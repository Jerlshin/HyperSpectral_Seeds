"""Everything a pipeline needs, built once, in the one order that is correct.

``train.py`` used to hold this inline, which meant the RNG-ordering invariant
below was a comment in an entrypoint rather than a property of a function any
caller could reuse. With two pipelines (single-stage and three-stage), three
ablation drivers and a protocol sweep all needing the same setup, it has to be
constructible from one place or the arms of an experiment stop being comparable
for reasons nobody can see.

The ordering invariant
──────────────────────
::

    set_seed(cfg.seed) -> DataStore -> build_model -> ModelEMA

Every ``kaiming_normal_``/``trunc_normal_``/``xavier_uniform_`` inside the
branches draws from the same global torch RNG stream, in the order the model's
``__init__`` constructs them. So the seed must be set before model construction
and **nothing between the two may consume the global stream**.
:func:`~spectralquadnet.data.loaders.build_split_bundle` is safe on both of its
paths: sklearn's ``train_test_split`` takes ``random_state=42`` and uses its own
``RandomState``, and the grouped path draws from a private
``np.random.default_rng(SPLIT_SEED)``. The tracker is built by the caller,
before this, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy.typing as npt
import torch
import torch.nn as nn

from spectralquadnet.data.loaders import (
    Splits,
    build_calib_loader,
    build_split_bundle,
    standardised_morphometrics,
)
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import build_model, count_parameters, parameter_breakdown
from spectralquadnet.tracking.base import ExperimentTracker
from spectralquadnet.tracking.global_step import GlobalStep
from spectralquadnet.utils.device import (
    RuntimePlan,
    apply_runtime_optimisations,
    configure_backend,
    describe_amp,
    describe_hardware,
    maybe_compile,
    resolve_runtime,
)
from spectralquadnet.utils.distributed import DistContext, wrap_for_training
from spectralquadnet.utils.seed import set_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.utils.data import DataLoader

    from spectralquadnet.config.schema import ExperimentConfig


@dataclass
class RunContext:
    """The resolved run: hardware, data, splits, model and its shadow.

    A pipeline reads this and owns nothing else. That is what lets
    ``pipeline=single`` and ``pipeline=three_stage`` be genuinely the same run
    up to the curriculum, which is A8's whole premise.
    """

    cfg: ExperimentConfig | Any
    device: torch.device
    plan: RuntimePlan
    dist: DistContext
    tracker: ExperimentTracker
    store: DataStore
    splits: Splits
    model: nn.Module
    ema: ModelEMA
    #: The model under whatever DDP / ``torch.compile`` wrappers apply. The
    #: forward and backward go through this; everything that reads attributes
    #: or parameter names uses :attr:`model`.
    train_module: nn.Module
    #: ``(N, 8)`` morphometrics standardised on **train alone**, or ``None``.
    #: Fitted once and shared by every loader: the fit is a function of
    #: ``splits.train``, and fitting it per loader let the eval splits' grain
    #: sizes set the origin of a feature the network then classifies with.
    morph: npt.NDArray[Any] | None
    #: Cross-stage monotone step axis for the structured backends (IC-1).
    clock: GlobalStep
    #: Un-augmented loader over ``calib``, or ``None`` when no calibration
    #: split was carved. Every *fitted* quantity reads this.
    calib_loader: DataLoader[Any] | None

    @property
    def select_split(self) -> str:
        """Which split the checkpoint decision reads — ``calib`` or ``val``."""
        requested = str(getattr(self.cfg.evaluation, "select_split", "calib"))
        if requested == "calib" and self.calib_loader is None:
            return "val"
        return requested

    def summary(self) -> dict[str, Any]:
        """Run facts worth writing into the results JSON."""
        return {
            "arch": str(self.cfg.model.arch),
            "pipeline": str(self.cfg.pipeline),
            "seed": int(self.cfg.seed),
            "split_scheme": str(self.cfg.data.split_scheme),
            "split_fold": int(self.cfg.data.split_fold),
            "select_split": self.select_split,
            "report_split": str(getattr(self.cfg.evaluation, "report_split", "test")),
            "parameters": count_parameters(self.model, trainable_only=False),
            "parameter_breakdown": parameter_breakdown(self.model),
            "split_report": self.splits.report.as_dict(),
        }


def build_run_context(
    cfg: ExperimentConfig | Any,
    tracker: ExperimentTracker,
    dist: DistContext,
) -> RunContext:
    """Resolve hardware, load data, build the splits, the model and its shadow.

    Raises:
        ValueError: ``data.split_scheme=grouped`` was requested but the realised
            split is not group-disjoint. IC-3's assertion: a protocol that
            silently degrades is worse than one that fails, because the number
            it produces still looks like a variety-recognition number.
    """
    # ══════════════════════════════════════════════════════════════════
    #  RNG ORDERING. DO NOT MOVE THIS CALL. See the module docstring.
    # ══════════════════════════════════════════════════════════════════
    set_seed(cfg.seed)

    device = dist.device
    plan = resolve_runtime(cfg.runtime, device, dist.world_size)
    backend_notes = configure_backend(device, cfg.runtime)

    store = DataStore.from_config(cfg.data, device)
    gb_on_disk = store.require_patches().size * 4 / 1e9
    vram = (
        f" | {torch.cuda.mem_get_info(device)[0] / 1e9:.1f} GB VRAM free"
        if device.type == "cuda"
        else ""
    )
    tracker.log_message(
        f"[DATA] ✓ Dataset (mmap): {gb_on_disk:.1f} GB on disk {vram} | "
        f"num_workers={plan.num_workers} (eval {plan.eval_num_workers}) | "
        f"pin_memory={plan.pin_memory} | persistent={plan.persistent_workers}",
        level="plain",
    )

    splits = build_split_bundle(cfg)
    for line in splits.report.summary():
        tracker.log_message(line, level="plain")
    assert_protocol_holds(cfg, splits, tracker)
    tracker.log_message(
        f"Samples/class (train): ~{len(splits.train) // cfg.data.num_classes}", level="plain"
    )

    model = build_model(cfg, store.require_wavelengths()).to(device)
    path_notes = apply_runtime_optimisations(model, plan)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    tracker.log_message(
        f"Model  : {cfg.model.arch}  ({count_parameters(model, trainable_only=False):,} params)",
        level="plain",
    )
    tracker.watch(model)
    tracker.log_message(f"Device : {device}", level="plain")
    for line in describe_hardware(device):
        tracker.log_message(line, level="plain")
    if backend_notes:
        tracker.log_message("Kernels: " + "  ".join(backend_notes), level="plain")
    if path_notes:
        tracker.log_message("Paths  : " + "  ".join(path_notes), level="plain")
    if dist.enabled:
        tracker.log_message(
            f"[DDP]  world_size={dist.world_size}  rank={dist.rank}  "
            f"SyncBatchNorm={bool(cfg.runtime.sync_batchnorm)}  "
            f"per-rank batch = global // {dist.world_size}",
            level="plain",
        )
    tracker.log_message(
        f"Graph  : torch.compile={'on' if plan.compile_enabled else 'off'}"
        f" ({plan.compile_backend}/{plan.compile_mode})  |  "
        f"fused AdamW={plan.fused_optimizer}  |  channels_last={plan.channels_last}  |  "
        f"TF32={bool(cfg.runtime.allow_tf32)}",
        level="plain",
    )
    tracker.log_message(f"Precis : {describe_amp(plan.amp_dtype, device)}", level="plain")

    # ── Hardware dispatch ─────────────────────────────────────────────
    # Order matters: SyncBatchNorm conversion and the DDP wrap must happen
    # before `torch.compile` sees the module, or the compiled graph captures
    # the un-converted BatchNorm. The EMA shadow is *not* wrapped — it takes no
    # gradient, so it needs neither replication nor a compiled graph, and
    # keeping it plain is what lets `save_ckpt` write one schema.
    if plan.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    train_module = wrap_for_training(model, dist, sync_batchnorm=bool(cfg.runtime.sync_batchnorm))
    train_module = maybe_compile(train_module, plan)

    morph = standardised_morphometrics(store, splits.train)
    calib_loader = build_calib_loader(
        cfg, store, device, splits.calib, train_idx=splits.train, plan=plan, dist=dist
    )

    return RunContext(
        cfg=cfg,
        device=device,
        plan=plan,
        dist=dist,
        tracker=tracker,
        store=store,
        splits=splits,
        model=model,
        ema=ema,
        train_module=train_module,
        morph=morph,
        clock=GlobalStep(),
        calib_loader=calib_loader,
    )


def assert_protocol_holds(
    cfg: ExperimentConfig | Any, splits: Splits, tracker: ExperimentTracker
) -> None:
    """Fail the run when ``grouped`` was requested but not achieved (IC-3).

    CHANGES §21, Phase 0: *"add a ``SplitReport`` assertion that fails the run
    if group-disjointness is violated when ``grouped`` is requested."* The
    report already measured this; nothing acted on it, so a protocol that
    degraded produced a number indistinguishable from one that did not.

    The check is on ``train_eval_group_disjoint`` — the contract that no
    acquisition bundle appears in both training and evaluation. It is
    deliberately **not** on ``val_test_group_disjoint``: with exactly two
    bundles per class, val and test are two halves of one held-out bundle and
    three-way disjointness is mathematically impossible. That is a
    data-collection ceiling, reported at startup and stated in the paper, not a
    bug to fail on — which is why ``evaluation.report_split=val_test`` scores
    the two together as one held-out set.

    Also warns when the reporting protocol is one the audit rejected, so a run
    that produces an uninterpretable number says so in its own log.
    """
    if str(cfg.data.split_scheme) == "grouped" and not splits.report.train_eval_group_disjoint:
        leaking = splits.report.classes_leaking_into_eval
        raise ValueError(
            f"data.split_scheme=grouped was requested but {len(leaking)} classes have eval "
            f"patches sharing an acquisition bundle with their training patches "
            f"(classes {leaking[:10]}{' …' if len(leaking) > 10 else ''}). The realised split "
            "is not the protocol that was asked for, and a number from it is a claim about "
            "acquisition sessions rather than about rice varieties (CHANGES §4.2). Set "
            "data.single_group_policy=patch_split to accept that explicitly, with the leak "
            "recorded in the split report."
        )

    select = str(getattr(cfg.evaluation, "select_split", "calib"))
    if select == "calib" and not splits.has_calib:
        tracker.log_message(
            "evaluation.select_split=calib but data.calib_frac carved no calibration split — "
            "falling back to selecting on `val`, which is the split the run also fits its "
            "margins and sampling weights on. Set data.calib_frac=0.15 (CHANGES §4.4).",
            level="warn",
        )
    if str(cfg.data.split_scheme) == "stratified":
        tracker.log_message(
            "Protocol note: every acquisition bundle appears in train AND in val/test, so "
            "this run measures variety recognition and bundle recognition in an unmeasured "
            "mixture. Report it beside a `grouped` run, never alone (CHANGES §19.2).",
            level="warn",
        )
