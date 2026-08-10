#!/usr/bin/env python3
"""SpectralQuadNet training entrypoint.

Usage
─────
    python train.py                                  # the reference experiment
    python train.py stage1.max_lr=1e-4               # single override
    python train.py -m stage1.max_lr=1e-4,5e-4,1e-3  # sweep
    python train.py tracking.backend=multi tracking.backends=[console,wandb]

Multi-GPU (NVIDIA, e.g. T4 x2 / A100 / H100)::

    torchrun --standalone --nproc_per_node=2 train.py

DDP with ``SyncBatchNorm``, so the two-GPU run computes the **same** function as
the one-GPU run: the batch is split, but the normalisation statistics are
all-reduced back to the global batch's, and the gradient average over equal
shards is the global-batch gradient. See
:mod:`spectralquadnet.utils.distributed`. Nothing about it engages under a
plain ``python train.py``.

Execution knobs — worker counts, ``torch.compile``, fused AdamW, progress
rendering, the diagnostics stride — live under ``cfg.runtime`` and are resolved
once here into a :class:`~spectralquadnet.utils.device.RuntimePlan`. None of
them may change a reported number; the two that would (TF32, channels_last) are
off by default.

Auto-resume: ``latest_completed_stage()`` probes stages 3 → 2 → 1, each
completed stage is loaded rather than retrained, and
``_pick_best_checkpoint`` — not the stage order — decides which checkpoint
the final evaluation runs on. Pointing ``output_dir`` at a directory with all
three stages present therefore skips straight to ``final_evaluation``.

``torch.cuda.mem_get_info(device)`` is guarded by ``device.type == "cuda"``,
since it raises on a CPU/MPS host; the VRAM figure is simply omitted off-CUDA
and the dataset size is still reported.

The split protocol is a config choice (``data.split_scheme``,
``data.calib_frac``) and is **printed at startup**, because after Tier 4 the
same code produces two numbers that mean different things: a patch-level split
reports how well the model recognises scans it has seen, and a scan-disjoint
one reports how well it recognises varieties. ``python train.py
data=spa40_90class_pfix`` selects the latter.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from spectralquadnet.config.schema import register_configs
from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import (
    EVAL_BATCH,
    build_calib_loader,
    build_eval_loader,
    build_loaders,
    build_split_bundle,
    build_train_loader,
    standardised_morphometrics,
)
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.engine.checkpoint import (
    _pick_best_checkpoint,
    latest_completed_stage,
    load_ckpt,
    load_stage_meta,
    stage_ckpt_path,
)
from spectralquadnet.engine.diagnostics import compute_class_difficulty
from spectralquadnet.engine.evaluate import collect_embeddings
from spectralquadnet.engine.stages.final_eval import final_evaluation
from spectralquadnet.engine.stages.stage1_progressive import run_stage1
from spectralquadnet.engine.stages.stage2_arcface import run_stage2
from spectralquadnet.engine.stages.stage3_sam_swa import run_stage3_swa
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.tracking import build_tracker
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.device import (
    apply_runtime_optimisations,
    configure_backend,
    describe_hardware,
    empty_cache,
    maybe_compile,
    release_memory,
    reset_compilation,
    resolve_device,
    resolve_runtime,
)
from spectralquadnet.utils.distributed import (
    DistContext,
    init_distributed,
    shutdown,
    wrap_for_training,
)
from spectralquadnet.utils.seed import set_seed

# Makes the dataclass schemas available to Hydra's composition, so a typo in a
# YAML field fails at startup instead of hours into Stage 1.
register_configs()

#: How `latest_completed_stage()`'s return value reads in the resume banner.
_RESUME_LABELS = {0: "starting fresh", 1: "Stage 1 done", 2: "Stages 1–2 done", 3: "all done"}


class _LateBoundStdout(io.TextIOBase):
    """``sys.stdout`` resolved per write instead of at handler construction.

    The console tracker's progress bar redirects ``sys.stdout`` while it is live,
    so that a stray write scrolls *above* the bar rather than through it.
    ``logging.StreamHandler(sys.stdout)`` captures the stream object before that
    redirect exists and keeps writing to the original descriptor, which lands its
    output on top of the live region — the ``P1[INFO] …`` collision, and the
    reason the traceback from a crash mid-run arrives interleaved with a bar
    frame. Looking the attribute up per write puts logging back under whatever
    redirect is installed at that moment, and leaves it unchanged when none is.
    """

    def write(self, s: str) -> int:
        stream = sys.stdout
        written = stream.write(s)
        stream.flush()
        return written

    def flush(self) -> None:
        sys.stdout.flush()


def _run(cfg: DictConfig, tracker: ExperimentTracker, dist: DistContext) -> None:
    """Run the full auto-resuming training pipeline: Stages 1-3 plus final evaluation."""
    # ══════════════════════════════════════════════════════════════════
    #  RNG ORDERING. DO NOT MOVE THIS CALL.
    #
    #  Every `kaiming_normal_` / `trunc_normal_` / `xavier_uniform_` inside the
    #  branches' `_init_weights` draws from the same global torch RNG stream,
    #  in the order `SpectralQuadNet.__init__` constructs them — so the seed
    #  must be set before model construction, and the required call order is
    #  exactly:
    #
    #      load config → set_seed(cfg.seed) → DataStore → SpectralQuadNet → ModelEMA
    #
    #  Nothing between this line and the model construction below may consume
    #  the global RNG. `build_split_bundle` is safe on both of its paths:
    #  sklearn's `train_test_split` takes `random_state=42` and uses its own
    #  RandomState, and the grouped path draws from a private
    #  `np.random.default_rng(SPLIT_SEED)`. Neither touches the global torch or
    #  NumPy stream. Building the tracker happens before this call for the same
    #  reason.
    # ══════════════════════════════════════════════════════════════════
    set_seed(cfg.seed)

    # The device is the rank's under `torchrun` and `cfg.device`'s otherwise;
    # `init_distributed` has already made that choice, in `main`, because a rank
    # must own its device before the process group is joined.
    device = dist.device
    plan = resolve_runtime(cfg.runtime, device, dist.world_size)
    backend_notes = configure_backend(device, cfg.runtime)
    ckpt_s1 = stage_ckpt_path(cfg, 1)
    ckpt_s2 = stage_ckpt_path(cfg, 2)
    ckpt_s3 = stage_ckpt_path(cfg, 3)
    done_stage = latest_completed_stage(cfg)

    tracker.banner(
        f"Auto-resume: {_RESUME_LABELS.get(done_stage, f'stage {done_stage} done')}",
        [f"Output dir : {cfg.output_dir}"],
    )
    tracker.log_message(f"Latest completed stage: {done_stage}")

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
    all_labels = splits.labels
    train_idx, val_idx, test_idx = splits.train, splits.val, splits.test
    # P-1 / P-5 (T4-1 / T4-5). The protocol is reported, not assumed: the
    # report says which scans crossed which boundary and which classes it could
    # not keep group-disjoint, so a run's log records what its number means.
    for line in splits.report.summary():
        tracker.log_message(line, level="plain")
    tracker.log_message(
        f"Samples/class (train): ~{len(train_idx) // cfg.data.num_classes}", level="plain"
    )

    model = SpectralQuadNet(
        cfg=cfg,
        physical_wl=store.require_wavelengths(),
        num_classes=cfg.data.num_classes,
        num_bands=cfg.data.num_bands,
        dropout=cfg.stage1.dropout,
        wl_embed_dim=cfg.model.wl_embed_dim,
    ).to(device)

    # Before the EMA deep-copy and before any wrapper, so the shadow, the DDP
    # replica, the compiled graph and Stage 3's two SWA copies all run the same
    # paths. Sets no parameter and no buffer — see the function's docstring.
    path_notes = apply_runtime_optimisations(model, plan)

    ema = ModelEMA(model, decay=cfg.ema_decay)

    tracker.log_message(
        "Model  : SpectralQuadNet v4 (4× AuxHead deep supervision + P3 oversampling)",
        level="plain",
    )
    # Logs a `Params : …M` line on the console backend; structured backends
    # additionally attach gradient histograms.
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
        f"fused AdamW={plan.fused_optimizer}  |  channels_last={plan.channels_last}",
        level="plain",
    )

    # ── Hardware dispatch ─────────────────────────────────────────────
    # Order matters: SyncBatchNorm conversion and the DDP wrap must happen
    # before `torch.compile` sees the module, or the compiled graph captures
    # the un-converted BatchNorm. The EMA shadow is *not* wrapped — it takes no
    # gradient, so it needs neither replication nor a compiled graph, and
    # keeping it plain is what lets `save_ckpt` write one schema.
    if plan.channels_last and device.type == "cuda":
        # `Module.to`'s stubs do not carry the memory-format overload, though
        # the runtime accepts it; NHWC is opt-in precisely because it changes
        # which convolution kernels cuDNN selects.
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    train_model: torch.nn.Module = wrap_for_training(
        model, dist, sync_batchnorm=bool(cfg.runtime.sync_batchnorm)
    )
    train_model = maybe_compile(train_model, plan)

    # ── Calibration loader (P-5 / T4-5) ────────────────────────────────
    # `None` when no calib split was carved, which puts the margins, the CDWS
    # weights and the Phase-3 oversampling weights back on `val` — the
    # pre-Tier-4 behaviour, and the one every archived checkpoint was made
    # under.
    calib_ldr = build_calib_loader(
        cfg, store, device, splits.calib, train_idx=train_idx, plan=plan, dist=dist
    )

    # ── Morphometrics, standardised on train alone (P-4 / T4-4, FU-4) ──
    # `None` unless `data.morphology_path` names an array that exists, in which
    # case the model substitutes zeros — the mean of the standardised feature.
    # Fitted **once** and handed to every `build_loaders` call below: the fit is
    # a function of `train_idx` alone, and each of the four calls used to redo
    # it over all 8,624 rows.
    morph = standardised_morphometrics(store, train_idx)

    # ── Stage 1 phase loaders (per-phase aug profiles) ─────────────────
    def _s1_ldr(aug_str: str) -> DataLoader[Any]:
        ds = RiceSeedDataset(
            train_idx,
            aug_strength=aug_str,
            store=store,
            data_cfg=cfg.data,
            device=device,
            morph=morph,
        )
        return build_train_loader(ds, cfg.stage1.batch, seed=int(cfg.seed), plan=plan, dist=dist)

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1
    # ══════════════════════════════════════════════════════════════════
    if done_stage < 1:
        tracker.log_message("[RUN] Stage 1", level="plain")
        phase_loaders = {1: _s1_ldr("heavy"), 2: _s1_ldr("medium"), 3: _s1_ldr("very_light")}
        _, val_ldr1, _ = build_loaders(
            cfg,
            store,
            device,
            train_idx,
            val_idx,
            test_idx,
            cfg.stage1.batch,
            train_aug="none",
            plan=plan,
            dist=dist,
            morph=morph,
        )
        run_stage1(
            cfg,
            store,
            model,
            ema,
            phase_loaders,
            val_ldr1,
            device,
            ckpt_s1,
            tracker,
            calib_ldr=calib_ldr,
            plan=plan,
            dist=dist,
            train_module=train_model,
        )
        tracker.log_message("Reloading best Stage 1 checkpoint ...")
        load_ckpt(ckpt_s1, model, ema, device)
        # A stage boundary drops the stage's optimiser state, its scheduler, its
        # phase loaders and their prefetch queues all at once, and the reload
        # above has just materialised a checkpoint's worth of tensors on top.
        release_memory(device)
    else:
        tracker.log_message("[SKIP] Stage 1 → loading checkpoint", level="plain")
        load_ckpt(ckpt_s1, model, ema, device)

    meta_s1 = load_stage_meta(cfg, 1)
    class_f1_s1 = meta_s1.get("class_f1", {})
    cdws_wts_s1 = meta_s1.get("cdws_weights", {})
    arcface_done = meta_s1.get("arcface_init_done", False)
    s1_best_f1 = meta_s1.get("val_f1", meta_s1.get("val_acc", 0.0))
    tracker.log_message(
        f"Stage 1 → F1={s1_best_f1:.3f}  "
        f"hard classes={sum(1 for f in class_f1_s1.values() if f < 0.5)}"
    )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2
    # ══════════════════════════════════════════════════════════════════
    if done_stage < 2:
        if not arcface_done:
            # HD-2(iii) / T2-9. The head is the same one Stage 1 just trained,
            # so this is a re-seeding rather than a bootstrap: spherical
            # k-means on the Stage-1 embeddings puts every sub-centre inside
            # its own class's data, instead of two random decoys 9-18 degrees
            # from a single prototype that then never win and never learn.
            tracker.log_message("Seeding ArcFace sub-centres by spherical k-means")
            # Un-augmented, un-shuffled and nothing dropped: the clusters must
            # describe the data, not the augmentation profile, and `build_loaders`'
            # train loader would drop the last partial batch.
            seed_ldr = build_eval_loader(
                RiceSeedDataset(
                    train_idx,
                    aug_strength="none",
                    store=store,
                    data_cfg=cfg.data,
                    device=device,
                    morph=morph,
                ),
                plan=plan,
                batch_size=256,
            )
            emb, emb_y = collect_embeddings(ema.shadow, seed_ldr, device)
            stats = {"classes_seeded": 0.0, "min_separation": -1.0}
            for head in (model.arcface_head, ema.shadow.arcface_head):
                stats = head.init_subcentres_from_embeddings(emb, emb_y)
            tracker.log_message(
                f"Seeded {int(stats['classes_seeded'])} classes; "
                f"worst within-class sub-centre cosine {stats['min_separation']:.3f}"
            )

        if not class_f1_s1:
            tracker.log_message("No class_f1 in Stage 1 meta — recomputing", level="warn")
            _, val_cd, _ = build_loaders(
                cfg,
                store,
                device,
                train_idx,
                val_idx,
                test_idx,
                128,
                plan=plan,
                dist=dist,
                morph=morph,
            )
            class_f1_s1, cdws_wts_s1 = compute_class_difficulty(
                cfg,
                ema.shadow,
                calib_ldr if calib_ldr is not None else val_cd,
                device,
                "Stage 1 (recomputed)",
                tracker=tracker,
            )

        tracker.log_message("[RUN] Stage 2", level="plain")
        tr2, va2, _ = build_loaders(
            cfg,
            store,
            device,
            train_idx,
            val_idx,
            test_idx,
            cfg.stage2.batch,
            balanced=True,
            all_labels=all_labels,
            # "very_light" adds mild spectral noise so ArcFace doesn't overfit
            # the train set's exact spectral signatures.
            train_aug="very_light",
            class_weights=cdws_wts_s1,
            plan=plan,
            dist=dist,
            morph=morph,
        )
        run_stage2(
            cfg,
            model,
            ema,
            tr2,
            va2,
            device,
            ckpt_s2,
            tracker,
            calib_ldr=calib_ldr,
            plan=plan,
            dist=dist,
            train_module=train_model,
        )
        tracker.log_message("Reloading best Stage 2 checkpoint ...")
        load_ckpt(ckpt_s2, model, ema, device)
        release_memory(device)  # see the Stage 1 boundary above
    else:
        tracker.log_message("[SKIP] Stage 2 → loading checkpoint", level="plain")
        load_ckpt(ckpt_s2, model, ema, device)

    meta_s2 = load_stage_meta(cfg, 2)
    # class_f1 is in the Stage-2 sidecar too; nothing downstream reads it, since
    # Stage 3 takes its sampling weights from cdws_weights below.
    cdws_wts_s2 = meta_s2.get("cdws_weights", {})
    s2_best_f1 = meta_s2.get("val_f1", meta_s2.get("s2_val_f1", meta_s2.get("val_acc", 0.0)))
    tracker.log_message(f"Stage 2 → F1={s2_best_f1:.3f}")

    # Stage 3's double backward and per-cycle margin vector invalidate a
    # compiled graph's guards on every cycle; the graphs are dropped rather than
    # left to recompile. `run_stage3_swa` disables compilation for its own run.
    if plan.compile_enabled:
        tracker.log_message("Dropping compiled graphs before Stage 3")
    reset_compilation()

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 3
    # ══════════════════════════════════════════════════════════════════
    if done_stage < 3:
        if not cdws_wts_s2:
            tracker.log_message(
                "No cdws_weights in Stage 2 meta — falling back to Stage 1", level="warn"
            )
            cdws_wts_s2 = cdws_wts_s1

        tracker.log_message("[RUN] Stage 3 (SAM + Greedy SWA)", level="plain")
        tr3, va3, _ = build_loaders(
            cfg,
            store,
            device,
            train_idx,
            val_idx,
            test_idx,
            cfg.stage2.batch,
            balanced=True,
            all_labels=all_labels,
            train_aug="light",
            class_weights=cdws_wts_s2,
            plan=plan,
            dist=dist,
            morph=morph,
        )
        run_stage3_swa(
            cfg,
            model,
            ema,
            tr3,
            va3,
            device,
            ckpt_s3,
            prev_best_f1=s2_best_f1,
            tracker=tracker,
            plan=plan,
            dist=dist,
            train_module=train_model,
        )
    else:
        tracker.log_message("[SKIP] Stage 3 → loading checkpoint", level="plain")
        load_ckpt(ckpt_s3, model, ema, device)
        meta_s3 = load_stage_meta(cfg, 3)
        tracker.log_message(
            f"Stage 3 → snaps={meta_s3.get('swa_n_snapshots', '?')}  "
            f"rejected={meta_s3.get('swa_n_rejected', '?')}  "
            f"F1={meta_s3.get('val_f1', meta_s3.get('val_acc', 0)):.3f}"
        )

    # ══════════════════════════════════════════════════════════════════
    #  FINAL EVALUATION
    # ══════════════════════════════════════════════════════════════════
    best_final_ckpt = _pick_best_checkpoint(cfg, ckpt_s1, ckpt_s2, ckpt_s3)
    tracker.log_message(f"Best checkpoint (by val_f1): {best_final_ckpt}")

    # Stage 3 leaves two extra model copies behind; the allocator gets them back
    # before the twelve-view TTA pass allocates its own activations. Collected
    # first, because the SWA probe and the averaged copy are reachable from
    # cycles that only the cycle collector breaks — see `release_memory`.
    release_memory(device)

    _, _, test_ldr = build_loaders(
        cfg,
        store,
        device,
        train_idx,
        val_idx,
        test_idx,
        EVAL_BATCH,
        plan=plan,
        dist=dist,
        morph=morph,
    )
    final_evaluation(cfg, model, ema, test_ldr, device, best_final_ckpt, tracker, dist=dist)


@hydra.main(config_path="configs", config_name="experiment/output_v12_spa40", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose the config, set up logging and the tracker, then run.

    Logs to both a ``training.log`` file in the output directory and stdout.
    Any exception escaping :func:`_run` is logged at ``critical`` and exits
    with status 1, so a crashed run is visible in both the log file and the
    process exit code.

    Under ``torchrun`` the process group is joined **first** — before the model,
    the store or the tracker exist — because every rank has to own its CUDA
    device before anything allocates on it. Only rank 0 gets a tracker; the
    others get a :class:`~spectralquadnet.tracking.base.NullTracker`, so two
    GPUs do not write two interleaved copies of the same progress bar into one
    terminal, and rank 0's ``training.log`` stays the record of the run.
    """
    # `cfg.device` is the fallback, not a suggestion: without a launcher this is
    # the only thing that chooses, and `device=cpu` has to keep meaning CPU.
    # Under `torchrun` the local rank wins, because a rank must own exactly one
    # device and nothing else can be true.
    dist = init_distributed(cfg.runtime, fallback=resolve_device(cfg.device))

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler(_LateBoundStdout())]
    if dist.is_main:
        handlers.insert(0, logging.FileHandler(os.path.join(cfg.output_dir, "training.log")))
    logging.basicConfig(
        level=logging.INFO if dist.is_main else logging.WARNING,
        format=("%(asctime)s | %(levelname)s | %(message)s")
        if not dist.enabled
        else (f"%(asctime)s | rank{dist.rank} | %(levelname)s | %(message)s"),
        handlers=handlers,
        force=True,  # @hydra.main installs its own handlers first
    )

    tracker: ExperimentTracker = build_tracker(cfg) if dist.is_main else NullTracker()
    if dist.is_main:
        tracker.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    try:
        _run(cfg, tracker, dist)
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)
    finally:
        tracker.close()
        empty_cache()
        shutdown(dist)


if __name__ == "__main__":
    main()
