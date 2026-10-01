"""S11 — the S10 P0 instrumentation and the X2 pathway switch, contract by contract.

Every test here pins one claim the S11 study page makes about the code:

* **P0.4 / F54** — ``single.aux_weight_schedule``: ``legacy`` (default) is the
  old call bit for bit; ``fixed`` applies ``model.aux_head_weight``; the loop
  logs exactly what it applied.
* **P0.3** — ``train/loss_main``, ``train/acc_dominant``, ``train/acc_plain``.
* **P0.5 / D22** — model-declared gradient groups partition the parameters;
  the ``legacy`` clip partition is unchanged (``fuse`` stays in ``backbone``),
  the ``model`` one moves ``fuse`` into ``fusion``; finite-step means.
* **P0.2** — the clean-fit subset is fixed and class-stratified, and measuring
  it neither consumes the global RNG nor leaves a model in the wrong mode.
* **P0.1** — DDP evaluation is de-duplicated; logits are written; provenance.
* **X2** — ``model.pathways`` masks a pathway in train and eval, freezes it and
  removes its parameters from every gradient.

The end-to-end claim — that none of this moves a training number — is gate
G-neutral (``docs/research/evidence/S11_frozen_arms_execution/code/g_neutral.py``)
and the slow smoke test ``tests/smoke/test_s11_arms.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.loaders import dedup_index, gathered_positions
from spectralquadnet.engine.clean_fit import CleanFitProbe, clean_fit_subset, measure_fit
from spectralquadnet.engine.diagnostics import (
    compute_branch_influence,
    declared_grad_groups,
    module_grad_norm_tensors,
)
from spectralquadnet.engine.train_epoch import (
    _accumulate_finite,
    _finite_means,
    _nonfinite,
    train_one_epoch,
)
from spectralquadnet.losses.auxiliary import (
    _aux_loss_weight,
    describe_aux_weight_schedule,
    single_stage_aux_weight,
)
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import build_model
from spectralquadnet.models.spectral_seed_net import GRAD_GROUPS, SpectralSeedNet, resolve_pathways
from spectralquadnet.optim.param_groups import (
    CLIP_GROUPS,
    build_optimizer_s1,
    clip_grad_norm_by_group,
    clip_group_table,
    split_by_clip_group,
)
from spectralquadnet.reporting.artifacts import RunArtifacts
from spectralquadnet.tracking.base import NullTracker
from spectralquadnet.utils.provenance import COMMIT_ENV, code_revision, training_regime

N_BANDS, N_CLASSES, SIDE, BATCH = 8, 6, 16, 4

#: A structurally complete SpectralSeedNet at a size that runs in milliseconds.
TINY = [
    "data=ablation/u430k32_grouped",
    f"data.num_bands={N_BANDS}",
    f"data.num_classes={N_CLASSES}",
    "model.stem_channels=16",
    "model.stem_folded_depth=1",
    "model.spatial_width_mult=0.25",
    "model.spectral_hidden=32",
    "model.index_bank_size=8",
    "model.continuum_depths=4",
    "model.aux_head_hidden=16",
]


def tiny_cfg(*extra: str) -> Any:
    return load_experiment_config(overrides=[*TINY, *extra])


def tiny_model(*extra: str, seed: int = 0) -> SpectralSeedNet:
    torch.manual_seed(seed)
    model = build_model(tiny_cfg(*extra), torch.linspace(0.0, 1.0, N_BANDS))
    assert isinstance(model, SpectralSeedNet)
    return model


def tiny_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(BATCH, N_BANDS, SIDE, SIDE, generator=gen) + 0.2
    x[..., :2, :] = 0.0
    y = torch.arange(BATCH) % N_CLASSES
    morph = torch.randn(BATCH, 8, generator=gen)
    return x, y, morph


class Recorder(NullTracker):
    """A tracker that keeps every scalar it is handed."""

    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        self.scalars.update(tags)

    def log_message(self, *args: Any, **kwargs: Any) -> None:
        return None


# ══════════════════════════════════════════════════════════════════════
#  P0.4 — the auxiliary-weight schedule (S10 F54)
# ══════════════════════════════════════════════════════════════════════


def test_the_shipped_default_is_the_legacy_schedule() -> None:
    cfg = load_experiment_config()
    assert cfg.single.aux_weight_schedule == "legacy"


@pytest.mark.parametrize("total", [150, 200])
def test_legacy_is_the_old_call_bit_for_bit(total: int) -> None:
    cfg = load_experiment_config()
    for ep in range(1, total + 1):
        assert single_stage_aux_weight(cfg, ep, total) == _aux_loss_weight(cfg, ep, total)


def test_legacy_applies_065_to_025_not_the_configured_02() -> None:
    """F54, as numbers: what every SpectralSeedNet run so far trained with."""
    cfg = load_experiment_config()
    first, last = single_stage_aux_weight(cfg, 1, 150), single_stage_aux_weight(cfg, 150, 150)
    assert first == pytest.approx(0.65 * (1 - 0.7 / 150))
    assert last == pytest.approx(0.25)
    assert float(cfg.model.aux_head_weight) == 0.2
    assert "NOT applied" in describe_aux_weight_schedule(cfg, 150)


def test_fixed_applies_the_documented_weight_on_every_epoch() -> None:
    cfg = load_experiment_config(overrides=["single.aux_weight_schedule=fixed"])
    assert {single_stage_aux_weight(cfg, ep, 150) for ep in range(1, 151)} == {0.2}


def test_an_unknown_schedule_is_refused() -> None:
    cfg = load_experiment_config(overrides=["single.aux_weight_schedule=linear"])
    with pytest.raises(ValueError, match="aux_weight_schedule"):
        single_stage_aux_weight(cfg, 1, 150)


def test_x4_switches_the_aux_term_off_through_the_keys_the_loop_reads() -> None:
    cfg = load_experiment_config(
        overrides=["stage1.aux_loss_weight_init=0.0", "stage1.aux_loss_weight_final=0.0"]
    )
    assert {single_stage_aux_weight(cfg, ep, 200) for ep in range(1, 201)} == {0.0}


# ══════════════════════════════════════════════════════════════════════
#  P0.3 — the epoch loop logs what it applied, and honest fit series
# ══════════════════════════════════════════════════════════════════════


def _one_epoch(
    cfg: Any, model: nn.Module, *, use_mixup: bool, arc_m: float, aux_weight: float | None
) -> tuple[Recorder, tuple[float, float]]:
    x, y, morph = tiny_batch()
    mask = (x.sum(1) > 0).float()
    ds = TensorDataset(x.repeat(2, 1, 1, 1), y.repeat(2), mask.repeat(2, 1, 1), morph.repeat(2, 1))
    loader = DataLoader(ds, batch_size=BATCH)
    opt = build_optimizer_s1(cfg, model, 1e-3)
    rec = Recorder()
    out = train_one_epoch(
        cfg,
        model,
        loader,
        opt,
        nn.CrossEntropyLoss(label_smoothing=0.1),
        None,
        ModelEMA(model, decay=0.999),
        torch.device("cpu"),
        use_mixup=use_mixup,
        mixup_alpha=0.35,
        arc_m=arc_m,
        current_ep=3,
        total_ep=150,
        tracker=rec,
        aux_weight=aux_weight,
    )
    return rec, out


def test_the_logged_aux_weight_is_the_applied_one() -> None:
    cfg = tiny_cfg()
    rec, _ = _one_epoch(cfg, tiny_model(), use_mixup=True, arc_m=0.0, aux_weight=0.37)
    assert rec.scalars["sched/aux_weight_applied"] == pytest.approx(0.37)
    rec, _ = _one_epoch(cfg, tiny_model(), use_mixup=True, arc_m=0.0, aux_weight=None)
    assert rec.scalars["sched/aux_weight_applied"] == _aux_loss_weight(cfg, 3, 150)


def test_the_aux_weight_reaches_the_loss() -> None:
    """Same model, same batch: the epoch loss moves with the weight passed."""
    cfg = tiny_cfg()
    _, (loss_a, _) = _one_epoch(cfg, tiny_model(), use_mixup=False, arc_m=0.0, aux_weight=0.0)
    _, (loss_b, _) = _one_epoch(cfg, tiny_model(), use_mixup=False, arc_m=0.0, aux_weight=1.0)
    assert loss_b > loss_a


def test_honest_fit_series_are_logged() -> None:
    cfg = tiny_cfg()
    rec, _ = _one_epoch(cfg, tiny_model(), use_mixup=True, arc_m=0.0, aux_weight=0.2)
    for key in ("train/loss_main", "train/acc_dominant", "train/acc_plain"):
        assert key in rec.scalars and math.isfinite(rec.scalars[key]), key
    rec, _ = _one_epoch(cfg, tiny_model(), use_mixup=False, arc_m=0.3, aux_weight=0.2)
    # Under a margin the plain logits can only score at least as well.
    assert rec.scalars["train/acc_plain"] >= rec.scalars["train/acc_dominant"] - 1e-9


def test_without_mixup_the_dominant_label_is_the_label() -> None:
    from spectralquadnet.engine.train_epoch import _train_telemetry

    logits = torch.eye(3) * 5
    ya, yb = torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1])
    t = _train_telemetry({"main": logits}, logits, torch.tensor(1.0), ya, yb, 1.0)
    assert t["train/acc_dominant"].item() == 1.0
    t = _train_telemetry({"main": logits}, logits, torch.tensor(1.0), ya, yb, 0.3)
    assert t["train/acc_dominant"].item() == 0.0  # λ < 0.5: yb dominates, and it is wrong


def test_acc_plain_is_omitted_rather_than_mislabelled() -> None:
    """Under a margin, a model without `main_plain` gets no `acc_plain` at all."""
    from spectralquadnet.engine.train_epoch import _train_telemetry

    logits, y = torch.eye(3) * 5, torch.tensor([0, 1, 2])
    margined = _train_telemetry({"main": logits}, logits, torch.tensor(1.0), y, y, 1.0, True)
    assert "train/acc_plain" not in margined
    plain = _train_telemetry(
        {"main": logits, "main_plain": logits}, logits, torch.tensor(1.0), y, y, 1.0, True
    )
    assert plain["train/acc_plain"].item() == 1.0


def test_per_module_gradient_norms_use_the_declared_groups() -> None:
    cfg = tiny_cfg()
    rec, _ = _one_epoch(cfg, tiny_model(), use_mixup=False, arc_m=0.0, aux_weight=0.2)
    for label, _ in GRAD_GROUPS:
        assert f"grad_norm/{label}" in rec.scalars, label
    assert rec.scalars["grad_norm/nonfinite_steps"] == 0.0


# ══════════════════════════════════════════════════════════════════════
#  P0.5 / D22 — gradient groups and the clip partition
# ══════════════════════════════════════════════════════════════════════


def _owners(model: nn.Module, table: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, str]:
    owner: dict[str, str] = {}
    for name, _ in model.named_parameters():
        for label, prefixes in table:
            if not prefixes or name.startswith(prefixes):
                owner[name] = label
                break
    return owner


def test_every_parameter_is_in_exactly_one_declared_gradient_group() -> None:
    model = tiny_model()
    owner = _owners(model, GRAD_GROUPS)
    assert set(owner) == {n for n, _ in model.named_parameters()}
    assert declared_grad_groups(model) == GRAD_GROUPS


def test_the_legacy_partition_keeps_the_sweeps_grouping() -> None:
    """F53, preserved on purpose: `fuse` and the aux head clip with the backbone."""
    model = tiny_model()
    groups = split_by_clip_group(model, "legacy")
    ids = {label: {id(p) for p in ps} for label, ps in groups.items()}
    assert id(model.fuse[1].weight) in ids["backbone"]
    assert id(model.aux_head_spatial.net[0].weight) in ids["backbone"]
    assert id(model.embed_net.mlp[0].weight) in ids["fusion"]
    assert clip_group_table(model, "legacy") == CLIP_GROUPS


def test_the_model_partition_moves_fuse_into_fusion() -> None:
    model = tiny_model()
    groups = split_by_clip_group(model, "model")
    ids = {label: {id(p) for p in ps} for label, ps in groups.items()}
    assert id(model.fuse[1].weight) in ids["fusion"]
    assert id(model.embed_net.mlp[0].weight) in ids["fusion"]
    assert id(model.aux_head_spatial.net[0].weight) in ids["backbone"]


@pytest.mark.parametrize("partition", ["legacy", "model"])
def test_each_partition_covers_every_trainable_parameter_once(partition: str) -> None:
    model = tiny_model()
    seen = [id(p) for ps in split_by_clip_group(model, partition).values() for p in ps]
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(p) for p in model.parameters() if p.requires_grad}


def test_an_unknown_partition_is_refused() -> None:
    with pytest.raises(ValueError, match="clip_partition"):
        clip_group_table(tiny_model(), "per_layer")


def test_the_shipped_default_partition_is_legacy() -> None:
    assert load_experiment_config().clip_partition == "legacy"


def test_partitions_agree_when_no_clip_binds() -> None:
    """D22's premise: at a threshold no group reaches, the partition is moot."""
    a, b = tiny_model(seed=1), tiny_model(seed=1)
    x, y, morph = tiny_batch()
    for model in (a, b):
        model.train()
        torch.manual_seed(11)  # the same dropout masks in both forwards
        out = model(x, labels=y, arc_m=0.0, morph=morph)
        (out["main"].logsumexp(1).mean() + out["aux_spatial"].mean()).backward()
    clip_grad_norm_by_group(a, 1e6, "legacy")
    clip_grad_norm_by_group(b, 1e6, "model")
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        if pa.grad is not None:
            assert torch.equal(pa.grad, pb.grad)


def test_finite_means_skip_overflow_steps_and_count_them() -> None:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}
    for v in (2.0, float("inf"), 4.0):
        _accumulate_finite(sums, counts, {"grad_norm/preclip_backbone": torch.tensor(v)})
    assert _finite_means(sums, counts)["grad_norm/preclip_backbone"] == pytest.approx(3.0)

    class _Report:
        preclip = {"head": torch.tensor(1.0), "backbone": torch.tensor(float("nan"))}

    assert _nonfinite(_Report()).item() == 1.0


def test_module_norms_omit_a_frozen_pathway() -> None:
    model = tiny_model("model.pathways=[spectral]")
    x, y, morph = tiny_batch()
    model.train()
    model(x, labels=y, arc_m=0.0, morph=morph)["main"].logsumexp(1).mean().backward()
    norms = module_grad_norm_tensors(model, GRAD_GROUPS)
    assert {"stem", "tail", "proj", "aux"}.isdisjoint(norms)
    assert {"spectral", "fuse", "embed", "head"} <= set(norms)


# ══════════════════════════════════════════════════════════════════════
#  X2 — the pathway switch
# ══════════════════════════════════════════════════════════════════════


def test_the_default_is_both_pathways_and_labels_name_them() -> None:
    model = tiny_model()
    assert model.pathways == ("spatial", "spectral")
    assert model.pathway_labels() == ("SPATIAL", "SPECTRAL")
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("bad", [[], ["spatial", "spatial"], ["texture"]])
def test_an_invalid_pathway_set_is_refused(bad: list[str]) -> None:
    with pytest.raises(ValueError, match="pathways"):
        resolve_pathways(bad)


def test_spectral_only_freezes_the_spatial_path_and_its_aux_head() -> None:
    model = tiny_model("model.pathways=[spectral]")
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(n.startswith(("spatial.", "aux_head_spatial.")) for n in frozen)
    x, y, morph = tiny_batch()
    model.train()
    out = model(x, labels=y, arc_m=0.0, morph=morph)
    assert "aux_spatial" not in out, "the aux term must vanish with its pathway"
    out["main"].logsumexp(1).mean().backward()
    for name, p in model.named_parameters():
        if name.startswith(("spatial.", "aux_head_spatial.")):
            assert p.grad is None, name
    assert model.spectral.mlp[0].weight.grad is not None


def test_spectral_only_logits_ignore_the_spatial_arrangement() -> None:
    """Permuting foreground pixels keeps the mean spectrum; only space changes."""
    x, _, morph = tiny_batch()
    x = torch.rand_like(x) + 0.2  # every pixel foreground, so the mask is unchanged
    perm = torch.randperm(SIDE * SIDE, generator=torch.Generator().manual_seed(3))
    shuffled = x.flatten(2)[..., perm].reshape_as(x)
    only = tiny_model("model.pathways=[spectral]").eval()
    full = tiny_model().eval()
    with torch.no_grad():
        assert torch.allclose(only(x, morph=morph), only(shuffled, morph=morph), atol=1e-4)
        assert not torch.allclose(full(x, morph=morph), full(shuffled, morph=morph), atol=1e-3)


def test_spatial_only_ignores_the_spectral_path_and_the_morphometrics() -> None:
    model = tiny_model("model.pathways=[spatial]")
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(n.startswith("spectral.") for n in frozen)
    x, _, morph = tiny_batch()
    model.eval()
    with torch.no_grad():
        a = model(x, morph=morph)
        b = model(x, morph=morph + 5.0)
    assert torch.equal(a, b)
    full = tiny_model().eval()
    with torch.no_grad():
        assert not torch.allclose(full(x, morph=morph), full(x, morph=morph + 5.0))


def test_a_disabled_pathway_keeps_the_live_pathways_initial_weights() -> None:
    """Every module is still constructed in order, so the RNG draws are the full model's."""
    full, only = tiny_model(seed=5), tiny_model("model.pathways=[spectral]", seed=5)
    for (n, a), (_, b) in zip(full.state_dict().items(), only.state_dict().items(), strict=True):
        assert torch.equal(a, b), n


def test_the_optimizer_holds_no_frozen_parameter() -> None:
    cfg = tiny_cfg("model.pathways=[spectral]")
    model = tiny_model("model.pathways=[spectral]")
    held = {id(p) for g in build_optimizer_s1(cfg, model, 1e-3).param_groups for p in g["params"]}
    assert all(id(p) not in held for p in model.parameters() if not p.requires_grad)


def test_the_influence_probe_names_the_two_pathways() -> None:
    model = tiny_model()
    x, y, morph = tiny_batch()
    mask = (x.sum(1) > 0).float()
    loader = DataLoader(TensorDataset(x, y, mask, morph), batch_size=BATCH)
    shares = compute_branch_influence(model, loader, torch.device("cpu"), max_batches=1)
    assert set(shares) == {"SPATIAL", "SPECTRAL"}


# ══════════════════════════════════════════════════════════════════════
#  P0.2 — the clean-fit subset and its measurement
# ══════════════════════════════════════════════════════════════════════


def test_the_subset_is_fixed_class_stratified_and_inside_train() -> None:
    labels = np.repeat(np.arange(90), 96)
    train = np.random.default_rng(1).choice(labels.size, 3683, replace=False)
    a = clean_fit_subset(train, labels, 1000, seed=0)
    np.random.seed(123)
    torch.manual_seed(123)
    b = clean_fit_subset(train, labels, 1000, seed=0)
    assert np.array_equal(a, b), "independent of the global RNG"
    assert a.size == 1000 and len(np.unique(a)) == 1000
    assert set(a) <= set(train)
    counts = np.bincount(labels[a], minlength=90)
    assert counts.max() - counts.min() <= 1
    assert not np.array_equal(a, clean_fit_subset(train, labels, 1000, seed=1))


def test_a_subset_larger_than_train_is_all_of_train() -> None:
    labels = np.arange(20) % 4
    assert np.array_equal(clean_fit_subset(np.arange(20), labels, 50, 0), np.arange(20))


class _Store:
    def __init__(self, patches: np.ndarray, labels: np.ndarray) -> None:
        self.patches, self.labels = patches, labels

    def require_patches(self) -> np.ndarray:
        return self.patches

    def require_labels(self) -> np.ndarray:
        return self.labels


def _probe_inputs(n: int = 24) -> tuple[_Store, Any, np.ndarray]:
    rng = np.random.default_rng(0)
    patches = (rng.random((n, N_BANDS, SIDE, SIDE)) + 0.2).astype(np.float32)
    labels = (np.arange(n) % N_CLASSES).astype(np.int64)
    data_cfg = SimpleNamespace(max_cutout_bands=1, noise_std=0.0, cutmix_bands=0, cutmix_spatial=0)
    return _Store(patches, labels), data_cfg, labels


def test_measuring_fit_matches_a_manual_eval_and_leaves_the_rng_alone() -> None:
    store, data_cfg, labels = _probe_inputs()
    probe = CleanFitProbe.build(
        store=store,  # type: ignore[arg-type]
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        train_rows=np.arange(len(labels)),
        labels=labels,
        morph=None,
        n=12,
        seed=0,
    )
    assert probe is not None and probe.rows.size == 12
    model = tiny_model()
    model.train()
    state = torch.get_rng_state()
    fit = measure_fit(model, probe.loader, torch.device("cpu"))
    assert torch.equal(state, torch.get_rng_state()), "the probe must not draw from the global RNG"
    assert model.training, "the train flag is restored"

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(store.patches[probe.rows])
        y = torch.from_numpy(labels[probe.rows])
        logits = model(x)
    assert fit.acc == pytest.approx((logits.argmax(1) == y).float().mean().item())
    assert fit.ce == pytest.approx(nn.functional.cross_entropy(logits, y).item(), rel=1e-5)
    assert fit.n == 12


def test_the_probe_payload_records_the_subset_and_the_final_epoch(tmp_path) -> None:
    store, data_cfg, labels = _probe_inputs()
    probe = CleanFitProbe.build(
        store=store,  # type: ignore[arg-type]
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        train_rows=np.arange(len(labels)),
        labels=labels,
        morph=None,
        n=12,
        seed=0,
    )
    assert probe is not None
    model = tiny_model()
    fit = probe.measure(model, model, torch.device("cpu"))
    probe.record(4, fit, improved=True)
    probe.record(7, fit, improved=False)
    path = probe.write(tmp_path, final_epoch=7)
    import json

    payload = json.loads(path.read_text())
    assert payload["subset"]["rows"] == [int(r) for r in probe.rows]
    assert payload["final"]["epoch"] == 7 and payload["at_best_checkpoint"]["epoch"] == 4
    assert set(CleanFitProbe.scalars(fit)) == {
        "fit/clean_train_acc_live",
        "fit/clean_train_ce_live",
        "fit/clean_train_acc_ema",
        "fit/clean_train_ce_ema",
    }


def test_zero_kernels_switches_the_probe_off() -> None:
    store, data_cfg, labels = _probe_inputs()
    assert (
        CleanFitProbe.build(
            store=store,  # type: ignore[arg-type]
            data_cfg=data_cfg,
            device=torch.device("cpu"),
            train_rows=np.arange(len(labels)),
            labels=labels,
            morph=None,
            n=0,
            seed=0,
        )
        is None
    )


# ══════════════════════════════════════════════════════════════════════
#  P0.1 — DDP de-duplication, logits, provenance
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("n, world", [(9, 2), (4311, 2), (10, 3), (5, 4), (8, 2)])
def test_dedup_scores_every_kernel_exactly_once(n: int, world: int) -> None:
    ds = TensorDataset(torch.arange(n))
    shards = [
        list(DistributedSampler(ds, num_replicas=world, rank=r, shuffle=False))
        for r in range(world)
    ]
    gathered = np.concatenate(shards)  # what gather_concat returns: rank order, padded
    loader = DataLoader(
        ds, sampler=DistributedSampler(ds, num_replicas=world, rank=0, shuffle=False)
    )
    assert np.array_equal(gathered_positions(loader), gathered)
    keep = dedup_index(loader)
    assert keep is not None
    assert np.array_equal(gathered[keep], np.arange(n)), "once each, in dataset order"
    assert (len(gathered) > n) == (n % world != 0)


def test_single_process_needs_no_dedup() -> None:
    assert dedup_index(DataLoader(TensorDataset(torch.arange(5)))) is None


def test_logits_are_written_as_float16_with_rows_and_targets(tmp_path) -> None:
    art = RunArtifacts.for_run(tmp_path)
    logits = np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32) * 20
    path = art.write_logits("calib_tta", logits, np.arange(7) % 5, rows=np.arange(100, 107))
    data = np.load(path)
    assert data["logits"].dtype == np.float16
    assert np.allclose(data["logits"], logits, atol=0.02)
    assert np.array_equal(data["rows"], np.arange(100, 107))


def test_the_code_revision_names_a_commit(monkeypatch) -> None:
    monkeypatch.delenv(COMMIT_ENV, raising=False)
    rev = code_revision()
    if rev["source"] == "git":
        assert len(rev["commit"]) == 40 and isinstance(rev["dirty"], bool)
    monkeypatch.setenv(COMMIT_ENV, "abc123")
    assert code_revision() == {"commit": "abc123", "dirty": None, "source": f"env:{COMMIT_ENV}"}


def test_the_regime_block_records_what_x1_applies() -> None:
    cfg = load_experiment_config(
        overrides=[
            "single.mixup_epochs=30",
            "single.arcface_m=0.0",
            "single.margin_warmup_start=31",
            "single.margin_warmup_end=31",
            "grad_clip=50.0",
            "single.epochs=200",
            "single.patience=40",
        ]
    )
    regime = training_regime(cfg, morph_input=True)
    assert regime["mixup_epochs"] == 30 and regime["epochs"] == 200 and regime["grad_clip"] == 50.0
    assert regime["aux_weight_schedule"] == "legacy"
    # D21 guard 3: X1 runs the sweep's aux schedule as a function of progress.
    assert regime["aux_weight_applied"]["mean"] == pytest.approx(0.425, abs=0.002)
    assert regime["pathways"] == ["spatial", "spectral"]
