"""Stage 3's margin schedule and greedy SWA (**T2-1**, **T2-2**, **T2-3** / OP-4.2-4.5).

Three defects, each of which broke a property SWA needs.

* **T2-1 (C-7a, C-7b).** Stage 3 passed ``arc_m = 0.25 + 0.05 cos(pi e / E)``, a
  *scalar*, which overrode the per-class ``margins`` buffer entirely — throwing
  away Stage 2's calibration at entry — and moved every epoch, so no two
  snapshots were iterates of the same objective. It now scales the whole vector
  by ``kappa``, stepping only at cycle boundaries.
* **T2-2 (C-7c).** The greedy filter read
  ``f1_live >= best_live_f1 * 0.98`` immediately after
  ``best_live_f1 = max(best_live_f1, f1_live)``. Whatever ``f1_live`` is, that
  test is ``f1_live >= f1_live * 0.98`` — true for every non-negative F1. The
  filter was provably a no-op and ``swa/n_rejected`` was 0 by construction.
  Acceptance now evaluates the **candidate average**.
* **T2-3 (C-7d).** The first snapshot was taken inside Adam's second-moment
  warm-up, so the average carried a transient that no later snapshot could
  dilute. The first ``swa_warmup_cycles`` cycles are now discarded.

The stage-level tests run a real Stage 3 on synthetic data and are marked
``slow``; the arithmetic ones do not need a stage at all.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectralquadnet.engine.stages.stage3_sam_swa import _blend, run_stage3_swa
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.tracking.base import NullTracker

pytestmark = pytest.mark.regression

DEVICE = torch.device("cpu")
SAMPLES, BATCH, SPATIAL = 8, 4, 64


class CaptureTracker(NullTracker):
    """A ``NullTracker`` that keeps every scalar it is handed, per step."""

    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}
        self.history: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        self.scalars.update(tags)
        self.history.append((step, dict(tags)))

    def series(self, key: str) -> list[float]:
        return [tags[key] for _, tags in self.history if key in tags]


@pytest.fixture(scope="module")
def synthetic_loader(cfg) -> DataLoader:
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(SAMPLES, cfg.data.num_bands, SPATIAL, SPATIAL, generator=gen)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=False)


def short_stage3(cfg, tmp_path, *, epochs: int, cycle_len: int, warmup: int, greedy: bool = True):
    small = cfg.copy()
    small.output_dir = str(tmp_path)
    small.stage3.epochs = epochs
    small.stage3.cycle_len = cycle_len
    small.stage3.swa_warmup_cycles = warmup
    small.stage3.greedy = greedy
    return small


def run(stage_cfg, physical_wl, loader, tmp_path, tracker=None, margins=None):
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(stage_cfg, physical_wl).to(DEVICE)
    if margins is not None:
        model.arcface_head.margins.copy_(margins)
    ema = ModelEMA(model, decay=stage_cfg.ema_decay)
    best = run_stage3_swa(
        stage_cfg,
        model,
        ema,
        loader,
        loader,
        DEVICE,
        best_ckpt=str(tmp_path / "best_stage3.pth"),
        prev_best_f1=0.0,
        tracker=tracker,
    )
    meta = json.loads((tmp_path / "stage3_meta.json").read_text())
    return model, meta, best


# ══════════════════════════════════════════════════════════════════════
#  T2-1 · the per-class margin vector is kept, scaled, and frozen per cycle
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_the_margin_vector_stays_non_constant_throughout_stage_3(cfg, physical_wl, tmp_path):
    """T2-1's criterion, first half: ``arcface_head.margins`` is non-constant throughout.

    Stage 2 leaves a calibrated per-class vector. The scalar schedule replaced
    it with one number on Stage 3's first epoch; the multiplicative one scales
    it, so the spread survives to the end.
    """
    gen = torch.Generator().manual_seed(9)
    calibrated = 0.20 + 0.30 * torch.rand(cfg.data.num_classes, generator=gen)
    stage_cfg = short_stage3(cfg, tmp_path, epochs=4, cycle_len=2, warmup=0)
    loader = DataLoader(
        TensorDataset(
            torch.randn(SAMPLES, cfg.data.num_bands, SPATIAL, SPATIAL, generator=gen),
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        ),
        batch_size=BATCH,
    )

    model, _, _ = run(stage_cfg, physical_wl, loader, tmp_path, margins=calibrated)

    final = model.arcface_head.margins
    assert float(final.std()) > 0.0, "the vector collapsed to a scalar"
    ratios = final / calibrated
    assert torch.allclose(
        ratios, ratios[0].expand_as(ratios), atol=1e-6
    ), "the anneal must be multiplicative — every class scaled by one kappa"


@pytest.mark.slow
def test_the_margin_is_frozen_within_a_cycle_and_steps_between(cfg, physical_wl, tmp_path):
    """T2-1's criterion, second half: the objective is constant across each cycle."""
    gen = torch.Generator().manual_seed(10)
    stage_cfg = short_stage3(cfg, tmp_path, epochs=4, cycle_len=2, warmup=0)
    loader = DataLoader(
        TensorDataset(
            torch.randn(SAMPLES, cfg.data.num_bands, SPATIAL, SPATIAL, generator=gen),
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        ),
        batch_size=BATCH,
    )
    tracker = CaptureTracker()

    run(stage_cfg, physical_wl, loader, tmp_path, tracker=tracker)

    kappas = tracker.series("sched/margin_kappa")
    assert len(kappas) == 4
    assert kappas[0] == kappas[1], "kappa changed inside cycle 1"
    assert kappas[2] == kappas[3], "kappa changed inside cycle 2"
    assert kappas[1] != kappas[2], "kappa must step at the cycle boundary"


def test_the_legacy_scalar_would_have_overridden_the_vector(cfg, physical_wl):
    """The companion: ``arc_m`` is an override, so a scalar schedule *is* a discard.

    Not incidental to the fix — it is the fix. Passing a float makes the head
    ignore ``margins`` for every sample in the batch.
    """
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl).train()
    model.arcface_head.margins.copy_(torch.linspace(0.2, 0.5, cfg.data.num_classes))
    x = torch.randn(4, cfg.data.num_bands, SPATIAL, SPATIAL)
    y = torch.tensor([0, 1, 2, 3])

    with torch.no_grad():
        per_class = model(x, labels=y)["main"]
        scalar = model(x, labels=y, arc_m=0.25)["main"]

    assert not torch.allclose(per_class, scalar)


# ══════════════════════════════════════════════════════════════════════
#  T2-2 · greedy acceptance evaluates the candidate average
# ══════════════════════════════════════════════════════════════════════


def test_the_old_acceptance_test_was_a_tautology(cfg):
    """C-7c, restated as arithmetic: the filter could not reject anything.

    ``best_live_f1 = max(best_live_f1, f1_live)`` runs first, so the test is
    ``f1_live >= max(best, f1_live) * 0.98``, and ``max(best, f1_live)`` is
    ``f1_live`` whenever ``f1_live`` is the larger — and when it is not, the
    factor 0.98 still has to be cleared. Only the second case can ever reject,
    and it needs a **2% drop from the best ever seen** to do it.
    """
    best = 0.0
    rejected = 0
    for f1_live in (0.10, 0.55, 0.54, 0.83, 0.82, 0.99):
        best = max(best, f1_live)
        if not f1_live >= best * 0.98:
            rejected += 1

    assert rejected == 0


def test_blend_produces_the_running_mean(cfg):
    """``(n-1)/n * average + 1/n * snapshot`` is the mean of the ``n`` snapshots."""
    snapshots = [{"w": torch.tensor([float(i)])} for i in range(1, 5)]
    average = {k: v.clone() for k, v in snapshots[0].items()}
    for n, snapshot in enumerate(snapshots[1:], start=2):
        average = _blend(average, snapshot, n)

    assert float(average["w"]) == pytest.approx(2.5)


def test_blend_returns_a_new_dict(cfg):
    """OP-4.4 requires scoring the candidate *before* it becomes the average."""
    average = {"w": torch.tensor([1.0])}
    snapshot = {"w": torch.tensor([3.0])}

    candidate = _blend(average, snapshot, 2)

    assert float(average["w"]) == 1.0
    assert float(candidate["w"]) == 2.0


def test_blend_takes_integer_buffers_from_the_snapshot(cfg):
    """``num_batches_tracked`` has no meaningful mean; averaging it would also fail on ints."""
    average = {"n": torch.tensor(4)}
    snapshot = {"n": torch.tensor(9)}

    assert int(_blend(average, snapshot, 2)["n"]) == 9


@pytest.mark.slow
def test_the_greedy_filter_can_now_reject(cfg, physical_wl, synthetic_loader, tmp_path):
    """T2-2's criterion: ``swa/n_rejected > 0`` on a normal run.

    Eight one-epoch cycles on a model that is not learning anything from eight
    synthetic patches — exactly the regime where an added snapshot should stop
    improving the average, which the old filter could not express.
    """
    stage_cfg = short_stage3(cfg, tmp_path, epochs=8, cycle_len=1, warmup=0)
    tracker = CaptureTracker()

    _, meta, _ = run(stage_cfg, physical_wl, synthetic_loader, tmp_path, tracker=tracker)

    assert meta["swa_n_rejected"] > 0
    assert meta["swa_n_snapshots"] + meta["swa_n_rejected"] == 8
    assert tracker.scalars["swa/n_rejected"] == float(meta["swa_n_rejected"])


@pytest.mark.slow
def test_a_non_greedy_run_accepts_every_candidate(cfg, physical_wl, synthetic_loader, tmp_path):
    """``greedy=false`` must still be the "average everything" control it names."""
    stage_cfg = short_stage3(cfg, tmp_path, epochs=4, cycle_len=1, warmup=0, greedy=False)

    _, meta, _ = run(stage_cfg, physical_wl, synthetic_loader, tmp_path)

    assert meta["swa_n_rejected"] == 0
    assert meta["swa_n_snapshots"] == 4


# ══════════════════════════════════════════════════════════════════════
#  T2-3 · the first cycles are discarded
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_the_first_accepted_snapshot_is_past_the_warm_up(
    cfg, physical_wl, synthetic_loader, tmp_path
):
    """T2-3's criterion: the first accepted snapshot index is ``>= warmup + 1``.

    At the shipped ``swa_warmup_cycles = 3`` that is "index >= 4", the number
    §3.6 derives from ``1/(1-beta2)`` over the steps in a cycle.
    """
    stage_cfg = short_stage3(cfg, tmp_path, epochs=6, cycle_len=1, warmup=3)

    _, meta, _ = run(stage_cfg, physical_wl, synthetic_loader, tmp_path)

    assert meta["swa_first_accepted_cycle"] == 4
    assert meta["swa_n_snapshots"] + meta["swa_n_rejected"] == 3


def test_the_shipped_warm_up_is_the_derived_three(cfg):
    """The config value is the plan's ``ceil((1/(1-beta2)) / steps_per_cycle)``, not a guess."""
    assert cfg.stage3.swa_warmup_cycles == 3


@pytest.mark.slow
def test_a_stage_shorter_than_the_warm_up_falls_back_to_the_live_model(
    cfg, physical_wl, synthetic_loader, tmp_path
):
    """Nothing accepted must still produce a checkpoint, not a crash or an empty average."""
    stage_cfg = short_stage3(cfg, tmp_path, epochs=2, cycle_len=1, warmup=5)

    _, meta, best = run(stage_cfg, physical_wl, synthetic_loader, tmp_path)

    assert meta["swa_n_snapshots"] == 0
    assert meta["swa_first_accepted_cycle"] == 0
    assert (tmp_path / "best_stage3.pth").exists()
    assert 0.0 <= best <= 1.0
