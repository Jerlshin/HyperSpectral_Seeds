"""T4-5 / P-5 — the calibration split, and what reads it.

C-9 (§2.1.2, §2.1.4): one 1,294-patch validation split did six jobs at once. It
fitted the 90 per-class margins, the 8,100-entry confusion matrix, the 90 CDWS
sampling weights and the Phase-3 oversampling weights — 270+ fitted parameters
by the plan's count — *and then* chose which epoch to keep. Fitting and
selecting on the same sample is what the val→test gap 0-C measured is made of:
+0.011 on the two checkpoints selected that way, −0.003 on the one that was
not.

P-5's remedy is an inner split carved from train, so this module asserts two
things and their negations:

* every **fitted** quantity is measured on ``calib``;
* every **selection** decision is measured on ``val``;
* and with ``calib_ldr=None`` both fall back to ``val``, which is the
  pre-Tier-4 behaviour every archived checkpoint was produced under.

The stages are driven with stub collaborators rather than a real 15 M-parameter
training run, because what T4-5 changes is *which loader a call receives* —
running real epochs would measure the optimiser, not the routing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectralquadnet.data.loaders import build_calib_loader, build_split_bundle
from spectralquadnet.engine.stages import stage1_progressive, stage2_arcface
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.heads import AdaptiveSubcenterArcFaceHead

DEVICE = torch.device("cpu")
NUM_CLASSES = 6
EMBED = 8


class _StubModel(nn.Module):
    """The smallest module the two stages' non-training code paths accept."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(EMBED, EMBED)
        self.arcface_head = AdaptiveSubcenterArcFaceHead(EMBED, NUM_CLASSES, K=2)

    def set_dropout(self, p: float) -> None:
        self.dropout = p

    def set_subcentre_tau(self, tau: float) -> None:
        self.arcface_head.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - never called
        return self.arcface_head(self.backbone(x))


def _named_loader(name: str) -> DataLoader[Any]:
    """A loader carrying an identity, so a call site can be traced to its split."""
    loader = DataLoader(TensorDataset(torch.zeros(4, EMBED), torch.zeros(4, dtype=torch.long)))
    loader.split_name = name  # type: ignore[attr-defined]
    return loader


@pytest.fixture
def stage_cfg(cfg, tmp_path):
    """The composed config, shrunk to one epoch and pointed at ``tmp_path``."""
    small = cfg.copy()
    small.output_dir = str(tmp_path)
    small.data.num_classes = NUM_CLASSES
    small.stage1.epochs = 1
    small.stage2.epochs = 1
    return small


@pytest.fixture
def traced(monkeypatch):
    """Replace both stages' fitted-quantity calls with recorders.

    Returns a dict mapping each fitted quantity to the ``split_name`` of the
    loader it was handed.

    The doubles mirror the real signatures positionally on purpose — that is
    what lets them assert *which loader argument* each fitted quantity was
    handed. So they have to track those signatures: ``dist`` and ``detail``
    are the optional parameters the throughput refactor added.
    """
    seen: dict[str, str] = {}

    def fake_difficulty(cfg, shadow, loader, device, label="", tracker=None, step=0, detail=True):
        seen[f"cdws_{label}"] = loader.split_name
        return {c: 0.5 for c in range(NUM_CLASSES)}, {c: 1.0 for c in range(NUM_CLASSES)}

    def fake_pr(model, loader, device, num_classes, dist=None):
        seen["margins"] = loader.split_name
        precision = {c: 0.5 for c in range(num_classes)}
        recall = {c: 0.4 for c in range(num_classes)}
        return precision, recall, torch.zeros(num_classes, num_classes)

    def fake_evaluate(model, loader, device, dist=None):
        seen.setdefault("selection", loader.split_name)
        return 0.5, 0.5

    def fake_train(*args, **kwargs):
        return 0.1, 0.5

    def fake_save(*args, **kwargs):
        return None

    for module in (stage1_progressive, stage2_arcface):
        monkeypatch.setattr(module, "compute_class_difficulty", fake_difficulty)
        monkeypatch.setattr(module, "evaluate", fake_evaluate)
        monkeypatch.setattr(module, "train_one_epoch", fake_train)
        monkeypatch.setattr(module, "save_ckpt", fake_save)
    monkeypatch.setattr(stage2_arcface, "evaluate_pr_and_confusion", fake_pr)
    monkeypatch.setattr(
        stage1_progressive, "build_phase3_loader", lambda *a, **k: _named_loader("train")
    )
    return seen


def _run_stage1(cfg, traced, calib: DataLoader[Any] | None):
    model = _StubModel()
    phase_loaders = {p: _named_loader("train") for p in (1, 2, 3)}
    stage1_progressive.run_stage1(
        cfg,
        store=None,
        model=model,
        ema=ModelEMA(model, decay=0.99),
        loaders_by_phase=phase_loaders,
        val_ldr=_named_loader("val"),
        device=DEVICE,
        best_ckpt="unused.pth",
        tracker=None,
        calib_ldr=calib,
    )
    return traced


def _run_stage2(cfg, traced, calib: DataLoader[Any] | None):
    model = _StubModel()
    stage2_arcface.run_stage2(
        cfg,
        model=model,
        ema=ModelEMA(model, decay=0.99),
        train_ldr=_named_loader("train"),
        val_ldr=_named_loader("val"),
        device=DEVICE,
        best_ckpt="unused.pth",
        tracker=None,
        calib_ldr=calib,
    )
    return traced


# ══════════════════════════════════════════════════════════════════════
#  The routing — T4-5's criterion
# ══════════════════════════════════════════════════════════════════════


def test_stage1_fits_on_calib_and_selects_on_val(stage_cfg, traced) -> None:
    """CDWS weights and the Phase-3 oversampling F1 come off ``calib``."""
    seen = _run_stage1(stage_cfg, traced, _named_loader("calib"))
    assert seen["cdws_S1"] == "calib"
    assert seen["selection"] == "val"


def test_stage2_fits_its_270_parameters_on_calib(stage_cfg, traced) -> None:
    """The 90 margins and the (90, 90) confusion matrix — §4.2's "270 fitted
    parameters no longer touch the selection split"."""
    seen = _run_stage2(stage_cfg, traced, _named_loader("calib"))
    assert seen["margins"] == "calib"
    assert seen["cdws_S2"] == "calib"
    assert seen["selection"] == "val"


@pytest.mark.parametrize("runner", [_run_stage1, _run_stage2])
def test_without_a_calib_split_everything_falls_back_to_val(stage_cfg, traced, runner) -> None:
    """The companion: this is the pre-Tier-4 behaviour, and it is C-9.

    Every fitted quantity lands on the same loader that selects the
    checkpoint. The archived checkpoints were produced exactly this way, so
    the fallback has to keep working — but it is the defect, not the default
    to aim for.
    """
    seen = runner(stage_cfg, traced, None)
    assert set(seen.values()) == {"val"}


def test_the_phase3_oversampling_weights_are_fitted_on_calib(stage_cfg, traced) -> None:
    """The Phase-2→3 boundary measurement is a fitted quantity too.

    It sets ``HardClassOversampledSampler``'s weights, so measuring it on val
    lets the selection split shape the training distribution — a second,
    quieter channel of the same leak.
    """
    stage_cfg.stage1.epochs = 3
    stage_cfg.stage1.phase1_frac = 0.34
    stage_cfg.stage1.phase2_frac = 0.33
    seen = _run_stage1(stage_cfg, traced, _named_loader("calib"))
    assert seen["cdws_Phase2→3"] == "calib"


# ══════════════════════════════════════════════════════════════════════
#  The split itself
# ══════════════════════════════════════════════════════════════════════


def test_calib_is_disjoint_from_every_other_split(cfg) -> None:
    from omegaconf import OmegaConf

    bundle = build_split_bundle(OmegaConf.merge(cfg, {"data": {"calib_frac": 0.15}}))
    calib = set(bundle.calib.tolist())
    assert calib
    assert calib.isdisjoint(bundle.train.tolist())
    assert calib.isdisjoint(bundle.val.tolist())
    assert calib.isdisjoint(bundle.test.tolist())


def test_calib_comes_out_of_train_not_out_of_val(cfg) -> None:
    """P-5: "an inner split carved from *train*".

    Taking it from val would shrink the selection split rather than clean it,
    and taking it from test would spend the only untouched sample there is.
    """
    from omegaconf import OmegaConf

    plain = build_split_bundle(cfg)
    with_calib = build_split_bundle(OmegaConf.merge(cfg, {"data": {"calib_frac": 0.15}}))

    assert np.array_equal(np.sort(plain.val), np.sort(with_calib.val))
    assert np.array_equal(np.sort(plain.test), np.sort(with_calib.test))
    assert set(with_calib.calib.tolist()) <= set(plain.train.tolist())
    assert len(with_calib.train) + len(with_calib.calib) == len(plain.train)


def test_calib_never_contributes_a_gradient(cfg) -> None:
    """``build_splits``' ``train_idx`` excludes calib, so no loader can train on it."""
    from omegaconf import OmegaConf

    from spectralquadnet.data.loaders import build_splits

    with_calib = OmegaConf.merge(cfg, {"data": {"calib_frac": 0.15}})
    _, train_idx, _, _ = build_splits(with_calib)
    bundle = build_split_bundle(with_calib)
    assert set(train_idx.tolist()).isdisjoint(bundle.calib.tolist())


def test_calib_covers_every_class(cfg) -> None:
    """A calibration split missing a class cannot calibrate that class's margin."""
    from omegaconf import OmegaConf

    bundle = build_split_bundle(OmegaConf.merge(cfg, {"data": {"calib_frac": 0.15}}))
    assert len(np.unique(bundle.labels[bundle.calib])) == 90


def test_no_calib_loader_is_built_when_no_calib_split_exists(cfg) -> None:
    assert build_calib_loader(cfg, None, DEVICE, np.empty(0, dtype=np.int64)) is None
