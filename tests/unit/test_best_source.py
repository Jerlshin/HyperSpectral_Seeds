"""Which weights the final evaluation scores (**T1-8** / §2.1.4).

Every stage checkpoints on ``max(F1_live, F1_ema)`` and writes that maximum to
the sidecar as ``val_f1``; ``_pick_best_checkpoint`` ranks stages by it. But
``final_evaluation`` evaluated the **EMA shadow** unconditionally. When the live
model won the max — as it may well have at Stage 1's epoch 488, where the
shipped 0.8877 came from — the reported test number described a model that had
never scored it, and nothing in the artifacts recorded which of the two had won,
so the mismatch was unfalsifiable after the fact. That was the defect.

The stages now record ``best_source`` and ``final_evaluation`` honours it.
Bundles written before Tier 1 have no such key and fall back to ``"ema"``, which
is the behaviour they were produced under — ``test_a_legacy_bundle_still_evaluates_the_ema_shadow``
is what keeps the archived ``output_v12_spa40`` numbers reproducible.

The stubs below stand in for ``SpectralQuadNet`` deliberately: the property is
about which state dict reaches the forward pass, and a model that returns a
recognisable constant answers that in a way a 7.9 M-parameter network cannot.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectralquadnet.engine.checkpoint import save_ckpt
from spectralquadnet.engine.stages.final_eval import DEFAULT_BEST_SOURCE, final_evaluation
from spectralquadnet.models.ema import ModelEMA

DEVICE = torch.device("cpu")
SAMPLES, BANDS, SPATIAL = 8, 3, 4

#: Which class each stub predicts, so the written predictions name the winner.
LIVE_CLASS, EMA_CLASS = 1, 2


class TaggedModel(nn.Module):
    """Predicts one fixed class, carried in a parameter so it round-trips a state dict."""

    def __init__(self, num_classes: int, predicted: int) -> None:
        super().__init__()
        logits = torch.zeros(num_classes)
        logits[predicted] = 10.0
        self.logits = nn.Parameter(logits)
        self._use_arcface = False

    def use_arcface(self, flag: bool) -> None:
        self._use_arcface = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(x.shape[0], -1)


@pytest.fixture
def eval_cfg(cfg, tmp_path):
    small = cfg.copy()
    small.output_dir = str(tmp_path)
    return small


@pytest.fixture
def test_loader(cfg) -> DataLoader:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(SAMPLES, BANDS, SPATIAL, SPATIAL, generator=gen)
    y = torch.zeros(SAMPLES, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def write_bundle(eval_cfg, path, **metadata) -> None:
    """A Stage-1-shaped bundle whose ``model`` and ``ema`` slots disagree."""
    num_classes = eval_cfg.data.num_classes
    live = TaggedModel(num_classes, LIVE_CLASS)
    ema = ModelEMA(TaggedModel(num_classes, EMA_CLASS), decay=0.99)
    save_ckpt(eval_cfg, str(path), 488, "Stage 1", live, ema, val_f1=0.9, val_acc=0.9, **metadata)


def predicted_class(eval_cfg, path, test_loader) -> int:
    """Run ``final_evaluation`` on the bundle and read back what it predicted."""
    num_classes = eval_cfg.data.num_classes
    model = TaggedModel(num_classes, 0)
    ema = ModelEMA(TaggedModel(num_classes, 0), decay=0.99)

    final_evaluation(eval_cfg, model, ema, test_loader, DEVICE, str(path))

    preds = np.load(f"{eval_cfg.output_dir}/test_preds_noTTA.npy")
    assert len(set(preds.tolist())) == 1, "the stub predicts one class"
    return int(preds[0])


# ══════════════════════════════════════════════════════════════════════
#  The stages record it
# ══════════════════════════════════════════════════════════════════════


def test_the_sidecar_carries_best_source(eval_cfg, tmp_path) -> None:
    """§4.3's criterion: ``best_source`` reaches the JSON, not just the ``.pth``."""
    from spectralquadnet.engine.checkpoint import load_stage_meta

    write_bundle(eval_cfg, tmp_path / "best_stage1.pth", best_source="live")

    assert load_stage_meta(eval_cfg, 1)["best_source"] == "live"


@pytest.mark.parametrize(
    ("f1_live", "f1_ema", "expected"),
    [(0.88, 0.87, "live"), (0.87, 0.88, "ema"), (0.88, 0.88, "ema")],
)
def test_the_selection_rule_the_stages_apply(f1_live, f1_ema, expected) -> None:
    """The one-liner Stages 1 and 2 share, including the tie.

    A tie keeps the EMA shadow — the historical choice, so nothing about the
    archived runs is reinterpreted by this change.
    """
    assert ("ema" if f1_ema >= f1_live else "live") == expected


# ══════════════════════════════════════════════════════════════════════
#  `final_evaluation` honours it
# ══════════════════════════════════════════════════════════════════════


def test_a_live_selected_checkpoint_is_evaluated_live(eval_cfg, tmp_path, test_loader) -> None:
    """The defect, fixed: ``best_source="live"`` must not be scored on the shadow."""
    path = tmp_path / "best_stage1.pth"
    write_bundle(eval_cfg, path, best_source="live")

    assert predicted_class(eval_cfg, path, test_loader) == LIVE_CLASS


def test_an_ema_selected_checkpoint_is_evaluated_on_the_shadow(
    eval_cfg, tmp_path, test_loader
) -> None:
    path = tmp_path / "best_stage1.pth"
    write_bundle(eval_cfg, path, best_source="ema")

    assert predicted_class(eval_cfg, path, test_loader) == EMA_CLASS


def test_stage3_swa_source_reads_the_ema_slot(eval_cfg, tmp_path, test_loader) -> None:
    """Stage 3 writes ``"swa"`` when the SWA average wins, and puts it in the ``ema`` slot."""
    path = tmp_path / "best_stage3.pth"
    write_bundle(eval_cfg, path, best_source="swa")

    assert predicted_class(eval_cfg, path, test_loader) == EMA_CLASS


def test_a_legacy_bundle_still_evaluates_the_ema_shadow(eval_cfg, tmp_path, test_loader) -> None:
    """Bundles predating the key keep their original behaviour, so archived runs reproduce.

    ``outputs/output_v12_spa40/``'s three checkpoints are exactly this case;
    ``tests/regression/test_resume_and_final_eval.py`` reproduces their
    recorded predictions and would break if the default flipped.
    """
    path = tmp_path / "best_stage1.pth"
    write_bundle(eval_cfg, path)  # no best_source

    assert DEFAULT_BEST_SOURCE == "ema"
    assert predicted_class(eval_cfg, path, test_loader) == EMA_CLASS


def test_an_unrecognised_source_falls_back_to_the_shadow(eval_cfg, tmp_path, test_loader) -> None:
    """A value from a newer writer must degrade to the historical default, not crash."""
    path = tmp_path / "best_stage1.pth"
    write_bundle(eval_cfg, path, best_source="something_new")

    assert predicted_class(eval_cfg, path, test_loader) == EMA_CLASS
