"""S13 Y1 — the late-fusion scorer follows the frozen rule and never selects on held-out.

Rule (``preregistration_s12.json → arms.Y1_decoupled_pathways``): fuse the two
networks' log-softmax with ``w ∈ {0, 0.05, …, 1}`` chosen by macro-F1 on calib
TTA logits (ties → closest to 0.5), apply it unchanged to held-out TTA logits,
report ``w = 0.5`` beside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectralquadnet.experiments import fusion
from spectralquadnet.reporting.artifacts import RunArtifacts
from spectralquadnet.reporting.session import SessionMap

C = 4


def _cell(root: Path, name: str, passes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Path:
    out = root / name
    art = RunArtifacts.for_run(out)
    for tag, (logits, targets, rows) in passes.items():
        art.write_logits(tag, logits, targets, rows=rows)
    art.write_manifest({"run": {"report_split": "val_test"}, "results": {}})
    return out


def _onehot_logits(pred: np.ndarray, scale: float = 5.0) -> np.ndarray:
    z = np.zeros((len(pred), C))
    z[np.arange(len(pred)), pred] = scale
    return z


def test_the_grid_is_the_frozen_one() -> None:
    assert fusion.WEIGHT_GRID[0] == 0.0 and fusion.WEIGHT_GRID[-1] == 1.0
    assert len(fusion.WEIGHT_GRID) == 21 and fusion.EQUAL_WEIGHT == 0.5


def test_fusion_is_a_weighted_sum_of_log_probabilities() -> None:
    a, b = np.random.default_rng(0).normal(size=(2, 5, C))
    z = fusion.fuse(a, b, 0.3)
    assert np.allclose(z, 0.3 * fusion.log_softmax(a) + 0.7 * fusion.log_softmax(b))
    assert np.allclose(np.exp(fusion.log_softmax(a)).sum(1), 1.0)


def test_the_weight_is_chosen_on_calib_and_ties_go_to_one_half() -> None:
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    # Both networks perfect on calib → every w ties → 0.5.
    perfect = fusion.CellLogits(_onehot_logits(y), y, None)
    w, table = fusion.choose_weight(perfect, perfect, C)
    assert w == 0.5 and len(table) == 21
    # Spectral perfect, spatial always wrong → only large w win; among the
    # winners the one closest to 0.5 is chosen.
    wrong = fusion.CellLogits(_onehot_logits((y + 1) % C), y, None)
    w, table = fusion.choose_weight(perfect, wrong, C)
    best = max(table.values())
    winners = [float(k) for k, v in table.items() if v == best]
    assert w == min(winners, key=lambda v: abs(v - 0.5)) and w > 0.5


def test_misaligned_cells_are_refused() -> None:
    y = np.array([0, 1, 2, 3])
    a = fusion.CellLogits(_onehot_logits(y), y, np.arange(4))
    b = fusion.CellLogits(_onehot_logits(y), y, np.array([0, 1, 3, 2]))
    with pytest.raises(fusion.FusionError, match="different rows"):
        fusion.choose_weight(a, b, C)
    c = fusion.CellLogits(_onehot_logits(y), y[::-1].copy(), np.arange(4))
    with pytest.raises(fusion.FusionError, match="targets differ"):
        fusion.choose_weight(a, c, C)


def test_held_out_never_moves_the_weight(tmp_path: Path) -> None:
    """Two runs with identical calib and opposite held-out logits pick the same w."""
    rng = np.random.default_rng(1)
    y_cal = rng.integers(0, C, 40)
    y_held = rng.integers(0, C, 60)
    rows_cal, rows_held = np.arange(40), np.arange(100, 160)
    cal_a = rng.normal(size=(40, C)) + 3 * np.eye(C)[y_cal]
    cal_b = rng.normal(size=(40, C)) + 1 * np.eye(C)[y_cal]
    weights = []
    for flip in (False, True):
        held_a = rng.normal(size=(60, C)) + (0 if flip else 4) * np.eye(C)[y_held]
        held_b = rng.normal(size=(60, C)) + (4 if flip else 0) * np.eye(C)[y_held]
        passes_a = {"calib_tta": (cal_a, y_cal, rows_cal), "val_test_tta": (held_a, y_held, rows_held),
                    "val_test_no_tta": (held_a, y_held, rows_held)}
        passes_b = {"calib_tta": (cal_b, y_cal, rows_cal), "val_test_tta": (held_b, y_held, rows_held),
                    "val_test_no_tta": (held_b, y_held, rows_held)}
        root = tmp_path / str(flip)
        man = fusion.fuse_cells(
            _cell(root, "spectral", passes_a), _cell(root, "spatial", passes_b), root / "fused",
            num_classes=C, sessions=None, n_boot=0,
        )
        weights.append(man["fusion"]["weight"])
    assert weights[0] == weights[1]


def test_the_fused_cell_reads_like_a_model_cell(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    y_cal, y_held = rng.integers(0, C, 40), rng.integers(0, C, 60)
    rows_cal, rows_held = np.arange(40), np.arange(40, 100)
    passes = lambda s: {  # noqa: E731
        "calib_tta": (rng.normal(size=(40, C)) + s * np.eye(C)[y_cal], y_cal, rows_cal),
        "val_test_tta": (rng.normal(size=(60, C)) + s * np.eye(C)[y_held], y_held, rows_held),
        "val_test_no_tta": (rng.normal(size=(60, C)) + s * np.eye(C)[y_held], y_held, rows_held),
    }
    labels = np.concatenate([y_cal, y_held])
    session_of_row = (labels % 2).astype(np.int64)  # classes 0,2 → session 0; 1,3 → 1
    smap = SessionMap(
        session_of_row=session_of_row, labels=labels.astype(np.int64),
        train_sessions={c: frozenset({c % 2}) for c in range(C)}, session_names={}, num_classes=C,
    )
    man = fusion.fuse_cells(
        _cell(tmp_path, "a", passes(2.0)), _cell(tmp_path, "b", passes(1.0)), tmp_path / "fused",
        num_classes=C, sessions=smap, n_boot=50, run={"split_fold": 0, "seed": 0},
    )
    saved = json.loads((tmp_path / "fused" / "results" / "run.json").read_text())
    assert saved["results"]["tta"]["macro_f1"] == man["results"]["tta"]["macro_f1"]
    assert {"tta", "no_tta"} <= set(saved["results"])
    assert "same_session" in saved["results"]["tta"]["session"]
    assert saved["results"]["tta"]["macro_f1_ci"]["n_boot"] == 50
    assert saved["fusion"]["equal_weight"]["weight"] == 0.5
    assert saved["run"]["select_split"] == "calib" and saved["run"]["split_fold"] == 0
    assert (tmp_path / "fused" / "results" / "preds_val_test_tta.npy").exists()
    assert (tmp_path / "fused" / "results" / "logits_val_test_tta.npz").exists()


def test_a_missing_logits_file_is_refused(tmp_path: Path) -> None:
    y = np.arange(4)
    a = _cell(tmp_path, "a", {"calib_tta": (_onehot_logits(y), y, None)})
    b = _cell(tmp_path, "b", {})
    with pytest.raises(fusion.FusionError, match="does not exist"):
        fusion.fuse_cells(a, b, tmp_path / "f", num_classes=C, sessions=None, n_boot=0)
