"""S13 — every arm type end to end through ``train.py`` on the synthetic cube, then Y1's fusion.

The unit tests pin each piece; this checks they reach a real run's outputs:

* Y1 — ``model.pathways=[spectral]`` / ``[spatial]`` cells score, write calib and
  held-out logits ±TTA, and fuse (weight on calib) into a results tree shaped
  like a model's, with the same/cross-session breakdown;
* Y2 — ``model.spatial_mixstyle=true`` trains and is recorded in the regime;
* Y3 — the lean keys build the smaller model and are recorded;
* every run reports the training-rows session κ (D26).

Slow: opt-in with ``--run-slow``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _helpers import REPO_ROOT, TRAIN, tiny_overrides

pytestmark = pytest.mark.slow

R1_TINY = [
    "single.mixup_epochs=1",
    "single.arcface_m=0.0",
    "single.margin_warmup_start=3",
    "single.margin_warmup_end=3",
    "grad_clip=50.0",
    "single.epochs=2",
    "single.patience=3",
]
LEAN = [
    "model.spectral_descriptor=snv_morph",
    "model.spatial_tail_strides=[2,2,2,1]",
    "model.cbam_min_hw=3",
]


@pytest.fixture(scope="module")
def scan_table(synthetic_dataset, tmp_path_factory) -> str:
    """Three sessions over the synthetic bundles, so the session metrics are defined."""
    groups = np.load(synthetic_dataset["groups_path"])
    path = tmp_path_factory.mktemp("sessions") / "scan_table.csv"
    lines = ["scan_id,session_id,session"]
    lines += [f"{g},{g % 3},S{g % 3}" for g in np.unique(groups)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _overrides(dataset: dict[str, str], out: Path, scan_table: str, *extra: str) -> list[str]:
    base = tiny_overrides(dataset, out, **{"tracking.backend": "jsonl"})
    return [*base, f"data.scan_table_path={scan_table}", *R1_TINY, *extra]


def _run(dataset: dict[str, str], out: Path, scan_table: str, *extra: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(TRAIN), *_overrides(dataset, out, scan_table, *extra)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=2400, check=False,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stdout[-6000:]}{result.stderr[-3000:]}"
    return json.loads((out / "results" / "run.json").read_text())


def test_y1_cells_score_and_fuse_on_calib(synthetic_dataset, scan_table, tmp_path) -> None:
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.data.loaders import build_split_bundle
    from spectralquadnet.experiments import fusion
    from spectralquadnet.reporting.session import SessionMap

    spectral = _run(synthetic_dataset, tmp_path / "spec", scan_table, "model.pathways=[spectral]")
    spatial = _run(synthetic_dataset, tmp_path / "spat", scan_table, "model.pathways=[spatial]")
    assert spectral["run"]["regime"]["pathways"] == ["spectral"]
    assert set(spectral["session_probe"]["representations"]) == {"embedding", "spectral"}
    assert set(spatial["session_probe"]["representations"]) == {"embedding", "spatial"}

    overrides = [
        o for o in _overrides(synthetic_dataset, tmp_path / "x", scan_table) if not o.startswith("hydra.")
    ]
    cfg = load_experiment_config(overrides=overrides)
    smap, _ = SessionMap.from_config(cfg.data, build_split_bundle(cfg), int(cfg.data.num_classes))
    assert smap is not None
    man = fusion.fuse_cells(
        tmp_path / "spec", tmp_path / "spat", tmp_path / "fused",
        num_classes=int(cfg.data.num_classes), sessions=smap, n_boot=16,
    )
    assert man["fusion"]["weight"] in fusion.WEIGHT_GRID
    assert {"tta", "no_tta"} <= set(man["results"])
    assert "cross_session" in man["results"]["tta"]["session"]
    n_held = len(np.load(tmp_path / "spec" / "results" / "preds_val_test_tta.npy"))
    assert man["results"]["tta"]["n_samples"] == n_held


def test_y2_mixstyle_trains_and_is_recorded(synthetic_dataset, scan_table, tmp_path) -> None:
    man = _run(synthetic_dataset, tmp_path / "y2", scan_table, "model.spatial_mixstyle=true")
    assert man["run"]["regime"]["spatial_mixstyle"] is True
    assert man["session_probe"]["representations"]["embedding"]["kappa"] is not None


def test_y3_lean_builds_the_smaller_model(synthetic_dataset, scan_table, tmp_path) -> None:
    full = _run(synthetic_dataset, tmp_path / "full", scan_table)
    lean = _run(synthetic_dataset, tmp_path / "lean", scan_table, *LEAN)
    regime = lean["run"]["regime"]
    assert regime["spectral_descriptor"] == "snv_morph" and regime["cbam_min_hw"] == 3
    assert regime["spatial_tail_strides"] == [2, 2, 2, 1]
    assert lean["run"]["parameters"] < full["run"]["parameters"]
    assert full["run"]["regime"]["spectral_descriptor"] == "full"
    assert (tmp_path / "lean" / "results" / "session_probe.json").exists()
