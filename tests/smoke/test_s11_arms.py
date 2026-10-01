"""S11 — every frozen arm type, end to end through ``train.py``, on the synthetic cube.

The unit tests pin each P0 contract in isolation; this checks that they reach a
real run's outputs, for the regimes X1, X2 and X4 actually use (in miniature):

* ``results/run.json`` names the code revision and the regime *as applied*;
* ``clean_fit.json`` holds the fixed subset and a final-epoch live/EMA value;
* ``results/logits_*.npz`` exist for the reported split and calib, ±TTA;
* ``metrics.jsonl`` carries the honest series (``sched/aux_weight_applied`` …);
* X2's arms train what they claim to (pathways, morphometrics present or not).

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

X1_TINY = [
    "single.mixup_epochs=1",
    "single.arcface_m=0.0",
    "single.margin_warmup_start=2",
    "single.margin_warmup_end=2",
    "grad_clip=50.0",
    "single.epochs=3",
    "single.patience=3",
]
X4_TINY = [
    *X1_TINY,
    "single.mixup_epochs=0",
    "single.label_smooth_hi=0.0",
    "single.label_smooth_lo=0.0",
    "stage1.aux_loss_weight_init=0.0",
    "stage1.aux_loss_weight_final=0.0",
    "single.dropout=0.0",
    "single.aug_profile=none",
]


def _run(dataset: dict[str, str], out: Path, *extra: str) -> dict:
    overrides = tiny_overrides(dataset, out, **{"tracking.backend": "jsonl"})
    overrides += ["single.clean_fit_kernels=40", *extra]
    result = subprocess.run(
        [sys.executable, str(TRAIN), *overrides],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=2400,
        check=False,
    )
    assert result.returncode == 0, (
        f"train.py failed:\n{result.stdout[-6000:]}{result.stderr[-3000:]}"
    )
    return json.loads((out / "results" / "run.json").read_text())


def _series(out: Path) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for line in (out / "metrics.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "scalars":
            for k, v in event["metrics"].items():
                series.setdefault(k, []).append(v)
    return series


def test_an_x1_like_run_records_provenance_fit_and_logits(synthetic_dataset, tmp_path) -> None:
    out = tmp_path / "x1"
    manifest = _run(synthetic_dataset, out, *X1_TINY)
    run = manifest["run"]

    assert run["code"]["source"] in ("git", "unavailable") or run["code"]["source"].startswith(
        "env"
    )
    if run["code"]["source"] == "git":
        assert len(run["code"]["commit"]) == 40 and isinstance(run["code"]["dirty"], bool)
    regime = run["regime"]
    assert regime["mixup_epochs"] == 1 and regime["grad_clip"] == 50.0
    assert regime["aux_weight_schedule"] == "legacy" and regime["clip_partition"] == "legacy"
    assert regime["pathways"] == ["spatial", "spectral"] and regime["morphometrics_input"] is True
    assert run["output_dir"] == str(out)

    fit = json.loads((out / "clean_fit.json").read_text())
    assert fit["subset"]["n"] == 40 and len(fit["subset"]["rows"]) == 40
    assert fit["final"] is not None and fit["final"]["epoch"] == fit["final_epoch"]
    assert 0.0 <= fit["final"]["live"]["acc"] <= 1.0
    assert fit["at_best_checkpoint"] is not None

    for split in ("val_test_no_tta", "val_test_tta", "calib_no_tta", "calib_tta"):
        data = np.load(out / "results" / f"logits_{split}.npz")
        assert data["logits"].dtype == np.float16
        assert len(data["logits"]) == len(data["targets"]) == len(data["rows"])
    assert set(manifest["logits"]["splits"]) == {
        "val_test_no_tta",
        "val_test_tta",
        "calib_no_tta",
        "calib_tta",
    }
    # The logits and the scored predictions describe the same pass.
    held = np.load(out / "results" / "logits_val_test_no_tta.npz")
    preds = np.load(out / "results" / "preds_val_test_no_tta.npy")
    # (float16 storage can only flip an argmax between near-tied classes)
    assert np.mean(held["logits"].astype(np.float32).argmax(1) == preds) > 0.97

    series = _series(out)
    for key in (
        "sched/aux_weight_applied",
        "sched/aux_weight_configured",
        "train/loss_main",
        "train/acc_dominant",
        "train/acc_plain",
        "fit/clean_train_acc_live",
        "fit/clean_train_acc_ema",
        "grad_norm/stem",
        "grad_norm/nonfinite_steps",
    ):
        assert key in series, key
    assert "sched/aux_weight" not in series, "the ambiguous pre-S11 key is gone"
    applied = series["sched/aux_weight_applied"]
    assert applied[0] > 0.25 and all(v >= 0.25 for v in applied)  # legacy: decays to its floor
    meta = json.loads((out / "stage1_meta.json").read_text())
    assert "clean_fit" in meta


def test_an_x4_like_run_applies_no_aux_term(synthetic_dataset, tmp_path) -> None:
    out = tmp_path / "x4"
    manifest = _run(synthetic_dataset, out, *X4_TINY)
    assert manifest["run"]["regime"]["aux_weight_applied"]["mean"] == 0.0
    assert manifest["run"]["regime"]["aug_profile"] == "none"
    assert set(_series(out)["sched/aux_weight_applied"]) == {0.0}
    fit = json.loads((out / "clean_fit.json").read_text())
    assert fit["final"]["epoch"] == 3, "X4 is measured on its final epoch"


@pytest.mark.parametrize(
    "arm, override, pathways, morph",
    [
        ("spectral_only", "model.pathways=[spectral]", ["spectral"], True),
        ("spatial_only", "model.pathways=[spatial]", ["spatial"], True),
        ("no_morph", "data.morphology_path=''", ["spatial", "spectral"], False),
    ],
)
def test_the_x2_arms_train_what_they_claim(
    synthetic_dataset, tmp_path, arm: str, override: str, pathways: list[str], morph: bool
) -> None:
    out = tmp_path / arm
    manifest = _run(synthetic_dataset, out, override)
    regime = manifest["run"]["regime"]
    assert regime["pathways"] == pathways
    assert regime["morphometrics_input"] is morph
    series = _series(out)
    if arm == "spectral_only":
        # The aux head sits on the disabled pathway: no aux term, no spatial gradient.
        assert "loss/branch_spatial_raw" not in series
        assert "grad_norm/stem" not in series and "grad_norm/spectral" in series
    if arm == "spatial_only":
        assert "grad_norm/spectral" not in series and "grad_norm/stem" in series
