"""S20 fixed CPU feature probes and paired class-cluster uncertainty.

All fitted transforms are trained on the outer training rows. The calibration carve
shares that scan and is only used for temperatures/fusion weights. It is never used
as evidence of acquisition transfer. Dataset metadata is not a model input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectralquadnet.data.prep.multimodal import sha256

Array = npt.NDArray[Any]


def verify_plan(path: Path) -> dict[str, Any]:
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if sha256(path) != expected:
        raise ValueError("Frozen plan hash mismatch")
    plan: dict[str, Any] = json.loads(path.read_text())
    for name, expected_hash in plan["input_hashes"].items():
        if sha256(Path(name)) != expected_hash:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    return plan


def make_features(data: Path, embeddings: Path, axes: dict[str, list[int]]) -> dict[str, Array]:
    summary = np.load(data / "hsi_summary.npy")
    hm = np.load(data / "morphology.npy")
    rm = np.load(data / "rgb_morphology.npy")
    rd = np.load(data / "rgb_descriptors.npy")
    features = {
        "rgb_shape": rm,
        "rgb_color": rd[:, :15],
        "rgb_texture_shape": np.c_[rm, rd[:, 15:]],
        "rgb_all": np.c_[rm, rd],
    }
    for axis, ids in axes.items():
        features["hsi_q" + axis] = np.c_[summary[:, :, ids].reshape(len(summary), -1), hm]
    for axis in ["32", "214_own"]:
        features["hsi_mean" + axis] = np.c_[summary[:, 0, axes[axis]], hm]
    for axis in ["32", "214_own"]:
        mean = summary[:, 0, axes[axis]]
        shape = (mean - mean.mean(1, keepdims=True)) / np.maximum(mean.std(1, keepdims=True), 1e-6)
        features["hsi_snvmean" + axis] = np.c_[shape, hm]
    for view in ["rgb", "gray", "rgb32", "silhouette"]:
        features["dino_" + view] = np.load(embeddings / (view + ".npy"))
    features["concat32_dino"] = np.c_[features["hsi_q32"], features["dino_rgb"]]
    for name, values in features.items():
        if len(values) != len(summary) or not np.isfinite(values).all():
            raise ValueError(f"Nonfinite/misaligned feature block {name}")
    return features


def fit_probe(
    x: Array,
    y: Array,
    train: Array,
    calib: Array,
    test: Array,
    export: Path | None = None,
) -> tuple[Array, Array]:
    """Fixed shrinkage LDA; training-only scaling and covariance, no refit on calib."""
    if set(train) & (set(calib) | set(test)) or set(calib) & set(test):
        raise ValueError("Overlapping partitions")
    x = np.asarray(x, dtype=np.float64)
    model = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )
    model.fit(x[train], y[train])
    if not np.array_equal(model[-1].classes_, np.arange(int(y.max()) + 1)):
        raise ValueError("Training partition lacks a class")
    if export is not None:
        if export.exists():
            raise FileExistsError("Probe export already exists")
        # Numeric-only inference artifact: no pickle or executable estimator object.
        np.savez_compressed(
            export,
            mean=model[0].mean_,
            scale=model[0].scale_,
            coef=model[-1].coef_,
            intercept=model[-1].intercept_,
            classes=model[-1].classes_,
            train_rows=train,
        )
    return model.decision_function(x[calib]), model.decision_function(x[test])


def probe_logits(path: Path, x: Array) -> Array:
    """Apply an exported probe to the same ordered feature schema, without fitting."""
    with np.load(path, allow_pickle=False) as state:
        if x.ndim != 2 or x.shape[1] != len(state["mean"]) or not np.isfinite(x).all():
            raise ValueError("Invalid probe input dimensions/values")
        if np.any(state["scale"] <= 0):
            raise ValueError("Invalid exported scale")
        scores: Array = ((x - state["mean"]) / state["scale"]) @ state["coef"].T + state[
            "intercept"
        ]
    return scores[:, 0] if scores.shape[1] == 1 else scores


def calibrate_temperature(logits: Array, labels: Array, temperatures: list[float]) -> float:
    if not temperatures or min(temperatures) <= 0:
        raise ValueError("Temperatures must be positive")
    losses = [
        log_loss(labels, softmax(logits / t, axis=1), labels=np.arange(logits.shape[1]))
        for t in temperatures
    ]
    return float(temperatures[int(np.argmin(losses))])


def select_fusion(hsi: Array, rgb: Array, y: Array, weights: list[float]) -> float:
    if not weights or min(weights) < 0 or max(weights) > 1:
        raise ValueError("Fusion weights outside [0,1]")
    scores = [
        f1_score(y, (a * hsi + (1 - a) * rgb).argmax(1), average="macro", zero_division=0)
        for a in weights
    ]
    # Frozen deterministic tie-break prefers the first weight in the manifest.
    return float(weights[int(np.argmax(scores))])


def class_metrics(y: Array, pred: Array, classes: int) -> dict[str, Array]:
    matrix = np.bincount(y.astype(int) * classes + pred.astype(int), minlength=classes**2).reshape(
        classes, classes
    )
    support = matrix.sum(1)
    tp = np.diag(matrix)
    f1 = 2 * tp / np.maximum(matrix.sum(0) + support, 1)
    return {"f1": f1, "recall": tp / np.maximum(support, 1), "support": support, "correct": tp}


def cluster_interval(values: Array, n_boot: int = 2000, seed: int = 20261004) -> list[float]:
    """Percentile interval resampling varieties, retaining both fold directions.

    Input is classes x folds (or one aggregate column). For F1, resample the
    per-class contributions from fixed confusion matrices; precision denominators
    stay fixed. This describes class heterogeneity, not new-session uncertainty.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Empty/nonfinite class metric")
    rng = np.random.default_rng(seed)
    samples = a[rng.integers(len(a), size=(n_boot, len(a)))].mean(axis=(1, 2))
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def aligned_logits(path: Path, indices: Array, labels: Array) -> Array:
    """Load frozen historical logits only after exact row/target identity checks."""
    with np.load(path) as data:
        rows = data["rows"]
        if len(rows) != len(np.unique(rows)) or set(rows) != set(indices):
            raise ValueError("Historical prediction rows do not match the frozen split")
        if not np.array_equal(data["targets"], labels[rows]):
            raise ValueError("Historical prediction targets mismatch")
        order = {int(row): i for i, row in enumerate(rows)}
        logits = np.asarray(data["logits"][[order[int(i)] for i in indices]], dtype=np.float64)
    if (
        logits.ndim != 2
        or logits.shape[1] != int(labels.max()) + 1
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Invalid historical logits")
    return logits
