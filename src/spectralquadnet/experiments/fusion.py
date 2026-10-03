"""Late fusion of two single-pathway cells, weight chosen on calib (S13 Y1, H19; P0).

S12 F65 found that equal-weight late fusion of the separately trained spectral-
and spatial-only networks scores like the jointly trained network and keeps more
cross-session recall. ``preregistration_s12.json → arms.Y1_decoupled_pathways``
froze the test of it:

    For each (fold, seed) fuse the two networks' log-softmax:
    z = w·log p_spectral + (1−w)·log p_spatial, w ∈ {0, 0.05, …, 1} chosen by
    macro-F1 on calib (TTA logits; ties → closest to 0.5), applied unchanged to
    held-out TTA logits. Equal weight (w = 0.5) reported beside.

This module is that rule and nothing else. It reads the two cells' saved float16
logits (``results/logits_<split>_<variant>.npz``, D18), refuses unless both were
scored on the same kernels in the same order, picks ``w`` on **calib only**, and
writes a results tree shaped like a model's — ``results/run.json`` with
``results.{tta,no_tta}`` (metrics, bootstrap CI, same/cross-session breakdown),
predictions, fused log-probabilities and the session tables — so the S13 reading
treats a fused cell exactly like a trained one. ``results.tta`` is the frozen
number; ``fusion.equal_weight`` and ``results.no_tta`` (the calib-chosen ``w`` on
the un-augmented logits) are reported beside it and decide nothing.

No held-out row is read before ``w`` is fixed, and nothing is selected on one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spectralquadnet.reporting.artifacts import RESULTS_DIR, RunArtifacts, load_manifest
from spectralquadnet.reporting.metrics import macro_f1, score
from spectralquadnet.reporting.session import SessionMap, session_report

#: The frozen weight grid: 0, 0.05, …, 1 (21 values).
WEIGHT_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))
#: Reported beside the calib-chosen weight; decides nothing.
EQUAL_WEIGHT: float = 0.5
#: Two calib scores closer than this are a tie (they differ only in float noise).
TIE_TOL: float = 1e-12

RULE: str = (
    "z = w*log_softmax(spectral) + (1-w)*log_softmax(spatial); w in {0, 0.05, ..., 1} chosen by "
    "macro-F1 on calib TTA logits, ties -> closest to 0.5; applied unchanged to held-out TTA "
    "logits; w = 0.5 reported beside (preregistration_s12.json, arms.Y1_decoupled_pathways)"
)


class FusionError(ValueError):
    """The two cells cannot be fused: missing logits, or different kernels."""


@dataclass(frozen=True)
class CellLogits:
    """One saved pass: ``(n, C)`` logits, ``(n,)`` targets, ``(n,)`` patch rows or ``None``."""

    logits: npt.NDArray[np.float64]
    targets: npt.NDArray[np.int64]
    rows: npt.NDArray[np.int64] | None


def load_logits(run_dir: str | Path, split_tag: str) -> CellLogits:
    """``results/logits_<split_tag>.npz`` of one cell."""
    path = Path(run_dir) / RESULTS_DIR / f"logits_{split_tag}.npz"
    if not path.exists():
        raise FusionError(f"{path} does not exist — was the cell scored with evaluation.save_logits?")
    with np.load(path) as payload:
        rows = payload["rows"].astype(np.int64) if "rows" in payload.files else None
        return CellLogits(
            logits=payload["logits"].astype(np.float64),
            targets=payload["targets"].astype(np.int64),
            rows=rows,
        )


def log_softmax(z: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Row-wise log-softmax in float64."""
    z = np.asarray(z, dtype=np.float64)
    shifted = z - z.max(axis=1, keepdims=True)
    out: npt.NDArray[np.float64] = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return out


def fuse(
    spectral: npt.NDArray[Any], spatial: npt.NDArray[Any], w: float
) -> npt.NDArray[np.float64]:
    """``w·log p_spectral + (1−w)·log p_spatial``."""
    out: npt.NDArray[np.float64] = float(w) * log_softmax(spectral) + (1.0 - float(w)) * log_softmax(
        spatial
    )
    return out


def _aligned(a: CellLogits, b: CellLogits, what: str) -> None:
    if a.logits.shape != b.logits.shape:
        raise FusionError(f"{what}: logits {a.logits.shape} vs {b.logits.shape}")
    if not np.array_equal(a.targets, b.targets):
        raise FusionError(f"{what}: the two cells' targets differ — not the same kernels")
    if (a.rows is None) != (b.rows is None) or (
        a.rows is not None and b.rows is not None and not np.array_equal(a.rows, b.rows)
    ):
        raise FusionError(f"{what}: the two cells scored different rows (or one recorded none)")


def choose_weight(
    spectral: CellLogits,
    spatial: CellLogits,
    num_classes: int,
    grid: tuple[float, ...] = WEIGHT_GRID,
) -> tuple[float, dict[str, float]]:
    """The calib-chosen ``w`` and every grid point's calib macro-F1.

    Highest macro-F1; among ties (within :data:`TIE_TOL`), the weight closest to
    0.5; among those, the smaller weight (deterministic).
    """
    _aligned(spectral, spatial, "calib")
    table = {
        f"{w:.2f}": macro_f1(
            spectral.targets, fuse(spectral.logits, spatial.logits, w).argmax(1), num_classes
        )
        for w in grid
    }
    best = max(table.values())
    tied = [w for w in grid if best - table[f"{w:.2f}"] <= TIE_TOL]
    chosen = min(tied, key=lambda w: (abs(w - EQUAL_WEIGHT), w))
    return float(chosen), table


def _variant(
    out: RunArtifacts | None,
    tag: str,
    spectral: CellLogits,
    spatial: CellLogits,
    w: float,
    *,
    num_classes: int,
    n_boot: int,
    seed: int,
    sessions: SessionMap | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Score one fused held-out variant; write its artifacts when ``out`` is given."""
    fused = fuse(spectral.logits, spatial.logits, w)
    preds = fused.argmax(1)
    result = score(
        spectral.targets, preds, num_classes=num_classes, split=tag, n_boot=n_boot, seed=seed,
        context={**context, "fusion_weight": w},
    )
    payload = result.as_dict()
    breakdown = None
    if sessions is not None and spectral.rows is not None:
        breakdown = session_report(preds, spectral.targets, spectral.rows, sessions, split=tag)
        payload["session"] = breakdown.as_dict()
    if out is not None:
        out.write_predictions(tag, preds, spectral.targets, rows=spectral.rows)
        out.write_logits(tag, fused, spectral.targets, rows=spectral.rows)
        out.write_result(result)
        if breakdown is not None:
            out.write_session(breakdown)
    return payload


def fuse_cells(
    spectral_dir: str | Path,
    spatial_dir: str | Path,
    out_dir: str | Path,
    *,
    num_classes: int,
    sessions: SessionMap | None,
    n_boot: int = 2000,
    seed: int = 0,
    run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse two finished cells into ``out_dir`` and return the written ``run.json``.

    Args:
        spectral_dir, spatial_dir: The ``model.pathways=[spectral]`` and
            ``[spatial]`` cells of one (fold, seed).
        sessions: The fold's session map (train ∪ calib define where each class
            was trained); ``None`` skips the same/cross-session breakdown.
        run: Extra identity merged into ``run.json → run`` (fold, seed, code …).

    Raises:
        FusionError: A logits file is missing or the cells scored different kernels.
    """
    spectral_dir, spatial_dir = Path(spectral_dir), Path(spatial_dir)
    man_a, man_b = load_manifest(spectral_dir), load_manifest(spatial_dir)
    if not man_a or not man_b:
        raise FusionError("both cells must be scored (results/run.json) before they are fused")
    report = str((man_a.get("run") or {}).get("report_split", "val_test"))
    if report != str((man_b.get("run") or {}).get("report_split", "val_test")):
        raise FusionError("the two cells reported different splits")

    # ── 1 · the weight, from calib alone ──────────────────────────────
    w, calib_table = choose_weight(
        load_logits(spectral_dir, "calib_tta"), load_logits(spatial_dir, "calib_tta"), num_classes
    )

    # ── 2 · held-out, scored once with that weight (and 0.5 beside) ───
    held = {
        key: (load_logits(spectral_dir, f"{report}_{key}"), load_logits(spatial_dir, f"{report}_{key}"))
        for key in ("tta", "no_tta")
    }
    for key, (a, b) in held.items():
        _aligned(a, b, f"{report}_{key}")
    if sessions is not None and held["tta"][0].rows is None:
        sessions = None  # no rows → no breakdown, as in final_eval

    out = RunArtifacts.for_run(out_dir)
    context = {"fused_from": [str(spectral_dir), str(spatial_dir)]}
    results = {
        key: _variant(
            out, f"{report}_{key}", a, b, w,
            num_classes=num_classes, n_boot=n_boot, seed=seed, sessions=sessions, context=context,
        )
        for key, (a, b) in held.items()
    }
    equal = _variant(
        None, f"{report}_tta_equal_weight", held["tta"][0], held["tta"][1], EQUAL_WEIGHT,
        num_classes=num_classes, n_boot=n_boot, seed=seed, sessions=sessions, context=context,
    )

    component_probe = {
        name: (man.get("session_probe") or {}).get("representations")
        for name, man in (("spectral_only", man_a), ("spatial_only", man_b))
    }
    manifest = {
        "run": {
            "arch": "late_fusion(spectral_only, spatial_only)",
            "report_split": report,
            "select_split": "calib",
            "fused_from": {
                "spectral_only": {"dir": str(spectral_dir), "code": (man_a.get("run") or {}).get("code")},
                "spatial_only": {"dir": str(spatial_dir), "code": (man_b.get("run") or {}).get("code")},
            },
            **(run or {}),
        },
        "fusion": {
            "rule": RULE,
            "weight": w,
            "grid": list(WEIGHT_GRID),
            "calib_macro_f1_tta": calib_table,
            "equal_weight": {"weight": EQUAL_WEIGHT, "results": {"tta": equal}},
        },
        "results": results,
        "logits": {"dtype": "float16", "kind": "fused log-probabilities", "splits": [
            f"{report}_{k}" for k in held
        ]},
        "session_probe_components": component_probe,
    }
    path = out.results / "run.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
