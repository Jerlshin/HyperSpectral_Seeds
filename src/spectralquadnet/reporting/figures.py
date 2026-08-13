"""Figures, rendered from the results tree rather than from a live model.

Every figure here is a function of files under ``results/``, so it can be
re-rendered after the fact, on a different machine, without a GPU or the 5.6 GB
cube. That is deliberate: the audited project's only figures were seven W&B
panels that covered Stage 1 alone, and there was no way to regenerate anything
once the run ended.

matplotlib is optional
──────────────────────
It is not in the core dependency set — training must not require a plotting
stack — so every entry point degrades to "wrote no figures" rather than raising.
:func:`available` is the check; the callers report the count they wrote.

Style
─────
Deliberately plain: no seaborn, no custom palette, colour-blind-safe
sequential maps, and every axis labelled with its units. These go into a paper,
where a figure that needs its own legend explained is a figure that will be
asked about.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.reporting.artifacts import RunArtifacts
    from spectralquadnet.reporting.metrics import ClassificationResult

_log = logging.getLogger(__name__)

#: Figure DPI. 150 is legible in a two-column layout without producing files
#: too large to attach to a tracker run.
DPI: int = 150

#: How many of the worst classes the per-class figure annotates by name.
HARDEST_K: int = 10


def available() -> bool:
    """Whether matplotlib can be imported in this environment."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _pyplot() -> Any:
    """A non-interactive pyplot, or ``None``.

    ``Agg`` is forced before ``pyplot`` is imported: a headless training host
    has no display, and the default backend selection raises rather than
    falling back on some platforms.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


# ══════════════════════════════════════════════════════════════════════
#  Individual figures
# ══════════════════════════════════════════════════════════════════════


def confusion_heatmap(
    confusion: npt.NDArray[Any], path: Path, title: str = "Confusion matrix", normalise: bool = True
) -> Path | None:
    """The full ``C × C`` confusion matrix, rows = true class.

    Row-normalised by default. At 90 classes the raw counts are dominated by
    the diagonal and the off-diagonal structure — which is the entire point of
    plotting it, since CHANGES §10.5 identifies a mutually-confusable cluster
    {41, 49, 51, 52, 70} that eight targeted mechanisms could not move — is
    invisible without it.
    """
    plt = _pyplot()
    if plt is None:
        return None
    cm = np.asarray(confusion, dtype=np.float64)
    if normalise:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1.0, None)

    fig, ax = plt.subplots(figsize=(8, 7))
    # The diagonal is ~1.0 and the interesting mass is <0.3, so the colour
    # scale is capped well below the max; otherwise every off-diagonal cell
    # renders as the same near-white.
    vmax = float(np.percentile(cm[~np.eye(len(cm), dtype=bool)], 99.5)) if len(cm) > 1 else 1.0
    im = ax.imshow(cm, cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-3), interpolation="nearest")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{title}\n(row-normalised, colour capped at the 99.5th off-diagonal pct)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(predicted | true)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def per_class_f1_bars(result: ClassificationResult, path: Path) -> Path | None:
    """Per-class F1, sorted, with the hardest ``HARDEST_K`` annotated.

    The figure that makes "the hard classes never moved" checkable: the same
    plot from two runs, side by side, shows whether the bottom of the
    distribution is the same set of class ids.
    """
    plt = _pyplot()
    if plt is None:
        return None
    items = sorted(result.per_class_f1.items(), key=lambda kv: (kv[1], kv[0]))
    classes = [c for c, _ in items]
    scores = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(scores)), scores, width=0.9, color="#4C72B0")
    ax.axhline(
        result.macro_f1,
        color="#C44E52",
        linestyle="--",
        linewidth=1.2,
        label=f"macro-F1 = {result.macro_f1:.3f}",
    )
    ax.set_xlabel(f"Class, sorted by F1 ({len(scores)} classes)")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Per-class F1 — {result.split}")
    for rank in range(min(HARDEST_K, len(classes))):
        ax.text(rank, scores[rank] + 0.02, str(classes[rank]), ha="center", fontsize=7, rotation=90)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def protocol_comparison(
    arms: dict[str, dict[str, float]], path: Path, title: str = "Protocol comparison"
) -> Path | None:
    """Mean ± range per arm — the figure CHANGES §19.2 asks the paper to lead with.

    Args:
        arms: ``{label: {"mean", "min", "max", ...}}`` from
            :func:`~spectralquadnet.reporting.metrics.mean_and_range`.

    The error bar is a **range over folds and seeds**, not a standard error.
    With 2 folds × 3 seeds there is no third fold to be had, and a range is the
    honest summary of six numbers where an SE would imply a sampling model the
    data does not support.
    """
    plt = _pyplot()
    if plt is None or not arms:
        return None
    labels = list(arms)
    means = [arms[k]["mean"] for k in labels]
    lower = [arms[k]["mean"] - arms[k]["min"] for k in labels]
    upper = [arms[k]["max"] - arms[k]["mean"] for k in labels]

    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(labels)), 4.5))
    ax.bar(labels, means, yerr=[lower, upper], capsize=6, color="#4C72B0", width=0.6)
    # Annotated above the whisker, so the mean is legible without reading the
    # axis — these bars sit near each other and the differences are small.
    for i in range(len(labels)):
        ax.text(i, means[i] + upper[i] + 0.012, f"{means[i]:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{title}\n(bars = mean, whiskers = min–max over folds × seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def ablation_forest(
    deltas: dict[str, tuple[float, float, float]], path: Path, title: str = "Ablation deltas"
) -> Path | None:
    """Forest plot of ``Δ macro-F1`` with intervals, against a zero line.

    Args:
        deltas: ``{arm: (delta, ci_lo, ci_hi)}`` relative to the reference arm.

    The zero line is the whole figure. An arm whose interval crosses it has not
    been shown to do anything, and CHANGES §23.2 is explicit that a properly
    powered negative result is a contribution — so the plot is drawn to make
    "crosses zero" the visually obvious category, not a footnote.
    """
    plt = _pyplot()
    if plt is None or not deltas:
        return None
    labels = list(deltas)
    values = [deltas[k][0] for k in labels]
    lo = [deltas[k][0] - deltas[k][1] for k in labels]
    hi = [deltas[k][2] - deltas[k][0] for k in labels]
    crosses = [deltas[k][1] <= 0.0 <= deltas[k][2] for k in labels]

    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.5 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.errorbar(
        values,
        y,
        xerr=[lo, hi],
        fmt="o",
        capsize=4,
        color="#4C72B0",
        ecolor="#8899BB",
        markersize=6,
    )
    for i, is_null in enumerate(crosses):
        if is_null:
            ax.plot(values[i], y[i], "o", color="#999999", markersize=6)
    ax.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ macro-F1 vs reference  (grey = interval crosses zero)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
#  Batch rendering
# ══════════════════════════════════════════════════════════════════════


def render_run_figures(
    artifacts: RunArtifacts, results: dict[str, ClassificationResult]
) -> list[Path]:
    """Every per-run figure, from one run's scored splits.

    Returns the paths written — empty when matplotlib is unavailable, which is
    a supported state and not an error.
    """
    if not available():
        _log.info("matplotlib not installed; skipping figures (pip install -e '.[figures]')")
        return []
    written: list[Path] = []
    for result in results.values():
        cm_path = artifacts.figures / f"confusion_{result.split}.png"
        if confusion_heatmap(result.confusion, cm_path, title=f"Confusion — {result.split}"):
            written.append(cm_path)
        pc_path = artifacts.figures / f"per_class_f1_{result.split}.png"
        if per_class_f1_bars(result, pc_path):
            written.append(pc_path)
    return written
