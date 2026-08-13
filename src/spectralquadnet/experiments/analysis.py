"""A9 — what *are* the persistent hard classes? (CHANGES §20, §10.5)

Not a training run. This reads an existing checkpoint and the dataset and asks a
question the audited project never asked, which CHANGES rates the highest-value
experiment in the grid because **it may change the research question rather than
the model**.

The finding it interrogates
───────────────────────────
Classes {41, 49, 51, 52, 70} are the bottom-5 at Stage-1 epoch 46 and still the
bottom-5 at Stage 3 — invariant to 470 epochs, three loss regimes, two samplers
and four difficulty-targeted mechanisms. Class 49 goes 0.28 → 0.24 → ~0.29. The
precision/recall table sharpens it: ``c49: R=0.36 P=0.33``, ``c52: R=0.47
P=0.30``, ``c42: R=0.50 P=0.70`` — recall *and* precision are low for most of
them, which is the signature of **mutual confusion within a cluster**, not of a
threshold in the wrong place. c52 over-claims, c42 under-claims: they are
trading predictions with each other.

The two hypotheses, and why the distinction matters
───────────────────────────────────────────────────
=========================  ==================================================
Hypothesis                 Consequence if true
=========================  ==================================================
**Spectrally inseparable** A well-characterised ceiling. Publishable as a
varieties (genetic)        result; no architecture fixes it.
**Segmentation failure**   A fixable bug. Would explain why eight
(broken/small kernels      difficulty-targeted mechanisms moved nothing, and
failing the shape gate     would change the recommendation completely.
asymmetrically)
=========================  ==================================================

CHANGES §10.5: *"This alternative is cheap to test and has never been tested."*
So :func:`segmentation_audit` is the load-bearing part of this module — it
compares the morphometric distributions and the patch-loss rate of the hard
cluster against matched easy classes, which is exactly what separates the two.

Outputs land under ``<output_dir>/results/a9_*.json`` and
``<output_dir>/figures/a9_*.png``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from spectralquadnet.experiments.baselines import mean_spectra
from spectralquadnet.reporting.artifacts import RunArtifacts
from spectralquadnet.reporting.figures import DPI, _pyplot, available

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

_log = logging.getLogger(__name__)

#: The persistent confusable cluster, from CHANGES §10.5's bottom-5 table across
#: six checkpoints. Overridable — the point is to characterise whichever classes
#: *this* run finds hard, and :func:`hard_classes_from_result` derives them.
DEFAULT_HARD_CLASSES: tuple[int, ...] = (41, 49, 51, 52, 70)

#: Column names of the persisted morphometrics, in order. Mirrors
#: ``data/prep/segmentation.py::MORPHOMETRIC_NAMES``.
MORPHOMETRIC_NAMES: tuple[str, ...] = (
    "area",
    "perimeter",
    "major_axis",
    "minor_axis",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "extent",
)


@dataclass
class ClusterReport:
    """What the hard classes look like, against a matched easy set."""

    hard_classes: list[int]
    easy_classes: list[int]
    #: Mean pairwise cosine between class-mean spectra, within each group and
    #: across. A hard cluster whose within-group similarity far exceeds the
    #: easy group's is the "spectrally inseparable" signature.
    spectral_similarity: dict[str, float] = field(default_factory=dict)
    #: Per-class morphometric means for both groups, plus the pooled
    #: standardised difference. The "segmentation failure" signature is an
    #: offset here — smaller area, lower solidity — not in the spectra.
    morphometrics: dict[str, Any] = field(default_factory=dict)
    #: Patches surviving the shape gate per class. A class-dependent loss rate
    #: is the strongest single tell for segmentation failure.
    patch_counts: dict[str, Any] = field(default_factory=dict)
    #: Off-diagonal confusion mass inside the hard cluster vs. leaving it.
    confusion: dict[str, float] = field(default_factory=dict)
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "hard_classes": self.hard_classes,
            "easy_classes": self.easy_classes,
            "spectral_similarity": self.spectral_similarity,
            "morphometrics": self.morphometrics,
            "patch_counts": self.patch_counts,
            "confusion": self.confusion,
            "verdict": self.verdict,
        }


def hard_classes_from_result(per_class_f1: dict[int, float], k: int = 5) -> list[int]:
    """The ``k`` worst classes of an actual run, ties broken by class id."""
    return [c for c, _ in sorted(per_class_f1.items(), key=lambda kv: (kv[1], kv[0]))[:k]]


def easy_classes_from_result(per_class_f1: dict[int, float], k: int = 5) -> list[int]:
    """The ``k`` best classes — the matched comparison group."""
    return [c for c, _ in sorted(per_class_f1.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _class_mean_spectra(
    x: npt.NDArray[Any], labels: npt.NDArray[Any], classes: list[int]
) -> npt.NDArray[np.float64]:
    """``(len(classes), C)`` — one mean spectrum per class, in the order given.

    float64 because the cosines downstream are compared at four decimal places
    and the input is float32 patch means.
    """
    stacked: npt.NDArray[np.float64] = np.stack(
        [x[labels == c].mean(axis=0) for c in classes]
    ).astype(np.float64)
    return stacked


def _mean_pairwise_cosine(a: npt.NDArray[Any], b: npt.NDArray[Any] | None = None) -> float:
    """Mean cosine similarity between rows of ``a`` (or between ``a`` and ``b``).

    Within-group excludes the diagonal — a class is trivially similar to itself
    and including it would inflate every within-group figure by ``1/n``.
    """
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    if b is None:
        gram = an @ an.T
        off = gram[~np.eye(len(an), dtype=bool)]
        return float(off.mean()) if off.size else 0.0
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return float((an @ bn.T).mean())


def spectral_separability(
    x: npt.NDArray[Any], labels: npt.NDArray[Any], hard: list[int], easy: list[int]
) -> dict[str, float]:
    """Are the hard classes' mean spectra closer to each other than the easy ones'?

    Three numbers. ``within_hard`` well above ``within_easy`` is the
    inseparable-varieties signature: the cluster's members genuinely look alike
    in the 40 bands the pipeline kept.
    """
    hard_means = _class_mean_spectra(x, labels, hard)
    easy_means = _class_mean_spectra(x, labels, easy)
    return {
        "within_hard": _mean_pairwise_cosine(hard_means),
        "within_easy": _mean_pairwise_cosine(easy_means),
        "hard_vs_easy": _mean_pairwise_cosine(hard_means, easy_means),
    }


def segmentation_audit(
    morphology: npt.NDArray[Any] | None,
    labels: npt.NDArray[Any],
    hard: list[int],
    easy: list[int],
    expected_per_class: int = 96,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The test CHANGES §10.5 says has never been run.

    Two independent tells that the hard classes are a *segmentation* problem
    rather than a genetic one:

    * **Patch loss.** 8,624 of 8,640 kernels survived the shape gate
      (``300 < area < 800``, ``ecc > 0.6``, ``solidity > 0.85``) — a silent,
      unaudited 0.19% class-dependent filter. If the missing 16 concentrate in
      the hard classes, the gate is selecting against them.
    * **Morphometric offset.** A standardised difference in area or solidity
      between the two groups says the surviving patches of the hard classes are
      systematically different objects — broken kernels, or kernels the crop
      clipped — not different varieties.

    Returns:
        ``(morphometrics, patch_counts)``, both JSON-serialisable.
    """
    counts = {int(c): int((labels == c).sum()) for c in sorted(set(hard) | set(easy))}
    patch_counts: dict[str, Any] = {
        "per_class": counts,
        "expected_per_class": expected_per_class,
        "hard_mean": float(np.mean([counts[c] for c in hard])) if hard else 0.0,
        "easy_mean": float(np.mean([counts[c] for c in easy])) if easy else 0.0,
        "hard_lost": int(sum(max(expected_per_class - counts[c], 0) for c in hard)),
        "easy_lost": int(sum(max(expected_per_class - counts[c], 0) for c in easy)),
    }

    if morphology is None:
        return (
            {
                "available": False,
                "reason": (
                    "data.morphology_path is unset or the array was never written. "
                    "Re-run scripts/prepare_dataset.py — without it the "
                    "segmentation-failure hypothesis cannot be tested at all, which is "
                    "the whole point of A9."
                ),
            },
            patch_counts,
        )

    morph = np.asarray(morphology, dtype=np.float64)
    hard_rows = morph[np.isin(labels, hard)]
    easy_rows = morph[np.isin(labels, easy)]
    # Pooled standard deviation, so the difference is in units a reader can
    # judge: |d| > 0.8 is a large effect by any convention, and that is the
    # threshold at which "these are different objects" becomes the simpler
    # explanation than "these are different varieties".
    pooled = np.sqrt(0.5 * (hard_rows.var(axis=0, ddof=1) + easy_rows.var(axis=0, ddof=1)))
    cohens_d = (hard_rows.mean(axis=0) - easy_rows.mean(axis=0)) / np.clip(pooled, 1e-12, None)

    names = MORPHOMETRIC_NAMES[: morph.shape[1]]
    morphometrics: dict[str, Any] = {
        "available": True,
        "columns": list(names),
        "hard_mean": {n: float(v) for n, v in zip(names, hard_rows.mean(axis=0), strict=False)},
        "easy_mean": {n: float(v) for n, v in zip(names, easy_rows.mean(axis=0), strict=False)},
        "cohens_d": {n: float(v) for n, v in zip(names, cohens_d, strict=False)},
        "max_abs_cohens_d": float(np.abs(cohens_d).max()),
    }
    return morphometrics, patch_counts


def confusion_concentration(
    confusion: npt.NDArray[Any], hard: list[int]
) -> dict[str, float]:
    """How much of the hard classes' error stays inside the cluster.

    A cluster trading predictions among itself has most of its off-diagonal mass
    on the other members. A cluster that is merely hard scatters it over all 89
    other classes. This is the number that distinguishes "mutual confusion"
    from "generally weak".
    """
    cm = np.asarray(confusion, dtype=np.float64)
    if not hard or cm.size == 0:
        return {}
    rows = cm[hard]
    total_error = float(rows.sum() - np.trace(cm[np.ix_(hard, hard)]))
    inside = float(rows[:, hard].sum() - np.trace(cm[np.ix_(hard, hard)]))
    n_classes = cm.shape[0]
    # What the share would be if the errors were spread uniformly over the other
    # classes — the null this is measured against.
    chance = (len(hard) - 1) / max(n_classes - 1, 1)
    return {
        "error_inside_cluster": inside,
        "error_total": total_error,
        "share_inside": inside / total_error if total_error > 0 else 0.0,
        "chance_share": float(chance),
        "concentration_vs_chance": (
            (inside / total_error) / chance if total_error > 0 and chance > 0 else 0.0
        ),
    }


def verdict(report: ClusterReport) -> str:
    """A stated reading of the evidence — never a silent one.

    Deliberately conservative: it names which hypothesis the numbers favour and
    says when they are inconclusive, rather than picking one. A9 exists to
    inform a research decision, and a diagnostic that always returns an answer
    is not informative.
    """
    lines: list[str] = []
    morph = report.morphometrics
    if morph.get("available") and morph.get("max_abs_cohens_d", 0.0) >= 0.8:
        worst = max(morph["cohens_d"].items(), key=lambda kv: abs(kv[1]))
        lines.append(
            f"SEGMENTATION signal: the hard classes differ from the easy ones by "
            f"d={worst[1]:+.2f} in `{worst[0]}` — a large effect. The surviving patches of "
            "these classes are systematically different objects. This is FIXABLE and would "
            "explain why eight difficulty-targeted mechanisms moved nothing."
        )
    lost = report.patch_counts
    if lost.get("hard_lost", 0) > 2 * max(lost.get("easy_lost", 0), 1):
        lines.append(
            f"SEGMENTATION signal: the shape gate discarded {lost['hard_lost']} kernels from "
            f"the hard classes against {lost['easy_lost']} from the easy ones — a "
            "class-dependent filter, applied silently."
        )

    sim = report.spectral_similarity
    if sim and sim.get("within_hard", 0.0) > sim.get("within_easy", 0.0) + 0.02:
        lines.append(
            f"GENETIC signal: the hard classes' mean spectra are more similar to each other "
            f"(cos={sim['within_hard']:.4f}) than the easy classes' are "
            f"(cos={sim['within_easy']:.4f}). If this dominates, the residual error is a "
            "ceiling to characterise rather than a bug to fix."
        )
    conc = report.confusion
    if conc.get("concentration_vs_chance", 0.0) > 3.0:
        lines.append(
            f"MUTUAL CONFUSION confirmed: {conc['share_inside']:.1%} of the cluster's errors "
            f"stay inside it, {conc['concentration_vs_chance']:.1f}x chance. These classes are "
            "trading predictions with each other, not failing independently."
        )

    if not lines:
        return (
            "INCONCLUSIVE. No signal exceeded its threshold: the morphometric difference is "
            "small, the patch-loss rate is not class-dependent, and the spectral similarity "
            "is unremarkable. The residual error is neither obviously segmentation-induced "
            "nor obviously genetic on this evidence."
        )
    return "\n".join(lines)


def run_analysis(
    cfg: ExperimentConfig | Any,
    per_class_f1: dict[int, float] | None = None,
    confusion: npt.NDArray[Any] | None = None,
    output_dir: str | Path | None = None,
    hard: list[int] | None = None,
    k: int = 5,
) -> ClusterReport:
    """Run A9 and write its report.

    Args:
        per_class_f1: A run's per-class F1, used to derive the hard and easy
            sets. ``None`` falls back to :data:`DEFAULT_HARD_CLASSES` and the
            five best-scoring classes cannot be derived, so a fixed comparison
            set is used instead.
        confusion: The run's confusion matrix, for the concentration measure.
    """
    labels = np.load(cfg.data.labels_path)
    if per_class_f1:
        hard_classes = hard or hard_classes_from_result(per_class_f1, k)
        easy_classes = easy_classes_from_result(per_class_f1, k)
    else:
        hard_classes = list(hard or DEFAULT_HARD_CLASSES)
        # Deterministic complement: the k classes furthest from the hard set by
        # id, which is arbitrary but stated, rather than a silent random pick.
        easy_classes = [c for c in sorted(set(int(v) for v in np.unique(labels))) if c not in hard_classes][:k]

    _log.info("A9: hard=%s  easy=%s", hard_classes, easy_classes)

    x = mean_spectra(cfg.data.patches_data)
    morphology = None
    morph_path = str(getattr(cfg.data, "morphology_path", "") or "")
    if morph_path and Path(morph_path).exists():
        morphology = np.load(morph_path)

    morphometrics, patch_counts = segmentation_audit(morphology, labels, hard_classes, easy_classes)
    report = ClusterReport(
        hard_classes=hard_classes,
        easy_classes=easy_classes,
        spectral_similarity=spectral_separability(x, labels, hard_classes, easy_classes),
        morphometrics=morphometrics,
        patch_counts=patch_counts,
        confusion=confusion_concentration(confusion, hard_classes) if confusion is not None else {},
    )
    report.verdict = verdict(report)

    if output_dir is not None:
        artifacts = RunArtifacts.for_run(output_dir)
        (artifacts.results / "a9_hard_classes.json").write_text(
            json.dumps(report.as_dict(), indent=2)
        )
        if available():
            _plot_spectra(x, labels, hard_classes, easy_classes, artifacts.figures / "a9_spectra.png")
        artifacts.write_manifest({"a9": report.as_dict()})
    return report


def _plot_spectra(
    x: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    hard: list[int],
    easy: list[int],
    path: Path,
) -> Path | None:
    """Class-mean spectra of both groups, overlaid.

    The figure a reader uses to judge the genetic hypothesis by eye: if the hard
    classes' curves lie on top of each other while the easy ones separate, the
    cluster is spectrally degenerate in the 40 bands the pipeline kept.
    """
    plt = _pyplot()
    if plt is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, group, title, cmap in (
        (axes[0], hard, "Hard classes", "autumn"),
        (axes[1], easy, "Easy classes", "winter"),
    ):
        colours = plt.get_cmap(cmap)(np.linspace(0.1, 0.85, max(len(group), 1)))
        for colour, c in zip(colours, group, strict=False):
            ax.plot(x[labels == c].mean(axis=0), color=colour, linewidth=1.4, label=f"class {c}")
        ax.set_title(title)
        ax.set_xlabel("Band index (SPA-selected)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean foreground reflectance (SNV)")
    fig.suptitle("A9 — class-mean spectra: is the hard cluster spectrally degenerate?")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
