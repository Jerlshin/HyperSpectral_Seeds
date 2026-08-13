"""Scoring a prediction array, with the uncertainty attached (CHANGES §19.4).

The audited project reported a single macro-F1 with no interval, on a split it
had also selected on, from one seed. CHANGES §4.5 prices the two omissions:
sampling noise on 1,294 samples is **±0.020** at 95%, and a running maximum over
~944 correlated selection events is worth an expected **+0.042**. Every delta the
run reported was smaller than either.

So a metric here is never a bare float. :class:`ClassificationResult` carries
macro-F1, balanced accuracy, per-class recall, the full confusion matrix and a
percentile bootstrap CI, and :func:`paired_bootstrap_delta` is what compares two
arms — paired, because the same resample scored for both arms removes the
split's own variance, which is the only comparison that answers "is this gap
outside noise?".

Macro-F1 is computed over a fixed ``labels=range(num_classes)`` rather than over
the labels present in a resample, so every replicate averages over the same 90
denominators. Without that, a bootstrap sample missing a rare class silently
changes the denominator and the interval is of the wrong quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

#: Default resample count. 2,000 is enough for a stable 95% percentile interval
#: on a ~1,300-sample split; the estimator's own Monte-Carlo error at that count
#: is well under the ±0.020 sampling noise it is measuring.
DEFAULT_N_BOOT: int = 2000

#: Two-sided coverage of the reported interval.
DEFAULT_ALPHA: float = 0.05


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval."""

    lo: float
    hi: float
    n_boot: int
    alpha: float

    def as_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "n_boot": self.n_boot, "alpha": self.alpha}

    def __str__(self) -> str:
        return f"[{self.lo:.4f}, {self.hi:.4f}]"


@dataclass
class ClassificationResult:
    """Everything CHANGES §19.4 requires be reported for one scored split.

    *"Macro-F1 primary; balanced accuracy, per-class recall, and the full 90×90
    confusion matrix alongside."*
    """

    split: str
    n_samples: int
    num_classes: int
    macro_f1: float
    weighted_f1: float
    accuracy: float
    balanced_accuracy: float
    per_class_f1: dict[int, float]
    per_class_precision: dict[int, float]
    per_class_recall: dict[int, float]
    per_class_support: dict[int, int]
    confusion: npt.NDArray[Any]
    macro_f1_ci: Interval | None = None
    #: Free-form run identity — architecture, protocol, seed, fold. Carried here
    #: so an aggregated table can be built from the result files alone.
    context: dict[str, Any] = field(default_factory=dict)

    def scalars(self, prefix: str) -> dict[str, float]:
        """Tracker tags for the headline numbers, e.g. ``test/f1_macro``."""
        tags = {
            f"{prefix}/f1_macro": self.macro_f1,
            f"{prefix}/f1_weighted": self.weighted_f1,
            f"{prefix}/acc": self.accuracy,
            f"{prefix}/balanced_acc": self.balanced_accuracy,
        }
        if self.macro_f1_ci is not None:
            tags[f"{prefix}/f1_macro_ci_lo"] = self.macro_f1_ci.lo
            tags[f"{prefix}/f1_macro_ci_hi"] = self.macro_f1_ci.hi
        return tags

    def per_class_rows(self) -> list[dict[str, Any]]:
        """One row per class — the table the hard-class analysis reads."""
        return [
            {
                "class": c,
                "f1": self.per_class_f1[c],
                "precision": self.per_class_precision[c],
                "recall": self.per_class_recall[c],
                "support": self.per_class_support[c],
            }
            for c in sorted(self.per_class_f1)
        ]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form. The confusion matrix is written separately."""
        return {
            "split": self.split,
            "n_samples": self.n_samples,
            "num_classes": self.num_classes,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1_ci": self.macro_f1_ci.as_dict() if self.macro_f1_ci else None,
            "per_class": self.per_class_rows(),
            "context": dict(self.context),
        }


def macro_f1(
    targets: npt.NDArray[Any], preds: npt.NDArray[Any], num_classes: int
) -> float:
    """Macro-F1 over a **fixed** label set — see the module docstring."""
    return float(
        f1_score(
            targets, preds, average="macro", zero_division=0, labels=list(range(num_classes))
        )
    )


def bootstrap_macro_f1(
    targets: npt.NDArray[Any],
    preds: npt.NDArray[Any],
    num_classes: int,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI on macro-F1, resampling patches with replacement.

    The resample is over *patches*, which is the right unit for the sampling
    noise on a fixed split. It is **not** a confidence interval on the
    protocol: under ``grouped`` the deeper uncertainty is that there are only
    two acquisition bundles per class, and no amount of resampling patches
    within a held-out bundle speaks to that. Report this interval alongside the
    fold-to-fold range, never instead of it (CHANGES §19.1, constraint 3).
    """
    rng = np.random.default_rng(seed)
    n = len(targets)
    scores = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = macro_f1(targets[idx], preds[idx], num_classes)
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(lo=float(lo), hi=float(hi), n_boot=n_boot, alpha=alpha)


def paired_bootstrap_delta(
    targets: npt.NDArray[Any],
    preds_a: npt.NDArray[Any],
    preds_b: npt.NDArray[Any],
    num_classes: int,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, Interval]:
    """``(delta, CI)`` for ``macro_f1(b) - macro_f1(a)``, paired on the resample.

    Paired is not a refinement here, it is the whole point: the two arms are
    scored on the *same* patches, so scoring both on each resample cancels the
    split's own variance and leaves the variance of the difference. An unpaired
    comparison of two intervals that overlap says nothing either way, which is
    how a +0.005 gap survived unexamined for 19 hours of compute.

    Returns:
        The observed delta and its interval. A delta whose interval spans zero
        is not evidence of an improvement.
    """
    rng = np.random.default_rng(seed)
    n = len(targets)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t = targets[idx]
        deltas[i] = macro_f1(t, preds_b[idx], num_classes) - macro_f1(t, preds_a[idx], num_classes)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    observed = macro_f1(targets, preds_b, num_classes) - macro_f1(targets, preds_a, num_classes)
    return observed, Interval(lo=float(lo), hi=float(hi), n_boot=n_boot, alpha=alpha)


def score(
    targets: npt.NDArray[Any],
    preds: npt.NDArray[Any],
    num_classes: int,
    split: str = "test",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    context: dict[str, Any] | None = None,
) -> ClassificationResult:
    """Score one prediction array into a :class:`ClassificationResult`.

    Args:
        n_boot: ``0`` skips the bootstrap, which is what the per-epoch path
            wants — 2,000 resamples of a macro-F1 is seconds, not milliseconds,
            and nothing selects on an interval.
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    labels = list(range(num_classes))
    precision, recall, per_f1, support = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    return ClassificationResult(
        split=split,
        n_samples=int(len(targets)),
        num_classes=num_classes,
        macro_f1=macro_f1(targets, preds, num_classes),
        weighted_f1=float(f1_score(targets, preds, average="weighted", zero_division=0)),
        accuracy=float((targets == preds).mean()) if len(targets) else 0.0,
        balanced_accuracy=float(balanced_accuracy_score(targets, preds)),
        per_class_f1={i: float(v) for i, v in enumerate(per_f1)},
        per_class_precision={i: float(v) for i, v in enumerate(precision)},
        per_class_recall={i: float(v) for i, v in enumerate(recall)},
        per_class_support={i: int(v) for i, v in enumerate(support)},
        confusion=np.asarray(confusion_matrix(targets, preds, labels=labels)),
        macro_f1_ci=(
            bootstrap_macro_f1(targets, preds, num_classes, n_boot, alpha, seed)
            if n_boot > 0
            else None
        ),
        context=dict(context or {}),
    )


def mean_and_range(values: list[float]) -> dict[str, float]:
    """``{"mean", "min", "max", "range", "sd", "n"}`` over repeated runs.

    CHANGES §19.4: *"Every number as mean ± range over 3 seeds. A single-seed
    delta is not a result."* The range rather than a standard error, because 3
    seeds × 2 folds is too few for an SE to mean much and the range is the
    honest summary of six numbers.
    """
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "sd": 0.0, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        # ddof=1: this is a sample of runs, not the population of them. Zero for
        # a single run, which is correct and also a useful tell in a table.
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }
