"""The classical estimators that stand in for the network, and their honest limits.

Why proxies at all
──────────────────
The full sweep is roughly ten thousand fits. At ~45 minutes per neural run that
is eight GPU-years; on mean spectra with a linear model it is hours on a laptop.
There is no version of this study that answers "which of twelve methods, at
which of twenty budgets" with the deployed model, so the choice is between a
proxy study with a neural confirmation stage and no study at all.

What a proxy conclusion is, and is not
──────────────────────────────────────
These models read the **foreground-masked mean spectrum**: one number per band
per patch. That representation discards every spatial and textural cue, which
is not a small omission — the project's own bracket is 0.5916 for LDA on mean
spectra against ~0.845 for the full model under the same (leaky) protocol, so
roughly 25 points of the deployed number live in structure these estimators
cannot see (CHANGES §19.4).

The consequence for *this* study is specific and directional. A mean-spectrum
model saturates when the spectrum's usable degrees of freedom are exhausted; a
network that reads band × space interactions has more ways to use a band, so
CHANGES F-3 predicts its curve keeps rising past the proxies' plateau. So:

* **A proxy plateau is a lower bound on the useful budget, not an upper one.**
  "The proxies stop improving at k" supports "k is enough for a mean-spectrum
  model" and does not support "k is enough".
* **Method rankings transfer better than budgets do.** Which bands carry class
  information is a property of the spectra; how many a model needs is a
  property of the model. The neural stage tests the budget claim specifically.

Three families, deliberately
────────────────────────────
``lda`` is generative and linear, ``linsvc`` discriminative and linear,
``extratrees`` nonlinear and non-parametric. A budget conclusion that holds in
all three is a conclusion about the spectra; one that holds in a single family
is a conclusion about that family, and the tables report per-proxy rather than
averaging over them.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

_log = logging.getLogger("spectralquadnet.bandstudy.proxies")


@dataclass(frozen=True)
class ProxySpec:
    """One proxy estimator, named and described."""

    name: str
    family: str
    #: Built per call, never stored fitted — a shared estimator across cells
    #: would carry the previous cell's coefficients into the next.
    factory: Callable[[int], Any]
    note: str
    #: Roughly how the fit scales with the band count, for the compute table.
    cost_note: str


def _lda(seed: int) -> Any:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # solver="svd" uses a pseudo-inverse, so it survives the k > n_per_class
    # regime that the 90-class problem is in at every budget above ~40.
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd", tol=1e-4))


def _linsvc(seed: int) -> Any:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    # C=0.1 is the value the repository's own baseline and band selector use.
    # Holding it fixed across budgets is the point: a per-budget tuned C would
    # make the curve a curve through a tuning procedure rather than through k,
    # and tuning it on anything but calib would leak.
    return make_pipeline(StandardScaler(), LinearSVC(C=0.1, max_iter=5000, random_state=seed))


def _extratrees(seed: int) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    # No scaler: trees are invariant to monotone per-feature transforms, and
    # inserting one would only cost time.
    return ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


PROXIES: dict[str, ProxySpec] = {
    "lda": ProxySpec(
        "lda",
        "generative-linear",
        _lda,
        "LDA on the mean spectrum — the repository's most important baseline "
        "(CHANGES §19.4), recomputed here at every budget.",
        "O(k^2 n) via SVD; the only proxy whose cost grows superlinearly in k.",
    ),
    "linsvc": ProxySpec(
        "linsvc",
        "discriminative-linear",
        _linsvc,
        "LinearSVC, C=0.1 fixed across budgets. More robust than LDA off-Gaussian.",
        "O(k n) per one-vs-rest problem, 90 of them.",
    ),
    "extratrees": ProxySpec(
        "extratrees",
        "nonlinear-ensemble",
        _extratrees,
        "Extremely randomised trees — the only proxy that can use a band "
        "conditionally on another, so a budget conclusion that holds here too "
        "is not an artefact of linearity.",
        "O(sqrt(k) n log n) per tree; nearly flat in k.",
    ),
}


def get(name: str) -> ProxySpec:
    """Look up a proxy spec.

    Raises:
        KeyError: With the available names.
    """
    try:
        return PROXIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown proxy estimator {name!r}. Available: {', '.join(sorted(PROXIES))}"
        ) from None


@dataclass
class ProxyScore:
    """One fit-and-score, with its cost and any failure recorded."""

    proxy: str
    split: str
    macro_f1: float = float("nan")
    accuracy: float = float("nan")
    balanced_accuracy: float = float("nan")
    fit_seconds: float = 0.0
    score_seconds: float = 0.0
    n_train: int = 0
    n_eval: int = 0
    n_features: int = 0
    failure: str | None = None
    ci: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failure is None and np.isfinite(self.macro_f1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "proxy": self.proxy,
            "split": self.split,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "fit_seconds": round(self.fit_seconds, 4),
            "score_seconds": round(self.score_seconds, 4),
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_features": self.n_features,
            "failure": self.failure,
            "ci": self.ci,
            **self.extra,
        }


def fit_and_score(
    name: str,
    x_train: npt.NDArray[Any],
    y_train: npt.NDArray[Any],
    x_eval: npt.NDArray[Any],
    y_eval: npt.NDArray[Any],
    num_classes: int,
    split: str,
    seed: int = 0,
    n_boot: int = 0,
) -> ProxyScore:
    """Fit one proxy on ``x_train`` and score it on ``x_eval``.

    Macro-F1 over a **fixed** ``labels=range(num_classes)``, via
    :func:`spectralquadnet.reporting.metrics.macro_f1`, so every cell in the
    study averages over the same 90 denominators. Without that, a budget whose
    predictions happen to miss a rare class entirely would be scored on 89
    classes and look better for it.

    Args:
        n_boot: Bootstrap resamples for a percentile CI on macro-F1. ``0`` — the
            default — skips it: the sweep runs thousands of cells and its
            uncertainty comes from the replicate spread, which is a better
            estimate here than a within-split resample. The ``confirm`` stage
            turns it on, where there is one number per configuration and the
            within-split noise is the thing being reported.

    Returns:
        A :class:`ProxyScore`, with ``failure`` set rather than an exception
        raised — a proxy that will not converge on one cell must not take the
        other nine thousand with it.
    """
    from spectralquadnet.reporting.metrics import bootstrap_macro_f1, macro_f1

    spec = get(name)
    score = ProxyScore(
        proxy=name,
        split=split,
        n_train=int(len(x_train)),
        n_eval=int(len(x_eval)),
        n_features=int(x_train.shape[1]) if x_train.ndim == 2 else 0,
    )
    try:
        # Both the fit and the predict are silenced, and numpy's floating-point
        # errors with them. At k = 1 on 90 classes LDA's scatter matrix is
        # singular and `predict` overflows the discriminant — sklearn emits one
        # ConvergenceWarning and numpy three RuntimeWarnings *per cell*, which
        # over ten thousand cells buries every line the operator needs to see.
        # The degenerate cells are not hidden: they are recorded, and the
        # analysis flags them.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            model = spec.factory(seed)
            started = time.perf_counter()
            model.fit(x_train, y_train)
            score.fit_seconds = time.perf_counter() - started

            started = time.perf_counter()
            preds = np.asarray(model.predict(x_eval))
            score.score_seconds = time.perf_counter() - started

            from sklearn.metrics import balanced_accuracy_score

            score.macro_f1 = macro_f1(np.asarray(y_eval), preds, num_classes)
            score.accuracy = float((np.asarray(y_eval) == preds).mean())
            score.balanced_accuracy = float(balanced_accuracy_score(y_eval, preds))
            if n_boot > 0:
                interval = bootstrap_macro_f1(
                    np.asarray(y_eval), preds, num_classes, n_boot=n_boot, seed=seed
                )
                score.ci = {"lo": interval.lo, "hi": interval.hi, "n_boot": interval.n_boot}
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        _log.error("proxy %s failed on %s: %s", name, split, exc)
        score.failure = f"{type(exc).__name__}: {exc}"
    return score
