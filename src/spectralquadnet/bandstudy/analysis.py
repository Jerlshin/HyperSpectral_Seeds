"""Reading the accumulated results, and saying what they show.

This module is where the study stops collecting numbers and starts making
claims, so its design constraints are about what it is *not* allowed to do.

**It cannot see the held-out split.** :func:`load_records` reads the proxy
stage's records, which are ``split == "calib"`` by construction, and every
decision function below takes those records as its only numeric input. The
confirm stage's records are loaded separately and only by :func:`confirm_table`,
which reports and never recommends.

**It cannot assume the answer.** The decision rules are pre-registered in
:class:`~spectralquadnet.bandstudy.config.BandStudyConfig` — a plateau
tolerance, a stability floor, a margin over the null — and every one of them
can return "no", including:

* *no plateau* — :func:`classify_trend` returns ``monotone_increasing`` when the
  curve is still climbing at the full band count, and the recommendation then
  says "use all 256 bands", which is a legitimate outcome of this experiment
  and the one CHANGES F-3 predicts;
* *no effective method* — if nothing beats the random null by ``null_margin``,
  :func:`rank_methods` marks every method ineffective and the recommendation
  falls back to ``uniform`` with that stated as the finding;
* *no stable selection* — a method whose replicates disagree is flagged, and a
  wavelength claim is not made from it.

**It reports the shape of the curve rather than its maximum.** A maximum over
~500 correlated cells is worth an estimated +0.042 to whoever reads it
(CHANGES §4.5), which is larger than most of the effects here.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from spectralquadnet.bandstudy import methods as bsmethods
from spectralquadnet.bandstudy import stability as bsstability
from spectralquadnet.bandstudy.config import BandStudyConfig
from spectralquadnet.bandstudy.pipeline import CANONICAL, load_inputs, load_selection
from spectralquadnet.bandstudy.store import RECORDS

_log = logging.getLogger("spectralquadnet.bandstudy.analysis")

#: The metric every decision reads. Macro-F1 rather than accuracy: the 90
#: classes are near-balanced but not exactly, and macro-F1 is what every other
#: table in this repository reports, so a band budget chosen on a different
#: metric from the one the model is scored on would not be a band budget for
#: that model.
METRIC = "macro_f1"


# ══════════════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════════════


def load_records(cfg: BandStudyConfig) -> pd.DataFrame:
    """The proxy stage's records as a frame.

    Raises:
        FileNotFoundError: The proxy stage has not run.
    """
    path = cfg.proxy_dir / RECORDS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python -m spectralquadnet.bandstudy.cli proxy` first."
        )
    frame = pd.read_json(path, lines=True)
    if frame.empty:
        raise FileNotFoundError(f"{path} is empty — the proxy stage recorded nothing.")
    frame["rep"] = frame["rep"].astype(str)
    off_protocol = frame[frame["split"] != "calib"]
    if not off_protocol.empty:
        # Defensive: the proxy stage only ever writes calib. If anything else
        # is in here, a decision made from this frame would be a decision made
        # on held-out data, and dropping the rows loudly beats using them.
        _log.error(
            "%d proxy records are not on the calib split and were DROPPED — decisions must "
            "never read a reported split",
            len(off_protocol),
        )
        frame = frame[frame["split"] == "calib"]
    return frame


def load_confirm(cfg: BandStudyConfig) -> pd.DataFrame:
    """The confirm stage's held-out records, or an empty frame."""
    path = cfg.confirm_dir / RECORDS
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_json(path, lines=True)
    return frame if not frame.empty else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
#  Curves
# ══════════════════════════════════════════════════════════════════════


def budget_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (fold, method, proxy, budget), with its uncertainty.

    Two distinct uncertainties are kept apart rather than pooled:

    ``draw_sd``
        Spread across the ``random`` null's independent draws at one budget.
        This is the variability of *the choice* when the choice is arbitrary,
        and it is the reference scale for "is this method's advantage real?".
    ``rep_sd``
        Spread across replicates — different stratified subsamples of the same
        training split, each re-selecting and re-fitting. This is the
        variability of the whole procedure under resampling, and it is the one
        a delta between two methods has to clear.

    ``score`` is always the **canonical** selection's value (the full training
    split), never a mean over replicates: the replicates exist to measure
    spread, and averaging them in would report a number no single runnable
    configuration produces.
    """
    canonical = frame[frame["rep"] == CANONICAL]
    reps = frame[frame["rep"] != CANONICAL]

    base = canonical.groupby(["fold", "method", "proxy", "budget"], as_index=False).agg(
        score=(METRIC, "mean"),
        draw_sd=(METRIC, lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
        n_draws=(METRIC, "size"),
        fit_seconds=("fit_seconds", "mean"),
        selection_seconds=("selection_seconds", "mean"),
        n_failed=("failure", lambda s: int(s.notna().sum())),
    )
    if not reps.empty:
        spread = reps.groupby(["fold", "method", "proxy", "budget"], as_index=False).agg(
            rep_mean=(METRIC, "mean"),
            rep_sd=(METRIC, lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
            rep_min=(METRIC, "min"),
            rep_max=(METRIC, "max"),
            n_reps=(METRIC, "size"),
        )
        base = base.merge(spread, on=["fold", "method", "proxy", "budget"], how="left")
    else:
        for column in ("rep_mean", "rep_sd", "rep_min", "rep_max"):
            base[column] = np.nan
        base["n_reps"] = 0

    base["family"] = base["method"].map(
        lambda m: bsmethods.METHODS[m].family if m in bsmethods.METHODS else "?"
    )
    base["uncertainty"] = base["rep_sd"].fillna(base["draw_sd"])
    return base.sort_values(["fold", "proxy", "method", "budget"]).reset_index(drop=True)


def noise_scale(curves: pd.DataFrame) -> float:
    """A single pooled σ for the study, used where a curve has none of its own.

    The median of every cell's replicate sd. A median rather than a mean
    because a handful of degenerate cells (a proxy that failed to converge at
    k = 1) would otherwise set the scale every later comparison is judged
    against.
    """
    values = curves["uncertainty"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else 0.0


# ══════════════════════════════════════════════════════════════════════
#  Trend classification
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrendVerdict:
    """What one budget curve does, and where it stops doing it."""

    fold: int
    method: str
    proxy: str
    n_points: int
    min_budget: int
    max_budget: int
    peak_budget: int
    peak_score: float
    score_at_max: float
    score_at_min: float
    #: Smallest budget within ``plateau_tol`` of the peak.
    plateau_budget: int
    plateau_score: float
    #: Maximum-distance-from-chord knee of the normalised curve — a second,
    #: independent locator that does not depend on the tolerance.
    knee_budget: int
    #: ``monotone_increasing`` | ``saturating`` | ``peaked_declining`` | ``flat``
    shape: str
    #: True only when the curve extends past the plateau. A plateau at the last
    #: recorded budget is the endpoint of a truncated curve, which is the exact
    #: defect CHANGES M-14 found in both shipped selections.
    plateau_demonstrable: bool
    #: Best score anywhere past the plateau, minus the score at it. Positive
    #: means more bands still help.
    headroom_past_plateau: float
    #: Fraction of the full-budget score reached at the plateau.
    retained_fraction: float
    #: Marginal macro-F1 per doubling of the band count, over the last octave.
    last_octave_gain: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _knee(budgets: Sequence[int], scores: Sequence[float]) -> int:
    """The point of maximum distance from the chord joining the curve's ends.

    A tolerance-free locator, in log-budget space because the budgets are
    log-spaced and a linear-x knee on a log-x grid is dominated by the largest
    budget's spacing. Reported beside the tolerance-based plateau so that a
    reader can see whether the two agree; when they disagree the curve does not
    have a clean knee, which is itself the finding.
    """
    if len(budgets) < 3:
        return int(budgets[0])
    x = np.log2(np.asarray(budgets, dtype=float))
    y = np.asarray(scores, dtype=float)
    x = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    span = y.max() - y.min()
    if span < 1e-12:
        return int(budgets[0])
    y = (y - y[0]) / span
    chord = y[0] + (y[-1] - y[0]) * x
    return int(np.asarray(budgets)[int(np.argmax(y - chord))])


def classify_trend(
    fold: int,
    method: str,
    proxy: str,
    budgets: Sequence[int],
    scores: Sequence[float],
    tol: float,
    sigma: float,
) -> TrendVerdict:
    """Decide what a budget curve does.

    The four shapes are exhaustive and each is a real possible answer:

    ``monotone_increasing``
        The curve is still climbing at the largest budget evaluated. More bands
        help; the study has not found a sufficient budget, only a lower bound.
    ``saturating``
        The curve reaches within ``tol`` of its peak well before the largest
        budget and stays there. Reduction is free.
    ``peaked_declining``
        The curve peaks in the interior and falls by more than ``max(tol, 2σ)``
        by the largest budget. More bands actively hurt — the classic
        curse-of-dimensionality signature for a fixed-capacity model at ~40
        samples per class.
    ``flat``
        The whole curve spans less than ``tol``. The band count is not what
        limits this estimator, and reading a plateau off it would be reading
        noise.
    """
    order = np.argsort(np.asarray(budgets))
    ks = [int(budgets[i]) for i in order]
    ys = [float(scores[i]) for i in order]

    peak_i = int(np.argmax(ys))
    peak = ys[peak_i]
    threshold = peak - tol
    plateau_i = next((i for i, v in enumerate(ys) if v >= threshold), peak_i)
    past = ys[plateau_i + 1 :]
    drop = peak - ys[-1]
    material = max(tol, 2.0 * sigma)
    spread = max(ys) - min(ys)

    if spread < tol:
        shape = "flat"
        reason = (
            f"the whole curve spans {spread:.4f} < tol {tol:.3f}: the band count is not what "
            "limits this estimator, so no plateau can be read off it"
        )
    elif drop > material:
        shape = "peaked_declining"
        reason = (
            f"peaks at k={ks[peak_i]} ({peak:.4f}) and falls {drop:.4f} by k={ks[-1]}, more "
            f"than max(tol, 2σ)={material:.4f} — extra bands cost accuracy here"
        )
    elif plateau_i >= len(ks) - 1:
        shape = "monotone_increasing"
        reason = (
            f"still within tol of its peak only at the last budget k={ks[-1]}: the curve has "
            "not been shown to plateau, and the sufficient budget is at or beyond the range "
            "evaluated"
        )
    else:
        shape = "saturating"
        reason = (
            f"reaches within {tol:.3f} of its peak {peak:.4f} at k={ks[plateau_i]}, with "
            f"{len(past)} larger budgets recorded past it"
        )

    # Marginal return over the final doubling of k that the grid contains.
    last_octave = float("nan")
    target = ks[-1] / 2.0
    prior = [i for i, k in enumerate(ks) if k <= target]
    if prior:
        i = prior[-1]
        last_octave = (ys[-1] - ys[i]) / max(math.log2(ks[-1] / max(ks[i], 1)), 1e-9)

    return TrendVerdict(
        fold=fold,
        method=method,
        proxy=proxy,
        n_points=len(ks),
        min_budget=ks[0],
        max_budget=ks[-1],
        peak_budget=ks[peak_i],
        peak_score=peak,
        score_at_max=ys[-1],
        score_at_min=ys[0],
        plateau_budget=ks[plateau_i],
        plateau_score=ys[plateau_i],
        knee_budget=_knee(ks, ys),
        shape=shape,
        plateau_demonstrable=plateau_i < len(ks) - 1,
        headroom_past_plateau=float(max(past) - ys[plateau_i]) if past else 0.0,
        retained_fraction=float(ys[plateau_i] / ys[-1]) if ys[-1] > 0 else float("nan"),
        last_octave_gain=float(last_octave),
        reason=reason,
    )


def trend_table(cfg: BandStudyConfig, curves: pd.DataFrame) -> pd.DataFrame:
    """A :class:`TrendVerdict` per (fold, method, proxy)."""
    sigma = noise_scale(curves)
    rows: list[dict[str, Any]] = []
    for (fold, method, proxy), block in curves.groupby(["fold", "method", "proxy"]):
        block = block.sort_values("budget")
        if len(block) < 2:
            continue
        rows.append(
            classify_trend(
                int(fold),
                str(method),
                str(proxy),
                block["budget"].tolist(),
                block["score"].tolist(),
                tol=cfg.plateau_tol,
                sigma=sigma,
            ).as_dict()
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  The null comparison
# ══════════════════════════════════════════════════════════════════════


def null_margins(curves: pd.DataFrame) -> pd.DataFrame:
    """Every method's advantage over a random subset of the same size.

    The single most informative table in the study, and the one most easily
    omitted. 256 VIS-NIR bands with neighbour correlations above 0.99 carry far
    fewer than 256 degrees of freedom, so a random 20-band subset already spans
    most of the usable spectrum. A method that does not beat that has not been
    shown to select; it has been shown to subset.

    ``margin_sd`` is the null's own spread across draws, so ``margin / margin_sd``
    is a z-score against the distribution of arbitrary choices at that budget.
    """
    null = curves[curves["method"] == "random"][
        ["fold", "proxy", "budget", "score", "draw_sd"]
    ].rename(columns={"score": "null_score", "draw_sd": "null_sd"})
    if null.empty:
        return pd.DataFrame()

    merged = curves[curves["method"] != "random"].merge(
        null, on=["fold", "proxy", "budget"], how="left"
    )
    merged["margin"] = merged["score"] - merged["null_score"]
    merged["margin_z"] = merged["margin"] / merged["null_sd"].replace(0.0, np.nan)
    return (
        merged[
            [
                "fold",
                "method",
                "family",
                "proxy",
                "budget",
                "score",
                "null_score",
                "null_sd",
                "margin",
                "margin_z",
            ]
        ]
        .sort_values(["fold", "proxy", "method", "budget"])
        .reset_index(drop=True)
    )


def rank_methods(
    cfg: BandStudyConfig, curves: pd.DataFrame, margins: pd.DataFrame, stability_df: pd.DataFrame
) -> pd.DataFrame:
    """One row per method: how well it scores, how reliably, and how stably.

    Deliberately several criteria rather than one composite score. A single
    ranking number would hide that the methods differ on *different* axes — one
    wins at tiny budgets and loses at large ones, another is the most stable and
    never the best — and "which method" has a different answer depending on
    which of those a downstream user cares about.

    ``effective`` is the study's pass/fail: does this method beat the random
    null by at least ``null_margin`` at some budget, on some proxy? A method
    that never does is reported as ineffective *on this data with these
    proxies*, which is a claim about the method's usefulness here and not about
    the method.
    """
    rows: list[dict[str, Any]] = []
    small = [k for k in sorted(curves["budget"].unique()) if k <= 40]

    for method, block in curves.groupby("method"):
        spec = bsmethods.METHODS.get(str(method))
        margin_block = margins[margins["method"] == method] if not margins.empty else pd.DataFrame()
        stab = (
            stability_df[stability_df["method"] == method]
            if not stability_df.empty
            else pd.DataFrame()
        )
        small_block = block[block["budget"].isin(small)]

        best_margin = (
            float(margin_block["margin"].max()) if not margin_block.empty else float("nan")
        )
        mean_margin = (
            float(margin_block["margin"].mean()) if not margin_block.empty else float("nan")
        )
        rows.append(
            {
                "method": method,
                "family": spec.family if spec else "?",
                "supervised": spec.supervised if spec else None,
                "kind": spec.kind if spec else "?",
                "mean_score_all_budgets": float(block["score"].mean()),
                "mean_score_k_le_40": (
                    float(small_block["score"].mean()) if not small_block.empty else float("nan")
                ),
                "best_score": float(block["score"].max()),
                "best_budget": int(block.loc[block["score"].idxmax(), "budget"]),
                "mean_margin_vs_random": mean_margin,
                "best_margin_vs_random": best_margin,
                "effective": bool(np.isfinite(best_margin) and best_margin >= cfg.null_margin),
                "mean_jaccard_k_le_40": (
                    float(stab[stab["budget"].isin(small)]["mean_jaccard"].mean())
                    if not stab.empty
                    else float("nan")
                ),
                "stable": (
                    bool(
                        stab[stab["budget"].isin(small)]["mean_jaccard"].mean()
                        >= cfg.stability_floor
                    )
                    if not stab.empty
                    and np.isfinite(stab[stab["budget"].isin(small)]["mean_jaccard"].mean())
                    else None
                ),
                "mean_selection_seconds": float(block["selection_seconds"].mean()),
                "n_failed_cells": int(block["n_failed"].sum()),
            }
        )

    frame = pd.DataFrame(rows)
    # Ranked on small budgets, because that is where a selection method can
    # differ from another one at all: by k = 128 every method has most of the
    # spectrum and the ranking is noise.
    return frame.sort_values("mean_score_k_le_40", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  Stability and redundancy
# ══════════════════════════════════════════════════════════════════════


def stability_table(cfg: BandStudyConfig, n_bands: int) -> pd.DataFrame:
    """Pairwise agreement between replicates, per (fold, method, budget).

    ``random`` is included on purpose and is the reference row: its agreement
    is chance agreement by construction, so a named method whose Jaccard sits
    at random's has been shown to select no more reproducibly than a coin.
    """
    rows: list[dict[str, Any]] = []
    for fold in sorted(cfg.folds):
        for method in cfg.methods:
            spec = bsmethods.METHODS.get(method)
            # `random`'s replicates are its draws (it has no others); every
            # other method's are its subsample replicates.
            if method == "random":
                payload = load_selection(cfg, fold, CANONICAL, method)
                per_budget = {
                    int(k): v for k, v in ((payload or {}).get("per_budget") or {}).items()
                }
                groups = {k: sets for k, sets in per_budget.items()}
            else:
                groups = {}
                for rep in [str(r) for r in range(cfg.replicates)]:
                    payload = load_selection(cfg, fold, rep, method)
                    for k, sets in ((payload or {}).get("per_budget") or {}).items():
                        groups.setdefault(int(k), []).append(sets[0])

            for budget, sets in sorted(groups.items()):
                report = bsstability.stability(sets, n_total=n_bands, budget=budget)
                rows.append(
                    {
                        "fold": fold,
                        "method": method,
                        "family": spec.family if spec else "?",
                        **report.as_dict(),
                    }
                )
    return pd.DataFrame(rows)


def cross_fold_agreement(cfg: BandStudyConfig, n_bands: int) -> pd.DataFrame:
    """Do the two folds' canonical selections agree?

    A harder test than replicate stability and a more interesting one: the two
    folds hold out *different acquisition bundles*, so a band set that survives
    both is a band set that does not depend on which tray the model happened to
    train on. Under this dataset's structure — one training bundle per class per
    fold — that is the closest available approximation to "would this
    generalise to a new acquisition?".
    """
    folds = sorted(cfg.folds)
    if len(folds) < 2:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method in cfg.methods:
        payloads = {f: load_selection(cfg, f, CANONICAL, method) for f in folds}
        if any(p is None for p in payloads.values()):
            continue
        budgets = set.intersection(
            *[set(int(k) for k in (p or {}).get("per_budget", {})) for p in payloads.values()]
        )
        for budget in sorted(budgets):
            sets = [(payloads[f] or {})["per_budget"][str(budget)][0] for f in folds]
            report = bsstability.stability(sets, n_total=n_bands, budget=budget)
            rows.append({"method": method, **report.as_dict()})
    return pd.DataFrame(rows)


def redundancy_table(cfg: BandStudyConfig, inputs: Any) -> pd.DataFrame:
    """Correlation, effective rank and spectral coverage of each canonical set.

    The correlation matrix is recomputed per fold on that fold's **training**
    rows: it is a reported diagnostic, and computing it over every row would
    put held-out spectra into the study's output for a number that would barely
    move.
    """
    rows: list[dict[str, Any]] = []
    for fold in sorted(cfg.folds):
        train = inputs.folds[fold].train
        corr = np.nan_to_num(np.corrcoef(np.asarray(inputs.spectra[train], dtype=np.float64).T))
        for method in cfg.methods:
            payload = load_selection(cfg, fold, CANONICAL, method)
            if payload is None or payload.get("failure"):
                continue
            for sets in (payload.get("per_budget") or {}).values():
                report = bsstability.redundancy(sets[0], corr, inputs.wavelengths)
                rows.append({"fold": fold, "method": method, **report.as_dict()})
    return pd.DataFrame(rows)


def wavelength_consensus(
    cfg: BandStudyConfig, inputs: Any, budget: int, supervised_only: bool = True
) -> pd.DataFrame:
    """How often each wavelength is selected, across methods, folds and replicates.

    The evidence behind any "these wavelengths matter for rice variety"
    sentence. Restricted to supervised methods by default because ``uniform``
    and ``pca_loading`` select without reference to the label, so including
    them would dilute a claim about discriminative wavelengths with a claim
    about where the bands are.

    Nothing here is a decision — the frequencies are computed from selections
    made on training rows, and no score enters.
    """
    pool = [
        m
        for m in cfg.methods
        if m != "random"
        and (not supervised_only or (bsmethods.METHODS.get(m) and bsmethods.METHODS[m].supervised))
    ]
    sets: list[list[int]] = []
    for fold in sorted(cfg.folds):
        for method in pool:
            for rep in [CANONICAL] + [str(r) for r in range(cfg.replicates)]:
                payload = load_selection(cfg, fold, rep, method)
                if payload is None or payload.get("failure"):
                    continue
                entry = (payload.get("per_budget") or {}).get(str(budget))
                if entry:
                    sets.append([int(b) for b in entry[0]])
    if not sets:
        return pd.DataFrame()

    frequency = bsstability.selection_frequency(sets, inputs.n_bands)
    frame = pd.DataFrame(
        {
            "band": np.arange(inputs.n_bands),
            "wavelength_nm": np.round(inputs.wavelengths, 2),
            "selection_frequency": np.round(frequency, 4),
            "n_selections": len(sets),
            "budget": budget,
        }
    )
    return frame.sort_values("selection_frequency", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  Flags
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Flag:
    """Something the results say that a reader should not have to find."""

    severity: str  # "critical" | "warning" | "info"
    code: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "evidence": self.evidence,
        }


def detect_flags(
    cfg: BandStudyConfig,
    curves: pd.DataFrame,
    trends: pd.DataFrame,
    margins: pd.DataFrame,
    ranking: pd.DataFrame,
    stability_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    cross_fold: pd.DataFrame,
    n_classes: int = 90,
) -> list[Flag]:
    """Everything worth raising, ordered by how much it changes the conclusion.

    Each check exists because the corresponding mistake is one this project or
    its literature has actually made, or one this study's design makes possible.
    """
    flags: list[Flag] = []
    sigma = noise_scale(curves)

    # ── Trend disagreements ───────────────────────────────────────────
    if not trends.empty:
        shapes = trends.groupby("shape").size().to_dict()
        flags.append(
            Flag(
                "info",
                "trend_census",
                "Curve shapes across (fold × method × proxy): "
                + ", ".join(f"{k} × {v}" for k, v in sorted(shapes.items())),
                {"shapes": shapes, "sigma": sigma},
            )
        )

        not_demonstrable = trends[~trends["plateau_demonstrable"]]
        if len(not_demonstrable) == len(trends) and len(trends):
            flags.append(
                Flag(
                    "critical",
                    "no_plateau_anywhere",
                    "No curve plateaus inside the evaluated range: every one is still within "
                    "tolerance of its peak only at the largest budget. The study has established a "
                    "LOWER bound on the useful band count and not an upper one — reporting a "
                    "reduction from this would repeat CHANGES M-14 with a different number.",
                    {"n_curves": int(len(trends)), "max_budget": int(trends["max_budget"].max())},
                )
            )
        elif len(not_demonstrable):
            flags.append(
                Flag(
                    "warning",
                    "some_plateaus_not_demonstrable",
                    f"{len(not_demonstrable)} of {len(trends)} curves reach their plateau only at "
                    "the last budget, so for those the plateau is the endpoint of the curve rather "
                    "than a feature of it.",
                    {
                        "cells": not_demonstrable[["fold", "method", "proxy"]].to_dict("records")[
                            :20
                        ]
                    },
                )
            )

        declining = trends[trends["shape"] == "peaked_declining"]
        if not declining.empty:
            flags.append(
                Flag(
                    "warning",
                    "more_bands_hurt",
                    f"{len(declining)} curves peak in the interior and fall materially by the "
                    "largest budget — extra bands actively cost accuracy for those estimators. With "
                    "~40 training patches per class this is the expected small-sample behaviour, and "
                    "it means 'use everything' is not automatically safe.",
                    {
                        "cells": declining[
                            ["fold", "method", "proxy", "peak_budget", "score_at_max"]
                        ].to_dict("records")[:20]
                    },
                )
            )

        # Does the budget conclusion depend on the estimator?
        per_proxy = trends.groupby("proxy")["plateau_budget"].median().to_dict()
        if len(per_proxy) > 1:
            lo, hi = min(per_proxy.values()), max(per_proxy.values())
            if hi >= 4 * max(lo, 1):
                flags.append(
                    Flag(
                        "warning",
                        "budget_is_model_dependent",
                        "The plateau budget differs by more than 4× across proxy model families "
                        f"({per_proxy}). 'How many bands are needed' is therefore a property of the "
                        "estimator as much as of the data, and the neural confirmation is not "
                        "optional — the proxies do not agree with each other, let alone with a CNN.",
                        {"median_plateau_by_proxy": per_proxy},
                    )
                )

        per_fold = trends.groupby("fold")["plateau_budget"].median().to_dict()
        if len(per_fold) > 1:
            lo, hi = min(per_fold.values()), max(per_fold.values())
            if hi >= 2 * max(lo, 1):
                flags.append(
                    Flag(
                        "warning",
                        "budget_is_fold_dependent",
                        f"The plateau budget differs by ≥2× between acquisition folds ({per_fold}). "
                        "With one training bundle per class per fold, that is a signal the budget is "
                        "partly a property of which tray was held out.",
                        {"median_plateau_by_fold": per_fold},
                    )
                )

    # ── Method effectiveness ──────────────────────────────────────────
    if not ranking.empty:
        ineffective = ranking[~ranking["effective"]]["method"].tolist()
        if ineffective:
            flags.append(
                Flag(
                    "warning",
                    "methods_no_better_than_random",
                    f"{len(ineffective)} method(s) never beat a random subset of the same size by "
                    f"{cfg.null_margin:.3f} macro-F1 at any budget on any proxy: "
                    f"{', '.join(ineffective)}. On this data, with these estimators, they are not "
                    "selecting — they are subsetting.",
                    {"methods": ineffective, "null_margin": cfg.null_margin},
                )
            )
        named = ranking[~ranking["method"].isin(["random", "uniform"])]
        if not named.empty and not named["effective"].any():
            flags.append(
                Flag(
                    "critical",
                    "no_method_beats_the_null",
                    "NO named selection method beats the random null anywhere. The honest reading is "
                    "that band identity does not matter much on this task at these budgets — only "
                    "band count does — and the cheapest method (uniform) should be preferred.",
                    {},
                )
            )
        best = ranking.iloc[0]
        if str(best["method"]) in ("uniform", "random"):
            flags.append(
                Flag(
                    "warning",
                    "null_wins",
                    f"The best-scoring method at k ≤ 40 is the null `{best['method']}`. Any published "
                    "claim that a particular selection algorithm is needed for this dataset is not "
                    "supported by these results.",
                    {
                        "method": str(best["method"]),
                        "mean_score_k_le_40": float(best["mean_score_k_le_40"]),
                    },
                )
            )

    # ── Stability ─────────────────────────────────────────────────────
    if not stability_df.empty:
        small = stability_df[(stability_df["budget"] <= 40) & (stability_df["method"] != "random")]
        unstable = (
            small.groupby("method")["mean_jaccard"]
            .mean()
            .pipe(lambda s: s[s < cfg.stability_floor])
        )
        if len(unstable):
            flags.append(
                Flag(
                    "warning",
                    "unstable_selections",
                    f"{len(unstable)} method(s) have mean replicate Jaccard below "
                    f"{cfg.stability_floor:.2f} at k ≤ 40: "
                    + ", ".join(f"{m} ({v:.2f})" for m, v in unstable.items())
                    + ". Their scores may be reproducible while their BAND SETS are not, so no "
                    "'these wavelengths matter' claim may be made from them.",
                    {"methods": {str(m): float(v) for m, v in unstable.items()}},
                )
            )
        chance = (
            small[small["method"] == "random"]["mean_jaccard"].mean() if not small.empty else np.nan
        )
        if np.isfinite(chance):
            flags.append(
                Flag(
                    "info",
                    "chance_stability",
                    f"Chance-level replicate agreement at k ≤ 40 is Jaccard ≈ {chance:.3f}; read every "
                    "method's stability against that, not against 0.",
                    {"chance_jaccard": float(chance)},
                )
            )

    if not cross_fold.empty:
        weak = cross_fold[(cross_fold["budget"] <= 40) & (cross_fold["mean_jaccard"] < 0.4)]
        if len(weak):
            flags.append(
                Flag(
                    "warning",
                    "folds_choose_different_bands",
                    f"{len(weak)} (method, budget) pairs select nearly disjoint bands in the two "
                    "acquisition folds (Jaccard < 0.4 at k ≤ 40). The chosen wavelengths depend on "
                    "which tray was held out, which is the acquisition-recognition problem this "
                    "repository exists to measure, appearing one level up in the pipeline.",
                    {"pairs": weak[["method", "budget", "mean_jaccard"]].to_dict("records")[:20]},
                )
            )

    # ── Redundancy ────────────────────────────────────────────────────
    if not redundancy_df.empty:
        small = redundancy_df[redundancy_df["budget"].between(10, 50)]
        if not small.empty:
            worst = small.groupby("method")["rank_efficiency"].mean().sort_values()
            flags.append(
                Flag(
                    "info",
                    "redundancy_census",
                    "Mean rank efficiency (independent directions per selected band) at 10 ≤ k ≤ 50, "
                    "worst first: " + ", ".join(f"{m} {v:.2f}" for m, v in worst.head(5).items()),
                    {"rank_efficiency": {str(m): float(v) for m, v in worst.items()}},
                )
            )
            clustered = small[small["wavelength_coverage"] < 0.35]
            if len(clustered):
                flags.append(
                    Flag(
                        "warning",
                        "narrow_spectral_coverage",
                        f"{len(clustered)} selections span less than 35% of the 385–1006 nm range. A "
                        "set concentrated in one region is fragile to any acquisition change that "
                        "affects that region, whatever it scores.",
                        {
                            "cells": clustered[
                                ["fold", "method", "budget", "wavelength_coverage"]
                            ].to_dict("records")[:20]
                        },
                    )
                )

    # ── Suspicious values ─────────────────────────────────────────────
    ceiling = curves[curves["score"] > 0.98]
    if not ceiling.empty:
        flags.append(
            Flag(
                "critical",
                "implausible_scores",
                f"{len(ceiling)} cells score above 0.98 macro-F1 across {n_classes} classes from "
                "mean spectra alone. On the real corpus the repository's own leaky-protocol "
                "reference for this representation is 0.5916, so a score near ceiling points at the "
                "split, the feature cache or the labels rather than at good bands.",
                {
                    "cells": ceiling[["fold", "method", "proxy", "budget", "score"]].to_dict(
                        "records"
                    )[:20]
                },
            )
        )

    # Scale-free version of "one or two bands should not do most of the job":
    # compare the tiny budgets against the same curve's full-budget score
    # rather than against a fixed macro-F1, so the check means the same thing on
    # a 12-class synthetic cube and on the 90-class corpus.
    full_budget = int(curves["budget"].max())
    reference = curves[curves["budget"] == full_budget][
        ["fold", "method", "proxy", "score"]
    ].rename(columns={"score": "full_score"})
    tiny = curves[curves["budget"] <= 2].merge(
        reference, on=["fold", "method", "proxy"], how="left"
    )
    tiny = tiny[(tiny["score"] > 0.6 * tiny["full_score"]) & (tiny["full_score"] > 0.3)]
    if not tiny.empty:
        flags.append(
            Flag(
                "warning",
                "implausible_at_tiny_budgets",
                f"{len(tiny)} cells reach more than 60% of their own full-budget score using ≤2 "
                f"bands, across {n_classes} classes (chance ≈ {1.0 / max(n_classes, 1):.3f}). One or "
                "two reflectance values should not do most of the work of separating varieties — "
                "check for a per-acquisition brightness offset surviving into the mean spectrum, "
                "which is the strongest single carrier of bundle identity on this dataset.",
                {
                    "cells": tiny[
                        ["fold", "method", "proxy", "budget", "score", "full_score"]
                    ].to_dict("records")[:20]
                },
            )
        )

    failed = curves[curves["n_failed"] > 0]
    if not failed.empty:
        flags.append(
            Flag(
                "warning",
                "failed_cells",
                f"{int(failed['n_failed'].sum())} proxy fits failed and are holes in the tables. "
                "Their reasons are in the records and the stage log.",
                {
                    "cells": failed[["fold", "method", "proxy", "budget", "n_failed"]].to_dict(
                        "records"
                    )[:20]
                },
            )
        )

    if not margins.empty:
        negative = (
            margins.groupby("method")["margin"].mean().pipe(lambda s: s[s < -cfg.null_margin])
        )
        if len(negative):
            flags.append(
                Flag(
                    "warning",
                    "worse_than_random",
                    "Method(s) averaging WORSE than a random subset of the same size: "
                    + ", ".join(f"{m} ({v:+.3f})" for m, v in negative.items())
                    + ". A selection criterion that anti-correlates with usefulness is a finding "
                    "about the criterion.",
                    {"methods": {str(m): float(v) for m, v in negative.items()}},
                )
            )

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(flags, key=lambda f: order.get(f.severity, 3))


# ══════════════════════════════════════════════════════════════════════
#  The recommendation
# ══════════════════════════════════════════════════════════════════════


def recommend(
    cfg: BandStudyConfig,
    curves: pd.DataFrame,
    trends: pd.DataFrame,
    ranking: pd.DataFrame,
    stability_df: pd.DataFrame,
    flags: list[Flag],
) -> dict[str, Any]:
    """Turn the tables into the choices the neural experiments have to make.

    Two budgets rather than one, because they answer different questions:

    ``recommended_budget``
        The **conservative** choice — the largest per-proxy plateau. Buying the
        agreement of every estimator family costs bands and is the right default
        when the next experiment is expensive and the proxies disagree.
    ``aggressive_budget``
        The smallest per-proxy plateau. Worth running as a second arm precisely
        because the gap between the two is the study's own uncertainty about the
        answer, made visible.

    When no curve plateaus, both collapse to the full band count and the
    recommendation says the study found a lower bound rather than an answer.
    """
    full = int(max(cfg.budgets))
    critical = [f.code for f in flags if f.severity == "critical"]

    if trends.empty:
        return {
            "status": "insufficient_data",
            "recommended_budget": full,
            "note": "no trend verdicts were computed; run the proxy stage",
            "confirm_list": [],
        }

    usable = trends[trends["shape"].isin(["saturating", "peaked_declining"])]
    if usable.empty:
        plateau_by_proxy = {p: full for p in sorted(trends["proxy"].unique())}
        status = "no_plateau"
    else:
        plateau_by_proxy = {
            str(p): int(np.ceil(block["plateau_budget"].max()))
            for p, block in usable.groupby("proxy")
        }
        # Any proxy with no plateau at all votes for the full band count; it has
        # not agreed to a reduction and averaging it out would silence it.
        for proxy in trends["proxy"].unique():
            plateau_by_proxy.setdefault(str(proxy), full)
        status = "plateau_found"

    recommended = int(max(plateau_by_proxy.values()))
    aggressive = int(min(plateau_by_proxy.values()))
    grid = sorted(cfg.budgets)
    recommended = min(grid, key=lambda k: (abs(k - recommended), -k))
    aggressive = min(grid, key=lambda k: (abs(k - aggressive), -k))

    # ── Method ────────────────────────────────────────────────────────
    at_budget = curves[curves["budget"] == recommended]
    method_scores = (
        at_budget[at_budget["method"] != "random"]
        .groupby("method")["score"]
        .mean()
        .sort_values(ascending=False)
    )
    effective = set(ranking[ranking["effective"]]["method"]) if not ranking.empty else set()
    stable = set()
    if not stability_df.empty:
        agreement = (
            stability_df[stability_df["budget"] <= max(40, recommended)]
            .groupby("method")["mean_jaccard"]
            .mean()
        )
        stable = set(agreement[agreement >= cfg.stability_floor].index)

    qualified = [m for m in method_scores.index if m in effective and m in stable]
    fallback_reason = ""
    if qualified:
        chosen = qualified[0]
        why = "highest calib macro-F1 at the recommended budget among methods that both beat the random null and select stably"
    elif [m for m in method_scores.index if m in effective]:
        chosen = next(m for m in method_scores.index if m in effective)
        why = "highest scoring method that beats the random null; NO method met the stability floor"
        fallback_reason = "stability_floor_unmet"
    else:
        chosen = "uniform"
        why = (
            "no method beat the random null by the pre-registered margin, so the cheapest and "
            "most reproducible choice — evenly spaced bands — is preferred"
        )
        fallback_reason = "no_effective_method"

    runner_up = [str(m) for m in method_scores.index[:4] if str(m) != chosen][:2]

    confirm_list = [
        {"fold": int(f), "method": chosen, "budget": recommended, "why": "recommended budget"}
        for f in sorted(cfg.folds)
    ]
    if aggressive != recommended:
        confirm_list += [
            {
                "fold": int(f),
                "method": chosen,
                "budget": aggressive,
                "why": "aggressive budget — the smallest any proxy plateaus at",
            }
            for f in sorted(cfg.folds)
        ]
    for alt in runner_up:
        confirm_list += [
            {"fold": int(f), "method": alt, "budget": recommended, "why": "runner-up method"}
            for f in sorted(cfg.folds)
        ]

    return {
        "status": status,
        "recommended_budget": recommended,
        "aggressive_budget": aggressive,
        "full_budget": full,
        "plateau_by_proxy": plateau_by_proxy,
        "recommended_method": chosen,
        "recommended_method_rationale": why,
        "fallback_reason": fallback_reason,
        "runner_up_methods": runner_up,
        "critical_flags": critical,
        "decision_inputs": {
            "metric": METRIC,
            "split_decisions_were_made_on": "calib",
            "plateau_tol": cfg.plateau_tol,
            "stability_floor": cfg.stability_floor,
            "null_margin": cfg.null_margin,
            "sigma_pooled": noise_scale(curves),
        },
        "confirm_list": confirm_list,
        "caveat": (
            "Every number behind this recommendation comes from classical models on "
            "foreground-masked MEAN SPECTRA. That representation discards all spatial "
            "structure, and the project's own bracket puts ~25 macro-F1 points of the deployed "
            "model in exactly what it discards. A proxy plateau is therefore a LOWER bound on "
            "the budget a spatial-spectral network needs (CHANGES F-3). Run the `neural` stage "
            "before treating the budget as settled."
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════


def run_analysis(cfg: BandStudyConfig) -> dict[str, Any]:
    """Compute every table, write them, and return the recommendation payload."""
    from spectralquadnet.reporting.tables import write_table

    inputs = load_inputs(cfg, quiet=True)
    frame = load_records(cfg)
    curves = budget_curves(frame)
    trends = trend_table(cfg, curves)
    margins = null_margins(curves)
    stability_df = stability_table(cfg, inputs.n_bands)
    cross_fold = cross_fold_agreement(cfg, inputs.n_bands)
    redundancy_df = redundancy_table(cfg, inputs)
    ranking = rank_methods(cfg, curves, margins, stability_df)
    flags = detect_flags(
        cfg,
        curves,
        trends,
        margins,
        ranking,
        stability_df,
        redundancy_df,
        cross_fold,
        n_classes=inputs.n_classes,
    )
    recommendation = recommend(cfg, curves, trends, ranking, stability_df, flags)
    consensus = wavelength_consensus(cfg, inputs, recommendation["recommended_budget"])

    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for name, table in {
        "curves": curves,
        "trends": trends,
        "null_margins": margins,
        "stability": stability_df,
        "cross_fold_agreement": cross_fold,
        "redundancy": redundancy_df,
        "method_ranking": ranking,
        "wavelength_frequency": consensus,
    }.items():
        path = cfg.analysis_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = str(path)

    (cfg.analysis_dir / "flags.json").write_text(
        json.dumps([f.as_dict() for f in flags], indent=2, default=str)
    )
    (cfg.analysis_dir / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2, default=str)
    )

    # Publication tables, in the repository's own two-format convention.
    write_table(
        cfg.tables_dir / "method_ranking",
        [
            "method",
            "family",
            "supervised",
            "mean F1 (k≤40)",
            "best F1",
            "at k",
            "margin vs random",
            "effective",
            "mean Jaccard (k≤40)",
            "select s",
        ],
        [
            [
                r["method"],
                r["family"],
                r["supervised"],
                f"{r['mean_score_k_le_40']:.4f}",
                f"{r['best_score']:.4f}",
                r["best_budget"],
                f"{r['best_margin_vs_random']:+.4f}",
                "yes" if r["effective"] else "NO",
                f"{r['mean_jaccard_k_le_40']:.3f}",
                f"{r['mean_selection_seconds']:.2f}",
            ]
            for r in ranking.to_dict("records")
        ],
        caption="Selection methods, ranked on calib macro-F1 at small budgets. "
        "`effective` = beats a random subset of the same size by the pre-registered margin.",
    )
    write_table(
        cfg.tables_dir / "trends",
        [
            "fold",
            "method",
            "proxy",
            "shape",
            "plateau k",
            "knee k",
            "peak k",
            "peak F1",
            "F1 at k_max",
            "plateau demonstrable",
            "headroom past plateau",
        ],
        [
            [
                r["fold"],
                r["method"],
                r["proxy"],
                r["shape"],
                r["plateau_budget"],
                r["knee_budget"],
                r["peak_budget"],
                f"{r['peak_score']:.4f}",
                f"{r['score_at_max']:.4f}",
                "yes" if r["plateau_demonstrable"] else "NO",
                f"{r['headroom_past_plateau']:+.4f}",
            ]
            for r in trends.to_dict("records")
        ],
        caption="What each budget curve does. `plateau demonstrable` is NO whenever the plateau "
        "is the last point on the curve — the defect CHANGES M-14 identified.",
    )
    if not consensus.empty:
        top = consensus.head(25)
        write_table(
            cfg.tables_dir / "consensus_wavelengths",
            ["band", "λ (nm)", "selection frequency", "n selections"],
            [
                [
                    r["band"],
                    f"{r['wavelength_nm']:.1f}",
                    f"{r['selection_frequency']:.3f}",
                    r["n_selections"],
                ]
                for r in top.to_dict("records")
            ],
            caption=f"Most consistently selected bands at k={recommendation['recommended_budget']}, "
            "over supervised methods × folds × replicates.",
        )

    return {
        "recommendation": recommendation,
        "flags": [f.as_dict() for f in flags],
        "tables": written,
        "n_cells": int(len(frame)),
        "n_curves": int(len(trends)),
    }


def consensus_regions(
    consensus: pd.DataFrame,
    wavelengths: Any,
    threshold: float = 0.5,
    max_gap_nm: float | None = None,
) -> list[tuple[float, float, int]]:
    """Contiguous wavelength regions selected at least ``threshold`` of the time.

    Args:
        max_gap_nm: Bands further apart than this start a new region. ``None``
            derives it from the cube's own median band spacing (six steps),
            which is what makes the merge mean the same thing on a 2.4 nm grid
            and on a 20 nm one. A fixed constant merged nothing on a coarse
            cube and produced a "region" table of one-band rows.
    """
    if consensus.empty:
        return []
    wl = np.asarray(wavelengths, dtype=float)
    if max_gap_nm is None:
        spacing = float(np.median(np.diff(np.sort(wl)))) if len(wl) > 1 else 1.0
        max_gap_nm = 6.0 * spacing
    bands = consensus[consensus["selection_frequency"] >= threshold]["band"].tolist()
    return bsstability.contiguous_regions(bands, wl, max_gap_nm=max_gap_nm)


def confirm_table(cfg: BandStudyConfig) -> pd.DataFrame:
    """The held-out results, tidied for reporting. **Never** an input to a decision."""
    frame = load_confirm(cfg)
    if frame.empty:
        return frame
    keep = [
        "fold",
        "method",
        "budget",
        "proxy",
        "why",
        "n_bands",
        METRIC,
        "balanced_accuracy",
        "accuracy",
        "ci",
        "failure",
    ]
    return (
        frame[[c for c in keep if c in frame.columns]]
        .sort_values(["fold", "proxy", "budget", "method"])
        .reset_index(drop=True)
    )


def load_flags(cfg: BandStudyConfig) -> list[dict[str, Any]]:
    path = cfg.analysis_dir / "flags.json"
    return json.loads(path.read_text()) if path.exists() else []


def load_recommendation(cfg: BandStudyConfig) -> dict[str, Any]:
    """The recommendation payload.

    Raises:
        FileNotFoundError: ``analyse`` has not run.
    """
    path = cfg.analysis_dir / "recommendation.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run `python -m spectralquadnet.bandstudy.cli analyse`."
        )
    return json.loads(path.read_text())  # type: ignore[no-any-return]
