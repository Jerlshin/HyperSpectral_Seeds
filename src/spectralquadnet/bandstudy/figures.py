"""Publication figures for the band study, rendered from the CSVs alone.

Every function here reads a frame the analysis stage already wrote, so the whole
figure set can be re-rendered on a different machine, months later, without the
36 GB cube, a GPU, or re-running anything. That property is why the analysis
writes CSVs before it writes pictures.

matplotlib is an optional extra (``pip install -e '.[figures]'``), so every
entry point degrades to "wrote no figures" rather than raising — the same
contract :mod:`spectralquadnet.reporting.figures` keeps.

Style follows that module's: no seaborn, no custom theme, colour-blind-safe
ordering, log-x wherever the budget is an axis (the grid is log-spaced, and on a
linear axis the interesting half of the curve is squeezed into the first
centimetre), and a stated uncertainty on every line that has one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_log = logging.getLogger("spectralquadnet.bandstudy.figures")

DPI: int = 150

#: Okabe–Ito, which stays distinguishable under the common colour-vision
#: deficiencies and prints legibly in greyscale.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#8C564B",
    "#7F7F7F",
    "#2CA02C",
    "#9467BD",
)

#: The nulls are drawn in grey and dashed wherever they appear, so "did anything
#: beat chance?" is answerable from the figure without reading the legend.
NULL_STYLE: dict[str, Any] = {"color": "#666666", "linestyle": "--", "linewidth": 1.6}


def available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _plt() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _colour(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════
#  Individual figures
# ══════════════════════════════════════════════════════════════════════


def budget_curves(curves: pd.DataFrame, path: Path, metric_label: str = "macro-F1") -> Path | None:
    """Score against band budget, one panel per proxy, folds averaged.

    The figure the whole study exists to produce. Three things are deliberate:

    * **Log-x.** The budgets are log-spaced; on a linear axis everything below
      k = 50 collapses into one tick and that is where the curves differ.
    * **The random null as a shaded band**, not a line. Its width is the spread
      across independent draws, so a method inside the band has not been shown
      to beat an arbitrary choice of the same size, and that is visible without
      consulting a table.
    * **The full band count is plotted**, so the reader can see for themselves
      whether the curve has stopped rising — the property both of the
      repository's shipped selections lack.
    """
    plt = _plt()
    if plt is None or curves.empty:
        return None

    proxies = sorted(curves["proxy"].unique())
    fig, axes = plt.subplots(
        1, len(proxies), figsize=(5.4 * len(proxies), 4.6), squeeze=False, sharey=True
    )
    methods = [m for m in sorted(curves["method"].unique()) if m != "random"]

    for ax, proxy in zip(axes[0], proxies, strict=True):
        block = curves[curves["proxy"] == proxy]

        null = (
            block[block["method"] == "random"]
            .groupby("budget")
            .agg(score=("score", "mean"), sd=("draw_sd", "mean"))
        )
        if not null.empty:
            sd = null["sd"].fillna(0.0)
            ax.fill_between(
                null.index,
                null["score"] - sd,
                null["score"] + sd,
                color="#BBBBBB",
                alpha=0.45,
                label="random null ±1 sd",
                zorder=1,
            )
            ax.plot(null.index, null["score"], label="random (null)", zorder=2, **NULL_STYLE)

        for i, method in enumerate(methods):
            line = (
                block[block["method"] == method]
                .groupby("budget")
                .agg(score=("score", "mean"), sd=("uncertainty", "mean"))
            )
            if line.empty:
                continue
            style = NULL_STYLE if method == "uniform" else {"color": _colour(i), "linewidth": 1.7}
            if method == "uniform":
                style = {**NULL_STYLE, "linestyle": ":", "color": "#333333"}
            ax.plot(
                line.index, line["score"], marker="o", markersize=3, label=method, zorder=3, **style
            )
            sd = line["sd"].fillna(0.0)
            if float(sd.sum()) > 0:
                ax.fill_between(
                    line.index,
                    line["score"] - sd,
                    line["score"] + sd,
                    color=style.get("color", "#333333"),
                    alpha=0.12,
                    zorder=1,
                )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Band budget k  (log scale)")
        ax.set_title(f"proxy: {proxy}")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel(f"{metric_label} on calib")
    axes[0][-1].legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle(
        "Band budget vs accuracy — shaded band is the random-subset null, "
        "lines are means over folds (±1 replicate sd)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def per_fold_curves(curves: pd.DataFrame, path: Path) -> Path | None:
    """The same curves, split by acquisition fold rather than averaged.

    Averaging two folds hides the question this dataset is built around: with
    one training bundle per class per fold, a conclusion that only holds on one
    fold is a conclusion about that tray.
    """
    plt = _plt()
    if plt is None or curves.empty:
        return None
    folds = sorted(curves["fold"].unique())
    proxies = sorted(curves["proxy"].unique())
    fig, axes = plt.subplots(
        len(folds),
        len(proxies),
        figsize=(4.6 * len(proxies), 3.6 * len(folds)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    methods = sorted(curves["method"].unique())
    for r, fold in enumerate(folds):
        for c, proxy in enumerate(proxies):
            ax = axes[r][c]
            block = curves[(curves["fold"] == fold) & (curves["proxy"] == proxy)]
            for i, method in enumerate(methods):
                line = block[block["method"] == method].sort_values("budget")
                if line.empty:
                    continue
                style = (
                    NULL_STYLE
                    if method in ("random", "uniform")
                    else {"color": _colour(i), "linewidth": 1.4}
                )
                ax.plot(line["budget"], line["score"], marker=".", label=method, **style)
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.3, which="both")
            if r == 0:
                ax.set_title(f"proxy: {proxy}")
            if c == 0:
                ax.set_ylabel(f"fold {fold}\nmacro-F1 (calib)")
            if r == len(folds) - 1:
                ax.set_xlabel("Band budget k")
    axes[0][-1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Budget curves per acquisition fold — do the folds agree?", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plateau_summary(trends: pd.DataFrame, path: Path) -> Path | None:
    """Where each curve plateaus, and whether the plateau is demonstrable.

    Hollow markers are the cells whose plateau is the last point on their curve
    — for those the "elbow" is an artefact of where the sweep stopped, and the
    figure says so at a glance rather than in a footnote.
    """
    plt = _plt()
    if plt is None or trends.empty:
        return None
    frame = trends.sort_values(["proxy", "plateau_budget"])
    labels = [f"{r.method} · {r.proxy} · f{r.fold}" for r in frame.itertuples()]
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.24 * len(frame) + 1.5)))
    demonstrable = frame["plateau_demonstrable"].to_numpy()
    ax.scatter(
        frame["plateau_budget"][demonstrable],
        y[demonstrable],
        s=34,
        color="#0072B2",
        label="plateau demonstrable",
        zorder=3,
    )
    ax.scatter(
        frame["plateau_budget"][~demonstrable],
        y[~demonstrable],
        s=44,
        facecolors="none",
        edgecolors="#D55E00",
        label="plateau is the curve's endpoint (not demonstrable)",
        zorder=3,
    )
    ax.scatter(
        frame["knee_budget"],
        y,
        s=12,
        color="#009E73",
        marker="x",
        label="knee (chord method)",
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Band budget")
    ax.set_title("Where each curve stops improving")
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def stability_curves(stability: pd.DataFrame, path: Path, floor: float = 0.5) -> Path | None:
    """Replicate agreement against budget, with chance drawn in.

    Jaccard rises with k for purely combinatorial reasons — two random k-subsets
    of 256 bands overlap more as k grows — so the ``random`` line is the only
    honest reference for the others and is always plotted.
    """
    plt = _plt()
    if plt is None or stability.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (method, block) in enumerate(stability.groupby("method")):
        line = block.groupby("budget")["mean_jaccard"].mean()
        style = NULL_STYLE if method == "random" else {"color": _colour(i), "linewidth": 1.5}
        ax.plot(
            line.index,
            line.to_numpy(),
            marker="o",
            markersize=3,
            label=f"{method} (chance)" if method == "random" else str(method),
            **style,
        )
    ax.axhline(
        floor, color="#D55E00", linestyle=":", linewidth=1.2, label=f"stability floor {floor:.2f}"
    )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Band budget k")
    ax.set_ylabel("Mean pairwise Jaccard across replicates")
    ax.set_title("Would a different sample of the same training split have chosen these bands?")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def redundancy_curves(redundancy: pd.DataFrame, path: Path) -> Path | None:
    """Rank efficiency and spectral coverage against budget.

    Two panels because they disagree usefully: a set can be near-orthogonal and
    still occupy one end of the spectrum, and a set can cover the whole range
    and still be six independent directions wearing forty names.
    """
    plt = _plt()
    if plt is None or redundancy.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for i, (method, block) in enumerate(redundancy.groupby("method")):
        style = (
            NULL_STYLE
            if method in ("random", "uniform")
            else {"color": _colour(i), "linewidth": 1.4}
        )
        eff = block.groupby("budget")["rank_efficiency"].mean()
        cov = block.groupby("budget")["wavelength_coverage"].mean()
        axes[0].plot(eff.index, eff.to_numpy(), marker=".", label=str(method), **style)
        axes[1].plot(cov.index, cov.to_numpy(), marker=".", label=str(method), **style)
    for ax, ylabel, title in (
        (axes[0], "Effective rank / k", "Independent directions per selected band"),
        (axes[1], "Fraction of 385–1006 nm spanned", "Spectral coverage"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Band budget k")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Redundancy of the selected sets", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def wavelength_frequency(
    consensus: pd.DataFrame,
    path: Path,
    mean_spectrum: npt.NDArray[Any] | None = None,
    budget: int | None = None,
) -> Path | None:
    """How often each wavelength is chosen, over the mean spectrum.

    Plotted against nanometres rather than band index, and overlaid on the
    corpus mean spectrum, because the question a reader actually has is "do the
    selected regions correspond to anything physical?" — a pigment absorption,
    a water band, the red edge — and that is unanswerable from indices.
    """
    plt = _plt()
    if plt is None or consensus.empty:
        return None
    frame = consensus.sort_values("wavelength_nm")
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.fill_between(
        frame["wavelength_nm"],
        0.0,
        frame["selection_frequency"],
        color="#0072B2",
        alpha=0.65,
        step="mid",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Fraction of selections containing this band")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)

    if mean_spectrum is not None and len(mean_spectrum) == len(frame):
        # `frame` is ordered by wavelength; the spectrum is indexed by band, so
        # it is gathered through the frame's own band column rather than assumed
        # to already be in the same order.
        spectrum = np.asarray(mean_spectrum, dtype=float)[frame["band"].to_numpy(dtype=int)]
        twin = ax.twinx()
        twin.plot(frame["wavelength_nm"], spectrum, color="#666666", linewidth=1.2, alpha=0.8)
        twin.set_ylabel("Corpus mean reflectance (arb.)", color="#666666")
        twin.tick_params(axis="y", colors="#666666")

    suffix = f" at k = {budget}" if budget else ""
    ax.set_title(
        f"Consistently selected wavelengths{suffix}\n" "(supervised methods × folds × replicates)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def method_margins(ranking: pd.DataFrame, path: Path, null_margin: float = 0.01) -> Path | None:
    """Each method's best advantage over a random subset of the same size.

    The zero line is the figure. A bar that does not clear it says the method
    never beat an arbitrary choice at any budget on any proxy, which is a
    complete answer to "should we use this method?" and is drawn to be
    unmissable.
    """
    plt = _plt()
    if plt is None or ranking.empty:
        return None
    frame = ranking[ranking["method"] != "random"].sort_values("best_margin_vs_random")
    colours = ["#0072B2" if bool(e) else "#BBBBBB" for e in frame["effective"].fillna(False)]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.42 * len(frame) + 1.4)))
    ax.barh(frame["method"], frame["best_margin_vs_random"], color=colours)
    ax.axvline(0.0, color="#000000", linewidth=1.1)
    ax.axvline(
        null_margin,
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label=f"pre-registered margin {null_margin:.3f}",
    )
    ax.set_xlabel("Best Δ macro-F1 vs a random subset of the same size")
    ax.set_title(
        "Does the method select, or merely subset?\n(grey = never cleared the margin)", fontsize=10
    )
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def compute_tradeoff(curves: pd.DataFrame, path: Path) -> Path | None:
    """Accuracy against the cost of the bands that bought it.

    Cost is the band count itself, not seconds: the proxies' fit times are a
    property of scikit-learn on this laptop, whereas a neural run's input cost
    scales with k on any machine, so k is the transferable axis. The Pareto
    front is drawn because "which budgets are not dominated" is the actual
    decision, and it is invisible in a table.
    """
    plt = _plt()
    if plt is None or curves.empty:
        return None
    best = curves.groupby(["budget"])["score"].max()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    scatter = curves.groupby(["budget", "method"])["score"].mean().reset_index()
    ax.scatter(
        scatter["budget"], scatter["score"], s=12, color="#BBBBBB", label="individual methods"
    )

    frontier_k: list[float] = []
    frontier_v: list[float] = []
    running = -np.inf
    for k in sorted(best.index):
        if best[k] > running:
            running = float(best[k])
            frontier_k.append(float(k))
            frontier_v.append(running)
    ax.plot(
        frontier_k,
        frontier_v,
        marker="o",
        color="#0072B2",
        linewidth=1.8,
        label="Pareto front (best score at or below this budget)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Band budget k  —  proportional to a neural run's input cost")
    ax.set_ylabel("macro-F1 on calib")
    ax.set_title("Performance / compute trade-off", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def confirm_figure(confirm: pd.DataFrame, path: Path) -> Path | None:
    """Held-out scores with bootstrap intervals, for the configurations chosen.

    Reported after the fact and never plotted against a budget sweep, because a
    held-out curve over budgets is a held-out selection surface and drawing one
    would invite exactly the reading the study is built to prevent.
    """
    plt = _plt()
    if plt is None or confirm.empty:
        return None
    frame = confirm.copy()
    frame["label"] = frame.apply(
        lambda r: f"{r['method']} k={r['budget']} · {r['proxy']} · f{r['fold']}", axis=1
    )
    frame = frame.sort_values("macro_f1")
    lo, hi = [], []
    for value in frame.to_dict("records"):
        ci = value.get("ci") or {}
        lo.append(value["macro_f1"] - float(ci.get("lo", value["macro_f1"])))
        hi.append(float(ci.get("hi", value["macro_f1"])) - value["macro_f1"])

    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.3 * len(frame) + 1.5)))
    ax.errorbar(
        frame["macro_f1"],
        np.arange(len(frame)),
        xerr=[lo, hi],
        fmt="o",
        capsize=3,
        color="#0072B2",
        ecolor="#8899BB",
        markersize=5,
    )
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["label"], fontsize=7)
    ax.set_xlabel("macro-F1 on val ∪ test  (95% bootstrap CI)")
    ax.set_title(
        "Held-out confirmation of the chosen configurations\n"
        "(scored once, after the recommendation was fixed)",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════
#  Batch
# ══════════════════════════════════════════════════════════════════════


def render_all(cfg: Any, mean_spectrum: npt.NDArray[Any] | None = None) -> list[Path]:
    """Every figure the analysis has data for. Returns the paths written."""
    if not available():
        _log.info("matplotlib not installed; skipping figures (pip install -e '.[figures]')")
        return []

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def read(name: str) -> pd.DataFrame:
        # See `report._read` on why the default NA sentinels are disabled: the
        # family column's `null_model` and other bare word values are data.
        path = Path(cfg.analysis_dir) / f"{name}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=False, na_values=[""])

    curves = read("curves")
    trends = read("trends")
    stability = read("stability")
    redundancy = read("redundancy")
    ranking = read("method_ranking")
    consensus = read("wavelength_frequency")

    jobs = [
        (budget_curves, (curves, cfg.figures_dir / "budget_curves.png"), {}),
        (per_fold_curves, (curves, cfg.figures_dir / "budget_curves_per_fold.png"), {}),
        (plateau_summary, (trends, cfg.figures_dir / "plateau_summary.png"), {}),
        (
            stability_curves,
            (stability, cfg.figures_dir / "selection_stability.png"),
            {"floor": cfg.stability_floor},
        ),
        (redundancy_curves, (redundancy, cfg.figures_dir / "redundancy.png"), {}),
        (
            method_margins,
            (ranking, cfg.figures_dir / "method_margins.png"),
            {"null_margin": cfg.null_margin},
        ),
        (compute_tradeoff, (curves, cfg.figures_dir / "compute_tradeoff.png"), {}),
        (
            wavelength_frequency,
            (consensus, cfg.figures_dir / "wavelength_frequency.png"),
            {
                "mean_spectrum": mean_spectrum,
                "budget": int(consensus["budget"].iloc[0]) if not consensus.empty else None,
            },
        ),
    ]
    for function, args, kwargs in jobs:
        try:
            result = function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a figure must not kill a report
            _log.error("figure %s failed: %s", function.__name__, exc)
            continue
        if result is not None:
            written.append(result)

    from spectralquadnet.bandstudy.analysis import confirm_table

    try:
        confirm = confirm_table(cfg)
        if not confirm.empty:
            result = confirm_figure(confirm, cfg.figures_dir / "confirm_heldout.png")
            if result is not None:
                written.append(result)
    except Exception as exc:  # noqa: BLE001
        _log.error("confirm figure failed: %s", exc)

    return written
