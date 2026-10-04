"""Snapshot the research evidence into docs/research/ and regenerate every figure.

    python docs/research/tools/build_assets.py            # evidence + figures
    python docs/research/tools/build_assets.py --figures  # figures only, from the snapshot

Why this exists
───────────────
`outputs/` and `dataset/` are git-ignored, so the raw evidence behind the research log
would otherwise live on one laptop. The `evidence` step copies the small, load-bearing
artifacts of each study into `docs/research/evidence/<study>/` (tracked), and the `figures`
step draws every generated figure *from that snapshot* — so a figure in the log can always
be recomputed from files that are in git, without the 30 GB cube.

Figures whose inputs are only numbers quoted in a document (the S01 audit figures) carry
those numbers inline below, each with its source section, because the console log and
W&B panels they came from are not in the repository.

Needs numpy, pandas, matplotlib. Does not import `spectralquadnet`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / "docs" / "research"
EVIDENCE = RESEARCH / "evidence"
FIGURES = RESEARCH / "figures"

# ── palette: the dataviz reference instance, light surface ───────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GAP_FILL = "#f0efec"  # neutral wash for "not measured" regions

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "text.color": INK, "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2, "lines.markersize": 5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 110, "savefig.dpi": 160,
})


def _save(fig: plt.Figure, study: str, name: str, source: str) -> None:
    fig.text(0.01, -0.01, f"source: {source}", fontsize=7, color=MUTED, ha="left", va="top")
    out = FIGURES / study / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  {out.relative_to(REPO)}")


def _label_end(ax: plt.Axes, x: float, y: float, text: str, color: str, dy: float = 0) -> None:
    """Direct label at a line's right end, in ink (identity comes from the swatch line)."""
    ax.plot([x], [y], "o", color=color, ms=5, mec=SURFACE, mew=1.5, zorder=4)
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                fontsize=8.5, color=INK_2, va="center")


# ══════════════════════════════════════════════════════════════════════════════
# evidence snapshot
# ══════════════════════════════════════════════════════════════════════════════

OUT = REPO / "outputs"
DS = REPO / "dataset"
BR = OUT / "band_research"
BS = OUT / "band_study"

SNAPSHOT: dict[str, list[tuple[Path, str]]] = {
    "S03_band_study_proxy": [
        (BS / "REPORT.md", "REPORT_2026-08-13.md"),
        (BS / "study.json", "study.json"),
        (BS / "analysis/recommendation.json", "recommendation.json"),
        (BS / "analysis/flags.json", "flags.json"),
        (BS / "analysis/curves.csv", "curves.csv"),
        (BS / "analysis/trends.csv", "trends.csv"),
        (BS / "analysis/method_ranking.csv", "method_ranking.csv"),
        (BS / "analysis/wavelength_frequency.csv", "wavelength_frequency.csv"),
        (BS / "analysis/tables/method_ranking.md", "tables/method_ranking.md"),
        (BS / "analysis/tables/trends.md", "tables/trends.md"),
        (BS / "analysis/tables/consensus_wavelengths.md", "tables/consensus_wavelengths.md"),
    ],
    "S05_band_research": [
        (BR / "preregistration.frozen.json", "preregistration.frozen.json"),
        (BR / "preregistration.sha256", "preregistration.sha256"),
        (BR / "preregistration2.json", "preregistration2.json"),
        (BR / "preregistration2.sha256", "preregistration2.sha256"),
        (BR / "decision_rule.json", "decision_rule.json"),
        (BR / "r1_budget.json", "r1_budget.json"),
        (BR / "a1_spectral.json", "a1_spectral.json"),
        (BR / "a3_summary.txt", "a3_summary.txt"),
        (BR / "confirm_lda.json", "confirm_lda.json"),
        (BR / "confirm2.json", "confirm2.json"),
        (BR / "cnn_calib.jsonl", "cnn_calib.jsonl"),
        (BR / "cnn_heldout.jsonl", "cnn_heldout.jsonl"),
        (BR / "cnn_test.jsonl", "cnn_test.jsonl"),
        (BR / "glw_canonical.json", "glw_canonical.json"),
        (BR / "arms_cnn_calib2.json", "arms_cnn_calib2.json"),
        (BR / "arms_cnn_heldout.json", "arms_cnn_heldout.json"),
        (OUT / "band_finalists/manifest.json", "finalists_manifest.json"),
    ],
    "S06_session_confound": [
        (BR / "a9_session_F.json", "a9_session_F.json"),
        (DS / "scan_table.csv", "scan_table.csv"),
    ],
    "S07_reflectance_calibration": [
        (DS / "radiometry.json", "radiometry.json"),
        (DS / "white_tiles.csv", "white_tiles.csv"),
        (DS / "wavelengths.csv", "wavelengths_215.csv"),
    ],
}


def snapshot_evidence() -> None:
    print("evidence:")
    for study, items in SNAPSHOT.items():
        for src, rel in items:
            dst = EVIDENCE / study / rel
            if not src.exists():
                print(f"  MISSING {src.relative_to(REPO)} — kept previous snapshot" if dst.exists()
                      else f"  MISSING {src.relative_to(REPO)}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied  {dst.relative_to(REPO)}")
    # The September study's analysis scripts are git-ignored with the rest of outputs/.
    # They are the only record of how the S05/S06 numbers were made, so archive them.
    code = EVIDENCE / "S05_band_research" / "code"
    code.mkdir(parents=True, exist_ok=True)
    for py in sorted(BR.glob("*.py")):
        shutil.copy2(py, code / py.name)
    print(f"  copied  {len(list(BR.glob('*.py')))} scripts -> {code.relative_to(REPO)}")
    # The tile QC arrays are 370 KB of npz; keep only the per-band saturation table the
    # S07 figure needs.
    ws = DS / "white_spectra.npz"
    if ws.exists():
        z = np.load(ws)
        sat = pd.DataFrame(z["saturated_frac"], columns=[f"{w:.2f}" for w in z["wavelengths"]])
        sat.insert(0, "session_id", z["session_id"])
        sat.insert(0, "scan_id", z["scan_id"])
        sat.round(4).to_csv(EVIDENCE / "S07_reflectance_calibration" / "tile_saturated_frac.csv",
                            index=False)
        print("  wrote   tile_saturated_frac.csv")
    # S03 figures are produced by the band-study CLI; copy them rather than redraw.
    src = BS / "analysis" / "figures"
    if src.exists():
        dst = FIGURES / "S03_band_study_proxy"
        dst.mkdir(parents=True, exist_ok=True)
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, dst / png.name)
        print(f"  copied  {len(list(src.glob('*.png')))} band-study figures")


# ══════════════════════════════════════════════════════════════════════════════
# S01 — audit figures (numbers quoted in CHANGES.md; the run log is not in git)
# ══════════════════════════════════════════════════════════════════════════════

def fig_s01() -> None:
    s = "S01_independent_audit"
    # CHANGES.md §2.3 / §9.1 — FLOP share vs end-of-training fused influence, per branch.
    branches = ["A · SpectralProfile", "B · Index bank", "C · Spatial 3-D CNN", "D · SpecFormer"]
    flop = [60.1, 0.0, 34.2, 5.7]
    infl = [5.6, 3.5, 87.4, 3.1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(branches))
    h = 0.36
    ax.barh(y - h / 2 - 0.01, flop, h, color=BLUE, label="share of forward FLOPs")
    ax.barh(y + h / 2 + 0.01, infl, h, color=ORANGE, label="share of fused influence (end)")
    for i in range(len(branches)):
        ax.text(flop[i] + 1, y[i] - h / 2, f"{flop[i]:.1f}%", va="center", fontsize=8, color=INK_2)
        ax.text(infl[i] + 1, y[i] + h / 2, f"{infl[i]:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, branches)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Compute went to the branch the model did not use")
    _save(fig, s, "s01_compute_vs_influence.png", "CHANGES.md §2.3, §9.1 (FLOPs are estimates)")

    # CHANGES.md §2.4 / §9.3 — what each stage cost and bought.
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    hours = [6.6, 2.7, 9.5]
    best = [0.842, 0.844, 0.847]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    a1.bar(stages, hours, color=BLUE, width=0.55)
    for i, v in enumerate(hours):
        a1.text(i, v + 0.2, f"{v:.1f} h", ha="center", fontsize=8.5, color=INK_2)
    a1.set_ylabel("wall clock (h)")
    a1.set_title("Cost")
    a1.grid(axis="x", visible=False)
    a2.axhspan(best[0] - 0.020, best[0] + 0.020, color=GAP_FILL, zorder=0)
    a2.text(2.35, best[0] + 0.018, "±0.020 sampling CI\naround Stage 1", fontsize=7.5,
            color=MUTED, ha="right", va="top")
    a2.plot(stages, best, "-o", color=BLUE)
    for i, v in enumerate(best):
        a2.annotate(f"{v:.3f}", (i, v), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK_2)
    a2.set_ylim(0.81, 0.875)
    a2.set_ylabel("best val macro-F1 (leaky split)")
    a2.set_title("Return")
    fig.suptitle("Stages 2–3: 65% of the wall clock for +0.005 macro-F1", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, s, "s01_stage_return.png", "CHANGES.md §2.4, §9.3")

    # CHANGES.md §4.5 — expected optimism of a running maximum, σ·sqrt(2 ln n).
    n = np.logspace(0, 3.2, 200)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for sd, c in [(0.008, AQUA), (0.012, BLUE), (0.016, VIOLET)]:
        b = sd * np.sqrt(2 * np.log(n))
        ax.plot(n, b, color=c)
        _label_end(ax, n[-1], b[-1], f"σ = {sd}", c)
    ax.axhline(0.005, color=RED, lw=1.2, ls="--")
    ax.text(1.2, 0.0065, "entire Stage 2 + 3 gain (+0.005)", fontsize=8, color=INK_2)
    ax.axvline(944, color=AXIS, lw=1)
    ax.text(900, 0.012, "~944 selection\nevents in the run", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1, 4000)
    ax.set_xlabel("number of checkpoint-selection events n (log)")
    ax.set_ylabel("expected upward bias of max (macro-F1)")
    ax.set_title("Reporting a maximum over correlated epochs inflates the score")
    _save(fig, s, "s01_selection_bias.png", "CHANGES.md §4.5 — E[max] − mean ≈ σ·√(2 ln n)")


# ══════════════════════════════════════════════════════════════════════════════
# S04 — engineering measurements (docs/06 §6.2–6.3, measured on an M5)
# ══════════════════════════════════════════════════════════════════════════════

def fig_s04() -> None:
    s = "S04_compute_engineering"
    panels = [
        ("Conv3d as stacked Conv2d (Metal)", "ms / step, batch 32", ("Conv3d", "decomposed"), (2103, 994)),
        ("Branch-A recompute (Metal)", "activation MB, batch 32", ("stored", "recomputed"), (4054, 1901)),
        ("Branch-A recompute at batch 128", "ms / sample", ("off (pages)", "on"), (780.1, 49.6)),
        ("torch.compile on Metal", "ms / forward, batch 32", ("eager", "inductor"), (437, 983)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, (title, unit, labels, vals) in zip(axes, panels, strict=True):
        better = vals[1] < vals[0]
        cols = [AXIS, BLUE if better else RED]
        ax.bar(labels, vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{v:g}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
        r = vals[0] / vals[1] if better else vals[1] / vals[0]
        ax.set_title(f"{title}\n×{r:.2f} {'better' if better else 'worse'}",
                     fontsize=9.5)
        ax.set_ylabel(unit, fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, s, "s04_runtime_measurements.png", "docs/06_EXECUTION_AND_HARDWARE.md §6.2–6.3")


# ══════════════════════════════════════════════════════════════════════════════
# S05 — September band research (outputs/band_research, SNV 256-band axis)
# ══════════════════════════════════════════════════════════════════════════════

def _wl256() -> np.ndarray:
    # The 256-band instrument axis: 383.22–1006.47 nm. The a1/a9 arrays address it.
    # Reconstructed from the 215-band axis file's instrument indices, which keep the
    # archive's own wavelength values; the dropped 41 bands are interpolated linearly
    # (the instrument grid is uniform at ~2.444 nm).
    w = pd.read_csv(EVIDENCE / "S07_reflectance_calibration" / "wavelengths_215.csv")
    idx = w.iloc[:, 0].to_numpy() - 1
    nm = w.iloc[:, -1].to_numpy(dtype=float)
    return np.interp(np.arange(256), idx, nm)


def _load_jsonl(p: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in p.read_text().splitlines() if line.strip()])


def fig_s05() -> None:
    s = "S05_band_research"
    ev = EVIDENCE / s
    wl = _wl256()

    # Per-band pixel SNR (spectral second-difference noise) — why 430 nm.
    a1 = json.loads((ev / "a1_spectral.json").read_text())
    snr = np.asarray(a1["snr_pixel_sd2"])
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.axvspan(wl[0], 430, color=GAP_FILL, zorder=0)
    ax.text(386, snr.max() * 0.93, "< 430 nm\nexcluded", fontsize=8, color=INK_2, va="top")
    ax.plot(wl, snr, color=BLUE)
    ax.axhline(10, color=RED, lw=1.2, ls="--")
    ax.text(wl[-1], 10.8, "SNR = 10", fontsize=8, color=INK_2, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("median pixel SNR (log)")
    ax.set_title("The blue end is noise-dominated: pixel SNR < 10 below ~430 nm")
    _save(fig, s, "s05_band_snr.png", "evidence/S05_band_research/a1_spectral.json (radiance, 256-band axis)")

    # Within-training-bundle budget curves (nested selection; 2 folds × 5 outer CV).
    txt = (ev / "a3_summary.txt").read_text().split("==== proxy lda_svd")[0].split("paired Δ")[0]
    rows = [ln.split() for ln in txt.splitlines() if ln.split() and ln.split()[0] in
            {"uniform", "uniform_snr", "glw", "mrmr", "spa", "random"}]
    ks = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    names = {"uniform": "uniform (383–1006)", "uniform_snr": "uniform430", "glw": "glw (greedy LDA)",
             "mrmr": "mRMR", "spa": "SPA", "random": "random"}
    order = ["uniform", "uniform_snr", "glw", "mrmr", "spa", "random"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    curves = {r[0]: np.array([np.nan if v == "NaN" else float(v) for v in r[1:]]) for r in rows}
    for c, m in zip(SERIES, order, strict=False):
        ax.plot(ks, curves[m], "-o", color=c, ms=3.5, label=names[m])
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("band budget k (log)")
    ax.set_ylabel("macro-F1 (5-fold CV inside training bundle)")
    ax.set_title("Within the training bundle: greedy LDA leads, mRMR and SPA trail even spacing")
    ax.legend(ncol=2, loc="lower right")
    _save(fig, s, "s05_within_train_budget.png", "evidence/S05_band_research/a3_summary.txt (LDA, shrinkage 1e-3)")

    # Held-out LDA confirmation (pre-registered, scored once).
    c = pd.read_json(ev / "confirm_lda.json")
    c["lo"] = c.ci.map(lambda v: v[0])
    c["hi"] = c.ci.map(lambda v: v[1])
    c["fam"] = c.arm.str.replace(r"_[km]\d+$", "", regex=True)
    g = c.groupby(["fam", "arm"]).agg(k=("k", "first"), f1=("f1", "mean"), lo=("lo", "mean"),
                                      hi=("hi", "mean")).reset_index()
    g.round(4).to_csv(ev / "heldout_lda_summary.csv", index=False)
    full = g.loc[g.arm == "uniform_k256", "f1"].item()
    fams = [("uniform", "uniform (383–1006)", BLUE), ("uniform430", "uniform430", ORANGE),
            ("glw", "glw (greedy LDA)", AQUA), ("uniform430+gstat3", "uniform430 + 3 gain stats", VIOLET)]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for fam, lab, col in fams:
        d = g[g.fam == fam].sort_values("k")
        d = d[d.k <= 192]
        ax.fill_between(d.k, d.lo, d.hi, color=col, alpha=0.12, lw=0)
        ax.plot(d.k, d.f1, "-o", color=col, ms=3.5)
        _label_end(ax, d.k.iloc[-1], d.f1.iloc[-1], lab, col, dy=7 if fam == "glw" else 0)
    ax.axhline(full, color=INK_2, lw=1, ls=":")
    ax.text(8.2, full + 0.006, f"full 256-band cube {full:.3f}", fontsize=8, color=INK_2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"])
    ax.set_xlim(7, 700)
    ax.set_xlabel("band budget k (log; gain stats add 3 features)")
    ax.set_ylabel("held-out macro-F1 (other bundle)")
    ax.set_title("Held-out, linear proxy: the < 430 nm bands and brightness still carry score")
    _save(fig, s, "s05_heldout_lda_budget.png", "evidence/S05_band_research/confirm_lda.json (2 folds; band = bootstrap CI)")

    # CNN proxy: calib (decision split) vs held-out (scored once).
    cal = _load_jsonl(ev / "cnn_calib.jsonl")
    held = _load_jsonl(ev / "cnn_heldout.jsonl")
    summ = []
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for ax, d, title in [(axes[0], cal, "calib (decision split, same bundle)"),
                         (axes[1], held, "held-out bundle (scored once)")]:
        for fam, col in [("uniform", BLUE), ("uniform430", ORANGE)]:
            x = d[d.arm.str.match(rf"^{fam}_k\d+$")]
            t = x.groupby("k").f1.agg(["mean", "std", "size"]).reset_index()
            summ.append(t.assign(family=fam, split=title.split()[0]))
            ax.fill_between(t.k, t["mean"] - t["std"].fillna(0), t["mean"] + t["std"].fillna(0),
                            color=col, alpha=0.15, lw=0)
            ax.plot(t.k, t["mean"], "-o", color=col, ms=3.5, label=fam)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128, 256], ["8", "16", "32", "64", "128", "256"])
        ax.set_xlabel("band budget k (log)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("macro-F1 (mean ± sd over fold × seed)")
    axes[0].legend(loc="lower center")
    fig.suptitle("Spatial-spectral proxy: 24–64 even bands beat the full cube on both splits",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    pd.concat(summ).round(4).to_csv(ev / "cnn_summary.csv", index=False)
    _save(fig, s, "s05_cnn_calib_vs_heldout.png", "evidence/S05_band_research/cnn_{calib,heldout}.jsonl")

    # The shipped finalist sets on the 215-band reflectance axis.
    man = json.loads((ev / "finalists_manifest.json").read_text())
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.axvspan(605.6, 708.2, color=GAP_FILL, zorder=0)
    ax.text(656.9, 4.6, "dropped:\ntile saturated", fontsize=7.5, color=INK_2, ha="center", va="top")
    ax.axvspan(383, 430, color=GAP_FILL, zorder=0, alpha=0.6)
    ax.text(406, 4.6, "< 430", fontsize=7.5, color=INK_2, ha="center", va="top")
    names = sorted(man["sets"], key=lambda n: man["sets"][n]["k"])
    for i, n in enumerate(names):
        w = man["sets"][n]["wavelengths_nm"]
        ax.plot(w, np.full(len(w), i), "|", color=BLUE, ms=11, mew=1.6)
    ax.set_yticks(range(len(names)), [f"k = {man['sets'][n]['k']}" for n in names])
    ax.set_xlim(380, 1010)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("wavelength (nm)")
    ax.set_title("Finalist sets: evenly spaced over the measured spectrum ≥ 430 nm")
    _save(fig, s, "s05_finalist_sets.png", "evidence/S05_band_research/finalists_manifest.json (215-band axis)")


# ══════════════════════════════════════════════════════════════════════════════
# S06 — the session confound
# ══════════════════════════════════════════════════════════════════════════════

def fig_s06() -> None:
    s = "S06_session_confound"
    ev = EVIDENCE / s
    st = pd.read_csv(ev / "scan_table.csv")
    ns = st.groupby("label").session_id.nunique()
    st["kind"] = np.where(st.label.map(ns) == 1, "same", "cross")
    per = st.groupby(["session_id", "kind"]).label.nunique().unstack(fill_value=0)
    dates = st.groupby("session_id").session.first().str.extract(r"(2017\d{4}-\d)")[0]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    x = np.arange(len(per))
    ax.bar(x, per["same"], 0.62, color=BLUE, label="varieties with both bundles in this session (73 total)")
    ax.bar(x, per["cross"], 0.62, bottom=per["same"] + 0.25, color=ORANGE,
           label="varieties with one bundle here, one elsewhere (17 total)")
    ax.set_xticks(x, [f"{i}\n{d[4:6]}/{d[6:8]}-{d[-1]}" for i, d in zip(per.index, dates, strict=True)], fontsize=8)
    ax.set_xlabel("acquisition session (id, 2017 date-run)")
    ax.set_ylabel("varieties")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("73 of 90 varieties were imaged entirely within one session")
    _save(fig, s, "s06_session_structure.png", "evidence/S06_session_confound/scan_table.csv")

    # Same- vs cross-session held-out recall.
    c2 = pd.read_json(EVIDENCE / "S05_band_research" / "confirm2.json")
    lda = c2.groupby("arm")[["f1", "rec_same", "rec_cross"]].mean()
    arms = [("LDA · full 256", lda.loc["uniform_k256"]),
            ("LDA · uniform430 k64 + gain", lda.loc["uniform430+gstat3_k64"]),
            ("LDA · session-clean k64", lda.loc["clean_uniform_k64"])]
    # CNN per-session recall was computed by a11_cnn_heldout_summary.py from preds/
    # (git-ignored); its printed table is reproduced here.
    cnn = {"CNN · uniform k64": (0.5947, 0.0031), "CNN · full 256": (0.5692, 0.0046)}
    labels = [a for a, _ in arms] + list(cnn)
    same = [r.rec_same for _, r in arms] + [v[0] for v in cnn.values()]
    cross = [r.rec_cross for _, r in arms] + [v[1] for v in cnn.values()]
    pd.DataFrame({"arm": labels, "recall_same_session": same, "recall_cross_session": cross}
                 ).round(4).to_csv(ev / "session_recall_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y - h / 2 - 0.01, same, h, color=BLUE, label="same-session varieties (73)")
    ax.barh(y + h / 2 + 0.01, cross, h, color=ORANGE, label="cross-session varieties (17)")
    for i in range(len(labels)):
        ax.text(same[i] + 0.008, y[i] - h / 2, f"{same[i]:.3f}", va="center", fontsize=8, color=INK_2)
        ax.text(cross[i] + 0.008, y[i] + h / 2, f"{cross[i]:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("held-out macro-recall")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("Every model scores ~0 on varieties whose bundles span two sessions", pad=26)
    _save(fig, s, "s06_same_vs_cross_recall.png",
          "evidence/S05_band_research/confirm2.json; CNN rows: a11_cnn_heldout_summary.py output")

    # Where in the spectrum the session fingerprint lives.
    F = np.asarray(json.loads((ev / "a9_session_F.json").read_text()))
    wl = _wl256()
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(wl, F[:256], color=BLUE)
    ax.axhline(2.2, color=RED, lw=1.2, ls="--")
    ax.text(wl[0], 2.35, "p ≈ 0.05", fontsize=8, color=INK_2, va="bottom")
    ax.axhline(1, color=AXIS, lw=1)
    ax.text(wl[0], 1.05, "null (F = 1)", fontsize=8, color=MUTED, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("session F-ratio (log)")
    ax.set_title("The session fingerprint peaks near the lamp maximum (~710 nm) and below 450 nm")
    ax.text(0.17, 0.95, "F of the gain stats:  log μ {:.1f}   log sd {:.1f}   −μ/sd {:.1f}".format(*F[256:]),
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK_2)
    _save(fig, s, "s06_session_F_by_wavelength.png",
          "evidence/S06_session_confound/a9_session_F.json (training rows only, mean of 2 folds)")


# ══════════════════════════════════════════════════════════════════════════════
# S07 — white-tile reflectance calibration
# ══════════════════════════════════════════════════════════════════════════════

def fig_s07() -> None:
    s = "S07_reflectance_calibration"
    ev = EVIDENCE / s
    p = ev / "tile_saturated_frac.csv"
    if not p.exists():
        print("  skip    s07 (no tile_saturated_frac.csv)")
        return
    d = pd.read_csv(p).sort_values(["session_id", "scan_id"])
    wl = np.array([float(c) for c in d.columns[2:]])
    m = d.iloc[:, 2:].to_numpy()
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=0.7, interpolation="nearest",
                   extent=[wl[0], wl[-1], len(d), 0])
    sess = d.session_id.to_numpy()
    for b in np.flatnonzero(np.diff(sess)) + 1:
        ax.axhline(b, color=AXIS, lw=0.6)
    ax.axvline(608.0, color=RED, lw=1, ls="--")
    ax.axvline(705.8, color=RED, lw=1, ls="--")
    ax.text(657, -3, "41 bands dropped", ha="center", fontsize=8, color=INK_2)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("scan (sorted by session; rules = session breaks)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("fraction of tile core pixels at 4,095 DN", fontsize=8.5, color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("The white tile clips at the lamp peak in sessions 0–7, never in session 8", pad=14)
    _save(fig, s, "s07_tile_saturation.png", "evidence/S07_reflectance_calibration/tile_saturated_frac.csv")


# ══════════════════════════════════════════════════════════════════════════════
# S09 — post-sweep forensics of the u430k32 protocol sweep
# Evidence is written directly into evidence/S09_post_sweep_forensics/ by the study's
# code/ scripts (extract_runs → controls → synthesis), so S09 has no SNAPSHOT entry.
# ══════════════════════════════════════════════════════════════════════════════

def fig_s09() -> None:
    s = "S09_post_sweep_forensics"
    ev = EVIDENCE / s
    if not (ev / "runs.csv").exists():
        print("  skip    s09 (no runs.csv — run the study's code/ scripts first)")
        return
    runs = pd.read_csv(ev / "runs.csv")
    g, st = runs[runs.arm == "grouped"], runs[runs.arm == "stratified"]
    PCOL = {"grouped": BLUE, "stratified": ORANGE}

    # 1 · The network against linear models on its own scalar inputs, by population.
    c5 = pd.read_csv(ev / "c5_tabular.csv")
    lda = c5[c5.model == "LDA"].set_index(["input", "protocol"])
    series = [
        ("SpectralSeedNet (2.85 M params, TTA)", {p: (x.f1_tta.mean(), x.same_recall_tta.mean(), x.cross_recall_tta.mean())
                                                  for p, x in (("grouped", g), ("stratified", st))}, BLUE),
        ("LDA · mean spectrum + 8 morphometrics", {p: tuple(lda.loc[("k32 spectrum + morphometrics", p),
                                                                    ["macro_f1", "same_recall", "cross_recall"]])
                                                   for p in ("grouped", "stratified")}, ORANGE),
        ("LDA · mean spectrum only", {p: tuple(lda.loc[("k32 spectrum", p), ["macro_f1", "same_recall", "cross_recall"]])
                                      for p in ("grouped", "stratified")}, AQUA),
        ("LDA · 8 morphometrics only", {p: tuple(lda.loc[("morphometrics only", p),
                                                         ["macro_f1", "same_recall", "cross_recall"]])
                                        for p in ("grouped", "stratified")}, YELLOW),
    ]
    cats = ["macro-F1\n(all 90)", "recall · same-session\nvarieties (73)", "recall · cross-session\nvarieties (17)"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    w = 0.19
    for ax, proto in zip(axes, ("grouped", "stratified"), strict=True):
        x = np.arange(3)
        for i, (name, vals, col) in enumerate(series):
            v = vals[proto]
            ax.bar(x + (i - 1.5) * (w + 0.01), v, w, color=col, label=name)
            for xi, vi in zip(x, v, strict=True):
                ax.text(xi + (i - 1.5) * (w + 0.01), vi + 0.012, f"{vi:.2f}", ha="center", fontsize=6.8,
                        color=INK_2, rotation=90, va="bottom")
        ax.set_xticks(x, cats, fontsize=8)
        ax.set_ylim(0, 0.92)
        ax.grid(axis="x", visible=False)
        ax.set_title({"grouped": "grouped — held-out acquisition bundle",
                      "stratified": "stratified — patch-level, both bundles in training"}[proto], fontsize=9.5)
    axes[0].set_ylabel("score on val + test (held-out)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2, fontsize=8)
    fig.suptitle("The network adds ≈ 0.05 over a linear model on its own scalar inputs; "
                 "shape alone matches its cross-session recall", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.13)
    _save(fig, s, "s09_network_vs_linear.png",
          "evidence/S09_post_sweep_forensics/{runs.csv, c5_tabular.csv} (network: mean of 6 runs per protocol)")

    # 2 · Fit ↔ held-out, per protocol.
    link = json.loads((ev / "fit_link.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, proto in zip(axes, ("stratified", "grouped"), strict=True):
        x = runs[runs.arm == proto]
        ax.scatter(x.final_train_loss, x.f1_tta, s=48, color=PCOL[proto], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for _, r in x.iterrows():
            ax.annotate(f"f{r.fold}s{r.seed}", (r.final_train_loss, r.f1_tta), xytext=(5, 3),
                        textcoords="offset points", fontsize=7.5, color=MUTED)
        k = link[f"{proto}: final_train_loss vs held-out F1"]
        ax.set_title(f"{proto}: r = {k['r']:.2f}  (95 % CI {k['ci95'][0]:.2f} … {k['ci95'][1]:.2f}, n = 6)",
                     fontsize=9.5)
        ax.set_xlabel("final training loss (margin-penalised CE)")
        ax.axvline(np.log(90), color=AXIS, lw=1, ls="--")
        ax.text(np.log(90), ax.get_ylim()[0], " ln 90 (uniform guess)", fontsize=7.5, color=MUTED, va="bottom")
    axes[0].set_ylabel("held-out macro-F1 (TTA)")
    fig.suptitle("Within the acquisition, a run scores as well as it fits its training data; across bundles it does not",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    _save(fig, s, "s09_fit_vs_heldout.png", "evidence/S09_post_sweep_forensics/{runs.csv, fit_link.json}")

    # 3 · Training dynamics.
    cv = pd.read_csv(ev / "curves.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    for (arm, fold, seed), d in cv.groupby(["arm", "fold", "seed"]):
        axes[0].plot(d.epoch, d.train_loss, color=PCOL[arm], lw=1, alpha=0.75)
        axes[1].plot(d.epoch, d.calib_f1_ema, color=PCOL[arm], lw=1, alpha=0.75)
    for ax in axes:
        ax.axvspan(0, 110.5, color=GAP_FILL, zorder=0)
        ax.axvspan(110.5, 130.5, color="#e6eefa", zorder=0)
    axes[0].axhline(np.log(90), color=INK_2, lw=1, ls="--")
    axes[0].text(2, np.log(90) + 0.1, "ln 90 — loss of a uniform guess", fontsize=8, color=INK_2)
    axes[0].text(55, 7.6, "mixup α = 0.35 (epochs 1–110)", ha="center", fontsize=8, color=INK_2)
    axes[0].text(120.5, 7.6, "margin\n0 to 0.3", ha="center", fontsize=8, color=INK_2)
    axes[0].set_ylabel("training loss")
    axes[1].set_ylabel("calib macro-F1 (EMA)")
    axes[1].set_xlabel("epoch")
    axes[1].plot([], [], color=BLUE, label="grouped (6 runs)")
    axes[1].plot([], [], color=ORANGE, label="stratified (6 runs)")
    axes[1].legend(loc="lower right")
    axes[0].set_title("Mixup holds the fit down for 110 epochs; the margin then lifts training loss above chance level")
    _save(fig, s, "s09_training_dynamics.png", "evidence/S09_post_sweep_forensics/curves.csv")

    # 4 · Data quantity vs acquisition: LDA learning curves and acquisition mixing.
    c1 = pd.read_csv(ev / "c1_learning_curve.csv")
    c2 = pd.read_csv(ev / "c2_acquisition_mix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    ax = axes[0]
    strat = c1[c1.protocol == "stratified (both bundles)"]
    grp = c1[c1.protocol.str.startswith("grouped")].groupby("n_per_class").macro_f1.mean()
    ax.plot(strat.n_per_class, strat.macro_f1, "-o", color=ORANGE)
    ax.plot(grp.index, grp.values, "-o", color=BLUE)
    ax.plot([strat.n_per_class.iloc[-1]], [strat.macro_f1.iloc[-1]], "o", color=ORANGE, ms=5, mec=SURFACE, mew=1.5)
    ax.text(4, 0.505, "stratified (both bundles)", fontsize=8.5, color=INK_2, ha="left")
    _label_end(ax, grp.index[-1], grp.values[-1], "grouped (one bundle)", BLUE)
    p8020 = c1[c1.protocol == "80/20 patch-level 5-fold"].iloc[0]
    ax.plot([p8020.n_per_class], [p8020.macro_f1], "D", color=INK_2, ms=6)
    ax.annotate(f"80/20, 5-fold: {p8020.macro_f1:.3f}", (p8020.n_per_class, p8020.macro_f1), xytext=(0, 9),
                textcoords="offset points", fontsize=8, color=INK_2, ha="center")
    ax.set_ylim(0.2, 0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("training kernels per class")
    ax.set_ylabel("LDA macro-F1 (mean spectrum, k32)")
    ax.set_title("More kernels from the same acquisition: small, saturating gain", fontsize=9.5)
    ax = axes[1]
    for design, col in (("n-matched", VIOLET), ("additive", AQUA)):
        d = c2[c2.design == design]
        ax.plot(d.k, d.macro_f1, "-o", color=col)
        _label_end(ax, d.k.iloc[-1], d.macro_f1.iloc[-1],
                   {"n-matched": "swap k kernels (total fixed)", "additive": "add k kernels"}[design], col)
    ax.set_xlim(-1, 40)
    ax.set_xlabel("k kernels/class of the test bundle moved into training")
    ax.set_ylabel("LDA macro-F1 on the rest of that bundle")
    ax.set_title("Kernels from the test acquisition: a large gain at equal n", fontsize=9.5)
    fig.suptitle("The grouped–stratified gap is acquisition coverage, not training-set size (linear proxy)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.03)
    _save(fig, s, "s09_quantity_vs_acquisition.png",
          "evidence/S09_post_sweep_forensics/{c1_learning_curve.csv, c2_acquisition_mix.csv} (10 repeats × 2 folds)")

    # 5 · Transfer line.
    t = pd.read_csv(ev / "transfer.csv")
    tf = json.loads((ev / "transfer_fit.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    net = t.source.str.startswith("SpectralSeedNet")
    ax.scatter(t[~net].stratified, t[~net].grouped, s=34, color=MUTED, edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="LDA / tabular controls (C3, C5)")
    ax.scatter(t[net].stratified, t[net].grouped, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label="SpectralSeedNet (TTA, no-TTA)")
    xx = np.linspace(0.15, 1.0, 50)
    ax.plot(xx, tf["intercept"] + tf["slope"] * xx, color=INK_2, lw=1.2)
    ax.plot(xx, xx, color=AXIS, lw=1, ls="--")
    ax.text(0.33, 0.355, "grouped = stratified", fontsize=7.5, color=MUTED, ha="center", rotation=38)
    ax.text(0.40, 0.76, f"fit: grouped ≈ {tf['slope']:.2f} × stratified\n(r = {tf['r']:.2f}, n = {tf['n']} model/input pairs)",
            fontsize=8, color=INK_2, ha="left", va="top")
    ax.axhline(0.84, color=RED, lw=1, ls=":")
    ax.text(0.16, 0.85, "grouped ceiling if same-session were perfect: 0.84", fontsize=7.5, color=INK_2)
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_xlabel("stratified macro-F1 (within-acquisition)")
    ax.set_ylabel("grouped macro-F1 (held-out bundle)")
    ax.legend(loc="upper left")
    ax.set_title("Every model keeps ≈ 73 % of its in-distribution score across bundles")
    _save(fig, s, "s09_transfer_line.png", "evidence/S09_post_sweep_forensics/{transfer.csv, transfer_fit.json}")

    # 6 · Session fingerprint after reflectance.
    f4 = pd.read_csv(ev / "c4_session_F_reflectance.csv")
    old = np.asarray(json.loads((EVIDENCE / "S06_session_confound" / "a9_session_F.json").read_text()))[:256]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(_wl256(), old, color=MUTED, lw=1.4)
    for col, key in ((BLUE, "F_reflectance"), (ORANGE, "F_snv_of_reflectance")):
        y = f4[key].to_numpy().copy()
        gap = np.flatnonzero(np.diff(f4.wavelength_nm) > 5)
        y = np.insert(y, gap + 1, np.nan)
        x = np.insert(f4.wavelength_nm.to_numpy(), gap + 1, np.nan)
        ax.plot(x, y, color=col)
    ax.text(712, old.max(), " SNV-256 (S06)", fontsize=8, color=INK_2, va="center")
    ax.text(1004, f4.F_reflectance.iloc[-1], " reflectance-215", fontsize=8, color=INK_2, va="center")
    ax.text(1004, f4.F_snv_of_reflectance.iloc[-1], " SNV of reflectance", fontsize=8, color=INK_2, va="center")
    ax.axvspan(608, 706, color=GAP_FILL, zorder=0)
    ax.text(657, 0.35, "dropped\n(tile clips)", ha="center", fontsize=7.5, color=MUTED)
    ax.axhline(2.2, color=RED, lw=1, ls="--")
    ax.axhline(1, color=AXIS, lw=1)
    ax.set_yscale("log")
    ax.set_xlim(380, 1080)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("session F-ratio (log)")
    ax.set_title("Reflectance removed the lamp-peak spike but left a broad NIR session offset (F ≈ 7–9)")
    _save(fig, s, "s09_session_F_reflectance.png",
          "evidence/S09_post_sweep_forensics/c4_session_F_reflectance.csv; S06 a9_session_F.json (training rows, 2 folds)")

    # 7 · Per-class: in-distribution vs held-out.
    pc = pd.read_csv(ev / "per_class.csv")
    m = pc.groupby(["class", "arm"]).f1.mean().unstack()
    cross = pc.groupby("class").cross_session.first()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    for flag, col, lab in ((False, BLUE, "same-session variety (73)"), (True, ORANGE, "cross-session variety (17)")):
        mm = m[cross == flag]
        ax.scatter(mm.stratified, mm.grouped, s=30, color=col, edgecolor=SURFACE, linewidth=1.1, label=lab, zorder=3)
    for c in (30, 41, 49, 51, 52, 79, 78):
        ax.annotate(str(c), (m.loc[c, "stratified"], m.loc[c, "grouped"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("per-class F1 · stratified (mean of 6 seeds)")
    ax.set_ylabel("per-class F1 · grouped (mean of 6 runs)")
    ax.legend(loc="upper left")
    ax.set_title("Cross-session varieties are learnable in-distribution and fail only across sessions", fontsize=10)
    _save(fig, s, "s09_per_class.png", "evidence/S09_post_sweep_forensics/per_class.csv (TTA)")


def fig_s10() -> None:
    s = "S10_training_architecture_review"
    ev = EVIDENCE / s
    if not (ev / "dynamics_epochs.csv").exists():
        print("  skip    s10 (no dynamics_epochs.csv — run the study's code/ scripts first)")
        return
    PCOL = {"grouped": BLUE, "stratified": ORANGE}

    # 1 · The schedule against what was learned, on one epoch axis (three panels, one y-scale each).
    dy = pd.read_csv(ev / "dynamics_epochs.csv")
    bud = pd.read_csv(ev / "schedule_budget.csv")
    bud = bud[bud.regime.str.startswith("shipped")].set_index("phase").share_of_cumulative_lr
    mean = dy.groupby(["arm", "epoch"]).mean(numeric_only=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25, 1.25]})
    for ax in axes:
        ax.axvspan(0.5, 110.5, color=GAP_FILL, zorder=0)
        ax.axvspan(110.5, 130.5, color="#e6eefa", zorder=0)
    lr = mean.loc["grouped"].lr / 5e-4
    axes[0].plot(lr.index, lr.values, color=INK_2, lw=1.6)
    axes[0].set_ylabel("LR / peak")
    axes[0].set_ylim(0, 1.08)
    axes[0].text(55, 0.5, f"mixup α 0.35 · epochs 1–110\n{bud['mixup']:.1%} of the cumulative LR",
                 ha="center", fontsize=8, color=INK_2)
    axes[0].text(120.5, 0.62, f"margin\nramp\n{bud['clean, margin ramp']:.1%}", ha="center", fontsize=8, color=INK_2)
    axes[0].text(140.5, 0.62, f"m = 0.30\n{bud['clean, margin 0.30']:.1%}", ha="center", fontsize=8, color=INK_2)
    for arm in ("grouped", "stratified"):
        f1 = mean.loc[arm].calib_f1_live
        axes[1].plot(f1.index, f1.values, color=PCOL[arm])
        _label_end(axes[1], f1.index[-1], f1.values[-1], f"{arm} (6 runs)", PCOL[arm],
                   dy={"grouped": -7, "stratified": 7}[arm])
        acc = mean.loc[arm].train_acc
        axes[2].plot(acc.index, acc.values, color=PCOL[arm])
    axes[1].set_ylabel("calib macro-F1 (live)")
    axes[1].annotate("mixup off: +0.02 in one epoch,\nat 18 % of the peak LR", (111, 0.668), xytext=(58, 0.18),
                     fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[2].axhline(0.506, color=MUTED, lw=1, ls="--")
    axes[2].text(2, 0.53, "ceiling of the logged value under mixup (scored against one of the two labels)",
                 fontsize=7.5, color=MUTED)
    axes[2].text(112, 0.86, "clean labels,\nmargin 0", fontsize=7.5, color=INK_2)
    axes[2].text(133, 0.38, "scored on margin-\npenalised logits", fontsize=7.5, color=INK_2)
    axes[2].set_ylabel("train accuracy as logged")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("epoch")
    axes[2].plot([], [], color=BLUE, label="grouped")
    axes[2].plot([], [], color=ORANGE, label="stratified")
    axes[2].legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    axes[0].set_title("The clean-label objective gets the last 3.6 % of the learning rate")
    _save(fig, s, "s10_schedule_vs_learning.png",
          "evidence/S10_training_architecture_review/{dynamics_epochs.csv, schedule_budget.csv} (means of 6 runs per arm)")

    # 2 · Where the spatial pathway's parameters sit, and which of them can learn.
    st = pd.read_csv(ev / "structure.csv")
    sp = st[st.module.str.startswith("spatial")].reset_index(drop=True)
    names = {"spatial.stem": "3-D stem", "spatial.proj": "pool + proj"}
    labels = []
    rb = cb = 0
    for m, k in zip(sp.module, sp.kind, strict=True):
        if m in names:
            labels.append(names[m])
        elif k == "ResBlock2D":
            labels.append(f"ResBlock {rb}")
            rb += 1
        else:
            labels.append(f"CBAM {cb}")
            cb += 1
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(sp))[::-1]
    live = (sp.params - sp.dead_params) / 1e3
    ax.barh(y, live, 0.62, color=BLUE, label="receives gradient")
    ax.barh(y, sp.dead_params / 1e3, 0.62, left=live, color=ORANGE, label="never receives gradient (multiplies padding)")
    for yi, (n, hw, dp) in enumerate(zip(sp.params[::-1], sp.out_hw[::-1], sp.dead_params[::-1], strict=True)):
        txt = f"{n / 1e3:,.0f} k · out {hw.replace('x', ' × ')}" + (f"   ({dp / 1e3:,.0f} k dead)" if dp else "")
        ax.text(n / 1e3 + 8, yi, txt, va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1080)
    ax.set_xlabel("parameters (thousands)  ·  label: count · output feature map")
    ax.legend(loc="upper right")
    ax.set_title("Four stride-2 blocks after a ÷4 stem take a 64 × 64 kernel to 1 × 1; "
                 "the last 3 × 3 conv sees a 2 × 2 map", fontsize=10)
    _save(fig, s, "s10_spatial_tail.png", "evidence/S10_training_architecture_review/structure.csv (k = 32, 64 × 64 input)")

    if not (ev / "ckpt_fit.csv").exists():
        print("  skip    s10 checkpoint figures (no ckpt_fit.csv)")
        return

    # 3 · Clean fit vs what the margin asked for, per run (selected checkpoint, training kernels).
    fit = pd.read_csv(ev / "ckpt_fit.csv")
    sel = fit[(fit.weights == fit.best_source) & (fit.split == "train")].copy()
    sel = sel.merge(dy.groupby(["arm", "fold", "seed"]).train_acc.last().rename("logged_end").reset_index(),
                    on=["arm", "fold", "seed"])
    sel = sel.sort_values(["arm", "acc"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(sel))
    for _, r in sel.iterrows():
        ax.plot([r.margin_satisfied, r.acc], [_, _], color=GRID, lw=2.2, zorder=1)
    for col, key, lab in ((BLUE, "acc", "clean accuracy (eval, no augmentation, margin 0)"),
                          (ORANGE, "margin_satisfied", "share satisfying the 0.30 rad margin"),
                          (AQUA, "logged_end", "train accuracy as logged at the last epoch")):
        ax.scatter(sel[key], y, s=46, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3, label=lab)
    ax.set_yticks(y, [f"{r.arm[:5]} f{r.fold} s{r.seed} · ep {r.best_epoch}" for _, r in sel.iterrows()], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("fraction of the run's own training kernels")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=8)
    ax.set_title("90 % of training kernels are classified correctly; a third satisfy the margin the loss is scored on",
                 fontsize=10, y=1.25)
    _save(fig, s, "s10_fit_vs_margin.png",
          "evidence/S10_training_architecture_review/{ckpt_fit.csv, dynamics_epochs.csv} (selected checkpoint per run)")

    # 4 · What each representation carries (linear probe, train → calib) and what the network leans on.
    pr = pd.read_csv(ev / "ckpt_probes.csv")
    m = pr.groupby(["probe", "representation", "arm"]).calib_macro_f1.mean().unstack()
    lin = m.loc["linear (LDA train→calib)"].sort_values("grouped")
    ko = m.loc["eval-time knock-out"]
    ko.loc["full network"] = m.loc[("linear (LDA train→calib)", "network (argmax cos)")]
    ko = ko.sort_values("grouped")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
    for ax, tab, title in ((axes[0], lin, "Linear probe (shrinkage LDA, fit on train, scored on calib)"),
                           (axes[1], ko, "Eval-time knock-outs of the trained network")):
        yy = np.arange(len(tab))
        for j, (arm, col) in enumerate((("grouped", BLUE), ("stratified", ORANGE))):
            ax.barh(yy + (j - 0.5) * 0.36, tab[arm], 0.34, color=col, label=arm)
        for yi, v in zip(yy, tab.max(axis=1), strict=True):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(yy, tab.index, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 0.85)
        ax.set_xlabel("calib macro-F1 (mean of 6 runs)")
        ax.set_title(title, fontsize=9.5)
    axes[0].legend(loc="lower right")
    fig.suptitle("The spectral pathway's output carries less than its own input; the trained network leans on morphometrics",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.03)
    _save(fig, s, "s10_representation_probes.png",
          "evidence/S10_training_architecture_review/ckpt_probes.csv (calib only; knock-outs are not retrained ablations)")


def fig_s12() -> None:
    s = "S12_frozen_arms_reading"
    ev = EVIDENCE / s
    if not (ev / "ladder_summary.csv").exists():
        print("  skip    s12 (no ladder_summary.csv — run the study's code/ scripts first)")
        return

    # 1 · The generalisation ladder: fit moved, nothing held-out did.
    lad = pd.read_csv(ev / "ladder_summary.csv")
    rungs = [("clean_train", "clean train\n(D18 subset)"), ("calib_acc", "calib\n(same bundle)"),
             ("same_recall", "held-out ·\nsame session"), ("cross_recall", "held-out ·\ncross session")]
    series = [("S08 shipped", BLUE, "S08 shipped (6 runs)"), ("X1", ORANGE, "X1 fit-first (6)"),
              ("X4", AQUA, "X4 no softeners (1)")]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    xs = np.arange(len(rungs))
    for arm, col, lab in series:
        r = lad[(lad.arm == arm) & (lad.protocol == "grouped")].iloc[0]
        ys = [r[k] for k, _ in rungs]
        ax.plot(xs, ys, color=col, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=lab)
        _label_end(ax, xs[-1], ys[-1], lab, col, dy={"S08 shipped": 8, "X1": -6, "X4": 18}[arm])
    r1 = lad[(lad.arm == "X1") & (lad.protocol == "grouped")].iloc[0]
    ax.annotate(f"fit +{r1.clean_train - 0.900:.2f}", (0, r1.clean_train), xytext=(10, 6),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.annotate("calib 0.714 vs 0.714", (1, r1.calib_acc), xytext=(10, 8), textcoords="offset points",
                fontsize=8, color=INK_2)
    ax.set_xticks(xs, [lab for _, lab in rungs], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (train, calib) · macro-recall (held-out)")
    ax.legend(loc="lower left")
    ax.set_title("X1 fits 0.98 of its training kernels and no held-out rung moves; X4 fits all and loses (grouped)", fontsize=10)
    _save(fig, s, "s12_ladder.png", "evidence/S12_frozen_arms_reading/ladder_summary.csv (selected checkpoints, TTA)")

    # 2 · The same-session / cross-session frontier.
    fr = pd.read_csv(ev / "frontier.csv")
    fu = pd.read_csv(ev / "pathway_fusion.csv")
    lin = pd.read_csv(ev / "linear_controls.csv")
    pix = pd.read_csv(ev / "pixel_controls.csv")
    lin = lin[lin.lda == "shrinkage"].groupby("representation")[["same_recall", "cross_recall"]].mean()
    pix = pix[pix.protocol == "grouped"].groupby("representation")[["same_recall", "cross_recall"]].mean()
    pix = pix.drop(index="mean + morph (S09 best linear)")          # same features as "raw + morph"
    linear = pd.concat([lin, pix[~pix.index.isin(lin.index)]])
    arms = [("S08 full (shipped)", BLUE, "S08 full"), ("X1 full", ORANGE, "X1 full"),
            ("X2 spatial_only", AQUA, "spatial only"), ("X2 no_morph", YELLOW, "no morph"),
            ("X2 spectral_only", MAGENTA, "spectral only"), ("X4 full", VIOLET, "X4")]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.scatter(linear.same_recall, linear.cross_recall, s=40, facecolor="none", edgecolor=MUTED, lw=1.2,
               zorder=2, label="linear controls (shrinkage LDA, fold means)")
    for name, r in linear.iterrows():
        short = name.replace(" (S09 best linear)", "").replace(" (spectral-path analogue)", "").replace(" + morph", " +m")
        ax.annotate(short, (r.same_recall, r.cross_recall),
                    xytext={"core + rim +m": (-58, 8)}.get(short, (4, -9)), textcoords="offset points",
                    fontsize=6.5, color=MUTED)
    for model, col, lab in arms:
        g = fr[fr.model == model]
        ax.scatter(g.same_recall, g.cross_recall, s=22, color=col, alpha=0.55, edgecolor="none", zorder=3)
        ax.scatter([g.same_recall.mean()], [g.cross_recall.mean()], s=90, color=col, edgecolor=SURFACE,
                   lw=2, zorder=4, label=f"{lab} (run · mean)")
        ax.annotate(lab, (g.same_recall.mean(), g.cross_recall.mean()),
                    xytext={"no morph": (-52, 8), "X1 full": (8, -8)}.get(lab, (7, 4)),
                    textcoords="offset points", fontsize=8.5, color=INK_2)
    f = fu[(fu.view == "tta") & (fu.model == "fusion equal")]
    ax.scatter([f.same_recall.mean()], [f.cross_recall.mean()], s=90, marker="D", color=GREEN,
               edgecolor=SURFACE, lw=2, zorder=5, label="late fusion: spectral + spatial (3 cells)")
    ax.annotate("late fusion", (f.same_recall.mean(), f.cross_recall.mean()), xytext=(7, 4),
                textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlabel("same-session macro-recall (73 varieties, held-out bundle)")
    ax.set_ylabel("cross-session macro-recall (17 varieties)")
    ax.set_xlim(0.1, 0.72)
    ax.set_ylim(0, 0.26)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Gaining same-session recall costs cross-session recall; the spectral-only network is at the far end",
                 fontsize=10)
    _save(fig, s, "s12_frontier.png", "evidence/S12_frozen_arms_reading/{frontier.csv, pathway_fusion.csv, "
          "linear_controls.csv, pixel_controls.csv} (grouped, TTA for networks)")

    # 3 · The training-rows session probe against held-out attraction.
    val = json.loads((ev / "session_probe_validation.json").read_text())
    tl = pd.DataFrame(val["table"])
    ep = pd.read_csv(ev / "embed_probe.csv")
    ep = ep[ep.representation == "embedding"]
    epv = json.loads((ev / "embed_probe_validation.json").read_text())
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(tl.kappa, tl.attraction_cross, s=46, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3,
               label=f"linear representations (13) · ρ = {val['spearman_kappa_vs_attraction']:.2f}")
    for _, r in tl.iterrows():
        if r.representation in ("morphometrics", "SNV + morph (spectral-path analogue)", "all pixel statistics + morph",
                                "raw + morph (S09 best linear)", "quantiles 10/50/90 + morph"):
            ax.annotate(r.representation.replace(" (spectral-path analogue)", "").replace(" (S09 best linear)", ""),
                        (r.kappa, r.attraction_cross), xytext=(6, -3), textcoords="offset points", fontsize=7.5,
                        color=INK_2)
    ax.scatter(ep.kappa, ep.heldout_attraction, s=46, marker="s", color=ORANGE, edgecolor=SURFACE, lw=1.5,
               zorder=3, label=f"network embeddings (11 checkpoints) · ρ = {epv['spearman_kappa_vs_attraction']:.2f}, "
                               f"r = {epv['pearson_kappa_vs_attraction']:.2f}")
    so = ep[ep.model == "X2 spectral_only"]
    ax.annotate("spectral-only networks", (so.kappa.mean(), so.heldout_attraction.mean()), xytext=(8, -12),
                textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.annotate("every network with the spatial pathway", (0.33, 0.47), xytext=(-150, 22),
                textcoords="offset points", fontsize=7.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("class-disjoint session decodability, Cohen's κ (training rows only)")
    ax.set_ylabel("held-out cross-session attraction")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.1, 0.7)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("A training-rows probe tracks held-out session attraction — finely for linear models, coarsely for networks",
                 fontsize=10)
    _save(fig, s, "s12_session_probe.png", "evidence/S12_frozen_arms_reading/{session_probe_validation.json, "
          "embed_probe.csv} (grouped fold 0 for networks; both folds for linear)")

    # 4 · Joint training vs fusing separately trained pathways (shipped regime, matched cells).
    fu = fu[fu.view == "tta"]
    order = [("spectral_only", "spectral only"), ("spatial_only", "spatial only"),
             ("full (S08, joint)", "joint full network"), ("fusion equal", "late fusion (equal)"),
             ("fusion calib-w", "late fusion (w on calib)")]
    m = fu.groupby("model")[["f1", "cross_recall", "attraction_cross"]].mean().loc[[k for k, _ in order]]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), sharey=True)
    yy = np.arange(len(order))[::-1]
    cols = [MUTED, MUTED, BLUE, GREEN, GREEN]
    for ax, (key, title) in zip(axes, (("f1", "macro-F1"), ("cross_recall", "cross-session recall"),
                                       ("attraction_cross", "cross-session attraction (lower = less session)"))):
        ax.barh(yy, m[key], 0.6, color=cols)
        for yi, v in zip(yy, m[key], strict=True):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(m[key]) * 1.25)
    axes[0].set_yticks(yy, [lab for _, lab in order], fontsize=8.5)
    fig.suptitle("Fusing separately trained pathways matches the jointly trained network and keeps more "
                 "cross-session recall", x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.06)
    _save(fig, s, "s12_pathways.png", "evidence/S12_frozen_arms_reading/pathway_fusion.csv (shipped regime; grouped "
          "f0 s0, f0 s1, f1 s1; TTA)")

    # 5 · Detector noise by session.
    nb = pd.read_csv(ev / "noise_by_session.csv")
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.bar(nb.session, nb.hf_median * 1e3, 0.66, color=BLUE)
    for x, v, n in zip(nb.session, nb.hf_median * 1e3, nb.n, strict=True):
        ax.text(x, v + 0.25, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
        ax.text(x, 0.5, f"n {n}", ha="center", fontsize=7, color=SURFACE)
    ax.set_xticks(nb.session, [f"s{s}\n{str(n)[9:19]}" for s, n in zip(nb.session, nb.session_name, strict=True)],
                  fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("median HF residual sd (× 10⁻³ reflectance)")
    ax.set_ylim(0, 24)
    ax.set_title("Detector noise steps up 25–30 % from session 5 on: a session fingerprint inside every kernel",
                 fontsize=10)
    _save(fig, s, "s12_noise_by_session.png", "evidence/S12_frozen_arms_reading/noise_by_session.csv "
          "(bands 8–23, per-kernel 3 × 3 high-frequency residual)")


def fig_s14() -> None:
    s = "S14_screen_reading"
    ev = EVIDENCE / s
    if not (ev / "arm_summary.csv").exists():
        print("  skip    s14 (no arm_summary.csv — run the study's code/ scripts first)")
        return
    arm = pd.read_csv(ev / "arm_summary.csv").set_index("arm")
    seed_sd = {"f1": 0.009933, "same": 0.013873, "cross": 0.009823, "attraction": 0.020362}

    def interval(a: str, k: str, strat: bool = False) -> tuple[float, float, float]:
        r = arm.loc[a]
        se_k = (r[f"{k}_hi"] - r[f"{k}_lo"]) / 3.92
        hw = 1.96 * np.sqrt(se_k ** 2 + (0.005806 if strat else seed_sd[k]) ** 2 / r.n_runs)
        return r[k], r[k] - hw, r[k] + hw

    # 1 · The screen: every frozen quantity against its threshold (screen interval = kernels + X1 seed sd).
    rows = [("Y1 fused (calib w)", "Y1 fused", BLUE), ("Y2 mixstyle", "Y2 MixStyle", ORANGE),
            ("Y3 lean grouped", "Y3 lean", AQUA)]
    panels = [("f1", "grouped macro-F1", 0.5108, 0.530786, "≥ 0.511 (H19a, H21a)", {"Y2 mixstyle"}),
              ("cross", "cross-session recall", 0.1666, 0.146637, "≥ 0.167 (H19b, H20)", {"Y3 lean grouped"}),
              ("attraction", "cross-session attraction (lower = less session)", 0.4240, 0.463958, "≤ 0.424 (H19b)",
               {"Y2 mixstyle", "Y3 lean grouped"})]
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for ax, (k, title, thr, ref, thr_lab, beside) in zip(axes[:3], panels, strict=True):
        yy = np.arange(len(rows))[::-1]
        for yi, (a, _lab, col) in zip(yy, rows, strict=True):
            v, lo, hi = interval(a, k)
            ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([v], [yi], "o", color=col, ms=9, mec=SURFACE, mew=2, zorder=3)
            for pf in str(arm.loc[a, f"{k}_per_fold"]).split():
                ax.plot([float(pf)], [yi + 0.22], "|", color=col, ms=8, mew=1.5, zorder=3)
            ax.annotate(f"{v:.3f}" + ("  (beside)" if a in beside else ""), (v, yi), xytext=(0, -14),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK_2, bbox=dict(fc=SURFACE, ec="none", pad=0.5))
        ax.axvline(thr, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.axvline(ref, color=MUTED, lw=1.2, zorder=1)
        ax.annotate(thr_lab, (thr, len(rows) - 0.45), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                    color=INK_2, ha="left", bbox=dict(fc=SURFACE, ec="none", pad=0.5))
        ax.text(ref, -0.75, f"X1 ref {ref:.3f}", fontsize=7.5, color=MUTED, ha="center", bbox=dict(fc=SURFACE, ec="none", pad=0.5))
        ax.set_yticks(yy, [lab for _, lab, _ in rows] if ax is axes[0] else [""] * len(rows), fontsize=8.5)
        ax.set_ylim(-1.0, len(rows) - 0.2)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9.5)
    ax = axes[3]
    srows = [("Y3 lean stratified", "Y3 lean (strat)", AQUA), ("Y4 80/20", "Y4 80/20", YELLOW)]
    yy = np.arange(len(srows))[::-1]
    for yi, (a, _lab, col) in zip(yy, srows, strict=True):
        v, lo, hi = interval(a, "f1", strat=True)
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([v], [yi], "o", color=col, ms=9, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK_2)
    ax.axvline(0.7090, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    ax.axvline(0.726984 + 0.03, color=INK_2, lw=1.2, ls=(0, (1, 2)))
    ax.axvline(0.726984, color=MUTED, lw=1.2)
    ax.annotate("≥ 0.709 (H21b)", (0.7090, 1.75), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color=INK_2, ha="left", bbox=dict(fc=SURFACE, ec="none", pad=0.5))
    ax.annotate("≤ 0.757\n(H15)", (0.757, 1.55), xytext=(-4, 0), textcoords="offset points", fontsize=7.5,
                color=INK_2, ha="right", bbox=dict(fc=SURFACE, ec="none", pad=0.5))
    ax.text(0.726984, -0.75, "X1 ref 0.727", fontsize=7.5, color=MUTED, ha="center", bbox=dict(fc=SURFACE, ec="none", pad=0.5))
    ax.set_yticks(yy, [lab for _, lab, _ in srows], fontsize=8.5)
    ax.set_ylim(-1.0, 2.1)
    ax.grid(axis="y", visible=False)
    ax.set_title("stratified macro-F1 (one run)", fontsize=9.5)
    fig.suptitle("S13 screen (seed 0): Y3 passes every bar it faces; Y1 keeps F1 but not robustness; Y2 moves nothing",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    _save(fig, s, "s14_screen.png", "evidence/S14_screen_reading/{arm_summary.csv, hypotheses.json} — dot = mean of "
          "seed-0 cells, ticks = folds, bar = kernel bootstrap combined with X1 seed sd (95 %); dashed = frozen threshold")

    # 2 · The same/cross-session plane: regime moves single pathways along the frontier; Y3 moves off it.
    c12 = pd.read_csv(EVIDENCE / "S12_frozen_arms_reading" / "cells.csv")
    c14 = pd.read_csv(ev / "cells.csv")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x1 = c12[(c12.arm == "X1") & (c12.variant == "grouped")]
    groups = [("X1 joint (R1, 6 runs)", x1, BLUE),
              ("Y3 lean (R1)", c14[c14.variant == "lean_grouped"], AQUA),
              ("Y2 MixStyle (R1)", c14[c14.variant == "mixstyle"], ORANGE),
              ("Y1 fused, calib w (R1)", c14[c14.variant == "fused"], YELLOW)]
    for lab, g, col in groups:
        ax.scatter(g.same_recall_tta, g.cross_recall_tta, s=24, color=col, alpha=0.55, edgecolor="none", zorder=3)
        mx, my = g.same_recall_tta.mean(), g.cross_recall_tta.mean()
        ax.scatter([mx], [my], s=100, color=col, edgecolor=SURFACE, lw=2, zorder=4, label=lab)
        ax.annotate(lab.split(" (")[0], (mx, my), xytext={"Y1 fused, calib w (R1)": (8, -10),
                                                         "Y2 MixStyle (R1)": (-78, -4)}.get(lab, (8, 4)),
                    textcoords="offset points", fontsize=8.5, color=INK_2)
    for pw, lab in (("spectral_only", "spectral-only"), ("spatial_only", "spatial-only")):
        a = c12[(c12.arm == "X2") & (c12.variant == pw) & c12.f1_tta.notna()]
        b = c14[c14.variant == pw]
        p0 = (a.same_recall_tta.mean(), a.cross_recall_tta.mean())
        p1 = (b.same_recall_tta.mean(), b.cross_recall_tta.mean())
        ax.scatter(*p0, s=70, facecolor="none", edgecolor=MUTED, lw=1.6, zorder=3)
        ax.scatter(*p1, s=70, color=MUTED, edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=6, shrinkB=6))
        ax.annotate(f"{lab}: shipped to R1", p1, xytext=(8, -12), textcoords="offset points", fontsize=8,
                    color=INK_2)
    ax.scatter([], [], s=60, facecolor="none", edgecolor=MUTED, lw=1.6, label="single pathway, shipped (X2)")
    ax.scatter([], [], s=60, color=MUTED, label="single pathway, R1 (Y1)")
    ax.set_xlabel("same-session macro-recall (73 varieties, held-out bundle, TTA)")
    ax.set_ylabel("cross-session macro-recall (17 varieties, TTA)")
    ax.set_xlim(0.45, 0.72)
    ax.set_ylim(0.07, 0.24)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("Fitting more slides single pathways along the frontier; the lean network moves up and right",
                 fontsize=10)
    _save(fig, s, "s14_frontier.png", "evidence/S14_screen_reading/cells.csv, evidence/S12_frozen_arms_reading/"
          "cells.csv (grouped; small dots = runs, large = mean; X2 spatial-only f1 s0 unscored)")

    # 3 · The fusion weight: calib picks the session-carrying pathway.
    g = pd.read_csv(ev / "fusion_grid.csv").groupby("w_spectral")[["calib_f1", "heldout_f1", "heldout_cross"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3))
    for ax, (k, title, thr, ref) in zip(axes, (
            ("calib_f1", "calib macro-F1 (what chooses w)", None, None),
            ("heldout_f1", "held-out macro-F1 (diagnostic)", 0.5108, 0.530786),
            ("heldout_cross", "held-out cross-session recall (diagnostic)", 0.1666, 0.146637)), strict=True):
        ax.plot(g.index, g[k], color=BLUE, lw=2)
        ax.axvline(0.30, color=INK_2, lw=1.2, ls=(0, (4, 3)))
        ax.axvline(0.48, color=MUTED, lw=1.2, ls=(0, (1, 2)))
        if thr is not None:
            ax.axhline(thr, color=INK_2, lw=1, ls=(0, (2, 2)))
            ax.axhline(ref, color=MUTED, lw=1)
            ax.text(0.01, thr, " frozen threshold", fontsize=7, color=INK_2, va="bottom")
            ax.text(0.01, ref, " X1 ref", fontsize=7, color=MUTED, va="bottom")
        ax.set_xlabel("w on the spectral network (1 − w on the spatial)")
        ax.set_title(title, fontsize=9.5)
    axes[0].text(0.31, axes[0].get_ylim()[0] + 0.01, "R1 calib choice 0.30", fontsize=7.5, color=INK_2, bbox=dict(fc=SURFACE, ec="none", pad=0.5))
    axes[0].text(0.49, axes[0].get_ylim()[0] + 0.04, "shipped-regime\ncalib choice ≈ 0.48", fontsize=7.5, color=MUTED)
    fig.suptitle("Calib rewards the session-carrying spatial network; only a spectral-heavy weight calib cannot find "
                 "reaches 0.167 cross-session recall", x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.05)
    _save(fig, s, "s14_fusion_weight.png", "evidence/S14_screen_reading/fusion_grid.csv (Y1, grouped, mean of f0/f1, "
          "TTA). Held-out curves are post hoc and chose nothing")


def fig_s16() -> None:
    s = "S16_replication_reading"
    ev = EVIDENCE / s
    if not (ev / "arm_summary.csv").exists():
        print("  skip    s16 (no arm_summary.csv — run the study's code/ scripts first)")
        return
    cells = pd.concat([pd.read_csv(ev / "cells.csv"), pd.read_csv(EVIDENCE / "S14_screen_reading" / "cells.csv"),
                       pd.read_csv(EVIDENCE / "S12_frozen_arms_reading" / "cells.csv")], ignore_index=True)
    x1 = cells[(cells.arm == "X1") & cells.variant.isin(["grouped", "stratified"])]
    y3 = cells[cells.variant.isin(["lean_grouped", "lean_stratified"])].drop_duplicates("cell")

    # 1 · The replication: every run, X1 vs v5, in each cell; seed 0 (the S13 screen) hollow.
    panels = [("f1_tta", "macro-F1 (held-out; stratified = within acquisition)",
               [("grouped", 0, "grouped f0"), ("grouped", 1, "grouped f1"), ("stratified", 0, "stratified")], None),
              ("cross_recall_tta", "cross-session recall (17 varieties)",
               [("grouped", 0, "f0"), ("grouped", 1, "f1")], (0.1666, "H21d ≥ 0.167")),
              ("attraction_cross_tta", "cross-session attraction (lower = less session)",
               [("grouped", 0, "f0"), ("grouped", 1, "f1")], (0.4240, "H21d ≤ 0.424"))]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for ax, (col, title, groups, thr) in zip(axes, panels, strict=True):
        for i, (proto, fold, _lab) in enumerate(groups):
            px = x1[(x1.variant == proto) & (x1.fold == fold)]
            py = y3[(y3.variant == f"lean_{proto}") & (y3.fold == fold)]
            for off, d, colr in ((-0.16, px, BLUE), (0.16, py, AQUA)):
                for _, r in d.iterrows():
                    hollow = colr == AQUA and r.seed == 0
                    ax.plot([i + off + (r.seed - 1) * 0.05], [r[col]], "o", ms=8, mew=2,
                            color=colr, mfc=SURFACE if hollow else colr, mec=colr if hollow else SURFACE, zorder=3)
                m = d[col].mean()
                ax.plot([i + off - 0.11, i + off + 0.11], [m, m], color=colr, lw=2, solid_capstyle="round", zorder=2)
        if thr is not None:
            ax.axhline(thr[0], color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
            ax.annotate(thr[1], (len(groups) - 0.55, thr[0]), xytext=(0, 3), textcoords="offset points",
                        fontsize=7.5, color=INK_2, ha="right", va="bottom", bbox=dict(fc=SURFACE, ec="none", pad=0.5))
        ax.set_xticks(range(len(groups)), [g[2] for g in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5)
    axes[0].plot([], [], "o", color=BLUE, mec=SURFACE, ms=8, label="X1 (shipped architecture, R1), seeds 0–2")
    axes[0].plot([], [], "o", color=AQUA, mec=SURFACE, ms=8, label="v5 = Y3 lean, seeds 1–2 (S15)")
    axes[0].plot([], [], "o", color=AQUA, mfc=SURFACE, mew=2, ms=8, label="v5, seed 0 (S13 screen)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylim(0.50, 0.80)
    fig.suptitle("The lean network replicates: every v5 run beats every X1 run in all three cells, and cross-session "
                 "recall rises with attraction down", x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    _save(fig, s, "s16_replication.png", "evidence/S16_replication_reading/cells.csv, S14 cells.csv (v5 seed 0), S12 "
          "cells.csv (X1); dots = runs (TTA), bars = means; dashed = frozen bar")

    # 2 · The dissection: Δ vs the seed-matched X1 cell on the same kernels, kernel-bootstrap 95 % CI.
    a = pd.read_csv(ev / "arm_summary.csv").set_index("arm")
    rows = [("Y5 desc_only, seed 0 (H22a)", "descriptor removed only", ORANGE),
            ("Y5 spatial_repair, seed 0 (H22b)", "spatial end map repaired only", VIOLET),
            ("Y3 grouped, seed 0 (S13)", "both (v5), seed 0", AQUA),
            ("Y3 grouped, seeds 0–2 (H21a)", "both (v5), seeds 0–2 vs X1 seeds 0–2", AQUA)]
    mets = [("f1", "Δ macro-F1"), ("same", "Δ same-session recall"), ("cross", "Δ cross-session recall"),
            ("attraction", "Δ attraction (down = less session)")]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.0), sharey=True)
    yy = np.arange(len(rows))[::-1]
    for ax, (k, title) in zip(axes, mets, strict=True):
        for yi, (arm, _lab, colr) in zip(yy, rows, strict=True):
            v, lo, hi = a.loc[arm, f"d_{k}"], a.loc[arm, f"d_{k}_lo"], a.loc[arm, f"d_{k}_hi"]
            filled = "seeds 0–2" in arm
            ax.plot([lo, hi], [yi, yi], color=colr, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([v], [yi], "o", ms=8, mew=2, color=colr, mfc=colr if (filled or "Y5" in arm) else SURFACE,
                    mec=SURFACE if (filled or "Y5" in arm) else colr, zorder=3)
            ax.annotate(f"{v:+.3f}", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK_2, bbox=dict(fc=SURFACE, ec="none", pad=0.4))
        ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(rows) - 0.3)
    axes[0].set_yticks(yy, [r[1] for r in rows], fontsize=8.5)
    fig.suptitle("Dissection (seed 0): the spatial end-map repair alone carries v5's robustness; the descriptor "
                 "removal alone moves F1 a little and attraction not at all", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.07)
    _save(fig, s, "s16_dissection.png", "evidence/S16_replication_reading/arm_summary.csv — Δ vs X1 seed 0 on the "
          "same kernels (bottom row: v5 seeds 0–2 vs X1 seeds 0–2, hierarchical bootstrap); bars = 95 % CI")

    # 3 · Who is rescued: cross-session accuracy by the kernel's acquisition session (3-seed means per fold).
    se = pd.read_csv(ev / "v5_sessions.csv").sort_values(["fold", "session"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    yy = np.arange(len(se))[::-1]
    for yi, (_, r) in zip(yy, se.iterrows(), strict=True):
        ax.plot([r.X1, r.Y3], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot([r.X1], [yi], "o", color=BLUE, ms=8, mec=SURFACE, mew=2, zorder=3)
        ax.plot([r.Y3], [yi], "o", color=AQUA, ms=8, mec=SURFACE, mew=2, zorder=3)
        if r.Y3 - r.X1 > 0.02:
            ax.annotate(f"{r.Y3 - r.X1:+.2f}", (max(r.Y3, r.X1), yi), xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color=INK_2, va="center")
    ax.set_yticks(yy, [f"fold {int(r.fold)} · session {int(r.session)} (n {int(r.n)})" for _, r in se.iterrows()], fontsize=8)
    ax.plot([], [], "o", color=BLUE, mec=SURFACE, ms=8, label="X1, mean of 3 seeds")
    ax.plot([], [], "o", color=AQUA, mec=SURFACE, ms=8, label="v5, mean of 3 seeds")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("held-out accuracy on cross-session kernels (TTA)")
    ax.set_xlim(-0.02, 0.5)
    ax.grid(axis="y", visible=False)
    ax.set_title("v5's cross-session gain sits in kernels from sessions 2, 5 and 8;\nkernels imaged in sessions 0, 1, 3, "
                 "4 and 7 stay at ≈ 0 for both networks", fontsize=10)
    _save(fig, s, "s16_sessions.png", "evidence/S16_replication_reading/v5_sessions.csv (grouped; the kernel's own "
          "acquisition session; 17 cross-session varieties)")


# S19 writes its descriptive evidence directly, like S09; no raw-output copy needed.
def fig_s19() -> None:
    """Regenerate the metadata audit figures using tracked S19 evidence only."""
    import runpy
    runpy.run_path(str(EVIDENCE / "S19_next_generation_strategy" / "code" / "draw_figures.py"),
                   run_name="__main__")


def fig_s20() -> None:
    """Regenerate S20 scientific figures from saved CPU-screen evidence only."""
    import runpy
    with plt.rc_context(matplotlib.rcParamsDefault):
        runpy.run_path(str(EVIDENCE / "S20_rgb_pathway" / "code" / "draw_figures.py"),
                       run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", action="store_true", help="skip the evidence snapshot")
    a = ap.parse_args()
    if not a.figures:
        snapshot_evidence()
    print("figures:")
    for f in (fig_s01, fig_s04, fig_s05, fig_s06, fig_s07, fig_s09, fig_s10, fig_s12, fig_s14, fig_s16, fig_s19, fig_s20, fig_s21):
        f()


def fig_s21() -> None:
    """Regenerate S21 scientific figures from saved complementary-fold evidence."""
    import runpy
    with plt.rc_context(matplotlib.rcParamsDefault):
        runpy.run_path(str(EVIDENCE / "S21_complementary_rgb" / "code" / "draw_figures.py"),
                      run_name="__main__")


if __name__ == "__main__":
    main()
