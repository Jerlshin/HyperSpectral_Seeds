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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", action="store_true", help="skip the evidence snapshot")
    a = ap.parse_args()
    if not a.figures:
        snapshot_evidence()
    print("figures:")
    for f in (fig_s01, fig_s04, fig_s05, fig_s06, fig_s07):
        f()


if __name__ == "__main__":
    main()
