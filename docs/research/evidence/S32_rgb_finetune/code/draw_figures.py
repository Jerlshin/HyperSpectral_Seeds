#!/usr/bin/env python
"""S32/S34 figure from archived evidence only: what training the RGB branch changes, by
acquisition direction, and the kernel-vs-scan error decomposition (S36 diagnostic)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "docs/research/evidence"
FIG = ROOT / "docs/research/figures/S32_rgb_finetune"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
TO, AWAY, SAME = "#cc6677", "#2f6fb5", "#8c8c8c"


def main() -> None:
    s32 = pd.read_csv(E / "S32_rgb_finetune/screen_results/summary.csv", index_col=0)
    s34 = pd.read_csv(E / "S34_multimodal_reassessment/screen_results/summary.csv", index_col=0)
    dec = pd.read_csv(E / "S36_next_generation_architecture/error_decomposition.csv")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    systems = [(s32, "hsi_tta", "HSI v5\nTTA"), (s32, "frozen_ref", "RGB frozen\nViT-L (S29)"),
               (s32, "ft_vitb_tta", "RGB trained\nViT-B"), (s34, "s29_head_l", "S29 learned\nsystem"),
               (s32, "equal_ft_vitb", "HSI + RGB trained\nequal (new baseline)")]
    w = .27
    for k, (src, arm, _) in enumerate(systems):
        for off, col, color in ((-w, "same_recall", SAME), (0, "from_session8", AWAY), (w, "to_session8", TO)):
            v = src.loc[arm, col]
            a.bar(k + off, v, w - .03, color=color)
            a.text(k + off, v + .01, f"{v:.2f}", ha="center", fontsize=7.5, color=MUTED)
    a.set_xticks(range(len(systems)), [f"{lab}\nF1 {src.loc[arm, 'f1']:.3f}" for src, arm, lab in systems])
    a.set_ylim(0, 1)
    a.set_ylabel("Macro-recall, held-out, 2 corrected folds")
    a.legend(handles=[Patch(color=SAME, label="Same-session varieties (73)"),
                      Patch(color=AWAY, label="Bridges, test outside session 8 (17)"),
                      Patch(color=TO, label="Bridges, test in session 8 (17)")], frameon=False, loc="upper left")
    a.set_title("Training the RGB branch moves the away-from-session-8 direction for the first time", loc="left",
                fontsize=9, color=MUTED)
    order = ["1", "2", "4", "8", "16", "all"]
    d = dec[dec.system == "equal_fusion"].assign(
        group=lambda x: x.cross.map({False: "same"}).fillna(x.destination.eq(8).map({True: "to", False: "from"})))
    curve = d.groupby(["group", "kernels"]).accuracy.mean().unstack("kernels")[order]
    for g, color, label in (("same", SAME, "Same-session scans"), ("from", AWAY, "Bridge scans, test outside session 8"),
                            ("to", TO, "Bridge scans, test in session 8")):
        b.plot(range(len(order)), curve.loc[g], "-o", color=color, lw=2, ms=6, label=label)
        b.text(len(order) - .85, curve.loc[g].iloc[-1], f"{curve.loc[g].iloc[-1]:.2f}", va="center", fontsize=8, color=MUTED)
    b.set_xticks(range(len(order)), order)
    b.set_xlabel("Kernels of one held-out scan pooled (mean log-probability)")
    b.set_ylabel("Scan accuracy, equal fusion")
    b.set_ylim(0, 1.05)
    b.legend(frameon=False, loc="lower right")
    b.set_title("Same-session errors are kernel noise; cross-session errors are systematic", loc="left", fontsize=9, color=MUTED)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle("A trained RGB branch lifts every system (equal fusion .698 vs .627); what remains is acquisition-systematic",
                 x=.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(.01, .005, "source: evidence/S32_rgb_finetune, S34_multimodal_reassessment screen_results/summary.csv; "
             "S36_next_generation_architecture/error_decomposition.csv", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, .03, 1, .95))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "trained_rgb.png", dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
