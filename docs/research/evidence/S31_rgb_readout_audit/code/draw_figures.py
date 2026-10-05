#!/usr/bin/env python
"""S31 figure from archived evidence only: RGB-only and equal-fusion F1 by frozen ViT-L readout,
and bridge recall by acquisition direction."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "docs/research/evidence/S31_rgb_readout_audit/screen_results"
FIG = ROOT / "docs/research/figures/S31_rgb_readout_audit"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
TO, AWAY = "#cc6677", "#2f6fb5"
RGB, FUSED = "#8c8c8c", "#2f6fb5"
ARMS = [("cls", "class token\n(S29)"), ("cls_fg", "+ fg-mean\ntoken"), ("cls_fg_tta", "+ 4 views"),
        ("last4", "last 4 layers\n(1 view)"), ("last4_tta", "last 4 layers\n+ 4 views"), ("cls_fg_tta_morph", "cls+fg, 4 views\n+ morphometrics")]


def main() -> None:
    s = pd.read_csv(E / "summary.csv", index_col=0)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    w = .38
    for k, (arm, _) in enumerate(ARMS):
        for off, key, color in ((-w / 2, arm, RGB), (w / 2, "equal_" + arm, FUSED)):
            v = s.loc[key, "f1"]
            a.bar(k + off, v, w - .04, color=color)
            a.text(k + off, v + .006, f"{v:.3f}", ha="center", fontsize=8, color=MUTED)
    a.axhline(s.loc["hsi_tta", "f1"], color=INK, lw=1, ls="--")
    a.set_xticks(range(len(ARMS)), [x[1] for x in ARMS])
    a.set_ylim(0, .9)
    a.set_ylabel("Macro-F1, held-out, mean of 2 corrected folds")
    a.legend(handles=[Patch(color=RGB, label="RGB alone (frozen ViT-L, shrinkage LDA)"),
                      Patch(color=FUSED, label="Equal fusion with HSI v5 TTA"),
                      Line2D([], [], color=INK, lw=1, ls="--", label="HSI v5 TTA alone")], frameon=False, loc="upper left")
    a.set_title("Readout, not backbone: +.126 RGB / +.050 fused F1 without training", loc="left", fontsize=9, color=MUTED)
    shown = [("hsi_tta", "HSI"), ("cls", "RGB\ncls"), ("last4_tta", "RGB\nlast4+views"), ("morph", "RGB\nmorph"),
             ("colour", "RGB\ncolour"), ("equal_cls", "HSI+\ncls"), ("equal_last4_tta", "HSI+\nlast4+views")]
    for k, (arm, _) in enumerate(shown):
        for off, col, color in ((-w / 2, "from_session8", AWAY), (w / 2, "to_session8", TO)):
            v = s.loc[arm, col]
            b.bar(k + off, v, w - .04, color=color)
            b.text(k + off, v + .006, f"{v:.2f}", ha="center", fontsize=7.5, color=MUTED)
    b.set_xticks(range(len(shown)), [x[1] for x in shown])
    b.set_ylim(0, .4)
    b.set_ylabel("Bridge macro-recall (17 varieties per direction)")
    b.legend(handles=[Patch(color=AWAY, label="Test bundle outside session 8"),
                      Patch(color=TO, label="Test bundle in session 8")], frameon=False, loc="upper left")
    b.set_title("Gains stay within acquisition; colour transfers at chance", loc="left", fontsize=9, color=MUTED)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen ViT-L leaves most of its RGB signal unused by the class-token probe; none of it transfers",
                 x=.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(.01, .005, "source: evidence/S31_rgb_readout_audit/screen_results/summary.csv", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, .03, 1, .95))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "readout_audit.png", dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
