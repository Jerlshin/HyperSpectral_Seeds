#!/usr/bin/env python
"""S27-S29 figure from archived evidence only: paired F1 effects and direction recall."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "docs/research/evidence"
FIG = ROOT / "docs/research/figures/S29_rgb_backbone_screen"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
TO, AWAY = "#cc6677", "#2f6fb5"


def main() -> None:
    h27 = json.loads((E / "S27_tta_trained_head/screen_results/hypothesis.json").read_text())
    h28 = json.loads((E / "S28_tta_head_seeds/screen_results/hypothesis.json").read_text())
    h29 = json.loads((E / "S29_rgb_backbone_screen/screen_results/hypothesis.json").read_text())
    d = h29["descriptive"]
    rows = [  # label, delta, ci, gated
        ("S27 head (ViT-S) − equal TTA\nH45, head seed 0", h27["delta_f1_equal_tta"], h27["f1_ci"], True),
        ("S28 heads 0–2 (ViT-S) − equal TTA\nH46", h28["delta_f1"], h28["f1_ci"], True),
        ("S29 head ViT-L − head ViT-S\nH47 (calib-selected backbone)", h29["primary"]["delta_f1"], h29["primary"]["f1_ci"], True),
        ("Equal fusion ViT-L − ViT-S", d["equal_l_vs_equal_s"]["delta_f1"], d["equal_l_vs_equal_s"]["f1_ci"], False),
        ("Head ViT-L − equal fusion ViT-L", d["head_l_vs_equal_l"]["delta_f1"], d["head_l_vs_equal_l"]["f1_ci"], False),
        ("RGB-only ViT-L − ViT-S", d["dino_l_vs_dino_s"]["delta_f1"], d["dino_l_vs_dino_s"]["f1_ci"], False),
    ]
    dirs = pd.read_csv(E / "S29_rgb_backbone_screen/screen_results/acquisition_directions.csv")
    systems = [("hsi_tta", "HSI\nTTA"), ("equal_s", "Equal\nViT-S"), ("head_s", "Head\nViT-S"),
               ("equal_l", "Equal\nViT-L"), ("head_l", "Head\nViT-L")]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, (_, delta, ci, gated) in enumerate(rows):
        y = len(rows) - 1 - i
        a.plot(ci, [y, y], color=INK, lw=2, solid_capstyle="round")
        a.plot(delta, y, "o", ms=8, color=INK if gated else "white", mec=INK, mew=2)
        a.text(ci[1] + .002, y, f"{delta:+.4f}", va="center", color=MUTED)
    a.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    a.axvline(0, color=MUTED, lw=1)
    a.axvline(.01, color=MUTED, lw=1, ls="--")
    a.set_ylim(-.9, len(rows) - .5)
    a.text(.0105, -.75, "practical gate .01", color=MUTED, fontsize=8)
    a.set_xlabel("Paired macro-F1 delta, held-out, mean of 2 corrected folds (95% variety CI)")
    a.set_title("Filled = frozen gate (all pass) · hollow = descriptive", loc="left", fontsize=9, color=MUTED)
    a.grid(axis="x", color=GRID, lw=.6)
    a.set_axisbelow(True)
    width = .38
    for k, (arm, _) in enumerate(systems):
        for off, direction, color in ((-width / 2, "from_session8", AWAY), (width / 2, "to_session8", TO)):
            r = dirs[(dirs.arm == arm) & (dirs.direction == direction)].iloc[0]
            b.bar(k + off, r.recall, width - .04, color=color, label=None)
            b.text(k + off, r.recall + .008, f"{r.recall:.2f}", ha="center", fontsize=8, color=MUTED)
    b.legend(handles=[Patch(color=AWAY, label="Test bundle outside session 8"),
                      Patch(color=TO, label="Test bundle in session 8")], frameon=False, loc="upper left")
    b.set_xticks(range(len(systems)), [s[1] for s in systems])
    b.set_ylabel("Cross-session macro-recall, 17 bridge varieties")
    b.set_ylim(0, .4)
    b.grid(axis="y", color=GRID, lw=.6)
    b.set_axisbelow(True)
    b.set_title("Gains reach only kernels tested in session 8", loc="left", fontsize=9, color=MUTED)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Learned TTA head and ViT-L RGB each add ≥ .01 F1; transfer away from session 8 does not move",
                 x=.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(.01, .005, "source: evidence/S27_tta_trained_head, S28_tta_head_seeds, S29_rgb_backbone_screen "
             "screen_results/{hypothesis.json, acquisition_directions.csv}", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, .03, 1, .95))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "system_progression.png", dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
