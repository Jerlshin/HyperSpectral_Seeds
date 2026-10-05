"""S23 controls and learned-head figure, from archived predictions only."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "docs/research/evidence/S23_frozen_multimodal/screen_results"
F = ROOT / "docs/research/figures/S23_frozen_multimodal"


def main():
    if not (E / "ARCHIVED.json").exists():
        raise FileNotFoundError("Archive S23 predictions first")
    F.mkdir(parents=True, exist_ok=True)
    summary = {v["arm"]: v for v in json.loads((E / "summary.json").read_text())}
    arms = ["hsi_single", "rgb", "equal_single", "learned_residual", "v5_rgb_equal"]
    labels = ["HSI single", "Frozen RGB", "Equal single", "Learned residual", "S22 equal TTA"]
    colors = ["#228833", "#4477aa", "#bbcc66", "#aa3377", "#cc6677"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title, ci_name in zip(axes,
        ["f1", "same_recall", "cross_recall"], ["Macro-F1", "Same-session recall", "Cross-session recall"],
        ["f1_ci", "same_ci", "cross_ci"], strict=True):
        values = np.array([summary[a][metric] for a in arms])
        cis = np.array([summary[a][ci_name] for a in arms]).T
        ax.bar(np.arange(5), values, color=colors)
        ax.errorbar(np.arange(5), values, yerr=np.maximum(np.vstack([values-cis[0], cis[1]-values]), 0),
                    fmt="none", color="black", capsize=3)
        ax.set_xticks(np.arange(5), labels, rotation=30, ha="right")
        ax.set_ylim(0, min(1.05, max(.3, cis.max()+.05)))
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("S23: fixed 23,514-parameter head, seed 0 on both corrected folds")
    fig.text(.02, .01, "source: S23 screen_results/summary.json; paired variety intervals exclude seed/session uncertainty", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(F / "learned_screen.png", dpi=180)
    plt.close(fig)
    directions = pd.read_csv(E / "acquisition_directions.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, (arm, label) in enumerate(zip(["equal_single", "learned_residual", "v5_rgb_equal"],
                                       ["Equal single", "Learned residual", "S22 equal TTA"], strict=True)):
        block = directions[directions.arm == arm].set_index("direction")
        values = block.loc[["to_session8", "from_session8"], "recall"].to_numpy()
        ax.bar(np.arange(2)+(j-1)*.23, values, width=.22, label=label)
    ax.set_xticks([0, 1], ["To session 8", "From session 8"])
    ax.set_ylabel("Macro recall across 17 bridge varieties")
    ax.set_title("Learning and the acquisition-direction controls")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(F / "acquisition_directions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
