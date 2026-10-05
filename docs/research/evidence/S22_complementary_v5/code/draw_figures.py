"""S22 scientific figures from archived corrected-fold predictions only."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "docs/research/evidence/S22_complementary_v5/screen_results"
F = ROOT / "docs/research/figures/S22_complementary_v5"


def main():
    if not (E / "ARCHIVED.json").exists():
        raise FileNotFoundError("Archive corrected-fold results first")
    F.mkdir(parents=True, exist_ok=True)
    summary = {v["arm"]: v for v in json.loads((E / "summary.json").read_text())}
    arms = ["dino_rgb", "hsi_q32", "fusion32_equal", "v5_tta", "v5_rgb_equal"]
    labels = ["Frozen RGB", "Linear HSI32", "Equal linear\nfusion", "v5 HSI TTA", "Equal v5\nfusion"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for panel, metric, title, ci_name in zip(ax, ["f1", "same_recall", "cross_recall"],
        ["Macro-F1", "Same-session recall (73 varieties)", "Cross-session recall (17 varieties)"],
        ["f1_ci", "same_ci", "cross_ci"], strict=True):
        values = np.array([summary[a][metric] for a in arms])
        cis = np.array([summary[a][ci_name] for a in arms]).T
        panel.bar(np.arange(5), values, color=["#4477aa", "#66aa99", "#bbcc66", "#228833", "#cc6677"])
        panel.errorbar(np.arange(5), values, yerr=np.maximum(np.vstack([values-cis[0], cis[1]-values]), 0), fmt="none", color="black", capsize=3)
        panel.set_xticks(np.arange(5), labels, rotation=30, ha="right")
        panel.set_title(title, fontsize=10)
        panel.set_ylim(0, max(.3, cis.max() + .05))
        panel.grid(axis="y", alpha=.2)
    fig.suptitle("S22: one-seed development screen on both corrected acquisition folds")
    fig.text(.02, .01, "source: S22 screen_results/summary.json; S21 fixed probes; paired variety intervals exclude seed/session uncertainty", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(F / "corrected_screen.png", dpi=180)
    plt.close(fig)
    directions = pd.read_csv(E / "acquisition_directions.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, (arm, label) in enumerate(zip(["dino_rgb", "v5_tta", "v5_rgb_equal"], ["RGB", "v5 HSI", "Equal v5 fusion"], strict=True)):
        block = directions[directions.arm == arm].set_index("direction")
        values = block.loc[["to_session8", "from_session8"], "recall"].to_numpy()
        ax.bar(np.arange(2)+(j-1)*.23, values, width=.22, label=label)
    ax.set_xticks([0, 1], ["To session 8", "From session 8"])
    ax.set_ylabel("Macro recall across the same 17 bridge varieties")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    ax.set_title("Both directions are retained; neither is a new-session test")
    fig.text(.02, .01, "source: S22 screen_results/acquisition_directions.csv; reused acquisitions", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(F / "acquisition_directions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
