"""S24 paired head gains and acquisition directions from compact saved evidence."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "docs/research/evidence/S24_branch_multimodal/screen_results"
F = ROOT / "docs/research/figures/S24_branch_multimodal"


def main():
    if not (E / "ARCHIVED.json").exists():
        raise FileNotFoundError("Archive branch results first")
    F.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((E / "paired_deltas.json").read_text())
    both = next(v for v in json.loads((ROOT / "docs/research/evidence/S23_frozen_multimodal/screen_results/paired_deltas.json").read_text())
                if v["control"] == "equal_single" and v["metric"] == "f1")
    values = [next(v for v in pairs if v["arm"] == a and v["control"] == "equal_single" and v["metric"] == "f1")
              for a in ("hsi_correction", "rgb_correction")] + [both]
    mean = np.array([v["mean"] for v in values])
    ci = np.array([v["ci"] for v in values]).T
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(np.arange(3), mean, color=["#228833", "#4477aa", "#aa3377"])
    axes[0].errorbar(np.arange(3), mean, yerr=np.maximum(np.vstack([mean-ci[0], ci[1]-mean]), 0),
                     fmt="none", color="black", capsize=4)
    axes[0].set_xticks(np.arange(3), ["HSI correction\n11,194 parameters", "RGB correction\n15,290 parameters", "Both corrections\n23,514 parameters"])
    axes[0].set_ylabel("Paired macro-F1 gain over equal single-view fusion")
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].axhline(.01, color="gray", ls="--", lw=.8)
    axes[0].grid(axis="y", alpha=.2)
    directions = pd.read_csv(E / "acquisition_directions.csv")
    for j, (arm, label) in enumerate(zip(["hsi_correction", "rgb_correction", "learned_residual", "v5_rgb_equal"],
                                       ["HSI correction", "RGB correction", "Both corrections", "Equal TTA"], strict=True)):
        block = directions[directions.arm == arm].set_index("direction")
        recall = block.loc[["to_session8", "from_session8"], "recall"].to_numpy()
        axes[1].bar(np.arange(2)+(j-1.5)*.2, recall, width=.19, label=label)
    axes[1].set_xticks([0, 1], ["Toward session 8", "Away from session 8"])
    axes[1].set_ylabel("Recall across the same 17 bridge varieties")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("S24: remove learned branches; both modalities remain in the fusion anchor")
    fig.text(.02, .01, "source: S24/S23 screen_results; seed 0, both corrected folds; variety intervals exclude seed/session uncertainty", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(F / "branch_screen.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
