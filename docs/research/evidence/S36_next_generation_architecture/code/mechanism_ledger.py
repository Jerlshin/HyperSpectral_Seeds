#!/usr/bin/env python
"""S36 mechanism ledger from archived per-class evidence only: every intervention tested in
S31-S38, its paired F1 effect (variety CI) and its away-from-session-8 bridge-recall effect (cell CI).
Outputs: evidence/S36_next_generation_architecture/mechanism_ledger.csv and the figure."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectralquadnet.experiments.screen_metrics import contrast

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / "docs/research/evidence"
OUT = E / "S36_next_generation_architecture/mechanism_ledger.csv"
FIG = ROOT / "docs/research/figures/S36_next_generation_architecture/mechanism_ledger.png"
INK, MUTED, GRID, GOOD, NULL = "#1f2328", "#59636e", "#d0d7de", "#2f6fb5", "#8c8c8c"
ROWS = [  # label, study folder, arm a, arm b, adopted?
    ("Train the RGB encoder (S32)", "S32_rgb_finetune", "ft_vitb_tta", "frozen_ref", True),
    ("…fused with v5 (S32)", "S32_rgb_finetune", "equal_ft_vitb", "equal_frozen_ref", True),
    ("Frozen readout: last-4 + views (S31)", "S31_rgb_readout_audit", "last4_tta", "cls", False),
    ("Larger backbone: partial ViT-L (S32)", "S32_rgb_finetune", "ft_vitl_tta", "ft_vitb_tta", False),
    ("Multi-layer + morphometrics (S37)", "S37_rgb_multilayer", "mlm_vitb_tta", "ft_vitb_tta", False),
    ("Measured-optics blur + render (S33)", "S33_rgb_acquisition", "acq_vitb_render", "ft_vitb_tta", False),
    ("Class-conditional rendering (S35)", "S35_regime_rendering", "equal_ccar_acq_vitb", "equal_acq_vitb_tta", False),
    ("Add frozen RGB to trained (S34)", "S34_multimodal_reassessment", "tri", "equal_trained", False),
    ("Kernel pairing vs shuffled (S34)", "S34_multimodal_reassessment", "equal_trained", "equal_trained_shuffled", False),
    ("HSI = spectral shape only (S38)", "S38_modality_roles", "role_hsi_snvmean214_own", "baseline", False),
]


def from_s8(per_class: pd.DataFrame, a: str, b: str) -> tuple[float, list[float]]:
    bridge = per_class[per_class.cross & (per_class.destination != 8)].set_index(["label", "fold"])
    d = (bridge[bridge.arm == a].recall - bridge[bridge.arm == b].recall).to_numpy()
    rng = np.random.default_rng(20261005)
    boot = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
    return float(d.mean()), [float(v) for v in np.quantile(boot, [.025, .975])]


def main() -> None:
    records = []
    for label, study, a, b, adopted in ROWS:
        pc = pd.read_csv(E / study / "screen_results/per_class.csv")
        c = contrast(pc, a, b)
        d, ci = from_s8(pc, a, b)
        records.append({"mechanism": label, "study": study, "a": a, "b": b, "adopted": adopted,
                        "delta_f1": c["delta_f1"], "f1_lo": c["f1_ci"][0], "f1_hi": c["f1_ci"][1],
                        "delta_from_s8": d, "from_s8_lo": ci[0], "from_s8_hi": ci[1]})
    df = pd.DataFrame(records)
    df.to_csv(OUT, index=False)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, (col, lo, hi, title) in zip(axes, (("delta_f1", "f1_lo", "f1_hi", "Paired macro-F1 Δ (variety CI)"),
                                               ("delta_from_s8", "from_s8_lo", "from_s8_hi",
                                                "Δ bridge recall, test outside session 8 (17 cells, CI)"))):
        for i, r in df.iterrows():
            y = len(df) - 1 - i
            color = GOOD if r.adopted else NULL
            ax.plot([r[lo], r[hi]], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(r[col], y, "o", ms=7, color=color)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel(title)
        ax.grid(axis="x", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(df)), df.mechanism[::-1])
    axes[0].set_title("Blue = adopted (SeedNet-MX) · grey = tested and not adopted", loc="left", fontsize=9, color=MUTED)
    fig.suptitle("Only training the encoder moved transfer; every other mechanism was neutral or harmful",
                 x=.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(.01, .005, "source: evidence/S31–S38 screen_results/per_class.csv → S36_next_generation_architecture/mechanism_ledger.csv",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, .03, 1, .95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, facecolor="white")
    print(df[["mechanism", "delta_f1", "delta_from_s8", "from_s8_lo", "from_s8_hi"]].round(4).to_string())


if __name__ == "__main__":
    main()
