#!/usr/bin/env python
"""S36 diagnostic (descriptive, no selection): how much of the remaining error is kernel-random
and how much is scan-systematic?

Every held-out scan holds kernels of one variety. Averaging log-probabilities over N kernels of
the same scan removes kernel-level noise; errors that survive at N = all are systematic for that
acquisition (variety confusion or acquisition shift). Components: S22 seed-0 v5 TTA (HSI), S32
fine-tuned ViT-B 4-view (RGB), their equal probability fusion (S34 baseline). Uses saved logits only.
Output: evidence/S36_next_generation_architecture/error_decomposition.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from spectralquadnet.experiments.rgb_probe import aligned_logits
from spectralquadnet.experiments.screen_metrics import Cohort

OUT = Path("docs/research/evidence/S36_next_generation_architecture/error_decomposition.csv")
SIZES = (1, 2, 4, 8, 16, 1000)  # 1000 = every kernel of the scan
DRAWS = 20


def main() -> None:
    cohort = Cohort.load()
    y = cohort.y
    groups = np.load("dataset_u430k32/groups.npy")
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    rng = np.random.default_rng(20261005)
    rows = []
    for fold in (0, 1):
        te = cohort.heldout(fold)
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        hsi = softmax(aligned_logits(Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_val_test_tta.npz"), te, y)
                      / t_hsi, axis=1)
        sel = json.loads(Path(f"outputs/s32_rgb_finetune/vitb_f{fold}/selection.json").read_text())
        with np.load(f"outputs/s32_rgb_finetune/vitb_f{fold}/outputs.npz") as o:
            rgb = softmax(o["logits"].mean(0)[te] / sel["temperature"], axis=1)
        for name, prob in (("hsi_tta", hsi), ("rgb_trained", rgb), ("equal_fusion", (hsi + rgb) / 2)):
            logp = np.log(np.clip(prob, 1e-12, None))
            for scan in np.unique(groups[te]):
                idx = np.flatnonzero(groups[te] == scan)
                label = int(y[te][idx[0]])
                for n in SIZES:
                    k = min(n, len(idx))
                    draws = 1 if k == len(idx) else DRAWS
                    correct = [logp[rng.choice(idx, k, replace=False)].mean(0).argmax() == label for _ in range(draws)]
                    rows.append({"fold": fold, "system": name, "scan": int(scan), "label": label,
                                 "cross": bool(cohort.cross[label]), "destination": int(cohort.session[te][idx[0]]),
                                 "kernels": "all" if n == 1000 else n, "accuracy": float(np.mean(correct))})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    summary = (df.assign(group=np.where(df.cross, np.where(df.destination == 8, "cross_to_s8", "cross_from_s8"), "same"))
               .groupby(["system", "group", "kernels"], sort=False).accuracy.mean().unstack("kernels"))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
