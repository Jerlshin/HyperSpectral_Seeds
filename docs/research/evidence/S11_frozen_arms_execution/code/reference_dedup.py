#!/usr/bin/env python3
"""How much did the DDP duplicate move the frozen S08 reference? (S11, F58)

Every grouped run of the S08 sweep scored one held-out kernel twice (odd split
size on two ranks). This re-scores each run's saved held-out predictions with the
duplicate removed — the kernel is identified from ``rows_val_test_*.npy`` — and
reports macro-F1 before and after. No inference, no new held-out evaluation: the
same predictions, counted once.

    python docs/research/evidence/S11_frozen_arms_execution/code/reference_dedup.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = Path("docs/research/evidence/S11_frozen_arms_execution/reference_dedup.csv")
rows = []
for d in sorted(glob.glob("outputs/experiments_u430k32/protocol/*__f*_s*")):
    res = Path(d) / "results"
    for variant in ("no_tta", "tta"):
        p = np.load(res / f"preds_val_test_{variant}.npy")
        t = np.load(res / f"targets_val_test_{variant}.npy")
        r = np.load(res / f"rows_val_test_{variant}.npy")
        _, first = np.unique(r, return_index=True)
        keep = np.sort(first)
        rows.append(dict(
            run=Path(d).name, variant=variant, n_written=len(p), n_kernels=len(keep),
            macro_f1_written=f1_score(t, p, average="macro"),
            macro_f1_dedup=f1_score(t[keep], p[keep], average="macro"),
            reported=json.loads((res / "run.json").read_text())["results"][variant]["macro_f1"],
        ))
df = pd.DataFrame(rows)
df["delta"] = df.macro_f1_dedup - df.macro_f1_written
df.to_csv(OUT, index=False, float_format="%.6f")
g = df[df.run.str.startswith("grouped") & (df.variant == "tta")]
print(df.to_string(index=False))
print(f"\ngrouped TTA mean: written {g.macro_f1_written.mean():.6f}  dedup {g.macro_f1_dedup.mean():.6f}  "
      f"max |delta| {df.delta.abs().max():.2e}")
