#!/usr/bin/env python
"""S40: every held-out / calib row of the S22 v5 T4 x2 cells was scored exactly once.

The S22 k32 cells ran the same runner and final evaluation under DDP on two T4s; the
2,157-row test halves are odd, so DistributedSampler padded them. S40 reuses that path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from spectralquadnet.experiments.frozen_rows import load_frozen_rows

PARTITION = Path("docs/research/evidence/S21_complementary_rgb/preregistration.json")
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_refl215_f16")
out: dict[str, dict[str, object]] = {}
for fold in (0, 1):
    rows = load_frozen_rows(PARTITION, DATA / "labels.npy", DATA / "groups.npy", fold)
    want = {"val_test": np.concatenate([rows.val, rows.test]), "calib": rows.calib}
    cell: dict[str, object] = {
        "val": len(rows.val),
        "test": len(rows.test),
        "calib": len(rows.calib),
    }
    for split, expected in want.items():
        for tag in ("tta", "no_tta"):
            got = np.load(
                Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_{split}_{tag}.npz")
            )["rows"]
            cell[f"{split}_{tag}"] = {
                "n": int(got.size),
                "unique": bool(np.unique(got).size == got.size),
                "equals_frozen": bool(np.array_equal(np.sort(got), np.sort(expected))),
            }
    out[f"f{fold}_s0"] = cell
json.dump(out, sys.stdout, indent=2)
