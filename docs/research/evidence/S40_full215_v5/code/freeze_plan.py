#!/usr/bin/env python
"""Write and hash configs/research/s40_full215_v5.json (S40). Run from the repository root.

Re-running it is allowed only before any S40 cell has started. Once a cell exists, a change
is an amendment: a new file that names this plan's hash (WORKFLOW §3.2.5).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from spectralquadnet.data.prep.multimodal import sha256

PLAN = Path("configs/research/s40_full215_v5.json")
DATASET = Path("dataset_refl215_f16")
PARTITION = Path("docs/research/evidence/S21_complementary_rgb/preregistration.json")
PARENT = Path("configs/research/s22_screening_amendment05.json")
DATASET_FILES = (
    "MANIFEST.json",
    "band_axis.json",
    "labels.npy",
    "groups.npy",
    "masks.npy",
    "morphology.npy",
    "scan_table.csv",
    "wavelengths.csv",
    "radiometry.json",
)


def main() -> int:
    if any(Path("outputs/s40_full215_v5").glob("f*_s*")):
        print("an S40 cell already exists: write an amendment instead of re-freezing")
        return 1
    parent = json.loads(PARENT.read_text())
    files = sorted(str(p) for p in Path("src/spectralquadnet").rglob("*.py"))
    files += sorted(str(p) for p in Path("configs").rglob("*.yaml"))
    files += [
        "train.py",
        "scripts/run_complementary_v5.py",
        "scripts/run_full215_v5.py",
        "scripts/build_refl215_f16.py",
        str(PARTITION),
        str(PARTITION.with_suffix(".sha256")),
    ]
    files += [str(DATASET / name) for name in DATASET_FILES]
    v5 = [o for o in parent["v5_overrides"] if not o.startswith("data=")]
    plan = {
        "study": "S40_full215_v5",
        "status": "frozen development run; CUDA fits unrun at sealing",
        "question": "What does SeedNet v5 do when it reads all 215 calibrated reflectance bands "
        "instead of the uniform430_k32 subset, on the same corrected folds, seed and regime?",
        "partition_plan": str(PARTITION),
        "v5_overrides": ["data=refl215_f16_grouped", *v5],
        "cells": [{"fold": 0, "seed": 0}, {"fold": 1, "seed": 0}],
        "runtime_overrides": parent["runtime_overrides"],
        "matched_reference": {
            "cells": ["outputs/s22_complementary_v5/f0_s0", "outputs/s22_complementary_v5/f1_s0"],
            "plan": str(PARENT),
            "plan_sha256": sha256(PARENT),
            "differs_only_in": "data= (uniform430_k32 float16, 32 bands -> all 215 bands float16)",
        },
        "model_adaptation": "none. Every width that depends on the band count is derived from "
        "data.num_bands=215: stem strides (8,2,2), kernels (15,5,5), folded "
        "depth 7 (448-channel fold), SNV descriptor 215+8, CutMix 43 / "
        "cutout 16 bands (the same fractions as k32). The R1 regime, global "
        "batch 128, fp16 + GradScaler on T4 x2, epochs and patience are unchanged.",
        "analysis": "Descriptive development comparison with the matched S22 k32 seed-0 cells: "
        "per-fold and mean macro-F1 (no-TTA and TTA), same/cross-session recall, "
        "from/to session 8, paired 2,000-resample class bootstrap of the delta. "
        "No gate: one seed, and seed SD is measured only at k32 (S39). Not a "
        "band-selection claim and not a paper claim until it is replicated.",
        "stop": "Exactly the two listed cells. No seed expansion, no extra arms, and no rerun of "
        "a finished cell without a new frozen plan.",
        "input_hashes": {name: sha256(Path(name)) for name in files},
        "frozen_at": "2026-10-06",
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "provenance": "Every training source, config, the runner, the S40 driver, the frozen "
        "partition and the dataset's identity files (its MANIFEST pins the cube's "
        "SHA-256; the driver re-hashes the cube before training) are hashed.",
        "training_host": "Kaggle notebook, GPU T4 x2 (two devices required by the driver).",
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    PLAN.with_suffix(".sha256").write_text(f"{sha256(PLAN)}  {PLAN.name}\n")
    print(f"{PLAN}  {sha256(PLAN)}  ({len(files)} hashed inputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
