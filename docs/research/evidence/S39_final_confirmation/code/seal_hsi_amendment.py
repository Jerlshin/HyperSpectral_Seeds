#!/usr/bin/env python
"""Seal the S39 HSI amendment of the S22 runner: v5 encoder seeds 1 and 2 on both corrected folds.

The only change from amendment05 is the authorized cell list. Architecture, R1 schedule, batch,
partitions, TTA, runtime (two T4, fp16) and input hashes are copied unchanged, and the parent hash
is recorded. No cell may be scored before the S39 contrasts are frozen.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import verify_plan

PARENT = Path("configs/research/s22_screening_amendment05.json")
PLAN = Path("configs/research/s39_hsi_seeds_amendment06.json")


def main() -> None:
    parent = verify_plan(PARENT)
    if PLAN.exists():
        raise FileExistsError("S39 HSI amendment already sealed")
    spec = dict(parent)
    spec.update({
        "status": "frozen S39 confirmation cells; CUDA fits unrun at sealing; needs owner authorization for Kaggle GPU quota",
        "amendment": "06 (S39): authorize v5 encoder seeds 1 and 2 on corrected folds 0 and 1 (4 fits) for the matched final "
                      "confirmation of SeedNet-MX (S36). Only the cell list changes; architecture, R1, batch, partitions, TTA, "
                      "runtime and input hashes are those of amendment05. Embeddings and train-row TTA are extracted on CPU "
                      "afterwards from the saved checkpoints, as in S24/S27.",
        "cells": [{"fold": f, "seed": s} for s in (1, 2) for f in (0, 1)],
        "parent_plan_sha256": sha256(PARENT),
        "frozen_at": "2026-10-05",
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "stop": "Exactly the four listed cells. No held-out row of a new cell is scored until configs/research/s39_final_confirmation.json "
                "(contrasts M1-M4) is frozen and verified.",
    })
    write_json(PLAN, spec)
    PLAN.with_suffix(".sha256").write_text(sha256(PLAN) + "  " + PLAN.name + "\n")
    print("Sealed", sha256(PLAN), json.dumps(spec["cells"]))


if __name__ == "__main__":
    main()
