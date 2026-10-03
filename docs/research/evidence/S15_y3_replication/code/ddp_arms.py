#!/usr/bin/env python3
"""S15 validation: every S15 arm configuration on a real two-rank ``torchrun`` job.

Each S15 cell runs on two GPUs on Kaggle. Y3 ran there in S13; the two dissection
arms are new key combinations — ``desc_only`` (SNV + morph descriptor on the
shipped tail, whose last block keeps S10 F48's zero-gradient taps) and
``spatial_repair`` (the lean tail and gate placement with the full descriptor).
This runs R1 (reference), Y3 and both Y5 arms for 2 epochs on 2 gloo/CPU ranks with
S13's harness (S11's synthetic cube: 7 classes × 2 bundles × 13 kernels, odd held-out
split; a three-session scan table) and checks what a Kaggle cell must deliver: the
job finishes on both ranks (no DDP unused-parameter error, barrier passed), every
held-out kernel is scored once, the session κ covers every training row once, and the
regime block names the arm. Seeds follow the plan (Y3 at 1, Y5 at 0).

Usage (repository root; ≈ 2–4 min on a laptop CPU)::

    python docs/research/evidence/S15_y3_replication/code/ddp_arms.py \\
        --out docs/research/evidence/S15_y3_replication/ddp_arms.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
S13_DDP = REPO / "docs/research/evidence/S13_representation_screening/code/ddp_arms.py"

ARMS: dict[str, dict[str, Any]] = {
    "R1": {"seed": 0},
    "Y3_lean_s1": {"model.spectral_descriptor": "snv_morph", "model.spatial_tail_strides": "[2,2,2,1]",
                   "model.cbam_min_hw": 3, "seed": 1},
    "Y5_desc_only": {"model.spectral_descriptor": "snv_morph", "seed": 0},
    "Y5_spatial_repair": {"model.spatial_tail_strides": "[2,2,2,1]", "model.cbam_min_hw": 3, "seed": 0},
}


def _s13_harness():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("s13_ddp_arms", S13_DDP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    h = _s13_harness()
    payload: dict[str, Any] = {"launcher": "torchrun --nproc_per_node=2 (gloo, CPU)",
                               "harness": "S13 ddp_arms.py (_build_dataset, BASE, _torchrun, _report)", "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        data = h._build_dataset(Path(tmp) / "data")
        for arm, extra in ARMS.items():
            h._torchrun({**data, **h.BASE, **extra}, Path(tmp) / arm)
            payload["arms"][arm] = h._report(Path(tmp) / arm)
            print(arm, json.dumps(payload["arms"][arm]), flush=True)
    a = payload["arms"]
    payload["dissection_partitions_y3"] = (
        a["R1"]["parameters"] - a["Y3_lean_s1"]["parameters"]
        == (a["R1"]["parameters"] - a["Y5_desc_only"]["parameters"])
        + (a["R1"]["parameters"] - a["Y5_spatial_repair"]["parameters"])
    )
    payload["all_ok"] = bool(
        all(x["each_kernel_once"] and x["probe_every_training_row_once"] and x["world_size"] == 2
            for x in a.values())
        and payload["dissection_partitions_y3"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    print("all_ok:", payload["all_ok"])
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
