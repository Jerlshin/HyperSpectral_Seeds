#!/usr/bin/env python3
"""Snapshot of the S13 plan as the runner builds it: cells, sources, exact commands, composition check.

Reads only the two frozen files and the configs (no data, no outputs). Writes
``frozen_plan.json`` beside this folder. Equivalent to ``scripts/run_s13.py --list
--check --dry-run --nproc-per-node 2``, kept as evidence.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from spectralquadnet.experiments import s13

EVID = Path(__file__).resolve().parents[1]
WATCHED = (
    "single.mixup_epochs", "single.arcface_m", "grad_clip", "single.epochs", "single.patience",
    "model.pathways", "model.spatial_mixstyle", "model.spectral_descriptor",
    "model.spatial_tail_strides", "model.cbam_min_hw", "data.split_scheme", "data.split_fold",
    "data.split_eval_frac", "seed", "evaluation.session_probe", "evaluation.save_logits",
)


def main() -> None:
    plan = s13.load_plan()
    cells = []
    for cell in plan.cells:
        cfg = s13.compose_cell(cell)
        resolved = {}
        for key in WATCHED:
            node = cfg
            for part in key.split("."):
                node = node[part]
            resolved[key] = list(node) if hasattr(node, "__iter__") and not isinstance(node, str) else node
        command = cell.spec().command(train_script="train.py", nproc_per_node=2)
        cells.append({
            "cell": cell.name, "source": cell.source, "data": cell.data,
            "arm_overrides": list(cell.arm_overrides), "resolved": resolved,
            "check": s13.check_cell(cell) or "ok",
            "command": " ".join(shlex.quote(c) for c in ["torchrun", "--standalone", "--nproc_per_node=2", *command[5:]]),
        })
    payload = {
        "preregistration_sha256": plan.hashes,
        "seed": plan.seed,
        "regime_r1": list(plan.cells[0].regime),
        "regime_check": s13.check_regime_is_r1(plan) or "ok",
        "gpu_cells": cells,
        "fused_cells": [{"cell": f.name, "inputs": [f.spectral, f.spatial]} for f in plan.fused],
    }
    (EVID / "frozen_plan.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    print(f"{len(cells)} GPU cells, {len(plan.fused)} fused; checks:",
          {c["cell"]: c["check"] for c in cells})


if __name__ == "__main__":
    main()
