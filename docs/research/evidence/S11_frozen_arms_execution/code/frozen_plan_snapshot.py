#!/usr/bin/env python3
"""Snapshot of the S11 execution plan: every frozen cell, its exact command, its check.

Writes ``frozen_plan.json`` — the 23 X1/X2/X4 cells as ``scripts/run_frozen.py``
will launch them on Kaggle T4 × 2 (``--nproc-per-node 2``), each with its frozen
source, its output directory, the problems ``frozen.check_cell`` finds (expected:
none) and the regime values it was checked against. Composition only — nothing
trains, no data is read.

Usage (repository root)::

    python docs/research/evidence/S11_frozen_arms_execution/code/frozen_plan_snapshot.py
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from spectralquadnet.experiments import frozen

OUT = Path("docs/research/evidence/S11_frozen_arms_execution/frozen_plan.json")


def main() -> int:
    plan = frozen.load_plan()
    cells = []
    for cell in plan.cells:
        spec = cell.spec()
        argv = spec.command(train_script="train.py", nproc_per_node=2)
        argv[0] = "python"  # the interpreter path is the operator's, not the plan's
        cells.append(
            {
                "cell": cell.name,
                "arm": cell.arm,
                "variant": cell.variant,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "arm_overrides": list(cell.arm_overrides),
                "source": cell.source,
                "frozen_command": cell.frozen_command,
                "output_dir": spec.output_dir,
                "command": " ".join(shlex.quote(a) for a in argv),
                "checked_values": frozen.expected_values(cell),
                "check_problems": frozen.check_cell(cell),
            }
        )
    payload = {
        "preregistration_sha256": plan.hashes,
        "added_by_s11": list(frozen.ADDED_BY_S11),
        "n_cells": len(cells),
        "all_compose": all(not c["check_problems"] for c in cells),
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{len(cells)} cells, all compose: {payload['all_compose']} → {OUT}")
    return 0 if payload["all_compose"] else 1


if __name__ == "__main__":
    sys.exit(main())
