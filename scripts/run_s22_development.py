#!/usr/bin/env python
"""Finish S22's amended screen, analyze, and run only its justified S23 successor.

Use --wait-pid for a currently live local fold0 fit. This coordinator never kills
processes and never launches a second copy while that process is alive.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import verify_plan

PLAN = Path("configs/research/s22_screening_amendment05.json")
STATE = Path("outputs/s22_development_execution.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    plan = verify_plan(PLAN)
    if STATE.exists():
        raise FileExistsError("Coordinator already started; inspect state before replay")
    state: dict[str, Any] = {"training_plan_sha256": sha256(PLAN), "status": "running", "steps": []}
    write_json(STATE, state)
    if args.wait_pid is not None:
        print("Waiting for existing training PID", args.wait_pid, flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(15)
    def launch(name: str, argv: list[str]) -> None:
        log = Path("outputs") / f"{name}.log"
        print("Starting", name, flush=True)
        start = time.monotonic()
        with log.open("x") as output:
            result = subprocess.run([sys.executable, *argv], stdout=output, stderr=subprocess.STDOUT, check=False)
        state["steps"].append({"name": name, "returncode": result.returncode,
                               "seconds": time.monotonic() - start, "log": str(log)})
        write_json(STATE, state)
        if result.returncode:
            state["status"] = "failed"
            write_json(STATE, state)
            raise RuntimeError(f"{name} failed; inspect {log}")
        print("Completed", name, flush=True)
    for cell in plan["cells"]:
        fold, seed = cell["fold"], cell["seed"]
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s{seed}")
        if (root / "results/run.json").exists():
            prior = json.loads((root / "provenance.json").read_text())
            if (prior["training_plan_sha256"], prior["fold"], prior["seed"]) != (sha256(PLAN), fold, seed):
                raise ValueError("Completed cell identity mismatch")
            continue
        command = ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   "scripts/run_complementary_v5.py", "train", "--plan", str(PLAN),
                   "--fold", str(fold), "--seed", str(seed)]
        if root.exists():
            command.append("--resume")
        launch(f"s22_coordinator_f{fold}_s{seed}", command)
    launch("s22_coordinator_analysis", ["scripts/analyze_s22_cuda_screening.py"])
    launch("s22_coordinator_archive", ["docs/research/evidence/S22_complementary_v5/code/read_screen.py"])
    launch("s22_coordinator_figures", ["docs/research/evidence/S22_complementary_v5/code/draw_figures.py"])
    gate = json.loads(Path("outputs/s22_fusion_analysis/hypothesis.json").read_text())["development_gate"]
    state["s22_development_gate"] = gate
    if gate:
        launch("s23_coordinator_freeze", ["scripts/run_frozen_multimodal.py", "freeze"])
        launch("s23_coordinator_run", ["scripts/run_frozen_multimodal.py", "run"])
    state["status"] = "complete"
    write_json(STATE, state)
    print("Development sequence complete; synthesize recorded evidence and update research registers.", flush=True)


if __name__ == "__main__":
    main()
