#!/usr/bin/env python3
"""Snapshot of the S15 plan as the runner builds it, the code-identity proof and the runtime estimate.

Writes beside this folder:
  frozen_plan.json   every cell: frozen source, resolved keys, composition check, exact torchrun command;
                     the three pre-registration hashes; R1; and that no cell repeats an S13 cell
  code_identity.json the content digest of the training code at aed5257 (from `git archive`, i.e. the
                     commit's own blobs) and in this working tree, and `git diff --stat aed5257` on the scope
  runtime_estimate.json  per-cell minutes from the S13 cells' own wall clock on the same runtime
                     (evidence/S14_screen_reading/cells.csv → wall_min)

Equivalent to `scripts/run_s15.py --list --check --dry-run --nproc-per-node 2`, kept as evidence. Needs git
history (aed5257) for code_identity.json; reads no data and no held-out row.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from spectralquadnet.experiments import s13, s15

REPO = Path(__file__).resolve().parents[5]
EVID = Path(__file__).resolve().parents[1]
WATCHED = (
    "single.mixup_epochs", "single.arcface_m", "grad_clip", "single.epochs", "single.patience",
    "model.pathways", "model.spatial_mixstyle", "model.spectral_descriptor",
    "model.spatial_tail_strides", "model.cbam_min_hw", "data.split_scheme", "data.split_fold",
    "data.split_eval_frac", "seed", "evaluation.session_probe", "evaluation.save_logits",
)


def snapshot(plan: s15.S15Plan) -> dict:
    cells = []
    for cell in plan.cells:
        cfg = s15.compose_cell(cell)
        resolved = {}
        for key in WATCHED:
            node = cfg
            for part in key.split("."):
                node = node[part]
            resolved[key] = list(node) if hasattr(node, "__iter__") and not isinstance(node, str) else node
        command = cell.spec().command(train_script="train.py", nproc_per_node=2)
        cells.append({
            "cell": cell.name, "source": cell.source, "data": cell.data, "seed": cell.seed,
            "arm_overrides": list(cell.arm_overrides), "resolved": resolved,
            "check": s15.check_cell(cell) or "ok",
            "command": " ".join(shlex.quote(c) for c in ["torchrun", "--standalone", "--nproc_per_node=2", *command[5:]]),
        })
    s13_names = {c.name for c in s13.load_plan().cells}
    return {
        "preregistration_sha256": plan.hashes,
        "regime_r1": list(plan.cells[0].regime),
        "regime_check": s15.check_regime_is_r1(plan) or "ok",
        "repeats_an_s13_cell": sorted(c.name for c in plan.cells if c.name in s13_names) or "none",
        "gpu_cells": cells,
    }


def code_identity() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        blob = subprocess.run(["git", "archive", s15.CODE_REFERENCE_COMMIT], cwd=REPO, capture_output=True,
                              check=True).stdout
        with tarfile.open(fileobj=BytesIO(blob)) as tar:
            tar.extractall(tmp, filter="data")
        ref = s15.code_digest(Path(tmp))
    cur = s15.code_digest(REPO)
    scope = ["src/spectralquadnet", "configs", "train.py", ":(exclude)src/spectralquadnet/experiments"]
    diff = subprocess.run(["git", "diff", "--stat", s15.CODE_REFERENCE_COMMIT, "--", *scope], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    return {
        "reference_commit": s15.CODE_REFERENCE_COMMIT,
        "scope": {"globs": s15.CODE_SCOPE, "files": s15.CODE_SCOPE_FILES, "excluded": s15.CODE_SCOPE_EXCLUDE},
        "digest_at_reference": ref[0], "n_files_at_reference": ref[1],
        "digest_pinned_in_s15": s15.CODE_REFERENCE_DIGEST,
        "digest_worktree": cur[0], "n_files_worktree": cur[1],
        "identical": ref[0] == cur[0] == s15.CODE_REFERENCE_DIGEST,
        "git_diff_stat_vs_reference_on_scope": diff or "(empty)",
    }


def runtime(plan: s15.S15Plan) -> dict:
    c14 = pd.read_csv(REPO / "docs/research/evidence/S14_screen_reading/cells.csv").set_index("cell")
    x1_epoch_s = 7.9  # S13 runtime_estimate: X1 grouped on T4 x 2 (S11 wall clock)
    per = {
        "lean_grouped": float(c14.loc[["Y3/lean_grouped__f0_s0", "Y3/lean_grouped__f1_s0"], "wall_min"].mean()),
        "lean_stratified": float(c14.loc["Y3/lean_stratified__f0_s0", "wall_min"]),
        "desc_only": 25.7,  # ≈ X1 grouped (the shipped tail): S13 runtime_estimate.csv
        "spatial_repair": float(c14.loc[["Y3/lean_grouped__f0_s0", "Y3/lean_grouped__f1_s0"], "wall_min"].mean()),
    }
    rows = [{"cell": c.name, "minutes": per[c.variant]} for c in plan.cells]
    total = sum(r["minutes"] for r in rows)
    return {"source": "S13 cells' own wall clock on Kaggle T4 x 2 (evidence/S14_screen_reading/cells.csv); "
                      f"desc_only ≈ X1 grouped ({x1_epoch_s} s/epoch, S13 runtime_estimate)",
            "per_cell": rows, "total_minutes": total, "total_hours": total / 60,
            "cap_hours_if_every_cell_runs_200_epochs": total / 60 * 200 / 180}


def main() -> None:
    plan = s15.load_plan()
    snap, ident, rt = snapshot(plan), code_identity(), runtime(plan)
    for name, obj in (("frozen_plan.json", snap), ("code_identity.json", ident), ("runtime_estimate.json", rt)):
        (EVID / name).write_text(json.dumps(obj, indent=1, default=str) + "\n")
        print("wrote", (EVID / name).relative_to(REPO))
    print({c["cell"]: c["check"] for c in snap["gpu_cells"]})
    print("code identical:", ident["identical"], ident["n_files_worktree"], "files")
    print(f"runtime ≈ {rt['total_minutes']:.0f} min ({rt['total_hours']:.1f} h)")


if __name__ == "__main__":
    main()
