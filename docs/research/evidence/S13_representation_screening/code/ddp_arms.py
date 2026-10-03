#!/usr/bin/env python3
"""S13 validation: every arm type on a real two-rank ``torchrun`` job, then Y1's fusion.

Each S13 cell runs on two GPUs on Kaggle. This runs each arm's *configuration*
— R1 (the reference), Y1 ``[spectral]`` and ``[spatial]``, Y2 masked MixStyle,
Y3 lean, Y4 80/20 — for 2 epochs on 2 gloo/CPU ranks, on S11's synthetic cube
(``ddp_dedup.build``: 7 classes × 2 bundles × 13 kernels, so the grouped held-out
split is **odd**, 91) plus a three-session scan table, and checks what a Kaggle
cell must deliver:

* the job finishes — no DDP unused-parameter error from Y2's parameter-free module
  or Y3's ``nn.Identity`` gates, and both ranks pass the end-of-stage barrier;
* every held-out kernel is scored exactly once (S11 P0.1 still holds);
* the training-rows session κ is present and computed on every training row once
  (its extraction gathers across ranks and de-duplicates);
* the regime block names the arm;
* Y1's two cells fuse (weight on calib) into a model-shaped results tree.

Usage (repository root; ≈ 3–5 min on a laptop CPU)::

    python docs/research/evidence/S13_representation_screening/code/ddp_arms.py \\
        --out docs/research/evidence/S13_representation_screening/ddp_arms.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[5]
S11_DDP = REPO / "docs/research/evidence/S11_frozen_arms_execution/code/ddp_dedup.py"

#: S11's miniature run (ddp_dedup.run), with R1's shape: margin 0, clip 50.
BASE: dict[str, Any] = {
    "data.gain_path": "", "data.num_bands": 8, "data.num_classes": 7, "data.cutmix_bands": 2,
    "data.max_cutout_bands": 1, "single.epochs": 2, "single.batch": 8, "single.warmup_ep": 1,
    "single.mixup_epochs": 1, "single.arcface_m": 0.0, "single.margin_warmup_start": 3,
    "single.margin_warmup_end": 3, "grad_clip": 50.0, "single.patience": 40,
    "model.stem_channels": 16, "model.stem_folded_depth": 1, "model.spatial_width_mult": 0.25,
    "model.spectral_hidden": 32, "model.index_bank_size": 8, "model.continuum_depths": 4,
    "model.aux_head_hidden": 16, "evaluation.bootstrap_samples": 0, "tta_spatial": 2,
    "tta_spectral": 1, "device": "cpu", "runtime.num_workers": 0, "runtime.compile": "off",
    "runtime.multi_gpu": "ddp", "tracking.backend": "jsonl",
}
ARMS: dict[str, dict[str, Any]] = {
    "R1": {},
    "Y1_spectral_only": {"model.pathways": "[spectral]"},
    "Y1_spatial_only": {"model.pathways": "[spatial]"},
    "Y2_mixstyle": {"model.spatial_mixstyle": "true"},
    "Y3_lean": {"model.spectral_descriptor": "snv_morph", "model.spatial_tail_strides": "[2,2,2,1]",
                "model.cbam_min_hw": 3},
    "Y4_8020": {"data.split_scheme": "stratified", "data.split_eval_frac": 0.2},
}


def _build_dataset(root: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("ddp_dedup", S11_DDP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data: dict[str, str] = module.build(root)
    groups = np.load(data["data.groups_path"])
    table = root / "scan_table.csv"
    table.write_text(
        "scan_id,session_id,session\n" + "".join(f"{g},{g % 3},S{g % 3}\n" for g in np.unique(groups))
    )
    return {**data, "data.scan_table_path": str(table)}


def _torchrun(overrides: dict[str, Any], out: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    args = {**overrides, "output_dir": str(out), "hydra.run.dir": str(out / "hydra")}
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
        "--master_addr=127.0.0.1", f"--master_port={port}", str(REPO / "train.py"),
        *[f"{k}={v}" for k, v in args.items()],
    ]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-3000:] + res.stderr[-3000:])


def _report(out: Path) -> dict[str, Any]:
    manifest = json.loads((out / "results" / "run.json").read_text())
    run = manifest["run"]
    sizes = run["split_report"]["sizes"]
    held = sizes["test"] if run.get("report_split") == "test" else sizes["val"] + sizes["test"]
    tag = f"{run.get('report_split', 'val_test')}_tta"
    preds = np.load(out / "results" / f"preds_{tag}.npy")
    probe = manifest.get("session_probe") or {}
    return {
        "world_size": (run.get("environment") or {}).get("world_size"),
        "held_out_kernels": int(held),
        "predictions_written": int(len(preds)),
        "each_kernel_once": int(len(preds)) == int(held),
        "regime": {k: run["regime"][k] for k in (
            "pathways", "spatial_mixstyle", "spectral_descriptor", "spatial_tail_strides",
            "cbam_min_hw", "split_eval_frac", "grad_clip", "arcface_m")},
        "parameters": run.get("parameters"),
        "session_probe_rows": probe.get("n_rows"),
        "train_rows": sizes["train"],
        "probe_every_training_row_once": probe.get("n_rows") == sizes["train"],
        "kappa": {k: (v or {}).get("kappa") for k, v in (probe.get("representations") or {}).items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from spectralquadnet.experiments import fusion

    payload: dict[str, Any] = {"launcher": "torchrun --nproc_per_node=2 (gloo, CPU)", "arms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        data = _build_dataset(Path(tmp) / "data")
        for arm, extra in ARMS.items():
            _torchrun({**data, **BASE, **extra}, Path(tmp) / arm)
            payload["arms"][arm] = _report(Path(tmp) / arm)
            print(arm, json.dumps(payload["arms"][arm]), flush=True)
        man = fusion.fuse_cells(
            Path(tmp) / "Y1_spectral_only", Path(tmp) / "Y1_spatial_only", Path(tmp) / "Y1_fused",
            num_classes=7, sessions=None, n_boot=0,
        )
        n_held = payload["arms"]["Y1_spectral_only"]["held_out_kernels"]
        payload["Y1_fused"] = {
            "weight": man["fusion"]["weight"],
            "n_scored": man["results"]["tta"]["n_samples"],
            "each_kernel_once": man["results"]["tta"]["n_samples"] == n_held,
        }
    payload["all_ok"] = bool(
        all(a["each_kernel_once"] and a["probe_every_training_row_once"] and a["world_size"] == 2
            for a in payload["arms"].values())
        and payload["Y1_fused"]["each_kernel_once"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    print("all_ok:", payload["all_ok"])
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
