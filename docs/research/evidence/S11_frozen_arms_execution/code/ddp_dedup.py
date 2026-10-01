#!/usr/bin/env python3
"""S11 validation of P0.1's DDP de-duplication on a real two-rank ``torchrun`` run.

``DistributedSampler`` pads a split whose size does not divide by the world size
with its own first kernels; before S11 the gathered predictions kept that
duplicate (S09 §8: 4,312 predictions for 4,311 held-out kernels). This trains a
miniature SpectralSeedNet for 2 epochs on 2 gloo/CPU ranks, on a synthetic cube
whose held-out split has an **odd** size (7 classes × 13 kernels = 91), and
counts what the final evaluation wrote — through any source tree, so the same
run can be made with the code before and after S11.

Usage (repository root)::

    python docs/research/evidence/S11_frozen_arms_execution/code/ddp_dedup.py \\
        --repo . --repo <worktree of the parent commit> \\
        --out docs/research/evidence/S11_frozen_arms_execution/ddp_dedup.json

macOS needs the explicit loopback rendezvous used here (``--standalone`` resolves
the host name to an address gloo cannot reach); Linux/Kaggle runs either.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

C, B, P, NB, S = 7, 2, 13, 8, 16  # 7 x 13 = 91 held-out kernels: odd on purpose


def build(root: Path) -> dict[str, str]:
    rng = np.random.default_rng(0)
    n = C * B * P
    patches = np.zeros((n, NB, S, S), np.float32)
    labels = np.zeros(n, np.int64)
    groups = np.zeros(n, np.int64)
    masks = np.ones((n, S, S), np.float16)
    morph = rng.normal(size=(n, 8)).astype(np.float32)
    r = 0
    for c in range(C):
        sig = rng.normal(size=(NB, 1, 1)) * 0.5
        for b in range(B):
            off = rng.normal(size=(NB, 1, 1)) * 0.1
            for _ in range(P):
                p = sig + off + rng.normal(size=(NB, S, S)) * 0.2
                p[:, :2] = 0
                masks[r, :2] = 0
                patches[r], labels[r], groups[r] = p, c, c * B + b
                r += 1
    root.mkdir(parents=True, exist_ok=True)
    arrays = dict(patches=patches, labels=labels, groups=groups, masks=masks, morphology=morph)
    for k, v in arrays.items():
        np.save(root / f"{k}.npy", v)
    wl = np.sort(rng.uniform(400, 1000, NB))
    (root / "wavelengths.csv").write_text(
        "Band,Wavelength (nm)\n" + "\n".join(f"{i},{v:.2f}" for i, v in enumerate(wl)) + "\n"
    )
    return {
        "data.patches_data": str(root / "patches.npy"),
        "data.labels_path": str(root / "labels.npy"),
        "data.groups_path": str(root / "groups.npy"),
        "data.masks_path": str(root / "masks.npy"),
        "data.morphology_path": str(root / "morphology.npy"),
        "data.wavelength_path": str(root / "wavelengths.csv"),
    }


def run(repo: Path, data: dict[str, str], out: Path) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        **data,
        "data.gain_path": "",
        "data.scan_table_path": "",
        "data.num_bands": NB,
        "data.num_classes": C,
        "data.cutmix_bands": 2,
        "data.max_cutout_bands": 1,
        "single.epochs": 2,
        "single.batch": 8,
        "single.warmup_ep": 1,
        "single.mixup_epochs": 1,
        "single.margin_warmup_start": 2,
        "single.margin_warmup_end": 2,
        "model.stem_channels": 16,
        "model.stem_folded_depth": 1,
        "model.spatial_width_mult": 0.25,
        "model.spectral_hidden": 32,
        "model.index_bank_size": 8,
        "model.continuum_depths": 4,
        "model.aux_head_hidden": 16,
        "evaluation.bootstrap_samples": 0,
        "tta_spatial": 2,
        "tta_spectral": 1,
        "device": "cpu",
        "runtime.num_workers": 0,
        "runtime.compile": "off",
        "runtime.multi_gpu": "ddp",
        "tracking.backend": "jsonl",
        "output_dir": str(out),
        "hydra.run.dir": str(out / "hydra"),
    }
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
        "--master_addr=127.0.0.1", f"--master_port={port}", str(repo / "train.py"),
        *[f"{k}={v}" for k, v in overrides.items()],
    ]
    res = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-3000:] + res.stderr[-3000:])

    results = out / "results"
    manifest = json.loads((results / "run.json").read_text())
    sizes = manifest["run"]["split_report"]["sizes"]
    report: dict[str, Any] = {
        "held_out_kernels": sizes["val"] + sizes["test"],
        "calib_kernels": sizes["calib"],
        "predictions_written": int(len(np.load(results / "preds_val_test_no_tta.npy"))),
        "support_sum": int(sum(c["support"] for c in manifest["results"]["no_tta"]["per_class"])),
    }
    for split in ("val_test_no_tta", "val_test_tta", "calib_no_tta", "calib_tta"):
        path = results / f"logits_{split}.npz"
        if path.exists():
            rows = np.load(path)["rows"]
            report[f"logits_{split}"] = {"n": int(len(rows)), "unique": int(len(np.unique(rows)))}
    fit = out / "clean_fit.json"
    if fit.exists():
        report["clean_fit_final_n"] = json.loads(fit.read_text())["final"]["live"]["n"]
    code = manifest["run"].get("code")
    report["code"] = code
    report["world_size"] = (manifest["run"].get("environment") or {}).get("world_size", 2)
    report["each_kernel_once"] = report["predictions_written"] == report["held_out_kernels"]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload: dict[str, Any] = {"world_size": 2, "runs": {}}
    with tempfile.TemporaryDirectory() as tmp:
        data = build(Path(tmp) / "data")
        for i, repo in enumerate(args.repo):
            report = run(repo.resolve(), data, Path(tmp) / f"run{i}")
            payload["runs"][str(repo)] = report
            print(repo, json.dumps(report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
