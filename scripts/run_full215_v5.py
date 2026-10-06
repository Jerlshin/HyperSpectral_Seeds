#!/usr/bin/env python
"""S40: SeedNet v5 on all 215 bands. Seed 0, both corrected complementary folds, Kaggle T4 x2.

The two cells of ``configs/research/s40_full215_v5.json`` are the only production
run this prepares. Each cell is trained by the S22 runner,
``scripts/run_complementary_v5.py`` (hash-pinned, unchanged), under ``torchrun``:
the same composition, frozen S21 rows, R1 regime, DDP evaluation, TTA and result
files as the k32 v5 cells it is compared with. Only ``data=`` differs.

Subcommands
───────────
    link     symlink ./dataset_refl215_f16 to the attached Kaggle dataset (found by
             its band_axis.json, so the input slug does not matter)
    check    verify the plan and every hashed input, re-hash the cube against its
             MANIFEST, load both frozen folds, report GPUs, RAM, disk and cell states
    run      check, then train every cell that is not done: a finished cell is
             skipped, an interrupted one resumes at its next epoch. Each finished
             cell's outputs are validated. COMPLETE.json is written at the end
    status   each cell's state, without checks
    archive  one .tar.gz of the study's outputs to download

Kaggle (README §10)::

    python scripts/run_full215_v5.py link
    python scripts/run_full215_v5.py check
    python scripts/run_full215_v5.py run --nproc-per-node 2 --stream
    python scripts/run_full215_v5.py archive
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from spectralquadnet.data.prep.multimodal import sha256
from spectralquadnet.data.prep.preslice import BAND_AXIS_FILE, MANIFEST_FILE, verify_presliced
from spectralquadnet.experiments.frozen_rows import load_frozen_rows
from spectralquadnet.experiments.rgb_probe import verify_plan

PLAN = Path("configs/research/s40_full215_v5.json")
OUTPUT_ROOT = Path("outputs/s40_full215_v5")
DATASET = Path("dataset_refl215_f16")
RUNNER = "scripts/run_complementary_v5.py"
N_BANDS = 215
TAGS = ("tta", "no_tta")


def log(msg: str) -> None:
    print(f"[S40] {msg}", flush=True)


def cell_dir(root: Path, cell: dict[str, int]) -> Path:
    return root / f"f{cell['fold']}_s{cell['seed']}"


def cell_state(out: Path) -> str:
    """``done`` / ``new`` / ``interrupted`` / ``empty`` / ``broken``."""
    if (out / "results" / "run.json").exists():
        return "done"
    if not out.exists():
        return "new"
    if (out / "provenance.json").exists():
        return "interrupted"
    if not any(out.iterdir()):
        return "empty"  # claimed by the runner, then killed before it wrote anything
    return "broken"


# ── check ─────────────────────────────────────────────────────────────


def check_dataset(dataset: Path, quick: bool) -> dict[str, Any]:
    if not (dataset / MANIFEST_FILE).exists():
        raise SystemExit(f"{dataset}/{MANIFEST_FILE} not found: link or build the 215-band cube")
    axis = json.loads((dataset / BAND_AXIS_FILE).read_text())
    if (
        axis["set"] != "all"
        or axis["source_n_bands"] != N_BANDS
        or len(axis["source_band_indices"]) != N_BANDS
    ):
        raise SystemExit(f"{dataset} is not the {N_BANDS}-of-{N_BANDS}-band cube: {axis['set']}")
    started = time.monotonic()
    problems = verify_presliced(dataset, checksums=not quick)
    if problems:
        raise SystemExit("dataset does not match its manifest:\n  " + "\n  ".join(problems))
    manifest = json.loads((dataset / MANIFEST_FILE).read_text())
    return {
        "path": str(dataset.resolve()),
        "manifest_sha256": sha256(dataset / MANIFEST_FILE),
        "patches_sha256": manifest["files"]["patches.npy"]["sha256"],
        "bytes": manifest["total_bytes"],
        "storage_dtype": axis["storage_dtype"],
        "rehashed": not quick,
        "verify_seconds": round(time.monotonic() - started, 1),
    }


def check_rows(plan: dict[str, Any], dataset: Path) -> dict[str, Any]:
    out = {}
    for fold in sorted({c["fold"] for c in plan["cells"]}):
        rows = load_frozen_rows(
            Path(plan["partition_plan"]), dataset / "labels.npy", dataset / "groups.npy", fold
        )
        parts = [rows.train, rows.calib, rows.val, rows.test]
        joined = np.concatenate(parts)
        if np.unique(joined).size != joined.size:
            raise SystemExit(f"fold {fold}: frozen partitions overlap")
        held = np.concatenate([rows.val, rows.test])
        groups = np.asarray(rows.groups)
        if set(groups[held].tolist()) & set(groups[rows.train].tolist()):
            raise SystemExit(f"fold {fold}: a held-out scan also appears in train")
        out[str(fold)] = {
            k: int(len(v)) for k, v in zip(("train", "calib", "val", "test"), parts, strict=True)
        }
    return out


def host_report(nproc: int, allow_cpu: bool, cube_bytes: int) -> dict[str, Any]:
    import torch

    gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if not allow_cpu and len(gpus) < nproc:
        raise SystemExit(
            f"{nproc} ranks requested but {len(gpus)} CUDA device(s) visible "
            "(Kaggle: Accelerator = GPU T4 x2)"
        )
    mem: dict[str, float] = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                mem[key + "_GB"] = round(int(value.split()[0]) / 1e6, 1)
        if mem.get("MemTotal_GB", 99.0) < cube_bytes / 1e9 + 8:
            log(
                f"WARNING host RAM {mem} leaves little room beside the {cube_bytes / 1e9:.1f} GB "
                "cube in the page cache; expect slower epochs, not wrong results"
            )
    free = shutil.disk_usage(".").free / 1e9
    if free < 3:
        raise SystemExit(f"only {free:.1f} GB free for checkpoints and logs")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [{"name": g.name, "memory_GB": round(g.total_memory / 1e9, 1)} for g in gpus],
        "disk_free_GB": round(free, 1),
        **mem,
    }


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check(args: argparse.Namespace) -> dict[str, Any]:
    plan = verify_plan(args.plan)  # plan hash + every hashed source, config and dataset file
    log(
        f"plan {args.plan} OK ({sha256(args.plan)[:12]}…), {len(plan['input_hashes'])} inputs match"
    )
    cells = plan["cells"]
    log(f"cells: {cells}")
    dataset = check_dataset(args.dataset, args.quick)
    log(
        f"dataset {dataset['path']}: {dataset['bytes'] / 1e9:.2f} GB {dataset['storage_dtype']}, "
        + ("re-hashed against MANIFEST" if dataset["rehashed"] else "sizes only (--quick)")
        + f" in {dataset['verify_seconds']} s"
    )
    rows = check_rows(plan, args.dataset)
    log(f"frozen rows: {rows}")
    host = host_report(args.nproc_per_node, args.allow_cpu, dataset["bytes"])
    log(f"host: {host}")
    states = {
        cell_dir(args.output_root, c).name: cell_state(cell_dir(args.output_root, c)) for c in cells
    }
    log(f"cells: {states}")
    if "broken" in states.values():
        raise SystemExit("a cell directory has files but no provenance.json; move it aside first")
    return {
        "plan_sha256": sha256(args.plan),
        "repo_commit": git_head(),
        "dataset": dataset,
        "rows": rows,
        "host": host,
        "cell_states": states,
    }


# ── run ───────────────────────────────────────────────────────────────


def stream_to(cmd: list[str], log_path: Path, env: dict[str, str], echo: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_path.open("a") as fh,
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1
        ) as proc,
    ):
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        return proc.wait()


def validate_cell(
    out: Path, plan_path: Path, plan: dict[str, Any], cell: dict[str, int], dataset: Path
) -> dict[str, Any]:
    """Fail unless the cell's outputs are complete, from this plan, and score each row once."""
    prov = json.loads((out / "provenance.json").read_text())
    if (prov["training_plan_sha256"], prov["fold"], prov["seed"]) != (
        sha256(plan_path),
        cell["fold"],
        cell["seed"],
    ):
        raise SystemExit(f"{out}: provenance does not match the plan cell")
    run = json.loads((out / "results" / "run.json").read_text())["run"]
    geometry = run["band_geometry"]
    if (
        run["band_selection"]
        or geometry["selected"] != N_BANDS
        or geometry.get("acquired", geometry["stored"]) != N_BANDS
    ):
        raise SystemExit(f"{out}: not a {N_BANDS}-band full-spectrum run: {geometry}")
    if not json.loads((out / "last_stage1.json").read_text())["finished"]:
        raise SystemExit(f"{out}: training state is not marked finished")
    for name in ("best_stage1.pth", "stage1_meta.json", "metrics.jsonl", "resolved_config.yaml"):
        if not (out / name).exists():
            raise SystemExit(f"{out}: {name} missing")
    rows = load_frozen_rows(
        Path(plan["partition_plan"]), dataset / "labels.npy", dataset / "groups.npy", cell["fold"]
    )
    expected = {
        "val_test": np.sort(np.concatenate([rows.val, rows.test])),
        "calib": np.sort(rows.calib),
    }
    n_classes = int(np.unique(rows.labels).size)
    report: dict[str, Any] = {"cell": cell, "band_geometry": geometry}
    for split, want in expected.items():
        for tag in TAGS:
            z = np.load(out / "results" / f"logits_{split}_{tag}.npz")
            got, logits, targets = z["rows"], z["logits"], z["targets"]
            if np.unique(got).size != got.size:
                raise SystemExit(f"{out}: {split}/{tag} scores a row twice")
            if not np.array_equal(np.sort(got), want):
                raise SystemExit(f"{out}: {split}/{tag} rows differ from the frozen {split} rows")
            if (
                logits.shape != (want.size, n_classes)
                or not np.isfinite(logits.astype(np.float32)).all()
            ):
                raise SystemExit(f"{out}: {split}/{tag} logits {logits.shape} malformed")
            if not np.array_equal(targets, rows.labels[got]):
                raise SystemExit(f"{out}: {split}/{tag} targets disagree with labels.npy")
            report[f"{split}_{tag}_rows"] = int(got.size)
    for tag in TAGS:
        if not np.array_equal(
            np.load(out / "results" / f"rows_val_test_{tag}.npy"),
            np.load(out / "results" / f"logits_val_test_{tag}.npz")["rows"],
        ):
            raise SystemExit(f"{out}: rows_val_test_{tag}.npy disagrees with its logits file")
    report["rows_scored_once"] = True
    return report


def rendezvous(args: argparse.Namespace) -> list[str]:
    if args.master_port:
        return [
            "--nnodes=1",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
        ]
    return ["--standalone"]


def run(args: argparse.Namespace) -> None:
    receipt = check(args)
    plan = json.loads(args.plan.read_text())
    root: Path = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    session = {
        "event": "session",
        "at": _dt.datetime.now().isoformat(timespec="seconds"),
        "nproc_per_node": args.nproc_per_node,
        **receipt,
    }
    with (root / "sessions.jsonl").open("a") as fh:
        fh.write(json.dumps(session) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(Path("src").resolve()), env.get("PYTHONPATH", "")])
    )
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    started = time.monotonic()
    checks = []
    for cell in plan["cells"]:
        out = cell_dir(root, cell)
        state = cell_state(out)
        if state == "done":
            log(f"{out.name}: done, skipped")
        else:
            if state == "empty":
                out.rmdir()
            resume = state == "interrupted"
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                *rendezvous(args),
                f"--nproc_per_node={args.nproc_per_node}",
                RUNNER,
                "train",
                "--plan",
                str(args.plan),
                "--fold",
                str(cell["fold"]),
                "--seed",
                str(cell["seed"]),
                "--output",
                str(out),
                *(["--resume"] if resume else []),
            ]
            log(f"{out.name}: {'resuming' if resume else 'starting'} — {' '.join(cmd)}")
            t0 = time.monotonic()
            code = stream_to(cmd, root / "logs" / f"{out.name}.log", env, args.stream)
            attempt = {
                "event": "attempt",
                "cell": out.name,
                "resume": resume,
                "returncode": code,
                "seconds": round(time.monotonic() - t0, 1),
                "at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            with (root / "sessions.jsonl").open("a") as fh:
                fh.write(json.dumps(attempt) + "\n")
            if code != 0:
                raise SystemExit(
                    f"{out.name} exited {code}; see {root / 'logs' / out.name}.log. "
                    "Re-run the same command to resume it."
                )
        report = validate_cell(out, args.plan, plan, cell, args.dataset)
        (root / "checks").mkdir(exist_ok=True)
        (root / "checks" / f"{out.name}.json").write_text(json.dumps(report, indent=2) + "\n")
        log(f"{out.name}: outputs validated — every frozen held-out and calib row scored once")
        checks.append(report)
    complete = {
        "study": plan["study"],
        "plan_sha256": sha256(args.plan),
        "cells": plan["cells"],
        "checks": checks,
        "session_seconds": round(time.monotonic() - started, 1),
        "completed_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (root / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")
    log(f"all {len(plan['cells'])} cells complete → {root / 'COMPLETE.json'}")


# ── link / status / archive ───────────────────────────────────────────


def link(args: argparse.Namespace) -> None:
    candidates = []
    for manifest in Path(args.input_root).rglob(MANIFEST_FILE):
        axis_path = manifest.parent / BAND_AXIS_FILE
        if axis_path.exists() and json.loads(axis_path.read_text()).get("set") == "all":
            candidates.append(manifest.parent)
    if len(candidates) != 1:
        raise SystemExit(f"expected one 215-band cube under {args.input_root}, found {candidates}")
    if args.dataset.is_symlink():
        args.dataset.unlink()
    elif args.dataset.exists():
        raise SystemExit(f"{args.dataset} exists and is not a symlink; not replacing it")
    args.dataset.symlink_to(candidates[0].resolve(), target_is_directory=True)
    log(f"{args.dataset} → {candidates[0]}")


def status(args: argparse.Namespace) -> None:
    plan = json.loads(args.plan.read_text())
    for cell in plan["cells"]:
        out = cell_dir(args.output_root, cell)
        state = cell_state(out)
        epoch = ""
        if (out / "last_stage1.json").exists():
            meta = json.loads((out / "last_stage1.json").read_text())
            epoch = f" (epoch {meta['epoch']}, finished={meta['finished']})"
        log(f"{out.name}: {state}{epoch}")


def archive(args: argparse.Namespace) -> None:
    dest = args.to or (
        Path("/kaggle/working/s40_full215_v5_outputs.tar.gz")
        if Path("/kaggle/working").is_dir()
        else Path("outputs/s40_full215_v5_outputs.tar.gz")
    )
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(args.output_root, arcname=str(args.output_root))
    log(f"{dest}  ({dest.stat().st_size / 1e6:.0f} MB) — unpack at the repository root")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("command", choices=["link", "check", "run", "status", "archive"])
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--nproc-per-node", type=int, default=2)
    p.add_argument("--stream", action="store_true", help="echo the training output live")
    p.add_argument(
        "--quick", action="store_true", help="check sizes only, skip re-hashing the cube"
    )
    p.add_argument("--allow-cpu", action="store_true", help="smoke tests only: no CUDA required")
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=0, help="explicit rendezvous (macOS smoke)")
    p.add_argument("--input-root", default="/kaggle/input", help="link: where datasets are mounted")
    p.add_argument("--to", type=Path, help="archive: destination .tar.gz")
    args = p.parse_args()
    {"link": link, "check": check, "run": run, "status": status, "archive": archive}[args.command](
        args
    )


if __name__ == "__main__":
    main()
