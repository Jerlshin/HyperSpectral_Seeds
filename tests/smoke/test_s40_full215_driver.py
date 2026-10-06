"""S40 end to end on CPU: the driver, the unchanged S22 runner, two gloo ranks, 215 bands.

A miniature version of the production run: a float16 215-band cube on the real
instrument's band axis, built by ``build_presliced --set all``, with frozen
complementary rows whose held-out halves have odd sizes, so ``DistributedSampler``
pads them, and the v5 overrides of the S40 plan with only magnitudes shrunk. The
driver is killed for real after the first cell's first saved epoch. It is re-run to
resume that cell and finish both, and run once more to skip them.
Every held-out and calib row must be scored exactly once.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from _helpers import REPO_ROOT

from spectralquadnet.data.prep.multimodal import sha256
from spectralquadnet.data.prep.preslice import build_presliced

pytestmark = pytest.mark.slow

DRIVER = REPO_ROOT / "scripts/run_full215_v5.py"
BANDS, SIDE, CLASSES, PER_SCAN = 215, 32, 6, 11
EPOCHS = 6


def _wavelengths() -> np.ndarray:
    """The instrument's 256-band grid with the 41 clipped bands (608-706 nm) removed."""
    grid = np.linspace(383.2, 1006.5, 256)
    keep = np.flatnonzero((grid < 607.9) | (grid > 705.9))
    return keep[:BANDS], grid[keep[:BANDS]]


@pytest.fixture(scope="module")
def mini(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("s40")
    src = root / "source"
    src.mkdir()
    rng = np.random.default_rng(0)
    n = CLASSES * 2 * PER_SCAN
    x = np.zeros((n, BANDS, SIDE, SIDE), dtype=np.float32)
    labels = np.repeat(np.arange(CLASSES), 2 * PER_SCAN)
    groups = np.repeat(np.arange(2 * CLASSES), PER_SCAN)
    masks = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    masks[:, 4:-4, 4:-4] = 1.0
    for i in range(n):
        spectrum = 0.3 + 0.1 * np.sin(np.linspace(0, 3, BANDS) * (1 + labels[i])) + 0.02 * groups[i]
        x[i] = (spectrum[:, None, None] + rng.normal(0, 0.02, (BANDS, SIDE, SIDE))) * masks[i]
    np.save(src / "patches.npy", x)
    np.save(src / "labels.npy", labels.astype(np.int64))
    np.save(src / "groups.npy", groups.astype(np.int64))
    np.save(src / "masks.npy", masks)
    np.save(src / "morphology.npy", rng.normal(size=(n, 8)).astype(np.float32))
    index, nm = _wavelengths()
    (src / "wavelengths.csv").write_text(
        "index,Wavelength (nm)\n"
        + "".join(f"{b + 1},{w:.6f}\n" for b, w in zip(index, nm, strict=True))
    )
    scans = [
        f"{s},S{s % 3}/V{s // 2}-0{s % 2 + 1},S{s % 3},{s % 3},V{s // 2},{s // 2},m{s},{PER_SCAN}"
        for s in range(2 * CLASSES)
    ]
    (src / "scan_table.csv").write_text(
        "scan_id,scan_key,session,session_id,variety,label,member,n_patches\n"
        + "\n".join(scans)
        + "\n"
    )
    (src / "radiometry.json").write_text("{}\n")
    data = root / "dataset_refl215_f16"
    build_presliced(src, data, None, set_name="all")

    splits = {}
    for fold in (0, 1):
        train_scan = groups % 2 == fold
        train_rows = np.flatnonzero(train_scan)
        calib = np.concatenate([train_rows[labels[train_rows] == c][:2] for c in range(CLASSES)])
        held = np.flatnonzero(~train_scan)  # 66 rows: val 33 / test 33, both odd under 2 ranks
        splits[str(fold)] = {
            "train": np.setdiff1d(train_rows, calib).tolist(),
            "calib": calib.tolist(),
            "val": held[::2].tolist(),
            "test": held[1::2].tolist(),
        }
    partition = root / "partition.json"
    partition.write_text(
        json.dumps(
            {
                "data": str(data),
                "splits": splits,
                "input_hashes": {
                    str(data / "labels.npy"): sha256(data / "labels.npy"),
                    str(data / "groups.npy"): sha256(data / "groups.npy"),
                },
            }
        )
    )
    partition.with_suffix(".sha256").write_text(sha256(partition) + "\n")

    production = json.loads(
        (REPO_ROOT / "configs/research/s22_screening_amendment05.json").read_text()
    )
    v5 = [o for o in production["v5_overrides"] if not o.startswith("data=")]
    paths = {
        k: data / f
        for k, f in [
            ("patches_data", "patches.npy"),
            ("labels_path", "labels.npy"),
            ("groups_path", "groups.npy"),
            ("masks_path", "masks.npy"),
            ("morphology_path", "morphology.npy"),
            ("wavelength_path", "wavelengths.csv"),
            ("scan_table_path", "scan_table.csv"),
        ]
    }
    plan = {
        "study": "S40_smoke",
        "partition_plan": str(partition),
        "v5_overrides": [
            "data=refl215_f16_grouped",
            *v5,
            *(f"data.{k}={v}" for k, v in paths.items()),
            f"data.num_classes={CLASSES}",
            f"single.epochs={EPOCHS}",
            "single.batch=8",
            "single.warmup_ep=1",
            "single.patience=20",
            "single.clean_fit_kernels=12",
            "evaluation.bootstrap_samples=16",
            "tta_spatial=2",
            "tta_spectral=1",
        ],
        "cells": [{"fold": 0, "seed": 0}, {"fold": 1, "seed": 0}],
        "runtime_overrides": [
            "runtime=default",
            "tracking=console_jsonl",
            "device=cpu",
            "runtime.multi_gpu=ddp",
            "runtime.num_workers=0",
            "runtime.eval_num_workers=0",
            "runtime.compile=off",
            "runtime.checkpoint_every=1",
            "runtime.progress=off",
        ],
        "input_hashes": {
            "scripts/run_complementary_v5.py": sha256(REPO_ROOT / "scripts/run_complementary_v5.py")
        },
    }
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    plan_path.with_suffix(".sha256").write_text(f"{sha256(plan_path)}  plan.json\n")
    return {"data": data, "plan": plan_path, "out": root / "outputs"}


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cmd(mini: dict[str, Path], command: str) -> list[str]:
    return [
        sys.executable,
        str(DRIVER),
        command,
        "--plan",
        str(mini["plan"]),
        "--dataset",
        str(mini["data"]),
        "--output-root",
        str(mini["out"]),
        "--nproc-per-node",
        "2",
        "--allow-cpu",
        "--master-port",
        str(_port()),
    ]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")])
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return env


def test_the_driver_trains_resumes_validates_and_skips(mini) -> None:
    cell0 = mini["out"] / "f0_s0"
    # 1 · start, then kill the whole launcher tree once epoch 1's training state is on disk.
    proc = subprocess.Popen(
        _cmd(mini, "run"),
        cwd=REPO_ROOT,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline and proc.poll() is None:
        meta = cell0 / "last_stage1.json"
        if (cell0 / "last_stage1.pth").exists() and meta.exists():
            state = json.loads(meta.read_text())
            if 1 <= state["epoch"] < EPOCHS - 1 and not state["finished"]:
                break
        time.sleep(0.2)
    assert proc.poll() is None, proc.stdout.read()[-4000:]
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    killed_at = json.loads((cell0 / "last_stage1.json").read_text())["epoch"]
    assert not (cell0 / "results" / "run.json").exists()

    # 2 · the same command resumes cell 0 at its next epoch and runs cell 1.
    second = subprocess.run(
        _cmd(mini, "run"), cwd=REPO_ROOT, env=_env(), capture_output=True, text=True, timeout=2400
    )
    assert second.returncode == 0, second.stdout[-6000:] + second.stderr[-3000:]
    assert "f0_s0: resuming" in second.stdout and "f1_s0: starting" in second.stdout
    log0 = (mini["out"] / "logs" / "f0_s0.log").read_text()
    assert f"continuing at epoch {killed_at + 1}" in log0
    assert "world_size=2" in log0
    assert "Spectral: 215 bands — the full acquired cube, no band selection" in log0
    complete = json.loads((mini["out"] / "COMPLETE.json").read_text())
    assert [c["cell"] for c in complete["checks"]] == [
        {"fold": 0, "seed": 0},
        {"fold": 1, "seed": 0},
    ]
    for check in complete["checks"]:
        assert check["rows_scored_once"] and check["val_test_tta_rows"] == 66
        assert check["calib_no_tta_rows"] == 2 * CLASSES
        assert check["band_geometry"]["selected"] == BANDS
    run = json.loads((cell0 / "results" / "run.json").read_text())["run"]
    assert run["band_selection"] is False and run["parameters"] > 0
    assert json.loads((cell0 / "last_stage1.json").read_text())["epoch"] == EPOCHS
    sessions = [
        json.loads(line) for line in (mini["out"] / "sessions.jsonl").read_text().splitlines()
    ]
    assert [s["event"] for s in sessions].count("session") == 2

    # 3 · a third run trains nothing.
    third = subprocess.run(
        _cmd(mini, "run"), cwd=REPO_ROOT, env=_env(), capture_output=True, text=True, timeout=600
    )
    assert third.returncode == 0, third.stdout[-4000:]
    assert third.stdout.count("done, skipped") == 2

    archive = subprocess.run(
        [*_cmd(mini, "archive"), "--to", str(mini["out"].parent / "a.tar.gz")],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
    )
    assert archive.returncode == 0 and (mini["out"].parent / "a.tar.gz").stat().st_size > 0
