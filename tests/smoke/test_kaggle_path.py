"""The Kaggle path end to end: pre-sliced cube → train → JSON lines → resume → torchrun.

Each piece has unit tests; these check that they compose in the entrypoint the
Kaggle notebook runs, on the synthetic two-bundles-per-class cube.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from _helpers import REPO_ROOT, TRAIN, tiny_overrides

from spectralquadnet.data.prep.preslice import build_presliced

pytestmark = pytest.mark.slow

KEEP = [0, 2, 4, 6]


@pytest.fixture(scope="module")
def presliced(synthetic_dataset, tmp_path_factory) -> dict[str, str]:
    source = Path(synthetic_dataset["patches_data"]).parent
    out = tmp_path_factory.mktemp("presliced") / "dataset_k4"
    build_presliced(source, out, KEEP, set_name="smoke_k4")
    names = {
        "patches_data": "patches.npy",
        "labels_path": "labels.npy",
        "groups_path": "groups.npy",
        "masks_path": "masks.npy",
        "morphology_path": "morphology.npy",
        "wavelength_path": "wavelengths.csv",
    }
    return {key: str(out / name) for key, name in names.items()}


def _overrides(dataset: dict[str, str], out: Path, **extra) -> list[str]:
    return tiny_overrides(
        dataset,
        out,
        **{
            "data.num_bands": len(KEEP),
            "data.cutmix_bands": 1,
            "data.max_cutout_bands": 1,
            "tracking.backend": "multi",
            "tracking.backends": "[console,jsonl]",
            **extra,
        },
    )


def _run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=2400,
        check=False,
        env=env,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


def test_a_presliced_run_is_reduced_logged_and_resumable(presliced, tmp_path) -> None:
    out = tmp_path / "run"
    first = _run([sys.executable, str(TRAIN), *_overrides(presliced, out)])
    assert first.returncode == 0, first.stdout[-6000:]
    assert "REDUCED arm via a pre-sliced cube" in first.stdout
    assert "4 of 8 bands" in first.stdout

    run = json.loads((out / "results" / "run.json").read_text())["run"]
    assert run["band_selection"] is True, "a pre-sliced cube must not pass for the full cube"
    assert run["band_geometry"]["acquired"] == 8

    records = _records(out)
    assert json.loads((out / "last_stage1.json").read_text())["finished"] is True
    assert [r["step"] for r in records if r["event"] == "epoch"] == [1, 2]
    assert any(r["event"] == "hyperparams" for r in records)
    assert any(
        r["event"] == "table" and r["tag"].startswith("hardest_classes/") for r in records
    ), "the final evaluation reaches the JSON record"

    # A session that died after epoch 2 of a longer schedule.
    meta_path = out / "last_stage1.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "finished": False}))

    second = _run([sys.executable, str(TRAIN), *_overrides(presliced, out, **{"single.epochs": 3})])
    assert second.returncode == 0, second.stdout[-6000:]
    assert "[SKIP]" not in second.stdout, "an interrupted stage is not a finished one"
    assert "[RESUME]" in second.stdout and "continuing at epoch 3" in second.stdout
    records = _records(out)
    assert [r["step"] for r in records if r["event"] == "epoch"] == [1, 2, 3]
    assert [r["event"] for r in records].count("run_start") == 2
    final = json.loads(meta_path.read_text())
    assert final["finished"] is True and final["epoch"] == 3


def test_the_run_trains_under_torchrun_on_two_ranks(presliced, tmp_path) -> None:
    """Two gloo ranks on CPU: the DDP plumbing the T4 x2 profile relies on, minus CUDA.

    An explicit loopback rendezvous rather than ``--standalone``, which on macOS
    resolves the host name to an address the gloo client cannot reach. Kaggle
    (Linux) runs ``--standalone`` as documented.
    """
    out = tmp_path / "ddp"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={_free_port()}",
        str(TRAIN),
        *_overrides(presliced, out, **{"runtime.multi_gpu": "ddp"}),
    ]
    env = dict(os.environ)
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    result = _run(command, env=env)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
    assert "world_size=2" in result.stdout
    assert (out / "results" / "run.json").exists()
    assert (out / "last_stage1.pth").exists()
    assert json.loads((out / "last_stage1.json").read_text())["finished"] is True
    starts = [r for r in _records(out) if r["event"] == "run_start"]
    assert len(starts) == 1, "only rank 0 writes the record"
