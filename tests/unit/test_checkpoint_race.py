"""S12 F71a — the final-epoch checkpoint race, pinned as a two-rank regression test (S13 P0).

``X2/spatial_only__f1_s0`` trained 150/150 epochs and was never scored: its best
checkpoint was the final epoch's, rank 0 wrote ``best_stage1.pth`` synchronously
and non-atomically, and rank 1 — which skips the write — went straight on to
reload it, read a truncated file, and desynchronised the next collective. The
fix is two parts: ``save_ckpt`` writes atomically (tmp + ``os.replace``), and the
stage ends with a barrier, so no rank reloads before rank 0 has finished.

These tests run :func:`run_single_stage` itself on two gloo ranks with the
training step and the selection score stubbed so the score improves on **every**
epoch (best epoch = last epoch), and with rank 0's write of ``best_stage1.pth``
made slow. Each rank then reloads the checkpoint exactly as the pipeline does
and reports the epoch it got. The sensitivity control removes the barrier and
must observe the race (rank 1 reloads the previous epoch).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

WORLD = 2
EPOCHS = 3
SLOW_WRITE_S = 1.5


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _rank(rank: int, port: str, out: str, with_barrier: bool) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD), CUDA_VISIBLE_DEVICES="",
    )
    from torch.utils.data import DataLoader, TensorDataset

    import spectralquadnet.engine.stages.single_stage as stage
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.engine.checkpoint import load_ckpt
    from spectralquadnet.models.ema import ModelEMA
    from spectralquadnet.models.registry import build_model
    from spectralquadnet.utils.distributed import DistContext, init_distributed, shutdown

    ctx = init_distributed(SimpleNamespace(multi_gpu="ddp", dist_timeout_s=120))
    try:
        cfg = load_experiment_config(
            overrides=[
                "data=ablation/u430k32_grouped",
                f"single.epochs={EPOCHS}",
                "single.patience=50",
                "single.clean_fit_kernels=0",
                "model.stem_channels=16",
                "model.spatial_width_mult=0.25",
                "model.spectral_hidden=32",
                "model.index_bank_size=8",
                "model.continuum_depths=4",
                "model.aux_head_hidden=16",
                f"output_dir={out}",
            ]
        )
        torch.manual_seed(0)
        model = build_model(cfg, torch.linspace(0.0, 1.0, int(cfg.data.num_bands)))
        ema = ModelEMA(model, decay=0.999)

        # Best epoch = last epoch: the selection score rises on every call.
        calls = {"n": 0}

        def rising(*_a: object, **_k: object) -> tuple[float, float]:
            calls["n"] += 1
            return 0.01 * calls["n"], 0.01 * calls["n"]

        stage.evaluate = rising  # type: ignore[assignment]
        stage.train_one_epoch = lambda *a, **k: (0.0, 0.0)  # type: ignore[assignment]
        stage.compute_class_difficulty = lambda *a, **k: ({}, {})  # type: ignore[assignment]
        if not with_barrier:
            DistContext.barrier = lambda self: None  # type: ignore[method-assign]

        # A slow disk under rank 0's write of the best checkpoint.
        real_save = torch.save

        def slow_save(obj: object, f: object, *a: object, **k: object) -> None:
            real_save(obj, f, *a, **k)
            if "best_stage1.pth" in str(f):
                time.sleep(SLOW_WRITE_S)

        torch.save = slow_save  # type: ignore[assignment]

        select = DataLoader(TensorDataset(torch.zeros(4)), batch_size=2)
        best = str(Path(out) / "best_stage1.pth")
        stage.run_single_stage(cfg, model, ema, select, select, ctx.device, best, dist=ctx)
        # What every pipeline does next, on every rank:
        try:
            epoch = int(load_ckpt(best, model, ema, ctx.device)["epoch"])
            error = None
        except Exception as exc:  # a truncated read is the original defect
            epoch, error = -1, f"{type(exc).__name__}: {exc}"
        Path(out, f"rank{rank}.json").write_text(json.dumps({"epoch": epoch, "error": error}))
    finally:
        shutdown(ctx)


def _run(tmp_path: Path, with_barrier: bool) -> dict[int, dict[str, object]]:
    mp.spawn(_rank, args=(_free_port(), str(tmp_path), with_barrier), nprocs=WORLD, join=True)
    return {r: json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(WORLD)}


def test_every_rank_reloads_the_final_epochs_checkpoint(tmp_path: Path) -> None:
    seen = _run(tmp_path, with_barrier=True)
    assert seen == {r: {"epoch": EPOCHS, "error": None} for r in range(WORLD)}, seen
    assert not list(tmp_path.glob("*.tmp*")), "an atomic write left its temporary file behind"


@pytest.mark.slow
def test_without_the_barrier_rank_1_reads_a_stale_checkpoint(tmp_path: Path) -> None:
    """Sensitivity control: the test above would fail on the pre-S13 ordering."""
    seen = _run(tmp_path, with_barrier=False)
    assert seen[0]["epoch"] == EPOCHS
    assert seen[1]["epoch"] != EPOCHS, seen


def test_a_failed_write_leaves_the_previous_checkpoint_intact(tmp_path: Path, monkeypatch) -> None:
    """Atomicity on its own: a write that dies half-way never replaces a good file."""
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.engine import checkpoint
    from spectralquadnet.models.ema import ModelEMA
    from spectralquadnet.models.registry import build_model

    cfg = load_experiment_config(
        overrides=["data=ablation/u430k32_grouped", f"output_dir={tmp_path}",
                   "model.stem_channels=16", "model.spatial_width_mult=0.25"]
    )
    torch.manual_seed(0)
    model = build_model(cfg, torch.linspace(0.0, 1.0, 32))
    ema = ModelEMA(model, decay=0.999)
    path = str(tmp_path / "best_stage1.pth")
    checkpoint.save_ckpt(cfg, path, 1, "Stage 1", model, ema, val_f1=0.1, val_acc=0.1)

    real_save = torch.save

    def dies_half_way(obj: object, f: object, *a: object, **k: object) -> None:
        Path(str(f)).write_bytes(b"\x80\x02truncated")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", dies_half_way)
    with pytest.raises(OSError, match="disk full"):
        checkpoint.save_ckpt(cfg, path, 2, "Stage 1", model, ema, val_f1=0.2, val_acc=0.2)
    monkeypatch.setattr(torch, "save", real_save)

    assert int(checkpoint.load_ckpt(path, model, ema, torch.device("cpu"))["epoch"]) == 1
    assert json.loads((tmp_path / "stage1_meta.json").read_text())["epoch"] == 1
    assert not list(tmp_path.glob("*.tmp*"))
