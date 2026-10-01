"""Mid-stage resume, the background checkpoint writer, the DDP sampler epoch and the T4 profile."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectralquadnet.engine.checkpoint import ArchitectureMismatchError
from spectralquadnet.engine.resume import (
    AsyncCheckpointWriter,
    last_ckpt_path,
    last_meta_path,
    mark_stage_started,
    restore_training_state,
    resumable_state_path,
    snapshot_training_state,
    stage_interrupted,
)
from spectralquadnet.models.ema import ModelEMA


class _Tiny(nn.Module):
    ARCH = "tiny_test_net"
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.lin(x))


def _training_objects(seed: int):
    torch.manual_seed(seed)
    model = _Tiny()
    ema = ModelEMA(model, decay=0.9)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: 1.0 / (1 + ep))
    return model, ema, opt, sched


def _step(model, ema, opt, sched) -> None:
    loss = model(torch.randn(8, 4)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    ema.update(model)


def test_a_restored_state_continues_exactly_where_the_snapshot_left_off(tmp_path) -> None:
    model, ema, opt, sched = _training_objects(0)
    for _ in range(3):
        _step(model, ema, opt, sched)
    bundle = snapshot_training_state(
        epoch=3, model=model, ema=ema, optimizer=opt, scheduler=sched, scaler=None, best_f1=0.5
    )
    path = tmp_path / "last_stage1.pth"
    torch.save(bundle, path)

    # The snapshot is a copy: training on does not reach into it.
    _step(model, ema, opt, sched)
    assert not torch.equal(bundle["model"]["lin.weight"], model.lin.weight.detach())

    fresh, fresh_ema, fresh_opt, fresh_sched = _training_objects(1)
    state = restore_training_state(
        str(path),
        model=fresh,
        ema=fresh_ema,
        optimizer=fresh_opt,
        scheduler=fresh_sched,
        scaler=None,
    )
    assert state["epoch"] == 3 and state["best_f1"] == 0.5
    for key, value in bundle["model"].items():
        assert torch.equal(fresh.state_dict()[key], value), key
    assert fresh_ema._num_updates == 3, "the EMA warm-up ramp resumes, not restarts"
    assert fresh_sched.last_epoch == 3
    assert fresh_opt.state_dict()["state"][0]["step"] == 3


def test_a_bundle_from_another_architecture_is_refused(tmp_path) -> None:
    model, ema, opt, sched = _training_objects(0)
    bundle = snapshot_training_state(
        epoch=1, model=model, ema=ema, optimizer=opt, scheduler=sched, scaler=None
    )
    bundle["arch"] = "spectral_seed_net"
    torch.save(bundle, tmp_path / "x.pth")
    with pytest.raises(ArchitectureMismatchError):
        restore_training_state(
            str(tmp_path / "x.pth"),
            model=model,
            ema=ema,
            optimizer=opt,
            scheduler=sched,
            scaler=None,
        )


def test_the_interruption_record_decides_completion(tmp_path) -> None:
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    assert not stage_interrupted(cfg, 1), "a run from before this module keeps its meaning"

    mark_stage_started(cfg, 1)
    assert stage_interrupted(cfg, 1)
    assert resumable_state_path(cfg, 1) is None, "interrupted before any state: start fresh"

    torch.save({"epoch": 4}, last_ckpt_path(cfg, 1))
    (tmp_path / "last_stage1.json").write_text(json.dumps({"epoch": 4, "finished": False}))
    assert resumable_state_path(cfg, 1) == last_ckpt_path(cfg, 1)

    mark_stage_started(cfg, 1)
    assert json.loads((tmp_path / "last_stage1.json").read_text())["epoch"] == 4, (
        "restarting an interrupted stage must not reset its record"
    )

    (tmp_path / "last_stage1.json").write_text(json.dumps({"epoch": 9, "finished": True}))
    assert not stage_interrupted(cfg, 1)
    assert resumable_state_path(cfg, 1) is None

    other = SimpleNamespace(output_dir=str(tmp_path / "rank1"))
    os.makedirs(other.output_dir)
    mark_stage_started(other, 1, is_main=False)
    assert not os.path.exists(last_meta_path(other, 1)), "only rank 0 writes"


def test_the_writer_puts_the_bundle_down_before_its_sidecar(tmp_path) -> None:
    writer = AsyncCheckpointWriter()
    path, meta = str(tmp_path / "s.pth"), str(tmp_path / "s.json")
    writer.submit({"w": torch.ones(3)}, path, {"epoch": 2, "finished": False}, meta)
    writer.wait()
    assert torch.equal(torch.load(path)["w"], torch.ones(3))
    record = json.loads(Path(meta).read_text())
    assert record["epoch"] == 2 and "saved_at" in record
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_a_failed_background_write_is_raised_not_dropped(tmp_path) -> None:
    writer = AsyncCheckpointWriter()
    missing = tmp_path / "no" / "such" / "dir"
    writer.submit({"w": torch.ones(1)}, str(missing / "s.pth"), {}, str(missing / "s.json"))
    with pytest.raises(RuntimeError, match="background checkpoint write failed"):
        writer.wait()
    writer.wait()  # the error is reported once


def test_a_disabled_writer_writes_nothing(tmp_path) -> None:
    writer = AsyncCheckpointWriter(enabled=False)
    writer.submit({}, str(tmp_path / "s.pth"), {}, str(tmp_path / "s.json"))
    writer.wait()
    assert list(tmp_path.iterdir()) == []


def test_a_ddp_sampler_draws_a_new_order_every_epoch() -> None:
    """Without `set_epoch` every epoch of a torchrun job replays epoch 0's order."""
    from torch.utils.data import DataLoader, DistributedSampler

    from spectralquadnet.engine.train_epoch import _advance_sampler_epoch

    data = list(range(32))
    sampler = DistributedSampler(data, num_replicas=2, rank=0, shuffle=True, seed=7)
    loader = DataLoader(data, batch_size=4, sampler=sampler)
    orders = []
    for epoch in (1, 2, 3):
        _advance_sampler_epoch(loader, epoch)
        orders.append([int(x) for batch in loader for x in batch])
    assert orders[0] != orders[1] != orders[2]
    _advance_sampler_epoch(loader, 1)
    assert [int(x) for batch in loader for x in batch] == orders[0], "seeded by epoch"
    _advance_sampler_epoch([[1, 2]], 5)  # anything without a sampler is left alone


def test_the_kaggle_profile_composes_on_both_live_experiments() -> None:
    from spectralquadnet.config.compose import QUADNET_FULL256_EXPERIMENT, load_experiment_config

    base = load_experiment_config()
    assert base.runtime.amp_dtype == "bf16", "the default experiment's inline block stands"
    for name in (None, QUADNET_FULL256_EXPERIMENT):
        args = (name,) if name else ()
        cfg = load_experiment_config(*args, overrides=["runtime=kaggle_t4x2"])
        assert cfg.runtime.amp_dtype == "fp16", "T4 has no bf16 Tensor Core path"
        assert cfg.runtime.multi_gpu == "ddp"
        assert cfg.runtime.sync_batchnorm is True
        assert cfg.runtime.num_workers == 4
        assert cfg.runtime.checkpoint_every == 1
        assert cfg.runtime.allow_tf32 is False
        # Only `runtime` moves: the experiment is otherwise the same one.
        assert cfg.data == base.data if name is None else True
        assert cfg.single == base.single
