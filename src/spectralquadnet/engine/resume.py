"""Mid-stage resume: the full training state every N epochs, written off the training thread.

The gap this closes
───────────────────
``best_stage1.pth`` + ``stage1_meta.json`` is written the first time the
selection F1 improves — epoch 1, typically — and the pair is also what marks a
stage *complete* (:func:`~spectralquadnet.engine.checkpoint.stage_exists`). So a
single-stage run killed at epoch 90 of 150 (a Kaggle session hitting its time
limit, a pre-empted node) was, on restart, "complete": the pipeline skipped
straight to the final evaluation and reported a model that had trained for 60%
of its schedule, with nothing in the results to say so.

Two files now carry what the pair cannot:

``last_stage{s}.pth``
    Weights, EMA shadow and its update count, optimiser, LR scheduler, loss
    scaler, the early-stopping counters and the RNG states — everything the
    epoch loop needs to continue, not just to evaluate.
``last_stage{s}.json``
    ``{"epoch", "finished", "best_f1", "no_improve", …}``. Written at the
    **start** of training with ``finished: false`` and rewritten after every
    ``.pth``, so its presence with ``finished: false`` means "interrupted", even
    when the interruption came before the first ``.pth``.

A stage is complete when its best-checkpoint pair exists **and** it has no
unfinished ``last_stage{s}.json``. Runs written before this module existed have
no such file and keep their meaning.

I/O
───
The snapshot is taken on the training thread — a device→host copy of ~4× the
parameter count, milliseconds at this model's size — and the serialisation and
disk write happen on a background thread, so the next epoch starts at once.
Each file is written to a temporary name and ``os.replace``d into place, the
``.json`` after the ``.pth``, so neither a crash nor a full disk can leave a
sidecar describing a half-written bundle. At most one write is in flight: the
next ``submit`` (and the stage's end) waits for the previous one, and re-raises
its error rather than dropping it. Only rank 0 writes.

Resume is not bit-exact — ``cudnn.benchmark``, the unseeded worker streams and
the DataLoader's generator restart — which is the same reproducibility
statement the rest of the pipeline makes (README §6).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from spectralquadnet.engine.checkpoint import ArchitectureMismatchError
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import describe
from spectralquadnet.utils.device import unwrap_model

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.amp import GradScaler

    from spectralquadnet.config.schema import ExperimentConfig

_log = logging.getLogger(__name__)


def last_ckpt_path(cfg: ExperimentConfig | Any, s: int) -> str:
    """Stage ``s``'s full training-state bundle under ``cfg.output_dir``."""
    return os.path.join(cfg.output_dir, f"last_stage{s}.pth")


def last_meta_path(cfg: ExperimentConfig | Any, s: int) -> str:
    """Stage ``s``'s training-state sidecar under ``cfg.output_dir``."""
    return os.path.join(cfg.output_dir, f"last_stage{s}.json")


def read_last_meta(cfg: ExperimentConfig | Any, s: int) -> dict[str, Any] | None:
    """The training-state sidecar, or ``None`` when this stage never wrote one."""
    path = last_meta_path(cfg, s)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return dict(json.load(handle))


def stage_interrupted(cfg: ExperimentConfig | Any, s: int) -> bool:
    """Whether stage ``s`` started training and did not finish it."""
    meta = read_last_meta(cfg, s)
    return meta is not None and not bool(meta.get("finished", False))


def resumable_state_path(cfg: ExperimentConfig | Any, s: int) -> str | None:
    """The bundle to resume stage ``s`` from, or ``None`` to start it fresh."""
    path = last_ckpt_path(cfg, s)
    return path if stage_interrupted(cfg, s) and os.path.isfile(path) else None


def mark_stage_started(cfg: ExperimentConfig | Any, s: int, is_main: bool = True) -> None:
    """Record that stage ``s`` is training, before anything else is written for it.

    Leaves an existing unfinished record alone: a resumed run is still the run
    that started, and its epoch counter is the one to keep.
    """
    if not is_main or stage_interrupted(cfg, s):
        return
    _write_json(last_meta_path(cfg, s), {"stage": s, "epoch": 0, "finished": False})


def _to_cpu(obj: Any) -> Any:
    """A host-side copy of every tensor in a nested state, detached from training."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)


def snapshot_training_state(
    *,
    epoch: int,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler | None,
    **counters: Any,
) -> dict[str, Any]:
    """Everything needed to continue after ``epoch``, copied to host memory."""
    core = unwrap_model(model)
    identity = describe(core)
    return {
        "epoch": int(epoch),
        "arch": identity.arch,
        "schema_version": identity.schema_version,
        "model": _to_cpu(core.state_dict()),
        "ema": _to_cpu(ema.state_dict()),
        # Private, but it is ModelEMA's whole warm-up state: restoring the
        # weights without it would restart the (1+n)/(10+n) decay ramp.
        "ema_updates": int(ema._num_updates),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_state(),
        **counters,
    }


def restore_training_state(
    path: str,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler | None,
) -> dict[str, Any]:
    """Load a :func:`snapshot_training_state` bundle into live training objects.

    Returns:
        The bundle, for its ``epoch`` and counters.

    Raises:
        ArchitectureMismatchError: The bundle was written by another model.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    core = unwrap_model(model)
    identity = describe(core)
    if state.get("arch") != identity.arch:
        raise ArchitectureMismatchError(
            f"{path} was written by {state.get('arch')!r} but this run builds "
            f"{identity.arch!r}; delete it or point output_dir elsewhere."
        )
    core.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    ema._num_updates = int(state.get("ema_updates", 0))
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if state.get("rng") is not None:
        _set_rng_state(state["rng"])
    return dict(state)


class AsyncCheckpointWriter:
    """Write training-state bundles on a background thread, one at a time.

    Args:
        enabled: ``False`` on every rank but 0 — every call is then a no-op.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.last_seconds = 0.0

    def submit(
        self, bundle: dict[str, Any], path: str, meta: dict[str, Any], meta_path: str
    ) -> None:
        """Queue ``bundle`` → ``path`` then ``meta`` → ``meta_path``; returns immediately."""
        if not self.enabled:
            return
        self.wait()
        self._thread = threading.Thread(
            target=self._write, args=(bundle, path, meta, meta_path), daemon=True
        )
        self._thread.start()

    def wait(self) -> None:
        """Block until the in-flight write is on disk; re-raise its error if it failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError(f"background checkpoint write failed: {error}") from error

    def _write(
        self, bundle: dict[str, Any], path: str, meta: dict[str, Any], meta_path: str
    ) -> None:
        started = time.perf_counter()
        try:
            tmp = f"{path}.tmp"
            torch.save(bundle, tmp)
            os.replace(tmp, path)
            meta = {**meta, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            _write_json(meta_path, meta)
        except BaseException as exc:  # surfaced by `wait`, never swallowed
            self._error = exc
        self.last_seconds = time.perf_counter() - started


def _write_json(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)
