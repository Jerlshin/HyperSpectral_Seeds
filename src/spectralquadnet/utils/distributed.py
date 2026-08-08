"""Multi-GPU execution: DDP with SyncBatchNorm, and why it is not DataParallel.

Launch
──────
::

    torchrun --standalone --nproc_per_node=2 train.py     # a T4 x2 box
    python train.py                                       # unchanged, one device

Nothing here activates unless ``torchrun`` (or an equivalent launcher) has set
``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` **and** ``WORLD_SIZE > 1``. A plain
``python train.py`` builds a :class:`DistContext` with ``enabled=False`` whose
every collective is a no-op, so the single-device path executes exactly the code
it executed before this module existed.

Why DDP and not DataParallel
────────────────────────────
The invariant this refactor is held to is that a two-GPU run reports the same
number as a one-GPU run. Both replication schemes split the batch, and the
model contains five ``BatchNorm1d`` layers in the fusion plus one in each of
Branches C and D — so a split batch means each replica normalises by *its
shard's* mean and variance rather than the batch's. That is a different
function, and no amount of gradient averaging repairs it, because the
normalisation happens inside the forward.

``torch.nn.SyncBatchNorm`` is the repair: it all-reduces the per-shard sums so
every replica normalises by the **global** batch statistics, which is by
construction the single-device quantity. It is available for DDP and not for
DataParallel, which is the whole reason for the choice; DataParallel's
single-process scatter/gather and its Python-level replication per step are
merely the secondary reasons. :func:`wrap_for_training` therefore converts
before it wraps, and ``runtime.sync_batchnorm=false`` is the documented,
non-invariant opt-out.

``convert_sync_batchnorm`` preserves parameter and buffer names exactly
(``weight``, ``bias``, ``running_mean``, ``running_var``,
``num_batches_tracked``), so a checkpoint written by a DDP run loads into a
single-device model and the reverse, and ``SCHEMA_VERSION`` does not move.

What each rank does
───────────────────
* **Gradients** — DDP all-reduces them with a *mean*. Each rank's loss is
  already a mean over its shard, so with equal shard sizes (guaranteed by
  ``drop_last=True`` on the training loaders and by
  :class:`DistributedBatchSampler` for the balanced ones) the reduced gradient
  is the global-batch gradient.
* **Evaluation** — sharded across ranks and gathered by
  :func:`gather_concat`, so macro-F1 is computed once, on rank 0, over the
  whole split. Never on a shard: a per-rank macro-F1 over 45 of 90 classes is
  not the metric, and averaging two of them is not it either.
* **Checkpoints and console output** — rank 0 only. Every rank holds identical
  weights, so a second writer would only race the first.
* **Decisions** — anything a rank computes and every rank must agree on (Stage
  3's greedy accept/reject, early stopping) goes through
  :func:`broadcast_object`. A rank that decided differently would desynchronise
  the collectives on the next step and hang the job.
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn

_log = logging.getLogger(__name__)

T = TypeVar("T")

#: Environment variables a launcher sets. Their joint presence is the signal;
#: this module never guesses a topology from ``torch.cuda.device_count()``,
#: because two independent single-GPU runs on one box must stay independent.
_LAUNCHER_VARS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


@dataclass(frozen=True)
class DistContext:
    """This rank's place in the job, or the degenerate single-process case.

    ``enabled=False`` is not an error path — it is what ``python train.py``
    produces, and every method below is written so that the single-process
    branch is the identity.
    """

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        """Whether this rank owns the console, the checkpoints and the metrics."""
        return self.rank == 0

    def barrier(self) -> None:
        """Wait for every rank. A no-op single-process."""
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def all_reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        """Replace ``value`` with its mean across ranks, in place, and return it."""
        if not (self.enabled and dist.is_initialized()):
            return value
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value.div_(self.world_size)

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Replace ``value`` with its sum across ranks, in place, and return it."""
        if not (self.enabled and dist.is_initialized()):
            return value
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def broadcast_object(self, obj: T, src: int = 0) -> T:
        """Hand rank ``src``'s ``obj`` to every rank.

        The one way a metric computed on rank 0 — a validation F1, an SWA
        accept — becomes a decision every rank makes identically. Ranks that
        disagreed here would enter different numbers of collectives on the next
        epoch and deadlock.
        """
        if not (self.enabled and dist.is_initialized()):
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=src)
        return box[0]


def _launched_by_torchrun() -> bool:
    return all(v in os.environ for v in _LAUNCHER_VARS)


def init_distributed(runtime_cfg: Any = None, fallback: torch.device | None = None) -> DistContext:
    """Join the process group when a launcher started us; otherwise stay single-process.

    Args:
        runtime_cfg: The ``runtime`` config group. ``multi_gpu="off"`` refuses
            to join even under ``torchrun``; ``"ddp"`` demands the launcher and
            raises when it is absent, so a mis-typed launch fails loudly
            instead of silently training on one GPU for eight hours.
        fallback: Device to record when distribution is off. Defaults to
            whatever ``resolve_device("auto")`` would pick.

    Raises:
        RuntimeError: ``multi_gpu="ddp"`` without a launcher, or with a CUDA
            device count below the local world size.
    """
    from spectralquadnet.utils.device import resolve_device

    mode = str(getattr(runtime_cfg, "multi_gpu", "auto")).lower()
    launched = _launched_by_torchrun() and int(os.environ.get("WORLD_SIZE", "1")) > 1

    if mode == "off" or (mode == "auto" and not launched):
        if mode == "off" and launched:
            _log.warning("runtime.multi_gpu=off under torchrun — this rank trains alone")
        return DistContext(device=fallback or resolve_device("auto"))

    if not launched:
        raise RuntimeError(
            "runtime.multi_gpu='ddp' needs a launcher that sets RANK/WORLD_SIZE/LOCAL_RANK. "
            "Launch with `torchrun --standalone --nproc_per_node=<gpus> train.py`, or set "
            "runtime.multi_gpu=auto to fall back to single-device training."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        if torch.cuda.device_count() <= local_rank:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are "
                "visible. Check --nproc_per_node against CUDA_VISIBLE_DEVICES."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        # Distribution without CUDA is only useful for testing the plumbing;
        # gloo makes that possible without pretending it is fast.
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=int(getattr(runtime_cfg, "dist_timeout_s", 1800))),
        )

    return DistContext(
        enabled=True, rank=rank, local_rank=local_rank, world_size=world_size, device=device
    )


def shutdown(ctx: DistContext) -> None:
    """Leave the process group. Idempotent, and safe to call from a ``finally``."""
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def wrap_for_training(model: nn.Module, ctx: DistContext, sync_batchnorm: bool = True) -> nn.Module:
    """Convert BatchNorm, then wrap in DDP. Returns ``model`` unchanged single-process.

    The conversion comes **first** and is the load-bearing half — see the module
    docstring. ``find_unused_parameters`` is deliberately left at ``False``: the
    auxiliary heads and every branch receive gradient on every training step
    (branch dropout scales a branch's contribution by zero but still routes
    gradient through its auxiliary head), so the extra graph traversal would
    cost a few percent and find nothing.

    ``SyncBatchNorm`` is **CUDA-only** — DDP raises ``"SyncBatchNorm layers only
    work with GPU modules"`` for a CPU module — so a gloo/CPU group (which
    exists to exercise this plumbing, not to train) skips the conversion and
    says so. That run is then *not* BN-invariant against a single process,
    which is exactly why it is not a training configuration.

    Note that ``convert_sync_batchnorm`` rebuilds the BatchNorm *leaves* but
    returns the same root object for a non-BatchNorm root, so ``model`` is
    mutated in place and every existing reference to it stays valid. The EMA
    shadow is deep-copied before this runs and keeps plain BatchNorm, which is
    correct: it is only ever evaluated, and in ``eval()`` the two are the same
    function.
    """
    if not ctx.enabled:
        return model
    if sync_batchnorm:
        if ctx.device.type == "cuda":
            # `convert_sync_batchnorm` is untyped in torch's stubs.
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # type: ignore[no-untyped-call]
        else:
            _log.warning(
                "SyncBatchNorm needs CUDA modules; running DDP on %s without it. Per-rank "
                "BatchNorm statistics make this NOT numerically equal to a single-device run.",
                ctx.device.type,
            )
    device_ids = [ctx.local_rank] if ctx.device.type == "cuda" else None
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
        broadcast_buffers=True,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
    )


def gather_concat(ctx: DistContext, local: torch.Tensor) -> torch.Tensor:
    """Concatenate a per-rank 1-D tensor across every rank, in rank order.

    Shard sizes differ whenever the split does not divide by the world size, so
    the length is exchanged first and every rank pads to the maximum before the
    fixed-size collective — then each contribution is trimmed back to its own
    length. Padding that survived into the result would show up as extra
    predictions for class 0.
    """
    if not (ctx.enabled and dist.is_initialized()):
        return local

    local = local.contiguous()
    count = torch.tensor([local.numel()], device=local.device, dtype=torch.long)
    counts = [torch.zeros_like(count) for _ in range(ctx.world_size)]
    dist.all_gather(counts, count)
    sizes = [int(c.item()) for c in counts]

    width = max(sizes)
    padded = torch.zeros(width, dtype=local.dtype, device=local.device)
    padded[: local.numel()] = local
    buckets = [torch.zeros_like(padded) for _ in range(ctx.world_size)]
    dist.all_gather(buckets, padded)
    return torch.cat([bucket[:n] for bucket, n in zip(buckets, sizes, strict=True)])
