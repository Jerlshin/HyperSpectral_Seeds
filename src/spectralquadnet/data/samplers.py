"""Custom batch/index samplers, and their two DDP shard wrappers.

Note: ``ClassBalancedBatchSampler.__iter__`` creates an **unseeded**
``np.random.default_rng()`` on every epoch, so its batch composition is a
source of run-to-run non-determinism even when everything else (weights,
data order, other RNGs) is seeded. That stays the default — it is the
behaviour every archived run was produced under — but it is no longer the only
option, because it is unusable under DDP: the ranks must compose the *same*
batch before slicing it between themselves, or the "global batch" whose
gradient DDP's all-reduce claims to compute does not exist. Passing ``seed``
makes the stream a deterministic function of the seed and the epoch.

The two shard wrappers below are the whole of the DDP data story, and both are
deliberately *slicing* rather than *replicating*: the global batch stays the
configured one and each rank sees ``1/world_size`` of it, so
``stage1.batch``/``stage2.bal_n_cls × bal_n_spc`` keep meaning what they say.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sized
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Sampler

_log = logging.getLogger(__name__)

# `Sampler` is generic; the classes below inherit it bare rather than as
# `Sampler[list[int]]`/`Sampler[int]` because subscripting the generic base
# trips `mypy --strict` on this torch version.


class ClassBalancedBatchSampler(Sampler):  # type: ignore[type-arg]
    """Draws n_cls classes per batch, n_spc samples per class, with optional CDWS weighting.

    Args:
        train_labels: Labels of the training split, positionally aligned with
            the dataset's ``indices``.
        n_cls: Classes per batch.
        n_spc: Samples per class per batch.
        class_weights: CDWS weights, normalised into a class-choice probability.
        seed: Base seed for the per-epoch stream. ``None`` — the default —
            keeps the historical unseeded ``default_rng()``. An integer makes
            epoch ``e`` a pure function of ``(seed, e)``, which is what DDP
            needs and what a bit-reproducible run wants.
    """

    def __init__(
        self,
        train_labels: npt.NDArray[Any],
        n_cls: int = 16,
        n_spc: int = 8,
        class_weights: dict[int, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_cls = n_cls
        self.n_spc = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n = len(train_labels) // (n_cls * n_spc)
        self.seed = seed
        self._epoch = 0
        if class_weights is not None:
            raw = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
            self.probs = raw / raw.sum()
        else:
            self.probs = None

    def set_epoch(self, epoch: int) -> None:
        """Advance the seeded stream. A no-op when ``seed is None``."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = (
            np.random.default_rng()
            if self.seed is None
            else np.random.default_rng([self.seed, self._epoch])
        )
        if self.seed is not None:
            self._epoch += 1
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False, p=self.probs)
            batch = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(rng.choice(pool, self.n_spc, replace=len(pool) < self.n_spc).tolist())
            yield batch

    def __len__(self) -> int:
        return self._n


class DistributedBatchShardSampler(Sampler):  # type: ignore[type-arg]
    """Hands each rank its contiguous slice of a batch sampler's batches.

    The inner sampler yields the **global** batch — all ``n_cls × n_spc`` of it
    — on every rank identically (which is why the inner sampler must be
    seeded), and this takes the ``rank``-th of ``world_size`` equal pieces. The
    union of the pieces is the batch, so DDP's gradient mean is the global
    batch's gradient and the class balance the sampler was built to guarantee
    survives the split.

    The alternative — giving each rank a *different* batch — would multiply the
    effective batch size by the world size, which is a hyperparameter change.

    Raises:
        ValueError: The batch does not divide by the world size; see
            :func:`~spectralquadnet.data.loaders._shard_batch` for why an
            uneven split is refused rather than approximated.
    """

    def __init__(self, inner: Sampler, rank: int, world_size: int) -> None:  # type: ignore[type-arg]
        self.inner = inner
        self.rank = int(rank)
        self.world_size = int(world_size)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.inner, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.inner:
            if len(batch) % self.world_size:
                raise ValueError(
                    f"batch of {len(batch)} does not divide by world_size {self.world_size}"
                )
            share = len(batch) // self.world_size
            yield list(batch)[self.rank * share : (self.rank + 1) * share]

    def __len__(self) -> int:
        return len(self.inner)  # type: ignore[arg-type]


class DistributedIndexShardSampler(Sampler):  # type: ignore[type-arg]
    """Deals an index sampler's stream round-robin across ranks.

    For :class:`HardClassOversampledSampler`, whose output is a flat stream of
    weighted draws rather than pre-formed batches. Round-robin rather than
    contiguous blocks so each rank's sub-stream has the same class-weight
    distribution as the whole; a contiguous split of a ``multinomial`` draw has
    that property too, but only in expectation, and round-robin has it per
    batch.

    Each rank drops the tail that does not divide evenly, so every rank
    contributes the same number of steps — a rank that ran one batch fewer
    would leave the others waiting in an all-reduce that never completes.
    """

    def __init__(self, inner: Sampler, rank: int, world_size: int) -> None:  # type: ignore[type-arg]
        self.inner = inner
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        for position, index in enumerate(self.inner):
            if position % self.world_size == self.rank:
                yield index

    def __len__(self) -> int:
        total = len(self.inner) if isinstance(self.inner, Sized) else 0
        return total // self.world_size


class HardClassOversampledSampler(Sampler):  # type: ignore[type-arg]
    """Weighted-with-replacement sampler that oversamples low-F1 classes.

    Per-class weight is ``(1 / (f1 + eps)) ** oversample_power``, clamped to
    ``max_weight`` and renormalised so the mean weight stays 1.0 (no change
    to the overall epoch length's implied sampling rate). Used for Stage 1
    Phase 3, built from the per-class F1 measured at the Phase 2 -> 3
    boundary.
    """

    def __init__(
        self,
        labels: npt.NDArray[Any],
        class_f1: dict[int, float],
        num_samples: int,
        oversample_power: float = 0.75,
        max_weight: float = 5.0,
        hard_f1_thresh: float = 0.50,
        eps: float = 0.05,
    ) -> None:
        self.num_samples = num_samples

        # ── Build per-class weights ────────────────────────────────────
        num_classes = int(np.max(labels)) + 1
        raw_weights: dict[int, float] = {}
        for c in range(num_classes):
            f1 = float(class_f1.get(c, 0.0))
            w = (1.0 / (f1 + eps)) ** oversample_power
            raw_weights[c] = min(w, max_weight)

        # Normalise so the mean weight stays at 1.0 (no overall rate change)
        mean_w = float(np.mean(list(raw_weights.values())))
        norm_weights = {c: w / mean_w for c, w in raw_weights.items()}

        # ── Assign per-sample weights ──────────────────────────────────
        sample_weights = np.array(
            [norm_weights.get(int(lbl), 1.0) for lbl in labels], dtype=np.float32
        )
        self._weights = torch.from_numpy(sample_weights)

        # ── Diagnostics ───────────────────────────────────────────────
        n_hard = sum(1 for f in class_f1.values() if f < hard_f1_thresh)
        hard_classes = sorted(
            [c for c, f in class_f1.items() if f < hard_f1_thresh], key=lambda c: class_f1[c]
        )
        _log.info(
            "[INFO] Phase-3 oversampling: %d/%d hard classes (F1 < %s)  |  power=%.2f  "
            "max_w=%.1f  n_samples=%s",
            n_hard,
            num_classes,
            hard_f1_thresh,
            oversample_power,
            max_weight,
            f"{num_samples:,}",
        )
        if hard_classes:
            worst5 = [(c, class_f1[c]) for c in hard_classes[:5]]
            _log.info("[INFO] Hardest classes (class_id, F1): %s", worst5)

    def __iter__(self) -> Iterator[int]:
        return iter(torch.multinomial(self._weights, self.num_samples, replacement=True).tolist())

    def __len__(self) -> int:
        return self.num_samples
