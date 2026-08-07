"""Train/val/test splits and DataLoader construction.

The ``random_state=42`` values in :func:`build_splits` are deliberately
hardcoded rather than sourced from ``cfg.seed``: trained checkpoints are
validated against the exact split those literals produce, so tying the split
to the run seed would risk silently re-partitioning the data whenever
``cfg.seed`` is overridden for an unrelated reason (e.g. varying model
initialisation across runs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.samplers import ClassBalancedBatchSampler, HardClassOversampledSampler

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.data.mmap_store import DataStore


def build_splits(
    cfg: ExperimentConfig | Any,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Stratified 70/15/15 train/val/test split at a fixed random state.

    Returns:
        ``(labels, train_idx, val_idx, test_idx)`` — the full label array and
        three disjoint index arrays into it.
    """
    labels = np.load(cfg.data.labels_path)
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(
    cfg: ExperimentConfig | Any,
    store: DataStore,
    device: torch.device | str,
    train_idx: npt.NDArray[Any],
    val_idx: npt.NDArray[Any],
    test_idx: npt.NDArray[Any],
    batch_train: int,
    balanced: bool = False,
    all_labels: npt.NDArray[Any] | None = None,
    train_aug: str = "none",
    class_weights: dict[int, float] | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Build train/val/test loaders sharing one ``DataStore``/config/device.

    The train loader uses a :class:`~spectralquadnet.data.samplers.ClassBalancedBatchSampler`
    when ``balanced=True``, otherwise plain shuffled batching; val/test are
    always shuffled=False at a fixed batch size of 256.

    Args:
        cfg: Composed experiment config.
        store: Memory-mapped patch store shared by all three datasets.
        device: Device patches are moved to on ``__getitem__``.
        train_idx: Indices for the training split.
        val_idx: Indices for the validation split.
        test_idx: Indices for the test split.
        batch_train: Batch size for the (non-balanced) train loader.
        balanced: Whether to use class-balanced batch sampling for training.
        all_labels: Full label array, required when ``balanced=True``.
        train_aug: Augmentation profile name for the training dataset.
        class_weights: Optional per-class weights for the balanced sampler.

    Returns:
        ``(train_loader, val_loader, test_loader)``.
    """
    # Annotated rather than inferred: the inferred value type is the union of the
    # three entries, which cannot be checked against `RiceSeedDataset`'s distinct
    # keyword types when unpacked with `**`.
    kw: dict[str, Any] = dict(store=store, data_cfg=cfg.data, device=device)

    ds = RiceSeedDataset(train_idx, aug_strength=train_aug, **kw)

    if balanced and all_labels is not None:
        samp = ClassBalancedBatchSampler(
            all_labels[train_idx],
            cfg.stage2.bal_n_cls,
            cfg.stage2.bal_n_spc,
            class_weights=class_weights,
        )
        tr_ldr = DataLoader(ds, batch_sampler=samp, num_workers=0)
    else:
        tr_ldr = DataLoader(ds, batch_size=batch_train, shuffle=True, drop_last=True, num_workers=0)

    va_ldr = DataLoader(
        RiceSeedDataset(val_idx, **kw), batch_size=256, shuffle=False, num_workers=0
    )
    te_ldr = DataLoader(
        RiceSeedDataset(test_idx, **kw), batch_size=256, shuffle=False, num_workers=0
    )
    return tr_ldr, va_ldr, te_ldr


def build_phase3_loader(
    cfg: ExperimentConfig | Any,
    store: DataStore,
    train_ds: Dataset[Any],
    class_f1: dict[int, float],
) -> DataLoader[Any]:
    """Build the Phase-3 DataLoader with hard-class oversampling.

    Uses :class:`~spectralquadnet.data.samplers.HardClassOversampledSampler`
    to give hard classes (low F1 from Phase 2) higher sampling probability.
    Falls back to a standard shuffled loader if oversampling is disabled in
    config, or if no per-class F1 was supplied.

    Args:
        cfg: Composed experiment config.
        store: Memory-mapped patch store, used to read training labels.
        train_ds: Phase-3 training dataset; must expose ``.indices`` (as
            :class:`~spectralquadnet.data.datasets.RiceSeedDataset` does).
        class_f1: Per-class F1 from the Phase 2 -> 3 boundary evaluation,
            driving the oversampling weights.

    Returns:
        DataLoader over ``train_ds``, oversampled or plain-shuffled.
    """
    if not cfg.stage1.p3_oversample or not class_f1:
        return DataLoader(
            train_ds, batch_size=cfg.stage1.batch, shuffle=True, drop_last=True, num_workers=0
        )

    labels = store.require_labels()
    # `.indices` is `RiceSeedDataset`'s, not the generic `Dataset` protocol's;
    # the parameter type stays the duck-typed `Dataset` so `DataLoader.dataset`
    # can be handed straight in without a downcast.
    train_labels = np.array(
        [int(labels[train_ds.indices[i]]) for i in range(len(train_ds.indices))]  # type: ignore[attr-defined]
    )
    sampler = HardClassOversampledSampler(
        labels=train_labels,
        class_f1=class_f1,
        num_samples=len(train_labels),
        oversample_power=cfg.stage1.p3_oversample_power,
        max_weight=cfg.stage1.p3_oversample_max_w,
        hard_f1_thresh=cfg.stage1.p3_hard_f1_thresh,
        eps=cfg.stage1.p3_oversample_eps,
    )
    return DataLoader(
        train_ds, batch_size=cfg.stage1.batch, sampler=sampler, drop_last=True, num_workers=0
    )
