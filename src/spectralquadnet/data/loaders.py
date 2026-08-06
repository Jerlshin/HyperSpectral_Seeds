"""Train/val/test splits and DataLoader construction.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=================================  ==============
Symbol                             Baseline lines
=================================  ==============
:func:`build_splits`               1674-1679
:func:`build_loaders`              1682-1708
:func:`build_phase3_loader`        1711-1743
=================================  ==============

Declared deviations, all mechanical:

* ``CONFIG[...]`` → ``cfg.<group>.<field>``.
* ``_GLOBAL_LABELS`` → ``store.require_labels()``.
* ``RiceSeedDataset(...)`` now receives ``store``/``data_cfg``/``device``
  explicitly instead of reading module globals.

The ``random_state=42`` values in :func:`build_splits` are **hardcoded in the
baseline** and are deliberately *not* promoted to ``cfg.seed``: the three trained
checkpoints were validated against the split those literals produce, so changing
them — even to an identical value — would risk silently re-partitioning the data
if ``cfg.seed`` were ever overridden. See REFACTOR_PLAN.md §6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(cfg.data.labels_path)
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(
    cfg: ExperimentConfig | Any,
    store: DataStore,
    device: torch.device | str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_train: int,
    balanced: bool = False,
    all_labels: np.ndarray | None = None,
    train_aug: str = "none",
    class_weights: dict[int, float] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kw = dict(store=store, data_cfg=cfg.data, device=device)

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
    train_ds: Dataset,
    class_f1: dict[int, float],
) -> DataLoader:
    """
    Build the Phase-3 DataLoader with hard-class oversampling.

    Uses HardClassOversampledSampler to give hard classes (low F1 from
    Phase 2) higher sampling probability.  Falls back to standard
    shuffled loader if oversampling is disabled in CONFIG.
    """
    if not cfg.stage1.p3_oversample or not class_f1:
        return DataLoader(
            train_ds, batch_size=cfg.stage1.batch, shuffle=True, drop_last=True, num_workers=0
        )

    labels = store.require_labels()
    train_labels = np.array(
        [int(labels[train_ds.indices[i]]) for i in range(len(train_ds.indices))]
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
