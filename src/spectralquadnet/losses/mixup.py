"""Batch-level mixup augmentation and its loss interpolation.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`_mixup`                         468-473
:func:`mixed_aug`                      475-478
:func:`mixed_loss`                     480-487
=====================================  ==============

All three are pure functions of their arguments. :func:`_mixup` draws from two
RNG streams — ``np.random.beta`` (global NumPy) for the interpolation weight and
``torch.randperm`` (global torch) for the pairing — exactly as the baseline did;
neither is reseeded or given an explicit generator here, so Stage 1's batch
composition keeps the reproducibility characteristics documented in
REFACTOR_PLAN.md §1.3.

Used by Stage 1 Phases 1-2 only: Phase 3 and Stage 2 disable mixup (ArcFace and
mixup are mutually exclusive — see ``engine/train_epoch.py``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixed_aug(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    return _mixup(x, y, alpha)


def mixed_loss(
    crit: nn.Module,
    logits: torch.Tensor,
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    # `nn.Module.__call__` is untyped upstream, so the sum is `Any` to mypy.
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)  # type: ignore[no-any-return]
