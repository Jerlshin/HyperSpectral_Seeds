"""Batch-level mixup augmentation and its loss interpolation.

All three functions are pure functions of their arguments. :func:`_mixup` draws
from two global RNG streams — ``np.random.beta`` for the interpolation weight
and ``torch.randperm`` for the pairing — neither reseeded nor given an
explicit generator here, so callers control reproducibility by seeding the
global NumPy/torch RNGs before invocation.

Used by Stage 1 Phases 1-2 only: Phase 3 and Stage 2 disable mixup, since
ArcFace and mixup are mutually exclusive (soft-interpolated labels are not
compatible with the angular-margin loss) — see ``engine/train_epoch.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Linearly interpolate a batch with a random permutation of itself.

    Args:
        x: Input batch, shape ``(B, ...)``.
        y: Labels for ``x``, shape ``(B,)``.
        alpha: Concentration parameter of the ``Beta(alpha, alpha)`` mixing
            coefficient; smaller values bias ``lam`` towards 0 or 1 (mild
            mixing), larger values bias it towards 0.5 (heavy mixing).

    Returns:
        ``(mixed_x, y, y_permuted, lam)`` — the mixed input, the two label
        sets to interpolate the loss between, and the mixing coefficient.
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixed_aug(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Public entry point for :func:`_mixup` with the Stage-1 default alpha."""
    return _mixup(x, y, alpha)


def mixed_loss(
    crit: nn.Module,
    logits: torch.Tensor,
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Interpolate the criterion between the two mixup label sets by ``lam``.

    Args:
        crit: Loss module called as ``crit(logits, labels)``.
        logits: Model output for the mixed input, shape ``(B, num_classes)``.
        ya: First (unpermuted) label set from :func:`_mixup`.
        yb: Second (permuted) label set from :func:`_mixup`.
        lam: Mixing coefficient from :func:`_mixup`; weights ``ya``'s loss
            term by ``lam`` and ``yb``'s by ``1 - lam``.

    Returns:
        Scalar interpolated loss.
    """
    # `nn.Module.__call__` is untyped upstream, so the sum is `Any` to mypy.
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)  # type: ignore[no-any-return]
