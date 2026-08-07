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

from typing import Literal, overload

import numpy as np
import torch
import torch.nn as nn

#: Without ``return_perm``, exactly what the pre-Tier-3 signature returned.
MixupOut = tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
#: With it, the permutation as well — so the side inputs can be paired the same way.
MixupOutPerm = tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]


@overload
def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float, return_perm: Literal[False] = ...
) -> MixupOut: ...


@overload
def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float, return_perm: Literal[True]
) -> MixupOutPerm: ...


def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float, return_perm: bool = False
) -> MixupOut | MixupOutPerm:
    """Linearly interpolate a batch with a random permutation of itself.

    Args:
        x: Input batch, shape ``(B, ...)``.
        y: Labels for ``x``, shape ``(B,)``.
        alpha: Concentration parameter of the ``Beta(alpha, alpha)`` mixing
            coefficient; smaller values bias ``lam`` towards 0 or 1 (mild
            mixing), larger values bias it towards 0.5 (heavy mixing).
        return_perm: Also return the permutation. The Tier-3 side inputs — the
            fill map and the morphometrics — have to be mixed with the *same*
            pairing the pixels were, and re-drawing one would silently pair each
            patch's spectrum with another patch's mask.

    Returns:
        ``(mixed_x, y, y_permuted, lam)``, plus ``idx`` when ``return_perm``.
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed: MixupOut = (lam * x + (1 - lam) * x[idx], y, y[idx], lam)
    return (*mixed, idx) if return_perm else mixed


@overload
def mixed_aug(
    x: torch.Tensor, y: torch.Tensor, alpha: float = ..., return_perm: Literal[False] = ...
) -> MixupOut: ...


@overload
def mixed_aug(
    x: torch.Tensor, y: torch.Tensor, alpha: float, return_perm: Literal[True]
) -> MixupOutPerm: ...


def mixed_aug(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, return_perm: bool = False
) -> MixupOut | MixupOutPerm:
    """Public entry point for :func:`_mixup` with the Stage-1 default alpha."""
    return _mixup(x, y, alpha, True) if return_perm else _mixup(x, y, alpha)


def mix_side_inputs(
    mask: torch.Tensor | None,
    morph: torch.Tensor | None,
    perm: torch.Tensor | None,
    lam: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Interpolate the Tier-3 side inputs with the same ``perm``/``lam`` as the pixels.

    A no-op when mixup is off (``perm is None``), which is every path outside
    Stage-1 Phases 1-2.

    Both quantities are continuous and both mix linearly for a reason, not by
    analogy. The fill map is a *coverage fraction* since P-3, so the mixed
    patch's coverage genuinely is the mixture of the two coverages; the
    morphometrics are standardised physical measurements, and the interpolated
    patch is, to first order, a seed of the interpolated size. Leaving them
    un-mixed would hand the network the anchor's geometry attached to a
    composite's spectrum — the same kind of mismatch that made the pre-Tier-1
    spectral TTA view off-manifold (C-8).
    """
    if perm is None:
        return mask, morph
    mixed_mask = lam * mask + (1 - lam) * mask[perm] if mask is not None else None
    mixed_morph = lam * morph + (1 - lam) * morph[perm] if morph is not None else None
    return mixed_mask, mixed_morph


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
