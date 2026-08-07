"""Deep-supervision auxiliary-head loss and its decay schedule.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`_aux_loss_weight`               1807-1823
:func:`_compute_aux_loss`              1826-1852
=====================================  ==============

§2.1 offers ``losses/mixup.py`` or a new ``losses/auxiliary.py`` as the home for
this pair; the latter is used because the two functions are about deep
supervision, not mixup — they merely *call* :func:`~spectralquadnet.losses.mixup.mixed_loss`
when mixup is active.

Declared deviation (:func:`_aux_loss_weight` only)
──────────────────────────────────────────────────
``CONFIG["aux_loss_weight_{init,final}"]`` → ``cfg.stage1.aux_loss_weight_{init,final}``.
The arithmetic, the ``max(...)`` floor and the ``0.7`` decay factor are unchanged.
Stage 3 does not use this schedule at all — it passes a fixed
``cfg.stage3.aux_loss_weight`` straight to ``train_one_epoch_sam``.

:func:`_compute_aux_loss` keeps the ``branch_weights`` table that gives the
spectral branches A/B twice the weight of the spatial branches C/D, and keeps the
arithmetic term-for-term.

Declared deviation (:func:`_compute_aux_loss`, Phase 4 / §4.2)
─────────────────────────────────────────────────────────────
The plan's "per-branch loss contribution" diagnostic asks this function to
"return the per-branch dict alongside the summed total instead of only the
total". It does so behind ``return_components=False``, which leaves every
pre-existing call site and its return value untouched. The accumulation is
unchanged — ``total = total + w * loss`` merely names ``w * loss`` first so the
same tensor can be recorded — so the summed result is bit-identical either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import torch
import torch.nn as nn

from spectralquadnet.losses.mixup import mixed_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


def _aux_loss_weight(cfg: ExperimentConfig | Any, current_ep: int, total_ep: int) -> float:
    """
    Linearly decay the auxiliary branch loss weight from
    ``aux_loss_weight_init`` to ``aux_loss_weight_final`` over training.

    The higher weight early in training forces each branch to be
    independently discriminative; as training matures the weight decays
    so the main fused head dominates.

    Floor is set to aux_loss_weight_final so branches never stop getting
    gradient — critical for keeping spectral branches (A/B) alive.
    """
    progress = current_ep / max(total_ep, 1)
    return max(
        cfg.stage1.aux_loss_weight_final,
        cfg.stage1.aux_loss_weight_init * (1.0 - progress * 0.7),  # slower decay
    )


@overload
def _compute_aux_loss(
    criterion: nn.Module,
    out: dict[str, torch.Tensor],
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
    use_mixup: bool,
    return_components: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def _compute_aux_loss(
    criterion: nn.Module,
    out: dict[str, torch.Tensor],
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
    use_mixup: bool,
    return_components: Literal[True],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


def _compute_aux_loss(
    criterion: nn.Module,
    out: dict[str, torch.Tensor],
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
    use_mixup: bool,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the summed auxiliary head loss across all four branches.

    Handles both standard CE and Mixup-interpolated targets.
    Branches A and B (spectral) get 2× weight relative to C/D (spatial)
    to force them to learn discriminative spectral features — without this
    bias the spatial branches dominate and A/B produce near-zero gradients.

    Args:
        return_components: Also return the weighted per-branch terms, keyed by
            ``aux_a``…``aux_d``, for the §4.2 per-branch loss diagnostic. The
            summed total is identical either way.
    """
    # Weight: spectral branches A/B get 2x, spatial branches C/D get 1x
    branch_weights = {"aux_a": 2.0, "aux_b": 2.0, "aux_c": 1.0, "aux_d": 1.0}
    total = torch.zeros((), device=ya.device)
    components: dict[str, torch.Tensor] = {}
    for k, w in branch_weights.items():
        if k not in out:
            continue
        if use_mixup:
            term = w * mixed_loss(criterion, out[k], ya, yb, lam)
        else:
            term = w * criterion(out[k], ya)
        total = total + term
        components[k] = term
    return (total, components) if return_components else total
