"""Deep-supervision auxiliary-head loss and its decay schedule.

These two functions are about deep supervision, not mixup — they merely
*call* :func:`~spectralquadnet.losses.mixup.mixed_loss` when mixup is active.

Stage 3 does not use :func:`_aux_loss_weight`'s decay schedule at all; it
passes a fixed ``cfg.stage3.aux_loss_weight`` straight to
``train_one_epoch_sam``.

:func:`_compute_aux_loss` can optionally return the weighted per-branch terms
alongside the summed total (``return_components=True``), for the per-branch
loss diagnostic in ``engine/diagnostics.py``; the summed total is identical
either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import torch
import torch.nn as nn

from spectralquadnet.losses.mixup import mixed_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


def _aux_loss_weight(cfg: ExperimentConfig | Any, current_ep: int, total_ep: int) -> float:
    """Linearly decay the auxiliary branch loss weight over training.

    The higher weight early in training forces each branch to be
    independently discriminative; as training matures the weight decays
    towards ``aux_loss_weight_final`` so the main fused head dominates. The
    floor is set to ``aux_loss_weight_final`` (never zero) so branches never
    stop getting gradient — critical for keeping the spectral branches (A/B)
    alive.

    Args:
        cfg: Composed experiment config, read for
            ``cfg.stage1.aux_loss_weight_{init,final}``.
        current_ep: Current epoch within Stage 1 (0-indexed).
        total_ep: Total number of Stage-1 epochs, used to compute progress.

    Returns:
        The auxiliary loss weight for this epoch.
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
    """Compute the summed auxiliary head loss across all four branches.

    Handles both standard cross-entropy and mixup-interpolated targets.
    Branches A and B (spectral) get 2x weight relative to C/D (spatial) to
    force them to learn discriminative spectral features — without this
    bias the spatial branches dominate and A/B produce near-zero gradients.

    Args:
        criterion: Base loss module, called as ``criterion(logits, labels)``.
        out: Model output dict; only the ``aux_a``..``aux_d`` keys present
            are summed (a missing key contributes nothing).
        ya: Primary labels (or mixup's unpermuted labels).
        yb: Mixup's permuted labels; ignored if ``use_mixup`` is False.
        lam: Mixup interpolation coefficient; ignored if ``use_mixup`` is False.
        use_mixup: Whether to interpolate via :func:`~spectralquadnet.losses.mixup.mixed_loss`
            instead of applying ``criterion`` directly.
        return_components: Also return the weighted per-branch terms, keyed
            by ``aux_a``..``aux_d``. The summed total is identical either way.

    Returns:
        The summed weighted auxiliary loss, or ``(total, components)`` when
        ``return_components`` is True.
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
