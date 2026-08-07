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

:class:`GradNormAuxWeights` is OP-2 / T2-6. :data:`DEFAULT_BRANCH_WEIGHTS` is a
constant chosen to compensate for a *measured* gradient collapse in the
spectral branches A/B, so it has to be re-tuned by hand whenever anything
upstream of it changes and silently over- or under-corrects in between. The
GradNorm rule computes the same compensation each epoch from the per-branch
gradient norms the loops already log under ``grad_norm/branch_{a,b,c,d}``,
which turns the balance from an assertion into a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

import torch
import torch.nn as nn

from spectralquadnet.losses.mixup import mixed_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: The fixed per-branch auxiliary weights, and the starting point of the
#: GradNorm feedback loop. Spectral branches A/B get 2×, spatial C/D get 1×.
DEFAULT_BRANCH_WEIGHTS: dict[str, float] = {
    "aux_a": 2.0,
    "aux_b": 2.0,
    "aux_c": 1.0,
    "aux_d": 1.0,
}

#: Bounds on any single GradNorm weight. Without them one epoch in which a
#: branch's gradient is near zero — a frozen head, a batch that skipped — sends
#: its weight to infinity and the update is not recoverable.
AUX_WEIGHT_BOUNDS: tuple[float, float] = (0.25, 4.0)


class GradNormAuxWeights:
    """GradNorm-style per-branch auxiliary weights (OP-2 / T2-6).

    .. math::
        \\omega_b^{(t+1)} = \\omega_b^{(t)}
        \\Big(\\frac{\\bar g}{g_b}\\Big)^{\\alpha},
        \\qquad \\bar g = \\tfrac14\\textstyle\\sum_b g_b,\\ \\alpha = 0.5

    Updated **once per epoch**, not per step: ``g_b`` is an epoch mean, and a
    feedback loop closed on a per-batch gradient norm oscillates rather than
    converges. The existing ``w_aux(e): 0.65 -> 0.25`` decay stays an outer
    multiplier, so this rule sets the *balance* between branches and the
    schedule sets the overall strength.

    ``alpha = 0`` freezes the weights at :data:`DEFAULT_BRANCH_WEIGHTS`, which
    is the pre-Tier-2 behaviour exactly.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        init: Mapping[str, float] | None = None,
        bounds: tuple[float, float] = AUX_WEIGHT_BOUNDS,
    ) -> None:
        self.alpha = float(alpha)
        self.bounds = bounds
        self.weights: dict[str, float] = dict(init or DEFAULT_BRANCH_WEIGHTS)

    def update(self, grad_norms: Mapping[str, float]) -> dict[str, float]:
        """Apply one GradNorm update from ``{"branch_a": norm, …}``.

        Branches absent from ``grad_norms``, or reporting a non-positive norm,
        keep their current weight — a branch that produced no gradient this
        epoch carries no information about the balance.

        Returns:
            The updated weights, keyed ``aux_a``…``aux_d``.
        """
        observed = {
            key: float(grad_norms[key.replace("aux_", "branch_")])
            for key in self.weights
            if grad_norms.get(key.replace("aux_", "branch_"), 0.0) > 0.0
        }
        if self.alpha == 0.0 or len(observed) < 2:
            return dict(self.weights)

        mean = sum(observed.values()) / len(observed)
        low, high = self.bounds
        for key, norm in observed.items():
            scaled = self.weights[key] * (mean / norm) ** self.alpha
            self.weights[key] = min(max(scaled, low), high)
        return dict(self.weights)

    def scalars(self) -> dict[str, float]:
        """The weights as tracker keys, ``aux_weight/branch_a``…"""
        return {f"aux_weight/{k.replace('aux_', 'branch_')}": v for k, v in self.weights.items()}


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
    weights: Mapping[str, float] | None = None,
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
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


def _compute_aux_loss(
    criterion: nn.Module,
    out: dict[str, torch.Tensor],
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
    use_mixup: bool,
    return_components: bool = False,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the summed auxiliary head loss across all four branches.

    Handles both standard cross-entropy and mixup-interpolated targets.
    Branches A and B (spectral) get 2x weight relative to C/D (spatial) to
    force them to learn discriminative spectral features — without this
    bias the spatial branches dominate and A/B produce near-zero gradients.
    ``weights`` replaces that fixed vector with a measured one; see
    :class:`GradNormAuxWeights`.

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
        weights: Per-branch weights keyed ``aux_a``..``aux_d``; defaults to
            :data:`DEFAULT_BRANCH_WEIGHTS`. Iteration order comes from the
            default table either way, so the summation order — and therefore
            the float — does not depend on the caller's dict.

    Returns:
        The summed weighted auxiliary loss, or ``(total, components)`` when
        ``return_components`` is True.
    """
    active = DEFAULT_BRANCH_WEIGHTS if weights is None else weights
    branch_weights = {k: float(active.get(k, v)) for k, v in DEFAULT_BRANCH_WEIGHTS.items()}
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
