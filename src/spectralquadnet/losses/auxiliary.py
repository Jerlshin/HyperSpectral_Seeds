"""Deep-supervision auxiliary-head loss and its decay schedule.

These two functions are about deep supervision, not mixup — they merely
*call* :func:`~spectralquadnet.losses.mixup.mixed_loss` when mixup is active.

Stage 3 does not use :func:`_aux_loss_weight`'s decay schedule at all; it
passes a fixed ``cfg.stage3.aux_loss_weight`` straight to
``train_one_epoch_sam``.

:func:`_compute_aux_loss` can optionally return the per-branch terms alongside
the summed total (``return_components=True``), for the per-branch loss
diagnostic in ``engine/diagnostics.py``; the summed total is identical either
way.

**IC-2 — both the raw and the weighted term are returned.** The components used
to be :math:`\\omega_b \\mathcal L_b` alone, and CHANGES.md §10.2 is what that
cost: :math:`\\omega_b` is itself non-stationary and spends most of training
pinned at a clip bound, so ``loss/branch_a`` collapsing from 11 to ~1 at epoch
10 was *the weight hitting its 0.25 floor*, not Branch A learning anything. A
reader judging branch health from that panel was reading the controller acting
on the branch. :class:`AuxComponents` carries both, and the loops log both under
``loss/branch_{a,b,c,d}_{raw,weighted}``.

:class:`GradNormAuxWeights` is OP-2 / T2-6. :data:`DEFAULT_BRANCH_WEIGHTS` is a
constant chosen to compensate for a *measured* gradient collapse in the
spectral branches A/B, so it has to be re-tuned by hand whenever anything
upstream of it changes and silently over- or under-corrects in between. The
GradNorm rule computes the same compensation each epoch from the per-branch
gradient norms the loops already log under ``grad_norm/branch_{a,b,c,d}``,
which turns the balance from an assertion into a measurement.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

import torch
import torch.nn as nn

from spectralquadnet.losses.mixup import mixed_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

_log = logging.getLogger(__name__)

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
#:
#: **These bounds are the measured output of the controller, not a safety net**
#: (CHANGES.md §8.2). ``aux_weight/branch_{b,c}`` sat pinned at 4.0 for hundreds
#: of epochs and ``branch_a`` at 0.25 from epoch ~10, so the "adaptive" vector
#: was in practice the constant ``(0.25, 4, 4, ~1–4)`` chosen here. That is why
#: :data:`~spectralquadnet.losses.auxiliary.GradNormAuxWeights` now defaults off
#: (``aux_gradnorm_alpha = 0.0``) rather than being re-tuned.
AUX_WEIGHT_BOUNDS: tuple[float, float] = (0.25, 4.0)


@dataclass(frozen=True)
class AuxComponents(Mapping[str, torch.Tensor]):
    """Per-branch auxiliary terms, both before and after the weight (IC-2).

    Iterating, indexing and ``.values()`` all address the **weighted** terms, so
    every caller written against the plain ``dict[str, Tensor]`` this replaces
    keeps working and the summed total is still ``sum(components.values())``.
    :attr:`raw` is the addition: :math:`\\mathcal L_b` itself, unmultiplied.

    Logging both is what makes the branch panels readable. With only the
    weighted term, a curve falling by 16× is indistinguishable between "the
    branch learned" and "the controller drove :math:`\\omega_b` to its floor" —
    and in the audited run it was always the second (CHANGES.md §10.2).
    """

    raw: dict[str, torch.Tensor] = field(default_factory=dict)
    weighted: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.weighted[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.weighted)

    def __len__(self) -> int:
        return len(self.weighted)

    def scalars(self) -> dict[str, torch.Tensor]:
        """Both series as tracker tags: ``loss/branch_a_raw``, ``…_weighted``, …"""
        tags: dict[str, torch.Tensor] = {}
        for key, value in self.raw.items():
            tags[f"loss/branch_{key.removeprefix('aux_')}_raw"] = value
        for key, value in self.weighted.items():
            tags[f"loss/branch_{key.removeprefix('aux_')}_weighted"] = value
        return tags


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

        Branches absent from ``grad_norms``, or reporting a norm that is not
        finite and positive, keep their current weight — a branch that produced
        no gradient this epoch, or produced one that overflowed, carries no
        information about the balance.

        **The ``isfinite`` half of that test is load-bearing under AMP**, and it
        is where a skipped batch used to become a dead run. An fp16 gradient
        that overflows is *routine* — catching it and skipping the optimiser
        step is the entire job of ``GradScaler`` — but the epoch's mean
        per-branch norm is accumulated before the scaler gets a say, so one
        such step makes the mean ``inf``. Left unfiltered, ``inf`` passes the
        ``> 0`` test, ``mean`` becomes ``inf``, ``(inf/inf) ** alpha`` is
        ``NaN``, and the bounds below do not catch it: ``min``/``max`` against
        ``NaN`` in Python return the ``NaN``. The weight is then ``NaN``
        forever, every subsequent loss is ``NaN``, every batch is skipped, and
        the epoch line still reports a mean over ``len(loader)`` as though the
        epoch had trained. bf16 makes the overflow itself unreachable; this
        keeps the *consequence* unreachable on the fp16 path too.

        Returns:
            The updated weights, keyed ``aux_a``…``aux_d``.
        """
        observed: dict[str, float] = {}
        for key in self.weights:
            norm = grad_norms.get(key.replace("aux_", "branch_"))
            if norm is None:
                continue
            norm = float(norm)
            if math.isfinite(norm) and norm > 0.0:
                observed[key] = norm
            elif not math.isfinite(norm):
                _log.warning(
                    "GradNorm ignoring a non-finite gradient norm for %s (%s); "
                    "its auxiliary weight is held at %.3f",
                    key,
                    norm,
                    self.weights[key],
                )
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
) -> tuple[torch.Tensor, AuxComponents]: ...


def _compute_aux_loss(
    criterion: nn.Module,
    out: dict[str, torch.Tensor],
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
    use_mixup: bool,
    return_components: bool = False,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, AuxComponents]:
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
        return_components: Also return an :class:`AuxComponents` carrying the
            per-branch terms keyed ``aux_a``..``aux_d``, **both** raw
            (:math:`\\mathcal L_b`) and weighted
            (:math:`\\omega_b \\mathcal L_b`). The summed total is identical
            either way.
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
    # Any other `aux_*` head the model emitted — `SpectralSeedNet` has exactly
    # one, `aux_spatial`. Appended in sorted order *after* the four known keys,
    # so the summation order for a four-branch model is byte-for-byte the one
    # the golden Stage-1 loss was captured under.
    branch_weights.update(
        {
            k: float(active.get(k, 1.0))
            for k in sorted(out)
            if k.startswith("aux_") and k not in branch_weights
        }
    )
    total = torch.zeros((), device=ya.device)
    raw: dict[str, torch.Tensor] = {}
    weighted: dict[str, torch.Tensor] = {}
    for k, w in branch_weights.items():
        if k not in out:
            continue
        # The raw CE is materialised first and the weight applied to it, rather
        # than the two being fused: `term` must stay bit-identical to what the
        # single-expression form produced, and `w * base` is that expression.
        base = mixed_loss(criterion, out[k], ya, yb, lam) if use_mixup else criterion(out[k], ya)
        term = w * base
        total = total + term
        if return_components:
            raw[k] = base
            weighted[k] = term
    if not return_components:
        return total
    return total, AuxComponents(raw=raw, weighted=weighted)
