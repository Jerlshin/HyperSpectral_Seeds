"""The two per-epoch training loops: AdamW (Stages 1-2) and SAM (Stage 3).

Both loops take an optional ``tracker``. When it is ``None``, nothing extra
happens: no diagnostic is accumulated and no device synchronisation is
added beyond the loop's normal work. When a tracker is supplied, two
families of scalars are recorded, and both come from values the loop
already computes:

* ``loss/branch_{a,b,c,d}`` — the weighted per-branch terms
  ``_compute_aux_loss`` sums internally, now returned alongside the total.
* ``grad_norm/{branch_a,…,cross_interaction,arcface_head}`` — the pre-clip
  gradient norm split by owner, sampled at the same place
  ``clip_grad_norm_`` runs, and gated on ``cfg.tracking.log_grad_norms``.

Both are accumulated as device tensors and resolved to floats **once per epoch**,
so the per-step cost is a handful of adds rather than a host synchronisation.

Behaviour worth keeping in mind when reading these loops:

* **AMP is silently disabled whenever a SupCon loss is passed** (``use_amp =
  (supcon is None) and (scaler is not None)``), so Stage 1 Phase 3 and all of
  Stage 2 train in full fp32.
* **Non-finite losses skip the batch**, zeroing grads rather than raising, in
  both loops.
* **The optimiser steps on the accumulation boundary only**, and the EMA is
  updated in the same block, so ``ema.update`` is called once per *optimiser*
  step rather than once per batch.
* **Mixup and ArcFace are mutually exclusive** and the combination raises.
* :func:`train_one_epoch_sam` returns the *first*-step loss and accuracy; the
  second (descent) step's loss is computed but only used for its gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from spectralquadnet.engine.diagnostics import branch_grad_norm_tensors
from spectralquadnet.losses.auxiliary import _aux_loss_weight, _compute_aux_loss
from spectralquadnet.losses.mixup import mixed_aug, mixed_loss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.sam import SAM
from spectralquadnet.tracking.base import ExperimentTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


def train_one_epoch(
    cfg: ExperimentConfig | Any,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler | None,
    ema: ModelEMA,
    device: torch.device,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    supcon: nn.Module | None = None,
    supcon_weight: float = 0.0,
    proto: nn.Module | None = None,
    proto_weight: float = 0.0,
    accum_steps: int = 1,
    arc_m: float | None = None,
    current_ep: int = 0,
    total_ep: int = 100,
    tracker: ExperimentTracker | None = None,
) -> tuple[float, float]:
    """Run one AdamW training epoch (Stage 1 or Stage 2).

    Combines the main classification loss with the per-branch auxiliary
    loss, and optionally SupCon/ProtoNCE contrastive terms; supports mixup
    and gradient accumulation. See the module docstring for cross-cutting
    behaviour (AMP-disable-under-SupCon, non-finite-loss skip, EMA update
    cadence).

    Returns:
        ``(mean_loss, mean_accuracy)`` over the epoch.
    """
    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)

    use_amp = (supcon is None) and (scaler is not None)
    aux_w = _aux_loss_weight(cfg, current_ep, total_ep)

    if model._use_arcface and use_mixup:
        raise ValueError("Mixup cannot be used with ArcFace.")

    # Diagnostics are off entirely without a tracker.
    want_diag = tracker is not None
    want_grad_norms = want_diag and bool(
        getattr(getattr(cfg, "tracking", None), "log_grad_norms", False)
    )
    diag_aux: dict[str, torch.Tensor] = {}
    diag_grad: dict[str, torch.Tensor] = {}
    n_aux = n_clip = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_in, ya, yb, lam = mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)
        aux_parts: dict[str, torch.Tensor] = {}

        with autocast(device_type=device.type, enabled=use_amp):
            # ── SupCon / ProtoNCE path ─────────────────────────────────
            if supcon is not None:
                out = model(x_in, ya, return_embed=True, arc_m=arc_m)
                logits = out["main"] if isinstance(out, dict) else out[0]
                emb = out.get("emb") if isinstance(out, dict) else out[1]

                cls_l = criterion(logits, ya)
                sc_l = supcon(emb, ya)
                pt_l = proto(emb, ya) if proto is not None else 0.0

                if isinstance(out, dict):
                    aux_l, aux_parts = _compute_aux_loss(
                        criterion, out, ya, yb, lam, use_mixup=False, return_components=True
                    )
                else:
                    aux_l = torch.zeros((), device=device)

                loss = (
                    (1 - supcon_weight - proto_weight) * cls_l
                    + supcon_weight * sc_l
                    + proto_weight * pt_l
                    + aux_w * aux_l
                )

            # ── Standard CE / Focal path ───────────────────────────────
            else:
                arc_labels = ya if (model._use_arcface and not use_mixup) else None
                out = model(x_in, labels=arc_labels, arc_m=arc_m)

                if isinstance(out, dict):
                    l_main = mixed_loss(criterion, out["main"], ya, yb, lam)
                    aux_l, aux_parts = _compute_aux_loss(
                        criterion, out, ya, yb, lam, use_mixup, return_components=True
                    )
                    loss = l_main + aux_w * aux_l
                    logits = out["main"]
                else:
                    logits = out
                    loss = mixed_loss(criterion, logits, ya, yb, lam)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if want_diag and aux_parts:
            _accumulate(diag_aux, aux_parts)
            n_aux += 1

        # `use_amp` already implies `scaler is not None`, but mypy cannot see the
        # correlation between the two locals, hence the ignores below.
        if use_amp:
            scaler.scale(loss / accum_steps).backward()  # type: ignore[union-attr]
        else:
            (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
            # Sampled after unscale_ and before the clip, so the norms are the
            # true pre-clip ones.
            if want_grad_norms:
                _accumulate(diag_grad, branch_grad_norm_tensors(model))
                n_clip += 1
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if use_amp:
                scaler.step(optimizer)  # type: ignore[union-attr]
                scaler.update()  # type: ignore[union-attr]
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == ya).float().mean().item()

    if tracker is not None:
        scalars = _diagnostic_scalars(diag_aux, n_aux, diag_grad, n_clip)
        if scalars:
            tracker.log_scalars(scalars, step=current_ep)

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


def train_one_epoch_sam(
    cfg: ExperimentConfig | Any,
    model: nn.Module,
    loader: DataLoader[Any],
    sam_opt: SAM,
    criterion: nn.Module,
    device: torch.device,
    supcon: nn.Module | None = None,
    supcon_weight: float = 0.0,
    proto: nn.Module | None = None,
    proto_weight: float = 0.0,
    arc_m: float | None = None,
    aux_weight: float = 0.0,  # fixed aux weight for Stage 3 (typically small)
    current_ep: int = 0,
    tracker: ExperimentTracker | None = None,
) -> tuple[float, float]:
    """Run one Sharpness-Aware Minimisation training epoch (Stage 3).

    Each batch runs SAM's two-step contract: an ascent step to the local
    worst case (:meth:`SAM.first_step`), then a descent step from the
    original weights using the gradient computed there
    (:meth:`SAM.second_step`). SAM is not compatible with AMP, so this loop
    always runs in fp32.

    Returns:
        ``(mean_loss, mean_accuracy)`` over the epoch, computed from the
        first (ascent) step's forward pass.
    """
    torch.set_default_dtype(torch.float32)
    model.train()
    total_loss = total_acc = 0.0

    # Diagnostics are off entirely without a tracker.
    want_diag = tracker is not None
    want_grad_norms = want_diag and bool(
        getattr(getattr(cfg, "tracking", None), "log_grad_norms", False)
    )
    diag_aux: dict[str, torch.Tensor] = {}
    diag_grad: dict[str, torch.Tensor] = {}
    n_aux = n_clip = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        aux_parts: dict[str, torch.Tensor] = {}

        # ── SAM first step ────────────────────────────────────────────
        sam_opt.zero_grad()
        out = model(x, labels=y, arc_m=arc_m, return_embed=(supcon is not None))
        logits = out["main"] if isinstance(out, dict) else out
        emb = out.get("emb") if isinstance(out, dict) else None

        loss = criterion(logits, y)
        if supcon is not None and emb is not None:
            loss = loss + supcon_weight * supcon(emb, y)
        if isinstance(out, dict) and aux_weight > 0.0:
            aux_l, aux_parts = _compute_aux_loss(
                criterion, out, y, y, 1.0, use_mixup=False, return_components=True
            )
            loss = loss + aux_weight * aux_l

        if not torch.isfinite(loss):
            sam_opt.zero_grad()
            continue

        if want_diag and aux_parts:
            _accumulate(diag_aux, aux_parts)
            n_aux += 1

        loss.backward()
        # Pre-clip, and on the ascent step: these are the gradients SAM uses to
        # find the adversarial weight perturbation.
        if want_grad_norms:
            _accumulate(diag_grad, branch_grad_norm_tensors(model))
            n_clip += 1
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        sam_opt.first_step(zero_grad=True)

        # ── SAM second step ───────────────────────────────────────────
        out2 = model(x, labels=y, arc_m=arc_m, return_embed=(supcon is not None))
        logits2 = out2["main"] if isinstance(out2, dict) else out2

        loss2 = criterion(logits2, y)
        if not torch.isfinite(loss2):
            sam_opt.zero_grad()
            continue

        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        sam_opt.second_step(zero_grad=True)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.detach().argmax(1) == y).float().mean().item()

    if tracker is not None:
        scalars = _diagnostic_scalars(diag_aux, n_aux, diag_grad, n_clip)
        if scalars:
            tracker.log_scalars(scalars, step=current_ep)

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════════════
#  Diagnostic accumulation helpers
# ══════════════════════════════════════════════════════════════════════


def _accumulate(sink: dict[str, torch.Tensor], values: dict[str, torch.Tensor]) -> None:
    """Add ``values`` into ``sink`` in place, detached and on-device.

    Nothing is transferred to the host here — this runs every step.
    """
    for key, value in values.items():
        detached = value.detach()
        sink[key] = detached if key not in sink else sink[key] + detached


def _diagnostic_scalars(
    aux_sums: dict[str, torch.Tensor],
    n_aux: int,
    grad_sums: dict[str, torch.Tensor],
    n_grad: int,
) -> dict[str, float]:
    """Resolve one epoch's accumulated diagnostics to means, in a single sync.

    Returns ``{"loss/branch_a": …, "grad_norm/branch_a": …}`` — empty when
    nothing was accumulated, so callers can skip the tracker call entirely.
    """
    keys: list[str] = []
    tensors: list[torch.Tensor] = []
    if n_aux:
        for key, value in aux_sums.items():
            keys.append(f"loss/branch_{key.removeprefix('aux_')}")
            tensors.append(value / n_aux)
    if n_grad:
        for key, value in grad_sums.items():
            keys.append(f"grad_norm/{key}")
            tensors.append(value / n_grad)
    if not keys:
        return {}
    values = torch.stack(tensors).cpu().tolist()
    return {k: float(v) for k, v in zip(keys, values, strict=True)}
