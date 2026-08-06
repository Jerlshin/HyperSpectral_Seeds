"""The two per-epoch training loops: AdamW (Stages 1-2) and SAM (Stage 3).

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`train_one_epoch`                1859-1960
:func:`train_one_epoch_sam`            1967-2027
=====================================  ==============

Declared deviations, both mechanical:

* ``CONFIG["grad_clip"]`` → ``cfg.grad_clip`` (one call site in
  :func:`train_one_epoch`, two in :func:`train_one_epoch_sam`).
* ``_aux_loss_weight(current_ep, total_ep)`` → ``_aux_loss_weight(cfg, …)``,
  since that helper now reads its two weights off the config.

Both take ``cfg`` as their leading parameter as a result, matching the
convention Phase 2 set in ``data/loaders.py``.

Behaviour worth keeping in mind when reading these loops — all of it inherited,
none of it introduced here:

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

from spectralquadnet.losses.auxiliary import _aux_loss_weight, _compute_aux_loss
from spectralquadnet.losses.mixup import mixed_aug, mixed_loss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.sam import SAM

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
) -> tuple[float, float]:
    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)

    use_amp = (supcon is None) and (scaler is not None)
    aux_w = _aux_loss_weight(cfg, current_ep, total_ep)

    if model._use_arcface and use_mixup:
        raise ValueError("Mixup cannot be used with ArcFace.")

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_in, ya, yb, lam = mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)

        with autocast(device_type=device.type, enabled=use_amp):
            # ── SupCon / ProtoNCE path ─────────────────────────────────
            if supcon is not None:
                out = model(x_in, ya, return_embed=True, arc_m=arc_m)
                logits = out["main"] if isinstance(out, dict) else out[0]
                emb = out.get("emb") if isinstance(out, dict) else out[1]

                cls_l = criterion(logits, ya)
                sc_l = supcon(emb, ya)
                pt_l = proto(emb, ya) if proto is not None else 0.0

                aux_l = (
                    _compute_aux_loss(criterion, out, ya, yb, lam, use_mixup=False)
                    if isinstance(out, dict)
                    else torch.zeros((), device=device)
                )

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
                    aux_l = _compute_aux_loss(criterion, out, ya, yb, lam, use_mixup)
                    loss = l_main + aux_w * aux_l
                    logits = out["main"]
                else:
                    logits = out
                    loss = mixed_loss(criterion, logits, ya, yb, lam)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        # `use_amp` already implies `scaler is not None`, but mypy cannot see the
        # correlation between the two locals, hence the ignores below.
        if use_amp:
            scaler.scale(loss / accum_steps).backward()  # type: ignore[union-attr]
        else:
            (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
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
) -> tuple[float, float]:
    torch.set_default_dtype(torch.float32)
    model.train()
    total_loss = total_acc = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # ── SAM first step ────────────────────────────────────────────
        sam_opt.zero_grad()
        out = model(x, labels=y, arc_m=arc_m, return_embed=(supcon is not None))
        logits = out["main"] if isinstance(out, dict) else out
        emb = out.get("emb") if isinstance(out, dict) else None

        loss = criterion(logits, y)
        if supcon is not None and emb is not None:
            loss = loss + supcon_weight * supcon(emb, y)
        if isinstance(out, dict) and aux_weight > 0.0:
            aux_l = _compute_aux_loss(criterion, out, y, y, 1.0, use_mixup=False)
            loss = loss + aux_weight * aux_l

        if not torch.isfinite(loss):
            sam_opt.zero_grad()
            continue

        loss.backward()
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

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n
