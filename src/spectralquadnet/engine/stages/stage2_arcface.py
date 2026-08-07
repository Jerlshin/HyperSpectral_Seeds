"""Stage 2 — sub-centre ArcFace with SGDR, SupCon/ProtoNCE and CDWS batches.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`run_stage2`                     2395-2501
=====================================  ==============

Orchestration only. The single declared deviation is the mechanical
``CONFIG[...]`` → ``cfg.stage2.*`` / ``cfg.model.*`` rewrite, which also gives
``build_optimizer_s2``, ``train_one_epoch``, ``compute_class_difficulty`` and
``save_ckpt`` their leading ``cfg`` argument.

Two details that read as arbitrary but are load-bearing:

* ``optimizer.param_groups[0]`` and ``[2]`` are the head and backbone learning
  rates — that indexing depends on the group order ``build_optimizer_s2``
  produces (head-wd, head-no-wd, backbone-wd, backbone-no-wd).
* The margin is passed explicitly only *during* warmup; once
  ``ep - 1 >= margin_warmup_ep`` the call site switches to ``arc_m=None`` so the
  head applies its own per-class adaptive margins instead of a global one.

Mixup is off for the whole stage — ``train_one_epoch`` raises if ArcFace and
mixup are combined.

Phase 4 (REFACTOR_PLAN.md §4.1) replaced every ``print`` here one-for-one with a
``tracker`` call; the SGDR restart marker that used to be appended to the epoch
line is now its own row cell. Observability-only, and every value logged was
already a local here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from spectralquadnet.engine.checkpoint import save_ckpt
from spectralquadnet.engine.diagnostics import compute_class_difficulty
from spectralquadnet.engine.evaluate import evaluate
from spectralquadnet.engine.train_epoch import train_one_epoch
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import build_optimizer_s2
from spectralquadnet.optim.schedulers import arcface_margin, sgdr_scheduler
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet


def run_stage2(
    cfg: ExperimentConfig | Any,
    model: SpectralQuadNet,
    ema: ModelEMA,
    train_ldr: DataLoader[Any],
    val_ldr: DataLoader[Any],
    device: torch.device,
    best_ckpt: str,
    class_f1: dict[int, float] | None = None,
    tracker: ExperimentTracker | None = None,
) -> float:
    trk = tracker if tracker is not None else NullTracker()
    model.set_dropout(cfg.stage2.dropout)
    model.use_arcface(True)
    model.freeze_head("linear")
    model.unfreeze_head("arcface")

    ema.reinit_from(model)
    ema.set_dropout(cfg.stage2.dropout)
    ema.shadow.use_arcface(True)

    if class_f1 is not None:
        model.arcface_head.update_margins_from_f1(class_f1)
        ema.shadow.arcface_head.update_margins_from_f1(class_f1)

    focal = FocalLoss(gamma=cfg.stage2.focal_gamma)
    supcon = SupConLoss(temperature=cfg.stage2.supcon_temp)
    proto = ProtoNCELoss(temperature=cfg.stage2.proto_temp)

    optimizer = build_optimizer_s2(cfg, model, cfg.stage2.head_lr, cfg.stage2.back_lr)
    scheduler = sgdr_scheduler(
        optimizer,
        warmup_ep=cfg.stage2.warmup_ep,
        T_0=cfg.stage2.sgdr_T0,
        T_mult=cfg.stage2.sgdr_Tmult,
        eta_min_frac=cfg.stage2.min_lr / cfg.stage2.head_lr,
    )

    sc_w = cfg.stage2.supcon_weight
    pt_w = cfg.stage2.proto_weight
    ep_total = cfg.stage2.epochs
    best_f1 = 0.0
    no_improve = 0

    r1 = cfg.stage2.warmup_ep + cfg.stage2.sgdr_T0
    r2 = r1 + cfg.stage2.sgdr_T0 * cfg.stage2.sgdr_Tmult

    trk.banner(
        f"Stage 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR  [{ep_total} ep]",
        [
            f"hLR={cfg.stage2.head_lr:.1e}  bLR={cfg.stage2.back_lr:.1e}  "
            f"SGDR T0={cfg.stage2.sgdr_T0} Tmult={cfg.stage2.sgdr_Tmult} "
            f"→ restarts ep {r1} & {r2}",
            f"ArcFace K={cfg.model.subcenter_K}  "
            f"m={cfg.stage2.arcface_m0}→{cfg.stage2.arcface_m}+Δ{cfg.stage2.arcface_m_delta}",
            f"Losses: Focal(γ={cfg.stage2.focal_gamma}) + SupCon(w={sc_w}) + ProtoNCE(w={pt_w})",
            f"Batch: {cfg.stage2.bal_n_cls} cls × {cfg.stage2.bal_n_spc} spc = "
            f"{cfg.stage2.bal_n_cls * cfg.stage2.bal_n_spc} | Primary metric: macro-F1",
        ],
    )

    for ep in range(1, ep_total + 1):
        warmup_done = (ep - 1) >= cfg.stage2.margin_warmup_ep
        m_now = (
            cfg.stage2.arcface_m
            if warmup_done
            else arcface_margin(
                ep - 1, cfg.stage2.arcface_m0, cfg.stage2.arcface_m, cfg.stage2.margin_warmup_ep
            )
        )
        arc_m = None if warmup_done else m_now
        ramp = min(1.0, ep / 10.0)
        sc_now = sc_w * ramp
        pt_now = pt_w * ramp

        tl, ta = train_one_epoch(
            cfg,
            model,
            train_ldr,
            optimizer,
            focal,
            scaler=None,
            ema=ema,
            device=device,
            scheduler=None,
            use_mixup=False,
            supcon=supcon,
            supcon_weight=sc_now,
            proto=proto,
            proto_weight=pt_now,
            arc_m=arc_m,
            current_ep=ep,
            total_ep=ep_total,
            tracker=trk,
        )
        scheduler.step()

        f1_live, acc_live = evaluate(model, val_ldr, device)
        f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1 = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        head_lr = optimizer.param_groups[0]["lr"]
        back_lr = optimizer.param_groups[2]["lr"]
        saved = ""

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1_s2, _cdws_s2 = compute_class_difficulty(
                cfg, ema.shadow, val_ldr, device, "S2", tracker=trk, step=ep
            )
            save_ckpt(
                cfg,
                best_ckpt,
                ep,
                "Stage 2",
                model,
                ema,
                val_f1=best_ep_f1,
                val_acc=best_ep_acc,
                class_f1=_cf1_s2,
                cdws_weights=_cdws_s2,
                s2_val_f1=best_ep_f1,
            )
            saved = "✓"  # rendered as its own row cell now, so no padding
        else:
            no_improve += 1

        rf = "↻R1" if ep == r1 else ("↻R2" if ep == r2 else "")
        trk.log_row(
            "stage2",
            {
                "Ep": f"{ep:03d}/{ep_total}",
                "Loss": f"{tl:.4f}",
                "Tr": f"{ta:.1%}",
                "F1 live/ema": f"{f1_live:.3f}/{f1_ema:.3f}",
                "Acc live/ema": f"{acc_live:.1%}/{acc_ema:.1%}",
                "hLR": f"{head_lr:.1e}",
                "bLR": f"{back_lr:.1e}",
                "m": f"{m_now:.3f}",
                "ckpt": saved,
                "sgdr": rf,
            },
            step=ep,
        )
        trk.log_scalars(
            {
                "train/loss": tl,
                "train/acc": ta,
                "val/f1_live": f1_live,
                "val/acc_live": acc_live,
                "val/f1_ema": f1_ema,
                "val/acc_ema": acc_ema,
                "val/f1_best": best_f1,
                "sched/head_lr": head_lr,
                "sched/back_lr": back_lr,
                "sched/arcface_margin": m_now,
                "sched/supcon_weight": sc_now,
                "sched/proto_weight": pt_now,
            },
            step=ep,
        )

        if no_improve >= cfg.stage2.patience:
            trk.log_message(f"Early stopping at epoch {ep}.", level="warn")
            break

    model.unfreeze_head("linear")
    return best_f1
