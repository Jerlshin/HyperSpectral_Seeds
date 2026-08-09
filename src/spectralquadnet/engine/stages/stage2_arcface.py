"""Stage 2 — sub-centre ArcFace with SGDR, SupCon/ProtoNCE and CDWS batches.

Orchestration only. Two details that read as arbitrary but are load-bearing:

* ``optimizer.param_groups[0]`` and ``[2]`` are the head and backbone learning
  rates — that indexing depends on the group order ``build_optimizer_s2``
  produces (head-wd, head-no-wd, backbone-wd, backbone-no-wd).
* The margin is passed explicitly only *during* warmup; once
  ``ep - 1 >= margin_warmup_ep`` the call site switches to ``arc_m=None`` so the
  head applies its own per-class adaptive margins instead of a global one.

Mixup is off for the whole stage — ``train_one_epoch`` raises if a non-zero
margin and mixup are combined.

The per-class margins are calibrated once at stage entry by HD-3's **signed**
rule (T2-8): ``M(c) = clip(m + m_delta (R_c - P_c), 0.20, 0.50)``, plus the
row-normalised confusion matrix that aims a pairwise term at the classes each
class is actually mistaken for. Both come from a fresh
:func:`~spectralquadnet.engine.evaluate.evaluate_pr_and_confusion` pass on the
shadow the stage inherits.

**Which split that pass runs on is P-5 (T4-5).** ``calib_ldr`` — never used
for a gradient, never used for selection — when a calibration split exists;
``val_ldr`` when it does not, which is the pre-Tier-4 behaviour and exactly the
contamination §2.1.4/C-9 is about: 270 fitted parameters read off the split
that also selects the checkpoint. Once, not per epoch, either way —
recalibrating every epoch would multiply that leak rather than reduce it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectralquadnet.engine.checkpoint import save_ckpt
from spectralquadnet.engine.diagnostics import (
    compute_class_difficulty,
    epoch_tag,
    should_render_details,
)
from spectralquadnet.engine.evaluate import evaluate, evaluate_pr_and_confusion
from spectralquadnet.engine.train_epoch import train_one_epoch
from spectralquadnet.losses.auxiliary import GradNormAuxWeights
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import build_optimizer_s2
from spectralquadnet.optim.schedulers import arcface_margin, sgdr_scheduler, subcentre_tau
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.device import RuntimePlan, release_memory
from spectralquadnet.utils.distributed import DistContext

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
    tracker: ExperimentTracker | None = None,
    calib_ldr: DataLoader[Any] | None = None,
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
    train_module: nn.Module | None = None,
) -> float:
    """Run Stage 2's ArcFace fine-tuning and return the best validation F1.

    Turns the shared head's margin on, warming it up to ``cfg.stage2.arcface_m``
    over ``margin_warmup_ep`` epochs (then letting the per-class adaptive
    margins take over), ramps in SupCon/ProtoNCE contrastive weights over the
    first 10 epochs, and trains under an SGDR schedule with separate
    head/backbone learning rates.

    Args:
        cfg: Composed experiment config.
        model: Model to train.
        ema: EMA shadow; re-initialised from ``model`` at stage entry.
        train_ldr: Stage-2 (CDWS-weighted) training loader.
        val_ldr: Validation loader for per-epoch evaluation.
        device: Device to train on.
        best_ckpt: Path to write the best-F1 checkpoint to.
        tracker: Experiment tracker for progress reporting; ``None`` logs nowhere.
        calib_ldr: Calibration loader (P-5 / T4-5). The 90 margins, the
            ``(90, 90)`` confusion matrix and the CDWS weights are all fitted
            here, so ``val_ldr`` carries no fitted parameter and stays a clean
            selection split. ``None`` fits them on ``val_ldr`` — the
            pre-Tier-4 behaviour.

    Returns:
        Best validation macro-F1 (across the live model and its EMA shadow)
        seen during the run.
    """
    trk = tracker if tracker is not None else NullTracker()
    dist = dist or DistContext()
    # See `run_stage1`: eager module for attributes and grouping, wrapped
    # module for the forward.
    fwd = train_module if train_module is not None else model
    detail_every = plan.diagnostics_interval if plan is not None else 1
    model.set_dropout(cfg.stage2.dropout)

    # Stage entry is a phase transition of the same shape as Stage 1's: the
    # re-init doubles the parameter footprint for the length of the copy, and it
    # lands on whatever Stage 1 left in the pool. See `run_stage1`.
    release_memory(device)
    ema.reinit_from(model)
    ema.set_dropout(cfg.stage2.dropout)
    release_memory(device)

    # One name, one meaning: `val_ldr` selects, `fit_ldr` fits (P-5 / T4-5).
    fit_ldr = calib_ldr if calib_ldr is not None else val_ldr
    fit_split = "calib" if calib_ldr is not None else "val"

    # ── HD-3 calibration (T2-8), on the calibration split (T4-5) ──────
    precision, recall, confusion = evaluate_pr_and_confusion(
        ema.shadow, fit_ldr, device, cfg.data.num_classes, dist
    )
    for head in (model.arcface_head, ema.shadow.arcface_head):
        head.update_margins_from_pr(precision, recall)
        head.set_confusion(confusion)
    hardest = sorted(recall, key=lambda c: recall[c])[:5]
    trk.log_message(
        f"HD-3 margins from R−P on {fit_split}. Five lowest-recall classes: "
        + "  ".join(
            f"c{c}: R={recall[c]:.2f} P={precision[c]:.2f} m={model.arcface_head.margins[c]:.2f}"
            for c in hardest
        )
    )

    focal = FocalLoss(gamma=cfg.stage2.focal_gamma)
    supcon = SupConLoss(temperature=cfg.stage2.supcon_temp)
    proto = ProtoNCELoss(temperature=cfg.stage2.proto_temp)

    optimizer = build_optimizer_s2(
        cfg,
        model,
        cfg.stage2.head_lr,
        cfg.stage2.back_lr,
        fused=plan.fused_optimizer if plan else False,
    )
    scheduler = sgdr_scheduler(
        optimizer,
        warmup_ep=cfg.stage2.warmup_ep,
        T_0=cfg.stage2.sgdr_T0,
        T_mult=cfg.stage2.sgdr_Tmult,
        eta_min_frac=cfg.stage2.min_lr / cfg.stage2.head_lr,
    )

    aux_weights = GradNormAuxWeights(alpha=cfg.aux_gradnorm_alpha)

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
            f"Fitted on: {fit_split} ({len(fit_ldr.dataset)} patches)  |  "  # type: ignore[arg-type]
            f"Selected on: val ({len(val_ldr.dataset)} patches)",  # type: ignore[arg-type]
        ],
    )

    trk.progress_start("stage2", ep_total, "Stage 2")
    for ep in range(1, ep_total + 1):
        detail = should_render_details(ep, detail_every)
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
        tau_now = subcentre_tau(
            ep, ep_total, cfg.model.subcenter_tau_init, cfg.model.subcenter_tau_final
        )
        model.set_subcentre_tau(tau_now)
        ema.shadow.set_subcentre_tau(tau_now)

        tl, ta = train_one_epoch(
            cfg,
            fwd,
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
            aux_weights=aux_weights,
        )
        scheduler.step()

        tag = epoch_tag(ep, ep_total)

        # Bracketed for the same reason as Stage 1's — see `run_stage1`.
        release_memory(device)
        f1_live, acc_live = evaluate(model, val_ldr, device, dist)
        f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device, dist)
        release_memory(device)
        best_ep_f1 = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        # See stage1_progressive.py — `final_eval` evaluates whichever of the
        # two won the max (T1-8 / §2.1.4).
        best_ep_source = "ema" if f1_ema >= f1_live else "live"
        head_lr = optimizer.param_groups[0]["lr"]
        back_lr = optimizer.param_groups[2]["lr"]
        saved = ""

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1_s2, _cdws_s2 = compute_class_difficulty(
                cfg,
                ema.shadow,
                fit_ldr,
                device,
                "S2",
                tracker=trk,
                step=ep,
                detail=detail,
                total_steps=ep_total,
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
                dist=dist,
                best_source=best_ep_source,
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
        # One transfer for the three margin summaries instead of three; the
        # vector is constant within an epoch, so this is a per-epoch cost that
        # used to be three separate queue drains.
        m_mean, m_min, m_max = (
            torch.stack(
                [
                    model.arcface_head.margins.mean(),
                    model.arcface_head.margins.min(),
                    model.arcface_head.margins.max(),
                ]
            )
            .cpu()
            .tolist()
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
                "sched/subcentre_tau": tau_now,
                "margin/mean": float(m_mean),
                "margin/min": float(m_min),
                "margin/max": float(m_max),
            },
            step=ep,
        )

        if plan is not None and plan.empty_cache_interval and ep % plan.empty_cache_interval == 0:
            release_memory(device)

        # Broadcast so every rank stops on the same epoch — see stage 1.
        if dist.broadcast_object(no_improve >= cfg.stage2.patience):
            trk.log_message(
                f"{tag}Early stopping ({no_improve} epochs without improvement).", level="warn"
            )
            break

    trk.progress_stop("stage2")
    return best_f1
