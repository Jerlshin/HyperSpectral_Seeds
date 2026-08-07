"""Stage 3 — Sharpness-Aware Minimisation with greedy SWA snapshotting.

Orchestration only. ``FocalLoss(gamma=1.0)``, the SupCon/ProtoNCE weights
0.02/0.01, and the 0.98 greedy-acceptance factor are fixed constants for
this stage rather than config fields.

The SWA average is *greedy*: a cycle-end snapshot only joins the running mean if
its live F1 is within 2% of the best seen so far, otherwise it is rejected and
counted. ``update_bn_stats`` then re-estimates BatchNorm statistics for the
averaged weights — without it the averaged model's BN buffers correspond to no
model that ever existed. The stage always saves, even when its F1 does not beat
Stage 2's; the ``note`` field records that, and ``_pick_best_checkpoint`` is what
ultimately decides which checkpoint the final evaluation loads.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from spectralquadnet.engine.checkpoint import save_ckpt, update_bn_stats
from spectralquadnet.engine.evaluate import evaluate
from spectralquadnet.engine.train_epoch import train_one_epoch_sam
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import _wd_groups
from spectralquadnet.optim.sam import SAM
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet


def run_stage3_swa(
    cfg: ExperimentConfig | Any,
    model: SpectralQuadNet,
    ema: ModelEMA,
    train_ldr: DataLoader[Any],
    val_ldr: DataLoader[Any],
    device: torch.device,
    best_ckpt: str,
    prev_best_f1: float,
    tracker: ExperimentTracker | None = None,
) -> float:
    """Run Stage 3's SAM training with greedy SWA snapshotting and return the SWA validation F1.

    Trains under SAM with cyclic LR restarts; at the end of every cycle, the
    live weights are greedily averaged into a running SWA state if their F1
    is within 2% of the best seen so far. After training, BatchNorm
    statistics are re-estimated for the averaged weights, the SWA model is
    evaluated, and its checkpoint is saved regardless of whether it beats
    ``prev_best_f1`` (with a ``note`` recording that comparison).

    Args:
        cfg: Composed experiment config.
        model: Model to train; switched onto the ArcFace head with branch
            dropout disabled.
        ema: EMA shadow; its state is overwritten with the final SWA weights
            at the end of the stage (not used for live EMA averaging here).
        train_ldr: Stage-3 training loader.
        val_ldr: Validation loader for per-epoch evaluation.
        device: Device to train on.
        best_ckpt: Path to write the Stage-3 checkpoint to.
        prev_best_f1: Stage 2's best validation F1, used only for the saved
            ``note`` comparison — Stage 3 always saves its own checkpoint.
        tracker: Experiment tracker for progress reporting; ``None`` logs nowhere.

    Returns:
        The SWA model's validation macro-F1.
    """
    trk = tracker if tracker is not None else NullTracker()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.disable()  # type: ignore[no-untyped-call]

    model.set_dropout(cfg.stage2.dropout)
    model.branch_drop_prob = 0.0
    ema.shadow.branch_drop_prob = 0.0
    model.use_arcface(True)
    ema.shadow.use_arcface(True)

    params = list(_wd_groups(cfg, model.named_parameters(), cfg.stage3.swa_lr))
    sam = SAM(
        params,
        optim.AdamW,
        rho=cfg.stage3.sam_rho,
        lr=cfg.stage3.swa_lr,
        weight_decay=cfg.weight_decay,
    )

    focal_s3 = FocalLoss(gamma=1.0)
    supcon_s3 = SupConLoss(temperature=0.10)
    proto_s3 = ProtoNCELoss(temperature=0.10)

    swa_state = None
    n_snap = 0
    n_rejected = 0
    best_live_f1 = 0.0
    aux_w_s3 = cfg.stage3.aux_loss_weight

    trk.banner(
        f"Stage 3 — SAM + Greedy SWA  [{cfg.stage3.epochs} epochs]",
        [
            f"SAM ρ={cfg.stage3.sam_rho}  Cycle={cfg.stage3.cycle_len} ep  "
            f"Peak LR={cfg.stage3.swa_lr:.0e}  aux_w={aux_w_s3}"
        ],
    )

    def _s3_margin(ep: int) -> float:
        return 0.25 + 0.05 * math.cos(math.pi * ep / cfg.stage3.epochs)

    for ep in range(1, cfg.stage3.epochs + 1):
        cycle_ep = (ep - 1) % cfg.stage3.cycle_len
        lr_now = cfg.stage3.swa_lr * (
            0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * cycle_ep / cfg.stage3.cycle_len))
        )
        for pg in sam.param_groups:
            pg["lr"] = lr_now

        tl, ta = train_one_epoch_sam(
            cfg,
            model,
            train_ldr,
            sam,
            focal_s3,
            device,
            supcon=supcon_s3,
            supcon_weight=0.02,
            proto=proto_s3,
            proto_weight=0.01,
            arc_m=_s3_margin(ep),
            aux_weight=aux_w_s3,
            current_ep=ep,
            tracker=trk,
        )

        f1_live, acc_live = evaluate(model, val_ldr, device)
        best_live_f1 = max(best_live_f1, f1_live)
        snap_info = ""

        if ep % cfg.stage3.cycle_len == 0:
            if not cfg.stage3.greedy or f1_live >= best_live_f1 * 0.98:
                n_snap += 1
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = copy.deepcopy(sd)
                else:
                    beta = 1.0 / float(n_snap)
                    for k in swa_state:
                        if swa_state[k].is_floating_point():
                            swa_state[k].mul_(1.0 - beta).add_(sd[k], alpha=beta)
                        else:
                            swa_state[k].copy_(sd[k])
                snap_info = f"★ snap {n_snap}"
            else:
                n_rejected += 1
                snap_info = f"✗ rejected (F1 {f1_live:.3f} < {best_live_f1 * 0.98:.3f})"

        trk.log_row(
            "stage3",
            {
                "Ep": f"{ep:03d}/{cfg.stage3.epochs}",
                "Loss": f"{tl:.4f}",
                "Tr": f"{ta:.1%}",
                "F1": f"{f1_live:.3f}",
                "Acc": f"{acc_live:.1%}",
                "LR": f"{lr_now:.2e}",
                "swa": snap_info,
            },
            step=ep,
        )
        trk.log_scalars(
            {
                "train/loss": tl,
                "train/acc": ta,
                "val/f1_live": f1_live,
                "val/acc_live": acc_live,
                "val/f1_best": best_live_f1,
                "sched/lr": lr_now,
                "sched/arcface_margin": _s3_margin(ep),
                "swa/n_snapshots": float(n_snap),
                "swa/n_rejected": float(n_rejected),
            },
            step=ep,
        )

    trk.log_message(f"Updating BN stats ({n_snap} accepted, {n_rejected} rejected) ...")
    if swa_state is None:
        trk.log_message("No snapshots accepted — using final live model.", level="warn")
        swa_state = copy.deepcopy(model.state_dict())

    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state)
    swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)
    f1_swa, acc_swa = evaluate(swa_model, val_ldr, device)
    trk.log_message(f"SWA val: F1={f1_swa:.3f}  Acc={acc_swa:.1%}", level="plain")
    trk.log_scalars({"swa/f1": f1_swa, "swa/acc": acc_swa}, step=cfg.stage3.epochs)

    ema.shadow.load_state_dict(swa_model.state_dict())
    ema.shadow.use_arcface(True)

    note = ""
    if f1_swa <= prev_best_f1:
        note = "val_f1 did not beat Stage 2; Stage 2 ckpt preferred for eval"
        trk.log_message(
            f"Stage 3 F1 {f1_swa:.3f} ≤ Stage 2 best {prev_best_f1:.3f} — Stage 2 preferred.",
            level="plain",
        )
    else:
        trk.log_message(
            f"Stage 3 F1 {f1_swa:.3f} > Stage 2 best {prev_best_f1:.3f} → saving.",
            level="plain",
        )

    save_ckpt(
        cfg,
        best_ckpt,
        cfg.stage3.epochs,
        "Stage 3",
        swa_model,
        ema,
        val_f1=f1_swa,
        val_acc=acc_swa,
        swa_n_snapshots=n_snap,
        swa_n_rejected=n_rejected,
        **({"note": note} if note else {}),
    )
    return f1_swa
