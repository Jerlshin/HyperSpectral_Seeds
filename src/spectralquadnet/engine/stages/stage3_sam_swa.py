"""Stage 3 — Sharpness-Aware Minimisation with greedy SWA snapshotting.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`run_stage3_swa`                 2508-2617
=====================================  ==============

Orchestration only; the single declared deviation is the mechanical
``CONFIG[...]`` → ``cfg.stage3.*`` / ``cfg.stage2.dropout`` / ``cfg.weight_decay``
rewrite, which also gives ``_wd_groups``, ``train_one_epoch_sam`` and
``save_ckpt`` their leading ``cfg`` argument. The nested ``_s3_margin`` schedule
stays nested — only ``phase_aware_lr`` was lifted into ``optim/schedulers.py``,
per §2's tree.

Three hardcoded baseline literals are preserved deliberately (§6 forbids
promoting them to config as part of a mechanical move): ``FocalLoss(gamma=1.0)``,
the SupCon/ProtoNCE weights ``0.02``/``0.01``, and the ``0.98`` greedy-acceptance
factor.

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
) -> float:
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

    w = 66
    print(f"\n{'═' * w}")
    print(f"  Stage 3 — SAM + Greedy SWA  [{cfg.stage3.epochs} epochs]")
    print(f"{'═' * w}")
    print(
        f"  SAM ρ={cfg.stage3.sam_rho}  Cycle={cfg.stage3.cycle_len} ep  "
        f"Peak LR={cfg.stage3.swa_lr:.0e}  aux_w={aux_w_s3}"
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
                snap_info = f"  ★ snap {n_snap}"
            else:
                n_rejected += 1
                snap_info = f"  ✗ rejected (F1 {f1_live:.3f} < {best_live_f1 * 0.98:.3f})"

        print(
            f"Ep {ep:03d}/{cfg.stage3.epochs} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
            f"F1 {f1_live:.3f}  Acc {acc_live:.1%} │ LR {lr_now:.2e}{snap_info}"
        )

    print(f"\nUpdating BN stats ({n_snap} accepted, {n_rejected} rejected) ...")
    if swa_state is None:
        print("[WARN] No snapshots accepted — using final live model.")
        swa_state = copy.deepcopy(model.state_dict())

    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state)
    swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)
    f1_swa, acc_swa = evaluate(swa_model, val_ldr, device)
    print(f"SWA val: F1={f1_swa:.3f}  Acc={acc_swa:.1%}")

    ema.shadow.load_state_dict(swa_model.state_dict())
    ema.shadow.use_arcface(True)

    note = ""
    if f1_swa <= prev_best_f1:
        note = "val_f1 did not beat Stage 2; Stage 2 ckpt preferred for eval"
        print(f"Stage 3 F1 {f1_swa:.3f} ≤ Stage 2 best {prev_best_f1:.3f} — Stage 2 preferred.")
    else:
        print(f"Stage 3 F1 {f1_swa:.3f} > Stage 2 best {prev_best_f1:.3f} → saving.")

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
