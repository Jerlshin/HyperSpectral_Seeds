"""Stage 1 — three-phase progressive augmentation with deep supervision.

Orchestration only: every unit of work — the epoch loop, evaluation,
class-difficulty measurement, checkpointing — is a call into ``engine/``,
``losses/`` or ``optim/``. ``phase_aware_lr`` is built as a standalone
factory in ``optim/schedulers.py`` (rather than defined inline here) so the
schedule is independently testable — see ``tests/unit/test_schedulers.py``.

All progress reporting goes through the injected ``tracker``: the banner
block calls :meth:`~spectralquadnet.tracking.base.ExperimentTracker.banner`,
status notices call ``log_message``, and the per-epoch line is a
``log_row``/``log_scalars`` pair.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from spectralquadnet.data.loaders import build_phase3_loader
from spectralquadnet.engine.checkpoint import save_ckpt
from spectralquadnet.engine.diagnostics import compute_class_difficulty
from spectralquadnet.engine.evaluate import evaluate
from spectralquadnet.engine.train_epoch import train_one_epoch
from spectralquadnet.losses.auxiliary import _aux_loss_weight
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import build_optimizer_s1
from spectralquadnet.optim.schedulers import phase_aware_lr
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.data.mmap_store import DataStore
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet


def run_stage1(
    cfg: ExperimentConfig | Any,
    store: DataStore,
    model: SpectralQuadNet,
    ema: ModelEMA,
    loaders_by_phase: dict[int, DataLoader[Any]],
    val_ldr: DataLoader[Any],
    device: torch.device,
    best_ckpt: str,
    tracker: ExperimentTracker | None = None,
) -> float:
    """Run Stage 1's full three-phase curriculum and return the best validation F1.

    Phase 1 (0 → ~25%):   heavy aug  + mixup  + high LS   → explore representation
    Phase 2 (~25 → ~60%): medium aug + mixup  + decay LS  → robustness consolidation
    Phase 3 (~60 → 100%): light aug  + Focal  + NO mixup  → discriminate hard classes
                          └─ uses HardClassOversampledSampler built from Phase 2 F1 scores

    Deep supervision via per-branch AuxiliaryHeads with decaying weight.
    Primary metric: macro-F1 (not accuracy).

    Args:
        cfg: Composed experiment config.
        store: Memory-mapped patch store, needed to build the Phase-3 loader.
        model: Model to train; the linear head is used (ArcFace stays frozen).
        ema: EMA shadow, re-initialised at the Phase 2 and Phase 3 boundaries.
        loaders_by_phase: Loaders for phases 1 and 2, keyed by phase number;
            the Phase-3 loader is built internally once Phase 3 begins.
        val_ldr: Validation loader for per-epoch evaluation.
        device: Device to train on.
        best_ckpt: Path to write the best-F1 checkpoint to.
        tracker: Experiment tracker for progress reporting; ``None`` logs nowhere.

    Returns:
        Best validation macro-F1 (across the live model and its EMA shadow)
        seen during the run.
    """
    trk = tracker if tracker is not None else NullTracker()
    model.use_arcface(False)
    model.unfreeze_head("linear")
    model.freeze_head("arcface")

    ep_total = cfg.stage1.epochs
    p1_end = int(ep_total * cfg.stage1.phase1_frac)
    p2_end = int(ep_total * (cfg.stage1.phase1_frac + cfg.stage1.phase2_frac))

    optimizer = build_optimizer_s1(cfg, model, cfg.stage1.max_lr)

    # Custom Phase-Aware Scheduler — see optim/schedulers.py::phase_aware_lr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, phase_aware_lr(cfg, p1_end, p2_end))

    # `device=` is what makes AMP real: a bare `GradScaler()` binds to CUDA, so on
    # any other accelerator it prints "CUDA is not available. Disabling." and
    # every scale/step call becomes a pass-through. Metal autocasts to fp16,
    # which needs the loss scaling this performs.
    scaler = GradScaler(device=device.type)
    ls_hi = cfg.stage1.label_smooth_hi
    ls_lo = cfg.stage1.label_smooth_lo
    best_f1 = 0.0
    no_improve = 0
    ema_reinited = [False, False]

    # Phase 3 contrastive losses — critical for embedding geometry
    supcon_p3 = SupConLoss(temperature=0.10)
    proto_p3 = ProtoNCELoss(temperature=0.10)

    # Phase 3 loader — built lazily at the Phase 2 → 3 boundary
    phase3_ldr: DataLoader[Any] | None = None
    class_f1_phase2: dict[int, float] = {}

    trk.banner(
        f"Stage 1 — 3-Phase Progressive Augmentation  [{ep_total} epochs max]",
        [
            f"Phase 1: ep 1–{p1_end}         heavy aug + mixup (α={cfg.stage1.mixup})",
            f"Phase 2: ep {p1_end + 1}–{p2_end}       medium aug + mixup",
            f"Phase 3: ep {p2_end + 1}–{ep_total}      very_light aug, Focal + SupCon + ProtoNCE",
            f"Label smooth: {ls_hi} → {ls_lo}  |  Aux w: "
            f"{cfg.stage1.aux_loss_weight_init} → {cfg.stage1.aux_loss_weight_final}",
            f"Oversample: {cfg.stage1.p3_oversample}  "
            f"power={cfg.stage1.p3_oversample_power}  "
            f"hard_thresh={cfg.stage1.p3_hard_f1_thresh}",
            f"P3 contrastive: SupCon(w={cfg.stage1.p3_supcon_weight}) "
            f"ProtoNCE(w={cfg.stage1.p3_proto_weight})",
            "Branch aux weights: A/B×2.0  C/D×1.0  |  Drop probs: A=0 B=0 C=0.30 D=0.20",
        ],
    )

    for ep in range(1, ep_total + 1):
        # ── Phase assignment ──────────────────────────────────────────
        if ep <= p1_end:
            phase = 1
        elif ep <= p2_end:
            phase = 2
        else:
            phase = 3

        # ── EMA re-init at phase boundaries ───────────────────────────
        if phase == 2 and not ema_reinited[0] and cfg.stage1.ema_reinit_phases:
            ema.reinit_from(model)
            trk.log_message(f"EMA re-init at Phase 2 (ep {ep})")
            ema_reinited[0] = True

        if phase == 3 and not ema_reinited[1] and cfg.stage1.ema_reinit_phases:
            ema.reinit_from(model)
            # Dropout is raised for Phase 3 to fight memorisation on the
            # lighter-augmentation, hard-class-oversampled data it trains on.
            p3_drop = cfg.stage1.p3_dropout
            model.set_dropout(p3_drop)
            ema.set_dropout(p3_drop)
            trk.log_message(f"EMA re-init at Phase 3 (ep {ep})  dropout→{p3_drop}")
            ema_reinited[1] = True

        # ── Phase 3 loader construction (once, at first Phase-3 epoch) ─
        if phase == 3 and phase3_ldr is None:
            trk.log_message("Phase 2→3 boundary: measuring per-class F1 for oversampling ...")
            class_f1_phase2, _ = compute_class_difficulty(
                cfg, ema.shadow, val_ldr, device, "Phase2→3", tracker=trk, step=ep
            )
            phase3_ldr = build_phase3_loader(
                cfg,
                store,
                train_ds=loaders_by_phase[3].dataset,
                class_f1=class_f1_phase2,
            )

        # ── Select active loader ──────────────────────────────────────
        if phase == 1:
            cur_ldr = loaders_by_phase[1]
        elif phase == 2:
            cur_ldr = loaders_by_phase[2]
        else:
            # Never None here: the block above builds it on the first Phase-3 epoch.
            cur_ldr = phase3_ldr  # type: ignore[assignment]  # oversampled Phase-3 loader

        # ── Loss function ─────────────────────────────────────────────
        t = (ep - 1) / max(ep_total - 1, 1)
        ls_now = ls_hi * (1 - t) + ls_lo * t

        # Phase 3: Focal loss — sharpens focus on hard samples
        # Phases 1–2: standard CrossEntropy + label smoothing
        if phase == 3:
            crit: nn.Module = FocalLoss(gamma=cfg.stage1.focal_gamma, label_smoothing=ls_now)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=ls_now)

        use_mx = phase != 3  # no Mixup in Phase 3

        # Phase 3 uses SupCon + ProtoNCE for better embedding geometry
        sc_w_now = cfg.stage1.p3_supcon_weight if phase == 3 else 0.0
        pt_w_now = cfg.stage1.p3_proto_weight if phase == 3 else 0.0

        tl, ta = train_one_epoch(
            cfg,
            model,
            cur_ldr,
            optimizer,
            crit,
            scaler,
            ema,
            device,
            scheduler=None,
            use_mixup=use_mx,
            mixup_alpha=cfg.stage1.mixup,
            supcon=supcon_p3 if phase == 3 else None,
            supcon_weight=sc_w_now,
            proto=proto_p3 if phase == 3 else None,
            proto_weight=pt_w_now,
            accum_steps=cfg.stage1.accum,
            current_ep=ep,
            total_ep=ep_total,
            tracker=trk,
        )
        scheduler.step()

        # ── Evaluate both live model and EMA ──────────────────────────
        f1_live, acc_live = evaluate(model, val_ldr, device)
        f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1 = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        lr_now = optimizer.param_groups[0]["lr"]
        aux_w_now = _aux_loss_weight(cfg, ep, ep_total)
        saved = ""

        # ── Checkpoint on F1 improvement ─────────────────────────────
        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1, _cdws = compute_class_difficulty(
                cfg, ema.shadow, val_ldr, device, "S1", tracker=trk, step=ep
            )
            save_ckpt(
                cfg,
                best_ckpt,
                ep,
                "Stage 1",
                model,
                ema,
                val_f1=best_ep_f1,
                val_acc=best_ep_acc,
                class_f1=_cf1,
                cdws_weights=_cdws,
                arcface_init_done=False,
                phase3_class_f1=class_f1_phase2,  # store phase-2 hard-class info
            )
            saved = "✓"  # rendered as its own row cell now, so no padding
        else:
            no_improve += 1

        trk.log_row(
            "stage1",
            {
                "Ep": f"{ep:03d}/{ep_total}",
                "Loss": f"{tl:.4f}",
                "Tr": f"{ta:.1%}",
                "F1 live/ema": f"{f1_live:.3f}/{f1_ema:.3f}",
                "Acc live/ema": f"{acc_live:.1%}/{acc_ema:.1%}",
                "LR": f"{lr_now:.2e}",
                "LS": f"{ls_now:.3f}",
                "auxW": f"{aux_w_now:.2f}",
                "Ph": f"P{phase}",
                "ckpt": saved,
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
                "sched/lr": lr_now,
                "sched/label_smooth": ls_now,
                "sched/aux_weight": aux_w_now,
                "sched/phase": float(phase),
            },
            step=ep,
        )

        if no_improve >= cfg.stage1.patience:
            trk.log_message(f"Early stopping at epoch {ep}.", level="warn")
            break

    model.unfreeze_head("arcface")
    return best_f1
