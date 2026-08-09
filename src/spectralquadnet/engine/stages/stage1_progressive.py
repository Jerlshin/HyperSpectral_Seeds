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

Since HD-1 (T2-10) this stage trains the **same** sub-centre cosine head as
Stages 2-3, at ``cfg.stage1.arcface_m = 0``. There is no head to select and
none to freeze; the transition to Stage 2 turns a margin on and changes
nothing else, which is what §2.4.6's six-way discontinuity was about.

Since P-5 (T4-5) the stage reads **two** evaluation loaders and keeps their
jobs apart: ``val_ldr`` decides which epoch is checkpointed, and ``calib_ldr``
is where the CDWS weights and the Phase-3 oversampling weights are measured.
Both were ``val_ldr`` before, which is C-9: the weights were fitted on the same
1,294 patches that then chose the checkpoint, and the val→test gap 0-C measured
(+0.011 on the two checkpoints selected that way, −0.003 on the one that was
not) is what that costs.
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
from spectralquadnet.engine.diagnostics import (
    compute_class_difficulty,
    epoch_tag,
    should_render_details,
)
from spectralquadnet.engine.evaluate import evaluate
from spectralquadnet.engine.train_epoch import train_one_epoch
from spectralquadnet.losses.auxiliary import GradNormAuxWeights, _aux_loss_weight
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import build_optimizer_s1
from spectralquadnet.optim.schedulers import phase_aware_lr, subcentre_tau
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.device import RuntimePlan, release_memory
from spectralquadnet.utils.distributed import DistContext

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
    calib_ldr: DataLoader[Any] | None = None,
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
    train_module: nn.Module | None = None,
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
        model: Model to train; the shared cosine head runs at zero margin.
        ema: EMA shadow, re-initialised at the Phase 2 and Phase 3 boundaries.
        loaders_by_phase: Loaders for phases 1 and 2, keyed by phase number;
            the Phase-3 loader is built internally once Phase 3 begins.
        val_ldr: Validation loader for per-epoch evaluation.
        device: Device to train on.
        best_ckpt: Path to write the best-F1 checkpoint to.
        tracker: Experiment tracker for progress reporting; ``None`` logs nowhere.
        calib_ldr: Calibration loader (P-5 / T4-5). Every *fitted* quantity —
            the CDWS weights and the Phase-3 oversampling weights — is measured
            on this split, so it is never measured on the split that also
            selects the checkpoint. ``None`` falls back to ``val_ldr``, the
            pre-Tier-4 behaviour.

    Returns:
        Best validation macro-F1 (across the live model and its EMA shadow)
        seen during the run.
    """
    trk = tracker if tracker is not None else NullTracker()
    dist = dist or DistContext()
    # `model` is always the eager `SpectralQuadNet`: it owns the attributes
    # (`arcface_head`, `set_dropout`) and the unprefixed parameter names the
    # optimiser split and the checkpoint schema are built on. `train_module` is
    # what the forward/backward actually goes through — the same module under a
    # DDP and/or `torch.compile` wrapper, or `model` itself single-process.
    fwd = train_module if train_module is not None else model
    # One name, one meaning: `val_ldr` selects, `fit_ldr` fits.
    fit_ldr = calib_ldr if calib_ldr is not None else val_ldr
    fit_split = "calib" if calib_ldr is not None else "val"
    detail_every = plan.diagnostics_interval if plan is not None else 1

    ep_total = cfg.stage1.epochs
    p1_end = int(ep_total * cfg.stage1.phase1_frac)
    p2_end = int(ep_total * (cfg.stage1.phase1_frac + cfg.stage1.phase2_frac))

    optimizer = build_optimizer_s1(
        cfg, model, cfg.stage1.max_lr, fused=plan.fused_optimizer if plan else False
    )

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

    # OP-2 / T2-6: the per-branch auxiliary balance, measured rather than asserted.
    aux_weights = GradNormAuxWeights(alpha=cfg.aux_gradnorm_alpha)

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
            f"Branch aux weights: GradNorm α={cfg.aux_gradnorm_alpha} from A/B×2.0 C/D×1.0  |  "
            f"Head: cosine (m={cfg.stage1.arcface_m})  τ: "
            f"{cfg.model.subcenter_tau_init} → {cfg.model.subcenter_tau_final}",
            f"Fitted on: {fit_split} ({len(fit_ldr.dataset)} patches)  |  "  # type: ignore[arg-type]
            f"Selected on: val ({len(val_ldr.dataset)} patches)",  # type: ignore[arg-type]
        ],
    )

    trk.progress_start("stage1", ep_total, "Stage 1")
    for ep in range(1, ep_total + 1):
        detail = should_render_details(ep, detail_every)
        # ── Phase assignment ──────────────────────────────────────────
        if ep <= p1_end:
            phase = 1
        elif ep <= p2_end:
            phase = 2
        else:
            phase = 3

        tag = epoch_tag(ep, ep_total)

        # ── EMA re-init at phase boundaries ───────────────────────────
        # Both boundaries are bracketed by a memory sweep: the re-init briefly
        # holds a second copy of every parameter, the incoming phase's loader
        # brings its own prefetch queue, and cuDNN re-autotunes for the new
        # epoch's first shapes. That peak landing on a pool still fragmented by
        # the outgoing phase is what OOMs at ep 181 — see `release_memory`.
        if phase == 2 and not ema_reinited[0] and cfg.stage1.ema_reinit_phases:
            release_memory(device)
            ema.reinit_from(model)
            trk.log_message(f"{tag}EMA re-init at Phase 2")
            ema_reinited[0] = True
            release_memory(device)

        if phase == 3 and not ema_reinited[1] and cfg.stage1.ema_reinit_phases:
            release_memory(device)
            ema.reinit_from(model)
            # Dropout is raised for Phase 3 to fight memorisation on the
            # lighter-augmentation, hard-class-oversampled data it trains on.
            p3_drop = cfg.stage1.p3_dropout
            model.set_dropout(p3_drop)
            ema.set_dropout(p3_drop)
            trk.log_message(f"{tag}EMA re-init at Phase 3  dropout→{p3_drop}")
            ema_reinited[1] = True
            release_memory(device)

        # ── Phase 3 loader construction (once, at first Phase-3 epoch) ─
        if phase == 3 and phase3_ldr is None:
            trk.log_message(
                f"{tag}Phase 2→3 boundary: measuring per-class F1 "
                f"for oversampling on {fit_split} ..."
            )
            class_f1_phase2, _ = compute_class_difficulty(
                cfg,
                ema.shadow,
                fit_ldr,
                device,
                "Phase2→3",
                tracker=trk,
                step=ep,
                total_steps=ep_total,
            )
            phase3_ldr = build_phase3_loader(
                cfg,
                store,
                train_ds=loaders_by_phase[3].dataset,
                class_f1=class_f1_phase2,
                plan=plan,
                dist=dist,
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

        # HD-2(i): every sub-centre keeps receiving gradient while tau is large.
        tau_now = subcentre_tau(
            ep, ep_total, cfg.model.subcenter_tau_init, cfg.model.subcenter_tau_final
        )
        model.set_subcentre_tau(tau_now)
        ema.shadow.set_subcentre_tau(tau_now)

        # Phase 3 uses SupCon + ProtoNCE for better embedding geometry
        sc_w_now = cfg.stage1.p3_supcon_weight if phase == 3 else 0.0
        pt_w_now = cfg.stage1.p3_proto_weight if phase == 3 else 0.0

        tl, ta = train_one_epoch(
            cfg,
            fwd,
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
            arc_m=cfg.stage1.arcface_m,
            current_ep=ep,
            total_ep=ep_total,
            tracker=trk,
            aux_weights=aux_weights,
        )
        scheduler.step()

        # ── Evaluate both live model and EMA ──────────────────────────
        # Two full passes over the validation split, each allocating its own
        # activations on top of whatever the epoch's backward left cached.
        # Sweeping either side keeps the evaluation from having to grow the pool
        # and hands the next epoch a clean one.
        release_memory(device)
        f1_live, acc_live = evaluate(model, val_ldr, device, dist)
        f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device, dist)
        release_memory(device)
        best_ep_f1 = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        # Which of the two `best_ep_f1` came from. `final_eval` evaluates that
        # one, so the reported test number describes the model that was
        # actually selected (T1-8 / §2.1.4). Ties keep the EMA shadow, the
        # historical choice.
        best_ep_source = "ema" if f1_ema >= f1_live else "live"
        lr_now = optimizer.param_groups[0]["lr"]
        aux_w_now = _aux_loss_weight(cfg, ep, ep_total)
        saved = ""

        # ── Checkpoint on F1 improvement ─────────────────────────────
        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1, _cdws = compute_class_difficulty(
                cfg,
                ema.shadow,
                fit_ldr,
                device,
                "S1",
                tracker=trk,
                step=ep,
                detail=detail,
                total_steps=ep_total,
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
                dist=dist,
                best_source=best_ep_source,
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
                "sched/subcentre_tau": tau_now,
                "sched/phase": float(phase),
            },
            step=ep,
        )

        # Extra periodic sweep on top of the per-epoch one above. Off by
        # default; the stage boundary in `train.py` always sweeps.
        if plan is not None and plan.empty_cache_interval and ep % plan.empty_cache_interval == 0:
            release_memory(device)

        # Under DDP every rank must reach the same verdict, or the ones that
        # kept going would block forever in the next epoch's all-reduce.
        if dist.broadcast_object(no_improve >= cfg.stage1.patience):
            trk.log_message(
                f"{tag}Early stopping ({no_improve} epochs without improvement).", level="warn"
            )
            break

    trk.progress_stop("stage1")
    return best_f1
