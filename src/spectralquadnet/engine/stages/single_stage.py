"""The collapsed curriculum — one stage, one objective, one schedule (IC-11).

CHANGES §17. Three stages became one because the measured return on the other
two was +0.005 macro-F1 for 65% of an 18.7-hour wall clock — 6.5 samples of a
1,294-sample split, against a ±0.020 sampling CI and an estimated +0.042 of
selection bias. Under any accounting that is negative expected value.

What this loop is
─────────────────
::

    epochs 1-150, early stop patience 25 on CALIB macro-F1

    loss    CE(label smoothing 0.10 -> 0.04)
            + model.aux_head_weight x aux CE on the spatial path
            + mixup(0.35) while ep <= single.mixup_epochs
    margin  0 until margin_warmup_start, cosine to single.arcface_m by _end
    sampler plain shuffled (classes are 91-96/class — already balanced)
    aug     one profile throughout + D4 + same-class CutMix
    optim   AdamW, warm-up then one cosine decay, per-group clip at 5.0
    amp     bf16 throughout, fp32 confined to the head and any similarity matrix
    ema     d_max 0.999, never re-initialised
    select  max(F1_live, F1_ema) on the split cfg.evaluation.select_split names

What it deliberately does not do
────────────────────────────────
No EMA re-initialisation (the audited version did it twice, at phase
boundaries, and it was never isolated). No dropout schedule (0.15/0.25/0.10 was
untested). No hard-class oversampling, no CDWS, no sub-centre temperature
anneal, no SGDR restarts, no GradNorm — CHANGES §6 classifies all of them
UNJUSTIFIED or REDUNDANT, and eight of them were built to move the same five
hard classes, which never moved under any of them (§10.5).

Optional Phase B
────────────────
``single.supcon_epochs > 0`` appends a class-balanced ``CE + w·SupCon`` tail.
Off by default: A6 has to show SupCon beats plain CE by more than run-to-run
variance *with the sampler controlled*, because the audited design changed the
loss, the augmentation profile and the sampler at the same epoch and therefore
measured none of them.

Selection
─────────
``fit_ldr``/``select_ldr`` are separated on purpose and both are usually
``calib``. The audited run fitted 270+ parameters on ``val``, selected the
checkpoint on ``val`` and reported ``val`` — see :class:`EvaluationConfig`.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from spectralquadnet.engine.checkpoint import save_ckpt
from spectralquadnet.engine.diagnostics import (
    compute_class_difficulty,
    epoch_tag,
    should_render_details,
)
from spectralquadnet.engine.evaluate import evaluate
from spectralquadnet.engine.train_epoch import train_one_epoch
from spectralquadnet.losses.contrastive import SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.control import set_dropout as set_module_dropout
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import build_optimizer_s1
from spectralquadnet.optim.schedulers import (
    single_stage_label_smoothing,
    single_stage_lr,
    single_stage_margin,
    single_stage_uses_mixup,
)
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.device import (
    RuntimePlan,
    describe_amp,
    make_grad_scaler,
    release_memory,
    resolve_amp_dtype,
)
from spectralquadnet.utils.distributed import DistContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: The stage label written into the checkpoint bundle and its sidecar. The
#: trailing digit is load-bearing: ``save_ckpt`` derives the sidecar filename
#: from it, and ``latest_completed_stage`` probes ``stage1_meta.json``. A
#: single-stage run therefore occupies the stage-1 slot, which is what makes
#: auto-resume and ``_pick_best_checkpoint`` work unchanged.
STAGE_LABEL = "Stage 1"


def run_single_stage(
    cfg: ExperimentConfig | Any,
    model: nn.Module,
    ema: ModelEMA,
    train_ldr: DataLoader[Any],
    select_ldr: DataLoader[Any],
    device: torch.device,
    best_ckpt: str,
    tracker: ExperimentTracker | None = None,
    fit_ldr: DataLoader[Any] | None = None,
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
    train_module: nn.Module | None = None,
    supcon_ldr: DataLoader[Any] | None = None,
) -> float:
    """Run the single-stage curriculum and return the best selection macro-F1.

    Args:
        model: The eager model — it owns the attributes (``arcface_head``,
            ``set_dropout``) and the unprefixed parameter names the optimiser
            split and the checkpoint schema are built on.
        train_ldr: Shuffled training loader.
        select_ldr: The split the checkpoint decision reads. Under the shipped
            protocol this is ``calib``; it is **never** the split the run
            reports.
        fit_ldr: Where per-class diagnostics are measured. Defaults to
            ``select_ldr``.
        train_module: What the forward/backward actually goes through — the
            same module under a DDP and/or ``torch.compile`` wrapper.
        supcon_ldr: Class-balanced loader for the optional Phase B tail.
            Required when ``single.supcon_epochs > 0``.

    Returns:
        Best ``max(F1_live, F1_ema)`` on ``select_ldr``.
    """
    trk = tracker if tracker is not None else NullTracker()
    dist = dist or DistContext()
    fwd = train_module if train_module is not None else model
    fit_ldr = fit_ldr if fit_ldr is not None else select_ldr
    detail_every = plan.diagnostics_interval if plan is not None else 1

    main_ep = int(cfg.single.epochs)
    phase_b_ep = max(int(cfg.single.supcon_epochs), 0)
    if phase_b_ep and supcon_ldr is None:
        raise ValueError(
            "single.supcon_epochs > 0 requires a class-balanced loader: SupCon at "
            "tau=0.10 needs several positives per anchor, and a plain shuffled batch of "
            "128 over 90 classes gives most anchors none. Pass `supcon_ldr`."
        )
    ep_total = main_ep + phase_b_ep

    optimizer = build_optimizer_s1(
        cfg, model, cfg.single.max_lr, fused=plan.fused_optimizer if plan else False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, single_stage_lr(cfg))

    amp_dtype = plan.amp_dtype if plan is not None else resolve_amp_dtype(device)
    scaler = make_grad_scaler(amp_dtype, device)

    # One dropout rate for the whole run. The 0.15 -> 0.25 -> 0.10 schedule the
    # three-stage curriculum ran was never isolated from the four other things
    # that changed at the same epochs.
    #
    # Applied through the shared helper rather than `model.set_dropout` so the
    # stage stays architecture-agnostic: it reaches `nn.MultiheadAttention`'s
    # float attribute too, which is the gap IC-14 closed.
    set_module_dropout(model, float(cfg.single.dropout))
    set_module_dropout(ema.shadow, float(cfg.single.dropout))

    supcon = SupConLoss(temperature=float(cfg.single.supcon_temp)) if phase_b_ep else None

    best_f1 = 0.0
    no_improve = 0
    select_split = str(getattr(cfg.evaluation, "select_split", "calib"))

    trk.banner(
        f"Single stage — {ep_total} epochs max  (CHANGES §17)",
        [
            f"Mixup α={cfg.single.mixup} for epochs 1–{cfg.single.mixup_epochs}, off after",
            f"Margin 0 → {cfg.single.arcface_m} over epochs "
            f"{cfg.single.margin_warmup_start}–{cfg.single.margin_warmup_end}  "
            f"(s={cfg.single.arcface_s})",
            f"Label smooth {cfg.single.label_smooth_hi} → {cfg.single.label_smooth_lo}  |  "
            f"focal γ={cfg.single.focal_gamma}  |  aux w={cfg.model.aux_head_weight} (fixed)",
            f"AdamW {cfg.single.max_lr} → {cfg.single.min_lr}, {cfg.single.warmup_ep}-epoch "
            f"warm-up, cosine  |  per-group clip {cfg.grad_clip}",
            f"Selected on: {select_split} ({len(select_ldr.dataset)} patches)  |  "  # type: ignore[arg-type]
            f"Early stop patience {cfg.single.patience}",
            (
                f"Phase B: {phase_b_ep} epochs CE + {cfg.single.supcon_weight}×SupCon "
                f"(τ={cfg.single.supcon_temp}), class-balanced"
                if phase_b_ep
                else "Phase B: off (A6 has not earned it)"
            ),
            f"Precision: {describe_amp(amp_dtype, device)}  |  head always fp32",
        ],
    )

    trk.progress_start("single", ep_total, "Single")
    for ep in range(1, ep_total + 1):
        ep_started = time.perf_counter()
        on_stride = should_render_details(ep, detail_every)
        tag = epoch_tag(ep, ep_total)

        # `supcon_ldr is not None` whenever `in_phase_b` — the constructor
        # raised otherwise — but the test is written so the narrowing is
        # visible rather than argued for in a comment.
        in_phase_b = ep > main_ep and supcon_ldr is not None
        cur_ldr = supcon_ldr if in_phase_b and supcon_ldr is not None else train_ldr
        # Phase B is a *tail*, so it inherits the end-of-schedule margin and
        # label smoothing rather than restarting anything.
        sched_ep = min(ep, main_ep)

        ls_now = single_stage_label_smoothing(sched_ep, cfg)
        margin_now = single_stage_margin(sched_ep, cfg)
        use_mx = (not in_phase_b) and single_stage_uses_mixup(ep, cfg)

        gamma = float(cfg.single.focal_gamma)
        crit: nn.Module = (
            FocalLoss(gamma=gamma, label_smoothing=ls_now)
            if gamma > 0.0
            else nn.CrossEntropyLoss(label_smoothing=ls_now)
        )

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
            mixup_alpha=float(cfg.single.mixup),
            supcon=supcon if in_phase_b else None,
            supcon_weight=float(cfg.single.supcon_weight) if in_phase_b else 0.0,
            # ProtoNCE is gone (IC-14 / CHANGES §7.2): it used in-batch class
            # means as prototypes — an 8-sample estimate of a 256-d unit vector
            # — to pull together the same positives SupCon was already pulling,
            # at the same temperature, on the same normalised embedding. Two
            # losses, one signal, and the higher-variance estimator of the two.
            proto=None,
            proto_weight=0.0,
            accum_steps=int(cfg.single.accum),
            arc_m=margin_now,
            current_ep=ep,
            total_ep=ep_total,
            tracker=trk,
            # IC-5: no GradNorm state is threaded, so the auxiliary weight is
            # exactly `model.aux_head_weight` and `aux_weight/*` is a constant.
            aux_weights=None,
            amp_dtype=amp_dtype if amp_dtype is not None else torch.bfloat16,
        )
        if not in_phase_b:
            scheduler.step()

        release_memory(device)
        f1_live, acc_live = evaluate(model, select_ldr, device, dist)
        f1_ema, acc_ema = evaluate(ema.shadow, select_ldr, device, dist)
        release_memory(device)

        best_ep_f1 = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        best_ep_source = "ema" if f1_ema >= f1_live else "live"
        lr_now = optimizer.param_groups[0]["lr"]
        improved = best_ep_f1 > best_f1

        class_f1_now: dict[int, float] = {}
        cdws_now: dict[int, float] = {}
        if improved or on_stride:
            class_f1_now, cdws_now = compute_class_difficulty(
                cfg,
                ema.shadow,
                fit_ldr,
                device,
                "Single",
                tracker=trk,
                step=ep,
                detail=True,
                total_steps=ep_total,
            )

        if improved:
            best_f1, no_improve = best_ep_f1, 0
            save_ckpt(
                cfg,
                best_ckpt,
                ep,
                STAGE_LABEL,
                model,
                ema,
                val_f1=best_ep_f1,
                val_acc=best_ep_acc,
                dist=dist,
                best_source=best_ep_source,
                class_f1=class_f1_now,
                cdws_weights=cdws_now,
                arcface_init_done=True,
                select_split=select_split,
                pipeline="single",
            )
        else:
            no_improve += 1

        trk.log_row(
            "single",
            {
                "dt": f"{time.perf_counter() - ep_started:.1f}s",
                "Loss": f"{tl:.4f}",
                "Tr": f"{ta:.1%}",
                "F1 live/ema": f"{f1_live:.3f}/{f1_ema:.3f}",
                "Acc live/ema": f"{acc_live:.1%}/{acc_ema:.1%}",
                "Best": f"{best_f1:.3f}",
                "LR": f"{lr_now:.2e}",
                "LS": f"{ls_now:.3f}",
                "m": f"{margin_now:.3f}",
                "mix": "on" if use_mx else "off",
                "Ph": "B" if in_phase_b else "A",
                "ckpt": "✓" if improved else "",
                "stale": "" if improved else f"{no_improve}/{cfg.single.patience}",
            },
            step=ep,
        )
        trk.log_scalars(
            {
                "train/loss": tl,
                "train/acc": ta,
                # Named `val/*` rather than `calib/*` so the series a reader
                # plots is the same one across every pipeline; `select/split`
                # in the run's results JSON records which split it came from.
                "val/f1_live": f1_live,
                "val/acc_live": acc_live,
                "val/f1_ema": f1_ema,
                "val/acc_ema": acc_ema,
                "val/f1_best": best_f1,
                "sched/lr": lr_now,
                "sched/label_smooth": ls_now,
                "sched/arcface_m": margin_now,
                "sched/mixup": float(use_mx),
                "sched/aux_weight": float(cfg.model.aux_head_weight),
            },
            step=ep,
        )

        if plan is not None and plan.empty_cache_interval and ep % plan.empty_cache_interval == 0:
            release_memory(device)

        # Phase B is a fixed-length tail and is not early-stopped: it is 30
        # epochs by construction and stopping it early would confound A6's
        # comparison with a variable budget.
        if not in_phase_b and dist.broadcast_object(no_improve >= int(cfg.single.patience)):
            trk.log_message(
                f"{tag}Early stopping ({no_improve} epochs without improvement).", level="warn"
            )
            break

    trk.progress_stop("single")
    return best_f1
