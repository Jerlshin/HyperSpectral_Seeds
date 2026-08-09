"""Stage 3 — Sharpness-Aware Minimisation with greedy SWA snapshotting.

Orchestration only. ``FocalLoss(gamma=1.0)`` and the SupCon/ProtoNCE weights
0.02/0.01 are fixed constants for this stage rather than config fields.

Four Tier-2 items reshape the stage; each fixes a property SWA needs and the
original did not have.

* **T2-1 (OP-4.2/4.3).** The stage keeps Stage 2's per-class margin *vector*
  and scales all of it by ``kappa: 1.0 -> cfg.stage3.margin_kappa_final``,
  stepping **only at cycle boundaries**. It used to pass a scalar
  ``arc_m = 0.25 + 0.05 cos(...)``, which discarded the calibration at entry
  (C-7a) and changed the objective every epoch, so SWA averaged a translating
  trajectory rather than an oscillating one (C-7b).
* **T2-2 (OP-4.4).** Greedy acceptance now evaluates the **candidate average**
  ``(n-1)/n * theta_bar + 1/n * theta_n`` against the current average, not the
  snapshot against the best live F1 ever seen. The old test was
  ``f1_live >= best_live_f1 * 0.98`` evaluated immediately after
  ``best_live_f1 = max(best_live_f1, f1_live)`` — provably never false, so
  ``swa/n_rejected`` was 0 by construction and "greedy" was a no-op (C-7c).
* **T2-3 (OP-4.5).** The first ``cfg.stage3.swa_warmup_cycles`` cycles are
  discarded before any candidate is considered, keeping Adam's
  ``1/(1-beta2)``-step second-moment transient out of the average (C-7d).
* **T2-4 (OP-5).** SAM runs in ASAM mode, so the ``rho``-budget is normalised
  by ``|theta|`` instead of being handed to whichever parameters have the
  largest raw gradient — which, at ``s = 48``, was the head.

``update_bn_stats`` then re-estimates BatchNorm statistics for the averaged
weights over a **shuffled, unweighted** rebuild of the training loader, because
the stage's own loader is CDWS-weighted and BN buffers estimated under a
re-weighted class prior are applied at test under the natural one
(T1-5 / §2.5.6(ii)). The stage always saves, even when its F1 does not beat
Stage 2's; the ``note`` field records that, and ``_pick_best_checkpoint``
decides which checkpoint the final evaluation loads.

**Two averaging schemes run side by side.** The EMA shadow is maintained
throughout, scored every epoch as ``val/f1_ema`` (as Stages 1–2 do), and
compared against the SWA average at the end; the better of the two is what the
bundle's ``ema`` slot receives, and ``best_source`` records which won
(T1-9 / §2.5.7 C-7f).
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from spectralquadnet.data.loaders import build_natural_prior_loader
from spectralquadnet.engine.checkpoint import save_ckpt, update_bn_stats
from spectralquadnet.engine.diagnostics import epoch_tag
from spectralquadnet.engine.evaluate import evaluate
from spectralquadnet.engine.train_epoch import train_one_epoch_sam
from spectralquadnet.losses.auxiliary import GradNormAuxWeights
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import _wd_groups, adamw_kwargs
from spectralquadnet.optim.sam import SAM
from spectralquadnet.optim.schedulers import stage3_margin_kappa
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.device import (
    RuntimePlan,
    release_memory,
    reset_compilation,
    unwrap_model,
)
from spectralquadnet.utils.distributed import DistContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet

StateDict = dict[str, torch.Tensor]


def _blend(average: StateDict, snapshot: StateDict, n: int) -> StateDict:
    """The candidate average ``(n-1)/n * average + 1/n * snapshot``, as a new dict.

    A *new* dict, not an in-place update: OP-4.4's whole point is that the
    candidate has to be scored **before** it is allowed to become the running
    average. Integer buffers (``num_batches_tracked``) take the snapshot's
    value outright, since their mean is not one of them.
    """
    beta = 1.0 / float(n)
    out: StateDict = {}
    for key, value in average.items():
        if value.is_floating_point():
            out[key] = value * (1.0 - beta) + snapshot[key] * beta
        else:
            out[key] = snapshot[key].clone()
    return out


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
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
    train_module: nn.Module | None = None,
) -> float:
    """Run Stage 3's SAM training with greedy SWA snapshotting and return its best validation F1.

    Trains under ASAM with cyclic LR restarts. At the end of every cycle past
    the warm-up, the candidate SWA average is formed and **evaluated**; it
    replaces the running average only if it scores better than the average it
    would replace. After training, BatchNorm statistics are re-estimated for
    the accepted average over a natural-prior loader, the SWA model and the EMA
    shadow are both evaluated, and the checkpoint is saved regardless of
    whether it beats ``prev_best_f1`` (with a ``note`` recording that
    comparison).

    Args:
        cfg: Composed experiment config.
        model: Model to train, with branch dropout disabled.
        ema: EMA shadow, re-initialised from ``model`` at stage entry and
            updated every optimiser step. At the end it holds whichever of the
            EMA average and the SWA average scored higher on validation.
        train_ldr: Stage-3 training loader.
        val_ldr: Validation loader for per-epoch evaluation.
        device: Device to train on.
        best_ckpt: Path to write the Stage-3 checkpoint to.
        prev_best_f1: Stage 2's best validation F1, used only for the saved
            ``note`` comparison — Stage 3 always saves its own checkpoint.
        tracker: Experiment tracker for progress reporting; ``None`` logs nowhere.

    Returns:
        The better of the SWA and EMA validation macro-F1 scores.
    """
    trk = tracker if tracker is not None else NullTracker()
    dist = dist or DistContext()
    # See `run_stage1`: eager module for attributes and grouping, wrapped
    # module for the forward.
    fwd = train_module if train_module is not None else model
    # SAM's double backward and the per-cycle margin vector both invalidate a
    # compiled graph's guards, so a compiled Stage 3 spends its time
    # recompiling. Dropped rather than left to thrash.
    reset_compilation()

    model.set_dropout(cfg.stage2.dropout)
    # Now that `branch_drop_prob` reaches the forward pass (T1-6), this actually
    # disables branch masking rather than being stored and ignored.
    model.branch_drop_prob = 0.0
    # The sub-centre assignment is already hard by Stage 3: annealing it again
    # here would be a third schedule for SWA's objective to chase.
    model.set_subcentre_tau(cfg.model.subcenter_tau_final)

    # The shadow tracks *this* stage's trajectory, so it starts from the weights
    # Stage 3 starts from rather than continuing Stage 2's average. Bracketed
    # like Stage 1's phase boundaries — see `run_stage1`.
    release_memory(device)
    ema.reinit_from(model)
    ema.set_dropout(cfg.stage2.dropout)
    ema.shadow.branch_drop_prob = 0.0
    ema.shadow.set_subcentre_tau(cfg.model.subcenter_tau_final)
    release_memory(device)

    # T2-1: Stage 2's calibrated margin vector is the thing being annealed, so
    # it is captured before the first cycle scales it.
    base_margins = model.arcface_head.margins.detach().clone()

    params = list(_wd_groups(cfg, model.named_parameters(), cfg.stage3.swa_lr))
    sam = SAM(
        params,
        optim.AdamW,
        rho=cfg.stage3.sam_rho,
        adaptive=cfg.stage3.sam_adaptive,
        lr=cfg.stage3.swa_lr,
        weight_decay=cfg.weight_decay,
        **adamw_kwargs(plan.fused_optimizer if plan else False, params),
    )

    focal_s3 = FocalLoss(gamma=1.0)
    supcon_s3 = SupConLoss(temperature=0.10)
    proto_s3 = ProtoNCELoss(temperature=0.10)
    aux_weights = GradNormAuxWeights(alpha=cfg.aux_gradnorm_alpha)

    swa_state: StateDict | None = None
    n_snap = 0
    n_rejected = 0
    first_accepted_cycle = 0
    f1_swa_running = 0.0
    best_live_f1 = 0.0
    aux_w_s3 = cfg.stage3.aux_loss_weight
    warmup_cycles = int(cfg.stage3.swa_warmup_cycles)

    # One scratch model for every candidate evaluation, rather than a deepcopy
    # per cycle: candidates are only ever scored, never trained. Unwrapped, so
    # its keys match the unprefixed `state_dict()` the candidates are blended
    # from — and so it does not carry a second DDP process group.
    probe = copy.deepcopy(unwrap_model(model))

    trk.banner(
        f"Stage 3 — {'A' if cfg.stage3.sam_adaptive else ''}SAM + Greedy SWA "
        f"[{cfg.stage3.epochs} epochs]",
        [
            f"SAM ρ={cfg.stage3.sam_rho}  Cycle={cfg.stage3.cycle_len} ep  "
            f"Peak LR={cfg.stage3.swa_lr:.0e}  aux_w={aux_w_s3}",
            f"Margin κ: 1.0 → {cfg.stage3.margin_kappa_final} (per cycle)  "
            f"SWA warm-up: {warmup_cycles} cycles discarded  "
            f"greedy={cfg.stage3.greedy}",
        ],
    )

    trk.progress_start("stage3", cfg.stage3.epochs, "Stage 3")
    for ep in range(1, cfg.stage3.epochs + 1):
        cycle_ep = (ep - 1) % cfg.stage3.cycle_len
        lr_now = cfg.stage3.swa_lr * (
            0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * cycle_ep / cfg.stage3.cycle_len))
        )
        for pg in sam.param_groups:
            pg["lr"] = lr_now

        # T2-1: one margin vector per cycle. Written at every epoch but only
        # *changing* at a boundary, so the objective is constant between two
        # snapshots — which is the property SWA's mean requires.
        kappa = stage3_margin_kappa(
            ep, cfg.stage3.cycle_len, cfg.stage3.epochs, cfg.stage3.margin_kappa_final
        )
        scaled = base_margins * kappa
        model.arcface_head.margins.copy_(scaled)
        ema.shadow.arcface_head.margins.copy_(scaled)

        tl, ta = train_one_epoch_sam(
            cfg,
            fwd,
            train_ldr,
            sam,
            focal_s3,
            device,
            supcon=supcon_s3,
            supcon_weight=0.02,
            proto=proto_s3,
            proto_weight=0.01,
            # None, not a scalar: the per-class vector above is the objective.
            arc_m=None,
            aux_weight=aux_w_s3,
            current_ep=ep,
            ema=ema,
            tracker=trk,
            aux_weights=aux_weights,
        )

        # SAM's double backward leaves the pool at its high-water mark, and a
        # cycle-boundary epoch scores a *third* model (the SWA candidate) right
        # after these two. Bracketed — see `run_stage1`.
        release_memory(device)
        f1_live, acc_live = evaluate(model, val_ldr, device, dist)
        f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device, dist)
        release_memory(device)
        best_live_f1 = max(best_live_f1, f1_live)
        snap_info = ""

        # ── Greedy SWA on the candidate average (T2-2, T2-3) ──────────
        if ep % cfg.stage3.cycle_len == 0:
            cycle = ep // cfg.stage3.cycle_len
            if cycle <= warmup_cycles:
                snap_info = f"· warm-up {cycle}/{warmup_cycles}"
            elif swa_state is None:
                swa_state = {
                    k: v.detach().clone() for k, v in unwrap_model(model).state_dict().items()
                }
                n_snap, f1_swa_running, first_accepted_cycle = 1, f1_live, cycle
                snap_info = f"★ snap 1 (F1 {f1_swa_running:.3f})"
            else:
                candidate = _blend(swa_state, unwrap_model(model).state_dict(), n_snap + 1)
                probe.load_state_dict(candidate)
                f1_cand, _ = evaluate(probe, val_ldr, device, dist)
                # Every rank scored the same candidate on the same (sharded and
                # re-joined) split, so they agree — but the accept/reject is
                # broadcast rather than assumed, because a float comparison two
                # ranks disagree on would fork the SWA average.
                accept = dist.broadcast_object(not cfg.stage3.greedy or f1_cand > f1_swa_running)
                if accept:
                    swa_state, n_snap, f1_swa_running = candidate, n_snap + 1, f1_cand
                    snap_info = f"★ snap {n_snap} (F1 {f1_cand:.3f})"
                else:
                    n_rejected += 1
                    snap_info = f"✗ rejected (avg {f1_cand:.3f} ≤ {f1_swa_running:.3f})"

        trk.log_row(
            "stage3",
            {
                "Ep": f"{ep:03d}/{cfg.stage3.epochs}",
                "Loss": f"{tl:.4f}",
                "Tr": f"{ta:.1%}",
                "F1 live/ema": f"{f1_live:.3f}/{f1_ema:.3f}",
                "Acc live/ema": f"{acc_live:.1%}/{acc_ema:.1%}",
                "LR": f"{lr_now:.2e}",
                "κ": f"{kappa:.3f}",
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
                "val/f1_ema": f1_ema,
                "val/acc_ema": acc_ema,
                "val/f1_best": best_live_f1,
                "sched/lr": lr_now,
                "sched/margin_kappa": kappa,
                "sched/arcface_margin": float(scaled.mean()),
                "sched/supcon_weight": 0.02,
                "sched/proto_weight": 0.01,
                "swa/n_snapshots": float(n_snap),
                "swa/n_rejected": float(n_rejected),
                "swa/f1_running": f1_swa_running,
            },
            step=ep,
        )

    trk.progress_stop("stage3")
    end_tag = epoch_tag(cfg.stage3.epochs, cfg.stage3.epochs)
    trk.log_message(f"{end_tag}Updating BN stats ({n_snap} accepted, {n_rejected} rejected) ...")
    if swa_state is None:
        trk.log_message(f"{end_tag}No snapshots accepted — using final live model.", level="warn")
        swa_state = {k: v.detach().clone() for k, v in unwrap_model(model).state_dict().items()}

    # The scratch probe has done its job; a whole extra model is worth freeing
    # before two more (`swa_model`, and `deepcopy`'s transient) are built.
    del probe
    release_memory(device)

    swa_model = copy.deepcopy(unwrap_model(model))
    swa_model.load_state_dict(swa_state)
    update_bn_stats(build_natural_prior_loader(train_ldr, plan=plan), swa_model, device)
    f1_swa, acc_swa = evaluate(swa_model, val_ldr, device, dist)
    trk.log_message(f"{end_tag}SWA val: F1={f1_swa:.3f}  Acc={acc_swa:.1%}", level="plain")

    # ── SWA vs EMA — score both, keep the better ──────────────────────
    f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device, dist)
    release_memory(device)
    best_source = "swa" if f1_swa >= f1_ema else "ema"
    best_f1 = max(f1_swa, f1_ema)
    best_acc = acc_swa if best_source == "swa" else acc_ema
    trk.log_message(
        f"{end_tag}EMA val: F1={f1_ema:.3f}  Acc={acc_ema:.1%}  →  keeping {best_source.upper()}",
        level="plain",
    )
    trk.log_scalars(
        {"swa/f1": f1_swa, "swa/acc": acc_swa, "ema/f1": f1_ema, "ema/acc": acc_ema},
        step=cfg.stage3.epochs,
    )

    # The bundle's `ema` slot is what `final_eval` loads when `best_source` is
    # not "live"; it carries the SWA average only when the SWA average won.
    if best_source == "swa":
        ema.shadow.load_state_dict(swa_model.state_dict())

    note = ""
    if best_f1 <= prev_best_f1:
        note = "val_f1 did not beat Stage 2; Stage 2 ckpt preferred for eval"
        trk.log_message(
            f"{end_tag}Stage 3 F1 {best_f1:.3f} ≤ Stage 2 best {prev_best_f1:.3f} "
            "— Stage 2 preferred.",
            level="plain",
        )
    else:
        trk.log_message(
            f"{end_tag}Stage 3 F1 {best_f1:.3f} > Stage 2 best {prev_best_f1:.3f} → saving.",
            level="plain",
        )

    save_ckpt(
        cfg,
        best_ckpt,
        cfg.stage3.epochs,
        "Stage 3",
        swa_model,
        ema,
        val_f1=best_f1,
        val_acc=best_acc,
        dist=dist,
        best_source=best_source,
        swa_val_f1=f1_swa,
        ema_val_f1=f1_ema,
        swa_n_snapshots=n_snap,
        swa_n_rejected=n_rejected,
        swa_first_accepted_cycle=first_accepted_cycle,
        **({"note": note} if note else {}),
    )
    return best_f1
