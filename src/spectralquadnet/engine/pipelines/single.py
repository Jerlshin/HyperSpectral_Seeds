"""The single-stage pipeline (IC-11 / CHANGES §17).

Builds the loaders the collapsed curriculum needs, runs it, and hands the best
checkpoint to the final evaluation. Auto-resume is the same mechanism the
three-stage pipeline uses — a completed run is detected by its stage-1
checkpoint and skipped straight to reporting — so pointing ``output_dir`` at a
finished run re-scores it rather than retraining it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import DataLoader

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import build_eval_loader, build_train_loader
from spectralquadnet.data.samplers import ClassBalancedBatchSampler
from spectralquadnet.engine.checkpoint import load_ckpt, stage_ckpt_path, stage_exists
from spectralquadnet.engine.stages.final_eval import final_evaluation
from spectralquadnet.engine.stages.single_stage import run_single_stage
from spectralquadnet.tracking.global_step import stage_tracker
from spectralquadnet.utils.device import release_memory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.engine.pipelines.context import RunContext


def run(ctx: RunContext) -> None:
    """Train the single stage (or reload it), then score the reporting split once."""
    cfg = ctx.cfg
    trk = ctx.tracker
    ckpt = stage_ckpt_path(cfg, 1)

    if stage_exists(cfg, 1):
        trk.log_message("[SKIP] Single stage → loading checkpoint", level="plain")
        load_ckpt(ckpt, ctx.model, ctx.ema, ctx.device)
    else:
        trk.log_message("[RUN] Single stage", level="plain")
        train_ldr = _train_loader(ctx)
        select_ldr = _select_loader(ctx)
        supcon_ldr = _supcon_loader(ctx) if int(cfg.single.supcon_epochs) > 0 else None

        best_select_f1 = run_single_stage(
            cfg,
            ctx.model,
            ctx.ema,
            train_ldr,
            select_ldr,
            ctx.device,
            ckpt,
            # Stage 1 of 1, so the offset starts at 0 and the rebasing is the
            # identity — but it is applied anyway, so the scalar stream a
            # single-stage run produces is the same shape as a three-stage
            # one's and the two are comparable panel for panel (IC-1).
            tracker=stage_tracker(trk, ctx.clock, stage=1),
            fit_ldr=ctx.calib_loader,
            plan=ctx.plan,
            dist=ctx.dist,
            train_module=ctx.train_module,
            supcon_ldr=supcon_ldr,
        )
        trk.log_message(f"Single stage → best {ctx.select_split} F1={best_select_f1:.4f}")
        trk.log_message("Reloading best checkpoint ...")
        load_ckpt(ckpt, ctx.model, ctx.ema, ctx.device)
        # The loaders and their persistent worker pools are the last thing
        # holding the phase's prefetch queues; dropping them before the
        # evaluation allocates its own is what keeps the peak bounded.
        del train_ldr, select_ldr, supcon_ldr
        release_memory(ctx.device)

    final_evaluation(
        cfg,
        ctx.model,
        ctx.ema,
        _report_loader(ctx),
        ctx.device,
        ckpt,
        trk,
        dist=ctx.dist,
        run_summary=ctx.summary(),
    )


# ══════════════════════════════════════════════════════════════════════
#  Loaders
# ══════════════════════════════════════════════════════════════════════


def _dataset_kwargs(ctx: RunContext) -> dict[str, Any]:
    return {
        "store": ctx.store,
        "data_cfg": ctx.cfg.data,
        "device": ctx.device,
        "morph": ctx.morph,
    }


def _train_loader(ctx: RunContext) -> DataLoader[Any]:
    """Plain shuffled batches under one augmentation profile.

    No class-balanced sampler and no oversampling: the dataset is 91–96 samples
    per class, so class imbalance is not a real problem here and CDWS was
    correcting something that does not exist (CHANGES §6, item 17).
    """
    ds = RiceSeedDataset(
        ctx.splits.train, aug_strength=str(ctx.cfg.single.aug_profile), **_dataset_kwargs(ctx)
    )
    return build_train_loader(
        ds, int(ctx.cfg.single.batch), seed=int(ctx.cfg.seed), plan=ctx.plan, dist=ctx.dist
    )


def _supcon_loader(ctx: RunContext) -> DataLoader[Any]:
    """``bal_n_cls × bal_n_spc`` class-balanced batches for the optional Phase B.

    SupCon at ``τ=0.10`` needs several positives per anchor; a shuffled batch of
    128 over 90 classes gives most anchors none. A6 must control this — the
    audited design changed the loss *and* the sampler at the same epoch.
    """
    ds = RiceSeedDataset(
        ctx.splits.train, aug_strength=str(ctx.cfg.single.aug_profile), **_dataset_kwargs(ctx)
    )
    sampler = ClassBalancedBatchSampler(
        ctx.splits.labels[ctx.splits.train],
        int(ctx.cfg.single.bal_n_cls),
        int(ctx.cfg.single.bal_n_spc),
        class_weights=None,
        seed=int(ctx.cfg.seed) if ctx.dist.enabled else None,
    )
    return DataLoader(ds, batch_sampler=sampler, **ctx.plan.loader_kwargs)


def _select_loader(ctx: RunContext) -> DataLoader[Any]:
    """The split the checkpoint decision reads — ``calib`` under the shipped protocol."""
    if ctx.select_split == "calib" and ctx.calib_loader is not None:
        return ctx.calib_loader
    return build_eval_loader(
        RiceSeedDataset(ctx.splits.val, **_dataset_kwargs(ctx)),
        plan=ctx.plan,
        dist=ctx.dist,
        persistent=True,
    )


def _report_loader(ctx: RunContext) -> DataLoader[Any]:
    """The held-out split, scored exactly once.

    ``report_split=val_test`` concatenates them, which is §19.1's rule for the
    grouped protocol: they are two halves of the *same* held-out bundle, so they
    are not independent of each other and reporting one after selecting on the
    other would still be partially self-fulfilling.
    """
    if str(ctx.cfg.evaluation.report_split) == "val_test":
        idx = np.sort(np.concatenate([ctx.splits.val, ctx.splits.test]))
    else:
        idx = ctx.splits.test
    return build_eval_loader(
        RiceSeedDataset(idx, **_dataset_kwargs(ctx)), plan=ctx.plan, dist=ctx.dist
    )


# Re-exported so the three-stage pipeline builds its reporting loader the same
# way: which split is reported is a protocol decision (CHANGES §19.1), not a
# per-pipeline one, and A8 compares the two pipelines on it.
report_loader = _report_loader
dataset_kwargs = _dataset_kwargs
__all__ = ["dataset_kwargs", "report_loader", "run"]
