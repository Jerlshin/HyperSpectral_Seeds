"""The audited Stage 1 → 2 → 3 pipeline, and A8's three arms.

This is the curriculum ``stratified_benchmark_rtx3060`` ran, moved out of
``train.py`` unchanged in behaviour. It is retained for one reason: **A8 is the
falsification test for 65% of the audited run's compute**, and it needs
``pipeline=stage1_only`` vs ``stage1_stage2`` vs ``three_stage`` to be the same
code stopped at different points rather than three separately-written loops.

The measured case against it, so nobody re-enables it by accident:

======  ==========  ================  ====================  ==================
Stage   Wall clock  Δ val macro-F1    Δ in eval samples     Hours per 0.001 F1
======  ==========  ================  ====================  ==================
1       6.6 h       (baseline 0.842)  —                     —
2       2.7 h       **+0.002**        **2.6 / 1294**        1,325
3       ~9.5 h      **+0.003**        **3.9 / 1294**        3,167
======  ==========  ================  ====================  ==================

Against a ±26-sample 95% CI on that split, and an estimated +0.042 of running-
maximum selection bias. Worse, Stage 2's best epoch is **19**, two epochs
*before* the per-class margin vector — its headline mechanism — takes over.

IC-1 is applied here rather than left to each stage: each stage gets a
:class:`~spectralquadnet.tracking.global_step.StepOffsetTracker` over the run's
shared clock, so Stage 2 and Stage 3 scalars land past Stage 1's last step
instead of being silently discarded by W&B. In the audited run that discarded
*every* Stage-2 and Stage-3 scalar — ``sam/grad_cos``, the margin curves, the
SWA accept/reject series — roughly 200 warnings' worth (CHANGES §10.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import build_eval_loader, build_loaders, build_train_loader
from spectralquadnet.engine.checkpoint import (
    _pick_best_checkpoint,
    latest_completed_stage,
    load_ckpt,
    load_stage_meta,
    stage_ckpt_path,
)
from spectralquadnet.engine.diagnostics import compute_class_difficulty
from spectralquadnet.engine.evaluate import collect_embeddings
from spectralquadnet.engine.pipelines.single import dataset_kwargs, report_loader
from spectralquadnet.engine.stages.final_eval import final_evaluation
from spectralquadnet.engine.stages.stage1_progressive import run_stage1
from spectralquadnet.engine.stages.stage2_arcface import run_stage2
from spectralquadnet.engine.stages.stage3_sam_swa import run_stage3_swa
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.tracking.global_step import stage_tracker
from spectralquadnet.utils.device import release_memory, reset_compilation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.engine.pipelines.context import RunContext

#: How ``latest_completed_stage()``'s return value reads in the resume banner.
_RESUME_LABELS = {0: "starting fresh", 1: "Stage 1 done", 2: "Stages 1–2 done", 3: "all done"}

#: A8's three arms: the highest stage each ``pipeline`` value runs to.
#: Same driver, same loaders, same seeds — stopped at different points, which is
#: what makes the three numbers a comparison rather than three anecdotes.
PIPELINE_LAST_STAGE: dict[str, int] = {
    "stage1_only": 1,
    "stage1_stage2": 2,
    "three_stage": 3,
}


def quadnet(ctx: RunContext) -> SpectralQuadNet:
    """``ctx.model`` as the architecture these stages are written against.

    The three-stage curriculum is not architecture-agnostic and does not pretend
    to be: Stage 2 seeds ArcFace sub-centres by spherical *k*-means on the
    head's own prototypes, Stage 3 anneals that head's per-class margin vector,
    and both read ``cfg.stage2.*`` keys that only exist for it. Pairing it with
    ``SpectralSeedNet`` — which has a K=1 head and no per-class vector — would
    run, and would quietly mean something different.

    So the requirement is checked rather than assumed, with a message that names
    the composition that *does* work.

    Raises:
        TypeError: ``cfg.pipeline`` is a three-stage arm but ``cfg.model.arch``
            is not the four-branch model.
    """
    if not isinstance(ctx.model, SpectralQuadNet):
        raise TypeError(
            f"pipeline={ctx.cfg.pipeline!r} is the audited three-stage curriculum and requires "
            f"model.arch='spectral_quadnet'; got {ctx.cfg.model.arch!r}. The collapsed "
            "curriculum for the replacement architecture is pipeline='single' (CHANGES §17)."
        )
    return ctx.model


def run(ctx: RunContext) -> None:
    """Run stages 1…N (N from :data:`PIPELINE_LAST_STAGE`), then report once."""
    cfg = ctx.cfg
    trk = ctx.tracker
    last_stage = PIPELINE_LAST_STAGE[str(cfg.pipeline)]
    model = quadnet(ctx)

    ckpts = {s: stage_ckpt_path(cfg, s) for s in (1, 2, 3)}
    done_stage = latest_completed_stage(cfg)
    trk.banner(
        f"Auto-resume: {_RESUME_LABELS.get(done_stage, f'stage {done_stage} done')}",
        [f"Output dir : {cfg.output_dir}", f"Arm        : {cfg.pipeline} (stages 1–{last_stage})"],
    )

    _stage1(ctx, model, ckpts[1], done_stage)
    meta_s1 = load_stage_meta(cfg, 1)
    class_f1_s1 = meta_s1.get("class_f1", {})
    cdws_s1 = meta_s1.get("cdws_weights", {})
    s1_best = meta_s1.get("val_f1", meta_s1.get("val_acc", 0.0))
    trk.log_message(
        f"Stage 1 → F1={s1_best:.3f}  hard classes={sum(1 for f in class_f1_s1.values() if f < 0.5)}"
    )

    cdws_s2, s2_best = cdws_s1, s1_best
    if last_stage >= 2:
        _stage2(ctx, model, ckpts[2], done_stage, meta_s1, class_f1_s1, cdws_s1)
        meta_s2 = load_stage_meta(cfg, 2)
        cdws_s2 = meta_s2.get("cdws_weights", {}) or cdws_s1
        s2_best = meta_s2.get("val_f1", meta_s2.get("val_acc", 0.0))
        trk.log_message(f"Stage 2 → F1={s2_best:.3f}")

    if last_stage >= 3:
        # Stage 3's double backward and per-cycle margin vector invalidate a
        # compiled graph's guards on every cycle; the graphs are dropped rather
        # than left to recompile.
        if ctx.plan.compile_enabled:
            trk.log_message("Dropping compiled graphs before Stage 3")
        reset_compilation()
        _stage3(ctx, model, ckpts[3], done_stage, cdws_s2, s2_best)

    active = [ckpts[s] for s in range(1, last_stage + 1)]
    best_ckpt = _pick_best_checkpoint(cfg, *active)
    trk.log_message(f"Best checkpoint (by selection F1): {best_ckpt}")

    # Stage 3 leaves two extra model copies behind; the allocator gets them back
    # before the twelve-view TTA pass allocates its own activations.
    release_memory(ctx.device)
    final_evaluation(
        cfg,
        model,
        ctx.ema,
        report_loader(ctx),
        ctx.device,
        best_ckpt,
        trk,
        dist=ctx.dist,
        run_summary=ctx.summary(),
    )


# ══════════════════════════════════════════════════════════════════════
#  Stages
# ══════════════════════════════════════════════════════════════════════


def _select_loader(ctx: RunContext) -> DataLoader[Any]:
    """The split every stage checkpoints on.

    ``calib`` when one was carved and the protocol asks for it — the audited
    run's ``calib_frac=0.0`` put the margins, Ω, the CDWS weights *and* the
    Phase-3 oversampling weights on the same 1,294 patches that then selected
    the checkpoint and produced the headline (CHANGES §4.4).
    """
    if ctx.select_split == "calib" and ctx.calib_loader is not None:
        return ctx.calib_loader
    return build_eval_loader(
        RiceSeedDataset(ctx.splits.val, **dataset_kwargs(ctx)),
        plan=ctx.plan,
        dist=ctx.dist,
        persistent=True,
    )


def _stage1(ctx: RunContext, model: SpectralQuadNet, ckpt: str, done_stage: int) -> None:
    cfg, trk = ctx.cfg, ctx.tracker
    if done_stage >= 1:
        trk.log_message("[SKIP] Stage 1 → loading checkpoint", level="plain")
        load_ckpt(ckpt, model, ctx.ema, ctx.device)
        # The clock still has to advance past the epochs the skipped stage
        # logged, or a resumed Stage 2 would restart the step axis at 1 and
        # reintroduce exactly the collision IC-1 removes.
        ctx.clock.advance(int(load_stage_meta(cfg, 1).get("epoch", 0)))
        return

    trk.log_message("[RUN] Stage 1", level="plain")

    def _phase_loader(aug: str) -> DataLoader[Any]:
        ds = RiceSeedDataset(ctx.splits.train, aug_strength=aug, **dataset_kwargs(ctx))
        return build_train_loader(
            ds, cfg.stage1.batch, seed=int(cfg.seed), plan=ctx.plan, dist=ctx.dist
        )

    phase_loaders = {1: _phase_loader("heavy"), 2: _phase_loader("medium"), 3: _phase_loader("very_light")}
    select_ldr = _select_loader(ctx)
    run_stage1(
        cfg,
        ctx.store,
        model,
        ctx.ema,
        phase_loaders,
        select_ldr,
        ctx.device,
        ckpt,
        stage_tracker(trk, ctx.clock, stage=1),
        calib_ldr=ctx.calib_loader,
        plan=ctx.plan,
        dist=ctx.dist,
        train_module=ctx.train_module,
    )
    ctx.clock.advance(int(load_stage_meta(cfg, 1).get("epoch", cfg.stage1.epochs)))
    trk.log_message("Reloading best Stage 1 checkpoint ...")
    load_ckpt(ckpt, model, ctx.ema, ctx.device)
    # A loader holds `persistent_workers`' pool for exactly as long as something
    # holds the loader; without this the Stage-1 phase and validation workers
    # stay resident through Stages 2 and 3 — processes that will never serve
    # another batch, each with torch imported and its own mapping of the cube.
    del phase_loaders, select_ldr
    release_memory(ctx.device)


def _stage2(
    ctx: RunContext,
    model: SpectralQuadNet,
    ckpt: str,
    done_stage: int,
    meta_s1: dict[str, Any],
    class_f1_s1: dict[int, float],
    cdws_s1: dict[int, float],
) -> None:
    cfg, trk = ctx.cfg, ctx.tracker
    if done_stage >= 2:
        trk.log_message("[SKIP] Stage 2 → loading checkpoint", level="plain")
        load_ckpt(ckpt, model, ctx.ema, ctx.device)
        ctx.clock.advance(int(load_stage_meta(cfg, 2).get("epoch", 0)))
        return

    if not meta_s1.get("arcface_init_done", False) and int(cfg.model.subcenter_K) > 1:
        # Re-seeding, not a bootstrap: the head is the one Stage 1 just trained,
        # so spherical k-means on its embeddings puts every sub-centre inside
        # its own class's data. Skipped entirely at K=1 (IC-9) — there is
        # nothing to separate, and the "worst within-class sub-centre cosine"
        # line it printed would be trivially 1.0 and actively misleading.
        trk.log_message("Seeding ArcFace sub-centres by spherical k-means")
        seed_ldr = build_eval_loader(
            RiceSeedDataset(ctx.splits.train, aug_strength="none", **dataset_kwargs(ctx)),
            plan=ctx.plan,
            batch_size=256,
        )
        emb, emb_y = collect_embeddings(ctx.ema.shadow, seed_ldr, ctx.device)
        stats: dict[str, float] = {}
        for head in (model.arcface_head, ctx.ema.shadow.arcface_head):
            stats = head.init_subcentres_from_embeddings(emb, emb_y)
        line = f"Seeded {int(stats.get('classes_seeded', 0))} classes"
        if "min_separation" in stats:
            line += f"; worst within-class sub-centre cosine {stats['min_separation']:.3f}"
        trk.log_message(line)
        del seed_ldr, emb, emb_y
        release_memory(ctx.device)

    if not class_f1_s1:
        trk.log_message("No class_f1 in Stage 1 meta — recomputing", level="warn")
        class_f1_s1, cdws_s1 = compute_class_difficulty(
            cfg, ctx.ema.shadow, _select_loader(ctx), ctx.device, "Stage 1 (recomputed)", tracker=trk
        )

    trk.log_message("[RUN] Stage 2", level="plain")
    tr2, _, _ = build_loaders(
        cfg,
        ctx.store,
        ctx.device,
        ctx.splits.train,
        ctx.splits.val,
        ctx.splits.test,
        cfg.stage2.batch,
        balanced=True,
        all_labels=ctx.splits.labels,
        train_aug="very_light",
        class_weights=cdws_s1,
        plan=ctx.plan,
        dist=ctx.dist,
        morph=ctx.morph,
    )
    select_ldr = _select_loader(ctx)
    run_stage2(
        cfg,
        model,
        ctx.ema,
        tr2,
        select_ldr,
        ctx.device,
        ckpt,
        stage_tracker(trk, ctx.clock, stage=2),
        calib_ldr=ctx.calib_loader,
        plan=ctx.plan,
        dist=ctx.dist,
        train_module=ctx.train_module,
    )
    ctx.clock.advance(int(load_stage_meta(cfg, 2).get("epoch", cfg.stage2.epochs)))
    trk.log_message("Reloading best Stage 2 checkpoint ...")
    load_ckpt(ckpt, model, ctx.ema, ctx.device)
    del tr2, select_ldr
    release_memory(ctx.device)


def _stage3(
    ctx: RunContext,
    model: SpectralQuadNet,
    ckpt: str,
    done_stage: int,
    cdws_s2: dict[int, float],
    prev_best: float,
) -> None:
    cfg, trk = ctx.cfg, ctx.tracker
    if done_stage >= 3:
        trk.log_message("[SKIP] Stage 3 → loading checkpoint", level="plain")
        load_ckpt(ckpt, model, ctx.ema, ctx.device)
        meta_s3 = load_stage_meta(cfg, 3)
        ctx.clock.advance(int(meta_s3.get("epoch", 0)))
        trk.log_message(
            f"Stage 3 → snaps={meta_s3.get('swa_n_snapshots', '?')}  "
            f"rejected={meta_s3.get('swa_n_rejected', '?')}  "
            f"F1={meta_s3.get('val_f1', meta_s3.get('val_acc', 0)):.3f}"
        )
        return

    trk.log_message("[RUN] Stage 3 (SAM + Greedy SWA)", level="plain")
    tr3, _, _ = build_loaders(
        cfg,
        ctx.store,
        ctx.device,
        ctx.splits.train,
        ctx.splits.val,
        ctx.splits.test,
        cfg.stage2.batch,
        balanced=True,
        all_labels=ctx.splits.labels,
        train_aug="light",
        class_weights=cdws_s2,
        plan=ctx.plan,
        dist=ctx.dist,
        morph=ctx.morph,
    )
    select_ldr = _select_loader(ctx)
    run_stage3_swa(
        cfg,
        model,
        ctx.ema,
        tr3,
        select_ldr,
        ctx.device,
        ckpt,
        prev_best_f1=prev_best,
        tracker=stage_tracker(trk, ctx.clock, stage=3),
        plan=ctx.plan,
        dist=ctx.dist,
        train_module=ctx.train_module,
    )
    ctx.clock.advance(int(load_stage_meta(cfg, 3).get("epoch", cfg.stage3.epochs)))
    del tr3, select_ldr
    release_memory(ctx.device)
