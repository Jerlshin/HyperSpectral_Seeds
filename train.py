#!/usr/bin/env python3
"""SpectralQuadNet training entrypoint — REFACTOR_PLAN.md §4.3.

Replaces ``HSI_modality_training/hsi_training.py`` @ ``886560f`` lines 2701-2858:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`_run`                           2701-2837  (was ``main``)
:func:`main`                           2843-2858  (was ``if __name__``)
=====================================  ==============

Usage
─────
    python train.py                                  # the reference experiment
    python train.py stage1.max_lr=1e-4               # single override
    python train.py -m stage1.max_lr=1e-4,5e-4,1e-3  # sweep
    python train.py tracking.backend=multi tracking.backends=[console,wandb]

The auto-resume orchestration is unchanged (§3.5): ``latest_completed_stage()``
probes stages 3 → 2 → 1, each completed stage is loaded rather than retrained,
and ``_pick_best_checkpoint`` — not the stage order — decides which checkpoint
the final evaluation runs on. Pointing ``output_dir`` at a directory with all
three stages present therefore skips straight to ``final_evaluation``.

Declared deviations from the baseline ``main()``
────────────────────────────────────────────────
1. ``CONFIG[...]`` → ``cfg.<group>.<field>`` throughout, and every collaborator
   gains the leading ``cfg``/``store``/``device`` arguments Phases 2-3 gave it.
2. ``print(...)`` → ``tracker`` calls (§4.1), one-for-one.
3. ``torch.cuda.mem_get_info(device)`` (baseline line 2718) is now guarded by
   ``device.type == "cuda"``. The baseline called it unconditionally, which
   raises on a CPU/MPS host; the VRAM figure is simply omitted off-CUDA and the
   dataset size is still reported.
4. ``CONFIG["device"]`` held a live ``torch.device``; it is now a strategy string
   resolved by ``utils/device.py`` (§4.3), because YAML cannot carry a device.

Everything else — stage order, the ArcFace bootstrap, the recompute fallbacks,
the augmentation strings passed to each stage's loaders, the batch sizes — is
byte-for-byte the same decision tree the baseline ran.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from spectralquadnet.config.schema import register_configs
from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import build_loaders, build_splits
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.engine.checkpoint import (
    _pick_best_checkpoint,
    latest_completed_stage,
    load_ckpt,
    load_stage_meta,
    stage_ckpt_path,
)
from spectralquadnet.engine.diagnostics import compute_class_difficulty
from spectralquadnet.engine.stages.final_eval import final_evaluation
from spectralquadnet.engine.stages.stage1_progressive import run_stage1
from spectralquadnet.engine.stages.stage2_arcface import run_stage2
from spectralquadnet.engine.stages.stage3_sam_swa import run_stage3_swa
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.tracking import build_tracker
from spectralquadnet.tracking.base import ExperimentTracker
from spectralquadnet.utils.device import resolve_device
from spectralquadnet.utils.seed import set_seed

# Makes the dataclass schemas available to Hydra's composition, so a typo in a
# YAML field fails at startup instead of hours into Stage 1 (§4.3).
register_configs()

#: Baseline line 2706 — how `latest_completed_stage()`'s return value reads.
_RESUME_LABELS = {0: "starting fresh", 1: "Stage 1 done", 2: "Stages 1–2 done", 3: "all done"}


def _run(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    """The baseline's ``main()`` body, reading ``cfg`` instead of ``CONFIG``."""
    # ══════════════════════════════════════════════════════════════════
    #  RNG ORDERING — REFACTOR_PLAN.md §3.6. DO NOT MOVE THIS CALL.
    #
    #  The baseline ran `set_seed(CONFIG["seed"])` as an import-time side
    #  effect (monolith line 200), so it always executed *before* `main()`
    #  and therefore before the model was built. Every `kaiming_normal_` /
    #  `trunc_normal_` / `xavier_uniform_` inside the branches' `_init_weights`
    #  draws from the same global torch RNG stream, in the order
    #  `SpectralQuadNet.__init__` constructs them — so the seed must be set
    #  here, and the required call order is exactly:
    #
    #      load config → set_seed(cfg.seed) → DataStore → SpectralQuadNet → ModelEMA
    #
    #  Nothing between this line and the model construction below may consume
    #  the global RNG. `build_splits` is safe: sklearn's `train_test_split`
    #  takes `random_state=42` and uses its own RandomState. Building the
    #  tracker happens before this call for the same reason.
    # ══════════════════════════════════════════════════════════════════
    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    ckpt_s1 = stage_ckpt_path(cfg, 1)
    ckpt_s2 = stage_ckpt_path(cfg, 2)
    ckpt_s3 = stage_ckpt_path(cfg, 3)
    done_stage = latest_completed_stage(cfg)

    tracker.banner(
        f"Auto-resume: {_RESUME_LABELS.get(done_stage, f'stage {done_stage} done')}",
        [f"Output dir : {cfg.output_dir}"],
    )
    tracker.log_message(f"Latest completed stage: {done_stage}")

    store = DataStore.from_config(cfg.data, device)

    gb_on_disk = store.require_patches().size * 4 / 1e9
    vram = (
        f" | {torch.cuda.mem_get_info(device)[0] / 1e9:.1f} GB VRAM free"
        if device.type == "cuda"
        else ""
    )
    tracker.log_message(
        f"[DATA] ✓ Dataset (mmap): {gb_on_disk:.1f} GB on disk {vram} | num_workers=0",
        level="plain",
    )

    all_labels, train_idx, val_idx, test_idx = build_splits(cfg)
    tracker.log_message(
        f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}",
        level="plain",
    )
    tracker.log_message(
        f"Samples/class (train): ~{len(train_idx) // cfg.data.num_classes}", level="plain"
    )

    model = SpectralQuadNet(
        cfg=cfg,
        physical_wl=store.require_wavelengths(),
        num_classes=cfg.data.num_classes,
        num_bands=cfg.data.num_bands,
        dropout=cfg.stage1.dropout,
        wl_embed_dim=cfg.model.wl_embed_dim,
    ).to(device)

    ema = ModelEMA(model, decay=cfg.ema_decay)

    tracker.log_message(
        "Model  : SpectralQuadNet v4 (4× AuxHead deep supervision + P3 oversampling)",
        level="plain",
    )
    # Replaces the baseline's `Params : …M` print: the console backend renders
    # exactly that line, and the structured backends attach gradient histograms.
    tracker.watch(model)
    tracker.log_message(f"Device : {device}", level="plain")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        tracker.log_message(
            f"[GPU]  {props.name}  |  VRAM {props.total_memory // 1024**3} GB  |  "
            f"TF32={torch.backends.cuda.matmul.allow_tf32}",
            level="plain",
        )

    # ── Stage 1 phase loaders (per-phase aug profiles) ─────────────────
    def _s1_ldr(aug_str: str) -> DataLoader[Any]:
        ds = RiceSeedDataset(
            train_idx, aug_strength=aug_str, store=store, data_cfg=cfg.data, device=device
        )
        return DataLoader(
            ds, batch_size=cfg.stage1.batch, shuffle=True, drop_last=True, num_workers=0
        )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1
    # ══════════════════════════════════════════════════════════════════
    if done_stage < 1:
        tracker.log_message("[RUN] Stage 1", level="plain")
        phase_loaders = {1: _s1_ldr("heavy"), 2: _s1_ldr("medium"), 3: _s1_ldr("very_light")}
        _, val_ldr1, _ = build_loaders(
            cfg,
            store,
            device,
            train_idx,
            val_idx,
            test_idx,
            cfg.stage1.batch,
            train_aug="none",
        )
        run_stage1(cfg, store, model, ema, phase_loaders, val_ldr1, device, ckpt_s1, tracker)
        tracker.log_message("Reloading best Stage 1 checkpoint ...")
        load_ckpt(ckpt_s1, model, ema, device)
    else:
        tracker.log_message("[SKIP] Stage 1 → loading checkpoint", level="plain")
        load_ckpt(ckpt_s1, model, ema, device)

    meta_s1 = load_stage_meta(cfg, 1)
    class_f1_s1 = meta_s1.get("class_f1", {})
    cdws_wts_s1 = meta_s1.get("cdws_weights", {})
    arcface_done = meta_s1.get("arcface_init_done", False)
    s1_best_f1 = meta_s1.get("val_f1", meta_s1.get("val_acc", 0.0))
    tracker.log_message(
        f"Stage 1 → F1={s1_best_f1:.3f}  "
        f"hard classes={sum(1 for f in class_f1_s1.values() if f < 0.5)}"
    )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2
    # ══════════════════════════════════════════════════════════════════
    if done_stage < 2:
        if not arcface_done:
            tracker.log_message("Bootstrapping ArcFace from linear head")
            # `nn.Module.__getattr__` is typed `Tensor | Module`, so the `.weight`
            # of the Sequential's last layer needs a cast for mypy; at runtime it
            # is the nn.Linear the baseline indexed the same way (line 2765).
            lw = cast(torch.Tensor, model.linear_head[-1].weight).data.clone()
            model.arcface_head.init_from_linear(lw)
            ema.shadow.arcface_head.init_from_linear(lw)

        if not class_f1_s1:
            tracker.log_message("No class_f1 in Stage 1 meta — recomputing", level="warn")
            _, val_cd, _ = build_loaders(cfg, store, device, train_idx, val_idx, test_idx, 128)
            class_f1_s1, cdws_wts_s1 = compute_class_difficulty(
                cfg, ema.shadow, val_cd, device, "Stage 1 (recomputed)", tracker=tracker
            )

        tracker.log_message("[RUN] Stage 2", level="plain")
        tr2, va2, _ = build_loaders(
            cfg,
            store,
            device,
            train_idx,
            val_idx,
            test_idx,
            cfg.stage2.batch,
            balanced=True,
            all_labels=all_labels,
            # FIXED: was "light" (no spectral aug) — "very_light" adds mild spectral noise
            # so ArcFace doesn't overfit the train set's exact spectral signatures
            train_aug="very_light",
            class_weights=cdws_wts_s1,
        )
        run_stage2(cfg, model, ema, tr2, va2, device, ckpt_s2, class_f1_s1, tracker)
        tracker.log_message("Reloading best Stage 2 checkpoint ...")
        load_ckpt(ckpt_s2, model, ema, device)
    else:
        tracker.log_message("[SKIP] Stage 2 → loading checkpoint", level="plain")
        load_ckpt(ckpt_s2, model, ema, device)

    meta_s2 = load_stage_meta(cfg, 2)
    class_f1_s2 = meta_s2.get("class_f1", {})  # noqa: F841 - parity with the baseline binding
    cdws_wts_s2 = meta_s2.get("cdws_weights", {})
    s2_best_f1 = meta_s2.get("val_f1", meta_s2.get("s2_val_f1", meta_s2.get("val_acc", 0.0)))
    tracker.log_message(f"Stage 2 → F1={s2_best_f1:.3f}")

    if hasattr(torch, "_dynamo"):
        tracker.log_message("Disabling torch.compile for Stage 3 stability")
        torch._dynamo.reset()

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 3
    # ══════════════════════════════════════════════════════════════════
    if done_stage < 3:
        if not cdws_wts_s2:
            tracker.log_message(
                "No cdws_weights in Stage 2 meta — falling back to Stage 1", level="warn"
            )
            cdws_wts_s2 = cdws_wts_s1

        tracker.log_message("[RUN] Stage 3 (SAM + Greedy SWA)", level="plain")
        tr3, va3, _ = build_loaders(
            cfg,
            store,
            device,
            train_idx,
            val_idx,
            test_idx,
            cfg.stage2.batch,
            balanced=True,
            all_labels=all_labels,
            train_aug="light",
            class_weights=cdws_wts_s2,
        )
        run_stage3_swa(
            cfg, model, ema, tr3, va3, device, ckpt_s3, prev_best_f1=s2_best_f1, tracker=tracker
        )
    else:
        tracker.log_message("[SKIP] Stage 3 → loading checkpoint", level="plain")
        load_ckpt(ckpt_s3, model, ema, device)
        meta_s3 = load_stage_meta(cfg, 3)
        tracker.log_message(
            f"Stage 3 → snaps={meta_s3.get('swa_n_snapshots', '?')}  "
            f"rejected={meta_s3.get('swa_n_rejected', '?')}  "
            f"F1={meta_s3.get('val_f1', meta_s3.get('val_acc', 0)):.3f}"
        )

    # ══════════════════════════════════════════════════════════════════
    #  FINAL EVALUATION
    # ══════════════════════════════════════════════════════════════════
    best_final_ckpt = _pick_best_checkpoint(cfg, ckpt_s1, ckpt_s2, ckpt_s3)
    tracker.log_message(f"Best checkpoint (by val_f1): {best_final_ckpt}")

    _, _, test_ldr = build_loaders(cfg, store, device, train_idx, val_idx, test_idx, 256)
    final_evaluation(cfg, model, ema, test_ldr, device, best_final_ckpt, tracker)


@hydra.main(config_path="configs", config_name="experiment/output_v12_spa40", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose the config, set up logging and the tracker, then run.

    The ``logging`` setup and the top-level handler reproduce the baseline's
    ``if __name__ == "__main__"`` block (lines 2843-2858): a ``training.log`` in
    the output directory plus a stdout stream, and a ``logging.critical`` +
    ``sys.exit(1)`` on any escaping exception.
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.output_dir, "training.log")),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # @hydra.main installs its own handlers first
    )

    tracker = build_tracker(cfg)
    tracker.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    try:
        _run(cfg, tracker)
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
