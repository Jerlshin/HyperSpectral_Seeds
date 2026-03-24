"""
optuna_aug_tuning.py — Stage 1 Augmentation HPO
════════════════════════════════════════════════════════════════════════════════
Tunes ONLY these RiceSeedDataset parameters for Stage 1 (Phase 1 / 2 / 3):

    _PROFILES       → heavy / medium / light  (band_drop, cutout, noise, warp, mult)
    _INTENSITY_SCALE → heavy / medium / light
    _WARP_RANGE     → heavy / medium / light
    CONFIG          → noise_std, max_cutout_bands

Nothing in Stage 2 / 3 is touched.

Design choices
──────────────
• Monotone hierarchy enforced via multiplicative fractions:
      medium_X  = heavy_X  × m_frac_X   (m_frac ∈ [0.20, 0.90])
      light_X   = medium_X × l_frac_X   (l_frac ∈ [0.00, 0.45])
  This guarantees  heavy ≥ medium ≥ light  for every param without
  wasting trials on invalid configurations.

• Each trial runs a condensed Stage 1 (default 90 epochs ≈ 31 % of 286).
  Phase boundaries are kept proportional to the original s1_phase1/2_frac.

• Optuna MedianPruner kills clearly bad trials early (after warm-up).
• TPESampler with multivariate=True exploits cross-param correlations.
• Results persist in SQLite so a crashed run resumes automatically.

Usage
─────
    # basic
    python optuna_aug_tuning.py

    # custom
    python optuna_aug_tuning.py --n_trials 80 --n_epochs 100 --study aug_v2

    # resume (SQLite auto-resumes)
    python optuna_aug_tuning.py --n_trials 120

After the study the best params are printed in a ready-to-paste block.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Silence noisy warnings before importing the training module ──────────────
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

# ── Import everything we need from the main training script ─────────────────
# new_hs_train.py must be in the same directory (or on sys.path).
# All module-level code in new_hs_train (print CONFIG, mkdir, set_seed …)
# runs once on import — that is harmless.
try:
    import new_hs_train as _T  # type: ignore
except ModuleNotFoundError:
    sys.exit(
        "[ERROR] new_hs_train.py not found. "
        "Place optuna_aug_tuning.py in the same directory."
    )

import optuna
from optuna.pruners  import MedianPruner
from optuna.samplers import TPESampler
from torch.amp import GradScaler

# Re-bind commonly used symbols for brevity
CONFIG            = _T.CONFIG
RiceSeedDataset   = _T.RiceSeedDataset
SpectralQuadNet   = _T.SpectralQuadNet
ModelEMA          = _T.ModelEMA
build_splits      = _T.build_splits
build_loaders     = _T.build_loaders
build_optimizer_s1 = _T.build_optimizer_s1
train_one_epoch   = _T.train_one_epoch
evaluate          = _T.evaluate
FocalLoss         = _T.FocalLoss
_aux_loss_weight  = _T._aux_loss_weight
set_seed          = _T.set_seed


# ══════════════════════════════════════════════════════════════════════════════
#  TUNING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TUNE_CFG: dict = {
    # ── Optuna study ───────────────────────────────────────────────────────────
    "n_trials":           60,          # total trials (increase for deeper search)
    "n_jobs":              1,          # parallel trials — keep 1 for single GPU
    "study_name":    "aug_hpo_stage1",
    "storage":  "sqlite:///aug_hpo_stage1.db",  # resumes automatically

    # ── Pruner (MedianPruner) ──────────────────────────────────────────────────
    "pruner_n_startup":   5,           # trials before pruning starts
    "pruner_n_warmup":   15,           # epochs inside each trial before pruning
    "pruner_interval":    3,           # check every N epochs

    # ── Condensed Stage 1 ─────────────────────────────────────────────────────
    "n_epochs":           90,          # epochs per trial (~31% of full 286)
    "patience_frac":     0.35,         # early-stop patience = n_epochs * frac

    # ── Search space bounds ───────────────────────────────────────────────────
    # Heavy (Phase 1) absolute ranges
    "heavy_band_drop": (0.03, 0.22),
    "heavy_cutout":    (0.02, 0.22),
    "heavy_noise":     (0.01, 0.18),
    "heavy_warp":      (0.01, 0.12),
    "heavy_mult":      (0.01, 0.18),

    # Medium fraction of heavy (enforces medium ≤ heavy)
    "m_frac_range":    (0.20, 0.90),

    # Light fraction of medium (enforces light ≤ medium)
    "l_frac_range":    (0.00, 0.45),

    # Intensity scale
    "intensity_h_range": (0.70, 1.30),
    "intensity_m_frac":  (0.35, 0.85),
    "intensity_l_frac":  (0.00, 0.55),

    # Warp range
    "warp_range_h":      (0.02, 0.14),
    "warp_range_m_frac": (0.20, 0.85),
    "warp_range_l_frac": (0.00, 0.50),

    # Global noise / cutout config
    "noise_std_range":        (0.010, 0.120),
    "max_cutout_bands_range": (3,     22),
}

# ── Snapshot of original defaults (restored after each trial) ──────────────
_ORIG_PROFILES        = copy.deepcopy(RiceSeedDataset._PROFILES)
_ORIG_INTENSITY_SCALE = copy.deepcopy(RiceSeedDataset._INTENSITY_SCALE)
_ORIG_WARP_RANGE      = copy.deepcopy(RiceSeedDataset._WARP_RANGE)
_ORIG_NOISE_STD       = CONFIG["noise_std"]
_ORIG_MAX_CUTOUT      = CONFIG["max_cutout_bands"]


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _patch_aug_params(params: dict) -> None:
    """Monkey-patch RiceSeedDataset class attributes and CONFIG in place."""

    RiceSeedDataset._PROFILES = {
        "heavy": dict(
            band_drop = params["h_band_drop"],
            cutout    = params["h_cutout"],
            noise     = params["h_noise"],
            warp      = params["h_warp"],
            mult      = params["h_mult"],
        ),
        "medium": dict(
            band_drop = params["m_band_drop"],
            cutout    = params["m_cutout"],
            noise     = params["m_noise"],
            warp      = params["m_warp"],
            mult      = params["m_mult"],
        ),
        "light": dict(
            band_drop = params["l_band_drop"],
            cutout    = params["l_cutout"],
            noise     = params["l_noise"],
            warp      = params["l_warp"],
            mult      = params["l_mult"],
        ),
        "none": None,
    }
    RiceSeedDataset._INTENSITY_SCALE = {
        "heavy":  params["intensity_h"],
        "medium": params["intensity_m"],
        "light":  params["intensity_l"],
    }
    RiceSeedDataset._WARP_RANGE = {
        "heavy":  params["warp_range_h"],
        "medium": params["warp_range_m"],
        "light":  params["warp_range_l"],
    }
    CONFIG["noise_std"]        = params["noise_std"]
    CONFIG["max_cutout_bands"] = params["max_cutout_bands"]


def _restore_defaults() -> None:
    """Restore original augmentation parameters after a trial."""
    RiceSeedDataset._PROFILES        = copy.deepcopy(_ORIG_PROFILES)
    RiceSeedDataset._INTENSITY_SCALE = copy.deepcopy(_ORIG_INTENSITY_SCALE)
    RiceSeedDataset._WARP_RANGE      = copy.deepcopy(_ORIG_WARP_RANGE)
    CONFIG["noise_std"]        = _ORIG_NOISE_STD
    CONFIG["max_cutout_bands"] = _ORIG_MAX_CUTOUT


def _suggest_params(trial: optuna.Trial) -> dict:
    """
    Sample all augmentation hyper-parameters for one trial.

    Monotone hierarchy heavy ≥ medium ≥ light is enforced through
    multiplicative fractions, not via Optuna constraints — every sample
    is valid, so no trial is wasted.
    """
    tc = TUNE_CFG

    # ── Heavy (Phase 1) — absolute values ───────────────────────────────────
    h_band_drop = trial.suggest_float("h_band_drop", *tc["heavy_band_drop"])
    h_cutout    = trial.suggest_float("h_cutout",    *tc["heavy_cutout"])
    h_noise     = trial.suggest_float("h_noise",     *tc["heavy_noise"])
    h_warp      = trial.suggest_float("h_warp",      *tc["heavy_warp"])
    h_mult      = trial.suggest_float("h_mult",      *tc["heavy_mult"])

    # ── Medium (Phase 2) — fraction of heavy ────────────────────────────────
    mf_lo, mf_hi = tc["m_frac_range"]
    m_f_band_drop = trial.suggest_float("m_f_band_drop", mf_lo, mf_hi)
    m_f_cutout    = trial.suggest_float("m_f_cutout",    mf_lo, mf_hi)
    m_f_noise     = trial.suggest_float("m_f_noise",     mf_lo, mf_hi)
    m_f_warp      = trial.suggest_float("m_f_warp",      mf_lo, mf_hi)
    m_f_mult      = trial.suggest_float("m_f_mult",      mf_lo, mf_hi)

    m_band_drop = h_band_drop * m_f_band_drop
    m_cutout    = h_cutout    * m_f_cutout
    m_noise     = h_noise     * m_f_noise
    m_warp      = h_warp      * m_f_warp
    m_mult      = h_mult      * m_f_mult

    # ── Light (Phase 3) — fraction of medium ────────────────────────────────
    lf_lo, lf_hi = tc["l_frac_range"]
    l_f_band_drop = trial.suggest_float("l_f_band_drop", lf_lo, lf_hi)
    l_f_cutout    = trial.suggest_float("l_f_cutout",    lf_lo, lf_hi)
    l_f_noise     = trial.suggest_float("l_f_noise",     lf_lo, lf_hi)
    l_f_warp      = trial.suggest_float("l_f_warp",      lf_lo, lf_hi)
    l_f_mult      = trial.suggest_float("l_f_mult",      lf_lo, lf_hi)

    l_band_drop = m_band_drop * l_f_band_drop
    l_cutout    = m_cutout    * l_f_cutout
    l_noise     = m_noise     * l_f_noise
    l_warp      = m_warp      * l_f_warp
    l_mult      = m_mult      * l_f_mult

    # ── Intensity scale ──────────────────────────────────────────────────────
    intensity_h     = trial.suggest_float("intensity_h",     *tc["intensity_h_range"])
    intensity_m_frac = trial.suggest_float("intensity_m_frac", *tc["intensity_m_frac"])
    intensity_l_frac = trial.suggest_float("intensity_l_frac", *tc["intensity_l_frac"])
    intensity_m = intensity_h * intensity_m_frac
    intensity_l = intensity_m * intensity_l_frac

    # ── Spectral warp range ──────────────────────────────────────────────────
    warp_range_h      = trial.suggest_float("warp_range_h",      *tc["warp_range_h"])
    warp_range_m_frac = trial.suggest_float("warp_range_m_frac", *tc["warp_range_m_frac"])
    warp_range_l_frac = trial.suggest_float("warp_range_l_frac", *tc["warp_range_l_frac"])
    warp_range_m = warp_range_h * warp_range_m_frac
    warp_range_l = warp_range_m * warp_range_l_frac

    # ── Global noise / cutout ────────────────────────────────────────────────
    noise_std        = trial.suggest_float("noise_std",        *tc["noise_std_range"])
    max_cutout_bands = trial.suggest_int(  "max_cutout_bands", *tc["max_cutout_bands_range"])

    return dict(
        # heavy
        h_band_drop=h_band_drop, h_cutout=h_cutout, h_noise=h_noise,
        h_warp=h_warp, h_mult=h_mult,
        # medium
        m_band_drop=m_band_drop, m_cutout=m_cutout, m_noise=m_noise,
        m_warp=m_warp, m_mult=m_mult,
        # light
        l_band_drop=l_band_drop, l_cutout=l_cutout, l_noise=l_noise,
        l_warp=l_warp, l_mult=l_mult,
        # intensity
        intensity_h=intensity_h, intensity_m=intensity_m, intensity_l=intensity_l,
        # warp range
        warp_range_h=warp_range_h, warp_range_m=warp_range_m, warp_range_l=warp_range_l,
        # global
        noise_std=noise_std, max_cutout_bands=max_cutout_bands,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CONDENSED STAGE 1 TRIAL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _run_condensed_stage1(
    model:         nn.Module,
    ema:           ModelEMA,
    phase_loaders: Dict[int, _T.DataLoader],
    val_ldr:       _T.DataLoader,
    device:        torch.device,
    trial:         optuna.Trial,
    n_epochs:      int,
) -> float:
    """
    Mirrors run_stage1 from new_hs_train with three changes:
      1. Runs for n_epochs instead of CONFIG["s1_epochs"].
      2. Reports per-epoch F1 to Optuna → enables early pruning.
      3. No checkpoint I/O.
    """
    model.use_arcface(False)
    model.unfreeze_head("linear")
    model.freeze_head("arcface")

    ep_total = n_epochs
    p1_end   = max(1, int(ep_total * CONFIG["s1_phase1_frac"]))
    p2_end   = max(p1_end + 1,
                   int(ep_total * (CONFIG["s1_phase1_frac"] + CONFIG["s1_phase2_frac"])))
    patience = max(10, int(ep_total * TUNE_CFG["patience_frac"]))

    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ep_total, eta_min=CONFIG["s1_min_lr"]
        )

    scaler       = GradScaler()
    ls_hi        = CONFIG["s1_label_smooth_hi"]
    ls_lo        = CONFIG["s1_label_smooth_lo"]
    best_f1      = 0.0
    no_improve   = 0
    ema_reinited = [False, False]

    for ep in range(1, ep_total + 1):

        # ── Phase assignment ─────────────────────────────────────────────────
        if   ep <= p1_end: phase = 1
        elif ep <= p2_end: phase = 2
        else:              phase = 3

        # ── EMA re-init at phase boundaries ──────────────────────────────────
        if phase == 2 and not ema_reinited[0] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            ema_reinited[0] = True
        if phase == 3 and not ema_reinited[1] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            ema_reinited[1] = True

        cur_ldr = phase_loaders[phase]

        # ── Loss function (mirrors run_stage1 exactly) ────────────────────────
        t      = (ep - 1) / max(ep_total - 1, 1)
        ls_now = ls_hi * (1.0 - t) + ls_lo * t
        if phase == 3:
            crit = FocalLoss(gamma=CONFIG["s1_focal_gamma"], label_smoothing=ls_now)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=ls_now)

        use_mx = (phase != 3)

        train_one_epoch(
            model, cur_ldr, optimizer, crit, scaler, ema, device,
            scheduler=None,
            use_mixup=use_mx,
            mixup_alpha=CONFIG["s1_mixup"],
            accum_steps=CONFIG["s1_accum"],
            current_ep=ep,
            total_ep=ep_total,
        )
        scheduler.step()

        # ── Evaluate ─────────────────────────────────────────────────────────
        f1_live, _ = evaluate(model,      val_ldr, device)
        f1_ema,  _ = evaluate(ema.shadow, val_ldr, device)
        ep_f1      = max(f1_live, f1_ema)

        if ep_f1 > best_f1:
            best_f1    = ep_f1
            no_improve = 0
        else:
            no_improve += 1

        # ── Report to Optuna every `pruner_interval` epochs ──────────────────
        if ep % TUNE_CFG["pruner_interval"] == 0:
            trial.report(ep_f1, step=ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned(
                    f"Pruned at epoch {ep} (F1={ep_f1:.4f})"
                )

        if no_improve >= patience:
            break

    return best_f1


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

# Build splits once — reused across all trials
_ALL_LABELS, _TRAIN_IDX, _VAL_IDX, _TEST_IDX = None, None, None, None


def _ensure_splits() -> None:
    global _ALL_LABELS, _TRAIN_IDX, _VAL_IDX, _TEST_IDX
    if _TRAIN_IDX is not None:
        return
    _ALL_LABELS, _TRAIN_IDX, _VAL_IDX, _TEST_IDX = build_splits()
    print(f"[TUNE] Splits: train={len(_TRAIN_IDX):,}  val={len(_VAL_IDX):,}  "
          f"test={len(_TEST_IDX):,}")


def objective(trial: optuna.Trial) -> float:
    device = CONFIG["device"]

    # Different seed per trial → diverse augmentation sampling
    set_seed(CONFIG["seed"] + trial.number * 7)

    # ── Sample hyper-parameters ──────────────────────────────────────────────
    params = _suggest_params(trial)
    _patch_aug_params(params)

    # ── Store derived (actual) values as Optuna user attributes ─────────────
    for k in ("m_band_drop", "m_cutout", "m_noise", "m_warp", "m_mult",
              "l_band_drop", "l_cutout", "l_noise", "l_warp", "l_mult",
              "intensity_m", "intensity_l", "warp_range_m", "warp_range_l"):
        trial.set_user_attr(k, float(params[k]))

    try:
        _ensure_splits()

        # ── Build per-phase DataLoaders ──────────────────────────────────────
        def _ldr(aug: str) -> _T.DataLoader:
            ds = RiceSeedDataset(_TRAIN_IDX, aug_strength=aug)
            return _T.DataLoader(
                ds, batch_size=CONFIG["s1_batch"],
                shuffle=True, drop_last=True, num_workers=0,
            )

        phase_loaders = {1: _ldr("heavy"), 2: _ldr("medium"), 3: _ldr("light")}
        _, val_ldr, _ = build_loaders(
            _TRAIN_IDX, _VAL_IDX, _TEST_IDX,
            CONFIG["s1_batch"], train_aug="none",
        )

        # ── Fresh model + EMA ────────────────────────────────────────────────
        model = SpectralQuadNet(
            num_classes = CONFIG["num_classes"],
            num_bands   = CONFIG["num_bands"],
            dropout     = CONFIG["s1_dropout"],
            wl_embed_dim= CONFIG["wl_embed_dim"],
            cfg         = CONFIG,
        ).to(device)
        ema = ModelEMA(model, decay=CONFIG["ema_decay"])

        t0     = time.time()
        best_f1 = _run_condensed_stage1(
            model, ema, phase_loaders, val_ldr, device,
            trial, TUNE_CFG["n_epochs"],
        )
        elapsed = time.time() - t0

        print(
            f"[Trial {trial.number:03d}]  F1={best_f1:.4f}  "
            f"({elapsed/60:.1f} min)  "
            f"h_band_drop={params['h_band_drop']:.3f}  "
            f"h_noise={params['h_noise']:.3f}  "
            f"noise_std={params['noise_std']:.4f}"
        )
        return best_f1

    except optuna.exceptions.TrialPruned:
        raise

    except Exception as exc:
        print(f"[Trial {trial.number}] FAILED: {exc}")
        return 0.0

    finally:
        # ── Always restore defaults so the next trial starts clean ──────────
        _restore_defaults()
        try:
            del model, ema
        except NameError:
            pass
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _resolved_params(trial: optuna.Trial) -> dict:
    """
    Reconstruct the actual (derived) parameter values from a completed trial.
    Optuna stores the sampled fractions; we recompute the actual aug values.
    """
    p = trial.params
    ua = trial.user_attrs  # derived values stored during the trial

    h = dict(
        band_drop = p["h_band_drop"],
        cutout    = p["h_cutout"],
        noise     = p["h_noise"],
        warp      = p["h_warp"],
        mult      = p["h_mult"],
    )
    m = dict(
        band_drop = ua.get("m_band_drop", h["band_drop"] * p.get("m_f_band_drop", 0.6)),
        cutout    = ua.get("m_cutout",    h["cutout"]    * p.get("m_f_cutout",    0.6)),
        noise     = ua.get("m_noise",     h["noise"]     * p.get("m_f_noise",     0.6)),
        warp      = ua.get("m_warp",      h["warp"]      * p.get("m_f_warp",      0.6)),
        mult      = ua.get("m_mult",      h["mult"]      * p.get("m_f_mult",      0.6)),
    )
    l = dict(
        band_drop = ua.get("l_band_drop", m["band_drop"] * p.get("l_f_band_drop", 0.0)),
        cutout    = ua.get("l_cutout",    m["cutout"]    * p.get("l_f_cutout",    0.0)),
        noise     = ua.get("l_noise",     m["noise"]     * p.get("l_f_noise",     0.0)),
        warp      = ua.get("l_warp",      m["warp"]      * p.get("l_f_warp",      0.0)),
        mult      = ua.get("l_mult",      m["mult"]      * p.get("l_f_mult",      0.0)),
    )
    intensity_h = p["intensity_h"]
    intensity_m = ua.get("intensity_m", intensity_h * p.get("intensity_m_frac", 0.6))
    intensity_l = ua.get("intensity_l", intensity_m * p.get("intensity_l_frac", 0.0))

    warp_range_h = p["warp_range_h"]
    warp_range_m = ua.get("warp_range_m", warp_range_h * p.get("warp_range_m_frac", 0.6))
    warp_range_l = ua.get("warp_range_l", warp_range_m * p.get("warp_range_l_frac", 0.0))

    return dict(
        heavy=h, medium=m, light=l,
        intensity=dict(heavy=intensity_h, medium=intensity_m, light=intensity_l),
        warp_range=dict(heavy=warp_range_h, medium=warp_range_m, light=warp_range_l),
        noise_std=p["noise_std"],
        max_cutout_bands=int(p["max_cutout_bands"]),
    )


def print_best_params(study: optuna.Study) -> None:
    """Print best params in a copy-paste-ready block for new_hs_train.py."""
    trial = study.best_trial
    rp    = _resolved_params(trial)
    h, m, l_ = rp["heavy"], rp["medium"], rp["light"]
    inten     = rp["intensity"]
    wr        = rp["warp_range"]

    bar = "═" * 72
    print(f"\n{bar}")
    print(f"  BEST TRIAL  #{trial.number}   val macro-F1 = {trial.value:.4f}")
    print(f"{bar}")
    print(f"\n── Copy these into RiceSeedDataset in new_hs_train.py ──────────────\n")
    print(
        f'    _PROFILES = {{\n'
        f'        # Phase 1 — representation shaping (tuned)\n'
        f'        "heavy": dict(\n'
        f'            band_drop={h["band_drop"]:.4f}, cutout={h["cutout"]:.4f},\n'
        f'            noise={h["noise"]:.4f}, warp={h["warp"]:.4f}, mult={h["mult"]:.4f},\n'
        f'        ),\n'
        f'        # Phase 2 — robustness consolidation (tuned)\n'
        f'        "medium": dict(\n'
        f'            band_drop={m["band_drop"]:.4f}, cutout={m["cutout"]:.4f},\n'
        f'            noise={m["noise"]:.4f}, warp={m["warp"]:.4f}, mult={m["mult"]:.4f},\n'
        f'        ),\n'
        f'        # Phase 3 — fine refinement (tuned)\n'
        f'        "light": dict(\n'
        f'            band_drop={l_["band_drop"]:.4f}, cutout={l_["cutout"]:.4f},\n'
        f'            noise={l_["noise"]:.4f}, warp={l_["warp"]:.4f}, mult={l_["mult"]:.4f},\n'
        f'        ),\n'
        f'        "none": None,\n'
        f'    }}\n'
    )
    print(
        f'    _INTENSITY_SCALE = {{\n'
        f'        "heavy":  {inten["heavy"]:.4f},\n'
        f'        "medium": {inten["medium"]:.4f},\n'
        f'        "light":  {inten["light"]:.4f},\n'
        f'    }}\n'
    )
    print(
        f'    _WARP_RANGE = {{\n'
        f'        "heavy":  {wr["heavy"]:.4f},\n'
        f'        "medium": {wr["medium"]:.4f},\n'
        f'        "light":  {wr["light"]:.4f},\n'
        f'    }}\n'
    )
    print(f'── Also update CONFIG ────────────────────────────────────────────────\n')
    print(f'    "noise_std":        {rp["noise_std"]:.4f},')
    print(f'    "max_cutout_bands": {rp["max_cutout_bands"]},')
    print(f"\n{bar}\n")

    # Save to JSON as well
    out_path = Path("aug_hpo_best_params.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "trial_number": trial.number,
                "val_f1":       trial.value,
                "profiles": {
                    "heavy":  h,
                    "medium": m,
                    "light":  l_,
                },
                "intensity_scale": inten,
                "warp_range":      wr,
                "noise_std":       rp["noise_std"],
                "max_cutout_bands": rp["max_cutout_bands"],
                "raw_optuna_params": trial.params,
            },
            f, indent=2,
        )
    print(f"[TUNE] Best params saved → {out_path.resolve()}")


def print_top_k_summary(study: optuna.Study, k: int = 10) -> None:
    """Print a ranked table of the top-k completed trials."""
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed.sort(key=lambda t: t.value, reverse=True)
    top = completed[:k]
    if not top:
        print("[TUNE] No completed trials yet.")
        return

    print(f"\n{'─'*72}")
    print(f"  Top-{len(top)} trials (by val macro-F1)")
    print(f"{'─'*72}")
    hdr = (f"{'Rank':>4}  {'Trial':>5}  {'F1':>7}  "
           f"{'h_drop':>7}  {'h_cut':>6}  {'h_nz':>6}  "
           f"{'m_drop':>7}  {'noise_std':>9}  {'max_cut':>7}")
    print(hdr)
    print("─" * len(hdr))
    for rank, t in enumerate(top, 1):
        p  = t.params
        ua = t.user_attrs
        print(
            f"{rank:>4}  {t.number:>5}  {t.value:>7.4f}  "
            f"{p.get('h_band_drop', 0):>7.4f}  {p.get('h_cutout', 0):>6.4f}  "
            f"{p.get('h_noise', 0):>6.4f}  "
            f"{ua.get('m_band_drop', 0):>7.4f}  "
            f"{p.get('noise_std', 0):>9.4f}  "
            f"{int(p.get('max_cutout_bands', 0)):>7}"
        )
    print(f"{'─'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:

    # ── Override TUNE_CFG from CLI ───────────────────────────────────────────
    TUNE_CFG["n_trials"]   = args.n_trials
    TUNE_CFG["n_epochs"]   = args.n_epochs
    TUNE_CFG["study_name"] = args.study_name
    if args.storage:
        TUNE_CFG["storage"] = args.storage

    # ── Load dataset into shared memory (once) ───────────────────────────────
    _T._load_data_mmap(CONFIG["patches_data"], CONFIG["labels_path"])
    _T._load_wavelengths_to_gpu(CONFIG["wavelength_path"], CONFIG["device"])

    if CONFIG["device"].type == "cuda":
        props = torch.cuda.get_device_properties(CONFIG["device"])
        print(f"[TUNE] GPU: {props.name}  VRAM={props.total_memory//1024**3} GB")

    print(f"[TUNE] Study     : {TUNE_CFG['study_name']}")
    print(f"[TUNE] Storage   : {TUNE_CFG['storage']}")
    print(f"[TUNE] Trials    : {TUNE_CFG['n_trials']}")
    print(f"[TUNE] Epochs/trial: {TUNE_CFG['n_epochs']}  "
          f"(~{TUNE_CFG['n_epochs']/CONFIG['s1_epochs']*100:.0f} % of full run)")
    print(f"[TUNE] Parameters: "
          f"5×heavy + 5×m_frac + 5×l_frac + 3×intensity + 3×warp_range "
          f"+ noise_std + max_cutout_bands  = 24 dims")
    print(f"[TUNE] Pruner    : MedianPruner  "
          f"(startup={TUNE_CFG['pruner_n_startup']}  "
          f"warmup={TUNE_CFG['pruner_n_warmup']} ep)")

    # ── Silence Optuna's info logging (keep warnings+) ───────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Create / resume study ─────────────────────────────────────────────────
    sampler = TPESampler(
        n_startup_trials = 10,          # random exploration before TPE kicks in
        multivariate     = True,        # models correlations between params
        seed             = CONFIG["seed"],
    )
    pruner = MedianPruner(
        n_startup_trials = TUNE_CFG["pruner_n_startup"],
        n_warmup_steps   = TUNE_CFG["pruner_n_warmup"],
        interval_steps   = TUNE_CFG["pruner_interval"],
    )

    study = optuna.create_study(
        study_name  = TUNE_CFG["study_name"],
        storage     = TUNE_CFG["storage"],
        direction   = "maximize",
        sampler     = sampler,
        pruner      = pruner,
        load_if_exists = True,          # auto-resume on restart
    )

    already_done = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    if already_done > 0:
        print(f"[TUNE] Resuming study — {already_done} trials already complete, "
              f"best so far = {study.best_value:.4f}")

    # ── Run optimisation ──────────────────────────────────────────────────────
    study.optimize(
        objective,
        n_trials         = TUNE_CFG["n_trials"],
        n_jobs           = TUNE_CFG["n_jobs"],
        show_progress_bar= False,
        gc_after_trial   = True,        # force GC between trials
    )

    # ── Final reporting ───────────────────────────────────────────────────────
    print_top_k_summary(study, k=10)
    print_best_params(study)

    total_completed = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    total_pruned = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ])
    print(f"[TUNE] Finished: {total_completed} complete, {total_pruned} pruned.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna HPO for Stage 1 augmentation parameters"
    )
    parser.add_argument(
        "--n_trials", type=int, default=TUNE_CFG["n_trials"],
        help=f"Total Optuna trials (default: {TUNE_CFG['n_trials']})",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=TUNE_CFG["n_epochs"],
        help=f"Epochs per trial (default: {TUNE_CFG['n_epochs']})",
    )
    parser.add_argument(
        "--study_name", type=str, default=TUNE_CFG["study_name"],
        help=f"Optuna study name (default: {TUNE_CFG['study_name']})",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="SQLite URL override (default: aug_hpo_stage1.db)",
    )
    args = parser.parse_args()

    import traceback, logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("aug_hpo_stage1.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n[TUNE] Interrupted by user — partial results saved.")
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)