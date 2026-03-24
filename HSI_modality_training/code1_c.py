# code
from __future__ import annotations

import contextlib
import copy
import json as _json
import logging
import math
import os
import random
import shutil
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

os.environ["NETWORKX_BACKEND"] = "nx-loopback"
os.environ["PYTHONWARNINGS"]   = "ignore"
warnings.filterwarnings("ignore", module="networkx")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Online softmax is disabled.*")
# silence optuna's own INFO spam – we write our own logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════
#  BASELINE CONFIG  (used as the canonical default / search centre)
# ══════════════════════════════════════════════════════════════════════

WL_MIN: float = 385.0
WL_MAX: float = 1000.0

CONFIG: dict = {
    # ── Paths ─────────────────────────────────────────────────────────
    "patches_data":    "./dataset/patches.npy",
    "labels_path":     "./dataset/labels.npy",
    "wavelength_path": "./dataset/wavelengths.csv",
    "output_dir":      "./output_v10/",

    # ── Dataset ───────────────────────────────────────────────────────
    "num_bands":       256,
    "num_classes":     90,

    # ── Stage 1 — 3-Phase Progressive Augmentation ────────────────────
    "s1_epochs":            400,
    "s1_phase1_frac":       0.15,
    "s1_phase2_frac":       0.35,

    "s1_batch":             128,
    "s1_max_lr":            5e-4,
    "s1_min_lr":            1e-6,
    "s1_dropout":           0.10,
    "s1_mixup":             0.10,
    "s1_patience":          120,
    "s1_accum":             1,
    "s1_focal_gamma":       1.5,
    "s1_label_smooth_hi":   0.00,
    "s1_label_smooth_lo":   0.00,
    "s1_ema_reinit_phases": True,

    # ── Stage 1 · Phase 3 — Hard-Class Oversampling ───────────────────
    "s1_p3_oversample":         False,
    "s1_p3_oversample_power":   0.40,
    "s1_p3_oversample_max_w":   5.0,
    "s1_p3_hard_f1_thresh":     0.50,
    "s1_p3_oversample_eps":     0.05,

    # ── Architecture ──────────────────────────────────────────────────
    "branch_drop_prob":    0.20,
    "subcenter_K":          3,
    "max_cutout_bands":     8,
    "noise_scale":          0.02,   # absolute noise std used by RiceSeedDataset

    # ── Auxiliary Classification Heads (per branch, Stage 1) ──────────
    "aux_head_hidden":       128,
    "aux_loss_weight_init":  0.50,
    "aux_loss_weight_final": 0.15,

    # ── Stage 2 ───────────────────────────────────────────────────────
    "s2_epochs":            120,
    "s2_batch":             128,
    "s2_head_lr":           2.5e-4,
    "s2_back_lr":           2.5e-5,
    "s2_min_lr":            1e-6,
    "s2_warmup_ep":          5,
    "s2_sgdr_T0":           10,
    "s2_sgdr_Tmult":         2,
    "s2_dropout":            0.10,
    "s2_patience":           40,
    "s2_arcface_s":         32.0,
    "s2_arcface_m":          0.35,
    "s2_arcface_m0":         0.02,
    "s2_arcface_m_delta":    0.10,
    "s2_margin_warmup_ep":   50,
    "s2_focal_gamma":         1.5,
    "cdws_max_weight":        3.0,
    "cdws_eps":               0.05,
    "supcon_weight":           0.25,
    "supcon_temp":             0.10,
    "proto_weight":            0.12,
    "proto_temp":              0.10,
    "bal_n_cls":               16,
    "bal_n_spc":                8,

    # ── Stage 3 ───────────────────────────────────────────────────────
    "s3_epochs":            100,
    "s3_swa_lr":            4e-5,
    "s3_cycle_len":           8,
    "s3_sam_rho":             0.05,
    "s3_greedy":            True,
    "s3_aux_loss_weight":    0.10,

    # ── Shared ────────────────────────────────────────────────────────
    "weight_decay":          2e-4,
    "grad_clip":              1.0,
    "ema_decay":             0.999,

    # ── TTA ───────────────────────────────────────────────────────────
    "tta_spatial":             8,
    "tta_spectral":            4,

    # ── Transformer Branch (SpecFormer) ───────────────────────────────
    "wl_embed_dim":           16,
    "specf_patch":            32,
    "specf_dim":             256,
    "specf_heads":             8,
    "specf_layers":            4,
    "specf_drop":             0.15,
    "fusion_heads":            4,
    "fusion_drop":            0.10,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":   42,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True

_GPU_PATCHES:   Optional[np.ndarray]   = None
_GLOBAL_LABELS: Optional[np.ndarray]   = None
_PHYSICAL_WL:   Optional[torch.Tensor] = None


# ══════════════════════════════════════════════════════════════════════
#  TUNING CONFIGURATION  (centralized — all Optuna knobs live here)
# ══════════════════════════════════════════════════════════════════════

TUNING_CONFIG: dict = {
    # ── Study settings ───────────────────────────────────────────────
    "n_trials_s1":          100,   # Optuna trials for Stage 1
    "n_trials_s2":          40,    # Optuna trials for Stage 2
    "n_trials_s3":          20,    # Optuna trials for Stage 3
    "n_startup_trials":      5,    # random trials before TPE kicks in

    # ── Directory layout ─────────────────────────────────────────────
    "tuning_dir":        "./optuna_tuning/",
    "trials_dir":        "./optuna_tuning/trials/",

    # ── Optuna SQLite storage (enables resume after crash) ────────────
    "s1_db":         "sqlite:///./optuna_tuning/study_s1.db",
    "s2_db":         "sqlite:///./optuna_tuning/study_s2.db",
    "s3_db":         "sqlite:///./optuna_tuning/study_s3.db",
    "s1_study_name": "hyperspectral_stage1",
    "s2_study_name": "hyperspectral_stage2",
    "s3_study_name": "hyperspectral_stage3",

    # ── Best config JSON paths (stage-level resume gate) ─────────────
    "best_config_s1": "./optuna_tuning/best_config_s1.json",
    "best_config_s2": "./optuna_tuning/best_config_s2.json",
    "best_config_s3": "./optuna_tuning/best_config_s3.json",

    # ── Best model checkpoint paths ───────────────────────────────────
    "best_ckpt_s1":  "./optuna_tuning/best_ckpt_s1.pth",
    "best_ckpt_s2":  "./optuna_tuning/best_ckpt_s2.pth",
    "best_ckpt_s3":  "./optuna_tuning/best_ckpt_s3.pth",

    # ── Optuna pruner ─────────────────────────────────────────────────
    # "none" → NopPruner (all trials run to completion via patience)
    # Pruning is disabled: every trial runs the full epoch budget and relies
    # on the patience mechanism for early stopping, not Optuna's pruner.
    "pruner":          "none",

    # ── Optuna sampler ────────────────────────────────────────────────
    # "tpe"    → Tree-structured Parzen Estimator (recommended)
    # "random" → RandomSampler
    # "cmaes"  → CmaEsSampler
    "sampler":         "tpe",

    # ── Misc ──────────────────────────────────────────────────────────
    "n_jobs":           1,      # parallel trials (>1 requires multi-GPU)
    "timeout_s1":    None,      # seconds limit per stage, None = unlimited
    "timeout_s2":    None,
    "timeout_s3":    None,
    "keep_trial_ckpts": False,  # delete per-trial subdirs after each trial

    # ══════════════════════════════════════════════════════════════════
    #  CENTRALIZED HYPERPARAMETER SEARCH SPACES
    #  All Optuna suggest_* ranges live here.  Modify ranges in ONE place
    #  and suggest_s{1,2,3}_params() will automatically pick them up.
    # ══════════════════════════════════════════════════════════════════

    # ── Stage 1 search space ─────────────────────────────────────────
    "s1_space": {
        # SpecFormer architecture (dim×heads joint)
        "specf_dim_heads":          ["128_4", "128_8", "192_4", "192_8", "256_4", "256_8"],
        "specf_layers":             [2, 3, 4, 6],            # categorical
        "specf_drop":               (0.05, 0.10),
        "specf_patch":              [16, 32, 48],
        "wl_embed_dim":             [4, 8, 12],
        "fusion_drop":              (0.05, 0.20),
        "fusion_heads":             [2, 4, 8],
        "subcenter_K":              [2, 3, 4],
        "branch_drop_prob":         (0.25, 0.50),
        # ArcFace head (baked at construction, used in S2)
        "s2_arcface_s":             (28.0, 40.0),
        "s2_arcface_m":             (0.30, 0.45),
        "s2_arcface_m_delta":       (0.08, 0.15),
        # Phase curriculum fractions (direct, phase3 = 1 - p1 - p2)
        "s1_phase1_frac":           (0.08, 0.22),
        "s1_phase2_frac":           (0.22, 0.55),
        # Augmentation intensity scales per profile
        # Each entry: (min, max) multiplier applied to base profile probs
        "aug_heavy_band_drop":      (0.05, 0.15),
        "aug_heavy_cutout":         (0.04, 0.12),
        "aug_heavy_noise":          (0.02, 0.08),
        "aug_heavy_warp":           (0.01, 0.06),
        "aug_heavy_mult":           (0.03, 0.08),
        "aug_medium_band_drop":     (0.02, 0.08),
        "aug_medium_cutout":        (0.02, 0.07),
        "aug_medium_noise":         (0.01, 0.05),
        "aug_medium_warp":          (0.01, 0.03),
        "aug_medium_mult":          (0.01, 0.05),
        "aug_light_band_drop":      (0.0,  0.03),
        "aug_light_cutout":         (0.0,  0.03),
        "aug_light_noise":          (0.0,  0.02),
        # Shared aug scale params (applied on top of profile probs)
        "band_drop_scale":          (0.5,  2.0),
        "noise_scale":              (0.01, 0.10),
        "cutout_scale":             (0.5,  2.0),
        "max_cutout_bands":         (4,    20),   # int range
        # Training dynamics
        "s1_max_lr":                (3e-4, 1e-3),
        "s1_min_lr":                (1e-7, 1e-6),
        "s1_batch":                 [64, 128, 256],
        "s1_dropout":               (0.20, 0.40),
        "s1_mixup":                 (0.30, 0.55),
        "s1_accum":                 [1, 2, 4],
        "s1_focal_gamma":           (1.0,  3.0),
        "s1_label_smooth_hi":       (0.0,  0.15),
        "s1_label_smooth_lo":       (0.0,  0.05),
        "s1_ema_reinit_phases":     [True, False],
        # Phase-3 oversampling
        "s1_p3_oversample":         [True, False],
        "s1_p3_oversample_power":   (0.20, 0.80),
        "s1_p3_oversample_max_w":   (3.0,  8.0),
        "s1_p3_hard_f1_thresh":     (0.30, 0.70),
        "s1_p3_oversample_eps":     (0.01, 0.15),
        # Aux heads
        "aux_head_hidden":          [64, 128, 256],
        "aux_loss_weight_init":     (0.60, 0.85),
        "aux_loss_weight_final":    (0.10, 0.20),
        # Shared regularisation
        "weight_decay":             (1e-5, 2e-4),
        "ema_decay":                (0.9985, 0.9999),
        "grad_clip":                (0.5,  5.0),
    },

    # ── Stage 2 search space ─────────────────────────────────────────
    "s2_space": {
        "s2_head_lr":               (1e-4,  1e-3),
        "s2_back_lr":               (5e-6,  1e-4),
        "s2_min_lr":                (1e-7,  1e-5),
        "s2_batch":                 [64, 128, 256],
        "s2_warmup_ep":             (2,    10),
        "s2_sgdr_T0":               (5,    20),
        "s2_sgdr_Tmult":            [1, 2],
        "s2_dropout":               (0.05, 0.25),
        "s2_arcface_m":             (0.20, 0.50),
        "s2_arcface_m0":            (0.01, 0.05),
        "s2_arcface_m_delta":       (0.05, 0.20),
        "s2_margin_warmup_ep":      (20,   80),
        "s2_focal_gamma":           (1.0,  3.0),
        "cdws_max_weight":          (1.5,  6.0),
        "cdws_eps":                 (0.02, 0.10),
        "supcon_weight":            (0.05, 0.50),
        "supcon_temp":              (0.05, 0.30),
        "proto_weight":             (0.02, 0.30),
        "proto_temp":               (0.05, 0.30),
        "bal_n_cls":                (8,    24),
        "bal_n_spc":                (4,    12),
        "weight_decay":             (1e-5, 5e-3),
        "grad_clip":                (0.5,  5.0),
        "ema_decay":                (0.990, 0.9999),
    },

    # ── Stage 3 search space ─────────────────────────────────────────
    "s3_space": {
        "s3_swa_lr":                (5e-6, 2e-4),
        "s3_cycle_len":             (4,    16),
        "s3_sam_rho":               (0.01, 0.15),
        "s3_greedy":                [True, False],
        "s3_aux_loss_weight":       (0.02, 0.25),
        "weight_decay":             (1e-5, 5e-3),
        "grad_clip":                (0.5,  5.0),
    },
}

# Architecture keys that are baked into the model graph and must be
# preserved from Stage-1 best config when building models for S2/S3.
_ARCH_KEYS: Tuple[str, ...] = (
    "specf_dim", "specf_heads", "specf_layers", "specf_drop",
    "specf_patch", "wl_embed_dim", "fusion_drop", "fusion_heads",
    "subcenter_K", "aux_head_hidden", "branch_drop_prob",
    "s2_arcface_s", "s2_arcface_m", "s2_arcface_m_delta",
    "num_bands", "num_classes",
)


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def _load_data_mmap(patches_path: str, labels_path: str) -> None:
    global _GPU_PATCHES, _GLOBAL_LABELS
    if _GPU_PATCHES is not None:
        return
    print("[DATA] Memory-mapping dataset from disk (Zero-RAM footprint)...")
    _GPU_PATCHES   = np.load(patches_path, mmap_mode='r')
    _GLOBAL_LABELS = np.load(labels_path)
    print(f"[DATA] ✓ Indexed {_GPU_PATCHES.shape[0]} samples via mmap.")


def _load_wavelengths_to_gpu(csv_path: str, device: torch.device) -> None:
    global _PHYSICAL_WL
    if _PHYSICAL_WL is not None:
        return
    print("[DATA] Loading physical wavelengths from CSV...")
    try:
        df      = pd.read_csv(csv_path, sep=None, engine="python")
        raw_wl  = df.iloc[:, -1].values.astype(np.float32)
        wl_norm = (raw_wl - raw_wl.min()) / (raw_wl.max() - raw_wl.min())
        _PHYSICAL_WL = torch.from_numpy(wl_norm).to(device)
        print(f"[DATA] ✓ Loaded physical wavelengths: {_PHYSICAL_WL.size(0)} bands.")
    except Exception as exc:
        raise RuntimeError(f"Failed to load wavelengths.csv: {exc}")


# ══════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

set_seed(CONFIG["seed"])


# ══════════════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.max_decay    = decay
        self._num_updates = 0
        self.shadow       = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @property
    def current_decay(self) -> float:
        n = self._num_updates
        return min(self.max_decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._num_updates += 1
        d  = self.current_decay
        lp = dict(model.named_parameters())
        for n, sp in self.shadow.named_parameters():
            if n in lp:
                sp.copy_(d * sp + (1.0 - d) * lp[n])
        lb = dict(model.named_buffers())
        for n, sb in self.shadow.named_buffers():
            if n in lb and sb.dtype.is_floating_point:
                sb.copy_(lb[n])

    def reinit_from(self, model: nn.Module) -> None:
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """Hyperspectral Rice Seed Dataset with phase-aware spectral+spatial aug.

    Augmentation profile probabilities and intensity scales are fully
    configurable through the constructor (and therefore tunable by Optuna).
    All per-profile parameters fall back to CONFIG defaults when not supplied.

    Per-profile probability keys (band_drop, cutout, noise, warp, mult):
        aug_heavy_*  / aug_medium_*  / aug_light_*  in CONFIG

    Global scale parameters (applied on top of profile probs):
        band_drop_scale  — multiplier on the band-dropout probability
        noise_scale      — absolute noise std (tunable; falls back to CONFIG["noise_scale"])
        cutout_scale     — multiplier on the cutout probability
        max_cutout_bands — maximum consecutive bands to zero out
    """

    # ── Default profile probabilities (overridden by CONFIG / constructor) ──
    _DEFAULT_PROFILES = {
        "heavy":  dict(band_drop=0.08, cutout=0.06, noise=0.04, warp=0.03, mult=0.05),
        "medium": dict(band_drop=0.05, cutout=0.04, noise=0.03, warp=0.02, mult=0.03),
        "light":  dict(band_drop=0.0,  cutout=0.0,  noise=0.0,  warp=0.0,  mult=0.0),
        "none":   None,
    }
    _DEFAULT_INTENSITY_SCALE = {"heavy": 1.0, "medium": 0.7, "light": 0.4}
    _DEFAULT_WARP_RANGE      = {"heavy": 0.05, "medium": 0.03, "light": 0.0}

    def __init__(
        self,
        indices:          np.ndarray,
        aug_strength:     str   = "none",
        # ── Per-profile probability overrides (None → use CONFIG / default) ──
        aug_heavy_band_drop:  Optional[float] = None,
        aug_heavy_cutout:     Optional[float] = None,
        aug_heavy_noise:      Optional[float] = None,
        aug_heavy_warp:       Optional[float] = None,
        aug_heavy_mult:       Optional[float] = None,
        aug_medium_band_drop: Optional[float] = None,
        aug_medium_cutout:    Optional[float] = None,
        aug_medium_noise:     Optional[float] = None,
        aug_medium_warp:      Optional[float] = None,
        aug_medium_mult:      Optional[float] = None,
        aug_light_band_drop:  Optional[float] = None,
        aug_light_cutout:     Optional[float] = None,
        aug_light_noise:      Optional[float] = None,
        # ── Global scale params (None → use CONFIG defaults) ────────────────
        band_drop_scale:  Optional[float] = None,
        noise_scale:      Optional[float] = None,
        cutout_scale:     Optional[float] = None,
        max_cutout_bands: Optional[int]   = None,
    ) -> None:
        self.patches      = _GPU_PATCHES
        self.labels       = _GLOBAL_LABELS
        self.indices      = indices
        self.aug_strength = str(aug_strength)

        # ── Build per-profile probability dict from overrides → CONFIG → default ──
        def _p(key: str, default: float) -> float:
            cfg_val = CONFIG.get(key)
            return cfg_val if cfg_val is not None else default

        heavy_base  = self._DEFAULT_PROFILES["heavy"]
        medium_base = self._DEFAULT_PROFILES["medium"]
        light_base  = self._DEFAULT_PROFILES["light"]

        self._profiles: Dict[str, Optional[dict]] = {
            "heavy": dict(
                band_drop = aug_heavy_band_drop  if aug_heavy_band_drop  is not None else _p("aug_heavy_band_drop",  heavy_base["band_drop"]),
                cutout    = aug_heavy_cutout     if aug_heavy_cutout     is not None else _p("aug_heavy_cutout",     heavy_base["cutout"]),
                noise     = aug_heavy_noise      if aug_heavy_noise      is not None else _p("aug_heavy_noise",      heavy_base["noise"]),
                warp      = aug_heavy_warp       if aug_heavy_warp       is not None else _p("aug_heavy_warp",       heavy_base["warp"]),
                mult      = aug_heavy_mult       if aug_heavy_mult       is not None else _p("aug_heavy_mult",       heavy_base["mult"]),
            ),
            "medium": dict(
                band_drop = aug_medium_band_drop if aug_medium_band_drop is not None else _p("aug_medium_band_drop", medium_base["band_drop"]),
                cutout    = aug_medium_cutout    if aug_medium_cutout    is not None else _p("aug_medium_cutout",    medium_base["cutout"]),
                noise     = aug_medium_noise     if aug_medium_noise     is not None else _p("aug_medium_noise",     medium_base["noise"]),
                warp      = aug_medium_warp      if aug_medium_warp      is not None else _p("aug_medium_warp",      medium_base["warp"]),
                mult      = aug_medium_mult      if aug_medium_mult      is not None else _p("aug_medium_mult",      medium_base["mult"]),
            ),
            "light": dict(
                band_drop = aug_light_band_drop  if aug_light_band_drop  is not None else _p("aug_light_band_drop",  light_base["band_drop"]),
                cutout    = aug_light_cutout     if aug_light_cutout     is not None else _p("aug_light_cutout",     light_base["cutout"]),
                noise     = aug_light_noise      if aug_light_noise      is not None else _p("aug_light_noise",      light_base["noise"]),
                warp      = 0.0,   # light phase: no warp
                mult      = 0.0,   # light phase: no multiplicative noise
            ),
            "none": None,
        }
        self.profile = self._profiles.get(self.aug_strength)

        # Intensity scale and warp range stay at fixed defaults per profile
        self.intensity_scale = self._DEFAULT_INTENSITY_SCALE.get(self.aug_strength, 0.0)
        self.warp_range      = self._DEFAULT_WARP_RANGE.get(self.aug_strength, 0.0)

        # ── Global scale params ─────────────────────────────────────────────
        self.band_drop_scale  = band_drop_scale  if band_drop_scale  is not None else float(CONFIG.get("band_drop_scale",  1.0))
        self.noise_scale      = noise_scale      if noise_scale      is not None else float(CONFIG.get("noise_scale", 0.02))
        self.cutout_scale     = cutout_scale     if cutout_scale     is not None else float(CONFIG.get("cutout_scale",     1.0))
        self.max_cutout_bands = max_cutout_bands if max_cutout_bands is not None else int(CONFIG.get("max_cutout_bands", 8))

    def __len__(self) -> int:
        return len(self.indices)

    def _band_dropout(self, x: torch.Tensor, prob: float) -> torch.Tensor:
        C    = x.shape[0]
        # band_drop_scale multiplies the raw profile probability
        effective_prob = min(prob * self.band_drop_scale, 1.0)
        mask = (torch.rand(C, device=x.device) > effective_prob).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x: torch.Tensor) -> torch.Tensor:
        x       = x.clone()
        C       = x.shape[0]
        # cutout_scale multiplies max_cutout_bands (clamped to valid range)
        max_cut = max(1, int(self.max_cutout_bands * self.cutout_scale))
        max_cut = min(max_cut, C)
        cut     = torch.randint(1, max_cut + 1, (1,)).item()
        start   = torch.randint(0, max(1, C - cut), (1,)).item()
        x[start:start + cut] = 0.0
        return x

    def _spectral_noise(self, x: torch.Tensor) -> torch.Tensor:
        # noise_scale is an absolute std value (already absorbed intensity_scale)
        sigma = self.noise_scale * self.intensity_scale
        mask  = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        return x + torch.randn_like(x) * sigma * mask

    def _spectral_warp(self, x: torch.Tensor) -> torch.Tensor:
        if self.warp_range <= 0:
            return x
        C, H, W = x.shape
        scale   = 1.0 + random.uniform(-self.warp_range, self.warp_range)
        new_C   = max(1, int(C * scale))
        if new_C == C:
            return x
        xp     = x.permute(1, 2, 0).reshape(-1, 1, C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s      = (new_C - C) // 2
            warped = warped[:, :, s:s + C]
        else:
            pad_l  = (C - new_C) // 2
            pad_r  = C - new_C - pad_l
            warped = F.pad(warped, (pad_l, pad_r))
        return warped.reshape(H, W, C).permute(2, 0, 1)

    def _multiplicative_noise(self, x: torch.Tensor) -> torch.Tensor:
        scale_std = 0.05 * self.intensity_scale
        mask      = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        factor    = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device) * scale_std
        return x * factor * mask

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [1])
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k, [1, 2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ri       = self.indices[idx]
        patch_np = np.array(self.patches[ri])
        patch    = torch.from_numpy(patch_np).to(CONFIG["device"], non_blocking=True)
        label    = torch.tensor(int(self.labels[ri]), dtype=torch.long, device=CONFIG["device"])
        if self.profile is not None:
            p = self.profile
            effective_cutout_prob = min(p["cutout"] * self.cutout_scale, 1.0)
            if torch.rand(1) < p["band_drop"] * self.band_drop_scale:
                patch = self._band_dropout(patch, p["band_drop"])
            if torch.rand(1) < effective_cutout_prob:
                patch = self._band_cutout(patch)
            if torch.rand(1) < p["noise"]:
                patch = self._spectral_noise(patch)
            if torch.rand(1) < p["warp"]:
                patch = self._spectral_warp(patch)
            if torch.rand(1) < p["mult"]:
                patch = self._multiplicative_noise(patch)
            patch = self._spatial(patch)
        return patch, label


# ══════════════════════════════════════════════════════════════════════
#  SAMPLERS
# ══════════════════════════════════════════════════════════════════════

class ClassBalancedBatchSampler(Sampler):
    """Draws n_cls classes per batch, n_spc samples per class, with optional CDWS weighting."""

    def __init__(
        self,
        train_labels:  np.ndarray,
        n_cls:         int                          = 16,
        n_spc:         int                          = 8,
        class_weights: Optional[Dict[int, float]]  = None,
    ) -> None:
        self.n_cls   = n_cls
        self.n_spc   = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)
        if class_weights is not None:
            raw        = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
            self.probs = raw / raw.sum()
        else:
            self.probs = None

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False, p=self.probs)
            batch  = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(
                    rng.choice(pool, self.n_spc, replace=len(pool) < self.n_spc).tolist()
                )
            yield batch

    def __len__(self) -> int:
        return self._n


class HardClassOversampledSampler(Sampler):
    """Stage 1 · Phase 3 — Class-Specific Oversampling Sampler."""

    def __init__(
        self,
        labels:           np.ndarray,
        class_f1:         Dict[int, float],
        num_samples:      int,
        oversample_power: float = 0.75,
        max_weight:       float = 5.0,
        hard_f1_thresh:   float = 0.50,
        eps:              float = 0.05,
    ) -> None:
        self.num_samples = num_samples
        num_classes      = int(np.max(labels)) + 1
        raw_weights: Dict[int, float] = {}
        for c in range(num_classes):
            f1             = float(class_f1.get(c, 0.0))
            w              = (1.0 / (f1 + eps)) ** oversample_power
            raw_weights[c] = min(w, max_weight)
        mean_w         = float(np.mean(list(raw_weights.values())))
        norm_weights   = {c: w / mean_w for c, w in raw_weights.items()}
        sample_weights = np.array(
            [norm_weights.get(int(lbl), 1.0) for lbl in labels], dtype=np.float32
        )
        self._weights = torch.from_numpy(sample_weights)
        n_hard        = sum(1 for f in class_f1.values() if f < hard_f1_thresh)
        hard_classes  = sorted(
            [c for c, f in class_f1.items() if f < hard_f1_thresh],
            key=lambda c: class_f1[c]
        )
        print(
            f"[INFO] Phase-3 oversampling: {n_hard}/{num_classes} hard classes "
            f"(F1 < {hard_f1_thresh})  |  power={oversample_power:.2f}  "
            f"max_w={max_weight:.1f}  n_samples={num_samples:,}"
        )
        if hard_classes:
            worst5 = [(c, class_f1[c]) for c in hard_classes[:5]]
            print(f"[INFO] Hardest classes (class_id, F1): {worst5}")

    def __iter__(self) -> Iterator[int]:
        return iter(
            torch.multinomial(self._weights, self.num_samples, replacement=True).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples


# ══════════════════════════════════════════════════════════════════════
#  CLASS DIFFICULTY WEIGHTS (CDWS)
# ══════════════════════════════════════════════════════════════════════

def build_cdws_weights(
    class_f1:    Dict[int, float],
    num_classes: int,
    max_w:       float = 3.0,
    eps:         float = 0.05,
) -> Dict[int, float]:
    raw  = {c: min(1.0 / (class_f1.get(c, 0.0) + eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w / mean for c, w in raw.items()}


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixed_aug(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    return _mixup(x, y, alpha)


def mixed_loss(
    crit:   nn.Module,
    logits: torch.Tensor,
    ya:     torch.Tensor,
    yb:     torch.Tensor,
    lam:    float,
) -> torch.Tensor:
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)


# ══════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C    = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        if self.ls > 0.0:
            with torch.no_grad():
                soft = torch.full_like(logits, self.ls / (C - 1))
                soft.scatter_(1, targets.view(-1, 1), 1.0 - self.ls)
            ce = -(soft * logp).sum(1)
        else:
            ce = F.nll_loss(logp, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss. Expects L2-normalised features."""

    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B         = features.shape[0]
        sim       = torch.mm(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        n_pos     = pos_mask.float().sum(1)
        if not (n_pos > 0).any():
            return torch.zeros((), device=features.device, requires_grad=True)
        sim_m    = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim_m - torch.logsumexp(sim_m, dim=1, keepdim=True)
        loss     = -(pos_mask.float() * log_prob.masked_fill(self_mask, 0.0)).sum(1)
        valid    = n_pos > 0
        return (loss[valid] / n_pos[valid]).mean()


class ProtoNCELoss(nn.Module):
    """Class-mean prototype contrastive CE."""

    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if len(classes) < 2:
            return (features * 0).sum()
        protos = F.normalize(
            torch.stack([features[labels == c].mean(0) for c in classes]), dim=1
        )
        sim   = torch.mm(features, protos.T) / self.temperature
        c2l   = {c.item(): i for i, c in enumerate(classes)}
        local = torch.tensor(
            [c2l[y.item()] for y in labels], dtype=torch.long, device=features.device
        )
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  SAM — Sharpness-Aware Minimisation
# ══════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs) -> None:
        defaults            = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step / second_step.")

    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        ns  = [p.grad.norm(p=2).to(dev)
               for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(ns), p=2).clamp(min=1e-6) if ns else torch.tensor(0.0)

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups


# ══════════════════════════════════════════════════════════════════════
#  ADAPTIVE SUB-CENTER ARCFACE
# ══════════════════════════════════════════════════════════════════════

class AdaptiveSubcenterArcFaceHead(nn.Module):
    def __init__(
        self,
        in_dim:      int,
        num_classes: int,
        K:           int   = 2,
        s:           float = 32.0,
        m_base:      float = 0.35,
        m_delta:     float = 0.10,
    ) -> None:
        super().__init__()
        self.K = K; self.C = num_classes
        self.s = s; self.m_base = m_base; self.m_delta = m_delta
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))

    def update_margins_from_f1(self, class_f1: Dict[int, float]) -> None:
        for c, f1 in class_f1.items():
            self.margins[c] = self.m_base + self.m_delta * (1.0 - min(float(f1), 1.0))

    def forward(
        self,
        x:        torch.Tensor,
        labels:   Optional[torch.Tensor] = None,
        global_m: Optional[float]        = None,
    ) -> torch.Tensor:
        x_n    = F.normalize(x, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        cosine = (
            F.linear(x_n, w_n)
            .clamp(-1 + 1e-6, 1 - 1e-6)
            .view(-1, self.C, self.K)
            .max(dim=2).values
        )
        if labels is None or not self.training:
            return cosine * self.s
        m_per  = (
            torch.full((x.shape[0],), global_m, device=x.device)
            if global_m is not None else self.margins[labels]
        )
        cosm  = torch.cos(m_per); sinm = torch.sin(m_per)
        th    = torch.cos(math.pi - m_per); mm = torch.sin(math.pi - m_per) * m_per
        sine  = torch.sqrt(torch.clamp(1 - cosine ** 2, min=1e-6))
        tgt_c = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        tgt_s = sine.gather(1, labels.view(-1, 1)).squeeze(1)
        phi   = tgt_c * cosm - tgt_s * sinm
        phi   = torch.where(tgt_c > th, phi, tgt_c - mm)
        oh    = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi.unsqueeze(1)) + ((1 - oh) * cosine)) * self.s

    def init_from_linear(self, linear_w: torch.Tensor) -> None:
        with torch.no_grad():
            wn = F.normalize(linear_w, dim=1)
            for k in range(self.K):
                noise = torch.randn_like(wn) * 0.01 * k
                self.weight[k::self.K].copy_(wn + noise)
        print(f"[INFO] ArcFace (K={self.K}) bootstrapped from linear head.")


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid    = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, mid, 1, bias=False), nn.GELU(),
            nn.Conv1d(mid, channels, 1, bias=False), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dilation: int = 1) -> None:
        super().__init__()
        pad        = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.se    = SEBlock1D(out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out      = F.gelu(self.norm1(self.conv1(x)))
        out      = self.norm2(self.conv2(out))
        out      = self.se(out)
        return F.gelu(out + identity)


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid     = max(c // r, 8)
        self.ch = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.GELU(),
                                 nn.Conv2d(mid, c, 1, bias=False))
        self.sp = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(
            self.ch(x.mean([2, 3], keepdim=True)) +
            self.ch(x.amax([2, 3], keepdim=True))
        )
        return x * self.sp(
            torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        )


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid     = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch)
            )
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(
            self.n3(self.c3(F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) +
            self.skip(x)
        )


class PhysicalWavelengthPE(nn.Module):
    def __init__(self, physical_wl: torch.Tensor, d_model: int) -> None:
        super().__init__()
        dev    = physical_wl.device
        half   = d_model // 2
        freq   = torch.exp(
            torch.arange(half, device=dev).float() *
            -(math.log(10000.0) / max(half - 1, 1))
        )
        pe           = torch.zeros(physical_wl.size(0), d_model, device=dev)
        pe[:, :half] = torch.sin(physical_wl.unsqueeze(1) * freq.unsqueeze(0))
        pe[:, half:] = torch.cos(physical_wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.transpose(0, 1).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A — SPECTRAL PROFILE
# ══════════════════════════════════════════════════════════════════════

class LargeKernelBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.norm   = nn.GroupNorm(1, dim)
        self.pw1    = nn.Conv1d(dim, dim * 4, 1, bias=False)
        self.act    = nn.GELU()
        self.pw2    = nn.Conv1d(dim * 4, dim, 1, bias=False)
        self.se     = SEBlock1D(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x   = self.dwconv(x)
        x   = self.norm(x)
        x   = self.pw1(x)
        x   = self.act(x)
        x   = self.pw2(x)
        x   = self.se(x)
        return x + res


class SpectralProfileBranch(nn.Module):
    def __init__(
        self,
        out_dim:      int                  = 256,
        tower_ch:     int                  = 96,
        wl_pe_module: Optional[nn.Module]  = None,
    ) -> None:
        super().__init__()
        self.wl_pe_module = wl_pe_module
        self.d1_conv      = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.d2_conv      = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        with torch.no_grad():
            self.d1_conv.weight[0, 0] = torch.tensor([-3, -2, -1, 0, 1, 2, 3]).float() / 28.0
            self.d2_conv.weight[0, 0] = torch.tensor([5,  0, -3, -4, -3, 0, 5]).float() / 42.0
        self.stem    = nn.Sequential(
            nn.Conv1d(3, tower_ch, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(1, tower_ch), nn.GELU()
        )
        self.tower_s = nn.Sequential(LargeKernelBlock1D(tower_ch, 7),  LargeKernelBlock1D(tower_ch, 7))
        self.tower_m = nn.Sequential(LargeKernelBlock1D(tower_ch, 15), LargeKernelBlock1D(tower_ch, 15))
        self.tower_l = nn.Sequential(LargeKernelBlock1D(tower_ch, 31), LargeKernelBlock1D(tower_ch, 31))
        self.fusion  = nn.Sequential(
            nn.Conv1d(tower_ch * 3, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch), nn.GELU(),
            LargeKernelBlock1D(tower_ch, 7)
        )
        self.attn_pool = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1), nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and m not in [self.d1_conv, self.d2_conv]:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ms: torch.Tensor) -> torch.Tensor:
        s        = ms.unsqueeze(1)
        s_smooth = F.avg_pool1d(s, kernel_size=5, stride=1, padding=2)
        d1       = self.d1_conv(s_smooth)
        d2       = self.d2_conv(s_smooth)
        x        = torch.cat([s, d1, d2], dim=1)
        x        = self.stem(x)
        if self.wl_pe_module is not None:
            x = self.wl_pe_module(x)
        x_fused  = self.fusion(
            torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1)
        )
        w = torch.softmax(self.attn_pool(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(
        self,
        num_bands:    int,
        out_dim:      int                 = 256,
        tower_ch:     int                 = 96,
        wl_pe_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_channels  = 9
        self.wl_pe_module = wl_pe_module
        self.stat_attn    = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.in_channels, 16, 1, bias=False), nn.GELU(),
            nn.Conv1d(16, self.in_channels, 1, bias=False), nn.Sigmoid()
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch), nn.GELU()
        )

        def _make_tower(kernel: int) -> nn.Sequential:
            return nn.Sequential(
                ResBlock1D(tower_ch, tower_ch, kernel),
                ResBlock1D(tower_ch, tower_ch, kernel)
            )

        self.tower_s   = _make_tower(3)
        self.tower_m   = _make_tower(7)
        self.tower_l   = _make_tower(15)
        self.fusion    = nn.Sequential(
            ResBlock1D(tower_ch * 3, tower_ch, 5),
            ResBlock1D(tower_ch, tower_ch, 5)
        )
        self.pool_attn = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1, bias=False), nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1, bias=False)
        )
        self.proj = nn.Sequential(
            nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ms, std, mx, skew, kurt, p10, p25, p75, p90):
        stats   = torch.stack([ms, std, mx, skew, kurt, p10, p25, p75, p90], dim=1)
        stats   = stats * self.stat_attn(stats)
        x       = self.input_proj(stats)
        if self.wl_pe_module is not None:
            x = self.wl_pe_module(x)
        x_fused = self.fusion(
            torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1)
        )
        w = torch.softmax(self.pool_attn(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, 1, groups=num_bands, bias=False),
            nn.Conv2d(num_bands, 64, 1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU()
        )
        self.stages = nn.Sequential(
            ResBlock2D(64,  128, 2), CBAM(128),
            ResBlock2D(128, 192, 2), CBAM(192),
            ResBlock2D(192, 256, 2), CBAM(256),
            ResBlock2D(256, out_dim, 2)
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    @staticmethod
    def _pn(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().clamp(1e-8).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.band_reduce(x))
        return self.proj(
            F.normalize(torch.cat([self._pn(h.mean([2, 3])), self._pn(h.amax([2, 3]))], 1),
                        dim=1, eps=1e-4)
        )


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER (spectral patch transformer)
# ══════════════════════════════════════════════════════════════════════

class MultiScaleSpectralTokenizer(nn.Module):
    def __init__(self, in_channels: int, d_model: int, stride: int = 8):
        super().__init__()
        out_c            = d_model // 3
        rem              = d_model - (out_c * 2)
        self.proj_small  = nn.Conv1d(in_channels, out_c, kernel_size=8,  stride=stride, padding=4)
        self.proj_medium = nn.Conv1d(in_channels, out_c, kernel_size=16, stride=stride, padding=8)
        self.proj_large  = nn.Conv1d(in_channels, rem,   kernel_size=32, stride=stride, padding=16)
        self.norm        = nn.GroupNorm(1, d_model)
        self.act         = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_s = self.proj_small(x)
        t_m = self.proj_medium(x)
        t_l = self.proj_large(x)
        min_len       = min(t_s.size(2), t_m.size(2), t_l.size(2))
        t_s, t_m, t_l = t_s[..., :min_len], t_m[..., :min_len], t_l[..., :min_len]
        tokens        = torch.cat([t_s, t_m, t_l], dim=1)
        return self.act(self.norm(tokens))


class _PreLNBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, drop: float) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d_ff, d), nn.Dropout(drop)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lx      = self.ln1(x)
        h, _    = self.attn(lx, lx, lx, need_weights=False)
        x       = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))


class SpecFormerBranch(nn.Module):
    def __init__(
        self,
        physical_wl: torch.Tensor,
        num_bands:   int   = 256,
        patch_size:  int   = 16,
        stride:      int   = 8,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        out_dim:     int   = 256,
        dropout:     float = 0.15,
    ) -> None:
        super().__init__()
        self.tokenizer     = MultiScaleSpectralTokenizer(in_channels=2, d_model=d_model, stride=stride)
        self.spec_cls      = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spec_cls, std=0.02)
        n_tokens             = (num_bands // stride) + 2
        self.spec_pos_embed  = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.spectral_blocks = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers // 2)
        ])
        self.spatial_cls     = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spatial_cls, std=0.02)
        self.spatial_blocks  = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers // 2)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    def forward(self, grid_ms: torch.Tensor) -> torch.Tensor:
        B, N, C    = grid_ms.shape
        deriv      = torch.diff(grid_ms, dim=2)
        deriv      = F.pad(deriv, (0, 1), mode='replicate')
        x_combo    = torch.stack([grid_ms, deriv], dim=2).view(B * N, 2, C)
        tokens     = self.tokenizer(x_combo).transpose(1, 2)
        cls_tokens = self.spec_cls.expand(B * N, -1, -1)
        tokens     = torch.cat([cls_tokens, tokens], dim=1)
        seq_len    = tokens.size(1)
        if seq_len <= self.spec_pos_embed.size(1):
            tokens = tokens + self.spec_pos_embed[:, :seq_len, :]
        for blk in self.spectral_blocks:
            tokens = blk(tokens)
        grid_features  = tokens[:, 0, :]
        spatial_tokens = grid_features.view(B, N, -1)
        spatial_cls    = self.spatial_cls.expand(B, -1, -1)
        spatial_tokens = torch.cat([spatial_cls, spatial_tokens], dim=1)
        for blk in self.spatial_blocks:
            spatial_tokens = blk(spatial_tokens)
        global_feature = self.norm(spatial_tokens[:, 0, :])
        return self.proj(global_feature)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH FUSION — Residual Cross-Modal Interaction
# ══════════════════════════════════════════════════════════════════════

class CrossModalInteraction(nn.Module):
    def __init__(
        self,
        num_modalities: int   = 4,
        d:              int   = 256,
        latent_tokens:  int   = 4,
        heads:          int   = 8,
        depth:          int   = 2,
        drop:           float = 0.1,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.d              = d
        self.branch_norms   = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_modalities)])
        self.latents        = nn.Parameter(torch.randn(latent_tokens, d) * 0.02)
        self.blocks         = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True),
                "self_attn":  nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True),
                "ff": nn.Sequential(
                    nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(),
                    nn.Dropout(drop), nn.Linear(d * 4, d),
                ),
            }))
        self.modality_gate = nn.Sequential(
            nn.Linear(d, d // 4), nn.GELU(),
            nn.Linear(d // 4, num_modalities), nn.Softmax(dim=-1),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop),
        )

    def forward(self, branches: List[torch.Tensor]):
        B       = branches[0].shape[0]
        tokens  = torch.stack([norm(b) for norm, b in zip(self.branch_norms, branches)], dim=1)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            attn_out, _ = blk["cross_attn"](latents, tokens, tokens)
            latents     = latents + attn_out
            sa_out, _   = blk["self_attn"](latents, latents, latents)
            latents     = latents + sa_out
            latents     = latents + blk["ff"](latents)
        fused          = latents.mean(dim=1)
        gate           = self.modality_gate(fused)
        weighted_modal = (tokens * gate.unsqueeze(-1)).sum(dim=1)
        fused          = fused + weighted_modal
        return self.output_proj(fused)


class EmbedNet(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 512, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(self.mlp(self.norm1(x)))
        return self.norm2(x)


# ══════════════════════════════════════════════════════════════════════
#  AUXILIARY CLASSIFICATION HEAD  (per branch, deep supervision)
# ══════════════════════════════════════════════════════════════════════

class AuxiliaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.trunc_normal_(self.net[2].weight, std=0.02)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH INFLUENCE & SPECTRAL STATS HELPERS
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_branch_influence(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    max_batches: int = 5,
) -> Dict[str, float]:
    model.eval()
    influences = torch.zeros(4, device=device)
    total      = 0
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x            = x.to(device, non_blocking=True)
        logits_full  = model(x)
        p_full       = torch.softmax(logits_full, dim=1)
        for b in range(4):
            mask      = torch.ones(4, device=device); mask[b] = 0.0
            logits_ab = model(x, branch_mask=mask)
            p_ab      = torch.softmax(logits_ab, dim=1).clamp(min=1e-10)
            influences[b] += F.kl_div(p_ab.log(), p_full, reduction="batchmean")
        total += 1
    if total == 0:
        return {"A": 0, "B": 0, "C": 0, "D": 0}
    influences /= total
    total_inf   = influences.sum().clamp(min=1e-8)
    influences  = influences / total_inf * 100.0
    return {k: float(influences[i]) for i, k in enumerate("ABCD")}


def extract_grid_spectra(x: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    B, C, H, W    = x.shape
    mask          = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()
    grid_sum      = F.adaptive_avg_pool2d(x * mask, (grid_size, grid_size))
    grid_mask_sum = F.adaptive_avg_pool2d(mask, (grid_size, grid_size))
    grid_mean     = grid_sum / grid_mask_sum.clamp(min=1e-5)
    return grid_mean.view(B, C, -1).transpose(1, 2)


def masked_spectral_stats(x: torch.Tensor):
    x32  = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)
    mask = (flat.abs().sum(1, keepdim=True) > 1e-5).float()
    cnt  = mask.sum(2).clamp(min=1.0)
    mean = (flat * mask).sum(2) / cnt
    centered = (flat - mean.unsqueeze(2)) * mask
    var  = (centered ** 2).sum(2) / cnt
    std  = torch.sqrt(var + 1e-5)
    mx   = flat.masked_fill(mask.expand_as(flat) == 0, -1e4).max(2).values
    mx   = mx.masked_fill(mx < -9999.0, 0.0)
    skew = torch.clamp(((centered ** 3).sum(2) / cnt) / (std ** 3 + 1e-4), -10.0, 10.0)
    kurt = torch.clamp(((centered ** 4).sum(2) / cnt) / (std ** 4 + 1e-4),   0.0, 20.0)
    flat_masked    = flat.masked_fill(mask.expand_as(flat) == 0, float("inf"))
    sorted_vals, _ = torch.sort(flat_masked, dim=2)

    def gather_percentile(vals, p_frac):
        idx          = (cnt * p_frac).long().clamp(max=H * W - 1)
        expanded_idx = idx.unsqueeze(2).expand(-1, C, -1)
        return torch.gather(vals, 2, expanded_idx).squeeze(2)

    p10, p25 = gather_percentile(sorted_vals, 0.10), gather_percentile(sorted_vals, 0.25)
    p75, p90 = gather_percentile(sorted_vals, 0.75), gather_percentile(sorted_vals, 0.90)
    return (
        torch.nan_to_num(mean, 0), torch.nan_to_num(std, 0),
        torch.nan_to_num(mx, 0),   torch.nan_to_num(skew, 0),
        torch.nan_to_num(kurt, 0), torch.nan_to_num(p10, 0),
        torch.nan_to_num(p25, 0),  torch.nan_to_num(p75, 0),
        torch.nan_to_num(p90, 0),
    )


# ══════════════════════════════════════════════════════════════════════
#  SPECTRALQUADNET
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    def __init__(
        self,
        num_classes:  int   = 90,
        num_bands:    int   = 256,
        dropout:      float = 0.30,
        wl_embed_dim: int   = 16,
        cfg:          Optional[dict] = None,
    ) -> None:
        super().__init__()
        global _PHYSICAL_WL
        cfg      = cfg or CONFIG
        tower_ch = 96
        self.branch_drop_prob = cfg.get("branch_drop_prob", 0.0)
        self.wl_pe_cnn        = PhysicalWavelengthPE(_PHYSICAL_WL, tower_ch)
        self.branch_a = SpectralProfileBranch(out_dim=256, tower_ch=tower_ch, wl_pe_module=self.wl_pe_cnn)
        self.branch_b = SpectralStatsBranch(num_bands=num_bands, out_dim=256, tower_ch=96, wl_pe_module=self.wl_pe_cnn)
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(
            physical_wl=_PHYSICAL_WL,
            num_bands=num_bands,
            patch_size=cfg["specf_patch"],
            stride=cfg["specf_patch"] // 2,
            d_model=cfg["specf_dim"],
            n_heads=cfg["specf_heads"],
            n_layers=cfg["specf_layers"],
            out_dim=256,
            dropout=cfg.get("specf_drop", 0.15),
        )
        self.cross_interaction = CrossModalInteraction(
            num_modalities=4, d=256, drop=cfg["fusion_drop"]
        )
        aux_hidden      = cfg.get("aux_head_hidden", 128)
        self.aux_head_a = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_b = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_c = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_d = AuxiliaryHead(256, aux_hidden, num_classes)
        self.embed_net  = EmbedNet(256, 512, dropout)
        self.linear_head  = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout * 0.4), nn.Linear(256, num_classes)
        )
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            256, num_classes,
            K=cfg.get("subcenter_K", 2),
            s=cfg["s2_arcface_s"],
            m_base=cfg["s2_arcface_m"],
            m_delta=cfg.get("s2_arcface_m_delta", 0.10),
        )
        self._use_arcface = False
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def use_arcface(self, flag: bool) -> None:
        self._use_arcface = flag

    def freeze_head(self, which: str) -> None:
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which: str) -> None:
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(True)

    def forward(
        self,
        x:            torch.Tensor,
        labels:       Optional[torch.Tensor]      = None,
        return_embed: bool                         = False,
        arc_m:        Optional[float]             = None,
        branch_mask:  Optional[torch.Tensor]      = None,
    ):
        ms, std, mx, skew, kurt, p10, p25, p75, p90 = masked_spectral_stats(x)
        grid_ms      = extract_grid_spectra(x, grid_size=4)
        B, N, C      = grid_ms.shape
        flat_grid_ms = grid_ms.reshape(B * N, C)
        ba_grid      = self.branch_a(flat_grid_ms)
        ba_raw       = ba_grid.view(B, N, -1).mean(dim=1)
        bb_raw       = self.branch_b(ms, std, mx, skew, kurt, p10, p25, p75, p90)
        bc_raw       = self.branch_c(x)
        bd_raw       = self.branch_d(grid_ms)
        if branch_mask is not None:
            ba = ba_raw * branch_mask[0]; bb = bb_raw * branch_mask[1]
            bc = bc_raw * branch_mask[2]; bd = bd_raw * branch_mask[3]
        elif self.training:
            drop_probs = torch.tensor([0.10, 0.10, 0.10, 0.10], device=ba_raw.device)
            keeps      = (torch.rand(4, device=ba_raw.device) > drop_probs).float()
            safe_idx   = torch.randint(0, 4, (), device=ba_raw.device)
            safe_mask  = F.one_hot(safe_idx, num_classes=4).float()
            keeps      = torch.maximum(keeps, safe_mask)
            ba = ba_raw * keeps[0]; bb = bb_raw * keeps[1]
            bc = bc_raw * keeps[2]; bd = bd_raw * keeps[3]
        else:
            ba, bb, bc, bd = ba_raw, bb_raw, bc_raw, bd_raw
        joint_token = self.cross_interaction([ba, bb, bc, bd])
        emb         = self.embed_net(joint_token)
        if self._use_arcface:
            logits = self.arcface_head(F.normalize(emb, dim=1), labels, global_m=arc_m)
        else:
            logits = self.linear_head(emb)
        if self.training:
            out = {
                "main":  logits,
                "aux_a": self.aux_head_a(ba_raw),
                "aux_b": self.aux_head_b(bb_raw),
                "aux_c": self.aux_head_c(bc_raw),
                "aux_d": self.aux_head_d(bd_raw),
            }
            if return_embed:
                out["emb"] = F.normalize(emb, dim=1)
            return out
        if return_embed:
            return logits, F.normalize(emb, dim=1)
        return logits


# ══════════════════════════════════════════════════════════════════════
#  TTA
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(
    model:      nn.Module,
    x:          torch.Tensor,
    n_spatial:  int = 8,
    n_spectral: int = 4,
) -> torch.Tensor:
    device        = x.device
    logits        = []
    spatial_views = [(k, f) for k in range(4) for f in (False, True)][:n_spatial]
    for k, flip in spatial_views:
        aug = torch.rot90(x, k, [2, 3])
        if flip:
            aug = torch.flip(aug, [3])
        with autocast(device_type=device.type):
            out = model(aug)
            logits.append(out["main"] if isinstance(out, dict) else out)
    scales = torch.linspace(0.95, 1.05, n_spectral, device=device)
    for s in scales:
        if s == 1.0:
            continue
        mean   = x.mean(dim=[2, 3], keepdim=True)
        aug_sp = mean + (x - mean) * s
        with autocast(device_type=device.type):
            out = model(aug_sp)
            logits.append(out["main"] if isinstance(out, dict) else out)
    return torch.stack(logits).mean(0)


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels,      random_state=42)
    va, te  = train_test_split(tmp,     test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(
    train_idx:     np.ndarray,
    val_idx:       np.ndarray,
    test_idx:      np.ndarray,
    batch_train:   int,
    balanced:      bool                         = False,
    all_labels:    Optional[np.ndarray]         = None,
    train_aug:     str                          = "none",
    class_weights: Optional[Dict[int, float]]  = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = RiceSeedDataset(train_idx, aug_strength=train_aug)
    if balanced and all_labels is not None:
        samp   = ClassBalancedBatchSampler(
            all_labels[train_idx],
            CONFIG["bal_n_cls"],
            CONFIG["bal_n_spc"],
            class_weights=class_weights,
        )
        tr_ldr = DataLoader(ds, batch_sampler=samp, num_workers=0)
    else:
        tr_ldr = DataLoader(ds, batch_size=batch_train, shuffle=True, drop_last=True, num_workers=0)
    va_ldr = DataLoader(RiceSeedDataset(val_idx),  batch_size=256, shuffle=False, num_workers=0)
    te_ldr = DataLoader(RiceSeedDataset(test_idx), batch_size=256, shuffle=False, num_workers=0)
    return tr_ldr, va_ldr, te_ldr


def build_phase3_loader(train_ds: Dataset, class_f1: Dict[int, float]) -> DataLoader:
    if not CONFIG["s1_p3_oversample"] or not class_f1:
        return DataLoader(
            train_ds, batch_size=CONFIG["s1_batch"],
            shuffle=True, drop_last=True, num_workers=0
        )
    train_labels = np.array(
        [int(_GLOBAL_LABELS[train_ds.indices[i]]) for i in range(len(train_ds.indices))]
    )
    sampler = HardClassOversampledSampler(
        labels           = train_labels,
        class_f1         = class_f1,
        num_samples      = len(train_labels),
        oversample_power = CONFIG["s1_p3_oversample_power"],
        max_weight       = CONFIG["s1_p3_oversample_max_w"],
        hard_f1_thresh   = CONFIG["s1_p3_hard_f1_thresh"],
        eps              = CONFIG["s1_p3_oversample_eps"],
    )
    return DataLoader(
        train_ds, batch_size=CONFIG["s1_batch"],
        sampler=sampler, drop_last=True, num_workers=0
    )


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISERS & SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def _wd_groups(named_params, lr: float) -> List[dict]:
    wd, no_wd = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_wd if (p.ndim == 1 or n.endswith(".bias")) else wd).append(p)
    return [
        {"params": wd,    "lr": lr, "weight_decay": CONFIG["weight_decay"]},
        {"params": no_wd, "lr": lr, "weight_decay": 0.0},
    ]


def build_optimizer_s1(model: nn.Module, lr: float) -> optim.AdamW:
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))


def build_optimizer_s2(model: nn.Module, head_lr: float, back_lr: float) -> optim.AdamW:
    hp, bp = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hp if n.startswith("arcface_head") else bp).append((n, p))
    return optim.AdamW(_wd_groups(hp, head_lr) + _wd_groups(bp, back_lr))


def build_optimizer_s3(model: nn.Module, lr: float) -> optim.AdamW:
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))


def sgdr_scheduler(
    optimizer:    optim.Optimizer,
    warmup_ep:    int   = 5,
    T_0:          int   = 10,
    T_mult:       int   = 2,
    eta_min_frac: float = 1e-3,
) -> optim.lr_scheduler.LambdaLR:
    def _l(ep: int) -> float:
        if ep < warmup_ep:
            return max(ep / max(warmup_ep, 1), 1e-6)
        t = ep - warmup_ep; clen = T_0; elapsed = 0
        while t >= elapsed + clen:
            elapsed += clen; clen = max(int(clen * T_mult), 1)
        ratio = (t - elapsed) / max(clen, 1)
        return eta_min_frac + 0.5 * (1 - eta_min_frac) * (1 + math.cos(math.pi * ratio))
    return optim.lr_scheduler.LambdaLR(optimizer, _l)


def arcface_margin(ep: int, m0: float, m_target: float, warmup_ep: int) -> float:
    if ep >= warmup_ep:
        return m_target
    return m0 + (m_target - m0) * 0.5 * (1 - math.cos(math.pi * ep / max(warmup_ep, 1)))


# ══════════════════════════════════════════════════════════════════════
#  AUXILIARY LOSS HELPERS
# ══════════════════════════════════════════════════════════════════════

def _aux_loss_weight(current_ep: int, total_ep: int) -> float:
    progress = current_ep / max(total_ep, 1)
    return max(
        CONFIG["aux_loss_weight_final"],
        CONFIG["aux_loss_weight_init"] * (1.0 - progress),
    )


def _compute_aux_loss(
    criterion: nn.Module,
    out:       dict,
    ya:        torch.Tensor,
    yb:        torch.Tensor,
    lam:       float,
    use_mixup: bool,
) -> torch.Tensor:
    aux_keys = ["aux_a", "aux_b", "aux_c", "aux_d"]
    total    = torch.zeros((), device=ya.device)
    for k in aux_keys:
        if k not in out:
            continue
        if use_mixup:
            total = total + mixed_loss(criterion, out[k], ya, yb, lam)
        else:
            total = total + criterion(out[k], ya)
    return total


# ══════════════════════════════════════════════════════════════════════
#  TRAIN ONE EPOCH  (AdamW — Stage 1 and Stage 2)
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    optimizer:     optim.Optimizer,
    criterion:     nn.Module,
    scaler:        Optional[GradScaler],
    ema:           ModelEMA,
    device:        torch.device,
    scheduler:     Optional[optim.lr_scheduler._LRScheduler] = None,
    use_mixup:     bool  = True,
    mixup_alpha:   float = 0.4,
    supcon:        Optional[nn.Module] = None,
    supcon_weight: float = 0.0,
    proto:         Optional[nn.Module] = None,
    proto_weight:  float = 0.0,
    accum_steps:   int   = 1,
    arc_m:         Optional[float] = None,
    current_ep:    int   = 0,
    total_ep:      int   = 100,
) -> Tuple[float, float]:
    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_amp = (supcon is None) and (scaler is not None)
    aux_w   = _aux_loss_weight(current_ep, total_ep)
    if model._use_arcface and use_mixup:
        raise ValueError("Mixup cannot be used with ArcFace.")
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_in, ya, yb, lam = mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)
        with autocast(device_type=device.type, enabled=use_amp):
            if supcon is not None:
                out    = model(x_in, ya, return_embed=True, arc_m=arc_m)
                logits = out["main"] if isinstance(out, dict) else out[0]
                emb    = out.get("emb") if isinstance(out, dict) else out[1]
                cls_l  = criterion(logits, ya)
                sc_l   = supcon(emb, ya)
                pt_l   = proto(emb, ya) if proto is not None else 0.0
                aux_l  = (
                    _compute_aux_loss(criterion, out, ya, yb, lam, use_mixup=False)
                    if isinstance(out, dict) else torch.zeros((), device=device)
                )
                loss = (
                    (1 - supcon_weight - proto_weight) * cls_l
                    + supcon_weight * sc_l
                    + proto_weight * pt_l
                    + aux_w * aux_l
                )
            else:
                arc_labels = ya if (model._use_arcface and not use_mixup) else None
                out        = model(x_in, labels=arc_labels, arc_m=arc_m)
                if isinstance(out, dict):
                    l_main = mixed_loss(criterion, out["main"], ya, yb, lam)
                    aux_l  = _compute_aux_loss(criterion, out, ya, yb, lam, use_mixup)
                    loss   = l_main + aux_w * aux_l
                    logits = out["main"]
                else:
                    logits = out
                    loss   = mixed_loss(criterion, logits, ya, yb, lam)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        if use_amp:
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            if use_amp:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)
        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == ya).float().mean().item()
    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════════════
#  TRAIN ONE EPOCH (SAM — Stage 3)
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch_sam(
    model:         nn.Module,
    loader:        DataLoader,
    sam_opt:       SAM,
    criterion:     nn.Module,
    device:        torch.device,
    supcon:        Optional[nn.Module] = None,
    supcon_weight: float = 0.0,
    proto:         Optional[nn.Module] = None,
    proto_weight:  float = 0.0,
    arc_m:         Optional[float] = None,
    aux_weight:    float = 0.0,
) -> Tuple[float, float]:
    torch.set_default_dtype(torch.float32)
    model.train()
    total_loss = total_acc = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        sam_opt.zero_grad()
        out    = model(x, labels=y, arc_m=arc_m, return_embed=(supcon is not None))
        logits = out["main"] if isinstance(out, dict) else out
        emb    = out.get("emb") if isinstance(out, dict) else None
        loss   = criterion(logits, y)
        if supcon is not None and emb is not None:
            loss = loss + supcon_weight * supcon(emb, y)
        if isinstance(out, dict) and aux_weight > 0.0:
            aux_l = _compute_aux_loss(criterion, out, y, y, 1.0, use_mixup=False)
            loss  = loss + aux_weight * aux_l
        if not torch.isfinite(loss):
            sam_opt.zero_grad()
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.first_step(zero_grad=True)
        out2    = model(x, labels=y, arc_m=arc_m, return_embed=(supcon is not None))
        logits2 = out2["main"] if isinstance(out2, dict) else out2
        loss2   = criterion(logits2, y)
        if not torch.isfinite(loss2):
            sam_opt.zero_grad()
            continue
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.second_step(zero_grad=True)
        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.detach().argmax(1) == y).float().mean().item()
    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_eval(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with autocast(device_type=device.type, enabled=False):
        for x, y in loader:
            x      = x.to(device, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            preds.append(logits.argmax(1).cpu()); targets.append(y.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    p, t = _run_eval(model, loader, device)
    return f1_score(t, p, average="macro", zero_division=0), accuracy_score(t, p)


def evaluate_per_class(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> Dict[int, float]:
    p, t   = _run_eval(model, loader, device)
    f1_arr = f1_score(t, p, average=None, zero_division=0, labels=list(range(num_classes)))
    return {i: float(v) for i, v in enumerate(f1_arr)}


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def stage_ckpt_path(s: int) -> str:
    return os.path.join(CONFIG["output_dir"], f"best_stage{s}.pth")


def stage_meta_path(s: int) -> str:
    return os.path.join(CONFIG["output_dir"], f"stage{s}_meta.json")


def stage_exists(s: int) -> bool:
    return os.path.isfile(stage_ckpt_path(s)) and os.path.isfile(stage_meta_path(s))


def latest_completed_stage() -> int:
    for s in (3, 2, 1):
        if stage_exists(s):
            return s
    return 0


def save_ckpt(
    path:    str,
    epoch:   int,
    stage:   str,
    model:   nn.Module,
    ema:     ModelEMA,
    val_f1:  float,
    val_acc: float,
    **metadata,
) -> None:
    bundle = {
        "epoch": epoch, "stage": stage,
        "model": model.state_dict(), "ema": ema.state_dict(),
        "val_f1": val_f1, "val_acc": val_acc,
        "use_arcface": model._use_arcface,
        **metadata,
    }
    torch.save(bundle, path)
    sidecar = {k: v for k, v in bundle.items()
               if k not in ("model", "ema") and _is_json_serialisable(v)}
    sn = int(stage.split()[-1]) if stage.split()[-1].isdigit() else 0
    with open(stage_meta_path(sn), "w") as f:
        _json.dump(sidecar, f, indent=2)


def _is_json_serialisable(v) -> bool:
    try:
        _json.dumps(v); return True
    except (TypeError, ValueError):
        return False


def load_stage_meta(s: int) -> dict:
    p = stage_meta_path(s)
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        raw = _json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            try:
                out[k] = {int(kk): vv for kk, vv in v.items()}; continue
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out


def load_ckpt(path: str, model: nn.Module, ema: ModelEMA, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    flag = ckpt.get("use_arcface", False)
    model.use_arcface(flag); ema.shadow.use_arcface(flag)
    return ckpt


def update_bn_stats(loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats(); m.momentum = None
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device, non_blocking=True))
    model.eval()


# ══════════════════════════════════════════════════════════════════════
#  CLASS DIFFICULTY
# ══════════════════════════════════════════════════════════════════════

def compute_class_difficulty(
    ema_shadow: nn.Module,
    val_ldr:    DataLoader,
    device:     torch.device,
    label:      str = "Stage",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    class_f1   = evaluate_per_class(ema_shadow, val_ldr, device, CONFIG["num_classes"])
    cdws_wts   = build_cdws_weights(
        class_f1, CONFIG["num_classes"], CONFIG["cdws_max_weight"], CONFIG["cdws_eps"]
    )
    macro      = float(np.mean(list(class_f1.values())))
    n_hard     = sum(1 for f in class_f1.values() if f < 0.50)
    branch_inf = compute_branch_influence(ema_shadow, val_ldr, device, max_batches=3)
    return class_f1, cdws_wts


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — 3-Phase Progressive Augmentation
#  (optional trial_callback for Optuna pruning)
# ══════════════════════════════════════════════════════════════════════

def run_stage1(
    model:            nn.Module,
    ema:              ModelEMA,
    loaders_by_phase: Dict[int, DataLoader],
    val_ldr:          DataLoader,
    device:           torch.device,
    best_ckpt:        str,
    trial_callback:   Optional[Callable] = None,
) -> float:
    model.use_arcface(False)
    model.unfreeze_head("linear")
    model.freeze_head("arcface")
    ep_total = CONFIG["s1_epochs"]
    p1_end   = int(ep_total * CONFIG["s1_phase1_frac"])
    p2_end   = int(ep_total * (CONFIG["s1_phase1_frac"] + CONFIG["s1_phase2_frac"]))
    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ep_total, eta_min=CONFIG["s1_min_lr"]
        )
    scaler           = GradScaler()
    ls_hi            = CONFIG["s1_label_smooth_hi"]
    ls_lo            = CONFIG["s1_label_smooth_lo"]
    best_f1          = 0.0
    no_improve       = 0
    ema_reinited     = [False, False]
    phase3_ldr:      Optional[DataLoader] = None
    class_f1_phase2: Dict[int, float]     = {}
    w = 66
    for ep in range(1, ep_total + 1):
        if   ep <= p1_end: phase = 1
        elif ep <= p2_end: phase = 2
        else:              phase = 3
        if phase == 2 and not ema_reinited[0] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            ema_reinited[0] = True
        if phase == 3 and not ema_reinited[1] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            ema_reinited[1] = True
        if phase == 3 and phase3_ldr is None:
            print(f"\n[INFO] Phase 2→3 boundary: measuring per-class F1 for oversampling ...")
            class_f1_phase2, _ = compute_class_difficulty(ema.shadow, val_ldr, device, "Phase2→3")
            phase3_ldr = build_phase3_loader(loaders_by_phase[3].dataset, class_f1_phase2)
        if   phase == 1: cur_ldr = loaders_by_phase[1]
        elif phase == 2: cur_ldr = loaders_by_phase[2]
        else:            cur_ldr = phase3_ldr
        t      = (ep - 1) / max(ep_total - 1, 1)
        ls_now = ls_hi * (1 - t) + ls_lo * t
        if phase == 3:
            crit = FocalLoss(gamma=CONFIG["s1_focal_gamma"], label_smoothing=ls_now)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=ls_now)
        use_mx = (phase != 3)
        tl, ta = train_one_epoch(
            model, cur_ldr, optimizer, crit, scaler, ema, device,
            scheduler=None, use_mixup=use_mx,
            mixup_alpha=CONFIG["s1_mixup"],
            accum_steps=CONFIG["s1_accum"],
            current_ep=ep, total_ep=ep_total,
        )
        scheduler.step()
        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1        = max(f1_live, f1_ema)
        best_ep_acc       = max(acc_live, acc_ema)
        lr_now            = optimizer.param_groups[0]["lr"]
        aux_w_now         = _aux_loss_weight(ep, ep_total)
        saved             = ""
        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1, _cdws = compute_class_difficulty(ema.shadow, val_ldr, device, "S1")
            save_ckpt(
                best_ckpt, ep, "Stage 1", model, ema,
                val_f1=best_ep_f1, val_acc=best_ep_acc,
                class_f1=_cf1, cdws_weights=_cdws,
                arcface_init_done=False,
                phase3_class_f1=class_f1_phase2,
            )
            saved = "  ✓"
        else:
            no_improve += 1
        # ── Optuna callback (pruning / reporting) ─────────────────────
        if trial_callback is not None:
            should_stop = trial_callback(ep, best_f1)
            if should_stop:
                print(f"[Optuna] Trial pruned at epoch {ep}.")
                break
        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break
    model.unfreeze_head("arcface")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR
# ══════════════════════════════════════════════════════════════════════

def run_stage2(
    model:          nn.Module,
    ema:            ModelEMA,
    train_ldr:      DataLoader,
    val_ldr:        DataLoader,
    device:         torch.device,
    best_ckpt:      str,
    class_f1:       Optional[Dict[int, float]] = None,
    trial_callback: Optional[Callable]         = None,
) -> float:
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)
    model.freeze_head("linear"); model.unfreeze_head("arcface")
    ema.reinit_from(model)
    ema.set_dropout(CONFIG["s2_dropout"]); ema.shadow.use_arcface(True)
    if class_f1 is not None:
        model.arcface_head.update_margins_from_f1(class_f1)
        ema.shadow.arcface_head.update_margins_from_f1(class_f1)
    focal  = FocalLoss(gamma=CONFIG["s2_focal_gamma"])
    supcon = SupConLoss(temperature=CONFIG["supcon_temp"])
    proto  = ProtoNCELoss(temperature=CONFIG["proto_temp"])
    optimizer = build_optimizer_s2(model, CONFIG["s2_head_lr"], CONFIG["s2_back_lr"])
    scheduler = sgdr_scheduler(
        optimizer,
        warmup_ep=CONFIG["s2_warmup_ep"],
        T_0=CONFIG["s2_sgdr_T0"],
        T_mult=CONFIG["s2_sgdr_Tmult"],
        eta_min_frac=CONFIG["s2_min_lr"] / CONFIG["s2_head_lr"],
    )
    sc_w     = CONFIG["supcon_weight"]; pt_w = CONFIG["proto_weight"]
    ep_total = CONFIG["s2_epochs"]
    best_f1  = 0.0; no_improve = 0
    r1 = CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"]
    r2 = r1 + CONFIG["s2_sgdr_T0"] * CONFIG["s2_sgdr_Tmult"]
    w  = 66
    for ep in range(1, ep_total + 1):
        warmup_done = (ep - 1) >= CONFIG["s2_margin_warmup_ep"]
        m_now       = (
            CONFIG["s2_arcface_m"] if warmup_done
            else arcface_margin(ep - 1, CONFIG["s2_arcface_m0"],
                                CONFIG["s2_arcface_m"], CONFIG["s2_margin_warmup_ep"])
        )
        arc_m  = None if warmup_done else m_now
        ramp   = min(1.0, ep / 10.0)
        sc_now = sc_w * ramp; pt_now = pt_w * ramp
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, focal, scaler=None, ema=ema,
            device=device, scheduler=None,
            use_mixup=False,
            supcon=supcon, supcon_weight=sc_now,
            proto=proto,   proto_weight=pt_now,
            arc_m=arc_m, current_ep=ep, total_ep=ep_total,
        )
        scheduler.step()
        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1        = max(f1_live, f1_ema)
        best_ep_acc       = max(acc_live, acc_ema)
        head_lr           = optimizer.param_groups[0]["lr"]
        back_lr           = optimizer.param_groups[2]["lr"]
        saved             = ""
        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1_s2, _cdws_s2   = compute_class_difficulty(ema.shadow, val_ldr, device, "S2")
            save_ckpt(
                best_ckpt, ep, "Stage 2", model, ema,
                val_f1=best_ep_f1, val_acc=best_ep_acc,
                class_f1=_cf1_s2, cdws_weights=_cdws_s2,
                s2_val_f1=best_ep_f1,
            )
            saved = "  ✓"
        else:
            no_improve += 1

        # ── Optuna callback ────────────────────────────────────────────
        if trial_callback is not None:
            should_stop = trial_callback(ep, best_f1)
            if should_stop:
                print(f"[Optuna] Trial pruned at epoch {ep}.")
                break
        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break
    model.unfreeze_head("linear")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 — SAM + Greedy SWA
# ══════════════════════════════════════════════════════════════════════

def run_stage3_swa(
    model:          nn.Module,
    ema:            ModelEMA,
    train_ldr:      DataLoader,
    val_ldr:        DataLoader,
    device:         torch.device,
    best_ckpt:      str,
    prev_best_f1:   float,
    trial_callback: Optional[Callable] = None,
) -> float:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.disable()
    model.set_dropout(CONFIG["s2_dropout"])
    model.branch_drop_prob     = 0.0
    ema.shadow.branch_drop_prob = 0.0
    model.use_arcface(True); ema.shadow.use_arcface(True)
    params    = list(_wd_groups(model.named_parameters(), CONFIG["s3_swa_lr"]))
    sam       = SAM(params, optim.AdamW, rho=CONFIG["s3_sam_rho"],
                    lr=CONFIG["s3_swa_lr"], weight_decay=CONFIG["weight_decay"])
    focal_s3  = FocalLoss(gamma=1.0)
    supcon_s3 = SupConLoss(temperature=0.10)
    proto_s3  = ProtoNCELoss(temperature=0.10)
    swa_state    = None
    n_snap       = 0; n_rejected = 0; best_live_f1 = 0.0
    aux_w_s3     = CONFIG["s3_aux_loss_weight"]
    ep_total     = CONFIG["s3_epochs"]
    w = 66

    def _s3_margin(ep: int) -> float:
        return 0.25 + 0.05 * math.cos(math.pi * ep / ep_total)

    for ep in range(1, ep_total + 1):
        cycle_ep = (ep - 1) % CONFIG["s3_cycle_len"]
        lr_now   = CONFIG["s3_swa_lr"] * (
            0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * cycle_ep / CONFIG["s3_cycle_len"]))
        )
        for pg in sam.param_groups:
            pg["lr"] = lr_now
        tl, ta = train_one_epoch_sam(
            model, train_ldr, sam, focal_s3, device,
            supcon=supcon_s3, supcon_weight=0.02,
            proto=proto_s3,   proto_weight=0.01,
            arc_m=_s3_margin(ep),
            aux_weight=aux_w_s3,
        )
        f1_live, acc_live = evaluate(model, val_ldr, device)
        best_live_f1      = max(best_live_f1, f1_live)
        snap_info         = ""
        if ep % CONFIG["s3_cycle_len"] == 0:
            if not CONFIG["s3_greedy"] or f1_live >= best_live_f1 * 0.98:
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
                snap_info   = f"  ✗ rejected (F1 {f1_live:.3f} < {best_live_f1*0.98:.3f})"
        print(
            f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
            f"F1 {f1_live:.3f}  Acc {acc_live:.1%} │ LR {lr_now:.2e}{snap_info}"
        )
        # ── Optuna callback ────────────────────────────────────────────
        if trial_callback is not None:
            should_stop = trial_callback(ep, best_live_f1)
            if should_stop:
                print(f"[Optuna] Trial pruned at epoch {ep}.")
                break
    print(f"\nUpdating BN stats ({n_snap} accepted, {n_rejected} rejected) ...")
    if swa_state is None:
        swa_state = copy.deepcopy(model.state_dict())
    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state); swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)
    f1_swa, acc_swa = evaluate(swa_model, val_ldr, device)
    print(f"SWA val: F1={f1_swa:.3f}  Acc={acc_swa:.1%}")
    ema.shadow.load_state_dict(swa_model.state_dict())
    ema.shadow.use_arcface(True)
    note = ""
    if f1_swa <= prev_best_f1:
        note = "val_f1 did not beat Stage 2; Stage 2 ckpt preferred for eval"
    save_ckpt(
        best_ckpt, ep_total, "Stage 3",
        swa_model, ema, val_f1=f1_swa, val_acc=acc_swa,
        swa_n_snapshots=n_snap, swa_n_rejected=n_rejected,
        **({"note": note} if note else {}),
    )
    return f1_swa


# ══════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(
    model:     nn.Module,
    ema:       ModelEMA,
    test_ldr:  DataLoader,
    device:    torch.device,
    best_ckpt: str,
) -> None:
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")
    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow; eval_model.eval()
    print(f"  ArcFace: {eval_model._use_arcface}  |  "
          f"Checkpoint: ep {ckpt['epoch']} | {ckpt['stage']} | "
          f"F1={ckpt.get('val_f1',0):.3f}  Acc={ckpt.get('val_acc',0):.1%}")
    print(f"  TTA: {CONFIG['tta_spatial']} spatial + {CONFIG['tta_spectral']} spectral "
          f"= {CONFIG['tta_spatial']+CONFIG['tta_spectral']} total views")
    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x = x.to(device, non_blocking=True)
            logits = (
                tta_predict(eval_model, x, CONFIG["tta_spatial"], CONFIG["tta_spectral"])
                if use_tta else eval_model(x)
            )
            preds.append(logits.argmax(1).cpu()); targets.append(y.cpu())
        p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p, t)
        print(
            f"\n  [{tag}]  F1(macro)={f1_score(t,p,average='macro',zero_division=0):.4f}  "
            f"F1(wt)={f1_score(t,p,average='weighted',zero_division=0):.4f}  "
            f"Acc={accuracy_score(t,p):.1%}"
        )
    p_tta, t_tta = results["TTA   "]
    print(f"\nClassification Report (TTA):\n")
    print(classification_report(t_tta, p_tta, zero_division=0))
    out = CONFIG["output_dir"]
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy",   p_tta)
    np.save(f"{out}/test_targets.npy",     t_tta)
    print(f"\nOutputs saved → {out}")


def _pick_best_checkpoint(*ckpt_paths: str) -> str:
    best_val, best_path = -1.0, ckpt_paths[-1]
    for p in ckpt_paths:
        if not os.path.isfile(p):
            continue
        try:
            sn   = int(os.path.basename(p).replace("best_stage", "").replace(".pth", ""))
            meta = load_stage_meta(sn)
            v    = meta.get("val_f1", meta.get("val_acc", None))
        except (ValueError, KeyError):
            v = None
        if v is None:
            try:
                v = torch.load(p, map_location="cpu", weights_only=False).get("val_f1", 0.0)
            except Exception:
                v = 0.0
        if v > best_val:
            best_val, best_path = v, p
    return best_path


# ══════════════════════════════════════════════════════════════════════
#  STANDARD TRAINING MAIN  (original pipeline, unchanged)
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    device     = CONFIG["device"]
    ckpt_s1    = stage_ckpt_path(1)
    ckpt_s2    = stage_ckpt_path(2)
    ckpt_s3    = stage_ckpt_path(3)
    done_stage = latest_completed_stage()

    labels_map = {0: "starting fresh", 1: "Stage 1 done", 2: "Stages 1–2 done", 3: "all done"}
    print(f"\n{'─'*66}")
    print(f"  Auto-resume: {labels_map.get(done_stage, f'stage {done_stage} done')}")
    print(f"  Output dir : {CONFIG['output_dir']}")
    print(f"{'─'*66}")
    print(f"[INFO] Latest completed stage: {done_stage}")

    _load_data_mmap(CONFIG["patches_data"], CONFIG["labels_path"])
    _load_wavelengths_to_gpu(CONFIG["wavelength_path"], device)

    all_labels, train_idx, val_idx, test_idx = build_splits()
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    model = SpectralQuadNet(
        num_classes=CONFIG["num_classes"],
        num_bands=CONFIG["num_bands"],
        dropout=CONFIG["s1_dropout"],
        wl_embed_dim=CONFIG["wl_embed_dim"],
        cfg=CONFIG,
    ).to(device)
    ema = ModelEMA(model, decay=CONFIG["ema_decay"])

    print(f"Model  : SpectralQuadNet (4× AuxHead deep supervision)")
    print(f"Params : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Device : {device}")

    def _s1_ldr(aug_str: str) -> DataLoader:
        ds = RiceSeedDataset(train_idx, aug_strength=aug_str)
        return DataLoader(ds, batch_size=CONFIG["s1_batch"],
                          shuffle=True, drop_last=True, num_workers=0)

    if done_stage < 1:
        print("\n[RUN] Stage 1")
        phase_loaders = {1: _s1_ldr("heavy"), 2: _s1_ldr("medium"), 3: _s1_ldr("light")}
        _, val_ldr1, _ = build_loaders(train_idx, val_idx, test_idx,
                                       CONFIG["s1_batch"], train_aug="none")
        run_stage1(model, ema, phase_loaders, val_ldr1, device, ckpt_s1)
        print("[INFO] Reloading best Stage 1 checkpoint ...")
        load_ckpt(ckpt_s1, model, ema, device)
    else:
        print("\n[SKIP] Stage 1 → loading checkpoint")
        load_ckpt(ckpt_s1, model, ema, device)

    meta_s1      = load_stage_meta(1)
    class_f1_s1  = meta_s1.get("class_f1",    {})
    cdws_wts_s1  = meta_s1.get("cdws_weights", {})
    arcface_done = meta_s1.get("arcface_init_done", False)
    s1_best_f1   = meta_s1.get("val_f1", meta_s1.get("val_acc", 0.0))
    print(f"[INFO] Stage 1 → F1={s1_best_f1:.3f}  "
          f"hard classes={sum(1 for f in class_f1_s1.values() if f<0.5)}")

    if done_stage < 2:
        if not arcface_done:
            print("\n[INFO] Bootstrapping ArcFace from linear head")
            lw = model.linear_head[-1].weight.data.clone()
            model.arcface_head.init_from_linear(lw)
            ema.shadow.arcface_head.init_from_linear(lw)
        if not class_f1_s1:
            print("[WARN] No class_f1 in Stage 1 meta — recomputing")
            _, val_cd, _ = build_loaders(train_idx, val_idx, test_idx, 128)
            class_f1_s1, cdws_wts_s1 = compute_class_difficulty(
                ema.shadow, val_cd, device, "Stage 1 (recomputed)"
            )
        print("\n[RUN] Stage 2")
        tr2, va2, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"],
            balanced=True, all_labels=all_labels,
            train_aug="light", class_weights=cdws_wts_s1,
        )
        run_stage2(model, ema, tr2, va2, device, ckpt_s2, class_f1_s1)
        print("[INFO] Reloading best Stage 2 checkpoint ...")
        load_ckpt(ckpt_s2, model, ema, device)
    else:
        print("\n[SKIP] Stage 2 → loading checkpoint")
        load_ckpt(ckpt_s2, model, ema, device)

    meta_s2     = load_stage_meta(2)
    class_f1_s2 = meta_s2.get("class_f1",    {})
    cdws_wts_s2 = meta_s2.get("cdws_weights", {})
    s2_best_f1  = meta_s2.get("val_f1", meta_s2.get("s2_val_f1", meta_s2.get("val_acc", 0.0)))
    print(f"[INFO] Stage 2 → F1={s2_best_f1:.3f}")

    if hasattr(torch, "_dynamo"):
        print("[INFO] Disabling torch.compile for Stage 3 stability")
        torch._dynamo.reset()

    if done_stage < 3:
        if not cdws_wts_s2:
            cdws_wts_s2 = cdws_wts_s1
        print("\n[RUN] Stage 3 (SAM + Greedy SWA)")
        tr3, va3, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"],
            balanced=True, all_labels=all_labels,
            train_aug="light", class_weights=cdws_wts_s2,
        )
        run_stage3_swa(model, ema, tr3, va3, device, ckpt_s3, prev_best_f1=s2_best_f1)
    else:
        print("\n[SKIP] Stage 3 → loading checkpoint")
        load_ckpt(ckpt_s3, model, ema, device)
        meta_s3 = load_stage_meta(3)
        print(f"[INFO] Stage 3 → snaps={meta_s3.get('swa_n_snapshots','?')}  "
              f"rejected={meta_s3.get('swa_n_rejected','?')}  "
              f"F1={meta_s3.get('val_f1', meta_s3.get('val_acc',0)):.3f}")

    best_final_ckpt = _pick_best_checkpoint(ckpt_s1, ckpt_s2, ckpt_s3)
    print(f"\n[INFO] Best checkpoint (by val_f1): {best_final_ckpt}")
    _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx, 256)
    final_evaluation(model, ema, test_ldr, device, best_final_ckpt)


# ══════════════════════════════════════════════════════════════════════════════
#  ██████████████████  OPTUNA TUNING PIPELINE  ██████████████████
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
#  TUNING UTILITIES
# ──────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def temp_config(overrides: dict):
    """
    Context manager: temporarily overlay CONFIG with ``overrides``.
    On exit, all keys are restored to their original values.
    CONFIG["device"] is never overridden (kept constant).
    """
    overrides.pop("device", None)
    old_vals  = {k: CONFIG[k] for k in overrides if k in CONFIG}
    new_keys  = [k for k in overrides if k not in CONFIG]
    CONFIG.update(overrides)
    try:
        yield CONFIG
    finally:
        CONFIG.update(old_vals)
        for k in new_keys:
            CONFIG.pop(k, None)


def _make_tuning_dirs() -> None:
    """Create all tuning output directories."""
    for d in [TUNING_CONFIG["tuning_dir"], TUNING_CONFIG["trials_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)


def setup_stage_logger(stage_name: str) -> logging.Logger:
    """
    Build a dedicated logger for a tuning stage.
    Writes to console (INFO) and a per-stage log file (DEBUG).
    """
    log_path = os.path.join(
        TUNING_CONFIG["tuning_dir"], f"tuning_{stage_name}.log"
    )
    logger = logging.getLogger(f"tune_{stage_name}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _save_best_config(path: str, config_dict: dict, extra: dict = None) -> None:
    """Persist best hyperparams to a JSON file."""
    out = {k: v for k, v in config_dict.items() if _is_json_serialisable(v)}
    if extra:
        out.update({k: v for k, v in extra.items() if _is_json_serialisable(v)})
    with open(path, "w") as f:
        _json.dump(out, f, indent=2)
    print(f"[Tuning] Best config saved → {path}")


def _load_best_config(path: str) -> dict:
    """Load a previously saved best config JSON."""
    with open(path) as f:
        return _json.load(f)


def _build_pruner() -> optuna.pruners.BasePruner:
    """
    Pruning is permanently disabled — every trial runs to completion and relies
    on patience for early stopping.  This always returns NopPruner.
    The "pruner" key in TUNING_CONFIG is kept for forward-compatibility but
    is not acted upon.
    """
    return optuna.pruners.NopPruner()


def _build_sampler() -> optuna.samplers.BaseSampler:
    name = TUNING_CONFIG["sampler"]
    ns   = TUNING_CONFIG["n_startup_trials"]
    seed = CONFIG["seed"]
    if name == "tpe":
        return optuna.samplers.TPESampler(n_startup_trials=ns, seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.RandomSampler(seed=seed)


def _make_trial_callback(
    trial:        optuna.Trial,
    total_epochs: int,
    report_every: int = 5,
) -> Callable:
    """
    Returns a per-epoch callback for Optuna metric reporting.
    Pruning is disabled — trials always run to completion (patience handles
    early stopping).  Returns False unconditionally.
    """
    def callback(ep: int, current_best_f1: float) -> bool:
        if ep % report_every == 0 or ep == total_epochs:
            trial.report(current_best_f1, step=ep)
        return False   # never prune
    return callback


def _trial_dir(stage: int, trial_number: int) -> str:
    d = os.path.join(TUNING_CONFIG["trials_dir"], f"s{stage}_trial_{trial_number}")
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_trial_dir(trial_dir: str) -> None:
    """Remove per-trial checkpoint directory to save disk space."""
    if TUNING_CONFIG.get("keep_trial_ckpts", False):
        return
    try:
        shutil.rmtree(trial_dir, ignore_errors=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
#  PARAM SUGGESTION FUNCTIONS  (centralised — one per stage)
# ──────────────────────────────────────────────────────────────────────

def suggest_s1_params(trial: optuna.Trial) -> dict:
    """
    Suggest all Stage-1 + architecture hyperparameters.
    All search ranges are read from TUNING_CONFIG["s1_space"] so there is a
    single place to edit them.  Architecture params are tuned here because
    Stage 2/3 inherit and freeze them.

    Every trial runs the FULL epoch budget (CONFIG["s1_epochs"]) — patience
    handles early stopping, Optuna pruning is disabled.
    """
    sp = TUNING_CONFIG["s1_space"]

    def _cat(name: str) -> Any:
        return trial.suggest_categorical(name, sp[name])

    def _flt(name: str, log: bool = False) -> float:
        lo, hi = sp[name]
        return trial.suggest_float(name, lo, hi, log=log)

    def _int(name: str) -> int:
        lo, hi = sp[name]
        return trial.suggest_int(name, lo, hi)

    # ── SpecFormer: dim & heads must be compatible → suggest as a pair ──
    dim_heads: str   = _cat("specf_dim_heads")
    specf_dim: int   = int(dim_heads.split("_")[0])
    specf_heads: int = int(dim_heads.split("_")[1])

    # ── Total curriculum length (controls p1 + p2) ──────────────────────
    total_frac: float = trial.suggest_float(
        "s1_total_phase12_frac", 0.60, 0.75
    )

    # ── Relative split (p1 smaller than p2) ─────────────────────────────
    phase1_ratio: float = trial.suggest_float(
        "s1_phase1_ratio", 0.25, 0.45
    )

    # ── Final values ────────────────────────────────────────────────────
    s1_phase1_frac: float = total_frac * phase1_ratio
    s1_phase2_frac: float = total_frac * (1.0 - phase1_ratio)

    # ── Oversampling (conditional) ─────────────────────────────────────
    s1_p3_oversample: bool = _cat("s1_p3_oversample")
    s1_p3_oversample_power: float = (
        _flt("s1_p3_oversample_power") if s1_p3_oversample
        else CONFIG["s1_p3_oversample_power"]
    )
    s1_p3_oversample_max_w: float = (
        _flt("s1_p3_oversample_max_w") if s1_p3_oversample
        else CONFIG["s1_p3_oversample_max_w"]
    )

    params: dict = {
        # ── Phase curriculum ──────────────────────────────────────────
        "s1_phase1_frac":            s1_phase1_frac,
        "s1_phase2_frac":            s1_phase2_frac,
        # ── Training ──────────────────────────────────────────────────
        "s1_max_lr":                 _flt("s1_max_lr",  log=True),
        "s1_min_lr":                 _flt("s1_min_lr",  log=True),
        "s1_batch":                  _cat("s1_batch"),
        "s1_dropout":                _flt("s1_dropout"),
        "s1_mixup":                  _flt("s1_mixup"),
        "s1_accum":                  _cat("s1_accum"),
        "s1_focal_gamma":            _flt("s1_focal_gamma"),
        "s1_label_smooth_hi":        _flt("s1_label_smooth_hi"),
        "s1_label_smooth_lo":        _flt("s1_label_smooth_lo"),
        "s1_ema_reinit_phases":      _cat("s1_ema_reinit_phases"),
        # ── Phase-3 oversampling ──────────────────────────────────────
        "s1_p3_oversample":          s1_p3_oversample,
        "s1_p3_oversample_power":    s1_p3_oversample_power,
        "s1_p3_oversample_max_w":    s1_p3_oversample_max_w,
        "s1_p3_hard_f1_thresh":      _flt("s1_p3_hard_f1_thresh"),
        "s1_p3_oversample_eps":      _flt("s1_p3_oversample_eps"),
        # ── Shared regularisation ─────────────────────────────────────
        "weight_decay":              _flt("weight_decay", log=True),
        "ema_decay":                 _flt("ema_decay"),
        "grad_clip":                 _flt("grad_clip"),
        # ── Aux heads ─────────────────────────────────────────────────
        "aux_head_hidden":           _cat("aux_head_hidden"),
        "aux_loss_weight_init":      _flt("aux_loss_weight_init"),
        "aux_loss_weight_final":     _flt("aux_loss_weight_final"),
        # ── Architecture ──────────────────────────────────────────────
        "specf_dim":                 specf_dim,
        "specf_heads":               specf_heads,
        "specf_layers":              _cat("specf_layers"),
        "specf_drop":                _flt("specf_drop"),
        "specf_patch":               _cat("specf_patch"),
        "wl_embed_dim":              _cat("wl_embed_dim"),
        "branch_drop_prob":          _flt("branch_drop_prob"),
        "fusion_drop":               _flt("fusion_drop"),
        "fusion_heads":              _cat("fusion_heads"),
        "subcenter_K":               _cat("subcenter_K"),
        # ── ArcFace head params (baked at construction time) ──────────
        "s2_arcface_s":              _flt("s2_arcface_s"),
        "s2_arcface_m":              _flt("s2_arcface_m"),
        "s2_arcface_m_delta":        _flt("s2_arcface_m_delta"),
        # ── Augmentation: per-profile probability overrides ───────────
        "aug_heavy_band_drop":       _flt("aug_heavy_band_drop"),
        "aug_heavy_cutout":          _flt("aug_heavy_cutout"),
        "aug_heavy_noise":           _flt("aug_heavy_noise"),
        "aug_heavy_warp":            _flt("aug_heavy_warp"),
        "aug_heavy_mult":            _flt("aug_heavy_mult"),
        "aug_medium_band_drop":      _flt("aug_medium_band_drop"),
        "aug_medium_cutout":         _flt("aug_medium_cutout"),
        "aug_medium_noise":          _flt("aug_medium_noise"),
        "aug_medium_warp":           _flt("aug_medium_warp"),
        "aug_medium_mult":           _flt("aug_medium_mult"),
        "aug_light_band_drop":       _flt("aug_light_band_drop"),
        "aug_light_cutout":          _flt("aug_light_cutout"),
        "aug_light_noise":           _flt("aug_light_noise"),
        # ── Augmentation: global scale params ─────────────────────────
        "band_drop_scale":           _flt("band_drop_scale"),
        "noise_scale":               _flt("noise_scale"),
        "cutout_scale":              _flt("cutout_scale"),
        "max_cutout_bands":          _int("max_cutout_bands"),
    }
    return params


def suggest_s2_params(trial: optuna.Trial) -> dict:
    """
    Suggest Stage-2 training hyperparameters.
    Architecture keys are NOT included here — they come from the best S1 config.
    All search ranges are read from TUNING_CONFIG["s2_space"].
    Every trial runs the full CONFIG["s2_epochs"] budget; patience handles stopping.
    """
    sp = TUNING_CONFIG["s2_space"]

    def _cat(name: str) -> Any:
        return trial.suggest_categorical(name, sp[name])

    def _flt(name: str, log: bool = False) -> float:
        lo, hi = sp[name]
        return trial.suggest_float(name, lo, hi, log=log)

    def _int(name: str) -> int:
        lo, hi = sp[name]
        return trial.suggest_int(name, lo, hi)

    params: dict = {
        "s2_head_lr":          _flt("s2_head_lr",        log=True),
        "s2_back_lr":          _flt("s2_back_lr",        log=True),
        "s2_min_lr":           _flt("s2_min_lr",         log=True),
        "s2_batch":            _cat("s2_batch"),
        "s2_warmup_ep":        _int("s2_warmup_ep"),
        "s2_sgdr_T0":          _int("s2_sgdr_T0"),
        "s2_sgdr_Tmult":       _cat("s2_sgdr_Tmult"),
        "s2_dropout":          _flt("s2_dropout"),
        "s2_arcface_m":        _flt("s2_arcface_m"),
        "s2_arcface_m0":       _flt("s2_arcface_m0"),
        "s2_arcface_m_delta":  _flt("s2_arcface_m_delta"),
        "s2_margin_warmup_ep": _int("s2_margin_warmup_ep"),
        "s2_focal_gamma":      _flt("s2_focal_gamma"),
        "cdws_max_weight":     _flt("cdws_max_weight"),
        "cdws_eps":            _flt("cdws_eps"),
        "supcon_weight":       _flt("supcon_weight"),
        "supcon_temp":         _flt("supcon_temp",        log=True),
        "proto_weight":        _flt("proto_weight"),
        "proto_temp":          _flt("proto_temp",         log=True),
        "bal_n_cls":           _int("bal_n_cls"),
        "bal_n_spc":           _int("bal_n_spc"),
        "weight_decay":        _flt("weight_decay",       log=True),
        "grad_clip":           _flt("grad_clip"),
        "ema_decay":           _flt("ema_decay"),
    }
    return params


def suggest_s3_params(trial: optuna.Trial) -> dict:
    """
    Suggest Stage-3 SAM + SWA hyperparameters.
    All search ranges are read from TUNING_CONFIG["s3_space"].
    Every trial runs the full CONFIG["s3_epochs"] budget; patience handles stopping.
    """
    sp = TUNING_CONFIG["s3_space"]

    def _cat(name: str) -> Any:
        return trial.suggest_categorical(name, sp[name])

    def _flt(name: str, log: bool = False) -> float:
        lo, hi = sp[name]
        return trial.suggest_float(name, lo, hi, log=log)

    def _int(name: str) -> int:
        lo, hi = sp[name]
        return trial.suggest_int(name, lo, hi)

    params: dict = {
        "s3_swa_lr":          _flt("s3_swa_lr",         log=True),
        "s3_cycle_len":       _int("s3_cycle_len"),
        "s3_sam_rho":         _flt("s3_sam_rho"),
        "s3_greedy":          _cat("s3_greedy"),
        "s3_aux_loss_weight": _flt("s3_aux_loss_weight"),
        "weight_decay":       _flt("weight_decay",       log=True),
        "grad_clip":          _flt("grad_clip"),
    }
    return params


# ──────────────────────────────────────────────────────────────────────
#  OPTUNA OBJECTIVE FUNCTIONS  (one per stage)
# ──────────────────────────────────────────────────────────────────────

def objective_s1(
    trial:  optuna.Trial,
    splits: Tuple,
    logger: logging.Logger,
) -> float:
    """
    Objective for Stage 1.
    Builds a fresh model and runs the FULL Stage-1 training loop
    (CONFIG["s1_epochs"] epochs, CONFIG["s1_patience"] early-stopping patience).
    Pruning is disabled — every trial runs to completion.
    Augmentation intensity parameters tuned by Optuna are passed directly
    into RiceSeedDataset so they are respected by the phase loaders.
    """
    all_labels, train_idx, val_idx, test_idx = splits
    device = CONFIG["device"]
    t_cfg  = TUNING_CONFIG

    params = suggest_s1_params(trial)

    # All trials run the full epoch budget — patience handles early stopping
    params["s1_epochs"]   = CONFIG["s1_epochs"]
    params["s1_patience"] = CONFIG["s1_patience"]

    t_dir               = _trial_dir(1, trial.number)
    params["output_dir"] = t_dir

    logger.info(
        f"[S1 Trial {trial.number}] "
        f"epochs={params['s1_epochs']}  patience={params['s1_patience']}  "
        f"lr={params['s1_max_lr']:.2e}  batch={params['s1_batch']}  "
        f"ph1={params['s1_phase1_frac']:.2f}  ph2={params['s1_phase2_frac']:.2f}  "
        f"specf={params['specf_dim']}×{params['specf_heads']}L{params['specf_layers']}  "
        f"aug_heavy_bd={params['aug_heavy_band_drop']:.3f}  "
        f"noise_scale={params['noise_scale']:.3f}  "
        f"max_cutout={params['max_cutout_bands']}"
    )

    val_f1: float = 0.0
    try:
        with temp_config(params):
            set_seed(CONFIG["seed"] + trial.number)

            model = SpectralQuadNet(
                num_classes=CONFIG["num_classes"],
                num_bands=CONFIG["num_bands"],
                dropout=CONFIG["s1_dropout"],
                wl_embed_dim=CONFIG["wl_embed_dim"],
                cfg=CONFIG,
            ).to(device)
            ema = ModelEMA(model, decay=CONFIG["ema_decay"])

            # ── Phase loaders: pass tuned aug params into dataset ────────
            def _s1_ldr(aug_str: str) -> DataLoader:
                ds = RiceSeedDataset(
                    train_idx,
                    aug_strength      = aug_str,
                    # per-profile probability overrides from params
                    aug_heavy_band_drop  = params.get("aug_heavy_band_drop"),
                    aug_heavy_cutout     = params.get("aug_heavy_cutout"),
                    aug_heavy_noise      = params.get("aug_heavy_noise"),
                    aug_heavy_warp       = params.get("aug_heavy_warp"),
                    aug_heavy_mult       = params.get("aug_heavy_mult"),
                    aug_medium_band_drop = params.get("aug_medium_band_drop"),
                    aug_medium_cutout    = params.get("aug_medium_cutout"),
                    aug_medium_noise     = params.get("aug_medium_noise"),
                    aug_medium_warp      = params.get("aug_medium_warp"),
                    aug_medium_mult      = params.get("aug_medium_mult"),
                    aug_light_band_drop  = params.get("aug_light_band_drop"),
                    aug_light_cutout     = params.get("aug_light_cutout"),
                    aug_light_noise      = params.get("aug_light_noise"),
                    # global scale params
                    band_drop_scale      = params.get("band_drop_scale"),
                    noise_scale          = params.get("noise_scale"),
                    cutout_scale         = params.get("cutout_scale"),
                    max_cutout_bands     = params.get("max_cutout_bands"),
                )
                return DataLoader(ds, batch_size=CONFIG["s1_batch"],
                                  shuffle=True, drop_last=True, num_workers=0)

            phase_loaders = {1: _s1_ldr("heavy"), 2: _s1_ldr("medium"), 3: _s1_ldr("light")}
            _, val_ldr, _ = build_loaders(train_idx, val_idx, test_idx,
                                          CONFIG["s1_batch"], train_aug="none")

            full_ep = params["s1_epochs"]
            ckpt_path = stage_ckpt_path(1)
            cb        = _make_trial_callback(trial, full_ep,
                                             report_every=max(1, full_ep // 20))

            val_f1 = run_stage1(model, ema, phase_loaders, val_ldr,
                                device, ckpt_path, trial_callback=cb)

        # ── Track & save new best ─────────────────────────────────────
        s1_ckpt_in_trial = os.path.join(t_dir, "best_stage1.pth")
        if os.path.isfile(s1_ckpt_in_trial):
            best_so_far: float = -1.0
            bcfg  = t_cfg["best_config_s1"]
            bckpt = t_cfg["best_ckpt_s1"]
            if os.path.isfile(bcfg):
                try:
                    prev = _load_best_config(bcfg)
                    best_so_far = float(prev.get("val_f1", -1.0))
                except Exception:
                    pass
            if val_f1 > best_so_far:
                shutil.copy2(s1_ckpt_in_trial, bckpt)
                _save_best_config(bcfg, params,
                                  extra={"val_f1": val_f1, "trial_number": trial.number})
                logger.info(
                    f"[S1 Trial {trial.number}] ✓ NEW BEST  F1={val_f1:.4f} "
                    f"(prev={best_so_far:.4f}) — ckpt saved."
                )

    except Exception as exc:
        logger.warning(f"[S1 Trial {trial.number}] FAILED: {exc}\n{traceback.format_exc()}")
        val_f1 = 0.0
    finally:
        _cleanup_trial_dir(t_dir)
        torch.cuda.empty_cache()

    logger.info(f"[S1 Trial {trial.number}] finished — val_F1={val_f1:.4f}")
    return val_f1


def objective_s2(
    trial:        optuna.Trial,
    splits:       Tuple,
    best_s1_ckpt: str,
    best_s1_cfg:  dict,
    logger:       logging.Logger,
) -> float:
    """
    Objective for Stage 2.
    Loads best Stage-1 weights and runs the FULL Stage-2 training loop
    (CONFIG["s2_epochs"] epochs, CONFIG["s2_patience"] early-stopping patience).
    Architecture is frozen to best_s1_cfg values.  Pruning is disabled.
    """
    all_labels, train_idx, val_idx, test_idx = splits
    device = CONFIG["device"]
    t_cfg  = TUNING_CONFIG

    params = suggest_s2_params(trial)
    # Inherit architecture from best S1
    for k in _ARCH_KEYS:
        if k in best_s1_cfg:
            params[k] = best_s1_cfg[k]

    # All trials run the full epoch budget — patience handles early stopping
    params["s2_epochs"]   = CONFIG["s2_epochs"]
    params["s2_patience"] = CONFIG["s2_patience"]

    t_dir               = _trial_dir(2, trial.number)
    params["output_dir"] = t_dir

    logger.info(
        f"[S2 Trial {trial.number}] "
        f"epochs={params['s2_epochs']}  patience={params['s2_patience']}  "
        f"hLR={params['s2_head_lr']:.2e}  bLR={params['s2_back_lr']:.2e}  "
        f"supcon_w={params['supcon_weight']:.3f}  proto_w={params['proto_weight']:.3f}"
    )

    val_f1: float = 0.0
    try:
        with temp_config(params):
            set_seed(CONFIG["seed"] + trial.number + 1000)

            model = SpectralQuadNet(
                num_classes=CONFIG["num_classes"],
                num_bands=CONFIG["num_bands"],
                dropout=CONFIG["s2_dropout"],
                wl_embed_dim=CONFIG["wl_embed_dim"],
                cfg=CONFIG,
            ).to(device)
            ema = ModelEMA(model, decay=CONFIG["ema_decay"])

            # Load best S1 weights
            load_ckpt(best_s1_ckpt, model, ema, device)

            # Bootstrap ArcFace from linear head
            lw = model.linear_head[-1].weight.data.clone()
            model.arcface_head.init_from_linear(lw)
            ema.shadow.arcface_head.init_from_linear(lw)

            # Recover class difficulty info from S1 checkpoint
            ckpt_data   = torch.load(best_s1_ckpt, map_location=device, weights_only=False)
            class_f1_s1 = ckpt_data.get("class_f1",    {})
            cdws_wts_s1 = ckpt_data.get("cdws_weights", {})
            if class_f1_s1:
                class_f1_s1 = {int(k): float(v) for k, v in class_f1_s1.items()}
                cdws_wts_s1 = {int(k): float(v) for k, v in cdws_wts_s1.items()}

            s2_batch: int = CONFIG.get("s2_batch", 128)
            tr2, va2, _ = build_loaders(
                train_idx, val_idx, test_idx, s2_batch,
                balanced=True, all_labels=all_labels,
                train_aug="light",
                class_weights=cdws_wts_s1 if cdws_wts_s1 else None,
            )

            full_ep   = params["s2_epochs"]
            ckpt_path = stage_ckpt_path(2)
            cb        = _make_trial_callback(trial, full_ep,
                                             report_every=max(1, full_ep // 15))

            val_f1 = run_stage2(
                model, ema, tr2, va2, device, ckpt_path,
                class_f1=class_f1_s1 if class_f1_s1 else None,
                trial_callback=cb,
            )

        # ── Track & save new best ─────────────────────────────────────
        s2_ckpt_in_trial = os.path.join(t_dir, "best_stage2.pth")
        if os.path.isfile(s2_ckpt_in_trial):
            best_so_far: float = -1.0
            bcfg  = t_cfg["best_config_s2"]
            bckpt = t_cfg["best_ckpt_s2"]
            if os.path.isfile(bcfg):
                try:
                    prev = _load_best_config(bcfg)
                    best_so_far = float(prev.get("val_f1", -1.0))
                except Exception:
                    pass
            if val_f1 > best_so_far:
                shutil.copy2(s2_ckpt_in_trial, bckpt)
                _save_best_config(bcfg, params,
                                  extra={"val_f1": val_f1, "trial_number": trial.number})
                logger.info(
                    f"[S2 Trial {trial.number}] ✓ NEW BEST  F1={val_f1:.4f} "
                    f"(prev={best_so_far:.4f}) — ckpt saved."
                )

    except Exception as exc:
        logger.warning(f"[S2 Trial {trial.number}] FAILED: {exc}\n{traceback.format_exc()}")
        val_f1 = 0.0
    finally:
        _cleanup_trial_dir(t_dir)
        torch.cuda.empty_cache()

    logger.info(f"[S2 Trial {trial.number}] finished — val_F1={val_f1:.4f}")
    return val_f1


def objective_s3(
    trial:        optuna.Trial,
    splits:       Tuple,
    best_s2_ckpt: str,
    best_s2_cfg:  dict,
    s2_best_f1:   float,
    logger:       logging.Logger,
) -> float:
    """
    Objective for Stage 3.
    Loads best Stage-2 weights and runs the FULL Stage-3 (SAM+SWA) loop
    (CONFIG["s3_epochs"] epochs).  Pruning is disabled.
    """
    all_labels, train_idx, val_idx, test_idx = splits
    device = CONFIG["device"]
    t_cfg  = TUNING_CONFIG

    params = suggest_s3_params(trial)
    # Inherit architecture from best S2 (which itself came from best S1)
    for k in _ARCH_KEYS:
        if k in best_s2_cfg:
            params[k] = best_s2_cfg[k]

    # All trials run the full epoch budget
    params["s3_epochs"] = CONFIG["s3_epochs"]

    t_dir               = _trial_dir(3, trial.number)
    params["output_dir"] = t_dir

    logger.info(
        f"[S3 Trial {trial.number}] "
        f"epochs={params['s3_epochs']}  "
        f"swa_lr={params['s3_swa_lr']:.2e}  rho={params['s3_sam_rho']:.3f}  "
        f"cycle={params['s3_cycle_len']}  greedy={params['s3_greedy']}"
    )

    val_f1: float = 0.0
    try:
        with temp_config(params):
            set_seed(CONFIG["seed"] + trial.number + 2000)

            model = SpectralQuadNet(
                num_classes=CONFIG["num_classes"],
                num_bands=CONFIG["num_bands"],
                dropout=CONFIG["s2_dropout"],
                wl_embed_dim=CONFIG["wl_embed_dim"],
                cfg=CONFIG,
            ).to(device)
            ema = ModelEMA(model, decay=CONFIG["ema_decay"])

            # Load best S2 weights
            load_ckpt(best_s2_ckpt, model, ema, device)

            ckpt_data   = torch.load(best_s2_ckpt, map_location=device, weights_only=False)
            cdws_wts_s2 = ckpt_data.get("cdws_weights", {})
            if cdws_wts_s2:
                cdws_wts_s2 = {int(k): float(v) for k, v in cdws_wts_s2.items()}

            s2_batch: int = CONFIG.get("s2_batch", 128)
            tr3, va3, _ = build_loaders(
                train_idx, val_idx, test_idx, s2_batch,
                balanced=True, all_labels=all_labels,
                train_aug="light",
                class_weights=cdws_wts_s2 if cdws_wts_s2 else None,
            )

            full_ep   = params["s3_epochs"]
            cycle_len: int = params.get("s3_cycle_len", CONFIG["s3_cycle_len"])
            ckpt_path = stage_ckpt_path(3)
            cb        = _make_trial_callback(trial, full_ep,
                                             report_every=max(1, cycle_len))

            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()

            val_f1 = run_stage3_swa(
                model, ema, tr3, va3, device, ckpt_path,
                prev_best_f1=s2_best_f1,
                trial_callback=cb,
            )

        # ── Track & save new best ─────────────────────────────────────
        s3_ckpt_in_trial = os.path.join(t_dir, "best_stage3.pth")
        if os.path.isfile(s3_ckpt_in_trial):
            best_so_far: float = -1.0
            bcfg  = t_cfg["best_config_s3"]
            bckpt = t_cfg["best_ckpt_s3"]
            if os.path.isfile(bcfg):
                try:
                    prev = _load_best_config(bcfg)
                    best_so_far = float(prev.get("val_f1", -1.0))
                except Exception:
                    pass
            if val_f1 > best_so_far:
                shutil.copy2(s3_ckpt_in_trial, bckpt)
                _save_best_config(bcfg, params,
                                  extra={"val_f1": val_f1, "trial_number": trial.number})
                logger.info(
                    f"[S3 Trial {trial.number}] ✓ NEW BEST  F1={val_f1:.4f} "
                    f"(prev={best_so_far:.4f}) — ckpt saved."
                )

    except Exception as exc:
        logger.warning(f"[S3 Trial {trial.number}] FAILED: {exc}\n{traceback.format_exc()}")
        val_f1 = 0.0
    finally:
        _cleanup_trial_dir(t_dir)
        torch.cuda.empty_cache()

    logger.info(f"[S3 Trial {trial.number}] finished — val_F1={val_f1:.4f}")
    return val_f1


# ──────────────────────────────────────────────────────────────────────
#  STAGE TUNING RUNNERS  (each creates/resumes its own Optuna study)
# ──────────────────────────────────────────────────────────────────────

def _log_study_summary(
    study:  optuna.Study,
    stage:  int,
    logger: logging.Logger,
) -> None:
    """Print and log a concise study summary."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    try:
        best_val    = study.best_value
        best_params = study.best_params
    except Exception:
        best_val    = float("nan")
        best_params = {}
    logger.info(
        f"\n{'═'*66}\n"
        f"  Stage {stage} HPO Summary\n"
        f"  Completed : {len(completed)}  Pruned : {len(pruned)}  Failed : {len(failed)}\n"
        f"  Best F1   : {best_val:.4f}\n"
        f"  Best params:\n"
        + "\n".join(f"    {k}: {v}" for k, v in best_params.items())
        + f"\n{'═'*66}"
    )


def _study_callback(logger: logging.Logger, stage: int) -> Callable:
    """Returns an Optuna study callback that logs after every trial."""
    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        state_str = trial.state.name
        val_str   = f"{trial.value:.4f}" if trial.value is not None else "N/A"
        logger.info(
            f"[S{stage}] Trial {trial.number:>4d} | {state_str:<12s} | "
            f"F1={val_str}  |  best so far={study.best_value:.4f}"
        )
    return cb


def run_stage1_tuning(splits: Tuple) -> Tuple[dict, str]:
    """
    Run Optuna Stage-1 HPO.

    Returns
    -------
    best_config : dict  — best hyperparameter config
    best_ckpt   : str   — path to best Stage-1 checkpoint
    """
    t_cfg  = TUNING_CONFIG
    logger = setup_stage_logger("s1")
    bcfg   = t_cfg["best_config_s1"]
    bckpt  = t_cfg["best_ckpt_s1"]

    # ── Stage-level resume: if already done, skip ─────────────────────
    if os.path.isfile(bcfg) and os.path.isfile(bckpt):
        cfg = _load_best_config(bcfg)
        logger.info(
            f"[S1 Tuning] Already complete (F1={cfg.get('val_f1',0):.4f}).  "
            f"Loading saved config & checkpoint."
        )
        return cfg, bckpt

    tune_ep_max = CONFIG["s1_epochs"]
    logger.info(
        f"\n{'═'*66}\n"
        f"  STAGE 1 HPO  — {t_cfg['n_trials_s1']} trials\n"
        f"  DB            : {t_cfg['s1_db']}\n"
        f"  Epoch budget  : {tune_ep_max} per trial (full training — patience controls stopping)\n"
        f"  Sampler       : {t_cfg['sampler']}  |  Pruner: {t_cfg['pruner']} (disabled)\n"
        f"{'═'*66}"
    )

    study = optuna.create_study(
        study_name     = t_cfg["s1_study_name"],
        storage        = t_cfg["s1_db"],
        direction      = "maximize",
        load_if_exists = True,
        sampler        = _build_sampler(),
        pruner         = _build_pruner(),
    )

    done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, t_cfg["n_trials_s1"] - done)
    logger.info(f"[S1 Tuning] {done} trials already done, running {remaining} more.")

    if remaining > 0:
        study.optimize(
            lambda trial: objective_s1(trial, splits, logger),
            n_trials          = remaining,
            n_jobs            = t_cfg["n_jobs"],
            timeout           = t_cfg["timeout_s1"],
            show_progress_bar = False,
            callbacks         = [_study_callback(logger, 1)],
        )

    _log_study_summary(study, 1, logger)

    if not os.path.isfile(bcfg) or not os.path.isfile(bckpt):
        raise RuntimeError(
            "[S1 Tuning] No best checkpoint found after tuning.  "
            "All trials may have failed — check tuning_s1.log."
        )

    best_config = _load_best_config(bcfg)
    logger.info(f"[S1 Tuning] DONE.  Best val_F1={best_config.get('val_f1',0):.4f}")
    return best_config, bckpt


def run_stage2_tuning(
    splits:       Tuple,
    best_s1_ckpt: str,
    best_s1_cfg:  dict,
) -> Tuple[dict, str]:
    """
    Run Optuna Stage-2 HPO.
    Architecture is frozen to best_s1_cfg; only S2 training params are searched.
    """
    t_cfg  = TUNING_CONFIG
    logger = setup_stage_logger("s2")
    bcfg   = t_cfg["best_config_s2"]
    bckpt  = t_cfg["best_ckpt_s2"]

    # ── Stage-level resume ────────────────────────────────────────────
    if os.path.isfile(bcfg) and os.path.isfile(bckpt):
        cfg = _load_best_config(bcfg)
        logger.info(
            f"[S2 Tuning] Already complete (F1={cfg.get('val_f1',0):.4f}).  "
            f"Loading saved config & checkpoint."
        )
        return cfg, bckpt

    tune_ep_max = CONFIG["s2_epochs"]
    logger.info(
        f"\n{'═'*66}\n"
        f"  STAGE 2 HPO  — {t_cfg['n_trials_s2']} trials\n"
        f"  DB            : {t_cfg['s2_db']}\n"
        f"  Epoch budget  : {tune_ep_max} per trial (full training — patience controls stopping)\n"
        f"  S1 best F1    : {best_s1_cfg.get('val_f1', '?')}\n"
        f"  S1 ckpt       : {best_s1_ckpt}\n"
        f"  Arch (frozen) : "
        + ", ".join(f"{k}={best_s1_cfg.get(k,'?')}" for k in _ARCH_KEYS if k in best_s1_cfg)
        + f"\n{'═'*66}"
    )

    study = optuna.create_study(
        study_name     = t_cfg["s2_study_name"],
        storage        = t_cfg["s2_db"],
        direction      = "maximize",
        load_if_exists = True,
        sampler        = _build_sampler(),
        pruner         = _build_pruner(),
    )

    done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, t_cfg["n_trials_s2"] - done)
    logger.info(f"[S2 Tuning] {done} trials already done, running {remaining} more.")

    if remaining > 0:
        study.optimize(
            lambda trial: objective_s2(trial, splits, best_s1_ckpt, best_s1_cfg, logger),
            n_trials          = remaining,
            n_jobs            = t_cfg["n_jobs"],
            timeout           = t_cfg["timeout_s2"],
            show_progress_bar = False,
            callbacks         = [_study_callback(logger, 2)],
        )

    _log_study_summary(study, 2, logger)

    if not os.path.isfile(bcfg) or not os.path.isfile(bckpt):
        raise RuntimeError(
            "[S2 Tuning] No best checkpoint found after tuning.  "
            "All trials may have failed — check tuning_s2.log."
        )

    best_config = _load_best_config(bcfg)
    logger.info(f"[S2 Tuning] DONE.  Best val_F1={best_config.get('val_f1',0):.4f}")
    return best_config, bckpt


def run_stage3_tuning(
    splits:       Tuple,
    best_s2_ckpt: str,
    best_s2_cfg:  dict,
) -> Tuple[dict, str]:
    """
    Run Optuna Stage-3 HPO.
    Architecture + S2 params are frozen; only SAM/SWA params are searched.
    """
    t_cfg  = TUNING_CONFIG
    logger = setup_stage_logger("s3")
    bcfg   = t_cfg["best_config_s3"]
    bckpt  = t_cfg["best_ckpt_s3"]

    # ── Stage-level resume ────────────────────────────────────────────
    if os.path.isfile(bcfg) and os.path.isfile(bckpt):
        cfg = _load_best_config(bcfg)
        logger.info(
            f"[S3 Tuning] Already complete (F1={cfg.get('val_f1',0):.4f}).  "
            f"Loading saved config & checkpoint."
        )
        return cfg, bckpt

    s2_best_f1: float = float(best_s2_cfg.get("val_f1", 0.0))
    tune_ep_max: int = CONFIG["s3_epochs"]
    logger.info(
        f"\n{'═'*66}\n"
        f"  STAGE 3 HPO  — {t_cfg['n_trials_s3']} trials\n"
        f"  DB            : {t_cfg['s3_db']}\n"
        f"  Epoch budget  : {tune_ep_max} per trial (full training — patience controls stopping)\n"
        f"  S2 best F1    : {s2_best_f1:.4f}  (SAM must beat this to accept snapshots)\n"
        f"  S2 ckpt       : {best_s2_ckpt}\n"
        f"{'═'*66}"
    )

    study = optuna.create_study(
        study_name     = t_cfg["s3_study_name"],
        storage        = t_cfg["s3_db"],
        direction      = "maximize",
        load_if_exists = True,
        sampler        = _build_sampler(),
        pruner         = _build_pruner(),
    )

    done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, t_cfg["n_trials_s3"] - done)
    logger.info(f"[S3 Tuning] {done} trials already done, running {remaining} more.")

    if remaining > 0:
        study.optimize(
            lambda trial: objective_s3(
                trial, splits, best_s2_ckpt, best_s2_cfg, s2_best_f1, logger
            ),
            n_trials          = remaining,
            n_jobs            = t_cfg["n_jobs"],
            timeout           = t_cfg["timeout_s3"],
            show_progress_bar = False,
            callbacks         = [_study_callback(logger, 3)],
        )

    _log_study_summary(study, 3, logger)

    if not os.path.isfile(bcfg) or not os.path.isfile(bckpt):
        raise RuntimeError(
            "[S3 Tuning] No best checkpoint found after tuning.  "
            "All trials may have failed — check tuning_s3.log."
        )

    best_config = _load_best_config(bcfg)
    logger.info(f"[S3 Tuning] DONE.  Best val_F1={best_config.get('val_f1',0):.4f}")
    return best_config, bckpt


# ──────────────────────────────────────────────────────────────────────
#  MASTER TUNING ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def tune_all_stages() -> None:
    """
    Sequential Stage 1 → Stage 2 → Stage 3 Optuna HPO pipeline.

    Stage-level resume: if a best_config_sN.json + best_ckpt_sN.pth pair
    already exists on disk, that stage is skipped entirely.
    """
    device = CONFIG["device"]
    _make_tuning_dirs()

    # ── Master log ────────────────────────────────────────────────────
    master_log = os.path.join(TUNING_CONFIG["tuning_dir"], "tuning_master.log")
    master_logger = logging.getLogger("tune_master")
    master_logger.setLevel(logging.INFO)
    if not master_logger.handlers:
        fh = logging.FileHandler(master_log, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        master_logger.addHandler(fh)
        master_logger.addHandler(ch)

    master_logger.info(
        f"\n{'█'*66}\n"
        f"  SpectralQuadNet  —  End-to-End Optuna HPO\n"
        f"  S1 trials: {TUNING_CONFIG['n_trials_s1']}  |  "
        f"S2 trials: {TUNING_CONFIG['n_trials_s2']}  |  "
        f"S3 trials: {TUNING_CONFIG['n_trials_s3']}\n"
        f"  All trials run at FULL epoch budget with patience early-stopping\n"
        f"  S1={CONFIG['s1_epochs']} ep  S2={CONFIG['s2_epochs']} ep  S3={CONFIG['s3_epochs']} ep\n"
        f"  Tuning dir : {TUNING_CONFIG['tuning_dir']}\n"
        f"  Device     : {device}\n"
        f"{'█'*66}"
    )

    # ── Load data (once; shared across all trials) ────────────────────
    master_logger.info("[Master] Loading dataset …")
    _load_data_mmap(CONFIG["patches_data"], CONFIG["labels_path"])
    _load_wavelengths_to_gpu(CONFIG["wavelength_path"], device)

    splits = build_splits()   # (all_labels, train_idx, val_idx, test_idx)
    all_labels, train_idx, val_idx, test_idx = splits
    master_logger.info(
        f"[Master] Dataset ready.  "
        f"Train={len(train_idx):,}  Val={len(val_idx):,}  Test={len(test_idx):,}"
    )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 HPO
    # ══════════════════════════════════════════════════════════════════
    master_logger.info("\n[Master] ── Starting Stage 1 HPO ──")
    best_cfg_s1, best_ckpt_s1 = run_stage1_tuning(splits)
    master_logger.info(
        f"[Master] Stage 1 HPO complete.  "
        f"Best F1={best_cfg_s1.get('val_f1', '?'):.4f}  "
        f"Ckpt={best_ckpt_s1}"
    )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2 HPO  (uses best S1 arch + ckpt)
    # ══════════════════════════════════════════════════════════════════
    master_logger.info("\n[Master] ── Starting Stage 2 HPO ──")
    best_cfg_s2, best_ckpt_s2 = run_stage2_tuning(splits, best_ckpt_s1, best_cfg_s1)
    master_logger.info(
        f"[Master] Stage 2 HPO complete.  "
        f"Best F1={best_cfg_s2.get('val_f1', '?'):.4f}  "
        f"Ckpt={best_ckpt_s2}"
    )

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 3 HPO  (uses best S2 arch + ckpt)
    # ══════════════════════════════════════════════════════════════════
    master_logger.info("\n[Master] ── Starting Stage 3 HPO ──")
    best_cfg_s3, best_ckpt_s3 = run_stage3_tuning(splits, best_ckpt_s2, best_cfg_s2)
    master_logger.info(
        f"[Master] Stage 3 HPO complete.  "
        f"Best F1={best_cfg_s3.get('val_f1', '?'):.4f}  "
        f"Ckpt={best_ckpt_s3}"
    )

    # ══════════════════════════════════════════════════════════════════
    #  FINAL EVALUATION on best checkpoint across all stages
    # ══════════════════════════════════════════════════════════════════
    master_logger.info("\n[Master] ── Final Evaluation ──")
    best_final_ckpt = _pick_best_checkpoint(best_ckpt_s1, best_ckpt_s2, best_ckpt_s3)
    master_logger.info(f"[Master] Best overall checkpoint: {best_final_ckpt}")

    # Build final model using the best architecture (from S3 cfg, which inherits S1)
    final_cfg = {**CONFIG, **best_cfg_s1, **best_cfg_s2, **best_cfg_s3}
    final_cfg["output_dir"] = TUNING_CONFIG["tuning_dir"]
    Path(final_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    with temp_config(final_cfg):
        final_model = SpectralQuadNet(
            num_classes=CONFIG["num_classes"],
            num_bands=CONFIG["num_bands"],
            dropout=CONFIG.get("s2_dropout", 0.10),
            wl_embed_dim=CONFIG["wl_embed_dim"],
            cfg=CONFIG,
        ).to(device)
        final_ema = ModelEMA(final_model, decay=CONFIG["ema_decay"])
        _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx, 256)
        final_evaluation(final_model, final_ema, test_ldr, device, best_final_ckpt)

    # ── Print consolidated best configs ───────────────────────────────
    master_logger.info(
        f"\n{'═'*66}\n"
        f"  HPO COMPLETE — BEST CONFIGURATIONS\n"
        f"{'═'*66}\n"
        f"  Stage 1 (arch + training):  {TUNING_CONFIG['best_config_s1']}\n"
        f"    val_F1 = {best_cfg_s1.get('val_f1', '?')}\n"
        f"  Stage 2 (training only):    {TUNING_CONFIG['best_config_s2']}\n"
        f"    val_F1 = {best_cfg_s2.get('val_f1', '?')}\n"
        f"  Stage 3 (SAM/SWA only):     {TUNING_CONFIG['best_config_s3']}\n"
        f"    val_F1 = {best_cfg_s3.get('val_f1', '?')}\n"
        f"  Final ckpt : {best_final_ckpt}\n"
        f"{'═'*66}"
    )


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SpectralQuadNet — Optuna HPO or standard training"
    )
    parser.add_argument(
        "--mode",
        choices=["tune", "train"],
        default="tune",
        help="'tune'  → run Optuna HPO (default)  |  'train' → run original training pipeline",
    )
    parser.add_argument(
        "--n_trials_s1", type=int, default=None,
        help="Override TUNING_CONFIG['n_trials_s1']"
    )
    parser.add_argument(
        "--n_trials_s2", type=int, default=None,
        help="Override TUNING_CONFIG['n_trials_s2']"
    )
    parser.add_argument(
        "--n_trials_s3", type=int, default=None,
        help="Override TUNING_CONFIG['n_trials_s3']"
    )
    parser.add_argument(
        "--keep_ckpts", action="store_true", default=False,
        help="Keep per-trial checkpoint subdirectories (large disk usage)"
    )
    args = parser.parse_args()

    # ── Apply CLI overrides to TUNING_CONFIG ──────────────────────────
    if args.n_trials_s1 is not None:
        TUNING_CONFIG["n_trials_s1"] = args.n_trials_s1
    if args.n_trials_s2 is not None:
        TUNING_CONFIG["n_trials_s2"] = args.n_trials_s2
    if args.n_trials_s3 is not None:
        TUNING_CONFIG["n_trials_s3"] = args.n_trials_s3
    TUNING_CONFIG["keep_trial_ckpts"] = args.keep_ckpts

    # ── Run ───────────────────────────────────────────────────────────
    try:
        if args.mode == "tune":
            tune_all_stages()
        else:
            main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.  "
              "All completed trials have been saved to the SQLite DBs and can be resumed.")
        sys.exit(0)
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)