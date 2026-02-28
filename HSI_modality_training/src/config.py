from __future__ import annotations

import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════
#  WAVELENGTH CONSTANTS
# ══════════════════════════════════════════════════════════════════════

WL_MIN: float = 385.0
WL_MAX: float = 1000.0


# ══════════════════════════════════════════════════════════════════════
#  MASTER CONFIG
# ══════════════════════════════════════════════════════════════════════

CONFIG: dict = {

    # ── Paths ────────────────────────────────────────────────────────
    "patches_data":         "./dataset/patches.npy",
    "labels_path":          "./dataset/labels.npy",
    "output_dir":           "./output_v10/",

    # ── Dataset ──────────────────────────────────────────────────────
    "num_bands":            256,
    "num_classes":          90,

    # Aug profile hyper-params (used inside RiceSeedDataset)
    "aug_max_cutout_bands": 20,
    "aug_noise_std":        0.02,

    # Data splits
    "val_split":            0.30,   # fraction for val+test combined
    "test_split":           0.50,   # fraction of val+test that becomes test
    "split_seed":           42,

    # ── Stage 1 — 3-Phase Progressive Augmentation ───────────────────
    "s1_epochs":            300,
    "s1_phase1_frac":       0.40,
    "s1_phase2_frac":       0.30,
    "s1_batch":             128,
    "s1_max_lr":            8e-4,
    "s1_dropout":           0.30,
    "s1_mixup":             0.40,
    "s1_patience":          50,
    "s1_accum":             1,
    "s1_focal_gamma":       2.0,
    "s1_label_smooth_hi":   0.05,
    "s1_label_smooth_lo":   0.00,
    "s1_ema_reinit_phases": True,

    # ── Architecture ─────────────────────────────────────────────────
    "branch_drop_prob":     0.05,
    "subcenter_K":          3,
    "branch_internal_drop": 0.10,   # dropout used inside SpectralProfileBranch
                                    # and SpectralStatsBranch projections

    # Wavelength positional encoding
    "wl_embed_dim":         16,

    # SpecFormer branch
    "specf_patch":          8,
    "specf_dim":            256,
    "specf_heads":          8,
    "specf_layers":         4,
    "specf_drop":           0.15,

    # Branch cross-attention fusion
    "fusion_heads":         4,
    "fusion_drop":          0.10,

    # ── Stage 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR ─
    "s2_epochs":            120,
    "s2_batch":             128,
    "s2_head_lr":           1.5e-4,
    "s2_back_lr":           1.5e-5,
    "s2_min_lr":            1e-7,
    "s2_warmup_ep":         5,
    "s2_sgdr_T0":           10,
    "s2_sgdr_Tmult":        2,
    "s2_dropout":           0.10,
    "s2_patience":          40,
    "s2_arcface_s":         32.0,
    "s2_arcface_m":         0.35,
    "s2_arcface_m0":        0.02,
    "s2_arcface_m_delta":   0.10,
    "s2_margin_warmup_ep":  50,
    "s2_focal_gamma":       1.5,

    # Class-difficulty weighting (CDWS)
    "cdws_max_weight":      3.0,
    "cdws_eps":             0.05,

    # Contrastive losses
    "supcon_weight":        0.25,
    "supcon_temp":          0.10,
    "proto_weight":         0.12,
    "proto_temp":           0.10,

    # Balanced sampler
    "bal_n_cls":            16,
    "bal_n_spc":            8,

    # ── Stage 3 — SAM + Greedy SWA ───────────────────────────────────
    "s3_epochs":            100,
    "s3_swa_lr":            4e-5,
    "s3_cycle_len":         8,
    "s3_sam_rho":           0.05,
    "s3_greedy":            True,
    "s3_focal_gamma":       1.0,
    "s3_supcon_temp":       0.10,
    "s3_proto_temp":        0.10,
    "s3_supcon_weight":     0.02,
    "s3_proto_weight":      0.01,

    # ── Shared training ───────────────────────────────────────────────
    "weight_decay":         2e-4,
    "grad_clip":            1.0,
    "ema_decay":            0.9995,

    # ── TTA ───────────────────────────────────────────────────────────
    "tta_spatial":          8,
    "tta_spectral":         4,

    # ── Evaluation / DataLoader ───────────────────────────────────────
    "eval_batch_size":      256,    # batch size for val and test loaders
    "test_num_workers":     4,
    "test_prefetch_factor": 2,

    # ── Runtime / system ─────────────────────────────────────────────
    "device":               torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":                 42,
    "num_workers":          12,
    "prefetch_factor":      4,
    "mmap_block_size":      64,
    "force_mmap":           False,
    "force_cpu_ram":        False,
}


# ══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT SETUP  (runs once on import)
# ══════════════════════════════════════════════════════════════════════

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.cuda.empty_cache()

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True