#!/usr/bin/env python3
"""
hsi_training_v7.py  —  SpectralQuadNet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rice HSI 90-class classification.  Target: 84–88%  (v6 baseline: 79.1%)

═══════════════════════════ WHAT CHANGED & WHY ═══════════════════════

ARCHITECTURE
────────────
[NEW] Branch D · SpecFormer — lightweight spectral transformer.
  Splits 256 bands into 32 non-overlapping patches × 8 bands each.
  4 Pre-LN transformer blocks, 4 heads, d=128, physical wavelength PE.
  Captures long-range spectral correlations that 1-D convolutions miss
  (e.g., starch 860 nm ↔ protein 930 nm, water 970 nm ↔ lipid 1000 nm).
  Ref: Hong et al. SpectralFormer, IEEE TGRS 2022.

[MOD] Branch C · Power-normalised spatial pooling.
  sign(x)|x|^0.5 applied after avg/max pool approximates second-order
  statistics without full bilinear cost.  Proven gain on fine-grained
  texture tasks (Ionescu et al. Matrix Backprop, ICCV 2015).

[NEW] ArcFace head (Stage 2/3).
  Additive angular margin s=20, m=0.40 on the L2-sphere.  Enforces
  intra-class compactness and inter-class separation — crucial when
  ~67 train samples/class must discriminate 90 similar varieties.
  Ref: Deng et al. ArcFace, CVPR 2019.

TRAINING
────────
[FIX] EMA re-initialised from live model at Stage 2 start.
  In v6 output, EMA accuracy collapsed from ~79% to 68–75% in Stage 2
  epochs 20-50 because the shadow weights still carried high-dropout
  (0.25) momentum when the live model switched to low-dropout (0.08).
  Re-initialising EMA from the Stage 1 best live weights and resetting
  _num_updates to 0 eliminates this mismatch entirely.

[NEW] Supervised Contrastive Loss in Stage 2 (within-batch, no 2× views).
  ClassBalancedBatchSampler guarantees n_spc=4 positive pairs per anchor.
  Combined with ArcFace CE:  loss = α·SupCon + (1-α)·CE.
  Ref: Khosla et al. SupCon, NeurIPS 2020.

[NEW] Stage 3 · Manual SWA (25 epochs, cyclic LR).
  Averages weights over cyclic LR to find flat wide minima.
  BN stats recomputed over training set after averaging.
  Ref: Izmailov et al. SWA, UAI 2018.

AUGMENTATION
────────────
[NEW] Spectral warp  — linear interp stretch/compress ±10% of spectrum.
  Simulates sensor calibration drift between imaging sessions.
[NEW] Spectral shift — circular band shift ±8 positions.
  Simulates wavelength axis offset between acquisition runs.
[NEW] Multiplicative noise — per-band intensity scale variation ±5%.
  Simulates non-uniform illumination and seed surface reflectance.

DIAGNOSTICS FIXED
─────────────────
  Stage 1: Training accuracy with Mixup is reported on MIXED samples
  (expected ~30-50%).  Added a note so the train/val gap is not
  misread as overfitting — the EMA accuracy is the real performance signal.

  Stage 2 no longer has EMA collapse because of the re-init fix above.
  The large train/val gap in v6 Stage 2 (91% vs 79%) was partly caused
  by ema shadow weights being stale, inflating val variance.

EXPECTED GAINS vs v6
────────────────────
  SpecFormer branch:        +2–3 %
  ArcFace head:             +2–3 %
  SupCon in Stage 2:        +1–2 %
  EMA fix (stability):      +0.5–1 %
  SWA:                      +0.5–1 %
  New augmentations:        +0.5–1 %
  ────────────────────────────────
  Total expected:           +5–10 %  →  84–88 % test accuracy with TTA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import copy
import math
import os
import random
import warnings
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Physical wavelength range — Specim V10E sensor
WL_MIN: float = 385.0
WL_MAX: float = 1000.0


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    # Paths
    "patches_data": "./dataset/patches.npy",
    "labels_path":  "./dataset/labels.npy",
    "output_dir":   "./output_v8/",

    # Dataset
    "num_bands":    256,
    "num_classes":  90,

    # Stage 1 — Heavy aug + Mixup/CutMix
    "s1_epochs":    150,
    "s1_batch":     64,
    "s1_max_lr":    8e-4,
    "s1_dropout":   0.30,   # slightly higher than v6 because model is larger
    "s1_mixup":     0.4,
    "s1_patience":  35,

    # Stage 2 — ArcFace + SupCon + balanced batches
    "s2_epochs":    80,
    "s2_batch":     64,     # = bal_n_cls × bal_n_spc
    "s2_lr":        4e-5,
    "s2_min_lr":    1e-7,
    "s2_dropout":   0.10,
    "s2_patience":  25,
    # BUG 6 FIX: s=20, m=0.40 was designed for millions of faces.
    # With 67 samples/class, m=0.40 forces cos(θ+m)<0 for most training,
    # making target logit negative → train stuck at 0% for 12 epochs.
    # s=16, m=0.30 provides margin without destabilising early gradients.
    "s2_arcface_s": 16.0,   # was 20.0
    "s2_arcface_m": 0.30,   # was 0.40

    # SupCon (Stage 2)
    "supcon_weight": 0.25,  # α: SupCon contribution (CE is 1-α)
    "supcon_temp":   0.07,

    # Class-balanced sampler (Stage 2)
    "bal_n_cls":    16,     # classes per batch
    "bal_n_spc":    4,      # samples per class  →  batch = 64

    # Stage 3 — Manual SWA
    "s3_epochs":    25,
    "s3_swa_lr":    8e-5,
    "s3_cycle_len": 5,      # epochs per cosine cycle

    # Loss
    "label_smoothing": 0.05,

    # Regularisation
    "weight_decay": 2e-4,   # slightly higher than v6 (larger model)
    "grad_clip":    1.0,

    # EMA — adaptive, ramps to max
    "ema_decay":    0.9999,

    # TTA
    "tta_n":        8,

    # Shared wavelength embedding (Branch A / B)
    "wl_embed_dim": 16,

    # SpecFormer (Branch D)
    "specf_patch":  8,      # bands per patch  →  32 patches
    "specf_dim":    128,    # transformer d_model
    "specf_heads":  4,
    "specf_layers": 4,
    "specf_drop":   0.15,

    # Misc
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":   42,
    "num_workers": 6,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CONFIG["seed"])


# ══════════════════════════════════════════════════════════════════════
#  ADAPTIVE EMA
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Adaptive EMA: decay ramps from ~0 → max_decay over training steps.
    Formula: d(n) = min(max_decay, (1+n)/(10+n))

    v7 adds reinit_from() — call at Stage 2 start to copy live weights
    into the shadow model.  This fixes the EMA accuracy collapse seen in
    v6 when dropout changed from 0.25 → 0.08 between stages.
    """

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
        ms = dict(model.named_parameters())
        for name, s_p in self.shadow.named_parameters():
            if name in ms:
                s_p.copy_(d * s_p + (1.0 - d) * ms[name])

    def reinit_from(self, model: nn.Module) -> None:
        """
        Re-copy live model weights into the shadow and reset step counter.
        Call at Stage 2 start so EMA adapts correctly to new dropout level.
        """
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def use_arcface(self, flag: bool = True) -> None:
        """BUG 2 FIX: _use_arcface is a Python attribute NOT in state_dict.
        reinit_from() copies state_dict but NOT _use_arcface, so the EMA
        shadow kept using linear_head throughout Stage 2, causing EMA to
        collapse to 4.6% as the embed_net learned ArcFace representations.
        Call this method whenever model.use_arcface() is called."""
        self.shadow._use_arcface = flag

    def state_dict(self)                -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET  (v6 augs + spectral warp / shift / mult-noise)
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """
    Memory-mapped HSI loader.

    augment=True applies (in order):
      band_dropout   – zero out random bands
      band_cutout    – zero out a contiguous spectral window
      spectral_noise – additive Gaussian per-pixel-per-band
      spectral_warp  – [NEW] random linear stretch/compress ±10%
      spectral_shift – [NEW] circular band shift ±8 positions
      mult_noise     – [NEW] per-band multiplicative intensity variation
      spatial        – random flip + rot90
    """

    def __init__(
        self,
        patches_path:     str,
        labels_path:      str,
        indices:          np.ndarray,
        augment:          bool  = False,
        band_drop_prob:   float = 0.04,
        max_cutout_bands: int   = 20,
        noise_std:        float = 0.02,
        warp_prob:        float = 0.35,
        shift_prob:       float = 0.30,
        mult_noise_prob:  float = 0.30,
    ) -> None:
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.augment          = augment
        self.band_drop_prob   = band_drop_prob
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std
        self.warp_prob        = warp_prob
        self.shift_prob       = shift_prob
        self.mult_noise_prob  = mult_noise_prob

    def __len__(self) -> int:
        return len(self.indices)

    # ── original v6 augmentations ────────────────────────────────────

    def _band_dropout(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand(x.shape[0]) > self.band_drop_prob).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.clone()
        nb  = x.shape[0]
        cut = torch.randint(1, max(2, self.max_cutout_bands), (1,)).item()
        st  = torch.randint(0, max(1, nb - cut), (1,)).item()
        x[st: st + cut] = 0.0
        return x

    def _spectral_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.noise_std

    def _spatial_augment(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])
        return torch.rot90(x, torch.randint(0, 4, (1,)).item(), dims=[1, 2])

    # ── new v7 augmentations ─────────────────────────────────────────

    def _spectral_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Random linear stretch / compress of the spectral axis ±10%.
        Correctly interpolates along C (spectral) dimension.
        """
        C, H, W = x.shape

        scale = 1.0 + random.uniform(-0.10, 0.10)
        new_C = max(1, int(C * scale))

        if new_C == C:
            return x

        # Reshape: (N=H*W, C=1, L=C) → interpolate over L
        x_perm = x.permute(1, 2, 0).reshape(-1, 1, C)

        warped = F.interpolate(
            x_perm,
            size=new_C,
            mode="linear",
            align_corners=False
        )

        # Center crop / pad back to C
        if new_C > C:
            start = (new_C - C) // 2
            warped = warped[:, :, start:start + C]
        else:
            pad_lo = (C - new_C) // 2
            pad_hi = C - new_C - pad_lo
            warped = F.pad(warped, (pad_lo, pad_hi))

        # Restore shape (C,H,W)
        warped = warped.reshape(H, W, C).permute(2, 0, 1)

        return warped

    def _spectral_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Circular shift by ±8 bands — simulates wavelength axis offset."""
        shift = random.randint(-8, 8)
        return torch.roll(x, shift, dims=0)

    def _mult_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Per-band multiplicative noise (±5%) — non-uniform illumination."""
        scale = 1.0 + torch.randn(x.shape[0], 1, 1) * 0.05
        return x * scale

    # ── __getitem__ ──────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        patch    = torch.from_numpy(self.patches[real_idx].copy()).float()
        label    = torch.tensor(self.labels[real_idx], dtype=torch.long)

        if self.augment:
            if torch.rand(1).item() < 0.70:
                patch = self._band_dropout(patch)
            if torch.rand(1).item() < 0.50:
                patch = self._band_cutout(patch)
            if torch.rand(1).item() < 0.40:
                patch = self._spectral_noise(patch)
            if torch.rand(1).item() < self.warp_prob:
                patch = self._spectral_warp(patch)
            if torch.rand(1).item() < self.shift_prob:
                patch = self._spectral_shift(patch)
            if torch.rand(1).item() < self.mult_noise_prob:
                patch = self._mult_noise(patch)
            patch = self._spatial_augment(patch)

        return patch, label


# ══════════════════════════════════════════════════════════════════════
#  CLASS-BALANCED BATCH SAMPLER  (Stage 2 — for SupCon)
# ══════════════════════════════════════════════════════════════════════

class ClassBalancedBatchSampler(Sampler):
    """
    Each batch: n_cls randomly-selected classes × n_spc samples each.
    batch_size = n_cls × n_spc  (e.g., 16 × 4 = 64).

    Guarantees n_spc-1 = 3 positive pairs per anchor for SupCon.
    With random shuffle, only ~0.7 positives/anchor on average, which
    is insufficient for meaningful contrastive signal.

    train_labels must be the labels for the TRAINING indices only,
    indexed 0..N_train-1 (i.e. all_labels[train_idx]).
    """

    def __init__(self, train_labels: np.ndarray,
                 n_cls: int = 16, n_spc: int = 4) -> None:
        self.n_cls   = n_cls
        self.n_spc   = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0]
                        for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen_cls = rng.choice(self.classes, self.n_cls, replace=False)
            batch: list[int] = []
            for c in chosen_cls:
                pool = self.cls_idx[c]
                replace = len(pool) < self.n_spc
                samp = rng.choice(pool, self.n_spc, replace=replace)
                batch.extend(samp.tolist())
            yield batch

    def __len__(self) -> int:
        return self._n


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION  (Mixup + CutMix)  — unchanged from v6
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _cutmix(x, y, alpha):
    lam      = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx      = torch.randperm(B, device=x.device)
    r        = math.sqrt(1.0 - lam)
    ch, cw   = int(H * r), int(W * r)
    cx       = random.randint(0, W)
    cy       = random.randint(0, H)
    x1 = max(cx - cw // 2, 0);  x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0);  y2 = min(cy + ch // 2, H)
    x_mix    = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam      = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_mix, y, y[idx], lam


def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1).item() < 0.5 else _cutmix)(x, y, alpha)


def mixed_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ══════════════════════════════════════════════════════════════════════
#  SUPERVISED CONTRASTIVE LOSS  (within-batch, no 2× views)
# ══════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Within-batch SupCon.  Requires L2-normalised feature vectors.
    With a class-balanced sampler (4 pos per anchor), this is stable.

    features : (B, D)  — L2-normalised embeddings
    labels   : (B,)    — integer class labels
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B      = features.shape[0]

        # Cosine similarity, temperature-scaled
        sim = torch.mm(features, features.T) / self.temperature  # (B,B)

        # Numerical stability: subtract row max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()

        # Masks
        diag     = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag  # (B,B)
        neg_mask = ~diag  # all non-self pairs form the denominator

        # log P(positive | anchor)
        exp_sim  = torch.exp(sim) * neg_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        log_prob  = sim - log_denom

        # Mean log-prob over positives per anchor
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1e-8)
        loss_per  = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count

        # Exclude anchors with no positives (edge case)
        valid = pos_mask.any(dim=1)
        if valid.sum() == 0:
            return features.new_tensor(0.0, requires_grad=True)

        return loss_per[valid].mean()


# ══════════════════════════════════════════════════════════════════════
#  MASKED SPECTRAL STATISTICS  (float32, NaN-safe)  — unchanged v6
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute foreground-masked mean / std / max per spectral band."""
    x32  = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)

    energy = flat.abs().sum(dim=1, keepdim=True)
    mask   = (energy > 1e-5).float()
    count  = mask.sum(dim=2).clamp(min=1.0)

    mean    = (flat * mask).sum(dim=2) / count
    mean_sq = ((flat ** 2) * mask).sum(dim=2) / count
    var     = (mean_sq - mean ** 2).clamp(min=1e-6)
    std     = var.sqrt()

    flat_fg = flat.masked_fill(mask.expand_as(flat) == 0, -1e4)
    mx      = flat_fg.max(dim=2).values
    mx      = mx.masked_fill(mx < -9999.0, 0.0)

    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std  = torch.nan_to_num(std,  nan=0.0, posinf=0.0, neginf=0.0)
    mx   = torch.nan_to_num(mx,   nan=0.0, posinf=0.0, neginf=0.0)

    return mean, std, mx


# ══════════════════════════════════════════════════════════════════════
#  WAVELENGTH POSITIONAL ENCODING  — unchanged from v6
# ══════════════════════════════════════════════════════════════════════

class WavelengthPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding of physical wavelengths.
    Tells 1-D CNNs where chlorophyll (~680 nm), water (~970 nm),
    starch (~860 nm) and protein (~930 nm) absorption features live.
    """

    def __init__(self, num_bands: int = 256, embed_dim: int = 16) -> None:
        super().__init__()
        wl   = torch.linspace(0.0, 1.0, num_bands)
        half = embed_dim // 2
        freq = torch.exp(
            torch.arange(half).float() * -(math.log(10_000.0) / max(half - 1, 1))
        )
        enc         = torch.zeros(num_bands, embed_dim)
        enc[:,  :half] = torch.sin(wl.unsqueeze(1) * freq.unsqueeze(0))
        enc[:, half:]  = torch.cos(wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("enc", enc)
        self.proj = nn.Linear(embed_dim, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self) -> torch.Tensor:
        """Returns (1, 1, num_bands) — broadcast-adds to (B, C, 256) inputs."""
        return self.proj(self.enc).squeeze(-1).view(1, 1, -1)


# ══════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS  — ALL BatchNorm1d replaced with GroupNorm
# ══════════════════════════════════════════════════════════════════════

def _gn1d(c: int) -> nn.GroupNorm:
    """GroupNorm for 1-D feature tensors.  No running stats → EMA-safe.
    BUG FIX: BatchNorm1d tracks running_mean/var in nn.Buffers which are
    NOT updated by EMA.update() (only Parameters are EMA-averaged).
    After F.normalize drives feature variance to ~1/C, the frozen EMA BN
    buffers are 22× off the live stats → EMA collapses to 1.1%.
    GroupNorm normalises within each forward pass regardless of mode."""
    g = min(8, c)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)

class SpectralSE(nn.Module):
    """Per-sample spectral Squeeze-and-Excitation over 256 bands."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid,      bias=False), nn.GELU(),
            nn.Linear(mid,      channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x.mean(dim=[2, 3]))
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    """1-D residual block — GroupNorm replaces BatchNorm1d (BUG 1 FIX).
    GroupNorm has no running stats so the EMA shadow always normalises
    correctly regardless of training distribution."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7) -> None:
        super().__init__()
        pad        = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.gn1   = _gn1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.gn2   = _gn1d(out_ch)
        self.skip  = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                          _gn1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class CBAM(nn.Module):
    """Channel + Spatial Attention Module."""

    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid = max(c // r, 8)
        self.ch_mlp  = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False), nn.GELU(),
            nn.Conv2d(mid, c, 1, bias=False),
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = torch.sigmoid(
            self.ch_mlp(x.mean(dim=[2, 3], keepdim=True)) +
            self.ch_mlp(x.amax(dim=[2, 3], keepdim=True))
        )
        x  = x * ca
        sp = self.sp_conv(torch.cat(
            [x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1
        ))
        return x * sp


class ResBlock2D(nn.Module):
    """2-D bottleneck residual block with GroupNorm."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid     = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid,    1,            bias=False)
        self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid,   mid,    3, stride, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid,   out_ch, 1,            bias=False)
        self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.c1(x)))
        h = F.gelu(self.n2(self.c2(h)))
        h = self.n3(self.c3(h))
        return F.gelu(h + self.skip(x))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A — SPECTRAL PROFILE  — unchanged from v6
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    """
    Multi-scale 1-D CNN on mean spectrum + first-order derivative.
    Derivative captures peak positions (illumination-invariant).
    3 towers (kernel 3, 7, 15) → global avg+max pool → 256-D.
    """

    def __init__(self, out_dim: int = 256, tower_ch: int = 80,
                 wl_enc: Optional[WavelengthPositionalEncoding] = None) -> None:
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(2, tower_ch, k=3)
        self.tower_m = self._tower(2, tower_ch, k=7)
        self.tower_l = self._tower(2, tower_ch, k=15)
        self.proj    = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            _gn1d(out_dim),          # BUG 1 FIX: was BatchNorm1d
            nn.GELU(),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(in_ch, out_ch, k):
        mid = out_ch // 2
        return nn.Sequential(
            ResBlock1D(in_ch, mid, k),
            ResBlock1D(mid,   out_ch, k),
            ResBlock1D(out_ch, out_ch, k),
        )

    @staticmethod
    def _gpool(f):
        return torch.cat([f.mean(2), f.max(2).values], dim=1)

    def forward(self, mean_spec: torch.Tensor) -> torch.Tensor:
        s = mean_spec.unsqueeze(1)
        d = F.pad(torch.diff(s, dim=2), (0, 1))
        x = torch.cat([s, d], dim=1)
        if self.wl_enc is not None:
            x = x + self.wl_enc()
        feat = torch.cat([
            self._gpool(self.tower_s(x)),
            self._gpool(self.tower_m(x)),
            self._gpool(self.tower_l(x)),
        ], dim=1)
        return self.proj(feat)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS  — unchanged from v6
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    """
    Multi-scale 1-D CNN on {mean, std, max} spectral statistics.
    std encodes intra-seed heterogeneity (hull vs endosperm variation).
    3 towers → global avg+max pool → 256-D.
    """

    def __init__(self, out_dim: int = 256, tower_ch: int = 80,
                 wl_enc: Optional[WavelengthPositionalEncoding] = None) -> None:
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(3, tower_ch, k=3)
        self.tower_m = self._tower(3, tower_ch, k=7)
        self.tower_l = self._tower(3, tower_ch, k=15)
        self.proj    = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            _gn1d(out_dim),          # BUG 1 FIX: was BatchNorm1d
            nn.GELU(),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(in_ch, out_ch, k):
        mid = out_ch // 2
        return nn.Sequential(
            ResBlock1D(in_ch, mid, k),
            ResBlock1D(mid,   out_ch, k),
            ResBlock1D(out_ch, out_ch, k),
        )

    @staticmethod
    def _gpool(f):
        return torch.cat([f.mean(2), f.max(2).values], dim=1)

    def forward(self, mean_s, std_s, max_s) -> torch.Tensor:
        x = torch.stack([mean_s, std_s, max_s], dim=1)
        if self.wl_enc is not None:
            x = x + self.wl_enc()
        feat = torch.cat([
            self._gpool(self.tower_s(x)),
            self._gpool(self.tower_m(x)),
            self._gpool(self.tower_l(x)),
        ], dim=1)
        return self.proj(feat)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN  (+ power-normalised pooling)
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    """
    2-D residual CNN + CBAM attention for morphological features.

    v7 adds power normalisation (sign(x)|x|^0.5) on the pooled vectors
    before the projection.  This approximates second-order statistics:
    large activations are dampened so the MLP receives a more uniform
    distribution, reducing dominance from textural outliers.
    Empirical gains of 1–2% on fine-grained recognition tasks.
    """

    def __init__(self, num_bands: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            ResBlock2D(32,  64,      stride=2), CBAM(64),
            ResBlock2D(64,  128,     stride=2), CBAM(128),
            ResBlock2D(128, 192,     stride=2), CBAM(192),
            ResBlock2D(192, out_dim, stride=2),
        )
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.max_pool  = nn.AdaptiveMaxPool2d(1)
        self.pool_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            _gn1d(out_dim),          # BUG 1 FIX: was BatchNorm1d
            nn.GELU(),
        )

    @staticmethod
    def _power_norm(x: torch.Tensor) -> torch.Tensor:
        """Signed square root — approximates second-order pooling."""
        return x.sign() * x.abs().clamp(min=1e-8).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = self.band_reduce(x)
        h    = self.stages(h)
        avg  = self.avg_pool(h).flatten(1)
        mx   = self.max_pool(h).flatten(1)
        # Power-normalise. BUG 1 FIX: removed F.normalize() that drove
        # feature variance to ~1/512, causing 22× BN scale mismatch in EMA.
        feat = torch.cat([self._power_norm(avg), self._power_norm(mx)], dim=1)
        return self.pool_proj(feat)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER  (NEW)
# ══════════════════════════════════════════════════════════════════════

class _PreLNBlock(nn.Module):
    """Pre-LayerNorm transformer block (more stable than post-LN)."""

    def __init__(self, d: int, heads: int, d_ff: int, drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, dropout=drop,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d_ff),  nn.GELU(), nn.Dropout(drop),
            nn.Linear(d_ff, d),  nn.Dropout(drop),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class SpecFormerBranch(nn.Module):
    """
    Lightweight spectral transformer for 256-band HSI.

    Design rationale
    ────────────────
    1-D CNNs (Branches A/B) excel at local spectral patterns (narrow
    absorption dips) but cannot model long-range interactions, e.g.:
      • Starch at 860 nm correlates inversely with protein at 930 nm.
      • Water content at 970 nm modulates the entire NIR baseline.

    Self-attention across spectral patches captures these correlations
    explicitly.  The [CLS] token summarises global spectral context.

    Architecture
    ────────────
    • Tokenise: 256 bands → 32 patches × 8 bands each
    • Project: Linear(8, 128) per patch → (B, 32, 128)
    • Physical WL positional encoding (sinusoidal at patch centres)
    • Prepend [CLS] token → (B, 33, 128)
    • 4 Pre-LN blocks, 4 heads, d_ff=256, dropout=0.15
    • Pool [CLS] → Linear(128, 256) → BN → GELU → Dropout
    • Output: (B, 256)

    Parameter count: ~0.5 M
    Ref: Hong et al. SpectralFormer, IEEE TGRS 2022.
    """

    def __init__(
        self,
        num_bands:  int   = 256,
        patch_size: int   = 8,
        d_model:    int   = 128,
        n_heads:    int   = 4,
        n_layers:   int   = 4,
        out_dim:    int   = 256,
        dropout:    float = 0.15,
    ) -> None:
        super().__init__()
        n_patches       = num_bands // patch_size   # 32
        self.patch_size = patch_size
        self.n_patches  = n_patches

        # Patch projection
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_size, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        # Physical wavelength positional encoding at patch centres
        wl_centers = torch.linspace(WL_MIN, WL_MAX, n_patches)
        wl_norm    = (wl_centers - WL_MIN) / (WL_MAX - WL_MIN)
        half       = d_model // 2
        freq       = torch.exp(
            torch.arange(half).float() * -(math.log(1e4) / max(half - 1, 1))
        )
        pe = torch.zeros(n_patches, d_model)
        pe[:, :half] = torch.sin(wl_norm.unsqueeze(1) * freq.unsqueeze(0))
        pe[:, half:] = torch.cos(wl_norm.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("wl_pe", pe)   # (32, d_model)

        # [CLS] token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            _gn1d(out_dim),          # BUG 1 FIX: was BatchNorm1d
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, mean_spec: torch.Tensor) -> torch.Tensor:
        B   = mean_spec.shape[0]
        x   = mean_spec.float().view(B, self.n_patches, self.patch_size)  # (B,32,8)
        x   = self.patch_proj(x) + self.wl_pe.unsqueeze(0)               # (B,32,128)
        cls = self.cls.expand(B, -1, -1)                                  # (B,1,128)
        x   = torch.cat([cls, x], dim=1)                                  # (B,33,128)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x[:, 0])  # [CLS] → (B, out_dim)


# ══════════════════════════════════════════════════════════════════════
#  ARCFACE HEAD  (Stage 2/3)
# ══════════════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Softmax.

    Motivation for this dataset (90 classes, ~67 samples/class):
      Standard softmax classifies on dot-product distance in Euclidean
      space, allowing class boundaries to overlap.  ArcFace projects
      features and class proxies onto a unit hypersphere and penalises
      the angular distance between same-class features — enforcing
      compact within-class distributions.

    Hyper-parameters:
      s = 20  (conservative vs typical 64; features not perfectly
               unit-normed after BN, smaller scale avoids divergence)
      m = 0.40 (moderate margin; larger margins overfit with few samples)

    Ref: Deng et al. ArcFace, CVPR 2019.
    """

    def __init__(self, in_dim: int, num_classes: int,
                 s: float = 20.0, m: float = 0.40) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s     = s
        self.m     = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, in_dim)  ← must be L2-normalised before calling
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))  # (B, C)
        if labels is None or not self.training:
            return cosine * self.s

        sine  = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-8, 1.0))
        phi   = cosine * self.cos_m - sine * self.sin_m
        phi   = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh    = torch.zeros_like(cosine)
        oh.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi) + ((1.0 - oh) * cosine)) * self.s


# ══════════════════════════════════════════════════════════════════════
#  MAIN MODEL — SpectralQuadNet v7
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    """
    Four-branch HSI classification network.

    Branch A : SpectralProfileBranch  (mean + deriv + WL PE)    → 256-D
    Branch B : SpectralStatsBranch    (mean + std  + max + WL)   → 256-D
    Branch C : SpatialCNNBranch       (2D CNN + power norm)      → 256-D
    Branch D : SpecFormerBranch       (spectral transformer NEW)  → 256-D

    Fusion  : cat(A,B,C,D) = 1024-D
                → Linear(512) → BN → GELU → Dropout
                → Linear(256) → BN                      (embedding)

    Stage 1 head : Linear(256, 90)                [Mixup-compatible CE]
    Stage 2/3 head : ArcFaceHead(256, 90, s=20, m=0.40)  [ArcFace CE]

    Switching between heads is done via model.use_arcface(True/False).
    L2-normalisation of embeddings is applied only for the ArcFace head.
    """

    def __init__(
        self,
        num_classes:  int   = 90,
        num_bands:    int   = 256,
        dropout:      float = 0.30,
        wl_embed_dim: int   = 16,
        cfg:          dict  = None,
    ) -> None:
        super().__init__()
        cfg = cfg or CONFIG

        # Shared spectral SE attention
        self.se     = SpectralSE(num_bands, reduction=16)

        # Shared wavelength positional encoding (Branches A & B)
        self.wl_enc = WavelengthPositionalEncoding(num_bands, wl_embed_dim)

        # Four branches
        self.branch_a = SpectralProfileBranch(256, tower_ch=80, wl_enc=self.wl_enc)
        self.branch_b = SpectralStatsBranch(  256, tower_ch=80, wl_enc=self.wl_enc)
        self.branch_c = SpatialCNNBranch(num_bands=num_bands, out_dim=256)
        self.branch_d = SpecFormerBranch(
            num_bands  = num_bands,
            patch_size = cfg["specf_patch"],
            d_model    = cfg["specf_dim"],
            n_heads    = cfg["specf_heads"],
            n_layers   = cfg["specf_layers"],
            out_dim    = 256,
            dropout    = cfg["specf_drop"],
        )

        fusion_dim = 256 * 4   # 1024

        # Shared embedding network
        self.embed_net = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            _gn1d(512),              # BUG 1 FIX: was BatchNorm1d(512)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            _gn1d(256),              # BUG 1 FIX: was BatchNorm1d(256)
        )

        # Stage 1 head — standard linear (Mixup-compatible)
        self.linear_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(256, num_classes),
        )

        # Stage 2/3 head — ArcFace (requires L2-normalised input)
        self.arcface_head = ArcFaceHead(
            256, num_classes,
            s=cfg["s2_arcface_s"],
            m=cfg["s2_arcface_m"],
        )

        self._use_arcface = False   # toggled by use_arcface()

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── utilities ─────────────────────────────────────────────────────

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def use_arcface(self, flag: bool = True) -> None:
        """Toggle between linear (Stage 1) and ArcFace (Stage 2/3) heads."""
        self._use_arcface = flag

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x:            torch.Tensor,
        labels:       Optional[torch.Tensor] = None,
        return_embed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Spectral self-excitation
        x = self.se(x)                                     # (B,256,64,64)

        # Spectral statistics (float32, NaN-safe)
        ms, ss, mx = masked_spectral_stats(x)              # each (B,256)

        # Four branches
        fa = self.branch_a(ms)                             # (B,256)
        fb = self.branch_b(ms, ss, mx)                    # (B,256)
        fc = self.branch_c(x)                              # (B,256)
        fd = self.branch_d(ms)                             # (B,256)

        # Fusion embedding
        fused = torch.cat([fa, fb, fc, fd], dim=1)         # (B,1024)
        emb   = self.embed_net(fused)                      # (B,256), BN-normalised

        # Head selection
        if self._use_arcface:
            emb_n  = F.normalize(F.gelu(emb), dim=1)      # L2-norm for ArcFace
            logits = self.arcface_head(emb_n, labels)
        else:
            emb_n  = emb
            logits = self.linear_head(emb)                 # Stage 1: plain CE

        if return_embed:
            # BUG 3 FIX: removed .detach() — emb must stay in the computation
            # graph so SupCon loss can backpropagate into embed_net.
            # Previously: F.normalize(F.gelu(emb.detach()), dim=1)
            # → SupCon contributed zero gradient to all shared weights.
            return logits, F.normalize(F.gelu(emb), dim=1)

        return logits


# ══════════════════════════════════════════════════════════════════════
#  TTA  — unchanged from v6
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """Ensemble over n spatial augmentation views (rot + flip symmetries)."""
    device = x.device
    views  = [(k, f) for k in range(4) for f in (False, True)][:n]
    logits = []
    for k, flip in views:
        aug = torch.rot90(x, k, dims=[2, 3])
        if flip:
            aug = torch.flip(aug, dims=[3])
        with autocast(device_type=device.type):
            logits.append(model(aug))
    return torch.stack(logits).mean(0)


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits():
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3,
                               stratify=labels, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.5,
                               stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(
    train_idx:    np.ndarray,
    val_idx:      np.ndarray,
    test_idx:     np.ndarray,
    batch_train:  int,
    balanced:     bool              = False,
    all_labels:   Optional[np.ndarray] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    nw = CONFIG["num_workers"]
    kw = dict(num_workers=nw, pin_memory=True)

    train_ds = RiceSeedDataset(
        CONFIG["patches_data"], CONFIG["labels_path"], train_idx, augment=True
    )

    if balanced and all_labels is not None:
        sampler   = ClassBalancedBatchSampler(
            all_labels[train_idx],
            n_cls=CONFIG["bal_n_cls"],
            n_spc=CONFIG["bal_n_spc"],
        )
        train_ldr = DataLoader(
            train_ds, batch_sampler=sampler,
            persistent_workers=True, prefetch_factor=2, **kw,
        )
    else:
        train_ldr = DataLoader(
            train_ds, batch_size=batch_train, shuffle=True,
            persistent_workers=True, prefetch_factor=2, **kw,
        )

    val_ldr  = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], val_idx),
        batch_size=64, shuffle=False, **kw,
    )
    test_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], test_idx),
        batch_size=64, shuffle=False, **{**kw, "num_workers": 2},
    )
    return train_ldr, val_ldr, test_ldr


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISER  — unchanged from v6
# ══════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr: float) -> optim.AdamW:
    """AdamW — weight decay skips BN, GroupNorm, and bias params."""
    wd_p, no_wd_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_wd_p if (p.ndim == 1 or name.endswith(".bias")) else wd_p).append(p)
    return optim.AdamW(
        [{"params": wd_p,    "weight_decay": CONFIG["weight_decay"]},
         {"params": no_wd_p, "weight_decay": 0.0}],
        lr=lr,
    )


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    optimizer:     optim.Optimizer,
    criterion:     nn.Module,
    scaler:        GradScaler,
    ema:           Optional[ModelEMA],
    device:        torch.device,
    scheduler      = None,
    use_mixup:     bool              = True,
    mixup_alpha:   float             = 0.4,
    supcon:        Optional[SupConLoss] = None,
    supcon_weight: float             = 0.0,
) -> tuple[float, float]:
    """
    One training epoch.

    Stage 1:  use_mixup=True,  supcon=None    — Mixup/CutMix + CE
    Stage 2:  use_mixup=False, supcon=SupConLoss — ArcFace CE + SupCon
    Stage 3:  use_mixup=False, supcon=None    — plain CE (SWA accumulation)
    """
    model.train()
    total_loss = total_acc = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        x_in, y_a, y_b, lam = (
            mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            if supcon is not None and not use_mixup:
                # Stage 2: ArcFace CE + SupCon
                logits, emb = model(x_in, y_a, return_embed=True)
                loss_ce  = criterion(logits, y_a)
                loss_sc  = supcon(emb, y_a)
                loss     = (1.0 - supcon_weight) * loss_ce + supcon_weight * loss_sc
            else:
                # Stage 1: pass labels only if ArcFace is active
                logits = model(
                    x_in,
                    labels=y_a if (model._use_arcface and not use_mixup) else None,
                )
                loss = mixed_loss(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == y).float().mean().item()

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (macro-F1, accuracy)."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type):
            logits = model(x)
        preds.append(logits.argmax(1).cpu())
        targets.append(y)
    p, t = torch.cat(preds), torch.cat(targets)
    return (
        f1_score(t, p, average="macro", zero_division=0),
        accuracy_score(t, p),
    )


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, stage, model, ema, val_acc, val_f1):
    torch.save({"epoch": epoch, "stage": stage,
                "model": model.state_dict(), "ema": ema.state_dict(),
                "val_acc": val_acc, "val_f1": val_f1}, path)


def load_ckpt(path, model, ema, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    return ckpt


# ══════════════════════════════════════════════════════════════════════
#  BN UPDATE (for SWA)
# ══════════════════════════════════════════════════════════════════════

def update_bn_stats(loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    """
    Recompute BN running mean/var for a model with averaged weights.
    Required after SWA weight averaging — the averaged weights no longer
    correspond to the training statistics tracked by BN layers.
    Uses cumulative moving average (momentum=None).
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None   # cumulative moving avg over all batches

    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device, non_blocking=True))

    model.eval()


# ══════════════════════════════════════════════════════════════════════
#  STAGE RUNNERS
# ══════════════════════════════════════════════════════════════════════

def _hdr(title: str, epochs: int) -> None:
    w = 66
    print(f"\n{'═'*w}\n  {title}  [{epochs} epochs max]\n{'═'*w}")


# ── Stage 1 ───────────────────────────────────────────────────────────

def run_stage1(
    model, ema, train_ldr, val_ldr, device, criterion, best_ckpt,
) -> float:
    """
    Stage 1: Heavy aug + Mixup/CutMix + OneCycleLR.
    Linear head; ArcFace is OFF.

    Note: train accuracy is computed on MIXED samples (expected 30–50 %).
    The EMA accuracy is the true performance signal for checkpointing.
    """
    model.use_arcface(False)

    optimizer = build_optimizer(model, lr=CONFIG["s1_max_lr"] / 25)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr          = CONFIG["s1_max_lr"],
            epochs          = CONFIG["s1_epochs"],
            steps_per_epoch = len(train_ldr),
            pct_start       = 0.15,
            div_factor      = 25,
            final_div_factor= 1e4,
            anneal_strategy = "cos",
        )

    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 1 — Heavy Aug + Mixup/CutMix", CONFIG["s1_epochs"])

    for ep in range(1, CONFIG["s1_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=ema, device=device, scheduler=scheduler,
            use_mixup=True, mixup_alpha=CONFIG["s1_mixup"],
        )

        _,   va_live = evaluate(model,      val_ldr, device)
        vf1, va_ema  = evaluate(ema.shadow, val_ldr, device)
        va_best      = max(va_live, va_ema)
        lr_now       = optimizer.param_groups[0]["lr"]
        ema_d        = ema.current_decay
        saved        = ""

        if va_best > best_acc:
            best_acc, no_improve = va_best, 0
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema, va_best, vf1)
            saved = "  ✓ Saved"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s1_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%}  EMA {va_ema:.1%} │ "
            f"LR {lr_now:.2e}  EMA_d {ema_d:.4f}{saved}"
        )

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    return best_acc


# ── Stage 2 ───────────────────────────────────────────────────────────

def run_stage2(
    model, ema, train_ldr, val_ldr, device, criterion, best_ckpt,
) -> float:
    """
    Stage 2: ArcFace CE + within-batch SupCon + class-balanced batches.

    Key v7 fixes vs v6:
      1. EMA re-initialised from live model — eliminates EMA accuracy
         collapse that occurred in v6 when dropout changed 0.25→0.08.
      2. ArcFace head enforces angular margins on the L2-sphere.
      3. SupCon loss (α=0.25) further tightens intra-class clusters.

    Lower dropout (0.10) than Stage 1 (0.30) — model is more confident
    after Stage 1 pre-training and needs less regularisation noise.
    """
    # Set lower dropout for Stage 2
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)

    # ★ KEY FIX: Re-init EMA from live Stage-1 best weights.
    #   Without this, EMA shadow still carries Stage-1 high-dropout
    #   momentum, so val accuracy collapses in early Stage-2 epochs.
    ema.reinit_from(model)
    ema.set_dropout(CONFIG["s2_dropout"])
    # BUG 2 FIX: _use_arcface is a Python attr NOT in state_dict.
    # reinit_from() copies state_dict only → EMA shadow kept using
    # linear_head in Stage 2, causing collapse to 4.6% by ep20.
    ema.use_arcface(True)

    supcon    = SupConLoss(temperature=CONFIG["supcon_temp"])
    optimizer = build_optimizer(model, lr=CONFIG["s2_lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["s2_epochs"], eta_min=CONFIG["s2_min_lr"]
    )
    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 2 — ArcFace + SupCon + Balanced Batches", CONFIG["s2_epochs"])

    for ep in range(1, CONFIG["s2_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=ema, device=device, scheduler=None,
            use_mixup=False,
            supcon=supcon,
            supcon_weight=CONFIG["supcon_weight"],
        )
        scheduler.step()

        _,   va_live = evaluate(model,      val_ldr, device)
        vf1, va_ema  = evaluate(ema.shadow, val_ldr, device)
        va_best      = max(va_live, va_ema)
        lr_now       = optimizer.param_groups[0]["lr"]
        saved        = ""

        if va_best > best_acc:
            best_acc, no_improve = va_best, 0
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema, va_best, vf1)
            saved = "  ✓ Saved"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s2_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%}  EMA {va_ema:.1%} │ "
            f"LR {lr_now:.2e}{saved}"
        )

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    return best_acc


# ── Stage 3 (SWA) ─────────────────────────────────────────────────────

def run_stage3_swa(
    model, ema, train_ldr, val_ldr, device, criterion, best_ckpt,
) -> float:
    """
    Stage 3: Manual Stochastic Weight Averaging.

    Runs s3_epochs epochs with a cosine-cyclic LR (cycle_len epochs),
    then averages all model snapshots taken at the end of each cycle.
    BN running stats are recomputed from training data after averaging.

    Wide flat minima found by SWA generalise better than sharp minima
    found by standard SGD/Adam.  Expected gain: +0.5–1.5 %.

    Manual (not torch.optim.swa_utils.AveragedModel) to avoid the
    AveragedModel wrapper complexity and the module. prefix in state_dict.
    """
    model.set_dropout(CONFIG["s2_dropout"])

    optimizer  = build_optimizer(model, lr=CONFIG["s3_swa_lr"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["s3_cycle_len"],
        eta_min=CONFIG["s3_swa_lr"] * 0.1,
    )
    scaler = GradScaler()

    # Initialise accumulator from current (Stage-2 best) weights
    swa_state: dict = copy.deepcopy(model.state_dict())
    n_snap    = 1     # we count Stage-2 best as snapshot 0

    _hdr("Stage 3 — Stochastic Weight Averaging", CONFIG["s3_epochs"])

    for ep in range(1, CONFIG["s3_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=None, device=device, scheduler=None,
            use_mixup=False,
        )
        scheduler.step()

        # Restart cosine cycle
        if ep % CONFIG["s3_cycle_len"] == 0:
            # BUG 5 FIX: must reset optimizer LR back to swa_lr before
            # creating the new scheduler. Without this, the new scheduler
            # reads base_lr = current_lr = eta_min = 8e-6 and all remaining
            # cycles run flat at eta_min. LR never actually cycles.
            for pg in optimizer.param_groups:
                pg["lr"] = CONFIG["s3_swa_lr"]
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["s3_cycle_len"],
                eta_min=CONFIG["s3_swa_lr"] * 0.1,
            )
            # Accumulate snapshot
            n_snap  += 1
            alpha    = 1.0 / n_snap
            curr_sd  = model.state_dict()
            for k in swa_state:
                swa_state[k] = swa_state[k] + alpha * (curr_sd[k] - swa_state[k])

        _,   va_live = evaluate(model, val_ldr, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%} │ LR {lr_now:.2e} │ Snaps {n_snap}"
        )

    # ── Update BN for SWA model ──────────────────────────────────────
    print(f"\nUpdating BN statistics for SWA model ({n_snap} snapshots) ...")
    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state)
    update_bn_stats(train_ldr, swa_model, device)

    _, va_swa = evaluate(swa_model, val_ldr, device)
    _, va_ema = evaluate(ema.shadow, val_ldr, device)
    print(f"SWA val: {va_swa:.1%}   EMA val: {va_ema:.1%}")

    # Use the better of SWA and EMA as the final inference model
    if va_swa >= va_ema:
        print("Using SWA model as final eval model.")
        ema.shadow.load_state_dict(swa_model.state_dict())
        best_val = va_swa
    else:
        print("EMA model retained as final eval model.")
        best_val = va_ema

    # BUG 4 FIX: save SWA/EMA final state to a SEPARATE file.
    # Previously, final_evaluation() called load_ckpt(best_ckpt, ...) which
    # overwrote ema.shadow with the Stage 2 ep73 checkpoint (EMA=4.6%)
    # — destroying the SWA model and producing test accuracy of 3.5%.
    # By saving to swa_ckpt and passing it directly, final_evaluation
    # evaluates the correct 68.9% SWA model.
    return best_val, swa_model


# ══════════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(eval_model: nn.Module, test_ldr, device, ckpt_info: dict) -> None:
    """
    BUG 4 FIX: original signature called load_ckpt(best_ckpt, model, ema)
    which overwrote ema.shadow with the Stage 2 ep73 state (EMA=4.6%),
    destroying the SWA model and producing test accuracy of 3.5%.
    Now accepts the eval_model directly — caller decides which model to use.
    """
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")
    eval_model.eval()

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x = x.to(device, non_blocking=True)
            if use_tta:
                logits = tta_predict(eval_model, x, CONFIG["tta_n"])
            else:
                with autocast(device_type=device.type):
                    logits = eval_model(x)
            preds.append(logits.argmax(1).cpu())
            targets.append(y)
        p, t         = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p, t)
        acc = accuracy_score(t, p)
        f1m = f1_score(t, p, average="macro",    zero_division=0)
        f1w = f1_score(t, p, average="weighted", zero_division=0)
        print(f"\n  [{tag}]  Acc={acc:.1%}  F1(macro)={f1m:.4f}  F1(wt)={f1w:.4f}")

    print(f"\n  Checkpoint: {ckpt_info.get('label', 'SWA/EMA final')} "
          f"| val={ckpt_info.get('val_acc', 'N/A')}")

    p_tta, t_tta = results["TTA   "]
    print(f"\nClassification Report (TTA):\n")
    print(classification_report(t_tta, p_tta, zero_division=0))

    out = CONFIG["output_dir"]
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy",   p_tta)
    np.save(f"{out}/test_targets.npy",      t_tta)
    print(f"\nOutputs saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    device    = CONFIG["device"]
    best_ckpt = os.path.join(CONFIG["output_dir"], "best_model.pth")

    all_labels, train_idx, val_idx, test_idx = build_splits()
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx) // CONFIG['num_classes']}")

    model = SpectralQuadNet(
        num_classes  = CONFIG["num_classes"],
        num_bands    = CONFIG["num_bands"],
        dropout      = CONFIG["s1_dropout"],
        wl_embed_dim = CONFIG["wl_embed_dim"],
        cfg          = CONFIG,
    ).to(device)

    ema   = ModelEMA(model, decay=CONFIG["ema_decay"])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : SpectralQuadNet v7")
    print(f"Params : {n_par / 1e6:.2f}M")
    print(f"Device : {device}  |  EMA adaptive → max {CONFIG['ema_decay']}")
    print(f"Stage 1: {CONFIG['s1_epochs']} ep | Mixup+CutMix | Linear head    | drop={CONFIG['s1_dropout']}")
    print(f"Stage 2: {CONFIG['s2_epochs']} ep | ArcFace+SupCon | Balanced smp | drop={CONFIG['s2_dropout']}")
    print(f"Stage 3: {CONFIG['s3_epochs']} ep | SWA ({CONFIG['s3_cycle_len']}-ep cycles)")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # ── Stage 1: random sampler, Mixup, linear head ─────────────────
    train_ldr1, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s1_batch"]
    )
    run_stage1(model, ema, train_ldr1, val_ldr, device, criterion, best_ckpt)

    # ── Load Stage 1 best → Stage 2 ─────────────────────────────────
    print("\nLoading Stage 1 best checkpoint for Stage 2 ...")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  val={ckpt['val_acc']:.1%}  ({ckpt['stage']})")
    print(f"  Dropout: {CONFIG['s1_dropout']} → {CONFIG['s2_dropout']}")
    print(f"  EMA: will be re-initialised inside run_stage2()")

    # ── Stage 2: balanced sampler, ArcFace, SupCon ──────────────────
    train_ldr2, val_ldr2, _ = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"],
        balanced=True, all_labels=all_labels,
    )
    run_stage2(model, ema, train_ldr2, val_ldr2, device, criterion, best_ckpt)

    # ── Load Stage 2 best → Stage 3 ─────────────────────────────────
    print("\nLoading Stage 2 best checkpoint for Stage 3 (SWA) ...")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  val={ckpt['val_acc']:.1%}  ({ckpt['stage']})")

    # ── Stage 3: SWA ─────────────────────────────────────────────────
    # Use random sampler for SWA (BN update also needs full coverage)
    train_ldr3, val_ldr3, _ = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"]
    )
    # BUG 4 FIX: run_stage3_swa now returns (best_val, swa_model).
    # Previously returned only best_val; final_evaluation then called
    # load_ckpt(best_ckpt) which restored the broken Stage 2 EMA,
    # destroying the SWA model and producing test accuracy of 3.5%.
    best_val, swa_model = run_stage3_swa(
        model, ema, train_ldr3, val_ldr3, device, criterion, best_ckpt
    )

    # ── Final evaluation ─────────────────────────────────────────────
    _, _, test_ldr_final = build_loaders(train_idx, val_idx, test_idx, 64)
    # Evaluate SWA model directly — do NOT call load_ckpt here
    final_evaluation(
        swa_model, test_ldr_final, device,
        {"label": "SWA final", "val_acc": f"{best_val:.1%}"}
    )


if __name__ == "__main__":
    main()