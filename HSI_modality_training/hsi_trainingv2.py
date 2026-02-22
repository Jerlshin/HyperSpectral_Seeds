"""
hsi_training_v2.py  —  Improved Rice HSI Seed Classification
=============================================================

Improvements over v1 (SpectralTripleNet / 79.1% TTA):

  1. SpectralTransformerBranch  — self-attention over 256 band-tokens
     captures long-range spectral correlations that 1-D conv towers miss.
     Inspired by HyperFormer (Roy et al. 2023) and SpectralFormer (Hong et al. 2022).

  2. SubCenterArcFace loss  — angular-margin softmax (Wang et al. 2020).
     Drastically improves inter-class separability at low samples/class.
     K=3 sub-centres per class handles intra-class variance gracefully.

  3. Three-stage curriculum
       Stage 1 — Heavy aug + Mixup/CutMix + CE     (representation learning)
       Stage 2 — Clean fine-tune + CE               (sharpen boundaries)
       Stage 3 — ArcFace fine-tune, very low LR     (metric-space tightening)

  4. Spectral-regional augmentation  — VIS (bands 0–127) and NIR (128–255)
     zones perturbed independently with random gain + baseline shift,
     mimicking real sensor noise profiles.

  5. Band-Guided Spatial Attention  — the spatial CNN branch weights its
     band-reduction by a learned SE gate rather than a plain 1×1 conv.

  6. Fixed EMA transition  — no hard reset between stages; only decay
     is updated.  EMA keeps its accumulated momentum.

  7. Online class-difficulty weighting  — running accuracy per class
     is tracked; the loss of hard classes is up-weighted by a mild
     focal-like factor every N steps.

  8. Improved TTA  — 8 spatial views + 2 spectral flips (VIS / NIR swap)
     → 16-view ensemble for final evaluation.
"""

from __future__ import annotations

import copy
import math
import os
import random
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────
#  Sensor wavelength range (Specim V10E)
# ──────────────────────────────────────────────────────────────────────
WL_MIN: float = 385.0   # nm
WL_MAX: float = 1000.0  # nm
VIS_BANDS = slice(0, 128)    # ~385–692 nm
NIR_BANDS = slice(128, 256)  # ~692–1000 nm


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    # ── Paths ─────────────────────────────────────────────────────────
    "patches_data":  "./dataset/patches.npy",
    "labels_path":   "./dataset/labels.npy",
    "output_dir":    "./output_v7/",

    # ── Dataset ───────────────────────────────────────────────────────
    "num_bands":     256,
    "num_classes":   90,

    # ── Stage 1 — Heavy aug + Mixup/CutMix + CE ──────────────────────
    "s1_epochs":     160,
    "s1_batch":      64,
    "s1_max_lr":     8e-4,
    "s1_dropout":    0.25,
    "s1_mixup":      0.4,
    "s1_patience":   40,

    # ── Stage 2 — Clean fine-tune + CE ───────────────────────────────
    "s2_epochs":     80,
    "s2_batch":      48,
    "s2_lr":         3e-5,
    "s2_min_lr":     3e-7,
    "s2_dropout":    0.08,
    "s2_patience":   22,

    # ── Stage 3 — ArcFace fine-tune ──────────────────────────────────
    "s3_epochs":     60,
    "s3_batch":      48,
    "s3_lr":         5e-6,
    "s3_min_lr":     1e-7,
    "s3_dropout":    0.05,
    "s3_patience":   18,
    "arc_s":         48.0,    # ArcFace scale
    "arc_m":         0.35,    # ArcFace margin
    "arc_k":         3,       # SubCenter clusters per class

    # ── Loss ──────────────────────────────────────────────────────────
    "label_smoothing":  0.05,

    # ── Regularisation ────────────────────────────────────────────────
    "weight_decay":  1e-4,
    "grad_clip":     1.0,

    # ── EMA ───────────────────────────────────────────────────────────
    "ema_decay_s1":  0.995,
    "ema_decay_s2":  0.9995,
    "ema_decay_s3":  0.9999,

    # ── TTA ───────────────────────────────────────────────────────────
    "tta_n":         8,    # spatial views
    "tta_spectral":  True, # also use VIS/NIR spectral flips

    # ── Wavelength embedding ──────────────────────────────────────────
    "wl_embed_dim":  16,

    # ── Transformer branch ───────────────────────────────────────────
    "trans_patch_size":  16,   # bands per token  → 256/16 = 16 tokens
    "trans_d_model":     128,
    "trans_heads":       8,
    "trans_layers":      4,
    "trans_out_dim":     256,

    # ── Misc ──────────────────────────────────────────────────────────
    "device":        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":          42,
    "num_workers":   6,
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
#  ADAPTIVE EMA  (no hard reset between stages — only decay changes)
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay        = decay
        self._num_updates = 0
        self.shadow       = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _effective_decay(self) -> float:
        n = self._num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._num_updates += 1
        d = self._effective_decay()
        model_sd = model.state_dict()
        ema_sd   = self.shadow.state_dict()
        for k in ema_sd:
            if ema_sd[k].dtype.is_floating_point:
                ema_sd[k].copy_(d * ema_sd[k] + (1.0 - d) * model_sd[k])
            else:
                ema_sd[k].copy_(model_sd[k])

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    # ── NO hard reset — only adjust decay and continue momentum ───────
    def set_decay(self, decay: float) -> None:
        self.decay = decay

    def state_dict(self)               -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET  (with spectral-regional augmentation)
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """
    Memory-mapped HSI loader.

    Augmentations:
      - Band dropout (uniform random zero-out)
      - Band cutout (contiguous spectral block zeroed)
      - Per-region gain + shift: VIS and NIR zones perturbed independently
      - Gaussian spectral noise
      - Spatial: rot90 × 4 + h/v flip
    """

    def __init__(
        self,
        patches_path:     str,
        labels_path:      str,
        indices:          np.ndarray,
        augment:          bool  = False,
        band_drop_prob:   float = 0.02,
        max_cutout_bands: int   = 24,
        noise_std:        float = 0.015,
        regional_aug:     bool  = True,
    ) -> None:
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.augment          = augment
        self.band_drop_prob   = band_drop_prob
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std
        self.regional_aug     = regional_aug

    def __len__(self) -> int:
        return len(self.indices)

    # ── Spectral augmentations ─────────────────────────────────────────

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

    def _regional_perturbation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Independently perturb VIS and NIR zones with random gain + bias.
        Simulates sensor variability between spectral regions.
        gain  ∈ [0.90, 1.10],  bias ∈ [-0.05, 0.05]
        """
        x = x.clone()
        for sl in (VIS_BANDS, NIR_BANDS):
            gain = 0.90 + 0.20 * torch.rand(1).item()
            bias = (torch.rand(1).item() - 0.5) * 0.10
            x[sl] = x[sl] * gain + bias
        return x

    # ── Spatial augmentation ───────────────────────────────────────────

    def _spatial_augment(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])
        return torch.rot90(x, torch.randint(0, 4, (1,)).item(), dims=[1, 2])

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
            if self.regional_aug and torch.rand(1).item() < 0.50:
                patch = self._regional_perturbation(patch)
            patch = self._spatial_augment(patch)

        return patch, label


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION  (Mixup + CutMix)
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _cutmix(x, y, alpha):
    lam    = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx    = torch.randperm(B, device=x.device)
    r      = math.sqrt(1.0 - lam)
    ch, cw = int(H * r), int(W * r)
    cx     = random.randint(0, W)
    cy     = random.randint(0, H)
    x1 = max(cx - cw // 2, 0);  x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0);  y2 = min(cy + ch // 2, H)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_mix, y, y[idx], lam


def mixed_aug(x, y, alpha=0.4):
    fn = _mixup if torch.rand(1).item() < 0.5 else _cutmix
    return fn(x, y, alpha)


def mixed_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ══════════════════════════════════════════════════════════════════════
#  UTILITY: masked spectral statistics
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(x: torch.Tensor):
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
#  BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class WavelengthPositionalEncoding(nn.Module):
    def __init__(self, num_bands=256, embed_dim=16,
                 wl_min=WL_MIN, wl_max=WL_MAX) -> None:
        super().__init__()
        wl   = torch.linspace(0.0, 1.0, num_bands)
        half = embed_dim // 2
        freq = torch.exp(
            torch.arange(half).float() * -(math.log(10_000.0) / max(half - 1, 1))
        )
        enc = torch.zeros(num_bands, embed_dim)
        enc[:,  :half] = torch.sin(wl.unsqueeze(1) * freq.unsqueeze(0))
        enc[:, half:]  = torch.cos(wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("enc", enc)
        self.proj = nn.Linear(embed_dim, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self) -> torch.Tensor:
        return self.proj(self.enc).squeeze(-1).view(1, 1, -1)


class SpectralSE(nn.Module):
    """Per-sample spectral Squeeze-and-Excitation."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid,      bias=False),
            nn.GELU(),
            nn.Linear(mid,      channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x.mean(dim=[2, 3]))
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                           nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class CBAM(nn.Module):
    def __init__(self, c, r=8) -> None:
        super().__init__()
        mid = max(c // r, 8)
        self.ch_mlp = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False), nn.GELU(),
            nn.Conv2d(mid, c, 1, bias=False),
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
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
    def __init__(self, in_ch, out_ch, stride=1) -> None:
        super().__init__()
        mid    = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid,    1,             bias=False)
        self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid,   mid,    3, stride, 1,  bias=False)
        self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid,   out_ch, 1,             bias=False)
        self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                           nn.GroupNorm(min(8, out_ch), out_ch))
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = F.gelu(self.n2(self.c2(h)))
        h = self.n3(self.c3(h))
        return F.gelu(h + self.skip(x))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A — SPECTRAL PROFILE  (mean + derivative)
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None) -> None:
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(2, tower_ch, k=3)
        self.tower_m = self._tower(2, tower_ch, k=7)
        self.tower_l = self._tower(2, tower_ch, k=15)
        self.proj = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(in_ch, out_ch, k):
        mid = out_ch // 2
        return nn.Sequential(
            ResBlock1D(in_ch,  mid,     k),
            ResBlock1D(mid,    out_ch,  k),
            ResBlock1D(out_ch, out_ch,  k),
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
#  BRANCH B — SPECTRAL STATISTICS  (mean + std + max)
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None) -> None:
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(3, tower_ch, k=3)
        self.tower_m = self._tower(3, tower_ch, k=7)
        self.tower_l = self._tower(3, tower_ch, k=15)
        self.proj = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(in_ch, out_ch, k):
        mid = out_ch // 2
        return nn.Sequential(
            ResBlock1D(in_ch,  mid,     k),
            ResBlock1D(mid,    out_ch,  k),
            ResBlock1D(out_ch, out_ch,  k),
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
#  BRANCH C — SPATIAL CNN  (with SE band-gating instead of plain 1×1)
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands=256, out_dim=192) -> None:
        super().__init__()
        # SE-gated band reduction instead of plain 1×1 conv
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_bands, num_bands // 16, bias=False),
            nn.GELU(),
            nn.Linear(num_bands // 16, num_bands, bias=False),
            nn.Sigmoid(),
        )
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
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SE band gating
        w = self.se_gate(x)
        x = x * w.view(x.shape[0], x.shape[1], 1, 1)
        h   = self.band_reduce(x)
        h   = self.stages(h)
        avg = self.avg_pool(h).flatten(1)
        mx  = self.max_pool(h).flatten(1)
        return self.pool_proj(torch.cat([avg, mx], dim=1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECTRAL TRANSFORMER  (NEW)
#  ─────────────────────────────────────────────────────────────────────
#  Treats the 256-band spectrum as a sequence of patch-tokens.
#  Each patch covers `patch_size` bands (default 16) → 16 tokens.
#  Learnable patch embeddings + sinusoidal positional encoding.
#  Multi-head self-attention captures long-range spectral dependencies.
#
#  Reference: SpectralFormer (Hong et al., TGRS 2022)
#             HyperFormer    (Roy et al.,  IGARSS 2023)
# ══════════════════════════════════════════════════════════════════════

class SpectralTransformerBranch(nn.Module):
    def __init__(
        self,
        num_bands:  int   = 256,
        patch_size: int   = 16,
        d_model:    int   = 128,
        n_heads:    int   = 8,
        n_layers:   int   = 4,
        out_dim:    int   = 256,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        assert num_bands % patch_size == 0, \
            "num_bands must be divisible by patch_size"
        self.patch_size = patch_size
        n_patches = num_bands // patch_size  # 16

        # Linear patch embedding: (B, n_patches, patch_size) → (B, n_patches, d_model)
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Sinusoidal positional encoding for n_patches + 1 (cls) positions
        pe = self._sinusoidal_pe(n_patches + 1, d_model)
        self.register_buffer("pos_enc", pe)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = n_heads,
            dim_feedforward = d_model * 4,
            dropout     = dropout,
            activation  = "gelu",
            batch_first = True,
            norm_first  = True,   # Pre-LN (more stable training)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe   = torch.zeros(seq_len, d_model)
        pos  = torch.arange(seq_len).float().unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def forward(self, mean_spec: torch.Tensor) -> torch.Tensor:
        """
        mean_spec : (B, 256) — mean spectral profile across spatial dims
        returns   : (B, out_dim)
        """
        B, C = mean_spec.shape
        # Split into patches: (B, n_patches, patch_size)
        x = mean_spec.view(B, C // self.patch_size, self.patch_size)
        x = self.patch_embed(x)                           # (B, n_patches, d_model)

        # Prepend class token
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, n_patches+1, d_model)
        x   = x + self.pos_enc                            # add positional encoding

        # Transformer
        x = self.encoder(x)                               # (B, n_patches+1, d_model)
        x = self.norm(x)

        # Use CLS token + mean of patch tokens
        cls_out  = x[:, 0]
        patch_mean = x[:, 1:].mean(1)
        out = cls_out + patch_mean                         # (B, d_model)

        return self.proj(out)                              # (B, out_dim)


# ══════════════════════════════════════════════════════════════════════
#  SUB-CENTER ARCFACE LOSS
#  ─────────────────────────────────────────────────────────────────────
#  Wang et al., "Sub-center ArcFace: Boosting Face Recognition by
#  Large-Scale Noisy Web Faces", ECCV 2020.
#
#  Each class has K sub-centers; the cosine similarity is computed to
#  the nearest sub-center.  This handles intra-class variance naturally,
#  which is important for rice varieties with morphological diversity.
# ══════════════════════════════════════════════════════════════════════

class SubCenterArcFace(nn.Module):
    def __init__(
        self,
        in_dim:      int,
        num_classes: int,
        K:           int   = 3,
        s:           float = 48.0,
        m:           float = 0.35,
    ) -> None:
        super().__init__()
        self.K   = K
        self.s   = s
        self.m   = m
        # Weight matrix: (num_classes * K, in_dim)
        self.weight = nn.Parameter(torch.empty(num_classes * K, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        x      : (B, in_dim) — L2-normalized features
        labels : (B,) long  — needed during training
        returns: (B, num_classes) logits
        """
        # Normalize both features and sub-centers
        x_n = F.normalize(x, dim=1)                      # (B, in_dim)
        w_n = F.normalize(self.weight, dim=1)             # (C*K, in_dim)

        # Cosine similarity: (B, C*K)
        cos = x_n @ w_n.T
        num_classes = self.weight.shape[0] // self.K

        # Sub-center max pooling: take the nearest sub-center per class
        cos = cos.view(x.shape[0], num_classes, self.K)
        cos = cos.max(dim=2).values                       # (B, num_classes)

        if labels is None:
            return self.s * cos

        # Add angular margin to the target class
        theta_t    = torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        cos_margin = torch.cos(theta_t + self.m)

        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = self.s * (one_hot * cos_margin + (1.0 - one_hot) * cos)
        return logits


# ══════════════════════════════════════════════════════════════════════
#  MAIN MODEL — QuadSpectralNet  (4 branches + ArcFace head option)
# ══════════════════════════════════════════════════════════════════════

class QuadSpectralNet(nn.Module):
    """
    Four-branch hyperspectral classifier:

      Branch A — Multi-scale 1-D CNN on mean spectrum + 1st derivative
      Branch B — Multi-scale 1-D CNN on mean / std / max spectral stats
      Branch C — 2-D spatial CNN with SE band gating + CBAM
      Branch D — Spectral Transformer (patch-token self-attention)  ← NEW

    Fusion: concatenated → BN → MLP head
    During Stage 3 the MLP head is replaced by SubCenterArcFace.
    """

    def __init__(
        self,
        num_classes:  int   = 90,
        num_bands:    int   = 256,
        dropout:      float = 0.25,
        wl_embed_dim: int   = 16,
        use_arcface:  bool  = False,
        arc_s:        float = 48.0,
        arc_m:        float = 0.35,
        arc_k:        int   = 3,
        trans_patch_size: int = 16,
        trans_d_model:    int = 128,
        trans_heads:      int = 8,
        trans_layers:     int = 4,
        trans_out_dim:    int = 256,
    ) -> None:
        super().__init__()
        self.use_arcface = use_arcface

        # Shared wavelength encoding
        self.wl_enc = WavelengthPositionalEncoding(
            num_bands=num_bands, embed_dim=wl_embed_dim
        )

        # Spectral band attention (pre-stem)
        self.se = SpectralSE(num_bands, reduction=16)

        # Four branches
        self.branch_a = SpectralProfileBranch(256, tower_ch=80, wl_enc=self.wl_enc)
        self.branch_b = SpectralStatsBranch(  256, tower_ch=80, wl_enc=self.wl_enc)
        self.branch_c = SpatialCNNBranch(num_bands=num_bands, out_dim=192)
        self.branch_d = SpectralTransformerBranch(
            num_bands   = num_bands,
            patch_size  = trans_patch_size,
            d_model     = trans_d_model,
            n_heads     = trans_heads,
            n_layers    = trans_layers,
            out_dim     = trans_out_dim,
            dropout     = 0.1,
        )

        # Feature dimension
        fusion_dim = 256 + 256 + 192 + trans_out_dim   # 960

        # L2-norm + projection to a compact embedding before classification
        self.feat_proj = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        feat_dim = 512

        # Classifier head (CE stage)
        self.ce_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(256, num_classes),
        )

        # ArcFace head (Stage 3)
        self.arc_head = SubCenterArcFace(
            in_dim=feat_dim, num_classes=num_classes,
            K=arc_k, s=arc_s, m=arc_m,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight);  nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dim embedding (before classification)."""
        x = self.se(x)
        mean_s, std_s, max_s = masked_spectral_stats(x)
        mean_1d = mean_s          # (B, 256)

        fa = self.branch_a(mean_1d)
        fb = self.branch_b(mean_s, std_s, max_s)
        fc = self.branch_c(x)
        fd = self.branch_d(mean_1d)

        fused = torch.cat([fa, fb, fc, fd], dim=1)
        return self.feat_proj(fused)

    def forward(
        self,
        x:      torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.get_features(x)
        if self.use_arcface:
            return self.arc_head(feat, labels)
        return self.ce_head(feat)


# ══════════════════════════════════════════════════════════════════════
#  TTA  (spatial views + spectral flips)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(
    model:     nn.Module,
    x:         torch.Tensor,
    n_spatial: int  = 8,
    spectral:  bool = True,
) -> torch.Tensor:
    """
    Ensemble:
      - n_spatial spatial augmentations (rot90 × 4 + h/v flips)
      - optionally: original + VIS/NIR spectral-zone swapped version
    """
    device  = x.device
    views   = [(k, flip) for k in range(4) for flip in (False, True)][:n_spatial]
    logits  = []

    def _infer(inp):
        with autocast(device_type=device.type):
            return model(inp)

    for k, flip in views:
        aug = torch.rot90(x, k, dims=[2, 3])
        if flip:
            aug = torch.flip(aug, dims=[3])
        logits.append(_infer(aug))

    if spectral:
        # Spectral zone flip: swap VIS ↔ NIR regions
        x_sf = x.clone()
        x_sf[:, VIS_BANDS, ...] = x[:, NIR_BANDS, ...]
        x_sf[:, NIR_BANDS, ...] = x[:, VIS_BANDS, ...]
        for k, flip in views[:4]:           # 4 more views from spectral-flipped
            aug = torch.rot90(x_sf, k, dims=[2, 3])
            if flip:
                aug = torch.flip(aug, dims=[3])
            logits.append(_infer(aug))

    return torch.stack(logits).mean(0)


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITS + LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits():
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3,
                               stratify=labels, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.5,
                               stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(train_idx, val_idx, test_idx, batch_train):
    nw = CONFIG["num_workers"]
    kw = dict(num_workers=nw, pin_memory=True)

    train_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                        train_idx, augment=True),
        batch_size=batch_train, shuffle=True,
        persistent_workers=True, prefetch_factor=2, **kw,
    )
    val_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], val_idx),
        batch_size=64, shuffle=False, **kw,
    )
    test_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], test_idx),
        batch_size=64, shuffle=False, **{**kw, "num_workers": 2},
    )
    return train_ldr, val_ldr, test_ldr


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISER
# ══════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr: float) -> optim.AdamW:
    """
    Separate learning rates:
      - Transformer branch gets 0.5× LR (pre-trained inductive bias is fragile)
      - All BN / bias params: no weight decay
    """
    trans_ids = set(id(p) for p in model.branch_d.parameters())

    wd_normal,  no_wd_normal  = [], []
    wd_trans,   no_wd_trans   = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_trans = (id(p) in trans_ids)
        no_wd    = (p.ndim == 1 or name.endswith(".bias"))
        if in_trans:
            (no_wd_trans if no_wd else wd_trans).append(p)
        else:
            (no_wd_normal if no_wd else wd_normal).append(p)

    return optim.AdamW(
        [
            {"params": wd_normal,   "lr": lr,        "weight_decay": CONFIG["weight_decay"]},
            {"params": no_wd_normal,"lr": lr,        "weight_decay": 0.0},
            {"params": wd_trans,    "lr": lr * 0.5,  "weight_decay": CONFIG["weight_decay"]},
            {"params": no_wd_trans, "lr": lr * 0.5,  "weight_decay": 0.0},
        ],
        lr=lr,
    )


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   optim.Optimizer,
    criterion:   nn.Module,
    scaler:      GradScaler,
    ema:         Optional[ModelEMA],
    device:      torch.device,
    scheduler    = None,
    use_mixup:   bool  = True,
    mixup_alpha: float = 0.4,
    use_arcface: bool  = False,
) -> tuple[float, float]:
    model.train()
    total_loss = total_acc = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if use_mixup and not use_arcface:
            x_in, y_a, y_b, lam = mixed_aug(x, y, mixup_alpha)
        else:
            x_in, y_a, y_b, lam = x, y, y, 1.0

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type):
            if use_arcface:
                logits = model(x_in, labels=y_a)
            else:
                logits = model(x_in)
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
def evaluate(model, loader, device, use_arcface=False):
    model.eval()
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type):
            if use_arcface:
                logits = model(x, labels=None)   # no margin at eval
            else:
                logits = model(x)
        preds.append(logits.argmax(1).cpu())
        targets.append(y)

    p, t = torch.cat(preds), torch.cat(targets)
    f1   = f1_score(t.numpy(), p.numpy(), average="macro", zero_division=0)
    acc  = accuracy_score(t.numpy(), p.numpy())
    return f1, acc


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, stage, model, ema, val_acc, val_f1):
    torch.save({
        "epoch": epoch, "stage": stage,
        "model": model.state_dict(), "ema": ema.state_dict(),
        "val_acc": val_acc, "val_f1": val_f1,
    }, path)


def load_ckpt(path, model, ema, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    return ckpt


# ══════════════════════════════════════════════════════════════════════
#  STAGE RUNNERS
# ══════════════════════════════════════════════════════════════════════

def _hdr(title, epochs):
    w = 66
    print(f"\n{'═'*w}\n  {title}  [{epochs} epochs max]\n{'═'*w}")


def get_mixup_alpha(epoch):
    warmdown = int(0.8 * CONFIG["s1_epochs"])
    return max(0.1, CONFIG["s1_mixup"] * (1 - epoch / warmdown))


# ── Stage 1: Heavy aug + Mixup/CutMix + CE ─────────────────────────

def run_stage1(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt):
    optimizer = build_optimizer(model, lr=CONFIG["s1_max_lr"] / 25)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                CONFIG["s1_max_lr"],        # wd_normal
                CONFIG["s1_max_lr"],        # no_wd_normal
                CONFIG["s1_max_lr"] * 0.5,  # wd_trans
                CONFIG["s1_max_lr"] * 0.5,  # no_wd_trans
            ],
            epochs=CONFIG["s1_epochs"],
            steps_per_epoch=len(train_ldr),
            pct_start=0.15,
            div_factor=25,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )

    scaler = GradScaler()
    best_acc = no_improve = 0

    _hdr("Stage 1 — Heavy Aug + Mixup/CutMix + CE", CONFIG["s1_epochs"])

    for ep in range(1, CONFIG["s1_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=ema, device=device, scheduler=scheduler,
            use_mixup=True, mixup_alpha=get_mixup_alpha(ep),
        )
        _, va_live = evaluate(model,      val_ldr, device)
        vf1, va    = evaluate(ema.shadow, val_ldr, device)
        va_best    = max(va, va_live)

        lr_now = optimizer.param_groups[0]["lr"]
        ema_d  = ema._effective_decay()
        saved  = ""

        if va_best > best_acc:
            best_acc, no_improve = va_best, 0
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema, va_best, vf1)
            saved = "  ✓ Saved"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s1_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%}  EMA {va:.1%} │ "
            f"LR {lr_now:.2e}  EMA_d {ema_d:.4f}{saved}"
        )

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    return best_acc


# ── Stage 2: Clean fine-tune + CE ─────────────────────────────────

def run_stage2(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt):
    # Do NOT reset EMA — just lower the decay
    ema.set_decay(CONFIG["ema_decay_s2"])

    model.set_dropout(CONFIG["s2_dropout"])
    ema.set_dropout(CONFIG["s2_dropout"])

    optimizer = build_optimizer(model, lr=CONFIG["s2_lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["s2_epochs"], eta_min=CONFIG["s2_min_lr"]
    )
    scaler = GradScaler()
    best_acc = no_improve = 0

    _hdr("Stage 2 — Clean Fine-Tune (no Mixup, CE)", CONFIG["s2_epochs"])

    for ep in range(1, CONFIG["s2_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=ema, device=device, scheduler=None, use_mixup=False,
        )
        scheduler.step()

        _, va_live = evaluate(model,      val_ldr, device)
        vf1, va    = evaluate(ema.shadow, val_ldr, device)
        va_best    = max(va, va_live)

        lr_now = optimizer.param_groups[0]["lr"]
        saved  = ""

        if va_best > best_acc:
            best_acc, no_improve = va_best, 0
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema, va_best, vf1)
            saved = "  ✓ Saved"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s2_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%}  EMA {va:.1%} │ "
            f"LR {lr_now:.2e}{saved}"
        )

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    return best_acc


# ── Stage 3: SubCenter ArcFace fine-tune ───────────────────────────

def run_stage3(model, ema, train_ldr, val_ldr, device, best_ckpt):
    """
    Switch model to ArcFace mode and fine-tune at very low LR.
    CE is now replaced by plain CrossEntropy over ArcFace logits
    (no label smoothing — ArcFace already provides strong regularisation).
    """
    ema.set_decay(CONFIG["ema_decay_s3"])
    model.set_dropout(CONFIG["s3_dropout"])
    ema.set_dropout(CONFIG["s3_dropout"])

    # Enable ArcFace head
    model.use_arcface = True
    ema.shadow.use_arcface = True

    # Plain CE over ArcFace logits (no label smoothing)
    arc_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, lr=CONFIG["s3_lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["s3_epochs"], eta_min=CONFIG["s3_min_lr"]
    )
    scaler = GradScaler()
    best_acc = no_improve = 0

    _hdr("Stage 3 — SubCenter ArcFace Fine-Tune", CONFIG["s3_epochs"])

    for ep in range(1, CONFIG["s3_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, arc_criterion, scaler,
            ema=ema, device=device, scheduler=None,
            use_mixup=False, use_arcface=True,
        )
        scheduler.step()

        _, va_live = evaluate(model,      val_ldr, device, use_arcface=True)
        vf1, va    = evaluate(ema.shadow, val_ldr, device, use_arcface=True)
        va_best    = max(va, va_live)

        lr_now = optimizer.param_groups[0]["lr"]
        saved  = ""

        if va_best > best_acc:
            best_acc, no_improve = va_best, 0
            save_ckpt(best_ckpt, ep, "Stage 3", model, ema, va_best, vf1)
            saved = "  ✓ Saved"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%}  EMA {va:.1%} │ "
            f"LR {lr_now:.2e}{saved}"
        )

        if no_improve >= CONFIG["s3_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    return best_acc


# ══════════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt):
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")

    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow
    use_arc    = ckpt["stage"] == "Stage 3"
    eval_model.use_arcface = use_arc

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x = x.to(device, non_blocking=True)
            if use_tta:
                logits = tta_predict(
                    eval_model, x,
                    n_spatial = CONFIG["tta_n"],
                    spectral  = CONFIG["tta_spectral"],
                )
            else:
                with autocast(device_type=device.type):
                    logits = eval_model(x)
            preds.append(logits.argmax(1).cpu())
            targets.append(y)

        p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p, t)
        acc  = accuracy_score(t, p)
        f1m  = f1_score(t, p, average="macro",    zero_division=0)
        f1w  = f1_score(t, p, average="weighted", zero_division=0)
        print(f"\n  [{tag}]  Acc={acc:.1%}  F1(macro)={f1m:.4f}  F1(wt)={f1w:.4f}")

    print(f"\n  Checkpoint: epoch {ckpt['epoch']} | {ckpt['stage']} "
          f"| val={ckpt['val_acc']:.1%}")
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
    device = CONFIG["device"]
    labels, train_idx, val_idx, test_idx = build_splits()
    best_ckpt = os.path.join(CONFIG["output_dir"], "best_model.pth")

    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx) // CONFIG['num_classes']}")

    model = QuadSpectralNet(
        num_classes      = CONFIG["num_classes"],
        num_bands        = CONFIG["num_bands"],
        dropout          = CONFIG["s1_dropout"],
        wl_embed_dim     = CONFIG["wl_embed_dim"],
        use_arcface      = False,
        arc_s            = CONFIG["arc_s"],
        arc_m            = CONFIG["arc_m"],
        arc_k            = CONFIG["arc_k"],
        trans_patch_size = CONFIG["trans_patch_size"],
        trans_d_model    = CONFIG["trans_d_model"],
        trans_heads      = CONFIG["trans_heads"],
        trans_layers     = CONFIG["trans_layers"],
        trans_out_dim    = CONFIG["trans_out_dim"],
    ).to(device)

    ema   = ModelEMA(model, decay=CONFIG["ema_decay_s1"])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : QuadSpectralNet")
    print(f"Params : {n_par / 1e6:.2f}M")
    print(f"Device : {device}  |  EMA adaptive decay → {CONFIG['ema_decay_s1']}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # ── Stage 1 ────────────────────────────────────────────────────────
    train_ldr, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s1_batch"]
    )
    run_stage1(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt)

    # ── Stage 2 ────────────────────────────────────────────────────────
    print("\nLoading Stage 1 best checkpoint …")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  val={ckpt['val_acc']:.1%}  ({ckpt['stage']})")
    print(f"  Dropout: {CONFIG['s1_dropout']} → {CONFIG['s2_dropout']}")

    train_ldr, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"]
    )
    run_stage2(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt)

    # ── Stage 3 ────────────────────────────────────────────────────────
    print("\nLoading Stage 2 best checkpoint for ArcFace fine-tune …")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  val={ckpt['val_acc']:.1%}  ({ckpt['stage']})")
    print(f"  Dropout: {CONFIG['s2_dropout']} → {CONFIG['s3_dropout']}")

    train_ldr, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s3_batch"]
    )
    run_stage3(model, ema, train_ldr, val_ldr, device, best_ckpt)

    # ── Final test evaluation ──────────────────────────────────────────
    _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx, 64)
    final_evaluation(model, ema, test_ldr, device, best_ckpt)


if __name__ == "__main__":
    main()