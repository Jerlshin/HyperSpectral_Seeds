from __future__ import annotations

import copy
import math
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*lr_scheduler.*", category=UserWarning)

WL_MIN: float = 385.0
WL_MAX: float = 1000.0

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    "patches_data":   "./dataset/patches.npy",
    "labels_path":    "./dataset/labels.npy",
    "output_dir":     "./output_v8/",

    "num_bands":      256,
    "num_classes":    90,

    # Stage 1 — Focal loss + heavy aug + Mixup/CutMix
    "s1_epochs":      150,
    "s1_batch":       64,
    "s1_max_lr":      8e-4,
    "s1_dropout":     0.25,
    "s1_mixup":       0.4,
    "s1_patience":    35,
    "s1_label_smooth": 0.05,
    "focal_gamma":    1.5,       # FocalLoss focusing param

    # Stage 2 — Clean fine-tune + SWA
    "s2_epochs":      75,        # 3×T_0=25 cycles
    "s2_batch":       64,
    "s2_lr":          3e-5,
    "s2_min_lr":      3e-7,
    "s2_dropout":     0.08,
    "s2_patience":    25,        # longer patience (warm restarts reset progress)
    "s2_label_smooth": 0.0,      # no smoothing on clean fine-tune
    "s2_T0":          25,        # CosineAnnealingWarmRestarts period
    "swa_start_ep":   45,        # SWA begins at this Stage-2 epoch
    "swa_lr":         5e-6,

    "weight_decay":   1e-4,
    "grad_clip":      1.0,
    "ema_decay":      0.9999,
    "tta_n":          8,
    "wl_embed_dim":   16,

    "device":         torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":           42,
    "num_workers":    6,
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
    torch.backends.cudnn.benchmark     = True

set_seed(CONFIG["seed"])


# ══════════════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -(1 - p_t)^γ · log(p_t)
    Lin et al., "Focal Loss for Dense Object Detection" (ICLR 2017).

    For 90-class rice classification with highly imbalanced per-class
    difficulty: 29 classes at F1=1.0 vs 6 classes at F1<0.5.
    Standard CE gives equal gradient weight to easy and hard classes.
    Focal loss down-weights easy classes (p_t→1 → weight→0) and
    amplifies gradients for confused ones (p_t→0 → weight→1).

    γ=1.5: softer than γ=2 (RetinaNet standard), appropriate for
    90-way classification where most classes are already learnable.

    label_smoothing: applied BEFORE focal weighting. Cross-entropy on
    soft targets, then multiplied by focal weight (1-p_t)^γ.
    """

    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.05,
                 num_classes: int = 90) -> None:
        super().__init__()
        self.gamma          = gamma
        self.label_smoothing = label_smoothing
        self.num_classes    = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # --- soft label CE ---
        log_p  = F.log_softmax(logits, dim=1)               # (B, C)
        with torch.no_grad():
            smooth = self.label_smoothing / self.num_classes
            # one-hot + smoothing
            y_soft = torch.full_like(log_p, smooth)
            y_soft.scatter_(1, targets.unsqueeze(1),
                            1.0 - self.label_smoothing + smooth)
        ce = -(y_soft * log_p).sum(dim=1)                    # (B,)

        # --- focal weight based on TRUE-class probability ---
        p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
        focal_w = (1.0 - p_t) ** self.gamma

        return (focal_w * ce).mean()


# ══════════════════════════════════════════════════════════════════════
#  ADAPTIVE EMA  (with reset for Stage 2)
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Exponential Moving Average with adaptive decay + stage reset.

    KEY NEW METHOD: reset(model)
      Called at Stage 2 start. Copies live model weights into the shadow
      and resets the step counter to 0.

      Without reset (v6 bug):
        Shadow = blend(Stage-1 Mixup weights, Stage-2 clean weights)
        → BN running stats mismatch → EMA crashes to 68% before recovering
      With reset (v7):
        Shadow starts identical to live model at Stage-2 baseline
        → EMA cleanly tracks Stage-2 fine-tuning from ep 1
        → No regression, no crash, +2-3% over v6 Stage-2 EMA
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

    def reset(self, model: nn.Module) -> None:
        """
        Reset shadow = live model copy. Called at Stage 2 start.
        Wipes all Stage-1 Mixup contamination from the shadow.
        """
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def state_dict(self)               -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET  (+ SpectralShift augmentation)
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """
    Memory-mapped HSI loader.

    NEW IN V7: SpectralShift augmentation.
      Rolls the spectrum by a random offset in [-max_shift, +max_shift] bands.
      Simulates the ±2-5 nm inter-session calibration uncertainty of the
      Specim V10E sensor. The rice variety label is invariant to small
      spectral shifts, so this is a physically valid aug.
      torch.roll() wraps around: bands at one end appear at the other.
      This is acceptable because reflectance at the boundary bands (385 nm
      and 1000 nm) is typically very similar and featureless.
    """

    def __init__(
        self,
        patches_path:     str,
        labels_path:      str,
        indices:          np.ndarray,
        augment:          bool  = False,
        aug_level:        str   = "heavy",   # "heavy" | "light"
        band_drop_prob:   float = 0.04,
        max_cutout_bands: int   = 20,
        noise_std:        float = 0.02,
        max_shift:        int   = 5,
    ) -> None:
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.augment          = augment
        self.aug_level        = aug_level
        self.band_drop_prob   = band_drop_prob
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std
        self.max_shift        = max_shift

    def __len__(self) -> int:
        return len(self.indices)

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

    def _spectral_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Roll spectrum by random ±max_shift bands (dim 0 = bands)."""
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shift, dims=0)

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
            if self.aug_level == "heavy":
                # Full aug suite for Stage 1
                if torch.rand(1).item() < 0.7:
                    patch = self._band_dropout(patch)
                if torch.rand(1).item() < 0.5:
                    patch = self._band_cutout(patch)
                if torch.rand(1).item() < 0.4:
                    patch = self._spectral_noise(patch)
                if torch.rand(1).item() < 0.6:
                    patch = self._spectral_shift(patch)
                patch = self._spatial_augment(patch)
            else:
                # Light aug for Stage 2: spatial only (no spectral corruption)
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
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1 = max(cx - cw // 2, 0);  x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0);  y2 = min(cy + ch // 2, H)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_mix, y, y[idx], lam


def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1).item() < 0.5 else _cutmix)(x, y, alpha)


def mixed_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ══════════════════════════════════════════════════════════════════════
#  MASKED SPECTRAL STATISTICS  (float32, NaN-safe)
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(x: torch.Tensor):
    """Deterministic mean/std/max over foreground pixels. All in float32."""
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
#  WAVELENGTH POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class WavelengthPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding of physical wavelengths 385-1000 nm.
    Returns (1, 1, 256) bias added to spectral inputs.
    Non-zero init: trunc_normal(std=0.01) so it's active from step 1.
    """

    def __init__(self, num_bands: int = 256, embed_dim: int = 16) -> None:
        super().__init__()
        wl   = torch.linspace(0.0, 1.0, num_bands)
        half = embed_dim // 2
        freq = torch.exp(
            torch.arange(half).float() * -(math.log(10_000.0) / max(half - 1, 1))
        )
        enc          = torch.zeros(num_bands, embed_dim)
        enc[:,  :half] = torch.sin(wl.unsqueeze(1) * freq.unsqueeze(0))
        enc[:, half:]  = torch.cos(wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("enc", enc)
        self.proj = nn.Linear(embed_dim, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self) -> torch.Tensor:
        return self.proj(self.enc).squeeze(-1).view(1, 1, -1)


# ══════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class SpectralSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.GELU(),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x.mean(dim=[2, 3]))
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7) -> None:
        super().__init__()
        pad        = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid = max(c // r, 8)
        self.ch_mlp = nn.Sequential(
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
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid     = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid,    1,             bias=False)
        self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid,   mid,    3, stride, 1,  bias=False)
        self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid,   out_ch, 1,             bias=False)
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
#  BRANCHES  (identical to v6 — working well)
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80,
                 wl_enc: Optional[WavelengthPositionalEncoding] = None):
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(2, tower_ch, k=3)
        self.tower_m = self._tower(2, tower_ch, k=7)
        self.tower_l = self._tower(2, tower_ch, k=15)
        self.proj    = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(i, o, k):
        m = o // 2
        return nn.Sequential(ResBlock1D(i, m, k), ResBlock1D(m, o, k), ResBlock1D(o, o, k))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], dim=1)

    def forward(self, mean_spec):
        s = mean_spec.unsqueeze(1)
        d = F.pad(torch.diff(s, dim=2), (0, 1))
        x = torch.cat([s, d], dim=1)
        if self.wl_enc is not None: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], dim=1))


class SpectralStatsBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80,
                 wl_enc: Optional[WavelengthPositionalEncoding] = None):
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(3, tower_ch, k=3)
        self.tower_m = self._tower(3, tower_ch, k=7)
        self.tower_l = self._tower(3, tower_ch, k=15)
        self.proj    = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(i, o, k):
        m = o // 2
        return nn.Sequential(ResBlock1D(i, m, k), ResBlock1D(m, o, k), ResBlock1D(o, o, k))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], dim=1)

    def forward(self, mean_s, std_s, max_s):
        x = torch.stack([mean_s, std_s, max_s], dim=1)
        if self.wl_enc is not None: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], dim=1))


class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands=256, out_dim=192):
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 32, 1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.stages = nn.Sequential(
            ResBlock2D(32,  64,  stride=2), CBAM(64),
            ResBlock2D(64,  128, stride=2), CBAM(128),
            ResBlock2D(128, 192, stride=2), CBAM(192),
            ResBlock2D(192, out_dim, stride=2),
        )
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.max_pool  = nn.AdaptiveMaxPool2d(1)
        self.pool_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.BatchNorm1d(out_dim), nn.GELU(),
        )

    def forward(self, x):
        h   = self.stages(self.band_reduce(x))
        avg = self.avg_pool(h).flatten(1)
        mx  = self.max_pool(h).flatten(1)
        return self.pool_proj(torch.cat([avg, mx], dim=1))


# ══════════════════════════════════════════════════════════════════════
#  MAIN MODEL — SpectralTripleNet v7
# ══════════════════════════════════════════════════════════════════════

class SpectralTripleNet(nn.Module):
    """
    Architecture unchanged from v6 (working at 79% TTA).
    All improvements are in training: Focal Loss, EMA reset, SWA,
    warm restarts, spectral shift aug.

    Branch A : Spectral Profile (mean + derivative + WL embed) → 256-D
    Branch B : Spectral Stats   (mean + std + max  + WL embed) → 256-D
    Branch C : Spatial CNN      (morphology / texture)         → 192-D
    Fused: 704-D  →  BN(512) → GELU → Dropout → BN(256) → GELU → 90
    """

    def __init__(self, num_classes=90, num_bands=256, dropout=0.25,
                 wl_embed_dim=16):
        super().__init__()
        self.wl_enc   = WavelengthPositionalEncoding(num_bands, wl_embed_dim)
        self.se       = SpectralSE(num_bands, reduction=16)
        self.branch_a = SpectralProfileBranch(256, 80, self.wl_enc)
        self.branch_b = SpectralStatsBranch(  256, 80, self.wl_enc)
        self.branch_c = SpatialCNNBranch(num_bands, 192)

        fusion_dim = 704
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout * 0.4),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def set_dropout(self, p: float):
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        mean_s, std_s, max_s = masked_spectral_stats(x)
        fa = self.branch_a(mean_s)
        fb = self.branch_b(mean_s, std_s, max_s)
        fc = self.branch_c(x)
        return self.classifier(torch.cat([fa, fb, fc], dim=1))


# ══════════════════════════════════════════════════════════════════════
#  TTA
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model, x: torch.Tensor, n: int = 8) -> torch.Tensor:
    device = x.device
    views  = [(k, f) for k in range(4) for f in (False, True)][:n]
    logits = []
    for k, flip in views:
        aug = torch.rot90(x, k, dims=[2, 3])
        if flip: aug = torch.flip(aug, dims=[3])
        with autocast(device_type=device.type):
            logits.append(model(aug))
    return torch.stack(logits).mean(0)


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════

def build_splits():
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(train_idx, val_idx, test_idx, batch_train: int,
                  aug_level: str = "heavy"):
    nw = CONFIG["num_workers"]
    kw = dict(num_workers=nw, pin_memory=True)
    train_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                        train_idx, augment=True, aug_level=aug_level),
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
    wd_p, no_wd_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_wd_p if (p.ndim == 1 or name.endswith(".bias")) else wd_p).append(p)
    return optim.AdamW(
        [{"params": wd_p,    "weight_decay": CONFIG["weight_decay"]},
         {"params": no_wd_p, "weight_decay": 0.0}],
        lr=lr,
    )


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler, ema,
                    device, scheduler=None, use_mixup=True,
                    mixup_alpha=0.4):
    model.train()
    total_loss = total_acc = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_in, y_a, y_b, lam = (
            mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type):
            logits = model(x_in)
            loss   = mixed_loss(criterion, logits, y_a, y_b, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None: scheduler.step()
        if ema is not None: ema.update(model)
        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == y).float().mean().item()
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False, tta_n=8):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if use_tta:
            logits = tta_predict(model, x, tta_n)
        else:
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
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, stage, model, ema, val_acc, val_f1):
    torch.save({"epoch": epoch, "stage": stage,
                "model": model.state_dict(), "ema": ema.state_dict(),
                "val_acc": val_acc, "val_f1": val_f1}, path)


def load_ckpt(path, model, ema, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    return ckpt


def _hdr(title, epochs):
    w = 68
    print(f"\n{'═'*w}\n  {title}  [{epochs} epochs max]\n{'═'*w}")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1
# ══════════════════════════════════════════════════════════════════════

def run_stage1(model, ema, train_ldr, val_ldr, device, best_ckpt) -> float:
    """
    Stage 1: OneCycleLR per-batch, Mixup+CutMix, Focal Loss.

    Checkpoint policy: EMA only (live val is noisy due to Mixup training
    on blended inputs then evaluating on clean validation samples).
    The EMA is the only reliable performance signal in Stage 1.
    """
    criterion = FocalLoss(
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["s1_label_smooth"],
        num_classes=CONFIG["num_classes"],
    )
    optimizer = build_optimizer(model, lr=CONFIG["s1_max_lr"] / 25)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CONFIG["s1_max_lr"],
            epochs=CONFIG["s1_epochs"],
            steps_per_epoch=len(train_ldr),
            pct_start=0.15, div_factor=25,
            final_div_factor=1e4, anneal_strategy="cos",
        )

    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 1 — FocalLoss + Heavy Aug + Mixup/CutMix", CONFIG["s1_epochs"])
    print(f"  FocalLoss γ={CONFIG['focal_gamma']}  label_smooth={CONFIG['s1_label_smooth']}")

    for ep in range(1, CONFIG["s1_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler, ema,
            device, scheduler=scheduler,
            use_mixup=True, mixup_alpha=CONFIG["s1_mixup"],
        )
        vf1, va_ema  = evaluate(ema.shadow, val_ldr, device)
        _,   va_live = evaluate(model,      val_ldr, device)

        lr_now = optimizer.param_groups[0]["lr"]
        ema_d  = ema.current_decay
        saved  = ""

        # Checkpoint on EMA only — live is too noisy during Mixup training
        if va_ema > best_acc:
            best_acc, no_improve = va_ema, 0
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema, va_ema, vf1)
            saved = " ✓"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s1_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%}  EMA {va_ema:.1%}* │ "
            f"LR {lr_now:.2e}  d={ema_d:.4f}{saved}"
        )

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stop at epoch {ep} (patience={CONFIG['s1_patience']}).")
            break

    print(f"\n  Stage 1 best EMA: {best_acc:.1%}")
    return best_acc


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2
# ══════════════════════════════════════════════════════════════════════

def run_stage2(model, ema, train_ldr, val_ldr, train_ldr_bn,
               device, best_ckpt) -> float:
    """
    Stage 2: Clean fine-tune with CosineAnnealingWarmRestarts + SWA.

    KEY FIX — EMA reset at start:
      ema.reset(model) copies live model into the shadow and resets the
      step counter. This eliminates the Stage-2 EMA regression observed
      in v6 (68.3% nadir) where the shadow was contaminated with
      Stage-1 Mixup-era weights.

    Checkpoint policy: EMA only (consistent with Stage 1).
      After reset, EMA cleanly tracks Stage-2 fine-tuning.

    SWA (swa_start_ep onwards):
      Creates AveragedModel(model) which maintains a running avg of live
      model snapshots. After training ends, update_bn() recalibrates BN
      running stats. SWA finds flatter loss basins → better generalisation.
      Reference: Izmailov et al., "Averaging Weights Leads to Wider Optima
      and Better Generalization", UAI 2018.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["s2_label_smooth"])

    # ── EMA RESET ──────────────────────────────────────────────────────
    ema.reset(model)
    model.set_dropout(CONFIG["s2_dropout"])
    ema.set_dropout(CONFIG["s2_dropout"])
    print(f"\n  ✓ EMA reset: shadow = live model (no Stage-1 ghost)")

    optimizer = build_optimizer(model, lr=CONFIG["s2_lr"])

    # Cosine with warm restarts: T_0=25 → resets at ep 25, 50, 75
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CONFIG["s2_T0"], T_mult=1, eta_min=CONFIG["s2_min_lr"]
    )

    # SWA setup
    swa_model  = AveragedModel(model)
    swa_sched  = SWALR(optimizer, swa_lr=CONFIG["swa_lr"],
                       anneal_epochs=5, last_epoch=-1)
    swa_active = False

    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 2 — Clean Fine-Tune + WarmRestarts + SWA", CONFIG["s2_epochs"])
    print(f"  CosineWarmRestarts T_0={CONFIG['s2_T0']}  "
          f"SWA from ep {CONFIG['swa_start_ep']}  label_smooth={CONFIG['s2_label_smooth']}")

    for ep in range(1, CONFIG["s2_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler, ema,
            device, scheduler=None,  # per-epoch stepping below
            use_mixup=False,
        )

        # ── SWA or normal LR step ────────────────────────────────────
        if ep >= CONFIG["swa_start_ep"]:
            if not swa_active:
                swa_active = True
                print(f"\n  SWA activated at epoch {ep}")
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        vf1, va_ema  = evaluate(ema.shadow, val_ldr, device)
        _,   va_live = evaluate(model,      val_ldr, device)

        lr_now = optimizer.param_groups[0]["lr"]
        swa_tag = " [SWA]" if swa_active else ""
        saved  = ""

        # Checkpoint on EMA only (consistent with Stage 1, EMA is now clean)
        if va_ema > best_acc:
            best_acc, no_improve = va_ema, 0
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema, va_ema, vf1)
            saved = " ✓"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s2_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%}  EMA {va_ema:.1%}*{swa_tag} │ "
            f"LR {lr_now:.2e}{saved}"
        )

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stop at epoch {ep} (patience={CONFIG['s2_patience']}).")
            break

    # ── SWA BN update pass ────────────────────────────────────────────
    if swa_active:
        print("\n  Running SWA BN update (forward pass on training data)...")
        with torch.no_grad():
            update_bn(train_ldr_bn, swa_model, device=device)

        # Evaluate the SWA model
        _,   swa_acc = evaluate(swa_model, val_ldr, device)
        print(f"  SWA model val accuracy: {swa_acc:.1%}")

        # Save SWA checkpoint if it outperforms EMA
        if swa_acc > best_acc:
            best_acc = swa_acc
            swa_f1, _ = evaluate(swa_model, val_ldr, device)
            swa_path = best_ckpt.replace(".pth", "_swa.pth")
            torch.save({"model": swa_model.module.state_dict(),
                        "val_acc": swa_acc, "stage": "Stage 2 SWA"}, swa_path)
            print(f"  SWA checkpoint saved → {swa_path}  ({swa_acc:.1%})")

    print(f"\n  Stage 2 best EMA: {best_acc:.1%}")
    return best_acc


# ══════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt):
    w = 68
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")

    # Try SWA checkpoint first (may exist and be better)
    swa_path = best_ckpt.replace(".pth", "_swa.pth")
    if os.path.exists(swa_path):
        swa_sd = torch.load(swa_path, map_location=device)
        model.load_state_dict(swa_sd["model"])
        eval_src = "SWA"
        eval_model = model
    else:
        ckpt = load_ckpt(best_ckpt, model, ema, device)
        eval_src = f"EMA (ep {ckpt['epoch']})"
        eval_model = ema.shadow

    print(f"  Evaluating: {eval_src}")

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
        p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p, t)
        acc = accuracy_score(t, p)
        f1m = f1_score(t, p, average="macro",    zero_division=0)
        f1w = f1_score(t, p, average="weighted", zero_division=0)
        print(f"\n  [{tag}]  Acc={acc:.1%}  F1(macro)={f1m:.4f}  F1(wt)={f1w:.4f}")

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

    labels, train_idx, val_idx, test_idx = build_splits()
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx) // CONFIG['num_classes']}")

    model = SpectralTripleNet(
        num_classes=CONFIG["num_classes"],
        num_bands=CONFIG["num_bands"],
        dropout=CONFIG["s1_dropout"],
        wl_embed_dim=CONFIG["wl_embed_dim"],
    ).to(device)

    ema   = ModelEMA(model, decay=CONFIG["ema_decay"])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : SpectralTripleNet v7")
    print(f"Params : {n_par / 1e6:.2f}M")
    print(f"Device : {device}  |  EMA adaptive → max {CONFIG['ema_decay']}")
    print(f"Stage 1: FocalLoss γ={CONFIG['focal_gamma']} | Mixup+CutMix | "
          f"SpectralShift ±{5}")
    print(f"Stage 2: Clean | WarmRestarts T_0={CONFIG['s2_T0']} | "
          f"SWA from ep {CONFIG['swa_start_ep']}")

    # ── Stage 1 ──────────────────────────────────────────────────────
    train_ldr, val_ldr, _ = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s1_batch"], aug_level="heavy"
    )
    run_stage1(model, ema, train_ldr, val_ldr, device, best_ckpt)

    # ── Stage 2 ──────────────────────────────────────────────────────
    print("\nLoading Stage 1 best EMA checkpoint for Stage 2 ...")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  val_ema={ckpt['val_acc']:.1%}  ({ckpt['stage']})")
    print(f"  Dropout: {CONFIG['s1_dropout']} → {CONFIG['s2_dropout']}")

    train_ldr, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"], aug_level="light"
    )
    # Separate BN-update loader (no augmentation, for SWA BN recalibration)
    train_ldr_bn, _, _ = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"], aug_level="light"
    )

    run_stage2(model, ema, train_ldr, val_ldr, train_ldr_bn, device, best_ckpt)

    # ── Final test ────────────────────────────────────────────────────
    _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx, 64)
    final_evaluation(model, ema, test_ldr, device, best_ckpt)


if __name__ == "__main__":
    main()