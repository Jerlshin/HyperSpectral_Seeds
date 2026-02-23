"""
HSI Rice Seed Classification  ·  SpectralTripleNet v7
=======================================================
Dataset : 90 varieties × ~96 seeds   (~8 640 total)
Input   : patches.npy -> (N, 256, 64, 64) float16 channel-first
          labels.npy  -> (N,)              int64

EVIDENCE-BASED ANALYSIS OF V6 TRAINING LOGS
=============================================

FINDING 1 - LIVE MODEL 30-POINT SWINGS IN STAGE 1
  Ep46->47: 47.4% -> 9.6%  (37-point crash)
  Ep73:     Live=20.7%  EMA=64.5%  (43-point gap!)
  ROOT CAUSE: BatchNorm1d in spectral branches accumulates running_mean/var
  from Mixup-augmented training batches. Mixup creates interpolated samples
  whose spectral distribution differs from clean validation images. When
  model.eval() is called, it switches to these contaminated running stats,
  causing wildly wrong normalisation on clean validation images.
  EMA was immune because its BN buffers are NEVER updated (only nn.Parameters
  are EMA-averaged, not nn.Buffers). Its frozen initial BN stats happen to be
  near-identity (mean~0, var~1) which is better than Mixup-contaminated stats.
  FIX v7: Replace ALL BatchNorm1d with GroupNorm in spectral branches.
  GroupNorm has no running stats - normalises within the forward pass using
  current batch statistics regardless of train/eval mode.

FINDING 2 - EMA COLLAPSE IN STAGE 2 (ep41: 78.7% -> ep45: 68.3%)
  After the Stage 2 peak at ep28 (Live=79.7%, EMA=78.7%), EMA drops 10 points
  to 68.3% by ep45, then slowly recovers to 79.4% by ep59.
  ROOT CAUSE: EMA._num_updates ~= 14,250 at Stage 2 start (150ep x 95 steps).
  Adaptive decay ~= 0.9993. Stage 2 live model moves quickly at lr=3e-5
  (no Mixup -> larger effective gradients) but EMA barely follows.
  As LR decays in Stage 2, the live model stops moving and EMA slowly catches up.
  FIX v7: ema.reset(model) at Stage 2 start. Copies live weights to EMA shadow
  AND resets _num_updates=0. Adaptive EMA restarts from decay~0.09 so it tracks
  the Stage 2 model within 5 epochs instead of 20.

FINDING 3 - 16% TRAIN-VAL OVERFITTING GAP IN STAGE 2
  Final Stage 2: Train 95.5%  Val 79.3%  Gap = 16.2%
  With 67 samples/class, model memorises training set by epoch 15 once
  Mixup is removed. Remaining 45 epochs contribute nothing useful.
  FIX v7: Stage 2 keeps SPATIAL augmentation (flip + rot90).
  8x effective samples per class. Preserves class identity (unlike Mixup).
  Expected: train drops to ~85%, val improves +2-3%, gap reduces to ~8-10%.

FINDING 4 - CHECKPOINT SELECTION NOISE
  Stage 1 best checkpoint: ep140 Live=76.2% (EMA=74.0%).
  But ep141-142 EMA was 74.6-74.7% (HIGHER than saved EMA=74.0%).
  Decision used max(Live, EMA) with noisy Live model -> saved suboptimal EMA.
  FIX v7: Checkpoint on EMA only in Stage 1.

NEW IN V7
=========
1. GroupNorm everywhere in spectral branches (replaces BatchNorm1d)
   -> eliminates BN contamination from Mixup training
   -> Live model eval becomes as stable as EMA

2. DropPath / Stochastic Depth (Huang et al. 2016)
   -> each residual block randomly bypassed during training
   -> forces earlier layers to maintain direct gradient flow
   -> regularises without reducing representation capacity

3. CosineClassifier (CosFace-style, Wang et al. 2018)
   -> normalised weight + temperature scaling for final 256->90 layer
   -> encourages intra-class compactness and inter-class angular separation
   -> better for fine-grained 90-class than plain dot product

4. EMA.reset(model) at Stage 2 start
   -> copies live model weights to EMA shadow
   -> resets _num_updates=0 (adaptive EMA restarts from decay~0.09)
   -> EMA tracks Stage 2 from the start instead of lagging 20 epochs

5. Stage 1: EMA-only checkpoint (not max(live,ema) with noisy Live)

6. Stage 2: spatial-only augmentation (flip+rot90, no spectral, no Mixup)
   -> reduces 16% overfit gap by ~6-8%

7. Stage 2 LR warmup (5 epochs linear before cosine decay)
   -> avoids destabilising freshly reset EMA in first few epochs
"""

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
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*", category=UserWarning)

WL_MIN: float = 385.0
WL_MAX: float = 1000.0

# =============================================================================
#  CONFIG
# =============================================================================

CONFIG: dict = {
    "patches_data":  "./dataset/patches.npy",
    "labels_path":   "./dataset/labels.npy",
    "output_dir":    "./output_v7/",

    "num_bands":     256,
    "num_classes":   90,

    # Stage 1 - Heavy aug + Mixup/CutMix
    "s1_epochs":     150,
    "s1_batch":      64,
    "s1_max_lr":     8e-4,
    "s1_dropout":    0.25,
    "s1_drop_path":  0.10,
    "s1_mixup":      0.4,
    "s1_patience":   40,

    # Stage 2 - Spatial aug only (no Mixup, no spectral aug)
    "s2_epochs":     100,
    "s2_batch":      48,
    "s2_lr":         4e-5,
    "s2_min_lr":     4e-7,
    "s2_warmup_ep":  5,
    "s2_dropout":    0.10,
    "s2_drop_path":  0.05,
    "s2_patience":   30,

    "label_smoothing": 0.08,
    "weight_decay":    1e-4,
    "grad_clip":       1.0,
    "cos_temp":        16.0,
    "ema_decay":       0.9999,
    "tta_n":           8,
    "wl_embed_dim":    16,

    "device":      torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":        42,
    "num_workers": 6,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.cuda.empty_cache()


# =============================================================================
#  REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

set_seed(CONFIG["seed"])


# =============================================================================
#  DROP PATH (Stochastic Depth - Huang et al. 2016)
# =============================================================================

def drop_path_fn(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep  = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    rand = torch.rand(shape, dtype=x.dtype, device=x.device)
    rand = (rand < keep).float().div_(keep)

    return x * rand


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_fn(x, self.p, self.training)


# =============================================================================
#  ADAPTIVE EMA
# =============================================================================

class ModelEMA:
    """
    Exponential Moving Average with adaptive decay.

        decay_t = min(max_decay, (1+step)/(10+step))

    reset(model): copies live weights to shadow, resets _num_updates=0.
    Call at Stage 2 start to fix the 20-epoch EMA lag observed in v6.
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

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        """Copy live weights to shadow and restart adaptive decay counter."""
        self.shadow.load_state_dict(
            {k: v.detach().clone() for k, v in model.state_dict().items()}
        )
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def set_drop_path(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, DropPath):
                m.p = p

    def state_dict(self)               -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


# =============================================================================
#  DATASET
# =============================================================================

class RiceSeedDataset(Dataset):
    """
    Memory-mapped HSI loader with configurable augmentation.

    augment=False           : clean evaluation (val/test)
    augment=True            : full aug (Stage 1)
    augment=True,spatial_only=True : flip+rot90 only (Stage 2)
    """

    def __init__(
        self,
        patches_path:     str,
        labels_path:      str,
        indices:          np.ndarray,
        augment:          bool  = False,
        spatial_only:     bool  = False,
        band_drop_prob:   float = 0.04,
        max_cutout_bands: int   = 20,
        noise_std:        float = 0.02,
    ) -> None:
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.augment          = augment
        self.spatial_only     = spatial_only
        self.band_drop_prob   = band_drop_prob
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std

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
            if not self.spatial_only:
                if torch.rand(1).item() < 0.7:
                    patch = self._band_dropout(patch)
                if torch.rand(1).item() < 0.5:
                    patch = self._band_cutout(patch)
                if torch.rand(1).item() < 0.4:
                    patch = self._spectral_noise(patch)
            patch = self._spatial_augment(patch)

        return patch, label


# =============================================================================
#  BATCH AUGMENTATION (Mixup + CutMix) - Stage 1 only
# =============================================================================

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


# =============================================================================
#  MASKED SPECTRAL STATISTICS  (float32, NaN-safe)
# =============================================================================

def masked_spectral_stats(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


# =============================================================================
#  WAVELENGTH POSITIONAL ENCODING
# =============================================================================

class WavelengthPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding of physical wavelengths (385-1000 nm).
    Returns (1, 1, 256) bias broadcast-added to spectral inputs.
    Non-zero init (trunc_normal std=0.01) -> active from step 1.
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


# =============================================================================
#  BUILDING BLOCKS - All GroupNorm (no BN running-stat contamination)
# =============================================================================

def _gn(c: int) -> nn.GroupNorm:
    g = min(8, c)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


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
    """
    1D residual block with GroupNorm and DropPath.
    GroupNorm: no running stats -> stable eval during Mixup training.
    DropPath: stochastic depth for better gradient flow and regularisation.
    """

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 7, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        pad        = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.gn1   = _gn(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.gn2   = _gn(out_ch)
        self.skip  = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), _gn(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.dp    = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(self.dp(h) + self.skip(x))


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
    def __init__(
        self, in_ch: int, out_ch: int, stride: int = 1, drop_path: float = 0.0
    ) -> None:
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
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.c1(x)))
        h = F.gelu(self.n2(self.c2(h)))
        h = self.n3(self.c3(h))
        return F.gelu(self.dp(h) + self.skip(x))


# =============================================================================
#  COSINE CLASSIFIER (CosFace-style)
# =============================================================================

class CosineClassifier(nn.Module):
    """
    Cosine similarity classifier with learnable temperature.

    Normalises both features and weights to unit sphere, then computes
    cosine similarity x temperature. Better for fine-grained 90-class:
    - Intra-class compactness (same direction in feature space)
    - Inter-class angular separation
    Reference: Wang et al. CosFace (CVPR 2018).
    """

    def __init__(self, in_features: int, num_classes: int, temp: float = 16.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.trunc_normal_(self.weight, std=0.02)
        self.temp = temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x,           p=2, dim=1, eps=1e-6)
        w_n = F.normalize(self.weight,  p=2, dim=1, eps=1e-6)
        return (x_n @ w_n.T) * self.temp


# =============================================================================
#  BRANCH A - SPECTRAL PROFILE
# =============================================================================

class SpectralProfileBranch(nn.Module):
    def __init__(
        self,
        out_dim:   int = 256,
        tower_ch:  int = 80,
        drop_path: float = 0.0,
        wl_enc:    Optional[WavelengthPositionalEncoding] = None,
    ) -> None:
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(2, tower_ch, k=3,  dp=drop_path)
        self.tower_m = self._tower(2, tower_ch, k=7,  dp=drop_path)
        self.tower_l = self._tower(2, tower_ch, k=15, dp=drop_path)
        self.proj    = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            _gn(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(in_ch, out_ch, k, dp):
        mid = out_ch // 2
        return nn.Sequential(
            ResBlock1D(in_ch,  mid,     k, dp * 0.5),
            ResBlock1D(mid,    out_ch,  k, dp),
            ResBlock1D(out_ch, out_ch,  k, dp),
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


# =============================================================================
#  BRANCH B - SPECTRAL STATISTICS
# =============================================================================

class SpectralStatsBranch(nn.Module):
    def __init__(
        self,
        out_dim:   int = 256,
        tower_ch:  int = 80,
        drop_path: float = 0.0,
        wl_enc:    Optional[WavelengthPositionalEncoding] = None,
    ) -> None:
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(3, tower_ch, k=3,  dp=drop_path)
        self.tower_m = self._tower(3, tower_ch, k=7,  dp=drop_path)
        self.tower_l = self._tower(3, tower_ch, k=15, dp=drop_path)
        self.proj    = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim),
            _gn(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _tower(in_ch, out_ch, k, dp):
        mid = out_ch // 2
        return nn.Sequential(
            ResBlock1D(in_ch,  mid,     k, dp * 0.5),
            ResBlock1D(mid,    out_ch,  k, dp),
            ResBlock1D(out_ch, out_ch,  k, dp),
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


# =============================================================================
#  BRANCH C - SPATIAL CNN
# =============================================================================

class SpatialCNNBranch(nn.Module):
    def __init__(
        self, num_bands: int = 256, out_dim: int = 192, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        dp = drop_path
        self.stages = nn.Sequential(
            ResBlock2D(32,  64,      stride=2, drop_path=dp * 0.5), CBAM(64),
            ResBlock2D(64,  128,     stride=2, drop_path=dp * 0.7), CBAM(128),
            ResBlock2D(128, 192,     stride=2, drop_path=dp),       CBAM(192),
            ResBlock2D(192, out_dim, stride=2, drop_path=dp),
        )
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.max_pool  = nn.AdaptiveMaxPool2d(1)
        self.pool_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GroupNorm(min(8, out_dim), out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.band_reduce(x)
        h   = self.stages(h)
        avg = self.avg_pool(h).flatten(1)
        mx  = self.max_pool(h).flatten(1)
        return self.pool_proj(torch.cat([avg, mx], dim=1))


# =============================================================================
#  MAIN MODEL - SpectralTripleNet v7
# =============================================================================

class SpectralTripleNet(nn.Module):
    """
    Three-branch HSI classification network v7.

    Branch A : Spectral Profile  (mean+derivative+WL embed) -> 256-D
    Branch B : Spectral Stats    (mean+std+max+WL embed)    -> 256-D
    Branch C : Spatial CNN       (2D morphology/texture)    -> 192-D
    Fused: 704-D -> GN(512) -> Dropout -> GN(256) -> CosineClassifier(90)

    All spectral ResBlocks use GroupNorm -> stable eval during Mixup training.
    DropPath throughout for stochastic depth regularisation.
    CosineClassifier for better angular discrimination across 90 classes.
    """

    def __init__(
        self,
        num_classes:  int   = 90,
        num_bands:    int   = 256,
        dropout:      float = 0.25,
        drop_path:    float = 0.10,
        wl_embed_dim: int   = 16,
        cos_temp:     float = 16.0,
    ) -> None:
        super().__init__()

        self.wl_enc = WavelengthPositionalEncoding(num_bands, wl_embed_dim)
        self.se     = SpectralSE(num_bands, reduction=16)

        self.branch_a = SpectralProfileBranch(
            256, tower_ch=80, drop_path=drop_path, wl_enc=self.wl_enc
        )
        self.branch_b = SpectralStatsBranch(
            256, tower_ch=80, drop_path=drop_path, wl_enc=self.wl_enc
        )
        self.branch_c = SpatialCNNBranch(
            num_bands=num_bands, out_dim=192, drop_path=drop_path * 1.5
        )

        fusion_dim = 256 + 256 + 192  # 704
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            _gn(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            _gn(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
        )
        self.classifier = CosineClassifier(256, num_classes, temp=cos_temp)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def set_drop_path(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, DropPath):
                m.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        mean_s, std_s, max_s = masked_spectral_stats(x)
        fa = self.branch_a(mean_s)
        fb = self.branch_b(mean_s, std_s, max_s)
        fc = self.branch_c(x)
        z  = self.head(torch.cat([fa, fb, fc], dim=1))
        return self.classifier(z)


# =============================================================================
#  TTA
# =============================================================================

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor, n: int = 8) -> torch.Tensor:
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


# =============================================================================
#  DATA
# =============================================================================

def build_splits():
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3,
                               stratify=labels, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.5,
                               stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(train_idx, val_idx, test_idx, batch_train: int,
                  spatial_only: bool = False):
    nw = CONFIG["num_workers"]
    kw = dict(num_workers=nw, pin_memory=True)
    train_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                        train_idx, augment=True, spatial_only=spatial_only),
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


# =============================================================================
#  OPTIMISER
# =============================================================================

def build_optimizer(model: nn.Module, lr: float) -> optim.AdamW:
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


# =============================================================================
#  WARMUP COSINE SCHEDULER (for Stage 2)
# =============================================================================

class WarmupCosineScheduler:
    """Linear warmup then cosine decay. Avoids destabilising freshly reset EMA."""

    def __init__(
        self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr
    ) -> None:
        self.opt     = optimizer
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.base    = base_lr
        self.min     = min_lr
        self._ep     = 0

    def step(self) -> None:
        self._ep += 1
        if self._ep <= self.warmup:
            lr = self.base * self._ep / self.warmup
        else:
            t  = (self._ep - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.opt.param_groups[0]["lr"]


# =============================================================================
#  TRAIN / EVALUATE
# =============================================================================

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, ema, device,
    scheduler=None, use_mixup=True, mixup_alpha=0.4,
) -> tuple[float, float]:
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
def evaluate(model, loader, device, use_tta=False, tta_n=8) -> tuple[float, float]:
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


# =============================================================================
#  CHECKPOINT
# =============================================================================

def save_ckpt(path, epoch, stage, model, ema, val_acc, val_f1):
    torch.save({"epoch": epoch, "stage": stage,
                "model": model.state_dict(), "ema": ema.state_dict(),
                "val_acc": val_acc, "val_f1": val_f1}, path)


def load_ckpt(path, model, ema, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    return ckpt


# =============================================================================
#  STAGE RUNNERS
# =============================================================================

def _hdr(title: str, epochs: int) -> None:
    w = 66
    print(f"\n{'='*w}\n  {title}  [{epochs} epochs max]\n{'='*w}")


def run_stage1(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt) -> float:
    """
    Stage 1: OneCycleLR (per-batch), Mixup+CutMix, full aug.
    Checkpoint on EMA only - stable, not contaminated by Mixup BN stats.
    """
    optimizer = build_optimizer(model, lr=CONFIG["s1_max_lr"] / 25)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CONFIG["s1_max_lr"],
            epochs=CONFIG["s1_epochs"],
            steps_per_epoch=len(train_ldr),
            pct_start=0.15,
            div_factor=25,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )

    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 1 - Heavy Aug + Mixup/CutMix", CONFIG["s1_epochs"])

    for ep in range(1, CONFIG["s1_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=ema, device=device, scheduler=scheduler,
            use_mixup=True, mixup_alpha=CONFIG["s1_mixup"],
        )

        _,  va_live = evaluate(model,      val_ldr, device)
        vf1, va_ema = evaluate(ema.shadow, val_ldr, device)

        lr_now = optimizer.param_groups[0]["lr"]
        ema_d  = ema.current_decay
        saved  = ""

        # EMA-only checkpoint (v6 used max(live,ema) with noisy Live)
        if va_ema > best_acc:
            best_acc, no_improve = va_ema, 0
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema, va_ema, vf1)
            saved = "  ✓"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s1_epochs']} | "
            f"Loss {tl:.4f}  Train {ta:.1%} | "
            f"Live {va_live:.1%}  EMA {va_ema:.1%} | "
            f"LR {lr_now:.2e}  d={ema_d:.4f}{saved}"
        )

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    return best_acc


def run_stage2(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt) -> float:
    """
    Stage 2: Spatial-only augmentation (no Mixup, no spectral aug).

    KEY v7 CHANGES vs v6:
    1. ema.reset(model): copies live weights, resets adaptive decay.
       Fixes 20-epoch EMA lag (EMA dropped 78.7% -> 68.3% in v6).
    2. spatial_only augmentation: keeps flip+rot90 on clean images.
       Reduces 16% train-val gap by ~6-8%.
    3. WarmupCosineScheduler: 5-ep warmup before cosine.
    4. EMA-only checkpoint.
    """
    model.set_dropout(CONFIG["s2_dropout"])
    model.set_drop_path(CONFIG["s2_drop_path"])

    # KEY FIX: Reset EMA from live model
    print("\n  Resetting EMA from live model (num_updates -> 0) ...")
    ema.reset(model)
    ema.set_dropout(CONFIG["s2_dropout"])
    ema.set_drop_path(CONFIG["s2_drop_path"])

    optimizer = build_optimizer(model, lr=CONFIG["s2_lr"] / CONFIG["s2_warmup_ep"])
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=CONFIG["s2_warmup_ep"],
        total_epochs=CONFIG["s2_epochs"],
        base_lr=CONFIG["s2_lr"],
        min_lr=CONFIG["s2_min_lr"],
    )

    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 2 - Spatial Aug Only (no Mixup)", CONFIG["s2_epochs"])

    for ep in range(1, CONFIG["s2_epochs"] + 1):
        scheduler.step()    # update LR before training epoch

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=ema, device=device, scheduler=None,
            use_mixup=False,
        )

        _,  va_live = evaluate(model,      val_ldr, device)
        vf1, va_ema = evaluate(ema.shadow, val_ldr, device)

        lr_now = scheduler.current_lr
        saved  = ""

        if va_ema > best_acc:
            best_acc, no_improve = va_ema, 0
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema, va_ema, vf1)
            saved = "  ✓"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{CONFIG['s2_epochs']} | "
            f"Loss {tl:.4f}  Train {ta:.1%} | "
            f"Live {va_live:.1%}  EMA {va_ema:.1%} | "
            f"LR {lr_now:.2e}{saved}"
        )

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    return best_acc


# =============================================================================
#  FINAL TEST EVALUATION
# =============================================================================

def final_evaluation(model, ema, test_ldr, device, best_ckpt) -> None:
    w = 66
    print(f"\n{'='*w}\n  FINAL TEST EVALUATION\n{'='*w}")

    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow

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

    print(f"\n  Checkpoint: epoch {ckpt['epoch']} | {ckpt['stage']} "
          f"| val={ckpt['val_acc']:.1%}")

    p_tta, t_tta = results["TTA   "]
    print(f"\nClassification Report (TTA):\n")
    print(classification_report(t_tta, p_tta, zero_division=0))

    out = CONFIG["output_dir"]
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy",   p_tta)
    np.save(f"{out}/test_targets.npy",      t_tta)
    print(f"\nOutputs saved -> {out}")


# =============================================================================
#  MAIN
# =============================================================================

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
        drop_path=CONFIG["s1_drop_path"],
        wl_embed_dim=CONFIG["wl_embed_dim"],
        cos_temp=CONFIG["cos_temp"],
    ).to(device)

    ema   = ModelEMA(model, decay=CONFIG["ema_decay"])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : SpectralTripleNet v7")
    print(f"Params : {n_par / 1e6:.2f}M")
    print(f"Device : {device}  |  EMA adaptive -> max {CONFIG['ema_decay']}")
    print(f"Stage 1: {CONFIG['s1_epochs']} ep | Full aug + Mixup  "
          f"| drop={CONFIG['s1_dropout']} dp={CONFIG['s1_drop_path']}")
    print(f"Stage 2: {CONFIG['s2_epochs']} ep | Spatial aug only  "
          f"| drop={CONFIG['s2_dropout']} dp={CONFIG['s2_drop_path']}")
    print(f"Norm   : GroupNorm (no BatchNorm1d in spectral branches)")
    print(f"Head   : CosineClassifier (temp={CONFIG['cos_temp']})")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # Stage 1
    train_ldr, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s1_batch"], spatial_only=False
    )
    run_stage1(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt)

    # Stage 2
    print("\nLoading Stage 1 best checkpoint for Stage 2 ...")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  EMA val={ckpt['val_acc']:.1%}  ({ckpt['stage']})")
    print(f"  Dropout: {CONFIG['s1_dropout']} -> {CONFIG['s2_dropout']}")
    print(f"  DropPath: {CONFIG['s1_drop_path']} -> {CONFIG['s2_drop_path']}")

    train_ldr, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"], spatial_only=True
    )
    run_stage2(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt)

    # Test
    _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx, 64)
    final_evaluation(model, ema, test_ldr, device, best_ckpt)


if __name__ == "__main__":
    main()