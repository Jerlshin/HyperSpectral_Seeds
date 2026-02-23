from __future__ import annotations

import copy
import math
import os
import random
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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

# Physical wavelength range — Specim V10E sensor (385–1000 nm)
WL_MIN: float = 385.0
WL_MAX: float = 1000.0


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    # Paths
    "patches_data":   "./dataset/patches.npy",
    "labels_path":    "./dataset/labels.npy",
    "output_dir":     "./output_v9/",

    # Dataset
    "num_bands":      256,
    "num_classes":    90,

    # Stage 1 — Heavy aug + Mixup/CutMix
    "s1_epochs":      150,
    "s1_batch":       64,
    "s1_max_lr":      8e-4,
    "s1_dropout":     0.30,
    "s1_mixup":       0.4,
    "s1_patience":    35,
    "s1_accum":       2,      # gradient accumulation steps → effective batch=128

    # Stage 2 — ArcFace + ProtoNCE + balanced batches
    "s2_epochs":      80,
    "s2_batch":       64,
    "s2_warmup_ep":   5,      # linear warmup epochs
    "s2_peak_lr":     1.2e-4, # peak LR after warmup
    "s2_min_lr":      1e-7,
    "s2_dropout":     0.10,
    "s2_patience":    25,
    "s2_arcface_s":   20.0,
    "s2_arcface_m":   0.40,   # final margin (warmed up from 0.05)
    "s2_arcface_m0":  0.05,   # initial margin
    "s2_margin_warmup_ep": 30, # epochs to ramp margin to s2_arcface_m

    # ProtoNCE (Stage 2)
    "proto_weight":   0.30,   # α: ProtoNCE contribution  (CE is 1-α)
    "proto_temp":     0.07,

    # Class-balanced sampler (Stage 2)
    "bal_n_cls":      16,
    "bal_n_spc":      4,      # batch = 16×4 = 64

    # Stage 3 — Manual SWA
    "s3_epochs":      25,
    "s3_swa_lr":      8e-5,
    "s3_cycle_len":   5,

    # Loss
    "label_smoothing": 0.05,

    # Regularisation
    "weight_decay":   2e-4,
    "grad_clip":      1.0,

    # EMA
    "ema_decay":      0.9999,

    # TTA
    "tta_n":          8,

    # Wavelength embedding (Branches A/B)
    "wl_embed_dim":   16,

    # SpecFormer (Branch D)
    "specf_patch":    8,
    "specf_dim":      128,
    "specf_heads":    4,
    "specf_layers":   4,
    "specf_drop":     0.15,

    # BranchCrossAttention (Fusion)
    "fusion_heads":   4,
    "fusion_drop":    0.10,

    # Misc
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
#  ADAPTIVE EMA
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Adaptive EMA: decay ramps from ~0 → max_decay over training steps.
    Formula: d(n) = min(max_decay, (1+n)/(10+n))
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
        d = self.current_decay

        # ── Parameters: EMA smoothing ─────────────────────────────────
        live_params   = dict(model.named_parameters())
        for name, s_p in self.shadow.named_parameters():
            if name in live_params:
                s_p.copy_(d * s_p + (1.0 - d) * live_params[name])

        # ── Buffers (BN running stats): direct copy  [BUG 1 FIX] ──────
        # BN running_mean/running_var are already EMA internally.
        # Applying EMA-of-EMA would over-smooth; just copy the live stats.
        live_buffers = dict(model.named_buffers())
        for name, s_b in self.shadow.named_buffers():
            if name in live_buffers and s_b.dtype.is_floating_point:
                s_b.copy_(live_buffers[name])

    def reinit_from(self, model: nn.Module) -> None:
        """
        Re-copy live model weights + buffers into shadow; reset step counter.
        Called at Stage 2 start so EMA adapts from the correct base weights.
        """
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def state_dict(self)                -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET  (spectral warp / shift / mult-noise — unchanged from v3)
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """
    Memory-mapped HSI loader.

    augment=True applies (in order):
      band_dropout   – zero out random bands
      band_cutout    – zero-out a contiguous spectral window
      spectral_noise – additive Gaussian per-pixel-per-band
      spectral_warp  – random linear stretch/compress ±10%
      spectral_shift – circular band shift ±8 positions
      mult_noise     – per-band multiplicative intensity variation ±5%
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

    def _spectral_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Random linear stretch/compress of the spectral axis ±10%."""
        C, H, W = x.shape
        scale   = 1.0 + random.uniform(-0.10, 0.10)
        new_C   = max(1, int(C * scale))
        if new_C == C:
            return x
        x_perm = x.permute(1, 2, 0).reshape(-1, 1, C)
        warped  = F.interpolate(x_perm, size=new_C, mode="linear",
                                align_corners=False)
        if new_C > C:
            start  = (new_C - C) // 2
            warped = warped[:, :, start:start + C]
        else:
            pad_lo = (C - new_C) // 2
            pad_hi = C - new_C - pad_lo
            warped = F.pad(warped, (pad_lo, pad_hi))
        return warped.reshape(H, W, C).permute(2, 0, 1)

    def _spectral_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Circular shift by ±8 bands — simulates wavelength axis offset."""
        shift = random.randint(-8, 8)
        return torch.roll(x, shift, dims=0)

    def _mult_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Per-band multiplicative noise ±5% — non-uniform illumination."""
        scale = 1.0 + torch.randn(x.shape[0], 1, 1) * 0.05
        return x * scale

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
#  CLASS-BALANCED BATCH SAMPLER  (Stage 2 — for ProtoNCE)
# ══════════════════════════════════════════════════════════════════════

class ClassBalancedBatchSampler(Sampler):
    """
    Each batch: n_cls randomly-selected classes × n_spc samples each.
    batch_size = n_cls × n_spc  (e.g., 16 × 4 = 64).

    Guarantees n_spc-1 = 3 in-class neighbours per anchor, which is
    the minimum needed for meaningful prototypical or contrastive loss.
    """

    def __init__(self, train_labels: np.ndarray,
                 n_cls: int = 16, n_spc: int = 4) -> None:
        self.n_cls   = n_cls
        self.n_spc   = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0]
                        for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen_cls = rng.choice(self.classes, self.n_cls, replace=False)
            batch: List[int] = []
            for c in chosen_cls:
                pool    = self.cls_idx[c]
                replace = len(pool) < self.n_spc
                samp    = rng.choice(pool, self.n_spc, replace=replace)
                batch.extend(samp.tolist())
            yield batch

    def __len__(self) -> int:
        return self._n


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION  (Mixup + CutMix)
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam      = float(np.random.beta(alpha, alpha))
    idx      = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _cutmix(x, y, alpha):
    lam       = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx       = torch.randperm(B, device=x.device)
    r         = math.sqrt(1.0 - lam)
    ch, cw    = int(H * r), int(W * r)
    cx        = random.randint(0, W)
    cy        = random.randint(0, H)
    x1 = max(cx - cw // 2, 0); x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0); y2 = min(cy + ch // 2, H)
    x_mix     = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam       = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_mix, y, y[idx], lam


def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1).item() < 0.5 else _cutmix)(x, y, alpha)


def mixed_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ══════════════════════════════════════════════════════════════════════
#  ARCFACE HEAD  (with per-call margin for warmup support)
# ══════════════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Softmax.

    The margin `m` is passed at call time so the training loop can implement
    a warmup schedule without re-building the head.

    Ref: Deng et al. ArcFace, CVPR 2019.
    """

    def __init__(self, in_dim: int, num_classes: int,
                 s: float = 20.0, m: float = 0.40) -> None:
        super().__init__()
        self.weight   = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s        = s
        self.default_m = m
        self._precompute(m)

    def _precompute(self, m: float) -> None:
        self._m    = m
        self._cosm = math.cos(m)
        self._sinm = math.sin(m)
        self._th   = math.cos(math.pi - m)
        self._mm   = math.sin(math.pi - m) * m

    def forward(
        self,
        x:      torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        m:      Optional[float]        = None,
    ) -> torch.Tensor:
        if m is not None and m != self._m:
            self._precompute(m)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))  # (B, C)
        cosine = cosine.clamp(-1+1e-6, 1-1e-6)
        
        if labels is None or not self.training:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-6))
        phi   = cosine * self._cosm - sine * self._sinm
        phi   = torch.where(cosine > self._th, phi, cosine - self._mm)
        oh    = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi) + ((1.0 - oh) * cosine)) * self.s


# ══════════════════════════════════════════════════════════════════════
#  PROTO-NCE LOSS  [NEW — replaces SupCon]
# ══════════════════════════════════════════════════════════════════════

class ProtoNCELoss(nn.Module):
    """
    Prototypical NCE Loss.

    For each batch (produced by ClassBalancedBatchSampler with n_spc=4
    per class), compute per-class prototype = mean of same-class L2-
    normalised embeddings.  Then apply cross-entropy over prototype-
    distance logits: each sample is pulled toward its class prototype
    and pushed away from all others.

    Advantages over within-batch SupCon:
      1. Gradient is cleaner — prototype averages reduce noise from
         individual hard/easy pair interactions.
      2. Each sample contributes to n_cls logit dimensions instead of
         n_pos pairs, giving a stronger learning signal per step.
      3. Numerically more stable with small n_spc (≥ 2 sufficient).

    Ref: Snell et al. Prototypical Networks, NeurIPS 2017.
         Li et al. Prototypical Contrastive Learning, EMNLP 2021.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (B, D)  L2-normalised embeddings
        labels   : (B,)    integer class labels
        """
        device  = features.device
        classes = labels.unique()
        C       = len(classes)

        if C < 2:
            return features.new_tensor(0.0, requires_grad=True)

        # Compute per-class prototypes  (C_batch, D)
        protos = torch.stack([
            features[labels == c].mean(0) for c in classes
        ])
        protos = F.normalize(protos, dim=1)     # unit-sphere prototypes

        # Pairwise cosine similarities  (B, C_batch)
        sim    = torch.mm(features, protos.T) / self.temperature

        # Remap original class labels to local [0, C_batch) indices
        class_to_local = {c.item(): i for i, c in enumerate(classes)}
        local_labels   = torch.tensor(
            [class_to_local[y.item()] for y in labels],
            dtype=torch.long, device=device,
        )

        return F.cross_entropy(sim, local_labels)


# ══════════════════════════════════════════════════════════════════════
#  MASKED SPECTRAL STATISTICS  (float32, NaN-safe)
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute foreground-masked mean / std / max per spectral band."""
    x32     = x.float()
    B, C, H, W = x32.shape
    flat    = x32.reshape(B, C, H * W)

    energy  = flat.abs().sum(dim=1, keepdim=True)
    mask    = (energy > 1e-5).float()
    count   = mask.sum(dim=2).clamp(min=1.0)

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
    Sinusoidal encoding of physical wavelengths.
    Tells 1-D CNNs where chlorophyll (~680 nm), water (~970 nm),
    starch (~860 nm) and protein (~930 nm) absorption features live.
    """

    def __init__(self, num_bands: int = 256, embed_dim: int = 16) -> None:
        super().__init__()
        wl    = torch.linspace(0.0, 1.0, num_bands)
        half  = embed_dim // 2
        freq  = torch.exp(
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
        """Returns (1, 1, num_bands) — broadcast-adds to (B, C, 256) inputs."""
        return self.proj(self.enc).squeeze(-1).view(1, 1, -1)


# ══════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class SpectralSE(nn.Module):
    """Per-sample spectral Squeeze-and-Excitation over 256 bands."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid       = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid,      bias=False), nn.GELU(),
            nn.Linear(mid,      channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x.mean(dim=[2, 3]))
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    """1-D residual block for spectral sequences."""

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
#  BRANCH A — SPECTRAL PROFILE
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    """
    Multi-scale 1-D CNN on mean spectrum + first-order derivative.
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
            nn.BatchNorm1d(out_dim),
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
#  BRANCH B — SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    """
    Multi-scale 1-D CNN on {mean, std, max} spectral statistics.
    std encodes intra-seed spectral heterogeneity (hull vs. endosperm).
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
            nn.BatchNorm1d(out_dim),
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
    2-D residual CNN + CBAM attention on band-reduced spatial volume.
    Power-normalised pooling (sign(x)|x|^0.5) approximates second-order
    statistics — dampens outlier activations.
    """

    def __init__(self, num_bands: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stages    = nn.Sequential(
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

    @staticmethod
    def _power_norm(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().clamp(min=1e-8).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = self.band_reduce(x)
        h    = self.stages(h)
        avg  = self.avg_pool(h).flatten(1)
        mx   = self.max_pool(h).flatten(1)
        feat = torch.cat([self._power_norm(avg), self._power_norm(mx)], dim=1)
        feat = F.normalize(feat, dim=1)
        return self.pool_proj(feat)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER  (lightweight spectral transformer)
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
    256 bands → 32 patches × 8 bands → (B,33,128) with [CLS] → 256-D.
    Physical wavelength PE at patch centres.
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
        n_patches       = num_bands // patch_size
        self.patch_size = patch_size
        self.n_patches  = n_patches

        self.patch_proj = nn.Sequential(
            nn.Linear(patch_size, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        wl_centers = torch.linspace(WL_MIN, WL_MAX, n_patches)
        wl_norm    = (wl_centers - WL_MIN) / (WL_MAX - WL_MIN)
        half       = d_model // 2
        freq       = torch.exp(
            torch.arange(half).float() * -(math.log(1e4) / max(half - 1, 1))
        )
        pe = torch.zeros(n_patches, d_model)
        pe[:, :half] = torch.sin(wl_norm.unsqueeze(1) * freq.unsqueeze(0))
        pe[:, half:] = torch.cos(wl_norm.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("wl_pe", pe)

        self.cls    = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.blocks = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, mean_spec: torch.Tensor) -> torch.Tensor:
        B   = mean_spec.shape[0]
        x   = mean_spec.float().view(B, self.n_patches, self.patch_size)
        x   = self.patch_proj(x) + self.wl_pe.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x[:, 0])


# ══════════════════════════════════════════════════════════════════════
#  BRANCH CROSS-ATTENTION FUSION  [NEW]
# ══════════════════════════════════════════════════════════════════════

class BranchCrossAttention(nn.Module):
    """
    Cross-branch attention fusion for four 256-D branch outputs.

    Treats each branch's feature vector as one "token" in a 4-token
    sequence, then applies a single Pre-LN transformer block so each
    branch can attend to and modulate the others.

    Motivation:
      Simple concatenation treats branches independently.  In fine-
      grained 90-class variety discrimination, synergies exist:
        • The spatial branch detects hull texture → should amplify the
          spectral absorption bands that correlate with hull composition.
        • SpecFormer detects starch/protein correlation → should inform
          what statistical moments (Branch B) to emphasise.
      Cross-attention discovers these interactions end-to-end.

    Architecture:
      input:  list of 4 × (B, 256) feature vectors
      stack:  (B, 4, 256) — 4 tokens of dim 256
      transformer: 1 Pre-LN block, 4 heads, d_ff=512, drop=0.10
      output: (B, 4, 256) → flatten → (B, 1024)

    Parameter overhead: ~160 K.
    Ref: "Cross-Modal Attention for Fine-Grained Visual Classification", 2021.
    """

    def __init__(self, d: int = 256, n_heads: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 2, d), nn.Dropout(dropout),
        )
        self.drop  = nn.Dropout(dropout)
        # Learnable per-branch residual scale (init=1 → neutral start)
        self.gate  = nn.Parameter(torch.ones(1))

    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        """
        branches: list of (B, d) tensors — one per branch
        returns : (B, n_branches * d)
        """
        x = torch.stack(branches, dim=1)         # (B, n_branches, d)
        # Pre-LN self-attention across the n_branches tokens
        h    = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x    = x + self.gate * self.drop(h)      # learnable residual scale
        x    = x + self.drop(self.ff(self.norm2(x)))
        return x.flatten(1)                      # (B, n_branches * d)


# ══════════════════════════════════════════════════════════════════════
#  MAIN MODEL — SpectralQuadNet v8
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    """
    Four-branch HSI classification network with cross-branch fusion.

    Branch A : SpectralProfileBranch  (mean + deriv + WL PE)      → 256-D
    Branch B : SpectralStatsBranch    (mean + std + max + WL)      → 256-D
    Branch C : SpatialCNNBranch       (2D CNN + power norm)        → 256-D
    Branch D : SpecFormerBranch       (spectral transformer)       → 256-D

    Fusion  : BranchCrossAttention(A,B,C,D) → 1024-D  [NEW v8]
                → Linear(512) → BN → GELU → Dropout
                → Linear(256) → BN                      (embedding)

    Stage 1 head : Linear(256, 90)                 [Mixup-compatible CE]
    Stage 2/3 head : ArcFaceHead(256, 90)          [ArcFace CE]

    HEAD FREEZE STRATEGY  [BUG 3 FIX]:
      Stage 1: arcface_head is FROZEN (no grad, no weight decay)
      Stage 2: linear_head  is FROZEN (no grad, no weight decay)
      Prevents cross-contamination and weight-decay corruption of the
      inactive head between stages.

    _use_arcface is saved/restored via checkpoint dict  [BUG 2 FIX].
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

        self.se     = SpectralSE(num_bands, reduction=16)
        self.wl_enc = WavelengthPositionalEncoding(num_bands, wl_embed_dim)

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

        # [NEW] Cross-branch attention fusion before MLP
        self.cross_attn = BranchCrossAttention(
            d       = 256,
            n_heads = cfg["fusion_heads"],
            dropout = cfg["fusion_drop"],
        )

        fusion_dim = 256 * 4   # 1024

        self.embed_net = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
        )

        # Stage 1 head
        self.linear_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(256, num_classes),
        )

        # Stage 2/3 head
        self.arcface_head = ArcFaceHead(
            256, num_classes,
            s=cfg["s2_arcface_s"],
            m=cfg["s2_arcface_m"],
        )

        self._use_arcface = False
        self._init_weights()

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

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def use_arcface(self, flag: bool = True) -> None:
        self._use_arcface = flag

    def freeze_head(self, which: str) -> None:
        """Freeze 'linear' or 'arcface' head — excluded from optimizer."""
        head = self.linear_head if which == "linear" else self.arcface_head
        for p in head.parameters():
            p.requires_grad_(False)

    def unfreeze_head(self, which: str) -> None:
        head = self.linear_head if which == "linear" else self.arcface_head
        for p in head.parameters():
            p.requires_grad_(True)

    def forward(
        self,
        x:            torch.Tensor,
        labels:       Optional[torch.Tensor] = None,
        return_embed: bool                   = False,
        arc_m:        Optional[float]        = None,
    ) -> torch.Tensor:

        x = self.se(x)                                     # (B,256,64,64)
        ms, ss, mx = masked_spectral_stats(x)              # each (B,256)

        fa = self.branch_a(ms)                             # (B,256)
        fb = self.branch_b(ms, ss, mx)                    # (B,256)
        fc = self.branch_c(x)                              # (B,256)
        fd = self.branch_d(ms)                             # (B,256)

        # [NEW] cross-branch attention before MLP fusion
        fused = self.cross_attn([fa, fb, fc, fd])          # (B,1024)
        emb   = self.embed_net(fused)                      # (B,256)

        if self._use_arcface:
            emb_n  = F.normalize(F.gelu(emb), dim=1)
            logits = self.arcface_head(emb_n, labels, m=arc_m)
        else:
            logits = self.linear_head(emb)

        if return_embed:
            return logits, F.normalize(F.gelu(emb.detach()), dim=1)

        return logits


# ══════════════════════════════════════════════════════════════════════
#  TTA
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
    train_idx:   np.ndarray,
    val_idx:     np.ndarray,
    test_idx:    np.ndarray,
    batch_train: int,
    balanced:    bool                  = False,
    all_labels:  Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
#  OPTIMISER  (weight decay skips BN, GroupNorm, biases)
# ══════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr: float) -> optim.AdamW:
    """
    Only optimise parameters that require grad (respects freeze_head()).
    Weight decay skips 1-D params (BN/GN scale/bias) and bias terms.
    """
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
#  ARCFACE MARGIN SCHEDULE
# ══════════════════════════════════════════════════════════════════════

def arcface_margin_schedule(
    epoch:     int,
    m0:        float,
    m_target:  float,
    warmup_ep: int,
) -> float:
    """
    Cosine warmup of ArcFace margin from m0 → m_target over warmup_ep.
    After warmup, holds at m_target.

    Motivation: the full margin at epoch 1 immediately suppresses the
    correct-class cosine below all negatives (train acc = 0% for first
    12 epochs in v3).  Starting from a small margin m0=0.05 and ramping
    up lets the head first establish a sensible directional layout,
    then sharpen it with the full margin.
    """
    if epoch >= warmup_ep:
        return m_target
    frac = epoch / max(warmup_ep, 1)
    # Cosine ramp: smooth and avoids discontinuities
    cos_val = 0.5 * (1.0 - math.cos(math.pi * frac))
    return m0 + (m_target - m0) * cos_val


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    optimizer:     optim.Optimizer,
    criterion:     nn.Module,
    scaler:        Optional[GradScaler],
    ema:           Optional[ModelEMA],
    device:        torch.device,
    scheduler                          = None,
    use_mixup:     bool                = True,
    mixup_alpha:   float               = 0.4,
    supcon:        Optional[nn.Module] = None,
    supcon_weight: float               = 0.0,
    accum_steps:   int                 = 1,
    arc_m:         Optional[float]     = None,
) -> Tuple[float, float]:
    """
    Unified training loop for all stages.

    Stage 1: Mixup + CE + AMP + Accum
    Stage 2: ArcFace + ProtoNCE (FP32, no AMP)
    Stage 3: CE (AMP)

    This version fixes:
      - AMP instability in Stage 2
      - Scheduler/optimizer ordering
      - Gradient scaling when AMP disabled
      - Partial-step gradient leaks
      - NaN propagation
    """

    model.train()

    total_loss = 0.0
    total_acc  = 0.0

    optimizer.zero_grad(set_to_none=True)

    # AMP only when no metric learning is used
    use_amp = (supcon is None) and (scaler is not None)

    for step, (x, y) in enumerate(loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_mixup:
            x_in, y_a, y_b, lam = mixed_aug(x, y, mixup_alpha)
        else:
            x_in, y_a, y_b, lam = x, y, y, 1.0

        with autocast(device_type=device.type, enabled=use_amp):

            if supcon is not None and not use_mixup:
                # Stage 2: ArcFace + ProtoNCE
                logits, emb = model(
                    x_in,
                    y_a,
                    return_embed=True,
                    arc_m=arc_m,
                )

                loss_ce = criterion(logits, y_a)
                loss_sc = supcon(emb, y_a)

                loss = (
                    (1.0 - supcon_weight) * loss_ce +
                    supcon_weight         * loss_sc
                )

            else:
                # Stage 1 / 3
                logits = model(
                    x_in,
                    labels=y_a if (model._use_arcface and not use_mixup) else None,
                    arc_m=arc_m,
                )

                loss = mixed_loss(
                    criterion, logits, y_a, y_b, lam
                )

        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at step {step}. Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = loss / accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG["grad_clip"]
            )
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * accum_steps

        with torch.no_grad():
            total_acc += (
                logits.argmax(1) == y
            ).float().mean().item()


    n = len(loader)

    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
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
#  CHECKPOINT HELPERS  [BUG 2 FIX: saves/restores _use_arcface flag]
# ══════════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, stage, model, ema, val_acc, val_f1):
    """
    Save checkpoint including the _use_arcface flag.
    """
    torch.save({
        "epoch":       epoch,
        "stage":       stage,
        "model":       model.state_dict(),
        "ema":         ema.state_dict(),
        "val_acc":     val_acc,
        "val_f1":      val_f1,
        "use_arcface": model._use_arcface,   # ← BUG 2 FIX
    }, path)


def load_ckpt(path, model, ema, device):
    """
    Load checkpoint and restore _use_arcface on both model and ema.shadow.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])

    # ── BUG 2 FIX: restore arcface flag ───────────────────────────────
    use_af = ckpt.get("use_arcface", False)
    model.use_arcface(use_af)
    ema.shadow.use_arcface(use_af)      # ← critical: fixes final_evaluation
    return ckpt


# ══════════════════════════════════════════════════════════════════════
#  BN UPDATE (for SWA)
# ══════════════════════════════════════════════════════════════════════

def update_bn_stats(loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    """
    Recompute BN running mean/var for averaged (SWA) weights.
    Uses cumulative moving average (momentum=None).
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None

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
    Stage 1: Heavy aug + Mixup/CutMix + OneCycleLR + gradient accumulation.
    """
    model.use_arcface(False)
    model.unfreeze_head("linear")
    model.freeze_head("arcface")        # ← BUG 3 FIX

    optimizer = build_optimizer(model, lr=CONFIG["s1_max_lr"] / 25)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr           = CONFIG["s1_max_lr"],
            epochs           = CONFIG["s1_epochs"],
            steps_per_epoch  = math.ceil(len(train_ldr) / CONFIG["s1_accum"]),
            pct_start        = 0.15,
            div_factor       = 25,
            final_div_factor = 1e4,
            anneal_strategy  = "cos",
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
            accum_steps=CONFIG["s1_accum"],
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

    # Unfreeze arcface_head so it's available for Stage 2
    model.unfreeze_head("arcface")
    return best_acc


# ── Stage 2 ───────────────────────────────────────────────────────────

def run_stage2(
    model, ema, train_ldr, val_ldr, device, criterion, best_ckpt,
) -> float:
    """
    Stage 2: ArcFace CE + ProtoNCE + class-balanced batches.
    """
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)
    model.freeze_head("linear")         # ← BUG 3 FIX
    model.unfreeze_head("arcface")

    ema.reinit_from(model)
    ema.set_dropout(CONFIG["s2_dropout"])
    ema.shadow.use_arcface(True)        # ← BUG 2 FIX (shadow also uses arcface)

    proto   = ProtoNCELoss(temperature=CONFIG["proto_temp"])
    optimizer = build_optimizer(model, lr=1e-5)   # warmup starts near 0

    # Linear warmup + cosine decay
    warmup_steps = CONFIG["s2_warmup_ep"] * len(train_ldr)
    total_steps  = CONFIG["s2_epochs"]   * len(train_ldr)
    peak_lr      = CONFIG["s2_peak_lr"]
    min_lr       = CONFIG["s2_min_lr"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return (min_lr / peak_lr) + 0.5 * (1 - min_lr / peak_lr) * (
            1 + math.cos(math.pi * t)
        )

    scheduler  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Override base LR so lambda is relative to peak_lr
    for pg in optimizer.param_groups:
        pg["lr"] = peak_lr

    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 2 — ArcFace + ProtoNCE + Balanced Batches", CONFIG["s2_epochs"])

    for ep in range(1, CONFIG["s2_epochs"] + 1):
        # ArcFace margin warmup
        m_now = arcface_margin_schedule(
            ep - 1,
            CONFIG["s2_arcface_m0"],
            CONFIG["s2_arcface_m"],
            CONFIG["s2_margin_warmup_ep"],
        )

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=ema, device=device, scheduler=scheduler,
            use_mixup=False,
            supcon=proto,
            supcon_weight=CONFIG["proto_weight"],
            arc_m=m_now,
        )

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
            f"LR {lr_now:.2e}  m={m_now:.3f}{saved}"
        )

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    # Unfreeze linear_head for Stage 3 (SWA evaluates both)
    model.unfreeze_head("linear")
    return best_acc


# ── Stage 3 (SWA) ─────────────────────────────────────────────────────

def run_stage3_swa(
    model, ema, train_ldr, val_ldr, device, criterion, best_ckpt,
) -> float:
    """
    Stage 3: Manual Stochastic Weight Averaging.

    Runs s3_epochs epochs with cosine-cyclic LR; accumulates model
    snapshots at the end of each cycle; updates BN after averaging.
    """
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)

    optimizer = build_optimizer(model, lr=CONFIG["s3_swa_lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["s3_cycle_len"],
        eta_min=CONFIG["s3_swa_lr"] * 0.1,
    )
    scaler = GradScaler()

    # Accumulator initialised from Stage-2 best weights
    swa_state: dict = copy.deepcopy(model.state_dict())
    n_snap          = 1

    _hdr("Stage 3 — Stochastic Weight Averaging", CONFIG["s3_epochs"])

    for ep in range(1, CONFIG["s3_epochs"] + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler,
            ema=None, device=device, scheduler=None,
            use_mixup=False,
        )
        scheduler.step()

        if ep % CONFIG["s3_cycle_len"] == 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["s3_cycle_len"],
                eta_min=CONFIG["s3_swa_lr"] * 0.1,
            )
            n_snap += 1
            alpha   = 1.0 / n_snap
            curr_sd = model.state_dict()
            for k in swa_state:
                swa_state[k] = swa_state[k] + alpha * (curr_sd[k] - swa_state[k])

        _, va_live = evaluate(model, val_ldr, device)
        lr_now     = optimizer.param_groups[0]["lr"]
        print(
            f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ "
            f"Loss {tl:.4f}  Train {ta:.1%} │ "
            f"Live {va_live:.1%} │ LR {lr_now:.2e} │ Snaps {n_snap}"
        )

    # BN update for SWA model
    print(f"\nUpdating BN statistics for SWA model ({n_snap} snapshots) ...")
    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state)
    swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)

    _, va_swa = evaluate(swa_model, val_ldr, device)
    _, va_ema = evaluate(ema.shadow, val_ldr, device)
    print(f"SWA val: {va_swa:.1%}   EMA val: {va_ema:.1%}")

    # Select best model for final evaluation
    if va_swa >= va_ema:
        print("Using SWA model as final eval model.")
        ema.shadow.load_state_dict(swa_model.state_dict())
        ema.shadow.use_arcface(True)
        best_val = va_swa
    else:
        print("EMA model retained as final eval model.")
        best_val = va_ema

    # ── BUG 4 FIX: save SWA result to best_ckpt ───────────────────────
    # Load the previously saved best to compare; if SWA is better, overwrite.
    prev_ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    if best_val > prev_ckpt.get("val_acc", 0.0):
        print(f"SWA val {best_val:.1%} > Stage-2 best {prev_ckpt['val_acc']:.1%} → saving.")
        save_ckpt(best_ckpt, CONFIG["s3_epochs"], "Stage 3",
                  swa_model, ema, best_val, 0.0)

    return best_val


# ══════════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION  [BUG 2 FIX: arcface flag restored by load_ckpt]
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt) -> None:
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")

    # load_ckpt now restores _use_arcface on both model and ema.shadow
    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow
    eval_model.eval()

    # Sanity check — emit which head is active
    print(f"  ArcFace head active: {eval_model._use_arcface}")
    print(f"  Checkpoint: epoch {ckpt['epoch']} | {ckpt['stage']} "
          f"| val={ckpt['val_acc']:.1%}")

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
        acc          = accuracy_score(t, p)
        f1m          = f1_score(t, p, average="macro",    zero_division=0)
        f1w          = f1_score(t, p, average="weighted", zero_division=0)
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

    print(f"\nModel  : SpectralQuadNet v8")
    print(f"Params : {n_par / 1e6:.2f}M")
    print(f"Device : {device}  |  EMA adaptive → max {CONFIG['ema_decay']}")
    print(f"Stage 1: {CONFIG['s1_epochs']} ep | Mixup+CutMix | linear head "
          f"| drop={CONFIG['s1_dropout']} | accum={CONFIG['s1_accum']}")
    print(f"Stage 2: {CONFIG['s2_epochs']} ep | ArcFace+ProtoNCE | balanced smp "
          f"| drop={CONFIG['s2_dropout']} | m warmup {CONFIG['s2_arcface_m0']}→{CONFIG['s2_arcface_m']}")
    print(f"Stage 3: {CONFIG['s3_epochs']} ep | SWA ({CONFIG['s3_cycle_len']}-ep cycles)")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # ── Stage 1: random sampler, Mixup, linear head ───────────────────
    train_ldr1, val_ldr, test_ldr = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s1_batch"]
    )
    run_stage1(model, ema, train_ldr1, val_ldr, device, criterion, best_ckpt)

    # ── Load Stage 1 best → Stage 2 ──────────────────────────────────
    print("\nLoading Stage 1 best checkpoint for Stage 2 ...")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  val={ckpt['val_acc']:.1%}  ({ckpt['stage']})")
    print(f"  Dropout: {CONFIG['s1_dropout']} → {CONFIG['s2_dropout']}")

    # ── Stage 2: balanced sampler, ArcFace, ProtoNCE ─────────────────
    train_ldr2, val_ldr2, _ = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"],
        balanced=True, all_labels=all_labels,
    )
    run_stage2(model, ema, train_ldr2, val_ldr2, device, criterion, best_ckpt)

    # ── Load Stage 2 best → Stage 3 ──────────────────────────────────
    print("\nLoading Stage 2 best checkpoint for Stage 3 (SWA) ...")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    print(f"  epoch={ckpt['epoch']}  val={ckpt['val_acc']:.1%}  ({ckpt['stage']})")

    # ── Stage 3: SWA ─────────────────────────────────────────────────
    train_ldr3, val_ldr3, _ = build_loaders(
        train_idx, val_idx, test_idx, CONFIG["s2_batch"]
    )
    run_stage3_swa(model, ema, train_ldr3, val_ldr3, device, criterion, best_ckpt)

    # ── Final evaluation ──────────────────────────────────────────────
    _, _, test_ldr_final = build_loaders(train_idx, val_idx, test_idx, 64)
    final_evaluation(model, ema, test_ldr_final, device, best_ckpt)


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback, sys, logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(CONFIG["output_dir"], "training.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        main()
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)