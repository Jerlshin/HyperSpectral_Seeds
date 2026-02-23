#!/usr/bin/env python3
"""
hsi_optuna_tune.py  —  SpectralQuadNet v8  |  Research-Grade Optuna HPO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Objective : Maximize macro-F1 on the 90-class Rice HSI validation set.

What is tuned
─────────────
  Architecture
    • wl_embed_dim, specf_patch, specf_dim, specf_heads, specf_layers,
      specf_drop, fusion_heads, fusion_drop, tower_ch (branches A/B)
  Augmentation
    • Heavy/Light aug probabilities per transform (band_drop, cutout,
      noise, warp, shift, mult), noise_std, max_cutout_bands
  Stage 1
    • batch, max_lr, dropout, mixup_alpha, label_smoothing, weight_decay,
      grad_clip, accum_steps, warmup_pct, ema_decay
  Stage 2
    • batch, head_lr, lr_ratio (head/backbone), min_lr_frac, warmup_ep,
      dropout, arcface_s, arcface_m, arcface_m0, margin_warmup_ep,
      label_smooth, proto_weight, proto_temp, bal_n_cls, bal_n_spc
  Stage 3
    • swa_lr, cycle_len, epochs

Search strategy
───────────────
  Sampler  : TPESampler (Multivariate=True, warm_start=True)
  Pruner   : HyperbandPruner (min_resource=10, reduction_factor=3)
  Trials   : configurable (default 60)
  Epochs   : scaled-down per trial (tuning_s1_epochs / s2 / s3 in TUNE_CFG)
             → full-length run can be triggered with TUNE_CFG["full_run"]=True

  Each trial runs all 3 stages end-to-end (same resume-safe structure as
  the original training script) so stage interactions are captured.

  Intermediate F1 values reported after every stage for Hyperband pruning.

Output
──────
  optuna_study.db   — SQLite study (persistent, resumable)
  optuna_best.json  — Best trial params + final macro-F1
  optuna_best.pth   — Model checkpoint of best trial
  optuna_history.csv

Usage
─────
  python hsi_optuna_tune.py                    # run 60 trials
  python hsi_optuna_tune.py --n-trials 120     # more exploration
  python hsi_optuna_tune.py --full-run         # full epoch budget
  python hsi_optuna_tune.py --study-name v2    # resume / new study name
  python hsi_optuna_tune.py --n-jobs 1         # parallelism (GPU advised = 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_CFG: dict = {
    "patches_data": "./dataset/patches.npy",
    "labels_path":  "./dataset/labels.npy",
    "output_dir":   "./optuna_output/",
    "num_bands":    256,
    "num_classes":  90,
    "device":       torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":         42,
    "num_workers":  6,
}

# ─── Epoch budget for tuning trials (increase for full run) ───────────────────
TUNE_CFG: dict = {
    # Reduced epoch budgets per trial  (fast exploration)
    "s1_epochs":   60,   # original 200  →  use 60 for sweeps
    "s2_epochs":   30,   # original 100  →  use 30
    "s3_epochs":   12,   # original  40  →  use 12
    # Patience fractions (relative to tuning epochs)
    "s1_patience_frac": 0.55,   # stop if no improve for 55% of s1 epochs
    "s2_patience_frac": 0.55,
    # Whether to keep trial checkpoints (True → large disk use)
    "keep_trial_ckpts": False,
}

WL_MIN: float = 385.0
WL_MAX: float = 1000.0

Path(BASE_CFG["output_dir"]).mkdir(parents=True, exist_ok=True)


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
        d = self.current_decay
        live_params = dict(model.named_parameters())
        for name, s_p in self.shadow.named_parameters():
            if name in live_params:
                s_p.copy_(d * s_p + (1.0 - d) * live_params[name])
        live_buffers = dict(model.named_buffers())
        for name, s_b in self.shadow.named_buffers():
            if name in live_buffers and s_b.dtype.is_floating_point:
                s_b.copy_(live_buffers[name])

    def reinit_from(self, model: nn.Module) -> None:
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def state_dict(self)                -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET  (parameterised aug profiles)
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """Aug profiles injected from trial config rather than hard-coded."""

    def __init__(
        self,
        patches_path:     str,
        labels_path:      str,
        indices:          np.ndarray,
        aug_profile:      Optional[dict] = None,
        max_cutout_bands: int   = 20,
        noise_std:        float = 0.02,
    ) -> None:
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.aug               = aug_profile       # dict or None
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std

    def __len__(self) -> int:
        return len(self.indices)

    def _band_dropout(self, x, p_band=0.04):
        mask = (torch.rand(x.shape[0]) > p_band).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x):
        x   = x.clone()
        nb  = x.shape[0]
        cut = torch.randint(1, max(2, self.max_cutout_bands), (1,)).item()
        st  = torch.randint(0, max(1, nb - cut), (1,)).item()
        x[st: st + cut] = 0.0
        return x

    def _spectral_noise(self, x):
        return x + torch.randn_like(x) * self.noise_std

    def _spectral_warp(self, x):
        C, H, W = x.shape
        scale   = 1.0 + random.uniform(-0.10, 0.10)
        new_C   = max(1, int(C * scale))
        if new_C == C:
            return x
        xp     = x.permute(1, 2, 0).reshape(-1, 1, C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s      = (new_C - C) // 2
            warped = warped[:, :, s:s + C]
        else:
            lo     = (C - new_C) // 2
            hi     = C - new_C - lo
            warped = F.pad(warped, (lo, hi))
        return warped.reshape(H, W, C).permute(2, 0, 1)

    def _spectral_shift(self, x):
        return torch.roll(x, random.randint(-8, 8), dims=0)

    def _mult_noise(self, x):
        return x * (1.0 + torch.randn(x.shape[0], 1, 1) * 0.05)

    def _spatial(self, x):
        if torch.rand(1) < 0.5: x = torch.flip(x, [2])
        if torch.rand(1) < 0.5: x = torch.flip(x, [1])
        return torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ri    = self.indices[idx]
        patch = torch.from_numpy(self.patches[ri].copy()).float()
        label = torch.tensor(self.labels[ri], dtype=torch.long)
        p     = self.aug
        if p is not None:
            if torch.rand(1) < p["band_drop"]:  patch = self._band_dropout(patch)
            if torch.rand(1) < p["cutout"]:     patch = self._band_cutout(patch)
            if torch.rand(1) < p["noise"]:      patch = self._spectral_noise(patch)
            if torch.rand(1) < p["warp"]:       patch = self._spectral_warp(patch)
            if torch.rand(1) < p["shift"]:      patch = self._spectral_shift(patch)
            if torch.rand(1) < p["mult"]:       patch = self._mult_noise(patch)
            patch = self._spatial(patch)
        return patch, label


# ══════════════════════════════════════════════════════════════════════
#  CLASS-BALANCED BATCH SAMPLER
# ══════════════════════════════════════════════════════════════════════

class ClassBalancedBatchSampler(Sampler):
    def __init__(self, train_labels: np.ndarray,
                 n_cls: int = 16, n_spc: int = 4) -> None:
        self.n_cls   = n_cls
        self.n_spc   = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False)
            batch: List[int] = []
            for c in chosen:
                pool = self.cls_idx[c]
                samp = rng.choice(pool, self.n_spc, replace=len(pool) < self.n_spc)
                batch.extend(samp.tolist())
            yield batch

    def __len__(self) -> int:
        return self._n


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION  (Mixup + CutMix)
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _cutmix(x, y, alpha):
    lam        = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx        = torch.randperm(B, device=x.device)
    r          = math.sqrt(1.0 - lam)
    ch, cw     = int(H * r), int(W * r)
    cx, cy     = random.randint(0, W), random.randint(0, H)
    x1 = max(cx - cw // 2, 0); x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0); y2 = min(cy + ch // 2, H)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam   = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_mix, y, y[idx], lam


def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1) < 0.5 else _cutmix)(x, y, alpha)


def mixed_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ══════════════════════════════════════════════════════════════════════
#  ARCFACE HEAD
# ══════════════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int,
                 s: float = 32.0, m: float = 0.30) -> None:
        super().__init__()
        self.weight    = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s         = s
        self.default_m = m
        self._cached_m = None
        self._precompute(m)

    def _precompute(self, m: float) -> None:
        self._cached_m = m
        self._cosm = math.cos(m)
        self._sinm = math.sin(m)
        self._th   = math.cos(math.pi - m)
        self._mm   = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                m: Optional[float] = None) -> torch.Tensor:
        if m is not None and m != self._cached_m:
            self._precompute(m)
        cosine = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        cosine = cosine.clamp(-1 + 1e-6, 1 - 1e-6)
        if labels is None or not self.training:
            return cosine * self.s
        sine  = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-6))
        phi   = cosine * self._cosm - sine * self._sinm
        phi   = torch.where(cosine > self._th, phi, cosine - self._mm)
        oh    = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi) + ((1.0 - oh) * cosine)) * self.s


# ══════════════════════════════════════════════════════════════════════
#  PROTO-NCE LOSS
# ══════════════════════════════════════════════════════════════════════

class ProtoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device  = features.device
        classes = labels.unique()
        if len(classes) < 2:
            return features.new_tensor(0.0, requires_grad=True)
        protos = torch.stack([features[labels == c].mean(0) for c in classes])
        protos = F.normalize(protos, dim=1)
        sim    = torch.mm(features, protos.T) / self.temperature
        c2l    = {c.item(): i for i, c in enumerate(classes)}
        local  = torch.tensor([c2l[y.item()] for y in labels],
                               dtype=torch.long, device=device)
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  MASKED SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(x: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x32  = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)
    energy  = flat.abs().sum(1, keepdim=True)
    mask    = (energy > 1e-5).float()
    count   = mask.sum(2).clamp(min=1.0)
    mean    = (flat * mask).sum(2) / count
    mean_sq = ((flat ** 2) * mask).sum(2) / count
    std     = (mean_sq - mean ** 2).clamp(min=1e-6).sqrt()
    flat_fg = flat.masked_fill(mask.expand_as(flat) == 0, -1e4)
    mx      = flat_fg.max(2).values
    mx      = mx.masked_fill(mx < -9999.0, 0.0)
    return (torch.nan_to_num(mean, 0), torch.nan_to_num(std, 0), torch.nan_to_num(mx, 0))


# ══════════════════════════════════════════════════════════════════════
#  WAVELENGTH POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class WavelengthPositionalEncoding(nn.Module):
    def __init__(self, num_bands: int = 256, embed_dim: int = 16) -> None:
        super().__init__()
        wl   = torch.linspace(0.0, 1.0, num_bands)
        half = embed_dim // 2
        freq = torch.exp(
            torch.arange(half).float() * -(math.log(10_000.0) / max(half - 1, 1)))
        enc  = torch.zeros(num_bands, embed_dim)
        enc[:, :half] = torch.sin(wl.unsqueeze(1) * freq.unsqueeze(0))
        enc[:, half:] = torch.cos(wl.unsqueeze(1) * freq.unsqueeze(0))
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
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x.mean([2, 3]))
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm1d(out_ch))
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid = max(c // r, 8)
        self.ch_mlp  = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.GELU(),
                                     nn.Conv2d(mid, c, 1, bias=False))
        self.sp_conv = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False),
                                     nn.Sigmoid())

    def forward(self, x):
        ca = torch.sigmoid(
            self.ch_mlp(x.mean([2, 3], keepdim=True)) +
            self.ch_mlp(x.amax([2, 3], keepdim=True)))
        x  = x * ca
        sp = self.sp_conv(
            torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1))
        return x * sp


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid,    1, bias=False)
        self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid,   mid,    3, stride, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid,   out_ch, 1, bias=False)
        self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                   nn.GroupNorm(min(8, out_ch), out_ch))
                     if (stride != 1 or in_ch != out_ch) else nn.Identity())

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = F.gelu(self.n2(self.c2(h)))
        h = self.n3(self.c3(h))
        return F.gelu(h + self.skip(x))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A — SPECTRAL PROFILE
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
            nn.Linear(tower_ch * 6, out_dim), nn.BatchNorm1d(out_dim),
            nn.GELU(), nn.Dropout(0.1))

    @staticmethod
    def _tower(in_ch, out_ch, k):
        mid = out_ch // 2
        return nn.Sequential(ResBlock1D(in_ch, mid, k), ResBlock1D(mid, out_ch, k),
                              ResBlock1D(out_ch, out_ch, k))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], dim=1)

    def forward(self, ms):
        s = ms.unsqueeze(1)
        d = F.pad(torch.diff(s, dim=2), (0, 1))
        x = torch.cat([s, d], dim=1)
        if self.wl_enc is not None: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], dim=1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80,
                 wl_enc: Optional[WavelengthPositionalEncoding] = None):
        super().__init__()
        self.wl_enc  = wl_enc
        self.tower_s = self._tower(3, tower_ch, k=3)
        self.tower_m = self._tower(3, tower_ch, k=7)
        self.tower_l = self._tower(3, tower_ch, k=15)
        self.proj    = nn.Sequential(
            nn.Linear(tower_ch * 6, out_dim), nn.BatchNorm1d(out_dim),
            nn.GELU(), nn.Dropout(0.1))

    @staticmethod
    def _tower(in_ch, out_ch, k):
        mid = out_ch // 2
        return nn.Sequential(ResBlock1D(in_ch, mid, k), ResBlock1D(mid, out_ch, k),
                              ResBlock1D(out_ch, out_ch, k))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], dim=1)

    def forward(self, ms, ss, mx):
        x = torch.stack([ms, ss, mx], dim=1)
        if self.wl_enc is not None: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], dim=1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands=256, out_dim=256):
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.GELU())
        self.stages = nn.Sequential(
            ResBlock2D(32,  64,      2), CBAM(64),
            ResBlock2D(64,  128,     2), CBAM(128),
            ResBlock2D(128, 192,     2), CBAM(192),
            ResBlock2D(192, out_dim, 2))
        self.pool_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.BatchNorm1d(out_dim), nn.GELU())

    @staticmethod
    def _pnorm(x): return x.sign() * x.abs().clamp(min=1e-8).sqrt()

    def forward(self, x):
        h    = self.stages(self.band_reduce(x))
        avg  = h.mean([2, 3])
        mx   = h.amax([2, 3])
        feat = F.normalize(torch.cat([self._pnorm(avg), self._pnorm(mx)], 1), dim=1)
        return self.pool_proj(feat)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER
# ══════════════════════════════════════════════════════════════════════

class _PreLNBlock(nn.Module):
    def __init__(self, d, heads, d_ff, drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(d_ff, d), nn.Dropout(drop))
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         need_weights=False)
        x = x + self.drop(h)
        return x + self.drop(self.ff(self.norm2(x)))


class SpecFormerBranch(nn.Module):
    def __init__(self, num_bands=256, patch_size=8, d_model=128,
                 n_heads=4, n_layers=4, out_dim=256, dropout=0.15):
        super().__init__()
        n_patches       = num_bands // patch_size
        self.patch_size = patch_size
        self.n_patches  = n_patches
        self.patch_proj = nn.Sequential(nn.Linear(patch_size, d_model, bias=False),
                                        nn.LayerNorm(d_model))
        wl   = torch.linspace(WL_MIN, WL_MAX, n_patches)
        wl_n = (wl - WL_MIN) / (WL_MAX - WL_MIN)
        half = d_model // 2
        freq = torch.exp(
            torch.arange(half).float() * -(math.log(1e4) / max(half - 1, 1)))
        pe   = torch.zeros(n_patches, d_model)
        pe[:, :half] = torch.sin(wl_n.unsqueeze(1) * freq.unsqueeze(0))
        pe[:, half:] = torch.cos(wl_n.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("wl_pe", pe)
        self.cls    = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([_PreLNBlock(d_model, n_heads, d_model * 2, dropout)
                                     for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.proj   = nn.Sequential(nn.Linear(d_model, out_dim), nn.BatchNorm1d(out_dim),
                                    nn.GELU(), nn.Dropout(dropout))

    def forward(self, ms):
        B = ms.shape[0]
        x = ms.float().view(B, self.n_patches, self.patch_size)
        x = self.patch_proj(x) + self.wl_pe.unsqueeze(0)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        for blk in self.blocks: x = blk(x)
        return self.proj(self.norm(x)[:, 0])


# ══════════════════════════════════════════════════════════════════════
#  BRANCH CROSS-ATTENTION FUSION
# ══════════════════════════════════════════════════════════════════════

class BranchCrossAttention(nn.Module):
    def __init__(self, d=256, n_heads=4, dropout=0.10):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d * 2, d), nn.Dropout(dropout))
        self.drop  = nn.Dropout(dropout)
        self.gate  = nn.Parameter(torch.ones(1))

    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(branches, dim=1)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gate * self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x.flatten(1)


# ══════════════════════════════════════════════════════════════════════
#  MAIN MODEL — SpectralQuadNet v8  (parameterised via cfg dict)
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    def __init__(self, num_classes=90, num_bands=256, dropout=0.30,
                 wl_embed_dim=16, tower_ch=80, cfg=None):
        super().__init__()
        cfg = cfg or {}
        specf_patch  = cfg.get("specf_patch",  8)
        specf_dim    = cfg.get("specf_dim",    128)
        specf_heads  = cfg.get("specf_heads",  4)
        specf_layers = cfg.get("specf_layers", 4)
        specf_drop   = cfg.get("specf_drop",   0.15)
        fusion_heads = cfg.get("fusion_heads", 4)
        fusion_drop  = cfg.get("fusion_drop",  0.10)
        arcface_s    = cfg.get("s2_arcface_s", 32.0)
        arcface_m    = cfg.get("s2_arcface_m", 0.30)

        self.se       = SpectralSE(num_bands, 16)
        self.wl_enc   = WavelengthPositionalEncoding(num_bands, wl_embed_dim)
        self.branch_a = SpectralProfileBranch(256, tower_ch, self.wl_enc)
        self.branch_b = SpectralStatsBranch(  256, tower_ch, self.wl_enc)
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(
            num_bands, specf_patch, specf_dim, specf_heads, specf_layers,
            256, specf_drop)
        self.cross_attn = BranchCrossAttention(256, fusion_heads, fusion_drop)
        self.embed_net  = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256))
        self.linear_head  = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout * 0.4), nn.Linear(256, num_classes))
        self.arcface_head = ArcFaceHead(256, num_classes, arcface_s, arcface_m)
        self._use_arcface = False
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

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def use_arcface(self, flag: bool) -> None:
        self._use_arcface = flag

    def freeze_head(self, which: str) -> None:
        head = self.linear_head if which == "linear" else self.arcface_head
        for p in head.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which: str) -> None:
        head = self.linear_head if which == "linear" else self.arcface_head
        for p in head.parameters(): p.requires_grad_(True)

    def forward(self, x, labels=None, return_embed=False, arc_m=None):
        x  = self.se(x)
        ms, ss, mx = masked_spectral_stats(x)
        fa = self.branch_a(ms)
        fb = self.branch_b(ms, ss, mx)
        fc = self.branch_c(x)
        fd = self.branch_d(ms)
        emb = self.embed_net(self.cross_attn([fa, fb, fc, fd]))
        if self._use_arcface:
            emb_n  = F.normalize(F.gelu(emb), dim=1)
            logits = self.arcface_head(emb_n, labels, m=arc_m)
        else:
            logits = self.linear_head(emb)
        if return_embed:
            return logits, F.normalize(F.gelu(emb.detach()), dim=1)
        return logits


# ══════════════════════════════════════════════════════════════════════
#  ARCFACE INIT FROM LINEAR HEAD
# ══════════════════════════════════════════════════════════════════════

def init_arcface_from_linear(model: nn.Module) -> None:
    linear = model.linear_head[-1]
    arc    = model.arcface_head
    with torch.no_grad():
        arc.weight.data.copy_(F.normalize(linear.weight.data.clone(), dim=1))


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits(cfg: dict):
    labels  = np.load(cfg["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(cfg: dict, trial_p: dict,
                  train_idx, val_idx, test_idx, batch_size,
                  stage: str,
                  balanced: bool = False,
                  all_labels=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    nw   = cfg["num_workers"]
    kw   = dict(num_workers=nw, pin_memory=True)

    # Build aug profile from trial params
    if stage == "heavy":
        aug_profile = {
            "band_drop": trial_p["aug_heavy_band_drop"],
            "cutout":    trial_p["aug_heavy_cutout"],
            "noise":     trial_p["aug_heavy_noise"],
            "warp":      trial_p["aug_heavy_warp"],
            "shift":     trial_p["aug_heavy_shift"],
            "mult":      trial_p["aug_heavy_mult"],
        }
    elif stage == "light":
        aug_profile = {
            "band_drop": trial_p["aug_light_band_drop"],
            "cutout":    trial_p["aug_light_cutout"],
            "noise":     trial_p["aug_light_noise"],
            "warp":      trial_p["aug_light_warp"],
            "shift":     trial_p["aug_light_shift"],
            "mult":      trial_p["aug_light_mult"],
        }
    else:
        aug_profile = None

    train_ds = RiceSeedDataset(
        cfg["patches_data"], cfg["labels_path"], train_idx,
        aug_profile=aug_profile,
        max_cutout_bands=trial_p["max_cutout_bands"],
        noise_std=trial_p["noise_std"],
    )

    if balanced and all_labels is not None:
        n_cls   = trial_p["bal_n_cls"]
        n_spc   = trial_p["bal_n_spc"]
        sampler = ClassBalancedBatchSampler(all_labels[train_idx], n_cls, n_spc)
        train_ldr = DataLoader(train_ds, batch_sampler=sampler,
                               persistent_workers=True, prefetch_factor=2, **kw)
    else:
        train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               persistent_workers=True, prefetch_factor=2, **kw)

    val_ldr  = DataLoader(
        RiceSeedDataset(cfg["patches_data"], cfg["labels_path"], val_idx),
        batch_size=64, shuffle=False, **kw)
    test_ldr = DataLoader(
        RiceSeedDataset(cfg["patches_data"], cfg["labels_path"], test_idx),
        batch_size=64, shuffle=False, **{**kw, "num_workers": 2})
    return train_ldr, val_ldr, test_ldr


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISERS
# ══════════════════════════════════════════════════════════════════════

def _wd_split(named_params, lr, weight_decay):
    wd, no_wd = [], []
    for name, p in named_params:
        if not p.requires_grad: continue
        (no_wd if (p.ndim == 1 or name.endswith(".bias")) else wd).append(p)
    return [{"params": wd,    "lr": lr, "weight_decay": weight_decay},
            {"params": no_wd, "lr": lr, "weight_decay": 0.0}]


def build_optimizer_s1(model, lr, weight_decay):
    groups = _wd_split(model.named_parameters(), lr, weight_decay)
    return optim.AdamW(groups)


def build_optimizer_s2(model, head_lr, backbone_lr, weight_decay):
    head_params, back_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.startswith("arcface_head"):
            head_params.append((name, p))
        else:
            back_params.append((name, p))
    groups = (_wd_split(head_params, head_lr,      weight_decay) +
              _wd_split(back_params, backbone_lr,  weight_decay))
    return optim.AdamW(groups)


def build_optimizer_s3(model, lr, weight_decay):
    groups = _wd_split(model.named_parameters(), lr, weight_decay)
    return optim.AdamW(groups)


# ══════════════════════════════════════════════════════════════════════
#  SCHEDULER
# ══════════════════════════════════════════════════════════════════════

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                 eta_min_frac=1e-3):
    def _lambda(ep):
        if ep < warmup_epochs:
            return ep / max(warmup_epochs, 1)
        t = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return eta_min_frac + 0.5 * (1 - eta_min_frac) * (1 + math.cos(math.pi * t))
    return optim.lr_scheduler.LambdaLR(optimizer, _lambda)


def arcface_margin(ep, m0, m_target, warmup_ep):
    if ep >= warmup_ep:
        return m_target
    frac = ep / max(warmup_ep, 1)
    return m0 + (m_target - m0) * 0.5 * (1 - math.cos(math.pi * frac))


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, ema, device,
    scheduler=None, use_mixup=True, mixup_alpha=0.4,
    supcon=None, supcon_weight=0.0, accum_steps=1,
    arc_m=None, grad_clip=1.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_amp = (supcon is None) and (scaler is not None)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_in, y_a, y_b, lam = (mixed_aug(x, y, mixup_alpha)
                                if use_mixup else (x, y, y, 1.0))

        with autocast(device_type=device.type, enabled=use_amp):
            if supcon is not None:
                logits, emb = model(x_in, y_a, return_embed=True, arc_m=arc_m)
                loss = ((1 - supcon_weight) * criterion(logits, y_a) +
                        supcon_weight * supcon(emb, y_a))
            else:
                logits = model(x_in,
                               labels=y_a if (model._use_arcface and not use_mixup) else None,
                               arc_m=arc_m)
                loss   = mixed_loss(criterion, logits, y_a, y_b, lam)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / accum_steps).backward() if not use_amp else \
            scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_amp: scaler.step(optimizer); scaler.update()
            else:       optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None: scheduler.step()
            if ema is not None:       ema.update(model)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == y).float().mean().item()

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
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
    return (f1_score(t, p, average="macro", zero_division=0),
            accuracy_score(t, p))


def update_bn_stats(loader, model, device):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats(); m.momentum = None
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device, non_blocking=True))
    model.eval()


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, stage, model, ema, val_acc, val_f1):
    torch.save({
        "epoch": epoch, "stage": stage,
        "model": model.state_dict(),
        "ema":   ema.state_dict(),
        "val_acc": val_acc, "val_f1": val_f1,
        "use_arcface": model._use_arcface,
    }, path)


def load_ckpt(path, model, ema, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    use_af = ckpt.get("use_arcface", False)
    model.use_arcface(use_af)
    ema.shadow.use_arcface(use_af)
    return ckpt


# ══════════════════════════════════════════════════════════════════════
#  STAGE RUNNERS  (parameterised — no global CONFIG dependency)
# ══════════════════════════════════════════════════════════════════════

def run_stage1(model, ema, train_ldr, val_ldr, device,
               trial_p, tune_cfg, trial_dir, trial) -> float:

    s1_epochs  = tune_cfg["s1_epochs"]
    patience   = max(5, int(s1_epochs * tune_cfg["s1_patience_frac"]))
    accum      = trial_p["s1_accum"]

    model.use_arcface(False)
    model.unfreeze_head("linear")
    model.freeze_head("arcface")

    criterion = nn.CrossEntropyLoss(label_smoothing=trial_p["label_smoothing"])
    optimizer = build_optimizer_s1(model, trial_p["s1_max_lr"] / 25,
                                   trial_p["weight_decay"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr           = trial_p["s1_max_lr"],
            epochs           = s1_epochs,
            steps_per_epoch  = math.ceil(len(train_ldr) / accum),
            pct_start        = trial_p["s1_warmup_pct"],
            div_factor       = 25,
            final_div_factor = 1e4,
            anneal_strategy  = "cos",
        )

    scaler     = GradScaler()
    best_acc   = 0.0
    best_f1    = 0.0
    no_improve = 0
    ckpt_path  = os.path.join(trial_dir, "best_s1.pth")

    for ep in range(1, s1_epochs + 1):
        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, criterion, scaler, ema, device,
            scheduler=scheduler, use_mixup=True,
            mixup_alpha=trial_p["s1_mixup"],
            accum_steps=accum,
            grad_clip=trial_p["grad_clip"],
        )
        _, va_live = evaluate(model,      val_ldr, device)
        vf1, va_ema = evaluate(ema.shadow, val_ldr, device)
        va_best    = max(va_live, va_ema)

        if va_best > best_acc:
            best_acc, best_f1, no_improve = va_best, vf1, 0
            save_ckpt(ckpt_path, ep, "Stage 1", model, ema, va_best, vf1)
        else:
            no_improve += 1

        # Hyperband pruning: report intermediate F1 every 5 epochs
        if ep % 5 == 0:
            trial.report(vf1, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if no_improve >= patience:
            break

    model.unfreeze_head("arcface")
    return best_f1, ckpt_path


def run_stage2(model, ema, train_ldr, val_ldr, device,
               trial_p, tune_cfg, trial_dir, trial, s1_f1_offset=0) -> float:

    s2_epochs  = tune_cfg["s2_epochs"]
    patience   = max(5, int(s2_epochs * tune_cfg["s2_patience_frac"]))

    model.set_dropout(trial_p["s2_dropout"])
    model.use_arcface(True)
    model.freeze_head("linear")
    model.unfreeze_head("arcface")
    ema.reinit_from(model)
    ema.set_dropout(trial_p["s2_dropout"])
    ema.shadow.use_arcface(True)

    criterion_s2 = nn.CrossEntropyLoss(label_smoothing=trial_p["s2_label_smooth"])
    proto        = ProtoNCELoss(temperature=trial_p["proto_temp"])

    head_lr     = trial_p["s2_head_lr"]
    backbone_lr = head_lr / trial_p["s2_lr_ratio"]
    min_lr_frac = trial_p["s2_min_lr_frac"]

    optimizer = build_optimizer_s2(model, head_lr, backbone_lr, trial_p["weight_decay"])
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=trial_p["s2_warmup_ep"],
        total_epochs=s2_epochs,
        eta_min_frac=min_lr_frac,
    )

    best_acc   = 0.0
    best_f1    = 0.0
    no_improve = 0
    ckpt_path  = os.path.join(trial_dir, "best_s2.pth")

    for ep in range(1, s2_epochs + 1):
        m_now   = arcface_margin(ep - 1, trial_p["s2_arcface_m0"],
                                 trial_p["s2_arcface_m"], trial_p["s2_margin_warmup_ep"])
        proto_w = min(trial_p["proto_weight"],
                      (ep / max(trial_p["s2_warmup_ep"] * 2, 1)) *
                      trial_p["proto_weight"])

        tl, ta  = train_one_epoch(
            model, train_ldr, optimizer, criterion_s2,
            scaler=None, ema=ema, device=device,
            scheduler=None, use_mixup=False,
            supcon=proto, supcon_weight=proto_w, arc_m=m_now,
            grad_clip=trial_p["grad_clip"],
        )
        scheduler.step()

        _, va_live  = evaluate(model,      val_ldr, device)
        vf1, va_ema = evaluate(ema.shadow, val_ldr, device)
        va_best     = max(va_live, va_ema)

        if va_best > best_acc:
            best_acc, best_f1, no_improve = va_best, vf1, 0
            save_ckpt(ckpt_path, ep, "Stage 2", model, ema, va_best, vf1)
        else:
            no_improve += 1

        if ep % 5 == 0:
            trial.report(vf1, step=tune_cfg["s1_epochs"] + ep + s1_f1_offset)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if no_improve >= patience:
            break

    model.unfreeze_head("linear")
    return best_f1, ckpt_path


def run_stage3_swa(model, ema, train_ldr, val_ldr, device,
                   trial_p, tune_cfg, trial_dir, trial, prev_best_val,
                   epoch_offset=0) -> float:

    s3_epochs = tune_cfg["s3_epochs"]

    model.set_dropout(trial_p["s2_dropout"])
    model.use_arcface(True)
    ema.shadow.use_arcface(True)

    optimizer    = build_optimizer_s3(model, trial_p["s3_swa_lr"], trial_p["weight_decay"])
    scaler       = GradScaler()
    criterion_s3 = nn.CrossEntropyLoss(label_smoothing=trial_p["s2_label_smooth"])
    cycle_len    = trial_p["s3_cycle_len"]

    swa_state: dict = copy.deepcopy(model.state_dict())
    n_snap          = 1

    best_f1    = 0.0
    ckpt_path  = os.path.join(trial_dir, "best_s3.pth")

    for ep in range(1, s3_epochs + 1):
        cycle_ep = (ep - 1) % cycle_len
        lr_now   = trial_p["s3_swa_lr"] * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cycle_ep / cycle_len)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        train_one_epoch(
            model, train_ldr, optimizer, criterion_s3, scaler, ema,
            device, scheduler=None, use_mixup=False,
            grad_clip=trial_p["grad_clip"],
        )

        if ep % cycle_len == 0:
            n_snap += 1
            a  = 1.0 / n_snap
            sd = model.state_dict()
            for k in swa_state:
                swa_state[k] = swa_state[k] + a * (sd[k] - swa_state[k])

    # BN update and final eval
    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state)
    swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)

    vf1_swa, va_swa = evaluate(swa_model,   val_ldr, device)
    vf1_ema, va_ema = evaluate(ema.shadow,  val_ldr, device)

    if va_swa >= va_ema:
        ema.shadow.load_state_dict(swa_model.state_dict())
        ema.shadow.use_arcface(True)
        best_f1 = vf1_swa
    else:
        best_f1 = vf1_ema

    if max(va_swa, va_ema) > prev_best_val:
        save_ckpt(ckpt_path, s3_epochs, "Stage 3",
                  swa_model, ema, max(va_swa, va_ema), best_f1)

    return best_f1, ckpt_path


# ══════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════

def objective(trial: optuna.Trial, base_cfg: dict, tune_cfg: dict,
              splits, study_dir: str) -> float:

    set_seed(base_cfg["seed"])
    device = base_cfg["device"]
    all_labels, train_idx, val_idx, test_idx = splits

    # ── Trial working directory ──────────────────────────────────────
    trial_dir = os.path.join(study_dir, f"trial_{trial.number:04d}")
    Path(trial_dir).mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    #  HYPER-PARAMETER SAMPLING
    # ════════════════════════════════════════════════════════════════

    # ── Architecture ────────────────────────────────────────────────
    wl_embed_dim  = trial.suggest_categorical("wl_embed_dim",  [8, 16, 32])
    tower_ch      = trial.suggest_categorical("tower_ch",      [64, 80, 96, 112])
    specf_patch   = trial.suggest_categorical("specf_patch",   [4, 8, 16])
    specf_dim     = trial.suggest_categorical("specf_dim",     [64, 128, 192, 256])
    specf_heads   = trial.suggest_categorical("specf_heads",   [2, 4, 8])
    specf_layers  = trial.suggest_int(        "specf_layers",  2, 6)
    specf_drop    = trial.suggest_float(      "specf_drop",    0.05, 0.25, step=0.05)
    fusion_heads  = trial.suggest_categorical("fusion_heads",  [2, 4, 8])
    fusion_drop   = trial.suggest_float(      "fusion_drop",   0.05, 0.20, step=0.05)

    # Constraint: specf_dim must be divisible by specf_heads
    if specf_dim % specf_heads != 0:
        raise optuna.TrialPruned()

    # ── Augmentation ────────────────────────────────────────────────
    aug_heavy_band_drop = trial.suggest_float("aug_heavy_band_drop", 0.40, 0.85, step=0.05)
    aug_heavy_cutout    = trial.suggest_float("aug_heavy_cutout",    0.25, 0.70, step=0.05)
    aug_heavy_noise     = trial.suggest_float("aug_heavy_noise",     0.20, 0.60, step=0.05)
    aug_heavy_warp      = trial.suggest_float("aug_heavy_warp",      0.15, 0.55, step=0.05)
    aug_heavy_shift     = trial.suggest_float("aug_heavy_shift",     0.15, 0.50, step=0.05)
    aug_heavy_mult      = trial.suggest_float("aug_heavy_mult",      0.15, 0.50, step=0.05)

    aug_light_band_drop = trial.suggest_float("aug_light_band_drop", 0.10, 0.45, step=0.05)
    aug_light_cutout    = trial.suggest_float("aug_light_cutout",    0.05, 0.35, step=0.05)
    aug_light_noise     = trial.suggest_float("aug_light_noise",     0.05, 0.35, step=0.05)
    aug_light_warp      = trial.suggest_float("aug_light_warp",      0.05, 0.30, step=0.05)
    aug_light_shift     = trial.suggest_float("aug_light_shift",     0.05, 0.30, step=0.05)
    aug_light_mult      = trial.suggest_float("aug_light_mult",      0.05, 0.30, step=0.05)

    noise_std        = trial.suggest_float("noise_std",        0.005, 0.05, log=True)
    max_cutout_bands = trial.suggest_int(  "max_cutout_bands", 10, 40)

    # ── Stage 1 ────────────────────────────────────────────────────
    s1_batch        = trial.suggest_categorical("s1_batch",    [32, 64, 128])
    s1_max_lr       = trial.suggest_float("s1_max_lr",         1e-4, 2e-3, log=True)
    s1_dropout      = trial.suggest_float("s1_dropout",        0.10, 0.45, step=0.05)
    s1_mixup        = trial.suggest_float("s1_mixup",          0.2,  0.6,  step=0.05)
    s1_warmup_pct   = trial.suggest_float("s1_warmup_pct",     0.05, 0.25, step=0.05)
    s1_accum        = trial.suggest_categorical("s1_accum",    [1, 2, 4])
    label_smoothing = trial.suggest_float("label_smoothing",   0.00, 0.15, step=0.01)

    # ── Shared ─────────────────────────────────────────────────────
    weight_decay    = trial.suggest_float("weight_decay",      5e-5, 5e-4, log=True)
    grad_clip       = trial.suggest_float("grad_clip",         0.5,  5.0,  log=True)
    ema_decay       = trial.suggest_categorical("ema_decay",   [0.999, 0.9995, 0.9999])

    # ── Stage 2 ─────────────────────────────────────────────────────
    s2_batch            = trial.suggest_categorical("s2_batch",  [32, 64])
    s2_head_lr          = trial.suggest_float("s2_head_lr",       5e-5, 5e-4, log=True)
    s2_lr_ratio         = trial.suggest_float("s2_lr_ratio",      5.0,  20.0, step=1.0)
    s2_min_lr_frac      = trial.suggest_float("s2_min_lr_frac",   1e-4, 1e-2, log=True)
    s2_warmup_ep        = trial.suggest_int(  "s2_warmup_ep",     2, 10)
    s2_dropout          = trial.suggest_float("s2_dropout",       0.05, 0.25, step=0.05)
    s2_arcface_s        = trial.suggest_categorical("s2_arcface_s", [16.0, 24.0, 32.0, 48.0, 64.0])
    s2_arcface_m        = trial.suggest_float("s2_arcface_m",      0.15, 0.55, step=0.05)
    s2_arcface_m0       = trial.suggest_float("s2_arcface_m0",     0.01, 0.10, step=0.01)
    s2_margin_warmup_ep = trial.suggest_int(  "s2_margin_warmup_ep", 10, 50, step=5)
    s2_label_smooth     = trial.suggest_categorical("s2_label_smooth", [0.0, 0.01, 0.02, 0.05])
    proto_weight        = trial.suggest_float("proto_weight",      0.0,  0.5,  step=0.05)
    proto_temp          = trial.suggest_float("proto_temp",        0.05, 0.25, step=0.01)

    # Class-balanced sampler
    bal_n_cls           = trial.suggest_categorical("bal_n_cls", [8, 12, 16, 20, 24])
    bal_n_spc           = trial.suggest_categorical("bal_n_spc", [4, 6, 8])

    # ── Stage 3 ─────────────────────────────────────────────────────
    s3_swa_lr   = trial.suggest_float("s3_swa_lr",   1e-5, 3e-4, log=True)
    s3_cycle_len = trial.suggest_categorical("s3_cycle_len", [3, 5, 7, 10])

    # ── Pack all params ─────────────────────────────────────────────
    trial_p: dict = dict(
        # Architecture
        wl_embed_dim=wl_embed_dim, tower_ch=tower_ch,
        specf_patch=specf_patch, specf_dim=specf_dim, specf_heads=specf_heads,
        specf_layers=specf_layers, specf_drop=specf_drop,
        fusion_heads=fusion_heads, fusion_drop=fusion_drop,
        s2_arcface_s=s2_arcface_s, s2_arcface_m=s2_arcface_m,
        # Augmentation
        aug_heavy_band_drop=aug_heavy_band_drop, aug_heavy_cutout=aug_heavy_cutout,
        aug_heavy_noise=aug_heavy_noise, aug_heavy_warp=aug_heavy_warp,
        aug_heavy_shift=aug_heavy_shift, aug_heavy_mult=aug_heavy_mult,
        aug_light_band_drop=aug_light_band_drop, aug_light_cutout=aug_light_cutout,
        aug_light_noise=aug_light_noise, aug_light_warp=aug_light_warp,
        aug_light_shift=aug_light_shift, aug_light_mult=aug_light_mult,
        noise_std=noise_std, max_cutout_bands=max_cutout_bands,
        # Stage 1
        s1_batch=s1_batch, s1_max_lr=s1_max_lr, s1_dropout=s1_dropout,
        s1_mixup=s1_mixup, s1_warmup_pct=s1_warmup_pct, s1_accum=s1_accum,
        label_smoothing=label_smoothing,
        # Shared
        weight_decay=weight_decay, grad_clip=grad_clip, ema_decay=ema_decay,
        # Stage 2
        s2_batch=s2_batch, s2_head_lr=s2_head_lr, s2_lr_ratio=s2_lr_ratio,
        s2_min_lr_frac=s2_min_lr_frac, s2_warmup_ep=s2_warmup_ep,
        s2_dropout=s2_dropout, s2_arcface_m0=s2_arcface_m0,
        s2_margin_warmup_ep=s2_margin_warmup_ep, s2_label_smooth=s2_label_smooth,
        proto_weight=proto_weight, proto_temp=proto_temp,
        bal_n_cls=bal_n_cls, bal_n_spc=bal_n_spc,
        # Stage 3
        s3_swa_lr=s3_swa_lr, s3_cycle_len=s3_cycle_len,
    )

    # ════════════════════════════════════════════════════════════════
    #  BUILD MODEL
    # ════════════════════════════════════════════════════════════════
    arch_cfg = dict(
        specf_patch=specf_patch, specf_dim=specf_dim, specf_heads=specf_heads,
        specf_layers=specf_layers, specf_drop=specf_drop,
        fusion_heads=fusion_heads, fusion_drop=fusion_drop,
        s2_arcface_s=s2_arcface_s, s2_arcface_m=s2_arcface_m,
    )

    model = SpectralQuadNet(
        num_classes=base_cfg["num_classes"],
        num_bands=base_cfg["num_bands"],
        dropout=s1_dropout,
        wl_embed_dim=wl_embed_dim,
        tower_ch=tower_ch,
        cfg=arch_cfg,
    ).to(device)
    ema = ModelEMA(model, ema_decay)

    # ════════════════════════════════════════════════════════════════
    #  STAGE 1
    # ════════════════════════════════════════════════════════════════
    train_ldr1, val_ldr1, _ = build_loaders(
        base_cfg, trial_p, train_idx, val_idx, test_idx,
        s1_batch, stage="heavy")

    s1_f1, ckpt_s1 = run_stage1(
        model, ema, train_ldr1, val_ldr1, device,
        trial_p, tune_cfg, trial_dir, trial)

    print(f"\n[Trial {trial.number}] Stage 1 best val F1 = {s1_f1:.4f}")

    # ════════════════════════════════════════════════════════════════
    #  STAGE 2
    # ════════════════════════════════════════════════════════════════
    # Reload best Stage-1 weights
    load_ckpt(ckpt_s1, model, ema, device)
    init_arcface_from_linear(model)
    init_arcface_from_linear(ema.shadow)

    train_ldr2, val_ldr2, _ = build_loaders(
        base_cfg, trial_p, train_idx, val_idx, test_idx,
        s2_batch, stage="light", balanced=True, all_labels=all_labels)

    s2_f1, ckpt_s2 = run_stage2(
        model, ema, train_ldr2, val_ldr2, device,
        trial_p, tune_cfg, trial_dir, trial, s1_f1_offset=0)

    print(f"[Trial {trial.number}] Stage 2 best val F1 = {s2_f1:.4f}")

    # ════════════════════════════════════════════════════════════════
    #  STAGE 3  (SWA)
    # ════════════════════════════════════════════════════════════════
    s2_ckpt      = torch.load(ckpt_s2, map_location=device, weights_only=False)
    s2_best_val  = s2_ckpt.get("val_acc", 0.0)
    load_ckpt(ckpt_s2, model, ema, device)

    train_ldr3, val_ldr3, _ = build_loaders(
        base_cfg, trial_p, train_idx, val_idx, test_idx,
        s2_batch, stage="light")

    s3_f1, ckpt_s3 = run_stage3_swa(
        model, ema, train_ldr3, val_ldr3, device,
        trial_p, tune_cfg, trial_dir, trial,
        prev_best_val=s2_best_val,
        epoch_offset=tune_cfg["s2_epochs"])

    print(f"[Trial {trial.number}] Stage 3 best val F1 = {s3_f1:.4f}")

    # ── Best F1 across all stages ────────────────────────────────────
    final_f1 = max(s1_f1, s2_f1, s3_f1)
    print(f"[Trial {trial.number}] ★ Final F1 = {final_f1:.4f}")

    # Save combined best F1 checkpoint tag
    trial.set_user_attr("s1_f1",    s1_f1)
    trial.set_user_attr("s2_f1",    s2_f1)
    trial.set_user_attr("s3_f1",    s3_f1)
    trial.set_user_attr("best_stage", "s1" if final_f1 == s1_f1 else
                                      "s2" if final_f1 == s2_f1 else "s3")

    # Optionally clean up trial checkpoints to save disk
    if not tune_cfg["keep_trial_ckpts"]:
        for f in ["best_s1.pth", "best_s2.pth"]:
            fp = os.path.join(trial_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    torch.cuda.empty_cache()
    return final_f1


# ══════════════════════════════════════════════════════════════════════
#  CALLBACKS  (live CSV logging + best ckpt promotion)
# ══════════════════════════════════════════════════════════════════════

class LiveCSVCallback:
    def __init__(self, path: str):
        self.path = path
        self.rows: List[dict] = []

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        row = {"trial": trial.number, "value": trial.value,
               "state": trial.state.name}
        row.update(trial.params)
        row.update({f"user_{k}": v for k, v in trial.user_attrs.items()})
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        print(f"\n[Callback] Trial {trial.number} | F1={trial.value:.4f} | "
              f"Best so far: {study.best_value:.4f}")


class BestCheckpointCallback:
    """Promotes best trial's Stage-3 checkpoint to study_dir/best_overall.pth."""
    def __init__(self, study_dir: str):
        self.study_dir  = study_dir
        self.best_value = -float("inf")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.value is None:
            return
        if trial.value > self.best_value:
            self.best_value = trial.value
            src_dir = os.path.join(self.study_dir, f"trial_{trial.number:04d}")
            for stage in ["best_s3.pth", "best_s2.pth", "best_s1.pth"]:
                src = os.path.join(src_dir, stage)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(self.study_dir, "best_overall.pth"))
                    print(f"[BestCkpt] Promoted trial {trial.number} {stage} → "
                          f"best_overall.pth  (F1={trial.value:.4f})")
                    break


# ══════════════════════════════════════════════════════════════════════
#  RESULTS EXPORT
# ══════════════════════════════════════════════════════════════════════

def export_results(study: optuna.Study, study_dir: str) -> None:
    best = study.best_trial
    out  = {
        "best_trial":  best.number,
        "best_f1":     best.value,
        "params":      best.params,
        "user_attrs":  best.user_attrs,
    }
    with open(os.path.join(study_dir, "optuna_best.json"), "w") as f:
        json.dump(out, f, indent=2)

    importance = optuna.importance.get_param_importances(study)
    imp_df     = pd.DataFrame(
        list(importance.items()), columns=["param", "importance"]
    ).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(study_dir, "param_importance.csv"), index=False)

    print("\n" + "═" * 60)
    print("  OPTUNA HPO COMPLETE")
    print("═" * 60)
    print(f"  Best trial  : {best.number}")
    print(f"  Best F1     : {best.value:.4f}")
    print(f"  Stage F1s   : S1={best.user_attrs.get('s1_f1', '?'):.4f}  "
          f"S2={best.user_attrs.get('s2_f1', '?'):.4f}  "
          f"S3={best.user_attrs.get('s3_f1', '?'):.4f}")
    print("\n  Top-10 most important hyperparameters:")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['param']:40s}  {row['importance']:.4f}")
    print(f"\n  Results saved to: {study_dir}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna HPO for SpectralQuadNet v8 — 90-class Rice HSI")
    p.add_argument("--n-trials",    type=int,   default=60,
                   help="Number of Optuna trials (default: 60)")
    p.add_argument("--study-name",  type=str,   default="sqn_hpo_v1",
                   help="Optuna study name / SQLite file name")
    p.add_argument("--output-dir",  type=str,   default="./optuna_output",
                   help="Output directory for study artifacts")
    p.add_argument("--full-run",    action="store_true",
                   help="Use full epoch budgets (s1=200, s2=100, s3=40)")
    p.add_argument("--n-jobs",      type=int,   default=1,
                   help="Parallel Optuna workers (1 recommended for GPU)")
    p.add_argument("--keep-ckpts",  action="store_true",
                   help="Keep per-trial intermediate checkpoints")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--timeout",     type=float, default=None,
                   help="Wall-clock timeout in seconds for the study")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    study_dir = args.output_dir
    Path(study_dir).mkdir(parents=True, exist_ok=True)

    base_cfg = dict(BASE_CFG)
    base_cfg["seed"] = args.seed

    tune_cfg = dict(TUNE_CFG)
    tune_cfg["keep_trial_ckpts"] = args.keep_ckpts
    if args.full_run:
        tune_cfg["s1_epochs"] = 200
        tune_cfg["s2_epochs"] = 100
        tune_cfg["s3_epochs"] = 40
        print("[INFO] Full epoch budget enabled (200/100/40).")
    else:
        print(f"[INFO] Tuning epoch budget: "
              f"S1={tune_cfg['s1_epochs']}  "
              f"S2={tune_cfg['s2_epochs']}  "
              f"S3={tune_cfg['s3_epochs']}")

    # Pre-load splits (shared across all trials for fair comparison)
    splits = build_splits(base_cfg)
    all_labels, train_idx, val_idx, test_idx = splits
    print(f"[INFO] Data splits — Train: {len(train_idx):,}  "
          f"Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    # ── Study setup ────────────────────────────────────────────────
    storage_path = f"sqlite:///{os.path.join(study_dir, args.study_name + '.db')}"

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,               # group correlated params (head_lr + lr_ratio)
        warn_independent_sampling=False,
        seed=args.seed,
        n_startup_trials=max(10, args.n_trials // 6),  # random exploration phase
        constant_liar=True,       # robust for n_jobs > 1
    )

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=tune_cfg["s1_epochs"] + tune_cfg["s2_epochs"],
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_path,
        direction="maximize",     # maximize macro-F1
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,      # resume if study already exists
    )

    # ── Warm-start with known-good configs ─────────────────────────
    known_good_params = [
        # Config mirroring original training script defaults
        {
            "wl_embed_dim": 16, "tower_ch": 80,
            "specf_patch": 8,   "specf_dim": 128, "specf_heads": 4,
            "specf_layers": 4,  "specf_drop": 0.15,
            "fusion_heads": 4,  "fusion_drop": 0.10,
            "aug_heavy_band_drop": 0.70, "aug_heavy_cutout": 0.50,
            "aug_heavy_noise": 0.40,     "aug_heavy_warp": 0.35,
            "aug_heavy_shift": 0.30,     "aug_heavy_mult": 0.30,
            "aug_light_band_drop": 0.30, "aug_light_cutout": 0.20,
            "aug_light_noise": 0.20,     "aug_light_warp": 0.15,
            "aug_light_shift": 0.15,     "aug_light_mult": 0.15,
            "noise_std": 0.02,  "max_cutout_bands": 20,
            "s1_batch": 64,     "s1_max_lr": 8e-4, "s1_dropout": 0.30,
            "s1_mixup": 0.4,    "s1_warmup_pct": 0.15, "s1_accum": 2,
            "label_smoothing": 0.05, "weight_decay": 2e-4, "grad_clip": 1.0,
            "ema_decay": 0.9999,
            "s2_batch": 64,     "s2_head_lr": 1.5e-4, "s2_lr_ratio": 10.0,
            "s2_min_lr_frac": 1e-3, "s2_warmup_ep": 5, "s2_dropout": 0.10,
            "s2_arcface_s": 32.0,   "s2_arcface_m": 0.30, "s2_arcface_m0": 0.02,
            "s2_margin_warmup_ep": 40, "s2_label_smooth": 0.0,
            "proto_weight": 0.20,   "proto_temp": 0.10,
            "bal_n_cls": 16,        "bal_n_spc": 4,
            "s3_swa_lr": 8e-5,      "s3_cycle_len": 5,
        },
    ]
    for p in known_good_params:
        study.enqueue_trial(p)

    # ── Callbacks ──────────────────────────────────────────────────
    csv_cb    = LiveCSVCallback(os.path.join(study_dir, "optuna_history.csv"))
    ckpt_cb   = BestCheckpointCallback(study_dir)

    # ── Run optimisation ──────────────────────────────────────────
    study.optimize(
        lambda t: objective(t, base_cfg, tune_cfg, splits, study_dir),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        callbacks=[csv_cb, ckpt_cb],
        gc_after_trial=True,
        catch=(RuntimeError,),     # GPU OOM → skip trial, do not abort study
    )

    # ── Export final results ────────────────────────────────────────
    export_results(study, study_dir)


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(BASE_CFG["output_dir"], "optuna.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Study interrupted by user. Partial results saved.")
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)