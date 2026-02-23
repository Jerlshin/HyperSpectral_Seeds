
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

    # ── Stage 1 ────────────────────────────────────────────────────────
    "s1_epochs":      200,
    "s1_batch":       64,
    "s1_max_lr":      8e-4,
    "s1_dropout":     0.30,
    "s1_mixup":       0.4,
    "s1_patience":    40,        # v6: 35 → 40 (model still improving at end)
    "s1_accum":       2,

    # ── Stage 2 ────────────────────────────────────────────────────────
    "s2_epochs":      110,
    "s2_batch":       64,
    "s2_head_lr":     1.5e-4,
    "s2_back_lr":     1.5e-5,
    "s2_min_lr":      1e-7,
    "s2_warmup_ep":   5,          # linear warmup before SGDR kicks in
    "s2_sgdr_T0":     15,         # v6: SGDR first cycle length
    "s2_sgdr_Tmult":  2,          # v6: cycle multiplier → restarts at ep20, ep50
    "s2_dropout":     0.10,
    "s2_patience":    35,         # v6: 30 → 35 (allow SGDR restarts to recover)
    "s2_arcface_s":   32.0,
    "s2_arcface_m":   0.35,       # v6: 0.30 → 0.35 (fine-grained best practice)
    "s2_arcface_m0":  0.02,
    "s2_margin_warmup_ep": 50,    # v6: 40 → 50 (slower warmup avoids early spikes)
    "s2_label_smooth": 0.0,
    "s2_focal_gamma": 1.5,        # v6: Focal loss gamma for hard class weighting

    # ProtoNCE
    "proto_weight":   0.20,
    "proto_temp":     0.10,

    # Class-balanced sampler
    "bal_n_cls":      16,
    "bal_n_spc":      4,

    # ── Stage 3 ────────────────────────────────────────────────────────
    "s3_epochs":      60,         # v6: 40 → 60 (model still learning at ep40)
    "s3_swa_lr":      5e-5,       # v6: 8e-5 → 5e-5 (reduces val dips at restarts)
    "s3_cycle_len":   8,          # v6: 5 → 8 (more time at low LR for stable snaps)

    # Shared
    "label_smoothing": 0.05,
    "weight_decay":   2e-4,
    "grad_clip":      1.0,
    "ema_decay":      0.9999,

    # TTA
    "tta_spatial":    8,          # spatial views (rot90 × flip)
    "tta_spectral":   4,          # v6: spectral shift views (±3, ±6 bands)

    # Architecture (unchanged)
    "wl_embed_dim":   16,
    "specf_patch":    8,
    "specf_dim":      128,
    "specf_heads":    4,
    "specf_layers":   4,
    "specf_drop":     0.15,
    "fusion_heads":   4,
    "fusion_drop":    0.10,

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
            if isinstance(m, nn.Dropout): m.p = p

    def state_dict(self)                -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    _AUG_PROFILES = {
        "heavy": dict(band_drop=0.70, cutout=0.50, noise=0.40,
                      warp=0.35, shift=0.30, mult=0.30),
        "light": dict(band_drop=0.30, cutout=0.20, noise=0.20,
                      warp=0.15, shift=0.15, mult=0.15),
        "none":  None,
    }

    def __init__(self, patches_path, labels_path, indices,
                 aug_strength="none", max_cutout_bands=20, noise_std=0.02):
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.probs            = self._AUG_PROFILES.get(aug_strength)
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std

    def __len__(self): return len(self.indices)

    def _band_dropout(self, x):
        return x * (torch.rand(x.shape[0]) > 0.04).float().view(-1,1,1)

    def _band_cutout(self, x):
        x   = x.clone(); nb = x.shape[0]
        cut = torch.randint(1, max(2, self.max_cutout_bands), (1,)).item()
        st  = torch.randint(0, max(1, nb - cut), (1,)).item()
        x[st:st+cut] = 0.0; return x

    def _spectral_noise(self, x):
        return x + torch.randn_like(x) * self.noise_std

    def _spectral_warp(self, x):
        C, H, W = x.shape
        scale   = 1.0 + random.uniform(-0.10, 0.10)
        new_C   = max(1, int(C * scale))
        if new_C == C: return x
        xp     = x.permute(1,2,0).reshape(-1,1,C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s      = (new_C - C) // 2
            warped = warped[:,:,s:s+C]
        else:
            lo     = (C - new_C) // 2
            warped = F.pad(warped, (lo, C - new_C - lo))
        return warped.reshape(H,W,C).permute(2,0,1)

    def _spectral_shift(self, x):
        return torch.roll(x, random.randint(-8, 8), dims=0)

    def _mult_noise(self, x):
        return x * (1.0 + torch.randn(x.shape[0],1,1) * 0.05)

    def _spatial(self, x):
        if torch.rand(1) < 0.5: x = torch.flip(x, [2])
        if torch.rand(1) < 0.5: x = torch.flip(x, [1])
        return torch.rot90(x, torch.randint(0,4,(1,)).item(), [1,2])

    def __getitem__(self, idx):
        ri    = self.indices[idx]
        patch = torch.from_numpy(self.patches[ri].copy()).float()
        label = torch.tensor(self.labels[ri], dtype=torch.long)
        if self.probs is not None:
            p = self.probs
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
    def __init__(self, train_labels, n_cls=16, n_spc=4):
        self.n_cls   = n_cls; self.n_spc = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False)
            batch  = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(rng.choice(pool, self.n_spc,
                                        replace=len(pool) < self.n_spc).tolist())
            yield batch

    def __len__(self): return self._n


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION (Mixup + CutMix)
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def _cutmix(x, y, alpha):
    lam       = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx       = torch.randperm(B, device=x.device)
    r         = math.sqrt(1.0 - lam)
    ch, cw    = int(H*r), int(W*r)
    cx, cy    = random.randint(0,W), random.randint(0,H)
    x1 = max(cx-cw//2,0); x2 = min(cx+cw//2,W)
    y1 = max(cy-ch//2,0); y2 = min(cy+ch//2,H)
    xm = x.clone(); xm[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    return xm, y, y[idx], 1.0 - (x2-x1)*(y2-y1)/(W*H)

def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1) < 0.5 else _cutmix)(x, y, alpha)

def mixed_loss(criterion, logits, y_a, y_b, lam):
    return lam*criterion(logits, y_a) + (1-lam)*criterion(logits, y_b)


# ══════════════════════════════════════════════════════════════════════
#  FOCAL LOSS  [v6 NEW — handles hard classes 41/49/51/52]
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy samples, focuses on hard classes.
    L = (1 - p_t)^γ × CE,  where p_t = softmax prob of correct class.
    γ=1.5 is conservative — preserves stable ArcFace angular geometry
    while emphasising the confusion zone.
    Ref: Lin et al. Focal Loss for Dense Object Detection, ICCV 2017.
    """
    def __init__(self, gamma: float = 1.5) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, reduction="none")
        p_t   = torch.exp(-ce)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


# ══════════════════════════════════════════════════════════════════════
#  ARCFACE HEAD
# ══════════════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    def __init__(self, in_dim, num_classes, s=32.0, m=0.35):
        super().__init__()
        self.weight    = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s         = s
        self.default_m = m
        self._cached_m = None
        self._precompute(m)

    def _precompute(self, m):
        self._cached_m = m
        self._cosm = math.cos(m); self._sinm = math.sin(m)
        self._th   = math.cos(math.pi - m)
        self._mm   = math.sin(math.pi - m) * m

    def forward(self, x, labels=None, m=None):
        if m is not None and m != self._cached_m: self._precompute(m)
        cosine = F.linear(F.normalize(x,dim=1), F.normalize(self.weight,dim=1))
        cosine = cosine.clamp(-1+1e-6, 1-1e-6)
        if labels is None or not self.training: return cosine * self.s
        sine  = torch.sqrt(torch.clamp(1-cosine**2, min=1e-6))
        phi   = cosine*self._cosm - sine*self._sinm
        phi   = torch.where(cosine > self._th, phi, cosine - self._mm)
        oh    = torch.zeros_like(cosine).scatter_(1, labels.view(-1,1).long(), 1.0)
        return ((oh*phi) + ((1-oh)*cosine)) * self.s


# ══════════════════════════════════════════════════════════════════════
#  PROTO-NCE LOSS
# ══════════════════════════════════════════════════════════════════════

class ProtoNCELoss(nn.Module):
    """Prototypical NCE: per-class mean prototype → NCE cross-entropy."""
    def __init__(self, temperature=0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        classes = labels.unique()
        if len(classes) < 2: return features.new_tensor(0.0, requires_grad=True)
        protos = F.normalize(torch.stack([features[labels==c].mean(0) for c in classes]), dim=1)
        sim    = torch.mm(features, protos.T) / self.temperature
        c2l    = {c.item(): i for i,c in enumerate(classes)}
        local  = torch.tensor([c2l[y.item()] for y in labels],
                               dtype=torch.long, device=features.device)
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  MASKED SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(x):
    x32  = x.float(); B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H*W)
    mask = (flat.abs().sum(1, keepdim=True) > 1e-5).float()
    cnt  = mask.sum(2).clamp(min=1.0)
    mean = (flat*mask).sum(2) / cnt
    std  = ((flat**2*mask).sum(2)/cnt - mean**2).clamp(min=1e-6).sqrt()
    mx   = flat.masked_fill(mask.expand_as(flat)==0, -1e4).max(2).values
    mx   = mx.masked_fill(mx < -9999.0, 0.0)
    return (torch.nan_to_num(mean,0), torch.nan_to_num(std,0), torch.nan_to_num(mx,0))


# ══════════════════════════════════════════════════════════════════════
#  WAVELENGTH POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class WavelengthPositionalEncoding(nn.Module):
    def __init__(self, num_bands=256, embed_dim=16):
        super().__init__()
        wl   = torch.linspace(0.0, 1.0, num_bands)
        half = embed_dim // 2
        freq = torch.exp(torch.arange(half).float() * -(math.log(1e4)/max(half-1,1)))
        enc  = torch.zeros(num_bands, embed_dim)
        enc[:,:half] = torch.sin(wl.unsqueeze(1)*freq.unsqueeze(0))
        enc[:,half:] = torch.cos(wl.unsqueeze(1)*freq.unsqueeze(0))
        self.register_buffer("enc", enc)
        self.proj = nn.Linear(embed_dim, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.01); nn.init.zeros_(self.proj.bias)

    def forward(self): return self.proj(self.enc).squeeze(-1).view(1,1,-1)


# ══════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class SpectralSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels//reduction, 16)
        self.gate = nn.Sequential(nn.Linear(channels,mid,bias=False), nn.GELU(),
                                  nn.Linear(mid,channels,bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x.mean([2,3])).view(x.shape[0],x.shape[1],1,1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel//2
        self.conv1 = nn.Conv1d(in_ch,out_ch,kernel,padding=pad,bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch,out_ch,kernel,padding=pad,bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch,out_ch,1,bias=False),
                                    nn.BatchNorm1d(out_ch))
                      if in_ch!=out_ch else nn.Identity())
    def forward(self, x):
        return F.gelu(self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x))))) + self.skip(x))


class CBAM(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(c//r, 8)
        self.ch  = nn.Sequential(nn.Conv2d(c,mid,1,bias=False),nn.GELU(),nn.Conv2d(mid,c,1,bias=False))
        self.sp  = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.Sigmoid())
    def forward(self, x):
        x = x * torch.sigmoid(self.ch(x.mean([2,3],keepdim=True)) + self.ch(x.amax([2,3],keepdim=True)))
        return x * self.sp(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1))


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid = max(out_ch//2, in_ch)
        self.c1 = nn.Conv2d(in_ch,mid,1,bias=False); self.n1 = nn.GroupNorm(min(8,mid),mid)
        self.c2 = nn.Conv2d(mid,mid,3,stride,1,bias=False); self.n2 = nn.GroupNorm(min(8,mid),mid)
        self.c3 = nn.Conv2d(mid,out_ch,1,bias=False); self.n3 = nn.GroupNorm(min(8,out_ch),out_ch)
        self.skip = (nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride=stride,bias=False),
                                   nn.GroupNorm(min(8,out_ch),out_ch))
                     if (stride!=1 or in_ch!=out_ch) else nn.Identity())
    def forward(self, x):
        return F.gelu(self.n3(self.c3(F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) + self.skip(x))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A — SPECTRAL PROFILE
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None):
        super().__init__()
        self.wl_enc  = wl_enc
        mk = lambda k: nn.Sequential(ResBlock1D(2,tower_ch//2,k),
                                     ResBlock1D(tower_ch//2,tower_ch,k),
                                     ResBlock1D(tower_ch,tower_ch,k))
        self.tower_s = mk(3); self.tower_m = mk(7); self.tower_l = mk(15)
        self.proj    = nn.Sequential(nn.Linear(tower_ch*6,out_dim),
                                     nn.BatchNorm1d(out_dim),nn.GELU(),nn.Dropout(0.1))
    @staticmethod
    def _gp(f): return torch.cat([f.mean(2),f.max(2).values],1)
    def forward(self, ms):
        s = ms.unsqueeze(1); d = F.pad(torch.diff(s,dim=2),(0,1))
        x = torch.cat([s,d],1)
        if self.wl_enc: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))],1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None):
        super().__init__()
        self.wl_enc  = wl_enc
        mk = lambda k: nn.Sequential(ResBlock1D(3,tower_ch//2,k),
                                     ResBlock1D(tower_ch//2,tower_ch,k),
                                     ResBlock1D(tower_ch,tower_ch,k))
        self.tower_s = mk(3); self.tower_m = mk(7); self.tower_l = mk(15)
        self.proj    = nn.Sequential(nn.Linear(tower_ch*6,out_dim),
                                     nn.BatchNorm1d(out_dim),nn.GELU(),nn.Dropout(0.1))
    @staticmethod
    def _gp(f): return torch.cat([f.mean(2),f.max(2).values],1)
    def forward(self, ms, ss, mx):
        x = torch.stack([ms,ss,mx],1)
        if self.wl_enc: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))],1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands=256, out_dim=256):
        super().__init__()
        self.band_reduce = nn.Sequential(nn.Conv2d(num_bands,32,1,bias=False),
                                         nn.GroupNorm(8,32),nn.GELU())
        self.stages = nn.Sequential(ResBlock2D(32,64,2),CBAM(64),
                                    ResBlock2D(64,128,2),CBAM(128),
                                    ResBlock2D(128,192,2),CBAM(192),
                                    ResBlock2D(192,out_dim,2))
        self.proj = nn.Sequential(nn.Linear(out_dim*2,out_dim),
                                  nn.BatchNorm1d(out_dim),nn.GELU())
    @staticmethod
    def _pn(x): return x.sign() * x.abs().clamp(1e-8).sqrt()
    def forward(self, x):
        h = self.stages(self.band_reduce(x))
        return self.proj(F.normalize(torch.cat([self._pn(h.mean([2,3])),
                                                self._pn(h.amax([2,3]))],1),dim=1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER
# ══════════════════════════════════════════════════════════════════════

class _PreLNBlock(nn.Module):
    def __init__(self, d, heads, d_ff, drop):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Dropout(drop),
                                  nn.Linear(d_ff,d),nn.Dropout(drop))
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        h,_ = self.attn(self.ln1(x),self.ln1(x),self.ln1(x),need_weights=False)
        x   = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))


class SpecFormerBranch(nn.Module):
    def __init__(self, num_bands=256, patch_size=8, d_model=128,
                 n_heads=4, n_layers=4, out_dim=256, dropout=0.15):
        super().__init__()
        n_p = num_bands // patch_size
        self.patch_size = patch_size; self.n_patches = n_p
        self.patch_proj = nn.Sequential(nn.Linear(patch_size,d_model,bias=False),
                                        nn.LayerNorm(d_model))
        wl_n = (torch.linspace(WL_MIN,WL_MAX,n_p)-WL_MIN)/(WL_MAX-WL_MIN)
        half = d_model//2
        freq = torch.exp(torch.arange(half).float() * -(math.log(1e4)/max(half-1,1)))
        pe   = torch.zeros(n_p, d_model)
        pe[:,:half] = torch.sin(wl_n.unsqueeze(1)*freq.unsqueeze(0))
        pe[:,half:] = torch.cos(wl_n.unsqueeze(1)*freq.unsqueeze(0))
        self.register_buffer("wl_pe", pe)
        self.cls    = nn.Parameter(torch.zeros(1,1,d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([_PreLNBlock(d_model,n_heads,d_model*2,dropout)
                                     for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.proj   = nn.Sequential(nn.Linear(d_model,out_dim),nn.BatchNorm1d(out_dim),
                                    nn.GELU(),nn.Dropout(dropout))
    def forward(self, ms):
        B = ms.shape[0]
        x = ms.float().view(B,self.n_patches,self.patch_size)
        x = self.patch_proj(x) + self.wl_pe.unsqueeze(0)
        x = torch.cat([self.cls.expand(B,-1,-1), x], 1)
        for blk in self.blocks: x = blk(x)
        return self.proj(self.norm(x)[:,0])


# ══════════════════════════════════════════════════════════════════════
#  BRANCH CROSS-ATTENTION FUSION
# ══════════════════════════════════════════════════════════════════════

class BranchCrossAttention(nn.Module):
    def __init__(self, d=256, n_heads=4, dropout=0.10):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(dropout),
                                  nn.Linear(d*2,d),nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(1))
    def forward(self, branches):
        x   = torch.stack(branches,1)
        h,_ = self.attn(self.ln1(x),self.ln1(x),self.ln1(x),need_weights=False)
        x   = x + self.gate * self.drop(h)
        return (x + self.drop(self.ff(self.ln2(x)))).flatten(1)


# ══════════════════════════════════════════════════════════════════════
#  SPECTRALQUADNET v8
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    def __init__(self, num_classes=90, num_bands=256, dropout=0.30,
                 wl_embed_dim=16, cfg=None):
        super().__init__()
        cfg = cfg or CONFIG
        self.se       = SpectralSE(num_bands, 16)
        self.wl_enc   = WavelengthPositionalEncoding(num_bands, wl_embed_dim)
        self.branch_a = SpectralProfileBranch(256, 80, self.wl_enc)
        self.branch_b = SpectralStatsBranch(  256, 80, self.wl_enc)
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(num_bands, cfg["specf_patch"], cfg["specf_dim"],
                                         cfg["specf_heads"], cfg["specf_layers"], 256,
                                         cfg["specf_drop"])
        self.cross_attn = BranchCrossAttention(256, cfg["fusion_heads"], cfg["fusion_drop"])
        self.embed_net  = nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),
                                        nn.GELU(),nn.Dropout(dropout),
                                        nn.Linear(512,256),nn.BatchNorm1d(256))
        self.linear_head  = nn.Sequential(nn.GELU(),nn.Dropout(dropout*0.4),
                                          nn.Linear(256,num_classes))
        self.arcface_head = ArcFaceHead(256, num_classes,
                                        cfg["s2_arcface_s"], cfg["s2_arcface_m"])
        self._use_arcface = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def set_dropout(self, p):
        for m in self.modules():
            if isinstance(m,nn.Dropout): m.p = p

    def use_arcface(self, flag): self._use_arcface = flag

    def freeze_head(self, which):
        head = self.linear_head if which=="linear" else self.arcface_head
        for p in head.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which):
        head = self.linear_head if which=="linear" else self.arcface_head
        for p in head.parameters(): p.requires_grad_(True)

    def forward(self, x, labels=None, return_embed=False, arc_m=None):
        x   = self.se(x)
        ms, ss, mx = masked_spectral_stats(x)
        emb = self.embed_net(self.cross_attn([
            self.branch_a(ms),
            self.branch_b(ms, ss, mx),
            self.branch_c(x),
            self.branch_d(ms),
        ]))
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

def init_arcface_from_linear(model):
    linear = model.linear_head[-1]
    arc    = model.arcface_head
    if not isinstance(linear, nn.Linear):
        raise RuntimeError("Expected last layer of linear_head to be nn.Linear")
    with torch.no_grad():
        arc.weight.data.copy_(F.normalize(linear.weight.data.clone(), dim=1))
    print("[INFO] ArcFace weights initialised from linear head.")


# ══════════════════════════════════════════════════════════════════════
#  TTA  [v6: +4 spectral-shift views = 12 total]
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor,
                n_spatial: int = 8, n_spectral: int = 4) -> torch.Tensor:
    """
    Test-Time Augmentation over:
      • 8 spatial views  : rot90(k=0..3) × flip(True/False)
      • 4 spectral views : torch.roll(±3, ±6 bands) along spectral axis
                           Exploits sensor bandwidth overlap — small
                           spectral shifts are semantically equivalent.

    Total: 12 views.  Ensemble by averaging logits.
    """
    device = x.device
    logits = []

    # ── Spatial ────────────────────────────────────────────────────────
    spatial = [(k, f) for k in range(4) for f in (False, True)][:n_spatial]
    for k, flip in spatial:
        aug = torch.rot90(x, k, [2,3])
        if flip: aug = torch.flip(aug, [3])
        with autocast(device_type=device.type):
            logits.append(model(aug))

    # ── Spectral shift ─────────────────────────────────────────────────
    #   shifts spread evenly around 0 with step = 256 / n_spectral
    step   = max(256 // max(n_spectral * 2, 1), 1)
    shifts = [step * i for i in range(1, n_spectral // 2 + 1)]
    shifts = [-s for s in shifts] + shifts                  # e.g. [-6,-3,3,6]
    for sh in shifts[:n_spectral]:
        aug = torch.roll(x, sh, dims=1)                    # roll along band axis
        with autocast(device_type=device.type):
            logits.append(model(aug))

    return torch.stack(logits).mean(0)


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits():
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(train_idx, val_idx, test_idx, batch_train,
                  balanced=False, all_labels=None, train_aug="none"):
    nw = CONFIG["num_workers"]
    kw = dict(num_workers=nw, pin_memory=True)
    ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                         train_idx, aug_strength=train_aug)
    if balanced and all_labels is not None:
        samp      = ClassBalancedBatchSampler(all_labels[train_idx],
                                              CONFIG["bal_n_cls"], CONFIG["bal_n_spc"])
        train_ldr = DataLoader(ds, batch_sampler=samp,
                               persistent_workers=True, prefetch_factor=2, **kw)
    else:
        train_ldr = DataLoader(ds, batch_size=batch_train, shuffle=True,
                               persistent_workers=True, prefetch_factor=2, **kw)
    val_ldr  = DataLoader(RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], val_idx),
                          batch_size=64, shuffle=False, **kw)
    test_ldr = DataLoader(RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], test_idx),
                          batch_size=64, shuffle=False, **{**kw, "num_workers":2})
    return train_ldr, val_ldr, test_ldr


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISERS
# ══════════════════════════════════════════════════════════════════════

def _wd_split(named_params, lr):
    wd, no_wd = [], []
    for name, p in named_params:
        if not p.requires_grad: continue
        (no_wd if (p.ndim==1 or name.endswith(".bias")) else wd).append(p)
    return [{"params": wd,    "lr": lr, "weight_decay": CONFIG["weight_decay"]},
            {"params": no_wd, "lr": lr, "weight_decay": 0.0}]


def build_optimizer_s1(model, lr):
    return optim.AdamW(_wd_split(model.named_parameters(), lr))


def build_optimizer_s2(model, head_lr, backbone_lr):
    """Discriminative LR: head faster, backbone slow to preserve Stage-1 repr."""
    head_params, back_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.startswith("arcface_head"): head_params.append((name, p))
        else:                               back_params.append((name, p))
    return optim.AdamW(_wd_split(head_params, head_lr) +
                       _wd_split(back_params, backbone_lr))


def build_optimizer_s3(model, lr):
    return optim.AdamW(_wd_split(model.named_parameters(), lr))


# ══════════════════════════════════════════════════════════════════════
#  LR SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def cosine_warmup_scheduler(optimizer, warmup_ep, total_ep, eta_min_frac=1e-3):
    """Monotone cosine with linear warmup — used for Stage 1 (via OneCycleLR)."""
    def _l(ep):
        if ep < warmup_ep: return ep / max(warmup_ep, 1)
        t = (ep-warmup_ep) / max(total_ep-warmup_ep, 1)
        return eta_min_frac + 0.5*(1-eta_min_frac)*(1+math.cos(math.pi*t))
    return optim.lr_scheduler.LambdaLR(optimizer, _l)


def sgdr_scheduler(optimizer, warmup_ep=5, T_0=15, T_mult=2,
                   eta_min_frac=1e-3) -> optim.lr_scheduler.LambdaLR:
    """
    SGDR (Stochastic Gradient Descent with Warm Restarts) per-epoch.
    Schedule: linear warmup → cosine cycle T_0 → restart → cosine cycle
              T_0×T_mult → restart → cosine cycle T_0×T_mult² → ...

    With warmup=5, T_0=15, T_mult=2:
      ep 0-4  : linear warmup (0 → peak)
      ep 5-19 : cosine annealing (T=15)
      ep 20   : RESTART → peak LR
      ep 20-49: cosine annealing (T=30)
      ep 50   : RESTART → peak LR
      ep 50+  : cosine annealing (T=60)

    Scale is multiplicative, so head_lr and backbone_lr both follow
    the same λ curve, preserving their ratio throughout.
    """
    def _l(ep):
        if ep < warmup_ep:
            return max(ep / max(warmup_ep, 1), 1e-6)
        t          = ep - warmup_ep
        cycle_len  = T_0
        elapsed    = 0
        while t >= elapsed + cycle_len:
            elapsed   += cycle_len
            cycle_len  = max(int(cycle_len * T_mult), 1)
        ratio = (t - elapsed) / max(cycle_len, 1)
        return eta_min_frac + 0.5*(1-eta_min_frac)*(1+math.cos(math.pi*ratio))
    return optim.lr_scheduler.LambdaLR(optimizer, _l)


# ══════════════════════════════════════════════════════════════════════
#  ARCFACE MARGIN WARMUP
# ══════════════════════════════════════════════════════════════════════

def arcface_margin(ep, m0, m_target, warmup_ep):
    if ep >= warmup_ep: return m_target
    return m0 + (m_target-m0)*0.5*(1-math.cos(math.pi*ep/max(warmup_ep,1)))


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler, ema, device,
                    scheduler=None, use_mixup=True, mixup_alpha=0.4,
                    supcon=None, supcon_weight=0.0, accum_steps=1,
                    arc_m=None):
    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_amp = (supcon is None) and (scaler is not None)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_in, y_a, y_b, lam = mixed_aug(x,y,mixup_alpha) if use_mixup else (x,y,y,1.0)

        with autocast(device_type=device.type, enabled=use_amp):
            if supcon is not None:
                logits, emb = model(x_in, y_a, return_embed=True, arc_m=arc_m)
                loss = (1-supcon_weight)*criterion(logits, y_a) + \
                       supcon_weight*supcon(emb, y_a)
            else:
                logits = model(x_in,
                               labels=y_a if (model._use_arcface and not use_mixup) else None,
                               arc_m=arc_m)
                loss   = mixed_loss(criterion, logits, y_a, y_b, lam)

        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at step {step}. Skipping.")
            optimizer.zero_grad(set_to_none=True); continue

        (scaler.scale(loss/accum_steps).backward() if use_amp
         else (loss/accum_steps).backward())

        if (step+1) % accum_steps == 0:
            if use_amp: scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            if use_amp: scaler.step(optimizer); scaler.update()
            else:       optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler: scheduler.step()
            if ema:       ema.update(model)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == y).float().mean().item()

    n = len(loader)
    return total_loss/n, total_acc/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type):
            logits = model(x)
        preds.append(logits.argmax(1).cpu()); targets.append(y)
    p, t = torch.cat(preds), torch.cat(targets)
    return f1_score(t,p,average="macro",zero_division=0), accuracy_score(t,p)


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def stage_ckpt_path(stage): return os.path.join(CONFIG["output_dir"],f"best_stage{stage}.pth")
def stage_exists(stage):    return os.path.isfile(stage_ckpt_path(stage))

def latest_completed_stage():
    for s in (3,2,1):
        if stage_exists(s): return s
    return 0

def save_ckpt(path, epoch, stage, model, ema, val_acc, val_f1):
    torch.save({"epoch":epoch,"stage":stage,"model":model.state_dict(),
                "ema":ema.state_dict(),"val_acc":val_acc,"val_f1":val_f1,
                "use_arcface":model._use_arcface}, path)

def load_ckpt(path, model, ema, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); ema.load_state_dict(ckpt["ema"])
    use_af = ckpt.get("use_arcface", False)
    model.use_arcface(use_af); ema.shadow.use_arcface(use_af)
    return ckpt


# ══════════════════════════════════════════════════════════════════════
#  BN UPDATE (SWA)
# ══════════════════════════════════════════════════════════════════════

def update_bn_stats(loader, model, device):
    model.train()
    for m in model.modules():
        if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
            m.reset_running_stats(); m.momentum = None
    with torch.no_grad():
        for x,_ in loader: model(x.to(device, non_blocking=True))
    model.eval()


# ══════════════════════════════════════════════════════════════════════
#  STAGE RUNNERS
# ══════════════════════════════════════════════════════════════════════

def _hdr(title, epochs):
    w=66; print(f"\n{'═'*w}\n  {title}  [{epochs} epochs max]\n{'═'*w}")


# ── Stage 1 ───────────────────────────────────────────────────────────

def run_stage1(model, ema, train_ldr, val_ldr, device, criterion, best_ckpt):
    model.use_arcface(False)
    model.unfreeze_head("linear")
    model.freeze_head("arcface")

    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"]/25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CONFIG["s1_max_lr"],
            epochs=CONFIG["s1_epochs"],
            steps_per_epoch=math.ceil(len(train_ldr)/CONFIG["s1_accum"]),
            pct_start=0.15, div_factor=25, final_div_factor=1e4, anneal_strategy="cos")

    scaler     = GradScaler()
    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 1 — Heavy Aug + Mixup/CutMix", CONFIG["s1_epochs"])

    for ep in range(1, CONFIG["s1_epochs"]+1):
        tl, ta = train_one_epoch(model, train_ldr, optimizer, criterion, scaler, ema,
                                 device, scheduler=scheduler, use_mixup=True,
                                 mixup_alpha=CONFIG["s1_mixup"], accum_steps=CONFIG["s1_accum"])
        _, va_live = evaluate(model,      val_ldr, device)
        vf1,va_ema = evaluate(ema.shadow, val_ldr, device)
        va_best    = max(va_live, va_ema)
        lr_now     = optimizer.param_groups[0]["lr"]
        saved      = ""

        if va_best > best_acc:
            best_acc, no_improve = va_best, 0
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema, va_best, vf1)
            saved = "  ✓ Saved"
        else:
            no_improve += 1

        print(f"Ep {ep:03d}/{CONFIG['s1_epochs']} │ Loss {tl:.4f}  Train {ta:.1%} │ "
              f"Live {va_live:.1%}  EMA {va_ema:.1%} │ LR {lr_now:.2e}  "
              f"EMA_d {ema.current_decay:.4f}{saved}")

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("arcface")
    return best_acc


# ── Stage 2 ───────────────────────────────────────────────────────────

def run_stage2(model, ema, train_ldr, val_ldr, device, best_ckpt):
    """
    Stage 2: ArcFace + Focal-CE + ProtoNCE + Discriminative LR + SGDR.

    v6 changes vs v5:
      • FocalLoss (γ=1.5) replaces CE — focuses on hard classes
      • SGDR (T0=15, Tmult=2) replaces monotone cosine — restarts at ep20,ep50
      • ArcFace margin target: 0.30 → 0.35, warmup: 40 → 50 epochs
      • Patience: 30 → 35 (allows SGDR restarts to recover)
    """
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)
    model.freeze_head("linear")
    model.unfreeze_head("arcface")

    ema.reinit_from(model)
    ema.set_dropout(CONFIG["s2_dropout"])
    ema.shadow.use_arcface(True)

    # ── Losses ────────────────────────────────────────────────────────
    focal = FocalLoss(gamma=CONFIG["s2_focal_gamma"])
    proto = ProtoNCELoss(temperature=CONFIG["proto_temp"])

    # ── Optimizer + SGDR scheduler ────────────────────────────────────
    optimizer = build_optimizer_s2(model, CONFIG["s2_head_lr"], CONFIG["s2_back_lr"])
    scheduler = sgdr_scheduler(
        optimizer,
        warmup_ep = CONFIG["s2_warmup_ep"],
        T_0       = CONFIG["s2_sgdr_T0"],
        T_mult    = CONFIG["s2_sgdr_Tmult"],
        eta_min_frac = CONFIG["s2_min_lr"] / CONFIG["s2_head_lr"],
    )

    best_acc   = 0.0
    no_improve = 0

    _hdr("Stage 2 — ArcFace + FocalCE + ProtoNCE + SGDR", CONFIG["s2_epochs"])
    print(f"  Head LR: {CONFIG['s2_head_lr']:.1e}  |  "
          f"Backbone LR: {CONFIG['s2_back_lr']:.1e}  (1/10×)")
    print(f"  SGDR: warmup={CONFIG['s2_warmup_ep']} ep, "
          f"T0={CONFIG['s2_sgdr_T0']}, Tmult={CONFIG['s2_sgdr_Tmult']} "
          f"→ restarts at ep {CONFIG['s2_warmup_ep']+CONFIG['s2_sgdr_T0']} and "
          f"ep {CONFIG['s2_warmup_ep']+CONFIG['s2_sgdr_T0']+CONFIG['s2_sgdr_T0']*CONFIG['s2_sgdr_Tmult']}")
    print(f"  Focal γ={CONFIG['s2_focal_gamma']}  |  "
          f"ArcFace margin: {CONFIG['s2_arcface_m0']} → {CONFIG['s2_arcface_m']} "
          f"over {CONFIG['s2_margin_warmup_ep']} ep")

    for ep in range(1, CONFIG["s2_epochs"]+1):
        m_now   = arcface_margin(ep-1, CONFIG["s2_arcface_m0"],
                                 CONFIG["s2_arcface_m"], CONFIG["s2_margin_warmup_ep"])
        proto_w = min(CONFIG["proto_weight"],
                      (ep/max(CONFIG["s2_warmup_ep"]*2,1))*CONFIG["proto_weight"])

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, focal, scaler=None, ema=ema,
            device=device, scheduler=None,   # scheduler stepped per-epoch below
            use_mixup=False, supcon=proto, supcon_weight=proto_w, arc_m=m_now,
        )
        scheduler.step()   # per-epoch

        _, va_live = evaluate(model,      val_ldr, device)
        vf1,va_ema = evaluate(ema.shadow, val_ldr, device)
        va_best    = max(va_live, va_ema)
        head_lr    = optimizer.param_groups[0]["lr"]
        back_lr    = optimizer.param_groups[2]["lr"]
        saved      = ""

        if va_best > best_acc:
            best_acc, no_improve = va_best, 0
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema, va_best, vf1)
            saved = "  ✓ Saved"
        else:
            no_improve += 1

        # Flag SGDR restarts in output
        restart_flag = ""
        if ep == CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"]: restart_flag = " ↻RESTART1"
        if ep == CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"] + \
                 CONFIG["s2_sgdr_T0"] * CONFIG["s2_sgdr_Tmult"]: restart_flag = " ↻RESTART2"

        print(f"Ep {ep:03d}/{CONFIG['s2_epochs']} │ Loss {tl:.4f}  Train {ta:.1%} │ "
              f"Live {va_live:.1%}  EMA {va_ema:.1%} │ "
              f"hLR {head_lr:.1e} bLR {back_lr:.1e}  m={m_now:.3f}{saved}{restart_flag}")

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("linear")
    return best_acc


# ── Stage 3 (SWA) ─────────────────────────────────────────────────────

def run_stage3_swa(model, ema, train_ldr, val_ldr, device, criterion,
                   best_ckpt, prev_best_val: float) -> float:
    """
    Stage 3: Stochastic Weight Averaging.

    v6 changes vs v5:
      • 60 epochs (was 40) — model still learning at ep40 in v5
      • ema=None — prevents BN-stat corruption of EMA shadow
      • Peak LR 8e-5 → 5e-5 — reduces val dips at cycle restarts
      • Cycle length 5 → 8 — more time at low LR for stable snapshots
    """
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)
    ema.shadow.use_arcface(True)

    optimizer    = build_optimizer_s3(model, CONFIG["s3_swa_lr"])
    scaler       = GradScaler()
    focal_s3     = FocalLoss(gamma=1.0)   # mild focal in Stage 3

    swa_state: dict = copy.deepcopy(model.state_dict())
    n_snap          = 1

    _hdr("Stage 3 — Stochastic Weight Averaging", CONFIG["s3_epochs"])
    print(f"  Peak LR: {CONFIG['s3_swa_lr']:.1e}  Cycle: {CONFIG['s3_cycle_len']} ep  "
          f"EMA: disabled (avoids BN corruption)")

    for ep in range(1, CONFIG["s3_epochs"]+1):
        # Cosine cycle within SWA — peak at cycle start, trough at end
        cycle_ep = (ep - 1) % CONFIG["s3_cycle_len"]
        lr_now   = CONFIG["s3_swa_lr"] * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cycle_ep / CONFIG["s3_cycle_len"]))
        )
        for pg in optimizer.param_groups: pg["lr"] = lr_now

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, focal_s3, scaler,
            ema=None,          # v6 FIX: no EMA in Stage 3 (prevents BN corruption)
            device=device, scheduler=None, use_mixup=False,
        )

        # Snapshot at cycle end (low LR = stable model point)
        if ep % CONFIG["s3_cycle_len"] == 0:
            n_snap += 1
            a  = 1.0 / n_snap
            sd = model.state_dict()
            for k in swa_state:
                swa_state[k] = swa_state[k] + a * (sd[k] - swa_state[k])

        _, va_live = evaluate(model, val_ldr, device)
        snap_marker = f"  ★ snap {n_snap}" if ep % CONFIG["s3_cycle_len"] == 0 else ""
        print(f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ Loss {tl:.4f}  Train {ta:.1%} │ "
              f"Live {va_live:.1%} │ LR {lr_now:.2e} │ Snaps {n_snap}{snap_marker}")

    # ── BN update for SWA model ────────────────────────────────────────
    print(f"\nUpdating BN stats for SWA model ({n_snap} snapshots) ...")
    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state)
    swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)

    _, va_swa = evaluate(swa_model, val_ldr, device)
    print(f"SWA val: {va_swa:.1%}")

    # SWA is the final model (EMA not tracked in Stage 3)
    ema.shadow.load_state_dict(swa_model.state_dict())
    ema.shadow.use_arcface(True)

    if va_swa > prev_best_val:
        print(f"Stage 3 val {va_swa:.1%} > Stage 2 best {prev_best_val:.1%} → saving.")
        save_ckpt(best_ckpt, CONFIG["s3_epochs"], "Stage 3",
                  swa_model, ema, va_swa, 0.0)

    return va_swa


# ══════════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt):
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")

    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow
    eval_model.eval()

    print(f"  ArcFace active : {eval_model._use_arcface}")
    print(f"  Checkpoint     : epoch {ckpt['epoch']} | {ckpt['stage']} "
          f"| val={ckpt['val_acc']:.1%}")
    print(f"  TTA views      : {CONFIG['tta_spatial']} spatial + "
          f"{CONFIG['tta_spectral']} spectral = "
          f"{CONFIG['tta_spatial']+CONFIG['tta_spectral']} total")

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x = x.to(device, non_blocking=True)
            if use_tta:
                logits = tta_predict(eval_model, x,
                                     CONFIG["tta_spatial"], CONFIG["tta_spectral"])
            else:
                with autocast(device_type=device.type):
                    logits = eval_model(x)
            preds.append(logits.argmax(1).cpu()); targets.append(y)
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

def main():
    device     = CONFIG["device"]
    ckpt_s1    = stage_ckpt_path(1)
    ckpt_s2    = stage_ckpt_path(2)
    ckpt_s3    = stage_ckpt_path(3)
    done_stage = latest_completed_stage()

    print(f"\n[INFO] Latest completed stage: {done_stage}")

    all_labels, train_idx, val_idx, test_idx = build_splits()
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx)//CONFIG['num_classes']}")

    model = SpectralQuadNet(
        num_classes=CONFIG["num_classes"], num_bands=CONFIG["num_bands"],
        dropout=CONFIG["s1_dropout"], wl_embed_dim=CONFIG["wl_embed_dim"], cfg=CONFIG,
    ).to(device)
    ema   = ModelEMA(model, CONFIG["ema_decay"])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : SpectralQuadNet v8 (v6 training)")
    print(f"Params : {n_par/1e6:.2f}M")
    print(f"Device : {device}")
    print(f"Key v6 changes:")
    print(f"  Stage 2 — FocalLoss γ={CONFIG['s2_focal_gamma']}, "
          f"SGDR restarts at ep {CONFIG['s2_warmup_ep']+CONFIG['s2_sgdr_T0']} & "
          f"ep {CONFIG['s2_warmup_ep']+CONFIG['s2_sgdr_T0']+CONFIG['s2_sgdr_T0']*CONFIG['s2_sgdr_Tmult']}, "
          f"m→{CONFIG['s2_arcface_m']}")
    print(f"  Stage 3 — {CONFIG['s3_epochs']} ep, cycle={CONFIG['s3_cycle_len']}, "
          f"peak_lr={CONFIG['s3_swa_lr']:.0e}, no EMA")
    print(f"  TTA    — {CONFIG['tta_spatial']}+{CONFIG['tta_spectral']}={CONFIG['tta_spatial']+CONFIG['tta_spectral']} views")

    criterion_s1 = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # ── Stage 1 ─────────────────────────────────────────────────────
    if done_stage < 1:
        print("\n[RUN] Stage 1")
        train_ldr1, val_ldr1, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s1_batch"], train_aug="heavy")
        run_stage1(model, ema, train_ldr1, val_ldr1, device, criterion_s1, ckpt_s1)
    else:
        print("\n[SKIP] Stage 1 → loading checkpoint")
        load_ckpt(ckpt_s1, model, ema, device)

    # ── Prepare Stage 2 ─────────────────────────────────────────────
    if done_stage < 2:
        print("\n[INFO] Initialising ArcFace from linear head")
        init_arcface_from_linear(model)
        init_arcface_from_linear(ema.shadow)

        print("\n[RUN] Stage 2")
        train_ldr2, val_ldr2, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"],
            balanced=True, all_labels=all_labels, train_aug="light")
        run_stage2(model, ema, train_ldr2, val_ldr2, device, ckpt_s2)
    else:
        print("\n[SKIP] Stage 2 → loading checkpoint")
        load_ckpt(ckpt_s2, model, ema, device)

    # ── Stage 3 (SWA) ───────────────────────────────────────────────
    if done_stage < 3:
        print("\n[RUN] Stage 3 (SWA)")
        s2_ckpt      = torch.load(ckpt_s2, map_location=device, weights_only=False)
        s2_best_val  = s2_ckpt.get("val_acc", 0.0)

        train_ldr3, val_ldr3, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"], train_aug="light")
        run_stage3_swa(model, ema, train_ldr3, val_ldr3, device,
                       criterion_s1, ckpt_s3, prev_best_val=s2_best_val)
    else:
        print("\n[SKIP] Stage 3 → loading checkpoint")
        load_ckpt(ckpt_s3, model, ema, device)

    # ── Final Evaluation ─────────────────────────────────────────────
    print("\n[INFO] Final Evaluation")
    final_ckpt = (ckpt_s3 if stage_exists(3)
                  else ckpt_s2 if stage_exists(2) else ckpt_s1)
    _, _, test_ldr_final = build_loaders(train_idx, val_idx, test_idx, 64)
    final_evaluation(model, ema, test_ldr_final, device, final_ckpt)


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