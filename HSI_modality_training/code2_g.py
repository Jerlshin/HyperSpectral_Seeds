# code final - Optuna Tuned Modular Version
from __future__ import annotations

import os
import copy, json as _json, math, random, warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

import optuna
from optuna.trial import TrialState

os.environ["NETWORKX_BACKEND"] = "nx-loopback"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", module="networkx")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Online softmax is disabled.*")

# ══════════════════════════════════════════════════════════════════════
#  BASE CONFIGURATION (Centralized defaults)
# ══════════════════════════════════════════════════════════════════════

WL_MIN: float = 385.0
WL_MAX: float = 1000.0

CONFIG: dict = {
    # ── Paths ─────────────────────────────────────────────────────────
    "patches_data":    "./dataset/patches.npy",
    "labels_path":     "./dataset/labels.npy",
    "wavelength_path": "./dataset/wavelengths.csv",
    "base_output_dir": "./optuna_runs/",

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
    "noise_std":            0.02,

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
    
    # Placeholders for dynamic trial directories
    "output_dir": "" 
}

_GPU_PATCHES:    Optional[torch.Tensor] = None
_GLOBAL_LABELS:  Optional[np.ndarray]  = None
_PHYSICAL_WL:    Optional[torch.Tensor] = None

# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING & REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════

def _load_data_mmap(patches_path: str, labels_path: str) -> None:
    global _GPU_PATCHES, _GLOBAL_LABELS
    if _GPU_PATCHES is not None: return
    print("[DATA] Memory-mapping dataset from disk (Zero-RAM footprint)...")
    _GPU_PATCHES = np.load(patches_path, mmap_mode='r') 
    _GLOBAL_LABELS = np.load(labels_path)
    print(f"[DATA] ✓ Indexed {_GPU_PATCHES.shape[0]} samples via mmap.")

def _load_wavelengths_to_gpu(csv_path: str, device: torch.device) -> None:
    global _PHYSICAL_WL
    if _PHYSICAL_WL is not None: return
    print("[DATA] Loading physical wavelengths from CSV...")
    df      = pd.read_csv(csv_path, sep=None, engine="python")
    raw_wl  = df.iloc[:, -1].values.astype(np.float32)
    wl_norm = (raw_wl - raw_wl.min()) / (raw_wl.max() - raw_wl.min())
    _PHYSICAL_WL = torch.from_numpy(wl_norm).to(device)
    print(f"[DATA] ✓ Loaded physical wavelengths: {_PHYSICAL_WL.size(0)} bands.")

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

# ══════════════════════════════════════════════════════════════════════
#  EMA & ARCHITECTURE BUILDING BLOCKS (Untouched functionality)
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.max_decay    = decay
        self._num_updates = 0
        self.shadow       = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
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
            if n in lp: sp.copy_(d * sp + (1.0 - d) * lp[n])
        lb = dict(model.named_buffers())
        for n, sb in self.shadow.named_buffers():
            if n in lb and sb.dtype.is_floating_point: sb.copy_(lb[n])
    def reinit_from(self, model: nn.Module) -> None:
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0
    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout): m.p = p
    def state_dict(self) -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)

class RiceSeedDataset(Dataset):
    _PROFILES = {
        "heavy": dict(band_drop=0.08, cutout=0.06, noise=0.04, warp=0.03, mult=0.05),
        "medium": dict(band_drop=0.05, cutout=0.04, noise=0.03, warp=0.02, mult=0.03),
        "light": dict(band_drop=0.0, cutout=0.0, noise=0.0, warp=0.0, mult=0.0),
        "none":  None,
    }
    _INTENSITY_SCALE = {"heavy": 1.0, "medium": 0.7, "light": 0.4}
    _WARP_RANGE      = {"heavy": 0.05, "medium": 0.03, "light": 0.0}

    def __init__(self, indices: np.ndarray, aug_strength: str = "none") -> None:
        self.patches         = _GPU_PATCHES
        self.labels          = _GLOBAL_LABELS
        self.indices         = indices
        self.aug_strength    = str(aug_strength)
        self.profile         = self._PROFILES.get(self.aug_strength)
        self.intensity_scale = self._INTENSITY_SCALE.get(self.aug_strength, 0.0)
        self.warp_range      = self._WARP_RANGE.get(self.aug_strength, 0.0)

    def __len__(self) -> int: return len(self.indices)

    def _band_dropout(self, x: torch.Tensor, prob: float) -> torch.Tensor:
        mask = (torch.rand(x.shape[0], device=x.device) > prob).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x: torch.Tensor) -> torch.Tensor:
        x       = x.clone()
        max_cut = max(1, CONFIG["max_cutout_bands"])
        cut     = torch.randint(1, max_cut + 1, (1,)).item()
        start   = torch.randint(0, max(1, x.shape[0] - cut), (1,)).item()
        x[start:start + cut] = 0.0
        return x

    def _spectral_noise(self, x: torch.Tensor) -> torch.Tensor:
        sigma = CONFIG["noise_std"] * self.intensity_scale
        mask  = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        return x + torch.randn_like(x) * sigma * mask

    def _spectral_warp(self, x: torch.Tensor) -> torch.Tensor:
        if self.warp_range <= 0: return x
        C, H, W = x.shape
        scale   = 1.0 + random.uniform(-self.warp_range, self.warp_range)
        new_C   = max(1, int(C * scale))
        if new_C == C: return x
        xp     = x.permute(1, 2, 0).reshape(-1, 1, C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s = (new_C - C) // 2
            warped = warped[:, :, s:s + C]
        else:
            pad_l = (C - new_C) // 2
            warped = F.pad(warped, (pad_l, C - new_C - pad_l))
        return warped.reshape(H, W, C).permute(2, 0, 1)

    def _multiplicative_noise(self, x: torch.Tensor) -> torch.Tensor:
        scale_std = 0.05 * self.intensity_scale
        mask      = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        factor    = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device) * scale_std
        return x * factor * mask

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5: x = torch.flip(x, [2])
        if torch.rand(1) < 0.5: x = torch.flip(x, [1])
        return torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ri    = self.indices[idx]
        patch_np = np.array(self.patches[ri])
        patch = torch.from_numpy(patch_np).to(CONFIG["device"], non_blocking=True)
        label = torch.tensor(int(self.labels[ri]), dtype=torch.long, device=CONFIG["device"])
        if self.profile is not None:
            p = self.profile
            if torch.rand(1) < p["band_drop"]: patch = self._band_dropout(patch, p["band_drop"])
            if torch.rand(1) < p["cutout"]: patch = self._band_cutout(patch)
            if torch.rand(1) < p["noise"]: patch = self._spectral_noise(patch)
            if torch.rand(1) < p["warp"]: patch = self._spectral_warp(patch)
            if torch.rand(1) < p["mult"]: patch = self._multiplicative_noise(patch)
            patch = self._spatial(patch)
        return patch, label

class ClassBalancedBatchSampler(Sampler):
    def __init__(self, train_labels: np.ndarray, n_cls: int = 16, n_spc: int = 8,
                 class_weights: Optional[Dict[int, float]] = None) -> None:
        self.n_cls, self.n_spc = n_cls, n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)
        if class_weights is not None:
            raw = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
            self.probs = raw / raw.sum()
        else: self.probs = None

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False, p=self.probs)
            batch  = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(rng.choice(pool, self.n_spc, replace=len(pool) < self.n_spc).tolist())
            yield batch
    def __len__(self) -> int: return self._n

class HardClassOversampledSampler(Sampler):
    def __init__(self, labels: np.ndarray, class_f1: Dict[int, float], num_samples: int,
                 oversample_power: float = 0.75, max_weight: float = 5.0,
                 hard_f1_thresh: float = 0.50, eps: float = 0.05) -> None:
        self.num_samples = num_samples
        num_classes = int(np.max(labels)) + 1
        raw_weights = {c: min((1.0 / (float(class_f1.get(c, 0.0)) + eps)) ** oversample_power, max_weight)
                       for c in range(num_classes)}
        mean_w = float(np.mean(list(raw_weights.values())))
        norm_weights = {c: w / mean_w for c, w in raw_weights.items()}
        self._weights = torch.from_numpy(np.array([norm_weights.get(int(lbl), 1.0) for lbl in labels], dtype=np.float32))

    def __iter__(self) -> Iterator[int]:
        return iter(torch.multinomial(self._weights, self.num_samples, replacement=True).tolist())
    def __len__(self) -> int: return self.num_samples

def build_cdws_weights(class_f1: Dict[int, float], num_classes: int, max_w: float = 3.0, eps: float = 0.05) -> Dict[int, float]:
    raw  = {c: min(1.0 / (class_f1.get(c, 0.0) + eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w / mean for c, w in raw.items()}

def mixed_aug(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixed_loss(crit: nn.Module, logits: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma, self.ls = gamma, label_smoothing
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C, logp = logits.shape[1], F.log_softmax(logits, dim=1)
        if self.ls > 0.0:
            soft = torch.full_like(logits, self.ls / (C - 1))
            soft.scatter_(1, targets.view(-1, 1), 1.0 - self.ls)
            ce = -(soft * logp).sum(1)
        else:
            ce = F.nll_loss(logp, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__(); self.temperature = temperature
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B         = features.shape[0]
        sim       = torch.mm(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        n_pos     = pos_mask.float().sum(1)
        if not (n_pos > 0).any(): return torch.zeros((), device=features.device, requires_grad=True)
        sim_m    = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim_m - torch.logsumexp(sim_m, dim=1, keepdim=True)
        loss     = -(pos_mask.float() * log_prob.masked_fill(self_mask, 0.0)).sum(1)
        valid    = n_pos > 0
        return (loss[valid] / n_pos[valid]).mean()

class ProtoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__(); self.temperature = temperature
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if len(classes) < 2: return (features * 0).sum()
        protos = F.normalize(torch.stack([features[labels == c].mean(0) for c in classes]), dim=1)
        sim   = torch.mm(features, protos.T) / self.temperature
        c2l   = {c.item(): i for i, c in enumerate(classes)}
        local = torch.tensor([c2l[y.item()] for y in labels], dtype=torch.long, device=features.device)
        return F.cross_entropy(sim, local)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs) -> None:
        defaults = dict(rho=rho, **kwargs)
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
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" in self.state[p]: p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def step(self, closure=None): raise NotImplementedError("Use first_step / second_step.")
    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        ns  = [p.grad.norm(p=2).to(dev) for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(ns), p=2).clamp(min=1e-6) if ns else torch.tensor(0.0)
    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups

class AdaptiveSubcenterArcFaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, K: int = 2, s: float = 32.0, m_base: float = 0.35, m_delta: float = 0.10) -> None:
        super().__init__()
        self.K, self.C, self.s, self.m_base, self.m_delta = K, num_classes, s, m_base, m_delta
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))
    def update_margins_from_f1(self, class_f1: Dict[int, float]) -> None:
        for c, f1 in class_f1.items(): self.margins[c] = self.m_base + self.m_delta * (1.0 - min(float(f1), 1.0))
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, global_m: Optional[float] = None) -> torch.Tensor:
        x_n, w_n = F.normalize(x, dim=1), F.normalize(self.weight, dim=1)
        cosine = F.linear(x_n, w_n).clamp(-1 + 1e-6, 1 - 1e-6).view(-1, self.C, self.K).max(dim=2).values
        if labels is None or not self.training: return cosine * self.s
        m_per = torch.full((x.shape[0],), global_m, device=x.device) if global_m is not None else self.margins[labels]
        cosm, sinm = torch.cos(m_per), torch.sin(m_per)
        th, mm = torch.cos(math.pi - m_per), torch.sin(math.pi - m_per) * m_per
        sine = torch.sqrt(torch.clamp(1 - cosine ** 2, min=1e-6))
        tgt_c, tgt_s = cosine.gather(1, labels.view(-1, 1)).squeeze(1), sine.gather(1, labels.view(-1, 1)).squeeze(1)
        phi = torch.where(tgt_c > th, tgt_c * cosm - tgt_s * sinm, tgt_c - mm)
        oh = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi.unsqueeze(1)) + ((1 - oh) * cosine)) * self.s
    def init_from_linear(self, linear_w: torch.Tensor) -> None:
        with torch.no_grad():
            wn = F.normalize(linear_w, dim=1)
            for k in range(self.K): self.weight[k::self.K].copy_(wn + torch.randn_like(wn) * 0.01 * k)

class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, mid, 1, bias=False), nn.GELU(), nn.Conv1d(mid, channels, 1, bias=False), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x * self.se(x)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1, self.norm1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False), nn.GroupNorm(1, out_ch)
        self.conv2, self.norm2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False), nn.GroupNorm(1, out_ch)
        self.se, self.skip = SEBlock1D(out_ch), nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return F.gelu(self.se(self.norm2(self.conv2(F.gelu(self.norm1(self.conv1(x)))))) + self.skip(x))

class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid = max(c // r, 8)
        self.ch = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.GELU(), nn.Conv2d(mid, c, 1, bias=False))
        self.sp = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(self.ch(x.mean([2, 3], keepdim=True)) + self.ch(x.amax([2, 3], keepdim=True)))
        return x * self.sp(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1))

class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid = max(out_ch // 2, in_ch)
        self.c1, self.n1 = nn.Conv2d(in_ch, mid, 1, bias=False), nn.GroupNorm(min(8, mid), mid)
        self.c2, self.n2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False), nn.GroupNorm(min(8, mid), mid)
        self.c3, self.n3 = nn.Conv2d(mid, out_ch, 1, bias=False), nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.GroupNorm(min(8, out_ch), out_ch)) if (stride != 1 or in_ch != out_ch) else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return F.gelu(self.n3(self.c3(F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) + self.skip(x))

class PhysicalWavelengthPE(nn.Module):
    def __init__(self, physical_wl: torch.Tensor, d_model: int) -> None:
        super().__init__()
        dev, half = physical_wl.device, d_model // 2
        freq = torch.exp(torch.arange(half, device=dev).float() * -(math.log(10000.0) / max(half - 1, 1)))
        pe = torch.zeros(physical_wl.size(0), d_model, device=dev)
        pe[:, :half], pe[:, half:] = torch.sin(physical_wl.unsqueeze(1) * freq.unsqueeze(0)), torch.cos(physical_wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("pe", pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.pe.transpose(0, 1).unsqueeze(0)

class LargeKernelBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1, self.act, self.pw2 = nn.Conv1d(dim, dim * 4, 1, bias=False), nn.GELU(), nn.Conv1d(dim * 4, dim, 1, bias=False)
        self.se = SEBlock1D(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.se(self.pw2(self.act(self.pw1(self.norm(self.dwconv(x))))))

class SpectralProfileBranch(nn.Module):
    def __init__(self, out_dim: int = 256, tower_ch: int = 96, wl_pe_module: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.wl_pe_module = wl_pe_module
        self.d1_conv, self.d2_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False), nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        with torch.no_grad():
            self.d1_conv.weight[0, 0] = torch.tensor([-3, -2, -1, 0, 1, 2, 3]).float() / 28.0
            self.d2_conv.weight[0, 0] = torch.tensor([5, 0, -3, -4, -3, 0, 5]).float() / 42.0
        self.stem = nn.Sequential(nn.Conv1d(3, tower_ch, kernel_size=7, padding=3, bias=False), nn.GroupNorm(1, tower_ch), nn.GELU())
        self.tower_s = nn.Sequential(LargeKernelBlock1D(tower_ch, 7), LargeKernelBlock1D(tower_ch, 7))
        self.tower_m = nn.Sequential(LargeKernelBlock1D(tower_ch, 15), LargeKernelBlock1D(tower_ch, 15))
        self.tower_l = nn.Sequential(LargeKernelBlock1D(tower_ch, 31), LargeKernelBlock1D(tower_ch, 31))
        self.fusion = nn.Sequential(nn.Conv1d(tower_ch * 3, tower_ch, 1, bias=False), nn.GroupNorm(1, tower_ch), nn.GELU(), LargeKernelBlock1D(tower_ch, 7))
        self.attn_pool = nn.Sequential(nn.Conv1d(tower_ch, tower_ch // 4, 1), nn.GELU(), nn.Conv1d(tower_ch // 4, 1, 1))
        self.proj = nn.Sequential(nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15))
    def forward(self, ms: torch.Tensor) -> torch.Tensor:
        s = ms.unsqueeze(1)
        s_smooth = F.avg_pool1d(s, kernel_size=5, stride=1, padding=2)
        d1, d2 = self.d1_conv(s_smooth), self.d2_conv(s_smooth)
        x = self.stem(torch.cat([s, d1, d2], dim=1))
        if self.wl_pe_module is not None: x = self.wl_pe_module(x)
        x_fused = self.fusion(torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1))
        w = torch.softmax(self.attn_pool(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))

class SpectralStatsBranch(nn.Module):
    def __init__(self, num_bands: int, out_dim: int = 256, tower_ch: int = 96, wl_pe_module: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.in_channels, self.wl_pe_module = 9, wl_pe_module
        self.stat_attn = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(self.in_channels, 16, 1, bias=False), nn.GELU(), nn.Conv1d(16, self.in_channels, 1, bias=False), nn.Sigmoid())
        self.input_proj = nn.Sequential(nn.Conv1d(self.in_channels, tower_ch, 1, bias=False), nn.GroupNorm(1, tower_ch), nn.GELU())
        def _make_tower(kernel: int) -> nn.Sequential: return nn.Sequential(ResBlock1D(tower_ch, tower_ch, kernel), ResBlock1D(tower_ch, tower_ch, kernel))
        self.tower_s, self.tower_m, self.tower_l = _make_tower(3), _make_tower(7), _make_tower(15)
        self.fusion = nn.Sequential(ResBlock1D(tower_ch * 3, tower_ch, 5), ResBlock1D(tower_ch, tower_ch, 5))
        self.pool_attn = nn.Sequential(nn.Conv1d(tower_ch, tower_ch // 4, 1, bias=False), nn.GELU(), nn.Conv1d(tower_ch // 4, 1, 1, bias=False))
        self.proj = nn.Sequential(nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15))
    def forward(self, ms, std, mx, skew, kurt, p10, p25, p75, p90):
        stats = torch.stack([ms, std, mx, skew, kurt, p10, p25, p75, p90], dim=1)
        x = self.input_proj(stats * self.stat_attn(stats))
        if self.wl_pe_module is not None: x = self.wl_pe_module(x)
        x_fused = self.fusion(torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1))
        w = torch.softmax(self.pool_attn(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.band_reduce = nn.Sequential(nn.Conv2d(num_bands, num_bands, 1, groups=num_bands, bias=False), nn.Conv2d(num_bands, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.GELU())
        self.stages = nn.Sequential(ResBlock2D(64, 128, 2), CBAM(128), ResBlock2D(128, 192, 2), CBAM(192), ResBlock2D(192, 256, 2), CBAM(256), ResBlock2D(256, out_dim, 2))
        self.proj = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.BatchNorm1d(out_dim), nn.GELU())
    @staticmethod
    def _pn(x: torch.Tensor) -> torch.Tensor: return x.sign() * x.abs().clamp(1e-8).sqrt()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.band_reduce(x))
        return self.proj(F.normalize(torch.cat([self._pn(h.mean([2, 3])), self._pn(h.amax([2, 3]))], 1), dim=1, eps=1e-4))

class MultiScaleSpectralTokenizer(nn.Module):
    def __init__(self, in_channels: int, d_model: int, stride: int = 8):
        super().__init__()
        out_c, rem = d_model // 3, d_model - ((d_model // 3) * 2)
        self.proj_small  = nn.Conv1d(in_channels, out_c, kernel_size=8,  stride=stride, padding=4)
        self.proj_medium = nn.Conv1d(in_channels, out_c, kernel_size=16, stride=stride, padding=8)
        self.proj_large  = nn.Conv1d(in_channels, rem,   kernel_size=32, stride=stride, padding=16)
        self.norm, self.act = nn.GroupNorm(1, d_model), nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_s, t_m, t_l = self.proj_small(x), self.proj_medium(x), self.proj_large(x)
        min_len = min(t_s.size(2), t_m.size(2), t_l.size(2))
        return self.act(self.norm(torch.cat([t_s[..., :min_len], t_m[..., :min_len], t_l[..., :min_len]], dim=1)))

class _PreLNBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, drop: float) -> None:
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2, self.drop = nn.LayerNorm(d), nn.Dropout(drop)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop), nn.Linear(d_ff, d), nn.Dropout(drop))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.ln1(x)
        h, _ = self.attn(lx, lx, lx, need_weights=False)
        x = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))

class SpecFormerBranch(nn.Module):
    def __init__(self, physical_wl: torch.Tensor, num_bands: int = 256, patch_size: int = 16, stride: int = 8, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, out_dim: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.tokenizer = MultiScaleSpectralTokenizer(in_channels=2, d_model=d_model, stride=stride)
        self.spec_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spec_cls, std=0.02)
        n_tokens = (num_bands // stride) + 2 
        self.spec_pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.spectral_blocks = nn.ModuleList([_PreLNBlock(d_model, n_heads, d_model * 2, dropout) for _ in range(n_layers // 2)])
        self.spatial_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spatial_cls, std=0.02)
        self.spatial_blocks = nn.ModuleList([_PreLNBlock(d_model, n_heads, d_model * 2, dropout) for _ in range(n_layers // 2)])
        self.norm, self.proj = nn.LayerNorm(d_model), nn.Sequential(nn.Linear(d_model, out_dim), nn.BatchNorm1d(out_dim), nn.GELU())
    def forward(self, grid_ms: torch.Tensor) -> torch.Tensor:
        B, N, C = grid_ms.shape
        deriv = F.pad(torch.diff(grid_ms, dim=2), (0, 1), mode='replicate') 
        x_combo = torch.stack([grid_ms, deriv], dim=2).view(B * N, 2, C)
        tokens = self.tokenizer(x_combo).transpose(1, 2)
        tokens = torch.cat([self.spec_cls.expand(B * N, -1, -1), tokens], dim=1)
        seq_len = tokens.size(1)
        if seq_len <= self.spec_pos_embed.size(1): tokens = tokens + self.spec_pos_embed[:, :seq_len, :]
        for blk in self.spectral_blocks: tokens = blk(tokens)
        spatial_tokens = torch.cat([self.spatial_cls.expand(B, -1, -1), tokens[:, 0, :].view(B, N, -1)], dim=1)
        for blk in self.spatial_blocks: spatial_tokens = blk(spatial_tokens)
        return self.proj(self.norm(spatial_tokens[:, 0, :]))

class CrossModalInteraction(nn.Module):
    def __init__(self, num_modalities: int = 4, d: int = 256, latent_tokens: int = 4, heads: int = 8, depth: int = 2, drop: float = 0.1):
        super().__init__()
        self.num_modalities, self.d = num_modalities, d
        self.branch_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_modalities)])
        self.latents = nn.Parameter(torch.randn(latent_tokens, d) * 0.02)
        self.blocks = nn.ModuleList([nn.ModuleDict({"cross_attn": nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True), "self_attn": nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True), "ff": nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(drop), nn.Linear(d * 4, d))}) for _ in range(depth)])
        self.modality_gate = nn.Sequential(nn.Linear(d, d // 4), nn.GELU(), nn.Linear(d // 4, num_modalities), nn.Softmax(dim=-1))
        self.output_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop))
    def forward(self, branches: List[torch.Tensor]):
        B = branches[0].shape[0]
        tokens = torch.stack([norm(b) for norm, b in zip(self.branch_norms, branches)], dim=1)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            attn_out, _ = blk["cross_attn"](latents, tokens, tokens); latents = latents + attn_out
            sa_out, _ = blk["self_attn"](latents, latents, latents); latents = latents + sa_out
            latents = latents + blk["ff"](latents)
        fused = latents.mean(dim=1)
        fused = fused + (tokens * self.modality_gate(fused).unsqueeze(-1)).sum(dim=1)
        return self.output_proj(fused)

class EmbedNet(nn.Module):
    def __init__(self, dim=256, hidden=512, drop=0.1):
        super().__init__()
        self.norm1, self.drop, self.norm2 = nn.LayerNorm(dim), nn.Dropout(drop), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim))
    def forward(self, x): return self.norm2(x + self.drop(self.mlp(self.norm1(x))))

class AuxiliaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))
        nn.init.trunc_normal_(self.net[0].weight, std=0.02); nn.init.zeros_(self.net[0].bias)
        nn.init.trunc_normal_(self.net[2].weight, std=0.02); nn.init.zeros_(self.net[2].bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

@torch.no_grad()
def compute_branch_influence(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 5) -> Dict[str, float]:
    model.eval()
    influences, total = torch.zeros(4, device=device), 0
    for i, (x, _) in enumerate(loader):
        if i >= max_batches: break
        x = x.to(device, non_blocking=True)
        p_full = torch.softmax(model(x), dim=1)
        for b in range(4):
            mask = torch.ones(4, device=device); mask[b] = 0.0
            p_ab = torch.softmax(model(x, branch_mask=mask), dim=1).clamp(min=1e-10)
            influences[b] += F.kl_div(p_ab.log(), p_full, reduction="batchmean")
        total += 1
    if total == 0: return {"A": 0, "B": 0, "C": 0, "D": 0}
    influences = (influences / total) / (influences / total).sum().clamp(min=1e-8) * 100.0
    return {k: float(influences[i]) for i, k in enumerate("ABCD")}

def extract_grid_spectra(x: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    B, C, H, W = x.shape
    mask = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()
    grid_mean = F.adaptive_avg_pool2d(x * mask, (grid_size, grid_size)) / F.adaptive_avg_pool2d(mask, (grid_size, grid_size)).clamp(min=1e-5)
    return grid_mean.view(B, C, -1).transpose(1, 2)

def masked_spectral_stats(x: torch.Tensor):
    x32 = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)
    mask = (flat.abs().sum(1, keepdim=True) > 1e-5).float()
    cnt = mask.sum(2).clamp(min=1.0)
    mean = (flat * mask).sum(2) / cnt
    centered = (flat - mean.unsqueeze(2)) * mask
    std = torch.sqrt((centered ** 2).sum(2) / cnt + 1e-5)
    mx = flat.masked_fill(mask.expand_as(flat) == 0, -1e4).max(2).values.masked_fill_(lambda a: a < -9999.0, 0.0)
    skew = torch.clamp(((centered**3).sum(2)/cnt) / (std**3 + 1e-4), -10.0, 10.0)
    kurt = torch.clamp(((centered**4).sum(2)/cnt) / (std**4 + 1e-4), 0.0, 20.0)
    flat_masked = flat.masked_fill(mask.expand_as(flat) == 0, float("inf"))
    sorted_vals, _ = torch.sort(flat_masked, dim=2)
    def gather_percentile(vals, p_frac):
        idx = (cnt * p_frac).long().clamp(max=H * W - 1).unsqueeze(2).expand(-1, C, -1)
        return torch.gather(vals, 2, idx).squeeze(2)
    p10, p25 = gather_percentile(sorted_vals, 0.10), gather_percentile(sorted_vals, 0.25)
    p75, p90 = gather_percentile(sorted_vals, 0.75), gather_percentile(sorted_vals, 0.90)
    return (torch.nan_to_num(mean, 0), torch.nan_to_num(std, 0), torch.nan_to_num(mx, 0), torch.nan_to_num(skew, 0), torch.nan_to_num(kurt, 0), torch.nan_to_num(p10, 0), torch.nan_to_num(p25, 0), torch.nan_to_num(p75, 0), torch.nan_to_num(p90, 0))    

class SpectralQuadNet(nn.Module):
    def __init__(self, num_classes: int = 90, num_bands: int = 256, dropout: float = 0.30, wl_embed_dim: int = 16, cfg: Optional[dict] = None) -> None:
        super().__init__()
        global _PHYSICAL_WL
        cfg = cfg or CONFIG
        tower_ch = 96
        self.branch_drop_prob = cfg.get("branch_drop_prob", 0.0)
        self.wl_pe_cnn = PhysicalWavelengthPE(_PHYSICAL_WL, tower_ch)
        self.branch_a = SpectralProfileBranch(out_dim=256, tower_ch=tower_ch, wl_pe_module=self.wl_pe_cnn)
        self.branch_b = SpectralStatsBranch(num_bands=num_bands, out_dim=256, tower_ch=96, wl_pe_module=self.wl_pe_cnn)
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(
            physical_wl=_PHYSICAL_WL, num_bands=num_bands, patch_size=cfg["specf_patch"], stride=cfg["specf_patch"] // 2,
            d_model=cfg["specf_dim"], n_heads=cfg["specf_heads"], n_layers=cfg["specf_layers"], out_dim=256, dropout=0.10
        )
        self.cross_interaction = CrossModalInteraction(num_modalities=4, d=256, drop=cfg["fusion_drop"])
        aux_hidden = cfg.get("aux_head_hidden", 128)
        self.aux_head_a = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_b = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_c = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_d = AuxiliaryHead(256, aux_hidden, num_classes)
        self.embed_net = EmbedNet(256, 512, dropout)
        self.linear_head  = nn.Sequential(nn.GELU(), nn.Dropout(dropout * 0.4), nn.Linear(256, num_classes))
        self.arcface_head = AdaptiveSubcenterArcFaceHead(256, num_classes, K=cfg.get("subcenter_K", 2), s=cfg["s2_arcface_s"], m_base=cfg["s2_arcface_m"], m_delta=cfg.get("s2_arcface_m_delta", 0.10))
        self._use_arcface = False
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.p = p
    def use_arcface(self, flag: bool) -> None: self._use_arcface = flag
    def freeze_head(self, which: str) -> None:
        for p in (self.linear_head if which == "linear" else self.arcface_head).parameters(): p.requires_grad_(False)
    def unfreeze_head(self, which: str) -> None:
        for p in (self.linear_head if which == "linear" else self.arcface_head).parameters(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, return_embed: bool = False, arc_m: Optional[float] = None, branch_mask: Optional[torch.Tensor] = None):
        ms, std, mx, skew, kurt, p10, p25, p75, p90 = masked_spectral_stats(x)
        grid_ms = extract_grid_spectra(x, grid_size=4)
        B, N, C = grid_ms.shape
        ba_raw = self.branch_a(grid_ms.reshape(B * N, C)).view(B, N, -1).mean(dim=1)
        bb_raw = self.branch_b(ms, std, mx, skew, kurt, p10, p25, p75, p90)
        bc_raw = self.branch_c(x)
        bd_raw = self.branch_d(grid_ms)

        if branch_mask is not None:
            ba, bb, bc, bd = ba_raw * branch_mask[0], bb_raw * branch_mask[1], bc_raw * branch_mask[2], bd_raw * branch_mask[3]
        elif self.training:
            keeps = torch.maximum((torch.rand(4, device=x.device) > torch.tensor([0.05, 0.05, 0.25, 0.10], device=x.device)).float(), F.one_hot(torch.randint(0, 4, (), device=x.device), num_classes=4).float())
            ba, bb, bc, bd = ba_raw * keeps[0], bb_raw * keeps[1], bc_raw * keeps[2], bd_raw * keeps[3]
        else:
            ba, bb, bc, bd = ba_raw, bb_raw, bc_raw, bd_raw

        emb = self.embed_net(self.cross_interaction([ba, bb, bc, bd]))
        logits = self.arcface_head(F.normalize(emb, dim=1), labels, global_m=arc_m) if self._use_arcface else self.linear_head(emb)

        if self.training:
            out = {"main": logits, "aux_a": self.aux_head_a(ba_raw), "aux_b": self.aux_head_b(bb_raw), "aux_c": self.aux_head_c(bc_raw), "aux_d": self.aux_head_d(bd_raw)}
            if return_embed: out["emb"] = F.normalize(emb, dim=1)
            return out
        return (logits, F.normalize(emb, dim=1)) if return_embed else logits

# ══════════════════════════════════════════════════════════════════════
#  HELPERS, LOADERS, ETC
# ══════════════════════════════════════════════════════════════════════

def build_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels,       random_state=42)
    va, te  = train_test_split(tmp,     test_size=0.5, stratify=labels[tmp],  random_state=42)
    return labels, tr, va, te

def build_loaders(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, batch_train: int, balanced: bool = False, all_labels: Optional[np.ndarray] = None, train_aug: str = "none", class_weights: Optional[Dict[int, float]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = RiceSeedDataset(train_idx, aug_strength=train_aug)
    if balanced and all_labels is not None:
        samp = ClassBalancedBatchSampler(all_labels[train_idx], CONFIG["bal_n_cls"], CONFIG["bal_n_spc"], class_weights=class_weights)
        tr_ldr = DataLoader(ds, batch_sampler=samp, num_workers=0)
    else: tr_ldr = DataLoader(ds, batch_size=batch_train, shuffle=True, drop_last=True, num_workers=0)
    return tr_ldr, DataLoader(RiceSeedDataset(val_idx), batch_size=256, shuffle=False, num_workers=0), DataLoader(RiceSeedDataset(test_idx), batch_size=256, shuffle=False, num_workers=0)

def build_phase3_loader(train_ds: Dataset, class_f1: Dict[int, float]) -> DataLoader:
    if not CONFIG["s1_p3_oversample"] or not class_f1:
        return DataLoader(train_ds, batch_size=CONFIG["s1_batch"], shuffle=True, drop_last=True, num_workers=0)
    train_labels = np.array([int(_GLOBAL_LABELS[train_ds.indices[i]]) for i in range(len(train_ds.indices))])
    sampler = HardClassOversampledSampler(labels=train_labels, class_f1=class_f1, num_samples=len(train_labels), oversample_power=CONFIG["s1_p3_oversample_power"], max_weight=CONFIG["s1_p3_oversample_max_w"], hard_f1_thresh=CONFIG["s1_p3_hard_f1_thresh"], eps=CONFIG["s1_p3_oversample_eps"])
    return DataLoader(train_ds, batch_size=CONFIG["s1_batch"], sampler=sampler, drop_last=True, num_workers=0)

def _wd_groups(named_params, lr: float) -> List[dict]:
    wd, no_wd = [], []
    for n, p in named_params:
        if not p.requires_grad: continue
        (no_wd if (p.ndim == 1 or n.endswith(".bias")) else wd).append(p)
    return [{"params": wd, "lr": lr, "weight_decay": CONFIG["weight_decay"]}, {"params": no_wd, "lr": lr, "weight_decay": 0.0}]

def sgdr_scheduler(optimizer: optim.Optimizer, warmup_ep: int = 5, T_0: int = 10, T_mult: int = 2, eta_min_frac: float = 1e-3) -> optim.lr_scheduler.LambdaLR:
    def _l(ep: int) -> float:
        if ep < warmup_ep: return max(ep / max(warmup_ep, 1), 1e-6)
        t, clen, elapsed = ep - warmup_ep, T_0, 0
        while t >= elapsed + clen: elapsed += clen; clen = max(int(clen * T_mult), 1)
        ratio = (t - elapsed) / max(clen, 1)
        return eta_min_frac + 0.5 * (1 - eta_min_frac) * (1 + math.cos(math.pi * ratio))
    return optim.lr_scheduler.LambdaLR(optimizer, _l)

def arcface_margin(ep: int, m0: float, m_target: float, warmup_ep: int) -> float:
    return m_target if ep >= warmup_ep else m0 + (m_target - m0) * 0.5 * (1 - math.cos(math.pi * ep / max(warmup_ep, 1)))

def _aux_loss_weight(current_ep: int, total_ep: int) -> float:
    return max(CONFIG["aux_loss_weight_final"], CONFIG["aux_loss_weight_init"] * (1.0 - (current_ep / max(total_ep, 1))))

def _compute_aux_loss(criterion: nn.Module, out: dict, ya: torch.Tensor, yb: torch.Tensor, lam: float, use_mixup: bool) -> torch.Tensor:
    total = torch.zeros((), device=ya.device)
    for k in ["aux_a", "aux_b", "aux_c", "aux_d"]:
        if k in out: total = total + (mixed_loss(criterion, out[k], ya, yb, lam) if use_mixup else criterion(out[k], ya))
    return total

def save_ckpt(path: str, epoch: int, stage: str, model: nn.Module, ema: ModelEMA, val_f1: float, val_acc: float, **metadata) -> None:
    bundle = {"epoch": epoch, "stage": stage, "model": model.state_dict(), "ema": ema.state_dict(), "val_f1": val_f1, "val_acc": val_acc, "use_arcface": model._use_arcface, **metadata}
    torch.save(bundle, path)

def load_ckpt(path: str, model: nn.Module, ema: ModelEMA, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    flag = ckpt.get("use_arcface", False)
    model.use_arcface(flag); ema.shadow.use_arcface(flag)
    return ckpt

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad(), autocast(device_type=device.type, enabled=False):
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            if not torch.isfinite(logits).all(): logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            preds.append(logits.argmax(1).cpu()); targets.append(y.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return f1_score(t, p, average="macro", zero_division=0), accuracy_score(t, p)

def evaluate_per_class(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[int, float]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu()); targets.append(y.cpu())
    f1_arr = f1_score(torch.cat(targets).numpy(), torch.cat(preds).numpy(), average=None, zero_division=0, labels=list(range(num_classes)))
    return {i: float(v) for i, v in enumerate(f1_arr)}

def compute_class_difficulty(ema_shadow: nn.Module, val_ldr: DataLoader, device: torch.device, label: str = "Stage") -> Tuple[Dict[int, float], Dict[int, float]]:
    class_f1 = evaluate_per_class(ema_shadow, val_ldr, device, CONFIG["num_classes"])
    cdws_wts = build_cdws_weights(class_f1, CONFIG["num_classes"], CONFIG["cdws_max_weight"], CONFIG["cdws_eps"])
    return class_f1, cdws_wts


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOPS (Optuna Integrated)
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, scaler: Optional[GradScaler], ema: ModelEMA, device: torch.device, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None, use_mixup: bool = True, mixup_alpha: float = 0.4, supcon: Optional[nn.Module] = None, supcon_weight: float = 0.0, proto: Optional[nn.Module] = None, proto_weight: float = 0.0, accum_steps: int = 1, arc_m: Optional[float] = None, current_ep: int = 0, total_ep: int = 100,
) -> Tuple[float, float]:
    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_amp = (supcon is None) and (scaler is not None)
    aux_w = _aux_loss_weight(current_ep, total_ep)

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_in, ya, yb, lam = mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)
        with autocast(device_type=device.type, enabled=use_amp):
            if supcon is not None:
                out = model(x_in, ya, return_embed=True, arc_m=arc_m)
                logits, emb = (out["main"], out["emb"]) if isinstance(out, dict) else (out[0], out[1])
                loss = (1 - supcon_weight - proto_weight) * criterion(logits, ya) + supcon_weight * supcon(emb, ya) + (proto_weight * proto(emb, ya) if proto else 0.0) + aux_w * (_compute_aux_loss(criterion, out, ya, yb, lam, False) if isinstance(out, dict) else 0.0)
            else:
                out = model(x_in, labels=(ya if (model._use_arcface and not use_mixup) else None), arc_m=arc_m)
                logits = out["main"] if isinstance(out, dict) else out
                loss = mixed_loss(criterion, logits, ya, yb, lam) + (aux_w * _compute_aux_loss(criterion, out, ya, yb, lam, use_mixup) if isinstance(out, dict) else 0.0)
        
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True); continue
        
        if use_amp: scaler.scale(loss / accum_steps).backward()
        else: (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if use_amp: scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            if use_amp: scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema: ema.update(model)

        total_loss += loss.item()
        with torch.no_grad(): total_acc += (logits.argmax(1) == ya).float().mean().item()
    n = max(len(loader), 1)
    return total_loss / n, total_acc / n

def run_stage1(model: nn.Module, ema: ModelEMA, loaders_by_phase: Dict[int, DataLoader], val_ldr: DataLoader, device: torch.device, best_ckpt: str, trial: optuna.Trial) -> float:
    model.use_arcface(False); model.unfreeze_head("linear"); model.freeze_head("arcface")
    ep_total, p1_end, p2_end = CONFIG["s1_epochs"], int(CONFIG["s1_epochs"] * CONFIG["s1_phase1_frac"]), int(CONFIG["s1_epochs"] * (CONFIG["s1_phase1_frac"] + CONFIG["s1_phase2_frac"]))
    optimizer = optim.AdamW(_wd_groups(model.named_parameters(), CONFIG["s1_max_lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_total, eta_min=CONFIG["s1_min_lr"])
    scaler, best_f1, no_improve, ema_reinited = GradScaler(), 0.0, 0, [False, False]
    phase3_ldr, class_f1_phase2 = None, {}

    for ep in range(1, ep_total + 1):
        phase = 1 if ep <= p1_end else 2 if ep <= p2_end else 3
        if phase == 2 and not ema_reinited[0] and CONFIG["s1_ema_reinit_phases"]: ema.reinit_from(model); ema_reinited[0] = True
        if phase == 3 and not ema_reinited[1] and CONFIG["s1_ema_reinit_phases"]: ema.reinit_from(model); ema_reinited[1] = True
        if phase == 3 and phase3_ldr is None:
            class_f1_phase2, _ = compute_class_difficulty(ema.shadow, val_ldr, device)
            phase3_ldr = build_phase3_loader(loaders_by_phase[3].dataset, class_f1_phase2)

        cur_ldr = loaders_by_phase[1] if phase == 1 else loaders_by_phase[2] if phase == 2 else phase3_ldr
        ls_now = CONFIG["s1_label_smooth_hi"] * (1 - (ep - 1) / max(ep_total - 1, 1)) + CONFIG["s1_label_smooth_lo"] * ((ep - 1) / max(ep_total - 1, 1))
        crit = FocalLoss(gamma=CONFIG["s1_focal_gamma"], label_smoothing=ls_now) if phase == 3 else nn.CrossEntropyLoss(label_smoothing=ls_now)
        
        train_one_epoch(model, cur_ldr, optimizer, crit, scaler, ema, device, use_mixup=(phase != 3), mixup_alpha=CONFIG["s1_mixup"], accum_steps=CONFIG["s1_accum"], current_ep=ep, total_ep=ep_total)
        scheduler.step()

        f1_live, _ = evaluate(model, val_ldr, device)
        f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1 = max(f1_live, f1_ema)

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1, _cdws = compute_class_difficulty(ema.shadow, val_ldr, device)
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema, val_f1=best_ep_f1, val_acc=acc_ema, class_f1=_cf1, cdws_weights=_cdws, arcface_init_done=False)
        else: no_improve += 1

        # Optuna Pruning check
        trial.report(best_ep_f1, ep)
        if trial.should_prune(): raise optuna.TrialPruned()
        if no_improve >= CONFIG["s1_patience"]: break

    model.unfreeze_head("arcface")
    return best_f1

def run_stage2(model: nn.Module, ema: ModelEMA, train_ldr: DataLoader, val_ldr: DataLoader, device: torch.device, best_ckpt: str, class_f1: Dict[int, float], trial: optuna.Trial) -> float:
    model.set_dropout(CONFIG["s2_dropout"]); model.use_arcface(True); model.freeze_head("linear"); model.unfreeze_head("arcface")
    ema.reinit_from(model); ema.set_dropout(CONFIG["s2_dropout"]); ema.shadow.use_arcface(True)
    if class_f1: model.arcface_head.update_margins_from_f1(class_f1); ema.shadow.arcface_head.update_margins_from_f1(class_f1)

    focal, supcon, proto = FocalLoss(gamma=CONFIG["s2_focal_gamma"]), SupConLoss(temperature=CONFIG["supcon_temp"]), ProtoNCELoss(temperature=CONFIG["proto_temp"])
    hp, bp = [], []
    for n, p in model.named_parameters():
        if p.requires_grad: (hp if n.startswith("arcface_head") else bp).append((n, p))
    optimizer = optim.AdamW(_wd_groups(hp, CONFIG["s2_head_lr"]) + _wd_groups(bp, CONFIG["s2_back_lr"]))
    scheduler = sgdr_scheduler(optimizer, warmup_ep=CONFIG["s2_warmup_ep"], T_0=CONFIG["s2_sgdr_T0"], T_mult=CONFIG["s2_sgdr_Tmult"], eta_min_frac=CONFIG["s2_min_lr"] / CONFIG["s2_head_lr"])
    
    best_f1, no_improve = 0.0, 0
    for ep in range(1, CONFIG["s2_epochs"] + 1):
        warmup_done = (ep - 1) >= CONFIG["s2_margin_warmup_ep"]
        m_now = CONFIG["s2_arcface_m"] if warmup_done else arcface_margin(ep - 1, CONFIG["s2_arcface_m0"], CONFIG["s2_arcface_m"], CONFIG["s2_margin_warmup_ep"])
        train_one_epoch(model, train_ldr, optimizer, focal, None, ema, device, use_mixup=False, supcon=supcon, supcon_weight=CONFIG["supcon_weight"] * min(1.0, ep / 10.0), proto=proto, proto_weight=CONFIG["proto_weight"] * min(1.0, ep / 10.0), arc_m=None if warmup_done else m_now, current_ep=ep, total_ep=CONFIG["s2_epochs"])
        scheduler.step()

        f1_live, _ = evaluate(model, val_ldr, device)
        f1_ema, acc_ema = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1 = max(f1_live, f1_ema)

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1_s2, _cdws_s2 = compute_class_difficulty(ema.shadow, val_ldr, device)
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema, val_f1=best_ep_f1, val_acc=acc_ema, class_f1=_cf1_s2, cdws_weights=_cdws_s2)
        else: no_improve += 1

        trial.report(best_ep_f1, ep)
        if trial.should_prune(): raise optuna.TrialPruned()
        if no_improve >= CONFIG["s2_patience"]: break
    
    model.unfreeze_head("linear")
    return best_f1

def run_stage3_swa(model: nn.Module, ema: ModelEMA, train_ldr: DataLoader, val_ldr: DataLoader, device: torch.device, best_ckpt: str, prev_best_f1: float, trial: optuna.Trial) -> float:
    model.set_dropout(CONFIG["s2_dropout"]); model.branch_drop_prob = 0.0; ema.shadow.branch_drop_prob = 0.0; model.use_arcface(True); ema.shadow.use_arcface(True)
    sam = SAM(list(_wd_groups(model.named_parameters(), CONFIG["s3_swa_lr"])), optim.AdamW, rho=CONFIG["s3_sam_rho"], lr=CONFIG["s3_swa_lr"], weight_decay=CONFIG["weight_decay"])
    focal_s3, supcon_s3, proto_s3 = FocalLoss(gamma=1.0), SupConLoss(temperature=0.10), ProtoNCELoss(temperature=0.10)
    
    swa_state, n_snap, best_live_f1 = None, 0, 0.0
    for ep in range(1, CONFIG["s3_epochs"] + 1):
        cycle_ep = (ep - 1) % CONFIG["s3_cycle_len"]
        lr_now = CONFIG["s3_swa_lr"] * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * cycle_ep / CONFIG["s3_cycle_len"])))
        for pg in sam.param_groups: pg["lr"] = lr_now

        model.train()
        for x, y in train_ldr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sam.zero_grad()
            out = model(x, labels=y, arc_m=0.25 + 0.05 * math.cos(math.pi * ep / CONFIG["s3_epochs"]), return_embed=True)
            loss = focal_s3(out["main"], y) + 0.02 * supcon_s3(out["emb"], y) + CONFIG["s3_aux_loss_weight"] * _compute_aux_loss(focal_s3, out, y, y, 1.0, False)
            if not torch.isfinite(loss): continue
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"]); sam.first_step(zero_grad=True)
            out2 = model(x, labels=y, arc_m=0.25 + 0.05 * math.cos(math.pi * ep / CONFIG["s3_epochs"]))
            loss2 = focal_s3(out2["main"] if isinstance(out2, dict) else out2, y)
            if not torch.isfinite(loss2): continue
            loss2.backward(); nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"]); sam.second_step(zero_grad=True)

        f1_live, _ = evaluate(model, val_ldr, device)
        best_live_f1 = max(best_live_f1, f1_live)

        if ep % CONFIG["s3_cycle_len"] == 0 and (not CONFIG["s3_greedy"] or f1_live >= best_live_f1 * 0.98):
            n_snap += 1
            sd = model.state_dict()
            if swa_state is None: swa_state = copy.deepcopy(sd)
            else:
                beta = 1.0 / float(n_snap)
                for k in swa_state:
                    if swa_state[k].is_floating_point(): swa_state[k].mul_(1.0 - beta).add_(sd[k], alpha=beta)
                    else: swa_state[k].copy_(sd[k])
        
        trial.report(best_live_f1, ep)
        if trial.should_prune(): raise optuna.TrialPruned()

    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state if swa_state else model.state_dict()); swa_model.use_arcface(True)
    
    swa_model.train()
    for m in swa_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.reset_running_stats(); m.momentum = None
    with torch.no_grad():
        for x, _ in train_ldr: swa_model(x.to(device, non_blocking=True))
    swa_model.eval()

    f1_swa, acc_swa = evaluate(swa_model, val_ldr, device)
    ema.shadow.load_state_dict(swa_model.state_dict())
    save_ckpt(best_ckpt, CONFIG["s3_epochs"], "Stage 3", swa_model, ema, val_f1=f1_swa, val_acc=acc_swa)
    return f1_swa

# ══════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVES
# ══════════════════════════════════════════════════════════════════════

def update_global_config(trial_cfg: dict):
    """Centralized config updater to dynamically inject parameters."""
    global CONFIG
    for k, v in trial_cfg.items(): CONFIG[k] = v

def objective_s1(trial: optuna.Trial, base_cfg: dict, all_labels, train_idx, val_idx, test_idx) -> float:
    cfg = copy.deepcopy(base_cfg)
    cfg["s1_max_lr"] = trial.suggest_float("s1_max_lr", 1e-5, 2e-3, log=True)
    cfg["s1_dropout"] = trial.suggest_float("s1_dropout", 0.0, 0.4)
    cfg["s1_mixup"] = trial.suggest_float("s1_mixup", 0.05, 0.3)
    cfg["s1_focal_gamma"] = trial.suggest_float("s1_focal_gamma", 1.0, 2.5)
    cfg["branch_drop_prob"] = trial.suggest_float("branch_drop_prob", 0.0, 0.4)
    cfg["specf_dim"] = trial.suggest_categorical("specf_dim", [128, 256])
    cfg["specf_layers"] = trial.suggest_categorical("specf_layers", [2, 4, 6])
    
    # Isolate output directory for this trial
    cfg["output_dir"] = os.path.join(cfg["base_output_dir"], f"stage1/trial_{trial.number}/")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    update_global_config(cfg)
    
    model = SpectralQuadNet(cfg["num_classes"], cfg["num_bands"], cfg["s1_dropout"], cfg["wl_embed_dim"], cfg).to(cfg["device"])
    ema = ModelEMA(model, decay=cfg["ema_decay"])
    
    ds_h, ds_m, ds_l = RiceSeedDataset(train_idx, "heavy"), RiceSeedDataset(train_idx, "medium"), RiceSeedDataset(train_idx, "light")
    loaders = {
        1: DataLoader(ds_h, batch_size=cfg["s1_batch"], shuffle=True, drop_last=True),
        2: DataLoader(ds_m, batch_size=cfg["s1_batch"], shuffle=True, drop_last=True),
        3: DataLoader(ds_l, batch_size=cfg["s1_batch"], shuffle=True, drop_last=True)
    }
    _, val_ldr, _ = build_loaders(train_idx, val_idx, test_idx, cfg["s1_batch"])
    
    ckpt_path = os.path.join(cfg["output_dir"], "best_stage1.pth")
    best_f1 = run_stage1(model, ema, loaders, val_ldr, cfg["device"], ckpt_path, trial)
    
    trial.set_user_attr("ckpt_path", ckpt_path)
    trial.set_user_attr("cfg", cfg)
    return best_f1


def objective_s2(trial: optuna.Trial, base_cfg: dict, s1_ckpt: str, all_labels, train_idx, val_idx, test_idx) -> float:
    cfg = copy.deepcopy(base_cfg)
    cfg["s2_head_lr"] = trial.suggest_float("s2_head_lr", 5e-5, 1e-3, log=True)
    cfg["s2_back_lr"] = trial.suggest_float("s2_back_lr", 1e-6, 1e-4, log=True)
    cfg["s2_dropout"] = trial.suggest_float("s2_dropout", 0.0, 0.4)
    cfg["s2_arcface_s"] = trial.suggest_float("s2_arcface_s", 16.0, 64.0)
    cfg["s2_arcface_m"] = trial.suggest_float("s2_arcface_m", 0.1, 0.6)
    cfg["supcon_weight"] = trial.suggest_float("supcon_weight", 0.05, 0.4)
    cfg["proto_weight"] = trial.suggest_float("proto_weight", 0.05, 0.3)
    
    cfg["output_dir"] = os.path.join(cfg["base_output_dir"], f"stage2/trial_{trial.number}/")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    update_global_config(cfg)
    
    model = SpectralQuadNet(cfg["num_classes"], cfg["num_bands"], cfg["s1_dropout"], cfg["wl_embed_dim"], cfg).to(cfg["device"])
    ema = ModelEMA(model, decay=cfg["ema_decay"])
    
    # Load optimal weights from stage 1
    loaded_ckpt = load_ckpt(s1_ckpt, model, ema, cfg["device"])
    class_f1_s1 = loaded_ckpt.get("class_f1", {})
    cdws_wts_s1 = loaded_ckpt.get("cdws_weights", {})
    
    # Bootstrap ArcFace if needed
    if not model._use_arcface:
        lw = model.linear_head[-1].weight.data.clone()
        model.arcface_head.init_from_linear(lw); ema.shadow.arcface_head.init_from_linear(lw)

    tr2, va2, _ = build_loaders(train_idx, val_idx, test_idx, cfg["s2_batch"], balanced=True, all_labels=all_labels, train_aug="light", class_weights=cdws_wts_s1)
    
    ckpt_path = os.path.join(cfg["output_dir"], "best_stage2.pth")
    best_f1 = run_stage2(model, ema, tr2, va2, cfg["device"], ckpt_path, class_f1_s1, trial)
    
    trial.set_user_attr("ckpt_path", ckpt_path)
    trial.set_user_attr("cfg", cfg)
    return best_f1


def objective_s3(trial: optuna.Trial, base_cfg: dict, s2_ckpt: str, all_labels, train_idx, val_idx, test_idx) -> float:
    cfg = copy.deepcopy(base_cfg)
    cfg["s3_swa_lr"] = trial.suggest_float("s3_swa_lr", 1e-6, 1e-4, log=True)
    cfg["s3_sam_rho"] = trial.suggest_float("s3_sam_rho", 0.01, 0.15)
    cfg["s3_aux_loss_weight"] = trial.suggest_float("s3_aux_loss_weight", 0.0, 0.2)
    
    cfg["output_dir"] = os.path.join(cfg["base_output_dir"], f"stage3/trial_{trial.number}/")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    update_global_config(cfg)
    
    model = SpectralQuadNet(cfg["num_classes"], cfg["num_bands"], cfg["s1_dropout"], cfg["wl_embed_dim"], cfg).to(cfg["device"])
    ema = ModelEMA(model, decay=cfg["ema_decay"])
    
    loaded_ckpt = load_ckpt(s2_ckpt, model, ema, cfg["device"])
    cdws_wts_s2 = loaded_ckpt.get("cdws_weights", {})
    prev_f1 = loaded_ckpt.get("val_f1", 0.0)

    tr3, va3, _ = build_loaders(train_idx, val_idx, test_idx, cfg["s2_batch"], balanced=True, all_labels=all_labels, train_aug="light", class_weights=cdws_wts_s2)
    
    ckpt_path = os.path.join(cfg["output_dir"], "best_stage3.pth")
    best_f1 = run_stage3_swa(model, ema, tr3, va3, cfg["device"], ckpt_path, prev_f1, trial)
    
    trial.set_user_attr("ckpt_path", ckpt_path)
    trial.set_user_attr("cfg", cfg)
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  MASTER PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════

def run_tuning_pipeline():
    print("=" * 60)
    print("  Starting End-to-End Modular Optuna Tuning Pipeline")
    print("=" * 60)

    device = CONFIG["device"]
    _load_data_mmap(CONFIG["patches_data"], CONFIG["labels_path"])
    _load_wavelengths_to_gpu(CONFIG["wavelength_path"], device)
    all_labels, train_idx, val_idx, test_idx = build_splits()
    
    storage_url = "sqlite:///optuna_hs_tuning.db"
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=5)

    # ──────────────────────────────────────────────────────────────────
    # TUNE STAGE 1
    # ──────────────────────────────────────────────────────────────────
    print("\n[INFO] Initializing Optuna Study for STAGE 1")
    study_s1 = optuna.create_study(study_name="HS_Stage1", direction="maximize", storage=storage_url, load_if_exists=True, pruner=pruner)
    
    # Check if we already have a successful trial
    if len([t for t in study_s1.trials if t.state == TrialState.COMPLETE]) < 10: # Assuming 10 trials desired
        study_s1.optimize(lambda t: objective_s1(t, CONFIG, all_labels, train_idx, val_idx, test_idx), n_trials=10, gc_after_trial=True)
    
    best_s1_cfg = study_s1.best_trial.user_attrs["cfg"]
    best_s1_ckpt = study_s1.best_trial.user_attrs["ckpt_path"]
    print(f"\n[SUCCESS] Stage 1 Tuning Complete! Best F1: {study_s1.best_value:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # TUNE STAGE 2
    # ──────────────────────────────────────────────────────────────────
    print("\n[INFO] Initializing Optuna Study for STAGE 2 (Loaded best S1 Architecture)")
    study_s2 = optuna.create_study(study_name="HS_Stage2", direction="maximize", storage=storage_url, load_if_exists=True, pruner=pruner)
    
    if len([t for t in study_s2.trials if t.state == TrialState.COMPLETE]) < 10:
        study_s2.optimize(lambda t: objective_s2(t, best_s1_cfg, best_s1_ckpt, all_labels, train_idx, val_idx, test_idx), n_trials=10, gc_after_trial=True)

    best_s2_cfg = study_s2.best_trial.user_attrs["cfg"]
    best_s2_ckpt = study_s2.best_trial.user_attrs["ckpt_path"]
    print(f"\n[SUCCESS] Stage 2 Tuning Complete! Best F1: {study_s2.best_value:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # TUNE STAGE 3
    # ──────────────────────────────────────────────────────────────────
    print("\n[INFO] Initializing Optuna Study for STAGE 3 (Loaded best S2 Weights)")
    if hasattr(torch, "_dynamo"): torch._dynamo.reset() # Prevent dynamo conflicts in S3
    
    study_s3 = optuna.create_study(study_name="HS_Stage3", direction="maximize", storage=storage_url, load_if_exists=True, pruner=pruner)
    
    if len([t for t in study_s3.trials if t.state == TrialState.COMPLETE]) < 10:
        study_s3.optimize(lambda t: objective_s3(t, best_s2_cfg, best_s2_ckpt, all_labels, train_idx, val_idx, test_idx), n_trials=5, gc_after_trial=True)
    
    print(f"\n[SUCCESS] Stage 3 Tuning Complete! Best F1: {study_s3.best_value:.4f}")
    
    print("\n" + "=" * 60)
    print("  ALL STAGES TUNED SUCCESSFULLY")
    print(f"  Final Best Model Weights: {study_s3.best_trial.user_attrs['ckpt_path']}")
    print("=" * 60)

if __name__ == "__main__":
    import traceback, sys, logging

    os.makedirs(CONFIG["base_output_dir"], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(CONFIG["base_output_dir"], "tuning_pipeline.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Redirect Optuna logs to our logger
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    
    try:
        run_tuning_pipeline()
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)