
from __future__ import annotations

import copy
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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

    # ── Stage 1 ──────────────────────────────────────────────────────
    # 3-phase aug curriculum:
    #   Phase 1 (0–40%): heavy aug + mixup → broad exploration
    #   Phase 2 (40–70%): medium aug + mixup → consolidation
    #   Phase 3 (70–100%): light aug + NO mixup + focal → discrimination
    "s1_epochs":              300,
    "s1_phase1_frac":         0.40,
    "s1_phase2_frac":         0.30,
    # Phase 3 = remaining 30% = 90 epochs (user's proven fade-out, improved)
    "s1_batch":               64,
    "s1_max_lr":              8e-4,
    "s1_dropout":             0.30,
    "s1_mixup":               0.4,
    "s1_patience":            60,
    "s1_accum":                2,
    "s1_focal_gamma":          2.0,   # focal in Phase 3 only
    "s1_label_smooth_hi":     0.10,   # start: high smoothing
    "s1_label_smooth_lo":     0.01,   # end: low smoothing
    "s1_ema_reinit_phases":   True,   # re-init EMA at each phase boundary

    # ── Architecture ─────────────────────────────────────────────────
    "branch_drop_prob":        0.10,  # stochastic branch dropout prob
    "subcenter_K":              2,    # sub-center ArcFace clusters per class

    # ── Stage 2 ──────────────────────────────────────────────────────
    "s2_epochs":              130,
    "s2_batch":               64,
    "s2_head_lr":             1.5e-4,
    "s2_back_lr":             1.5e-5,
    "s2_min_lr":              1e-7,
    "s2_warmup_ep":             5,
    "s2_sgdr_T0":              10,    # T0=10 → restarts at ep15, ep35, ep75
    "s2_sgdr_Tmult":            2,
    "s2_dropout":              0.10,
    "s2_patience":              45,
    "s2_arcface_s":            32.0,
    "s2_arcface_m":             0.35,
    "s2_arcface_m0":            0.02,
    "s2_arcface_m_delta":       0.10, # extra margin for hardest classes
    "s2_margin_warmup_ep":      50,
    "s2_focal_gamma":            1.5,
    "cdws_max_weight":           3.0,
    "cdws_eps":                  0.05,
    "supcon_weight":             0.15,
    "supcon_temp":               0.10,
    "proto_weight":              0.10,
    "proto_temp":                0.10,
    "bal_n_cls":                 16,
    "bal_n_spc":                  4,

    # ── Stage 3 ──────────────────────────────────────────────────────
    "s3_epochs":              100,
    "s3_swa_lr":              4e-5,
    "s3_cycle_len":             8,
    "s3_sam_rho":              0.05,
    "s3_greedy":              True,

    # Shared
    "weight_decay":           2e-4,
    "grad_clip":               1.0,
    "ema_decay":              0.9999,

    # TTA
    "tta_spatial":              8,
    "tta_spectral":             4,

    # Architecture
    "wl_embed_dim":            16,
    "specf_patch":              8,
    "specf_dim":              128,
    "specf_heads":              4,
    "specf_layers":             4,
    "specf_drop":              0.15,
    "fusion_heads":             4,
    "fusion_drop":             0.10,

    "device":     torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":       42,
    "num_workers": 6,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.cuda.empty_cache()


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
#  ADAPTIVE EMA
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Uses warm-up decay schedule: d = min(max_decay, (1+n)/(10+n))
    which starts near 0 and rises to max_decay quickly.

    v7: reinit_from() method explicitly resets at phase boundaries.
    This is critical: when Stage 1 switches from Phase 2 → Phase 3,
    the loss drops dramatically (CE→Focal, no mixup) and EMA with
    decay=0.9999 takes ~10000 steps to catch up. Re-init lets it
    track the new regime within 100 steps.
    """
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
        """Hard-copy live weights to shadow, reset step counter.
        Use at training-regime boundaries so EMA tracks new regime fast."""
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def state_dict(self)                -> dict: return self.shadow.state_dict()
    def load_state_dict(self, sd: dict):         self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET  —  supports named aug strengths
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    _AUG_HEAVY = dict(band_drop=0.65, cutout=0.50, noise=0.35,
                      warp=0.35, shift=0.30, mult=0.30)
    _AUG_MED   = dict(band_drop=0.35, cutout=0.25, noise=0.20,
                      warp=0.20, shift=0.15, mult=0.15)
    _AUG_LIGHT = dict(band_drop=0.25, cutout=0.15, noise=0.10,
                      warp=0.10, shift=0.10, mult=0.10)
    _NAMED     = {"heavy": _AUG_HEAVY, "medium": _AUG_MED,
                  "light": _AUG_LIGHT, "none": None}

    def __init__(self, patches_path, labels_path, indices,
                 aug_strength="none", max_cutout_bands=20, noise_std=0.02):
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.aug_strength     = aug_strength
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std

    def _probs(self) -> Optional[dict]:
        return self._NAMED.get(self.aug_strength) if isinstance(self.aug_strength, str) \
               else None

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
            s = (new_C - C) // 2; warped = warped[:,:,s:s+C]
        else:
            lo = (C - new_C) // 2; warped = F.pad(warped, (lo, C - new_C - lo))
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
        p     = self._probs()
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
#  SAMPLERS
# ══════════════════════════════════════════════════════════════════════

class ClassBalancedBatchSampler(Sampler):
    """
    Balanced batch sampler: draws n_cls classes × n_spc samples each.
    Optionally accepts per-class sampling weights (CDWS).

    CDWS (Class-Difficulty Weighted Sampler):
      w_c = 1 / (f1_c + eps),  hard classes sampled up to max_weight ×
      more frequently. This is curriculum sampling: the sampler forces
      the model to see hard classes more often within each batch.
    """
    def __init__(self, train_labels, n_cls=16, n_spc=4,
                 class_weights: Optional[Dict[int, float]] = None):
        self.n_cls   = n_cls; self.n_spc = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)
        if class_weights is not None:
            raw        = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
            self.probs = raw / raw.sum()
        else:
            self.probs = None

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False, p=self.probs)
            batch  = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(
                    rng.choice(pool, self.n_spc, replace=len(pool) < self.n_spc).tolist())
            yield batch

    def __len__(self): return self._n


def build_cdws_weights(class_f1: Dict[int, float],
                       num_classes: int,
                       max_weight: float = 3.0,
                       eps: float = 0.05) -> Dict[int, float]:
    """
    Build CDWS weights from per-class F1 scores.
    w_c = clip(1/(f1+eps), 1, max_weight), then normalize to mean=1.
    """
    raw  = {c: min(1.0 / (class_f1.get(c, 0.0) + eps), max_weight)
            for c in range(num_classes)}
    mean = np.mean(list(raw.values()))
    return {c: w / mean for c, w in raw.items()}


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION (Mixup + CutMix)
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def _cutmix(x, y, alpha):
    lam        = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx        = torch.randperm(B, device=x.device)
    r          = math.sqrt(1.0 - lam)
    ch, cw     = int(H*r), int(W*r)
    cx, cy     = random.randint(0,W), random.randint(0,H)
    x1 = max(cx-cw//2,0); x2 = min(cx+cw//2,W)
    y1 = max(cy-ch//2,0); y2 = min(cy+ch//2,H)
    xm = x.clone(); xm[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    return xm, y, y[idx], 1.0-(x2-x1)*(y2-y1)/(W*H)

def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1) < 0.5 else _cutmix)(x, y, alpha)

def mixed_loss(crit, logits, y_a, y_b, lam):
    return lam * crit(logits, y_a) + (1-lam) * crit(logits, y_b)


# ══════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    L = (1 − p_t)^γ × CE
    γ=2.0 in Stage 1 Phase 3  |  γ=1.5 in Stage 2  |  γ=1.0 in Stage 3.
    Ref: Lin et al., Focal Loss for Dense Object Detection, ICCV 2017.
    """
    def __init__(self, gamma: float = 1.5):
        super().__init__(); self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, reduction="none")
        p_t   = torch.exp(-ce)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    All samples sharing the same label form the positive set for each anchor.
    With n_spc=4 per class: each anchor has 3 positives (vs ProtoNCE's 1),
    giving 3× more gradient signal per batch.

    Ref: Khosla et al., Supervised Contrastive Learning, NeurIPS 2020.
    """
    def __init__(self, temperature: float = 0.10):
        super().__init__(); self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B         = features.shape[0]
        sim       = torch.mm(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        n_pos     = pos_mask.float().sum(1)
        if not (n_pos > 0).any():
            return features.new_tensor(0.0, requires_grad=True)
        sim_masked = sim.masked_fill(self_mask, float("-inf"))
        log_prob   = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)
        loss       = -(pos_mask.float() * log_prob).sum(1)
        valid      = n_pos > 0
        return (loss[valid] / n_pos[valid]).mean()


class ProtoNCELoss(nn.Module):
    """Prototypical NCE — class mean prototype as anchor."""
    def __init__(self, temperature: float = 0.10):
        super().__init__(); self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if len(classes) < 2: return features.new_tensor(0.0, requires_grad=True)
        protos  = F.normalize(
            torch.stack([features[labels==c].mean(0) for c in classes]), dim=1)
        sim     = torch.mm(features, protos.T) / self.temperature
        c2l     = {c.item(): i for i,c in enumerate(classes)}
        local   = torch.tensor([c2l[y.item()] for y in labels],
                                dtype=torch.long, device=features.device)
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  SAM — SHARPNESS-AWARE MINIMIZATION
# ══════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
    """
    SAM seeks parameters w* that lie in flat loss basins, not just local
    minima. Two-step: (1) perturb to local worst-case w+ε, (2) gradient
    at w+ε, restore w, step.

    Why for Stage 3: SWA averaging of multiple snapshots benefits greatly
    from each snapshot lying in a flat basin — sharp minima average poorly
    (the average falls in a valley between two peaks). SAM ensures each
    snapshot is flat, so their average is also a good model.

    Ref: Foret et al., SAM: Sharpness-Aware Minimization, ICLR 2021.
    """
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        assert rho >= 0
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
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
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step/second_step explicitly.")

    def _grad_norm(self) -> torch.Tensor:
        dev   = self.param_groups[0]["params"][0].device
        norms = [p.grad.norm(p=2).to(dev)
                 for group in self.param_groups
                 for p in group["params"] if p.grad is not None]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0)

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups


# ══════════════════════════════════════════════════════════════════════
#  ARCFACE — ADAPTIVE SUB-CENTER
# ══════════════════════════════════════════════════════════════════════

class AdaptiveSubcenterArcFaceHead(nn.Module):
    """
    Two extensions over standard ArcFace:

    (A) Sub-center (K centers per class):
        Each class has K cluster centers; target logit = nearest center.
        Relieves single-center bottleneck for bimodal class distributions
        (e.g. classes 51/52 may have two spectral subtypes).
        K=1 → standard ArcFace.
        Ref: Deng et al., Sub-center ArcFace, ECCV 2020.

    (B) Adaptive per-class margins:
        m_c = m_base + m_delta × (1 − f1_c)
        Hard classes → larger angular margin → stronger separation push.
        Set after Stage 1 via update_margins_from_f1().
    """
    def __init__(self, in_dim: int, num_classes: int,
                 K: int = 2, s: float = 32.0,
                 m_base: float = 0.35, m_delta: float = 0.10):
        super().__init__()
        self.K = K; self.C = num_classes
        self.s = s; self.m_base = m_base; self.m_delta = m_delta
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))

    def update_margins_from_f1(self, class_f1: Dict[int, float]) -> None:
        for c, f1 in class_f1.items():
            diff = 1.0 - min(float(f1), 1.0)
            self.margins[c] = self.m_base + self.m_delta * diff
        print(f"[INFO] Adaptive margins: mean={self.margins.mean():.3f}  "
              f"min={self.margins.min():.3f}  max={self.margins.max():.3f}")

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                global_m: Optional[float] = None) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        all_cos = F.linear(x_n, w_n).clamp(-1+1e-6, 1-1e-6)
        cosine  = all_cos.view(-1, self.C, self.K).max(dim=2).values

        if labels is None or not self.training:
            return cosine * self.s

        m_per = (torch.full((x.shape[0],), global_m, device=x.device)
                 if global_m is not None else self.margins[labels])

        cosm = torch.cos(m_per); sinm = torch.sin(m_per)
        th   = torch.cos(math.pi - m_per); mm = torch.sin(math.pi - m_per) * m_per

        sine    = torch.sqrt(torch.clamp(1 - cosine**2, min=1e-6))
        tgt_cos = cosine.gather(1, labels.view(-1,1)).squeeze(1)
        tgt_sin = sine.gather(1, labels.view(-1,1)).squeeze(1)
        phi     = tgt_cos * cosm - tgt_sin * sinm
        phi     = torch.where(tgt_cos > th, phi, tgt_cos - mm)

        oh  = torch.zeros_like(cosine).scatter_(1, labels.view(-1,1).long(), 1.0)
        out = cosine * (1 - oh) + phi.unsqueeze(1) * oh
        return out * self.s

    def init_from_linear(self, linear_weight: torch.Tensor) -> None:
        with torch.no_grad():
            w_n = F.normalize(linear_weight, dim=1)
            for k in range(self.K):
                noise = torch.randn_like(w_n) * 0.01 * k
                self.weight.data[k::self.K].copy_(w_n + noise)
        print(f"[INFO] Sub-center ArcFace K={self.K} init from linear head.")


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class SpectralSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels//reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.GELU(),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x.mean([2,3])).view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm1d(out_ch))
                      if in_ch != out_ch else nn.Identity())
    def forward(self, x):
        return F.gelu(self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x))))) + self.skip(x))


class CBAM(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(c//r, 8)
        self.ch = nn.Sequential(nn.Conv2d(c,mid,1,bias=False), nn.GELU(),
                                 nn.Conv2d(mid,c,1,bias=False))
        self.sp = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False), nn.Sigmoid())
    def forward(self, x):
        x = x * torch.sigmoid(self.ch(x.mean([2,3],keepdim=True)) +
                               self.ch(x.amax([2,3],keepdim=True)))
        return x * self.sp(torch.cat([x.mean(1,keepdim=True), x.amax(1,keepdim=True)], 1))


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
        return F.gelu(self.n3(self.c3(
            F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) + self.skip(x))


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


def masked_spectral_stats(x: torch.Tensor):
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
#  BRANCH A — SPECTRAL PROFILE  (v7: adds 2nd-order derivative)
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    """
    v7: 3-channel input = (signal, 1st deriv, 2nd deriv).
    The 2nd derivative captures spectral curvature / inflection points
    in absorption bands — diagnostically important for separating
    cultivars with similar overall reflectance but different band shapes.
    """
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None):
        super().__init__()
        self.wl_enc = wl_enc
        mk = lambda k: nn.Sequential(ResBlock1D(3, tower_ch//2, k),
                                     ResBlock1D(tower_ch//2, tower_ch, k),
                                     ResBlock1D(tower_ch, tower_ch, k))
        self.tower_s = mk(3); self.tower_m = mk(7); self.tower_l = mk(15)
        self.proj    = nn.Sequential(nn.Linear(tower_ch*6, out_dim),
                                     nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(0.1))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], 1)

    def forward(self, ms):
        s  = ms.unsqueeze(1)
        d1 = F.pad(torch.diff(s,  dim=2), (0,1))
        d2 = F.pad(torch.diff(d1, dim=2), (0,1))
        x  = torch.cat([s, d1, d2], 1)
        if self.wl_enc: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], 1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None):
        super().__init__()
        self.wl_enc = wl_enc
        mk = lambda k: nn.Sequential(ResBlock1D(3, tower_ch//2, k),
                                     ResBlock1D(tower_ch//2, tower_ch, k),
                                     ResBlock1D(tower_ch, tower_ch, k))
        self.tower_s = mk(3); self.tower_m = mk(7); self.tower_l = mk(15)
        self.proj    = nn.Sequential(nn.Linear(tower_ch*6, out_dim),
                                     nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(0.1))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], 1)

    def forward(self, ms, ss, mx):
        x = torch.stack([ms, ss, mx], 1)
        if self.wl_enc: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], 1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands=256, out_dim=256):
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 32, 1, bias=False), nn.GroupNorm(8,32), nn.GELU())
        self.stages = nn.Sequential(
            ResBlock2D(32,64,2), CBAM(64),
            ResBlock2D(64,128,2), CBAM(128),
            ResBlock2D(128,192,2), CBAM(192),
            ResBlock2D(192,out_dim,2))
        self.proj = nn.Sequential(nn.Linear(out_dim*2, out_dim),
                                  nn.BatchNorm1d(out_dim), nn.GELU())

    @staticmethod
    def _pn(x): return x.sign() * x.abs().clamp(1e-8).sqrt()

    def forward(self, x):
        h = self.stages(self.band_reduce(x))
        return self.proj(F.normalize(
            torch.cat([self._pn(h.mean([2,3])), self._pn(h.amax([2,3]))], 1), dim=1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER
# ══════════════════════════════════════════════════════════════════════

class _PreLNBlock(nn.Module):
    def __init__(self, d, heads, d_ff, drop):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d,d_ff), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(d_ff,d), nn.Dropout(drop))
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x    = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))


class SpecFormerBranch(nn.Module):
    def __init__(self, num_bands=256, patch_size=8, d_model=128,
                 n_heads=4, n_layers=4, out_dim=256, dropout=0.15):
        super().__init__()
        n_p             = num_bands // patch_size
        self.patch_size = patch_size; self.n_patches = n_p
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_size, d_model, bias=False), nn.LayerNorm(d_model))
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
        self.proj   = nn.Sequential(nn.Linear(d_model, out_dim),
                                    nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, ms):
        B = ms.shape[0]
        x = ms.float().view(B, self.n_patches, self.patch_size)
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
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d*2,d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(1))
    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        x    = torch.stack(branches, 1)
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x    = x + self.gate * self.drop(h)
        return (x + self.drop(self.ff(self.ln2(x)))).flatten(1)


# ══════════════════════════════════════════════════════════════════════
#  SPECTRALQUADNET v9
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    """
    v9 changes over v8:
      • Branch A: 3-channel (signal + d1 + d2 curvature)
      • Stochastic Branch Dropout: randomly zero 0–2 branches during
        training, forcing each branch to carry the full load independently.
        At inference all branches active = implicit ensemble of all combos.
      • AdaptiveSubcenterArcFaceHead (K=2) + per-class margins
    """
    def __init__(self, num_classes=90, num_bands=256, dropout=0.30,
                 wl_embed_dim=16, cfg=None):
        super().__init__()
        cfg = cfg or CONFIG
        self.branch_drop_prob = cfg.get("branch_drop_prob", 0.0)

        self.se        = SpectralSE(num_bands, 16)
        self.wl_enc    = WavelengthPositionalEncoding(num_bands, wl_embed_dim)
        self.branch_a  = SpectralProfileBranch(256, 80, self.wl_enc)
        self.branch_b  = SpectralStatsBranch(  256, 80, self.wl_enc)
        self.branch_c  = SpatialCNNBranch(num_bands, 256)
        self.branch_d  = SpecFormerBranch(num_bands, cfg["specf_patch"],
                                          cfg["specf_dim"], cfg["specf_heads"],
                                          cfg["specf_layers"], 256, cfg["specf_drop"])
        self.cross_attn   = BranchCrossAttention(256, cfg["fusion_heads"], cfg["fusion_drop"])
        self.embed_net    = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),  nn.BatchNorm1d(256))
        self.linear_head  = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout*0.4), nn.Linear(256, num_classes))
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            256, num_classes, K=cfg.get("subcenter_K", 2),
            s=cfg["s2_arcface_s"], m_base=cfg["s2_arcface_m"],
            m_delta=cfg.get("s2_arcface_m_delta", 0.10))
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

    def set_dropout(self, p: float):
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def use_arcface(self, flag: bool): self._use_arcface = flag

    def freeze_head(self, which: str):
        head = self.linear_head if which=="linear" else self.arcface_head
        for p in head.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which: str):
        head = self.linear_head if which=="linear" else self.arcface_head
        for p in head.parameters(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_embed: bool = False,
                arc_m: Optional[float] = None) -> torch.Tensor:
        x = self.se(x)
        ms, ss, mx = masked_spectral_stats(x)

        ba = self.branch_a(ms)
        bb = self.branch_b(ms, ss, mx)
        bc = self.branch_c(x)
        bd = self.branch_d(ms)

        # Stochastic Branch Dropout — only during training
        if self.training and self.branch_drop_prob > 0:
            live    = [ba, bb, bc, bd]
            dropped = 0
            for i in range(4):
                if dropped < 2 and torch.rand(1).item() < self.branch_drop_prob:
                    live[i] = torch.zeros_like(live[i]); dropped += 1
            ba, bb, bc, bd = live

        emb = self.embed_net(self.cross_attn([ba, bb, bc, bd]))

        if self._use_arcface:
            emb_n  = F.normalize(F.gelu(emb), dim=1)
            logits = self.arcface_head(emb_n, labels, global_m=arc_m)
        else:
            logits = self.linear_head(emb)

        if return_embed:
            return logits, F.normalize(F.gelu(emb.detach()), dim=1)
        return logits


# ══════════════════════════════════════════════════════════════════════
#  TTA  (8 spatial + 4 spectral = 12 views)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor,
                n_spatial: int = 8, n_spectral: int = 4) -> torch.Tensor:
    device = x.device; logits = []
    for k, flip in [(k,f) for k in range(4) for f in (False,True)][:n_spatial]:
        aug = torch.rot90(x, k, [2,3])
        if flip: aug = torch.flip(aug, [3])
        with autocast(device_type=device.type): logits.append(model(aug))
    step = max(256 // max(n_spectral*2, 1), 1)
    shifts = ([-step*i for i in range(1, n_spectral//2+1)] +
              [ step*i for i in range(1, n_spectral//2+1)])[:n_spectral]
    for sh in shifts:
        aug = torch.roll(x, sh, dims=1)
        with autocast(device_type=device.type): logits.append(model(aug))
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


def _make_ldr(train_idx, aug, batch, kw, balanced=False,
              all_labels=None, class_weights=None):
    ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                         train_idx, aug_strength=aug)
    if balanced and all_labels is not None:
        samp = ClassBalancedBatchSampler(
            all_labels[train_idx], CONFIG["bal_n_cls"], CONFIG["bal_n_spc"],
            class_weights=class_weights)
        return DataLoader(ds, batch_sampler=samp,
                          persistent_workers=True, prefetch_factor=2, **kw)
    return DataLoader(ds, batch_size=batch, shuffle=True,
                      persistent_workers=True, prefetch_factor=2, **kw)


def build_loaders(train_idx, val_idx, test_idx, batch_train=64,
                  balanced=False, all_labels=None, train_aug="none",
                  class_weights=None):
    kw = dict(num_workers=CONFIG["num_workers"], pin_memory=True)
    tr = _make_ldr(train_idx, train_aug, batch_train, kw,
                   balanced, all_labels, class_weights)
    va = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], val_idx),
        batch_size=64, shuffle=False, **kw)
    te = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], test_idx),
        batch_size=64, shuffle=False,
        **{**kw, "num_workers": 2})
    return tr, va, te


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
    head_params, back_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (head_params if name.startswith("arcface_head") else back_params).append((name, p))
    return optim.AdamW(_wd_split(head_params, head_lr) + _wd_split(back_params, backbone_lr))

def build_optimizer_s3(model, lr):
    return optim.AdamW(_wd_split(model.named_parameters(), lr))


# ══════════════════════════════════════════════════════════════════════
#  LR SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def sgdr_scheduler(optimizer, warmup_ep=5, T_0=10, T_mult=2,
                   eta_min_frac=1e-3) -> optim.lr_scheduler.LambdaLR:
    """
    Linear warmup → SGDR cosine cycles with multiplicative period growth.
    T0=10, Tmult=2: restarts at ep15, ep35, ep75 (3 restarts in 130 epochs).
    Each restart temporarily re-elevates LR, allowing escape from local minima.
    """
    def _l(ep):
        if ep < warmup_ep: return max(ep / max(warmup_ep,1), 1e-6)
        t = ep - warmup_ep
        clen = T_0; elapsed = 0
        while t >= elapsed + clen:
            elapsed += clen; clen = max(int(clen * T_mult), 1)
        ratio = (t - elapsed) / max(clen, 1)
        return eta_min_frac + 0.5*(1-eta_min_frac)*(1+math.cos(math.pi*ratio))
    return optim.lr_scheduler.LambdaLR(optimizer, _l)


def arcface_margin(ep, m0, m_target, warmup_ep):
    if ep >= warmup_ep: return m_target
    return m0 + (m_target-m0)*0.5*(1-math.cos(math.pi*ep/max(warmup_ep,1)))


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler, ema, device,
                    scheduler=None, use_mixup=True, mixup_alpha=0.4,
                    supcon=None, supcon_weight=0.0,
                    proto=None,  proto_weight=0.0,
                    accum_steps=1, arc_m=None):
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
                cls_loss    = criterion(logits, y_a)
                sc_loss     = supcon(emb, y_a)
                pt_loss     = proto(emb, y_a) if proto is not None else 0.0
                loss = ((1-supcon_weight-proto_weight)*cls_loss
                        + supcon_weight*sc_loss + proto_weight*pt_loss)
            else:
                logits = model(
                    x_in,
                    labels=(y_a if model._use_arcface and not use_mixup else None),
                    arc_m=arc_m)
                loss   = mixed_loss(criterion, logits, y_a, y_b, lam)

        if not torch.isfinite(loss):
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


def train_one_epoch_sam(model, loader, sam_optimizer, criterion, device,
                        supcon=None, supcon_weight=0.0,
                        proto=None, proto_weight=0.0, arc_m=None):
    """
    SAM training: two forward-backward passes per batch.
    Pass 1: compute loss, perturb weights to local worst case.
    Pass 2: compute loss at perturbed point, restore weights, take step.
    No AMP (ArcFace numerics sensitive to FP16).
    """
    model.train()
    total_loss = total_acc = 0.0

    def _compute_loss(x, y):
        if supcon is not None:
            lo, emb = model(x, y, return_embed=True, arc_m=arc_m)
            return ((1-supcon_weight-proto_weight) * criterion(lo, y)
                    + supcon_weight  * supcon(emb, y)
                    + proto_weight   * (proto(emb, y) if proto else 0)), lo
        else:
            lo = model(x, labels=y, arc_m=arc_m)
            return criterion(lo, y), lo

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        sam_optimizer.zero_grad()
        loss1, logits = _compute_loss(x, y)
        if not torch.isfinite(loss1): continue
        loss1.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_optimizer.first_step(zero_grad=True)

        loss2, _ = _compute_loss(x, y)
        if not torch.isfinite(loss2):
            # restore weights manually
            for group in sam_optimizer.param_groups:
                for p in group["params"]:
                    if "old_p" in sam_optimizer.state.get(p, {}):
                        p.data = sam_optimizer.state[p]["old_p"]
            sam_optimizer.zero_grad(); continue

        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_optimizer.second_step(zero_grad=True)

        total_loss += loss1.item()
        with torch.no_grad():
            total_acc += (logits.detach().argmax(1) == y).float().mean().item()

    n = len(loader)
    return total_loss/n, total_acc/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type): logits = model(x)
        preds.append(logits.argmax(1).cpu()); targets.append(y)
    p, t = torch.cat(preds), torch.cat(targets)
    return f1_score(t,p,average="macro",zero_division=0), accuracy_score(t,p)


@torch.no_grad()
def evaluate_per_class(model, loader, device, num_classes: int) -> Dict[int, float]:
    model.eval(); preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type): logits = model(x)
        preds.append(logits.argmax(1).cpu()); targets.append(y)
    p, t   = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    f1_arr = f1_score(t, p, average=None, zero_division=0,
                      labels=list(range(num_classes)))
    return {i: float(f) for i, f in enumerate(f1_arr)}


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
    torch.save({"epoch":epoch,"stage":stage,
                "model":model.state_dict(),"ema":ema.state_dict(),
                "val_acc":val_acc,"val_f1":val_f1,
                "use_arcface":model._use_arcface}, path)

def load_ckpt(path, model, ema, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); ema.load_state_dict(ckpt["ema"])
    flag = ckpt.get("use_arcface", False)
    model.use_arcface(flag); ema.shadow.use_arcface(flag)
    return ckpt

def update_bn_stats(loader, model, device):
    model.train()
    for m in model.modules():
        if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
            m.reset_running_stats(); m.momentum = None
    with torch.no_grad():
        for x,_ in loader: model(x.to(device, non_blocking=True))
    model.eval()

def _hdr(title, epochs):
    w=66; print(f"\n{'═'*w}\n  {title}  [{epochs} epochs max]\n{'═'*w}")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1  —  3-phase progressive augmentation
# ══════════════════════════════════════════════════════════════════════

def run_stage1(model, ema,
               ldr_heavy, ldr_medium, ldr_light, val_ldr,
               device, best_ckpt: str) -> float:
    """
    Phase 1 (0–40%): heavy aug + mixup + label_smooth=0.10→decaying
      → wide exploration, prevents overfit on small dataset
    Phase 2 (40–70%): medium aug + mixup + label_smooth decaying
      → consolidation: features solidify with lighter distortion
    Phase 3 (70–100%): light aug + NO mixup + FocalLoss γ=2
      → discrimination: near-clean data, hard-class focusing
      EMA re-initialized at Phase 3 start so it can quickly track
      the regime shift (loss drops ~2.3→0.9 immediately at transition).

    Progressive label smoothing: hi=0.10 → lo=0.01 linearly.
    High smoothing at start prevents the model from being overconfident
    on potentially mislabeled or ambiguous augmented samples.
    Low smoothing at end sharpens class boundaries before ArcFace.
    """
    model.use_arcface(False)
    model.unfreeze_head("linear"); model.freeze_head("arcface")

    ep_total = CONFIG["s1_epochs"]
    p1_end   = int(ep_total * CONFIG["s1_phase1_frac"])
    p2_end   = int(ep_total * (CONFIG["s1_phase1_frac"] + CONFIG["s1_phase2_frac"]))

    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"]/25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CONFIG["s1_max_lr"], epochs=ep_total,
            steps_per_epoch=math.ceil(len(ldr_heavy)/CONFIG["s1_accum"]),
            pct_start=0.25, div_factor=25, final_div_factor=1e4, anneal_strategy="cos")

    scaler       = GradScaler()
    focal_s1     = FocalLoss(gamma=CONFIG["s1_focal_gamma"])
    ls_hi        = CONFIG["s1_label_smooth_hi"]
    ls_lo        = CONFIG["s1_label_smooth_lo"]
    best_acc     = 0.0; no_improve = 0
    ema_reinited = [False, False]

    _hdr("Stage 1 — 3-Phase Progressive Augmentation", ep_total)
    print(f"  Phase 1: ep 1–{p1_end}:    heavy+mixup  LS={ls_hi:.2f}→decaying")
    print(f"  Phase 2: ep {p1_end+1}–{p2_end}: medium+mixup LS decaying")
    print(f"  Phase 3: ep {p2_end+1}–{ep_total}: light+focal(γ={CONFIG['s1_focal_gamma']}) NO mixup")

    for ep in range(1, ep_total+1):
        # ── Phase selection ──────────────────────────────────────────
        if ep <= p1_end:
            phase = 1; cur_ldr = ldr_heavy; use_mx = True
        elif ep <= p2_end:
            phase = 2; cur_ldr = ldr_medium; use_mx = True
        else:
            phase = 3; cur_ldr = ldr_light;  use_mx = False

        # ── EMA re-init at boundaries ────────────────────────────────
        if phase == 2 and not ema_reinited[0] and CONFIG.get("s1_ema_reinit_phases"):
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 2 boundary (ep {ep})")
            ema_reinited[0] = True
        if phase == 3 and not ema_reinited[1] and CONFIG.get("s1_ema_reinit_phases"):
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 3 boundary (ep {ep}) "
                  f"— critical: EMA now tracks clean-data regime")
            ema_reinited[1] = True

        # ── Progressive label smoothing ──────────────────────────────
        t         = (ep - 1) / max(ep_total - 1, 1)
        ls_now    = ls_hi * (1-t) + ls_lo * t
        criterion = (focal_s1 if phase == 3
                     else nn.CrossEntropyLoss(label_smoothing=ls_now))

        tl, ta = train_one_epoch(
            model, cur_ldr, optimizer, criterion, scaler, ema, device,
            scheduler=scheduler, use_mixup=use_mx,
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

        print(f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Train {ta:.1%} │ "
              f"Live {va_live:.1%}  EMA {va_ema:.1%} │ "
              f"LR {lr_now:.2e}  LS {ls_now:.3f} [P{phase}]{saved}")

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("arcface")
    return best_acc


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2  —  Sub-ctr ArcFace + SupCon + CDWS + Adaptive margins
# ══════════════════════════════════════════════════════════════════════

def run_stage2(model, ema, train_ldr, val_ldr, device, best_ckpt,
               class_f1: Optional[Dict[int,float]] = None) -> float:
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
        warmup_ep    = CONFIG["s2_warmup_ep"],
        T_0          = CONFIG["s2_sgdr_T0"],
        T_mult       = CONFIG["s2_sgdr_Tmult"],
        eta_min_frac = CONFIG["s2_min_lr"] / CONFIG["s2_head_lr"])

    sc_w  = CONFIG["supcon_weight"]
    pt_w  = CONFIG["proto_weight"]
    best_acc = 0.0; no_improve = 0
    ep_total = CONFIG["s2_epochs"]

    r1 = CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"]
    r2 = r1 + CONFIG["s2_sgdr_T0"] * CONFIG["s2_sgdr_Tmult"]

    _hdr("Stage 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR", ep_total)
    print(f"  Head LR: {CONFIG['s2_head_lr']:.1e}  Backbone LR: {CONFIG['s2_back_lr']:.1e}")
    print(f"  SGDR T0={CONFIG['s2_sgdr_T0']}, Tmult={CONFIG['s2_sgdr_Tmult']} "
          f"→ restarts ep {r1} & {r2}")
    print(f"  ArcFace K={CONFIG['subcenter_K']}  "
          f"m={CONFIG['s2_arcface_m0']}→{CONFIG['s2_arcface_m']}+Δ{CONFIG['s2_arcface_m_delta']}")
    print(f"  Losses: Focal(γ={CONFIG['s2_focal_gamma']}) + "
          f"SupCon(w={sc_w}) + ProtoNCE(w={pt_w})")

    for ep in range(1, ep_total+1):
        m_now  = arcface_margin(ep-1, CONFIG["s2_arcface_m0"],
                                CONFIG["s2_arcface_m"], CONFIG["s2_margin_warmup_ep"])
        ramp   = min(1.0, ep / 10.0)
        sc_now = sc_w * ramp; pt_now = pt_w * ramp

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, focal, scaler=None, ema=ema,
            device=device, scheduler=None,
            use_mixup=False, supcon=supcon, supcon_weight=sc_now,
            proto=proto, proto_weight=pt_now, arc_m=m_now)
        scheduler.step()

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

        rf = " ↻R1" if ep==r1 else (" ↻R2" if ep==r2 else "")
        print(f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Train {ta:.1%} │ "
              f"Live {va_live:.1%}  EMA {va_ema:.1%} │ "
              f"hLR {head_lr:.1e} bLR {back_lr:.1e}  m={m_now:.3f}{saved}{rf}")

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("linear")
    return best_acc


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3  —  SAM + Greedy SWA + Balanced Sampler
# ══════════════════════════════════════════════════════════════════════

def run_stage3_swa(model, ema, train_ldr, val_ldr, device,
                   best_ckpt: str, prev_best_val: float) -> float:
    """
    v7 Stage 3:
      • SAM optimizer: seeks flat loss basins ideal for SWA averaging.
        Flat-basin snapshots average into a model that's also in a flat
        basin — sharp minima average poorly.
      • Greedy SWA: snapshot accepted only if live val ≥ 98% of best
        seen so far. Filters out transient low-accuracy cycle starts.
      • ClassBalancedSampler + CDWS: same as Stage 2, hard classes sampled
        3× more frequently throughout Stage 3 fine-tuning.
      • 100 epochs: model still rising at ep60 in v6.
      • FocalLoss γ=1.0 + SupCon + ProtoNCE (mild) maintain metric structure.
    """
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True); ema.shadow.use_arcface(True)

    base_params = _wd_split(model.named_parameters(), CONFIG["s3_swa_lr"])
    sam = SAM(base_params, optim.AdamW, rho=CONFIG["s3_sam_rho"],
              lr=CONFIG["s3_swa_lr"], weight_decay=CONFIG["weight_decay"])

    focal_s3  = FocalLoss(gamma=1.0)
    supcon_s3 = SupConLoss(temperature=0.10)
    proto_s3  = ProtoNCELoss(temperature=0.10)

    swa_state : dict  = copy.deepcopy(model.state_dict())
    n_snap            = 1
    n_rejected        = 0
    best_live_s3      = 0.0

    _hdr("Stage 3 — SAM + Greedy SWA + Balanced Sampler", CONFIG["s3_epochs"])
    print(f"  SAM ρ={CONFIG['s3_sam_rho']}  Peak LR={CONFIG['s3_swa_lr']:.0e}  "
          f"Cycle={CONFIG['s3_cycle_len']} ep")
    print(f"  Greedy SWA: {'ON' if CONFIG['s3_greedy'] else 'OFF'}  |  "
          f"Losses: Focal(γ=1) + SupCon(0.05) + ProtoNCE(0.05)")

    for ep in range(1, CONFIG["s3_epochs"]+1):
        cycle_ep = (ep-1) % CONFIG["s3_cycle_len"]
        lr_now   = CONFIG["s3_swa_lr"] * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cycle_ep / CONFIG["s3_cycle_len"])))
        for pg in sam.param_groups: pg["lr"] = lr_now

        tl, ta = train_one_epoch_sam(
            model, train_ldr, sam, focal_s3, device,
            supcon=supcon_s3, supcon_weight=0.05,
            proto=proto_s3,   proto_weight=0.05,
            arc_m=CONFIG["s2_arcface_m"])

        _, va_live = evaluate(model, val_ldr, device)
        best_live_s3 = max(best_live_s3, va_live)

        snap_info = ""
        if ep % CONFIG["s3_cycle_len"] == 0:
            accept = (not CONFIG["s3_greedy"]) or (va_live >= best_live_s3 * 0.98)
            if accept:
                n_snap += 1
                a  = 1.0 / n_snap
                sd = model.state_dict()
                for k in swa_state:
                    swa_state[k] = swa_state[k] + a * (sd[k] - swa_state[k])
                snap_info = f"  ★ snap {n_snap}"
            else:
                n_rejected += 1
                snap_info = f"  ✗ rejected (live {va_live:.1%})"

        print(f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ Loss {tl:.4f}  Train {ta:.1%} │ "
              f"Live {va_live:.1%} │ LR {lr_now:.2e} │ Snaps {n_snap}{snap_info}")

    print(f"\nUpdating BN stats ({n_snap} snaps, {n_rejected} rejected) ...")
    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state)
    swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)

    _, va_swa = evaluate(swa_model, val_ldr, device)
    print(f"SWA val: {va_swa:.1%}")

    ema.shadow.load_state_dict(swa_model.state_dict())
    ema.shadow.use_arcface(True)

    if va_swa > prev_best_val:
        print(f"Stage 3 val {va_swa:.1%} > Stage 2 best {prev_best_val:.1%} → saving.")
        save_ckpt(best_ckpt, CONFIG["s3_epochs"], "Stage 3",
                  swa_model, ema, va_swa, 0.0)
    else:
        print(f"Stage 3 val {va_swa:.1%} ≤ {prev_best_val:.1%}. Keeping Stage 2 checkpoint.")

    return va_swa


# ══════════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt: str):
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")
    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow; eval_model.eval()
    print(f"  ArcFace : {eval_model._use_arcface}")
    print(f"  Ckpt    : ep {ckpt['epoch']} | {ckpt['stage']} | val={ckpt['val_acc']:.1%}")
    print(f"  TTA     : {CONFIG['tta_spatial']} spatial + {CONFIG['tta_spectral']} spectral "
          f"= {CONFIG['tta_spatial']+CONFIG['tta_spectral']} views")

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x = x.to(device, non_blocking=True)
            if use_tta:
                logits = tta_predict(eval_model, x,
                                     CONFIG["tta_spatial"], CONFIG["tta_spectral"])
            else:
                with autocast(device_type=device.type): logits = eval_model(x)
            preds.append(logits.argmax(1).cpu()); targets.append(y)
        p, t         = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p, t)
        acc = accuracy_score(t,p); f1m = f1_score(t,p,average="macro",zero_division=0)
        f1w = f1_score(t,p,average="weighted",zero_division=0)
        print(f"\n  [{tag}]  Acc={acc:.1%}  F1(macro)={f1m:.4f}  F1(wt)={f1w:.4f}")

    p_tta, t_tta = results["TTA   "]
    print(f"\nClassification Report (TTA):\n")
    print(classification_report(t_tta, p_tta, zero_division=0))

    out = CONFIG["output_dir"]
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy",   p_tta)
    np.save(f"{out}/test_targets.npy",     t_tta)
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
        dropout=CONFIG["s1_dropout"], wl_embed_dim=CONFIG["wl_embed_dim"],
        cfg=CONFIG).to(device)
    ema   = ModelEMA(model, CONFIG["ema_decay"])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : SpectralQuadNet v9 (v7 training)")
    print(f"Params : {n_par/1e6:.2f}M")
    print(f"Device : {device}")
    print(f"v7 key changes:")
    print(f"  S1: 3-phase aug, EMA reinit at boundaries, prog label smooth")
    print(f"  S2: Sub-ctr ArcFace K={CONFIG['subcenter_K']}, CDWS+SupCon, adaptive margins")
    print(f"  S3: SAM ρ={CONFIG['s3_sam_rho']}, Greedy SWA, 100ep, balanced sampler")

    # ── Stage 1 ─────────────────────────────────────────────────────
    if done_stage < 1:
        print("\n[RUN] Stage 1")
        kw = dict(num_workers=CONFIG["num_workers"], pin_memory=True,
                  persistent_workers=True, prefetch_factor=2)
        def _s1_ldr(aug):
            ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                                 train_idx, aug_strength=aug)
            return DataLoader(ds, batch_size=CONFIG["s1_batch"], shuffle=True, **kw)

        _, val_ldr1, _ = build_loaders(train_idx, val_idx, test_idx,
                                       CONFIG["s1_batch"], train_aug="none")
        run_stage1(model, ema,
                   _s1_ldr("heavy"), _s1_ldr("medium"), _s1_ldr("light"),
                   val_ldr1, device, ckpt_s1)
    else:
        print("\n[SKIP] Stage 1 → loading checkpoint")
        load_ckpt(ckpt_s1, model, ema, device)

    # ── Prepare Stage 2 ─────────────────────────────────────────────
    if done_stage < 2:
        # Init sub-center ArcFace from linear head
        print("\n[INFO] Initialising Sub-center ArcFace from linear head")
        lw = model.linear_head[-1].weight.data.clone()
        model.arcface_head.init_from_linear(lw)
        ema.shadow.arcface_head.init_from_linear(lw)

        # Per-class difficulty for CDWS + adaptive margins
        print("[INFO] Computing class-difficulty from Stage 1 model ...")
        _, val_cd, _ = build_loaders(train_idx, val_idx, test_idx, 64)
        class_f1     = evaluate_per_class(ema.shadow, val_cd, device, CONFIG["num_classes"])
        hard_cls     = sorted([(c,f) for c,f in class_f1.items() if f < 0.50],
                              key=lambda x: x[1])
        print(f"[INFO] Hard classes (F1<0.50): "
              f"{[(c,f'{f:.2f}') for c,f in hard_cls[:10]]}")
        cdws_wts = build_cdws_weights(class_f1, CONFIG["num_classes"],
                                      CONFIG["cdws_max_weight"], CONFIG["cdws_eps"])

        print("\n[RUN] Stage 2")
        tr2, va2, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"],
            balanced=True, all_labels=all_labels,
            train_aug="light", class_weights=cdws_wts)
        run_stage2(model, ema, tr2, va2, device, ckpt_s2, class_f1)
    else:
        print("\n[SKIP] Stage 2 → loading checkpoint")
        load_ckpt(ckpt_s2, model, ema, device)

    # ── Stage 3 ─────────────────────────────────────────────────────
    if done_stage < 3:
        print("\n[RUN] Stage 3 (SAM + Greedy SWA)")
        s2_best = torch.load(ckpt_s2, map_location=device,
                             weights_only=False).get("val_acc", 0.0)

        # Re-compute class difficulty after Stage 2 for updated CDWS
        _, val_cd3, _ = build_loaders(train_idx, val_idx, test_idx, 64)
        class_f1_s2   = evaluate_per_class(ema.shadow, val_cd3, device, CONFIG["num_classes"])
        cdws_s3       = build_cdws_weights(class_f1_s2, CONFIG["num_classes"],
                                           CONFIG["cdws_max_weight"], CONFIG["cdws_eps"])
        tr3, va3, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"],
            balanced=True, all_labels=all_labels,
            train_aug="light", class_weights=cdws_s3)
        run_stage3_swa(model, ema, tr3, va3, device, ckpt_s3, prev_best_val=s2_best)
    else:
        print("\n[SKIP] Stage 3 → loading checkpoint")
        load_ckpt(ckpt_s3, model, ema, device)

    # ── Final Evaluation ─────────────────────────────────────────────
    print("\n[INFO] Final Evaluation")
    final_ckpt = (ckpt_s3 if stage_exists(3)
                  else ckpt_s2 if stage_exists(2) else ckpt_s1)
    _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx, 64)
    final_evaluation(model, ema, test_ldr, device, final_ckpt)


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