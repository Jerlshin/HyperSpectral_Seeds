#!/usr/bin/env python3
"""
hsi_training_v7.py  —  SpectralQuadNet v9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rice HSI  90-class classification.  Resume-safe (skips completed stages).
Target: 85–90% TTA accuracy.

═════════════════════════ v6 → v7 DELTA LOG ════════════════════════

[S1-A] 3-PHASE PROGRESSIVE AUGMENTATION  (replaces binary fade-out)
  Phase 1 (0–40%): heavy aug + mixup  + high label-smooth (0.10)
  Phase 2 (40–70%): medium aug + mixup + decaying label-smooth
  Phase 3 (70–100%): light aug, NO mixup, Focal γ=2.0, low smooth (0.01)
  Motivation: user's binary fade-out gave +6pp — a smoother 3-phase
  curriculum should give an additional +1-2pp by avoiding the abrupt
  regime change at epoch 191.

[S1-B] EMA BOUNDARY REINIT
  EMA re-initialized at Phase 2 and Phase 3 boundaries.
  With decay=0.9999, EMA needs ~10k steps to forget the old regime.
  A hard reinit lets EMA track the new training regime immediately.
  In v6 EMA peaked at 68.5% (vs live 74.3%) — this is the fix.

[S1-C] PROGRESSIVE LABEL SMOOTHING
  Linear decay from ls_hi=0.10 → ls_lo=0.01 over all s1_epochs.
  High smoothing early prevents overconfident early predictions on
  noisy augmented batches; low smoothing late sharpens boundaries.

[S2-A] SUB-CENTER ARCFACE  K=2
  Each class gets K=2 cluster centres. The target logit uses the
  nearest centre per sample. Relieves the single-centre constraint
  for spectrally bimodal classes (51 F1=0.19, 52 F1=0.32, 49 F1=0.33).
  Ref: Wang et al., Sub-center ArcFace, ECCV 2020.

[S2-B] ADAPTIVE PER-CLASS MARGINS
  m_c = m_base + m_delta × (1 − F1_c).   Hard classes get larger margin.
  Computed from Stage 1 EMA F1 per class before Stage 2 starts.
  Easy classes (F1→1): m=0.35.  Hardest (F1→0): m=0.35+0.10=0.45.

[S2-C] SUPERVISED CONTRASTIVE LOSS  (SupCon, replaces ProtoNCE primary)
  All same-label samples = positives for each anchor.
  With 4 samples/class per batch → 3 positives/anchor vs ProtoNCE's 1.
  3× more contrastive gradient per batch.
  Ref: Khosla et al., NeurIPS 2020.  ProtoNCE kept as secondary term.

[S2-D] CLASS-DIFFICULTY WEIGHTED SAMPLER  (CDWS)
  Sampling weight w_c = clip(1/(F1_c + 0.05), 1, 3).
  Hard classes oversample up to 3×.  Computed from Stage 1 F1 scores.
  Applied on top of the existing ClassBalancedBatchSampler.

[S2-E] SGDR T0=10, Tmult=2 (vs v6 T0=15)
  More restart opportunities → restarts at ep15, ep35, ep75.
  Better escape from local minima in the sub-center ArcFace landscape.

[S3-A] SAM OPTIMIZER  (Sharpness-Aware Minimization)
  Finds flatter minima → better generalisation and better SWA targets.
  Two-step per batch: perturb → compute gradient → restore → step.
  rho=0.05 is standard for AdamW-based SAM.
  Ref: Foret et al., ICLR 2021.

[S3-B] GREEDY SWA
  Accept snapshot only if live val ≥ 98% of best_live_so_far.
  Prevents contaminating the SWA average with bad cycle-start weights.

[S3-C] CLASS-BALANCED SAMPLER IN STAGE 3
  Re-enabled (v6 used random sampler). Re-uses CDWS weights from Stage 2
  F1 scores so hard classes remain oversampled during SWA refinement.

[S3-D] 100 EPOCHS  (v6: 60)
  In v6 train acc was still rising at ep60 (48%→64%) — SWA cut off
  too early. Extended to 100 gives 12 full SWA cycles.

[ARCH-A] 2nd-ORDER SPECTRAL DERIVATIVE IN BRANCH A
  3-channel input: [signal, d1, d2].  d2 (curvature) captures
  absorption-band inflection points that are diagnostically informative
  for separating spectrally-similar rice cultivars.

[ARCH-B] STOCHASTIC BRANCH DROPOUT
  Each of the 4 branches is independently zeroed with p=0.10 during
  training. Forces each branch to carry the full representation load
  independently → stronger individual branches and a more robust ensemble.
  Similar to DropPath/stochastic depth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

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
    # ── Paths ─────────────────────────────────────────────────────────
    "patches_data":  "./dataset/patches.npy",
    "labels_path":   "./dataset/labels.npy",
    "output_dir":    "./output_v9/",

    # ── Dataset ───────────────────────────────────────────────────────
    "num_bands":     256,
    "num_classes":   90,

    # ── Stage 1: 3-phase progressive augmentation ─────────────────────
    "s1_epochs":          300,
    "s1_phase1_frac":     0.40,   # heavy aug + mixup
    "s1_phase2_frac":     0.30,   # medium aug + mixup
    # phase 3 = remaining 30%: light aug, NO mixup, focal ON
    "s1_batch":            64,
    "s1_max_lr":           8e-4,
    "s1_dropout":          0.30,
    "s1_mixup":            0.40,
    "s1_patience":         60,
    "s1_accum":             2,
    "s1_focal_gamma":       2.0,  # Phase 3 focal γ
    "s1_label_smooth_hi":  0.10,  # epoch 1 label smoothing
    "s1_label_smooth_lo":  0.01,  # final epoch label smoothing
    "s1_ema_reinit_phases": True, # re-init EMA at phase boundaries

    # ── Architecture extras ───────────────────────────────────────────
    "branch_drop_prob":    0.10,  # stochastic branch dropout
    "subcenter_K":          2,    # sub-centers per class in ArcFace

    # ── Stage 2 ───────────────────────────────────────────────────────
    "s2_epochs":           120,
    "s2_batch":             64,
    "s2_head_lr":          1.5e-4,
    "s2_back_lr":          1.5e-5,
    "s2_min_lr":           1e-7,
    "s2_warmup_ep":          5,
    "s2_sgdr_T0":           10,   # v7: 15→10, more restarts
    "s2_sgdr_Tmult":         2,   # restarts at ep15, ep35, ep75
    "s2_dropout":           0.10,
    "s2_patience":           40,
    "s2_arcface_s":         32.0,
    "s2_arcface_m":          0.35,
    "s2_arcface_m0":         0.02,
    "s2_arcface_m_delta":    0.10, # extra margin for hardest classes
    "s2_margin_warmup_ep":   50,
    "s2_focal_gamma":         1.5,
    # CDWS: oversampling weights for hard classes
    "cdws_max_weight":        3.0,
    "cdws_eps":               0.05,
    # Contrastive losses
    "supcon_weight":           0.15,
    "supcon_temp":             0.10,
    "proto_weight":            0.08,
    "proto_temp":              0.10,
    # Class-balanced batch sampler
    "bal_n_cls":               16,
    "bal_n_spc":                4,

    # ── Stage 3 ───────────────────────────────────────────────────────
    "s3_epochs":            100,   # v7: 60→100
    "s3_swa_lr":            4e-5,
    "s3_cycle_len":           8,
    "s3_sam_rho":             0.05, # SAM perturbation radius
    "s3_greedy":            True,   # greedy SWA snapshot selection

    # ── Shared ────────────────────────────────────────────────────────
    "weight_decay":         2e-4,
    "grad_clip":             1.0,
    "ema_decay":            0.9999,

    # ── TTA ───────────────────────────────────────────────────────────
    "tta_spatial":             8,
    "tta_spectral":            4,

    # ── Architecture ─────────────────────────────────────────────────
    "wl_embed_dim":           16,
    "specf_patch":             8,
    "specf_dim":             128,
    "specf_heads":             4,
    "specf_layers":            4,
    "specf_drop":             0.15,
    "fusion_heads":            4,
    "fusion_drop":            0.10,

    "device":    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":      42,
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
    Uses the warm-start decay formula: d = min(max_decay, (1+n)/(10+n)).
    Exposes reinit_from() for hard-reset at training-regime boundaries.
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
        """Hard-copy live weights → shadow and reset counter.
        Call at training-regime boundaries so EMA tracks the new regime
        immediately instead of lagging for thousands of steps.
        """
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def state_dict(self) -> dict:        return self.shadow.state_dict()
    def load_state_dict(self, sd: dict): self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET  — multi-profile aug + dynamic strength
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    _PROFILES = {
        "heavy":  dict(band_drop=0.65, cutout=0.50, noise=0.35,
                       warp=0.35,  shift=0.30, mult=0.30),
        "medium": dict(band_drop=0.35, cutout=0.25, noise=0.20,
                       warp=0.20,  shift=0.15, mult=0.15),
        "light":  dict(band_drop=0.25, cutout=0.15, noise=0.10,
                       warp=0.10,  shift=0.10, mult=0.10),
        "none":   None,
    }

    def __init__(self, patches_path, labels_path, indices,
                 aug_strength="none", max_cutout_bands=20, noise_std=0.02):
        self.patches          = np.load(patches_path, mmap_mode="r")
        self.labels           = np.load(labels_path)
        self.indices          = indices
        self.aug_strength     = aug_strength
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std

    def __len__(self): return len(self.indices)

    def _probs(self) -> Optional[dict]:
        return self._PROFILES.get(str(self.aug_strength))

    # ── Augmentation primitives ───────────────────────────────────────
    def _band_dropout(self, x):
        return x * (torch.rand(x.shape[0]) > 0.04).float().view(-1,1,1)

    def _band_cutout(self, x):
        x = x.clone(); nb = x.shape[0]
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
    Draws n_cls classes per batch, n_spc samples per class.
    Optional class_weights dict biases the class-selection probability
    toward hard classes (CDWS — Class-Difficulty Weighted Sampler).
    """
    def __init__(self, train_labels, n_cls=16, n_spc=4,
                 class_weights: Optional[Dict[int,float]] = None):
        self.n_cls   = n_cls; self.n_spc = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)
        if class_weights is not None:
            raw  = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
            self.probs = raw / raw.sum()
        else:
            self.probs = None

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls,
                                replace=False, p=self.probs)
            batch  = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(
                    rng.choice(pool, self.n_spc, replace=len(pool)<self.n_spc).tolist())
            yield batch

    def __len__(self): return self._n


def build_cdws_weights(class_f1: Dict[int,float], num_classes: int,
                       max_w: float = 3.0, eps: float = 0.05) -> Dict[int,float]:
    """
    w_c = clip(1 / (F1_c + eps), 1, max_w), then normalize to mean=1.
    Hard classes get up to max_w times more sampling budget.
    """
    raw = {c: min(1.0/(class_f1.get(c, 0.0)+eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w/mean for c,w in raw.items()}


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION  (Mixup + CutMix)
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def _cutmix(x, y, alpha):
    lam       = float(np.random.beta(alpha, alpha))
    B,C,H,W   = x.shape
    idx       = torch.randperm(B, device=x.device)
    r         = math.sqrt(1.0 - lam)
    ch,cw     = int(H*r), int(W*r)
    cx,cy     = random.randint(0,W), random.randint(0,H)
    x1=max(cx-cw//2,0); x2=min(cx+cw//2,W)
    y1=max(cy-ch//2,0); y2=min(cy+ch//2,H)
    xm=x.clone(); xm[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    return xm, y, y[idx], 1.0-(x2-x1)*(y2-y1)/(W*H)

def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1)<0.5 else _cutmix)(x,y,alpha)

def mixed_loss(crit, logits, ya, yb, lam):
    return lam*crit(logits,ya) + (1-lam)*crit(logits,yb)


# ══════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: L = (1-p_t)^γ × CE.
    γ=2.0 (Stage 1 Phase 3) | γ=1.5 (Stage 2) | γ=1.0 (Stage 3 mild).
    Ref: Lin et al., ICCV 2017.
    """
    def __init__(self, gamma: float = 1.5):
        super().__init__(); self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    All same-label samples are positives for each anchor.
    Expects L2-normalised feature vectors.
    With n_spc=4 per class in balanced batches → 3 positives per anchor
    (vs ProtoNCE's 1 prototype) — 3× more gradient per step.
    Ref: Khosla et al., NeurIPS 2020.
    """
    def __init__(self, temperature: float = 0.10):
        super().__init__(); self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B    = features.shape[0]
        sim  = torch.mm(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask  = (labels.unsqueeze(0)==labels.unsqueeze(1)) & ~self_mask
        n_pos = pos_mask.float().sum(1)
        if not (n_pos > 0).any():
            return features.new_tensor(0.0, requires_grad=True)
        sim_m    = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim_m - torch.logsumexp(sim_m, dim=1, keepdim=True)
        loss     = -(pos_mask.float() * log_prob).sum(1)
        valid    = n_pos > 0
        return (loss[valid] / n_pos[valid]).mean()


class ProtoNCELoss(nn.Module):
    """Class-mean prototype → contrastive CE. Secondary to SupCon."""
    def __init__(self, temperature: float = 0.10):
        super().__init__(); self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if len(classes) < 2: return features.new_tensor(0.0, requires_grad=True)
        protos = F.normalize(
            torch.stack([features[labels==c].mean(0) for c in classes]), dim=1)
        sim    = torch.mm(features, protos.T) / self.temperature
        c2l    = {c.item():i for i,c in enumerate(classes)}
        local  = torch.tensor([c2l[y.item()] for y in labels],
                               dtype=torch.long, device=features.device)
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  SAM  —  Sharpness-Aware Minimization
# ══════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
    """
    SAM: perturb weights to worst-case neighbour, then take gradient
    step at perturbed point. Result: weights in a flat basin of the
    loss landscape → better generalisation + better SWA averaging.

    Usage:
        sam = SAM(params, optim.AdamW, rho=0.05, lr=lr, ...)
        # forward + backward at w
        loss.backward()
        sam.first_step(zero_grad=True)
        # forward + backward at w+ε
        loss2.backward()
        sam.second_step(zero_grad=True)

    Ref: Foret et al., ICLR 2021.
    """
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step/second_step.")

    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        ns  = [p.grad.norm(p=2).to(dev)
               for g in self.param_groups for p in g["params"]
               if p.grad is not None]
        return torch.norm(torch.stack(ns), p=2) if ns else torch.tensor(0.0)

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups


# ══════════════════════════════════════════════════════════════════════
#  ADAPTIVE SUB-CENTER ARCFACE HEAD
# ══════════════════════════════════════════════════════════════════════

class AdaptiveSubcenterArcFaceHead(nn.Module):
    """
    Combines two ArcFace extensions:

    (A) Sub-center ArcFace (K per-class cluster centres):
        target logit = max cosine over K centres.
        K=1 → standard ArcFace.  K=2 → sub-center.
        Ref: Wang et al., ECCV 2020.

    (B) Adaptive per-class margins:
        m_c = m_base + m_delta × (1 - F1_c).
        Hard classes get extra margin; easy classes keep base margin.
        Updated from Stage 1 F1 scores via update_margins_from_f1().
    """
    def __init__(self, in_dim: int, num_classes: int,
                 K: int = 2, s: float = 32.0,
                 m_base: float = 0.35, m_delta: float = 0.10):
        super().__init__()
        self.K = K; self.C = num_classes
        self.s = s; self.m_base = m_base; self.m_delta = m_delta
        self.weight = nn.Parameter(torch.FloatTensor(num_classes*K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))

    def update_margins_from_f1(self, class_f1: Dict[int,float]):
        for c, f1 in class_f1.items():
            self.margins[c] = self.m_base + self.m_delta*(1.0 - min(float(f1),1.0))
        print(f"[INFO] Adaptive ArcFace margins — "
              f"mean={self.margins.mean():.3f}  "
              f"min={self.margins.min():.3f}  "
              f"max={self.margins.max():.3f}")

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                global_m: Optional[float] = None) -> torch.Tensor:
        x_n  = F.normalize(x, dim=1)
        w_n  = F.normalize(self.weight, dim=1)
        # [B, C*K] → [B, C] via per-class max
        cosine = (F.linear(x_n, w_n).clamp(-1+1e-6, 1-1e-6)
                   .view(-1, self.C, self.K).max(dim=2).values)

        if labels is None or not self.training:
            return cosine * self.s

        m_per  = (torch.full((x.shape[0],), global_m, device=x.device)
                  if global_m is not None else self.margins[labels])
        cosm   = torch.cos(m_per); sinm = torch.sin(m_per)
        th     = torch.cos(math.pi - m_per)
        mm     = torch.sin(math.pi - m_per) * m_per

        sine   = torch.sqrt(torch.clamp(1 - cosine**2, min=1e-6))
        tgt_c  = cosine.gather(1, labels.view(-1,1)).squeeze(1)
        tgt_s  = sine.gather(1,   labels.view(-1,1)).squeeze(1)
        phi    = tgt_c*cosm - tgt_s*sinm
        phi    = torch.where(tgt_c > th, phi, tgt_c - mm)

        oh  = torch.zeros_like(cosine).scatter_(1, labels.view(-1,1).long(), 1.0)
        return ((oh*phi.unsqueeze(1)) + ((1-oh)*cosine)) * self.s

    def init_from_linear(self, linear_w: torch.Tensor):
        """Bootstrap K sub-centre weights from linear head, plus small noise."""
        with torch.no_grad():
            wn = F.normalize(linear_w, dim=1)
            for k in range(self.K):
                noise = torch.randn_like(wn) * 0.01 * k
                self.weight[k::self.K].copy_(wn + noise)
        print(f"[INFO] Sub-center ArcFace (K={self.K}) bootstrapped from linear head.")


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE BUILDING BLOCKS
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
        return F.gelu(self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x)))))
                      + self.skip(x))


class CBAM(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(c//r, 8)
        self.ch = nn.Sequential(nn.Conv2d(c,mid,1,bias=False),nn.GELU(),
                                 nn.Conv2d(mid,c,1,bias=False))
        self.sp = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.Sigmoid())
    def forward(self, x):
        x = x * torch.sigmoid(self.ch(x.mean([2,3],keepdim=True))
                               + self.ch(x.amax([2,3],keepdim=True)))
        return x * self.sp(torch.cat([x.mean(1,keepdim=True),
                                       x.amax(1,keepdim=True)],1))


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid = max(out_ch//2, in_ch)
        self.c1=nn.Conv2d(in_ch,mid,1,bias=False);  self.n1=nn.GroupNorm(min(8,mid),mid)
        self.c2=nn.Conv2d(mid,mid,3,stride,1,bias=False); self.n2=nn.GroupNorm(min(8,mid),mid)
        self.c3=nn.Conv2d(mid,out_ch,1,bias=False); self.n3=nn.GroupNorm(min(8,out_ch),out_ch)
        self.skip=(nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride=stride,bias=False),
                                  nn.GroupNorm(min(8,out_ch),out_ch))
                   if (stride!=1 or in_ch!=out_ch) else nn.Identity())
    def forward(self, x):
        return F.gelu(self.n3(self.c3(F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x))))))))
                      + self.skip(x))


class WavelengthPositionalEncoding(nn.Module):
    def __init__(self, num_bands=256, embed_dim=16):
        super().__init__()
        wl   = torch.linspace(0.0,1.0,num_bands)
        half = embed_dim//2
        freq = torch.exp(torch.arange(half).float()*-(math.log(1e4)/max(half-1,1)))
        enc  = torch.zeros(num_bands, embed_dim)
        enc[:,:half] = torch.sin(wl.unsqueeze(1)*freq.unsqueeze(0))
        enc[:,half:] = torch.cos(wl.unsqueeze(1)*freq.unsqueeze(0))
        self.register_buffer("enc", enc)
        self.proj = nn.Linear(embed_dim, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
    def forward(self): return self.proj(self.enc).squeeze(-1).view(1,1,-1)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A  —  SPECTRAL PROFILE  (v7: adds 2nd derivative channel)
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    """
    3-channel input: [signal, d1 (1st derivative), d2 (2nd derivative)].
    2nd derivative = spectral curvature.  Inflection points in absorption
    bands are diagnostically informative for cultivar separation.
    """
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None):
        super().__init__()
        self.wl_enc = wl_enc
        mk = lambda k: nn.Sequential(
            ResBlock1D(3, tower_ch//2, k),       # 3 channels now
            ResBlock1D(tower_ch//2, tower_ch, k),
            ResBlock1D(tower_ch, tower_ch, k))
        self.tower_s = mk(3); self.tower_m = mk(7); self.tower_l = mk(15)
        self.proj    = nn.Sequential(nn.Linear(tower_ch*6, out_dim),
                                     nn.BatchNorm1d(out_dim), nn.GELU(),
                                     nn.Dropout(0.1))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], 1)

    def forward(self, ms):
        s  = ms.unsqueeze(1)                     # [B,1,256]
        d1 = F.pad(torch.diff(s,  dim=2),(0,1))  # 1st derivative
        d2 = F.pad(torch.diff(d1, dim=2),(0,1))  # 2nd derivative (curvature)
        x  = torch.cat([s, d1, d2], 1)           # [B,3,256]
        if self.wl_enc: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], 1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH B  —  SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(self, out_dim=256, tower_ch=80, wl_enc=None):
        super().__init__()
        self.wl_enc = wl_enc
        mk = lambda k: nn.Sequential(ResBlock1D(3,tower_ch//2,k),
                                     ResBlock1D(tower_ch//2,tower_ch,k),
                                     ResBlock1D(tower_ch,tower_ch,k))
        self.tower_s=mk(3); self.tower_m=mk(7); self.tower_l=mk(15)
        self.proj = nn.Sequential(nn.Linear(tower_ch*6,out_dim),
                                  nn.BatchNorm1d(out_dim),nn.GELU(),nn.Dropout(0.1))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], 1)

    def forward(self, ms, ss, mx):
        x = torch.stack([ms,ss,mx],1)
        if self.wl_enc: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], 1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C  —  SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands=256, out_dim=256):
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands,32,1,bias=False), nn.GroupNorm(8,32), nn.GELU())
        self.stages = nn.Sequential(
            ResBlock2D(32,64,2),   CBAM(64),
            ResBlock2D(64,128,2),  CBAM(128),
            ResBlock2D(128,192,2), CBAM(192),
            ResBlock2D(192,out_dim,2))
        self.proj = nn.Sequential(nn.Linear(out_dim*2,out_dim),
                                  nn.BatchNorm1d(out_dim),nn.GELU())

    @staticmethod
    def _pn(x): return x.sign() * x.abs().clamp(1e-8).sqrt()

    def forward(self, x):
        h = self.stages(self.band_reduce(x))
        return self.proj(F.normalize(
            torch.cat([self._pn(h.mean([2,3])), self._pn(h.amax([2,3]))], 1), dim=1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D  —  SPECFORMER
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
        n_p = num_bands//patch_size
        self.patch_size=patch_size; self.n_patches=n_p
        self.patch_proj = nn.Sequential(nn.Linear(patch_size,d_model,bias=False),
                                        nn.LayerNorm(d_model))
        wl_n = (torch.linspace(WL_MIN,WL_MAX,n_p)-WL_MIN)/(WL_MAX-WL_MIN)
        half = d_model//2
        freq = torch.exp(torch.arange(half).float()*-(math.log(1e4)/max(half-1,1)))
        pe   = torch.zeros(n_p,d_model)
        pe[:,:half]=torch.sin(wl_n.unsqueeze(1)*freq.unsqueeze(0))
        pe[:,half:]=torch.cos(wl_n.unsqueeze(1)*freq.unsqueeze(0))
        self.register_buffer("wl_pe",pe)
        self.cls    = nn.Parameter(torch.zeros(1,1,d_model))
        nn.init.trunc_normal_(self.cls,std=0.02)
        self.blocks = nn.ModuleList([_PreLNBlock(d_model,n_heads,d_model*2,dropout)
                                     for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.proj   = nn.Sequential(nn.Linear(d_model,out_dim),
                                    nn.BatchNorm1d(out_dim),nn.GELU(),nn.Dropout(dropout))

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

    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        x    = torch.stack(branches, 1)
        h,_  = self.attn(self.ln1(x),self.ln1(x),self.ln1(x),need_weights=False)
        x    = x + self.gate*self.drop(h)
        return (x + self.drop(self.ff(self.ln2(x)))).flatten(1)


# ══════════════════════════════════════════════════════════════════════
#  SPECTRAL STATISTICS HELPER
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(x: torch.Tensor):
    x32  = x.float(); B,C,H,W = x32.shape
    flat = x32.reshape(B,C,H*W)
    mask = (flat.abs().sum(1,keepdim=True)>1e-5).float()
    cnt  = mask.sum(2).clamp(min=1.0)
    mean = (flat*mask).sum(2)/cnt
    std  = ((flat**2*mask).sum(2)/cnt - mean**2).clamp(min=1e-6).sqrt()
    mx   = flat.masked_fill(mask.expand_as(flat)==0,-1e4).max(2).values
    mx   = mx.masked_fill(mx<-9999.0,0.0)
    return (torch.nan_to_num(mean,0), torch.nan_to_num(std,0),
            torch.nan_to_num(mx,0))


# ══════════════════════════════════════════════════════════════════════
#  SPECTRALQUADNET v9
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    """
    v9 upgrades vs v8:
      ▸ Branch A: 3-channel input [sig, d1, d2] — curvature features
      ▸ Stochastic Branch Dropout: p=0.10 per branch during training,
        min 2 branches always survive → forces independent competence
      ▸ AdaptiveSubcenterArcFaceHead (K=2, adaptive per-class margins)
    """
    def __init__(self, num_classes=90, num_bands=256,
                 dropout=0.30, wl_embed_dim=16, cfg=None):
        super().__init__()
        cfg = cfg or CONFIG
        self.branch_drop_prob = cfg.get("branch_drop_prob", 0.0)

        self.se       = SpectralSE(num_bands, 16)
        self.wl_enc   = WavelengthPositionalEncoding(num_bands, wl_embed_dim)
        self.branch_a = SpectralProfileBranch(256, 80, self.wl_enc)
        self.branch_b = SpectralStatsBranch(  256, 80, self.wl_enc)
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(num_bands, cfg["specf_patch"],
                                         cfg["specf_dim"], cfg["specf_heads"],
                                         cfg["specf_layers"], 256, cfg["specf_drop"])
        self.cross_attn = BranchCrossAttention(256, cfg["fusion_heads"],
                                               cfg["fusion_drop"])
        self.embed_net  = nn.Sequential(
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,256),  nn.BatchNorm1d(256))
        self.linear_head  = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout*0.4), nn.Linear(256, num_classes))
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            256, num_classes, K=cfg.get("subcenter_K",2),
            s=cfg["s2_arcface_s"], m_base=cfg["s2_arcface_m"],
            m_delta=cfg.get("s2_arcface_m_delta", 0.10))
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

    def set_dropout(self, p: float):
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def use_arcface(self, flag: bool): self._use_arcface = flag

    def freeze_head(self, which: str):
        h = self.linear_head if which=="linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which: str):
        h = self.linear_head if which=="linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(True)

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

        # ── Stochastic Branch Dropout ─────────────────────────────────
        if self.training and self.branch_drop_prob > 0:
            branches = [ba, bb, bc, bd]
            originals= [ba, bb, bc, bd]
            kept = 0
            for i in range(4):
                if torch.rand(1).item() < self.branch_drop_prob:
                    branches[i] = torch.zeros_like(branches[i])
                else:
                    kept += 1
            if kept < 2:  # ensure ≥2 branches survive
                rescue = random.randint(0,3)
                branches[rescue] = originals[rescue]
            ba, bb, bc, bd = branches

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
#  TTA  —  8 spatial + 4 spectral = 12 views
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor,
                n_spatial: int = 8, n_spectral: int = 4) -> torch.Tensor:
    device = x.device; logits = []
    combos = [(k,f) for k in range(4) for f in (False,True)][:n_spatial]
    for k, flip in combos:
        aug = torch.rot90(x, k, [2,3])
        if flip: aug = torch.flip(aug,[3])
        with autocast(device_type=device.type): logits.append(model(aug))
    step = max(256//(max(n_spectral,1)*2), 1)
    shifts = ([-step*i for i in range(1, n_spectral//2+1)] +
              [ step*i for i in range(1, n_spectral//2+1)])[:n_spectral]
    for sh in shifts:
        with autocast(device_type=device.type):
            logits.append(model(torch.roll(x, sh, dims=1)))
    return torch.stack(logits).mean(0)


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits():
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr,tmp  = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va,te   = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(train_idx, val_idx, test_idx, batch_train,
                  balanced=False, all_labels=None,
                  train_aug="none",
                  class_weights: Optional[Dict[int,float]] = None):
    nw = CONFIG["num_workers"]
    kw = dict(num_workers=nw, pin_memory=True)
    ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                         train_idx, aug_strength=train_aug)
    if balanced and all_labels is not None:
        samp   = ClassBalancedBatchSampler(all_labels[train_idx],
                                           CONFIG["bal_n_cls"], CONFIG["bal_n_spc"],
                                           class_weights=class_weights)
        tr_ldr = DataLoader(ds, batch_sampler=samp,
                            persistent_workers=True, prefetch_factor=2, **kw)
    else:
        tr_ldr = DataLoader(ds, batch_size=batch_train, shuffle=True,
                            persistent_workers=True, prefetch_factor=2, **kw)
    va_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], val_idx),
        batch_size=64, shuffle=False, **kw)
    te_ldr = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], test_idx),
        batch_size=64, shuffle=False, **{**kw,"num_workers":2})
    return tr_ldr, va_ldr, te_ldr


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISERS
# ══════════════════════════════════════════════════════════════════════

def _wd_groups(named_params, lr):
    wd, no_wd = [], []
    for n,p in named_params:
        if not p.requires_grad: continue
        (no_wd if (p.ndim==1 or n.endswith(".bias")) else wd).append(p)
    return [{"params":wd,    "lr":lr, "weight_decay":CONFIG["weight_decay"]},
            {"params":no_wd, "lr":lr, "weight_decay":0.0}]

def build_optimizer_s1(model, lr):
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))

def build_optimizer_s2(model, head_lr, back_lr):
    hp,bp = [],[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (hp if n.startswith("arcface_head") else bp).append((n,p))
    return optim.AdamW(_wd_groups(hp, head_lr) + _wd_groups(bp, back_lr))

def build_optimizer_s3(model, lr):
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))


# ══════════════════════════════════════════════════════════════════════
#  LR SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def sgdr_scheduler(optimizer, warmup_ep=5, T_0=10, T_mult=2,
                   eta_min_frac=1e-3) -> optim.lr_scheduler.LambdaLR:
    """
    Linear warmup → cosine cycle → restart → longer cosine cycle → …
    v7 default (T0=10, Tmult=2): restarts at ep15, ep35, ep75.
    """
    def _l(ep):
        if ep < warmup_ep:
            return max(ep/max(warmup_ep,1), 1e-6)
        t = ep - warmup_ep
        clen=T_0; elapsed=0
        while t >= elapsed+clen:
            elapsed += clen; clen = max(int(clen*T_mult),1)
        ratio = (t-elapsed)/max(clen,1)
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

    for step, (x,y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_in, ya, yb, lam = mixed_aug(x,y,mixup_alpha) if use_mixup else (x,y,y,1.0)

        with autocast(device_type=device.type, enabled=use_amp):
            if supcon is not None:
                logits, emb = model(x_in, ya, return_embed=True, arc_m=arc_m)
                cls_l  = criterion(logits, ya)
                sc_l   = supcon(emb, ya)
                pt_l   = proto(emb, ya) if proto is not None else 0.0
                loss   = ((1-supcon_weight-proto_weight)*cls_l
                          + supcon_weight*sc_l + proto_weight*pt_l)
            else:
                arc_labels = (ya if model._use_arcface and not use_mixup else None)
                logits = model(x_in, labels=arc_labels, arc_m=arc_m)
                loss   = mixed_loss(criterion, logits, ya, yb, lam)

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
            total_acc += (logits.argmax(1)==y).float().mean().item()

    n = max(len(loader),1)
    return total_loss/n, total_acc/n


def train_one_epoch_sam(model, loader, sam_opt, criterion, device,
                        supcon=None, supcon_weight=0.0,
                        proto=None, proto_weight=0.0, arc_m=None):
    """
    SAM training loop (Stage 3).
    Two forward-backward passes per batch:
      Pass 1: compute loss at w, perturb w → w + ε̂
      Pass 2: compute loss at w+ε̂, restore w, apply gradient step
    Result: weights in a flat basin → better generalisation & SWA targets.
    No AMP: ArcFace requires FP32 for numerical stability.
    """
    model.train()
    total_loss = total_acc = 0.0

    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ── Pass 1: forward at w ─────────────────────────────────────
        sam_opt.zero_grad()
        if supcon is not None:
            logits, emb = model(x, y, return_embed=True, arc_m=arc_m)
            loss = ((1-supcon_weight-proto_weight)*criterion(logits,y)
                    + supcon_weight*(supcon(emb,y) if supcon else 0)
                    + proto_weight*(proto(emb,y) if proto else 0))
        else:
            logits = model(x, labels=y, arc_m=arc_m)
            loss   = criterion(logits, y)

        if not torch.isfinite(loss): continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.first_step(zero_grad=True)

        # ── Pass 2: forward at w + ε ─────────────────────────────────
        if supcon is not None:
            logits2, emb2 = model(x, y, return_embed=True, arc_m=arc_m)
            loss2 = ((1-supcon_weight-proto_weight)*criterion(logits2,y)
                     + supcon_weight*(supcon(emb2,y) if supcon else 0)
                     + proto_weight*(proto(emb2,y) if proto else 0))
        else:
            logits2 = model(x, labels=y, arc_m=arc_m)
            loss2   = criterion(logits2, y)

        if not torch.isfinite(loss2):
            # restore weights manually on bad pass
            sam_opt.zero_grad()
            for g in sam_opt.param_groups:
                for p in g["params"]:
                    if "old_p" in sam_opt.state.get(p,{}):
                        p.data = sam_opt.state[p]["old_p"]
            continue

        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.second_step(zero_grad=True)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.detach().argmax(1)==y).float().mean().item()

    n = max(len(loader),1)
    return total_loss/n, total_acc/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); preds,targets=[],[]
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type):
            logits = model(x)
        preds.append(logits.argmax(1).cpu()); targets.append(y)
    p,t = torch.cat(preds), torch.cat(targets)
    return f1_score(t,p,average="macro",zero_division=0), accuracy_score(t,p)


@torch.no_grad()
def evaluate_per_class(model, loader, device, num_classes: int) -> Dict[int,float]:
    model.eval(); preds,targets=[],[]
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type): logits = model(x)
        preds.append(logits.argmax(1).cpu()); targets.append(y)
    p,t   = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    f1arr = f1_score(t,p,average=None,zero_division=0,labels=list(range(num_classes)))
    return {i:float(f) for i,f in enumerate(f1arr)}


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def stage_ckpt_path(s):  return os.path.join(CONFIG["output_dir"],f"best_stage{s}.pth")
def stage_exists(s):     return os.path.isfile(stage_ckpt_path(s))
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
            m.reset_running_stats(); m.momentum=None
    with torch.no_grad():
        for x,_ in loader: model(x.to(device, non_blocking=True))
    model.eval()


# ══════════════════════════════════════════════════════════════════════
#  CLASS DIFFICULTY  (computed after Stage 1)
# ══════════════════════════════════════════════════════════════════════

def compute_class_difficulty(model, ema, val_ldr, device) -> Dict[int,float]:
    """Per-class F1 from Stage 1 EMA model → drives CDWS + adaptive margins."""
    print("[INFO] Computing per-class difficulty from Stage 1 EMA ...")
    class_f1 = evaluate_per_class(ema.shadow, val_ldr, device, CONFIG["num_classes"])
    hard = [(c,f) for c,f in class_f1.items() if f < 0.50]
    hard.sort(key=lambda x:x[1])
    print(f"[INFO] Hard classes (F1<0.50): "
          f"{[(c,f'{f:.2f}') for c,f in hard[:10]]}")
    return class_f1


# ══════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════

def _hdr(title, n):
    w=66; print(f"\n{'═'*w}\n  {title}  [{n} epochs max]\n{'═'*w}")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1  —  3-PHASE PROGRESSIVE AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def run_stage1(model, ema, loaders_by_phase, val_ldr, device,
               best_ckpt: str) -> float:
    """
    3-phase curriculum:

    Phase 1 (0–40%):   heavy aug + mixup + high label-smooth
      → Explore broad hypothesis space under extreme augmentation.
    Phase 2 (40–70%):  medium aug + mixup + decaying label-smooth
      → Consolidate: features sharpen while still regularised.
    Phase 3 (70–100%): light aug + NO mixup + Focal(γ=2.0) + low smooth
      → Discriminate: near-clean data + focal sharpens hard boundaries.
      EMA re-initialised at phase start to track rapid improvements.

    Progressive label smoothing: ls_hi → ls_lo over all epochs.
    """
    model.use_arcface(False)
    model.unfreeze_head("linear"); model.freeze_head("arcface")

    ep_total = CONFIG["s1_epochs"]
    p1_end   = int(ep_total * CONFIG["s1_phase1_frac"])
    p2_end   = int(ep_total * (CONFIG["s1_phase1_frac"]+CONFIG["s1_phase2_frac"]))

    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"]/25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CONFIG["s1_max_lr"], epochs=ep_total,
            steps_per_epoch=math.ceil(len(loaders_by_phase[1])/CONFIG["s1_accum"]),
            pct_start=0.25, div_factor=25, final_div_factor=1e4,
            anneal_strategy="cos")

    scaler       = GradScaler()
    focal_p3     = FocalLoss(gamma=CONFIG["s1_focal_gamma"])
    ls_hi        = CONFIG["s1_label_smooth_hi"]
    ls_lo        = CONFIG["s1_label_smooth_lo"]
    best_acc     = 0.0
    no_improve   = 0
    ema_reinited = [False, False]

    _hdr("Stage 1 — 3-Phase Progressive Augmentation", ep_total)
    print(f"  Phase 1: ep 1–{p1_end}    heavy aug + mixup")
    print(f"  Phase 2: ep {p1_end+1}–{p2_end}  medium aug + mixup")
    print(f"  Phase 3: ep {p2_end+1}–{ep_total}  light aug, NO mixup, Focal γ={CONFIG['s1_focal_gamma']}")
    print(f"  Label smooth: {ls_hi} → {ls_lo}  |  EMA reinit at each phase boundary")

    for ep in range(1, ep_total+1):
        # ── Determine phase ───────────────────────────────────────────
        if   ep <= p1_end: phase=1; cur_ldr=loaders_by_phase[1]; use_mx=True
        elif ep <= p2_end: phase=2; cur_ldr=loaders_by_phase[2]; use_mx=True
        else:              phase=3; cur_ldr=loaders_by_phase[3]; use_mx=False

        # ── EMA hard-reset at phase boundaries ───────────────────────
        if phase==2 and not ema_reinited[0] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 2 boundary (ep {ep})")
            ema_reinited[0] = True
        if phase==3 and not ema_reinited[1] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 3 boundary (ep {ep})")
            ema_reinited[1] = True

        # ── Progressive label smoothing ───────────────────────────────
        t       = (ep-1) / max(ep_total-1, 1)
        ls_now  = ls_hi*(1-t) + ls_lo*t
        crit    = (focal_p3 if phase==3
                   else nn.CrossEntropyLoss(label_smoothing=ls_now))

        tl,ta = train_one_epoch(
            model, cur_ldr, optimizer, crit, scaler, ema, device,
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
#  STAGE 2  —  Sub-ctr ArcFace + SupCon + CDWS + Adaptive Margins
# ══════════════════════════════════════════════════════════════════════

def run_stage2(model, ema, train_ldr, val_ldr, device, best_ckpt,
               class_f1: Optional[Dict[int,float]] = None) -> float:
    """
    v7 Stage 2:
      • Sub-center ArcFace K=2 + adaptive per-class margins from Stage 1 F1
      • SupCon primary + ProtoNCE secondary contrastive losses
      • CDWS sampling: hard classes oversampled up to 3×
      • SGDR T0=10, Tmult=2 → restarts at ep15, ep35, ep75
    """
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
        optimizer, warmup_ep=CONFIG["s2_warmup_ep"],
        T_0=CONFIG["s2_sgdr_T0"], T_mult=CONFIG["s2_sgdr_Tmult"],
        eta_min_frac=CONFIG["s2_min_lr"]/CONFIG["s2_head_lr"])

    sc_w = CONFIG["supcon_weight"]; pt_w = CONFIG["proto_weight"]
    ep_total = CONFIG["s2_epochs"]
    best_acc = 0.0; no_improve = 0

    r1 = CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"]
    r2 = r1 + CONFIG["s2_sgdr_T0"]*CONFIG["s2_sgdr_Tmult"]

    _hdr("Stage 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR", ep_total)
    print(f"  Head LR: {CONFIG['s2_head_lr']:.1e}  |  "
          f"Backbone LR: {CONFIG['s2_back_lr']:.1e}  (1/10×)")
    print(f"  SGDR T0={CONFIG['s2_sgdr_T0']}, Tmult={CONFIG['s2_sgdr_Tmult']} "
          f"→ restarts at ep {r1} & ep {r2}")
    print(f"  ArcFace K={CONFIG['subcenter_K']}  "
          f"m={CONFIG['s2_arcface_m0']}→{CONFIG['s2_arcface_m']}"
          f"+Δ{CONFIG['s2_arcface_m_delta']}  over {CONFIG['s2_margin_warmup_ep']} ep")
    print(f"  Losses: FocalCE(γ={CONFIG['s2_focal_gamma']}) + "
          f"SupCon(w={sc_w}) + ProtoNCE(w={pt_w})")
    print(f"  Sampler: CDWS {'active' if class_f1 else 'n/a (uniform)'}")

    for ep in range(1, ep_total+1):
        m_now  = arcface_margin(ep-1, CONFIG["s2_arcface_m0"],
                                CONFIG["s2_arcface_m"], CONFIG["s2_margin_warmup_ep"])
        ramp   = min(1.0, ep/10.0)   # warm up contrastive losses over first 10 ep
        sc_now = sc_w * ramp; pt_now = pt_w * ramp

        tl,ta = train_one_epoch(
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
              f"hLR {head_lr:.1e} bLR {back_lr:.1e}  m={m_now:.3f}"
              f"{saved}{rf}")

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("linear")
    return best_acc


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3  —  SAM + GREEDY SWA + BALANCED SAMPLER
# ══════════════════════════════════════════════════════════════════════

def run_stage3_swa(model, ema, train_ldr, val_ldr, device,
                   best_ckpt: str, prev_best_val: float) -> float:
    """
    v7 Stage 3:
      • SAM optimizer (ρ=0.05): finds flat minima ideal for SWA averaging
      • Greedy SWA: accept snapshot only if live val ≥ 98% of session best
      • Class-balanced + CDWS sampler (same as Stage 2)
      • 100 epochs, 12 SWA cycles
      • FocalLoss γ=1.0 (mild) + SupCon(0.05) + ProtoNCE(0.05)
    """
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True); ema.shadow.use_arcface(True)

    params = list(_wd_groups(model.named_parameters(), CONFIG["s3_swa_lr"]))
    sam    = SAM(params, optim.AdamW, rho=CONFIG["s3_sam_rho"],
                 lr=CONFIG["s3_swa_lr"], weight_decay=CONFIG["weight_decay"])

    focal_s3  = FocalLoss(gamma=1.0)
    supcon_s3 = SupConLoss(temperature=0.10)
    proto_s3  = ProtoNCELoss(temperature=0.10)

    swa_state = copy.deepcopy(model.state_dict())
    n_snap    = 1; n_rejected = 0
    best_live = 0.0

    _hdr("Stage 3 — SAM + Greedy SWA + Balanced Sampler", CONFIG["s3_epochs"])
    print(f"  SAM ρ={CONFIG['s3_sam_rho']}  |  Cycle={CONFIG['s3_cycle_len']} ep  |  "
          f"Peak LR={CONFIG['s3_swa_lr']:.0e}")
    print(f"  Greedy SWA: {'✓' if CONFIG['s3_greedy'] else '✗'}  |  "
          f"Losses: FocalCE(γ=1) + SupCon(0.05) + ProtoNCE(0.05)")

    for ep in range(1, CONFIG["s3_epochs"]+1):
        cycle_ep = (ep-1) % CONFIG["s3_cycle_len"]
        lr_now   = CONFIG["s3_swa_lr"] * (
            0.1 + 0.9*0.5*(1+math.cos(math.pi*cycle_ep/CONFIG["s3_cycle_len"])))
        for pg in sam.param_groups: pg["lr"] = lr_now

        tl,ta = train_one_epoch_sam(
            model, train_ldr, sam, focal_s3, device,
            supcon=supcon_s3, supcon_weight=0.05,
            proto=proto_s3,  proto_weight=0.05,
            arc_m=CONFIG["s2_arcface_m"])

        _, va_live = evaluate(model, val_ldr, device)
        best_live  = max(best_live, va_live)

        snap_info = ""
        if ep % CONFIG["s3_cycle_len"] == 0:
            # Greedy acceptance: accept if ≥98% of this session's best
            if not CONFIG["s3_greedy"] or va_live >= best_live*0.98:
                n_snap += 1; a=1.0/n_snap
                sd = model.state_dict()
                for k in swa_state:
                    swa_state[k] = swa_state[k] + a*(sd[k]-swa_state[k])
                snap_info = f"  ★ snap {n_snap} accepted"
            else:
                n_rejected += 1
                snap_info  = f"  ✗ rejected (live {va_live:.1%} < {best_live*0.98:.1%})"

        print(f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ Loss {tl:.4f}  Train {ta:.1%} │ "
              f"Live {va_live:.1%} │ LR {lr_now:.2e} │ "
              f"Snaps {n_snap}{snap_info}")

    print(f"\nUpdating BN stats for SWA model "
          f"({n_snap} accepted, {n_rejected} rejected) ...")
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
        print(f"Stage 3 val {va_swa:.1%} ≤ Stage 2 best {prev_best_val:.1%}. "
              f"Keeping Stage 2 checkpoint.")

    return va_swa


# ══════════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt):
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")
    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow; eval_model.eval()

    print(f"  ArcFace active : {eval_model._use_arcface}")
    print(f"  Checkpoint     : epoch {ckpt['epoch']} | {ckpt['stage']} "
          f"| val={ckpt['val_acc']:.1%}")
    print(f"  TTA views      : {CONFIG['tta_spatial']} spatial + "
          f"{CONFIG['tta_spectral']} spectral = "
          f"{CONFIG['tta_spatial']+CONFIG['tta_spectral']} total")

    results = {}
    for tag, use_tta in [("No TTA",False),("TTA   ",True)]:
        preds,targets=[],[]
        for x,y in test_ldr:
            x = x.to(device, non_blocking=True)
            logits = (tta_predict(eval_model, x,
                                  CONFIG["tta_spatial"], CONFIG["tta_spectral"])
                      if use_tta else eval_model(x))
            preds.append(logits.argmax(1).cpu()); targets.append(y)
        p,t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p,t)
        acc = accuracy_score(t,p)
        f1m = f1_score(t,p,average="macro",    zero_division=0)
        f1w = f1_score(t,p,average="weighted", zero_division=0)
        print(f"\n  [{tag}]  Acc={acc:.1%}  F1(macro)={f1m:.4f}  F1(wt)={f1w:.4f}")

    p_tta,t_tta = results["TTA   "]
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
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  "
          f"Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx)//CONFIG['num_classes']}")

    model = SpectralQuadNet(
        num_classes=CONFIG["num_classes"], num_bands=CONFIG["num_bands"],
        dropout=CONFIG["s1_dropout"], wl_embed_dim=CONFIG["wl_embed_dim"],
        cfg=CONFIG).to(device)
    ema   = ModelEMA(model, CONFIG["ema_decay"])
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : SpectralQuadNet v9  (hsi_training_v7)")
    print(f"Params : {n_par/1e6:.2f}M")
    print(f"Device : {device}")
    print(f"v7 highlights:")
    print(f"  Stage 1 — 3-phase progressive aug, EMA reinit at phase boundaries, "
          f"progressive LS {CONFIG['s1_label_smooth_hi']}→{CONFIG['s1_label_smooth_lo']}")
    print(f"  Stage 2 — Sub-ctr ArcFace K={CONFIG['subcenter_K']}, CDWS, "
          f"SupCon+ProtoNCE, adaptive margins")
    print(f"  Stage 3 — SAM (ρ={CONFIG['s3_sam_rho']}), Greedy SWA, "
          f"{CONFIG['s3_epochs']} ep, balanced+CDWS sampler")
    print(f"  Arch   — Branch A: [sig+d1+d2], Stochastic Branch Dropout p={CONFIG['branch_drop_prob']}")

    # ── Helper: build Stage 1 phase loaders ──────────────────────────
    def _s1_ldr(aug_str):
        ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                             train_idx, aug_strength=aug_str)
        return DataLoader(ds, batch_size=CONFIG["s1_batch"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)

    # ── Stage 1 ──────────────────────────────────────────────────────
    if done_stage < 1:
        print("\n[RUN] Stage 1")
        phase_loaders = {1: _s1_ldr("heavy"),
                         2: _s1_ldr("medium"),
                         3: _s1_ldr("light")}
        _, val_ldr1, _ = build_loaders(train_idx, val_idx, test_idx,
                                       CONFIG["s1_batch"], train_aug="none")
        run_stage1(model, ema, phase_loaders, val_ldr1, device, ckpt_s1)
    else:
        print("\n[SKIP] Stage 1 → loading checkpoint")
        load_ckpt(ckpt_s1, model, ema, device)

    # ── Prepare Stage 2 ──────────────────────────────────────────────
    if done_stage < 2:
        print("\n[INFO] Bootstrapping Sub-center ArcFace from linear head")
        lw = model.linear_head[-1].weight.data.clone()
        model.arcface_head.init_from_linear(lw)
        ema.shadow.arcface_head.init_from_linear(lw)

        # Class difficulty → CDWS + adaptive margins
        _, val_cd, _ = build_loaders(train_idx, val_idx, test_idx, 64)
        class_f1  = compute_class_difficulty(model, ema, val_cd, device)
        cdws_wts  = build_cdws_weights(class_f1, CONFIG["num_classes"],
                                        CONFIG["cdws_max_weight"], CONFIG["cdws_eps"])

        print("\n[RUN] Stage 2")
        tr2,va2,_ = build_loaders(train_idx, val_idx, test_idx, CONFIG["s2_batch"],
                                  balanced=True, all_labels=all_labels,
                                  train_aug="light", class_weights=cdws_wts)
        run_stage2(model, ema, tr2, va2, device, ckpt_s2, class_f1)
    else:
        print("\n[SKIP] Stage 2 → loading checkpoint")
        load_ckpt(ckpt_s2, model, ema, device)

    # ── Stage 3 ──────────────────────────────────────────────────────
    if done_stage < 3:
        print("\n[RUN] Stage 3 (SAM + Greedy SWA)")
        s2c         = torch.load(ckpt_s2, map_location=device, weights_only=False)
        s2_best_val = s2c.get("val_acc", 0.0)

        # Recompute CDWS from Stage 2 model for refined hard-class weights
        _, val_cd3, _ = build_loaders(train_idx, val_idx, test_idx, 64)
        f1_s2     = evaluate_per_class(ema.shadow, val_cd3, device,
                                       CONFIG["num_classes"])
        cdws_s3   = build_cdws_weights(f1_s2, CONFIG["num_classes"],
                                        CONFIG["cdws_max_weight"], CONFIG["cdws_eps"])
        tr3,va3,_ = build_loaders(train_idx, val_idx, test_idx, CONFIG["s2_batch"],
                                  balanced=True, all_labels=all_labels,
                                  train_aug="light", class_weights=cdws_s3)
        run_stage3_swa(model, ema, tr3, va3, device, ckpt_s3,
                       prev_best_val=s2_best_val)
    else:
        print("\n[SKIP] Stage 3 → loading checkpoint")
        load_ckpt(ckpt_s3, model, ema, device)

    # ── Final evaluation ─────────────────────────────────────────────
    print("\n[INFO] Final Evaluation")
    final_ckpt = (ckpt_s3 if stage_exists(3)
                  else ckpt_s2 if stage_exists(2) else ckpt_s1)
    _,_,test_ldr = build_loaders(train_idx, val_idx, test_idx, 64)
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
        ])
    try:
        main()
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)