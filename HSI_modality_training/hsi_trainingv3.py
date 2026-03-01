# code7
from __future__ import annotations

import os
import copy, json as _json, math, os, random, warnings
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

os.environ["NETWORKX_BACKEND"] = "nx-loopback"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", module="networkx")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Online softmax is disabled.*")

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

WL_MIN: float = 385.0
WL_MAX: float = 1000.0

CONFIG: dict = {
    "patches_data":  "./dataset/patches.npy",
    "labels_path":   "./dataset/labels.npy",
    "output_dir":    "./output_v10/",

    "num_bands":     256,
    "num_classes":   90,

    # Stage 1 — progressive augmentation
    "s1_epochs":          300,
    "s1_phase1_frac":     0.40,
    "s1_phase2_frac":     0.30,
    "s1_batch":           128,
    "s1_max_lr":           2e-3,
    "s1_dropout":          0.10,
    "s1_mixup":            0.15,
    "s1_patience":         50,
    "s1_accum":             1,   
    "s1_focal_gamma":       2.0,
    "s1_label_smooth_hi":  0.00,
    "s1_label_smooth_lo":  0.00,
    "s1_ema_reinit_phases": True,

    # Architecture
    "branch_drop_prob":    0.15,
    "subcenter_K":          3,
    "max_cutout_bands":     8,
    "noise_std":            0.02,

    # Stage 2
    "s2_epochs":           120,
    "s2_batch":            128,  
    "s2_head_lr":          1.5e-4,
    "s2_back_lr":          1.5e-5,
    "s2_min_lr":           1e-7,
    "s2_warmup_ep":          5,
    "s2_sgdr_T0":           10,
    "s2_sgdr_Tmult":         2,
    "s2_dropout":           0.10,
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

    # Stage 3
    "s3_epochs":            100,
    "s3_swa_lr":            4e-5,
    "s3_cycle_len":           8,
    "s3_sam_rho":             0.05,
    "s3_greedy":            True,

    # Shared
    "weight_decay":         2e-4,
    "grad_clip":             1.0,
    "ema_decay":            0.999,

    # TTA
    "tta_spatial":             8,
    "tta_spectral":            4,

    # Architecture
    "wl_embed_dim":           16,
    "specf_patch":            16,
    "specf_dim":             256,
    "specf_heads":             8,
    "specf_layers":            4,
    "specf_drop":             0.15,
    "fusion_heads":            4,
    "fusion_drop":            0.10,

    "device":    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":      42,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.cuda.empty_cache()

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True

_GPU_PATCHES: Optional[torch.Tensor] = None
_GLOBAL_LABELS: Optional[np.ndarray] = None

# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def _load_data_to_gpu(patches_path: str, labels_path: str):
    global _GPU_PATCHES, _GLOBAL_LABELS

    if _GPU_PATCHES is not None:
        return

    device = CONFIG["device"]
    assert device.type == "cuda", "GPU mode required."

    print("[DATA] Loading full dataset into GPU VRAM...")

    mmap_arr = np.load(patches_path, mmap_mode="r")
    shape = mmap_arr.shape

    gpu_tensor = torch.empty(shape, dtype=torch.float32, device=device)

    for i in range(0, shape[0], 512):
        block = torch.from_numpy(
            mmap_arr[i:i+512].astype(np.float32)
        ).to(device, non_blocking=True)
        gpu_tensor[i:i+512].copy_(block)

    del mmap_arr
    torch.cuda.synchronize()

    _GPU_PATCHES = gpu_tensor
    _GLOBAL_LABELS = np.load(labels_path)

    print(f"[DATA] ✓ Loaded {_GPU_PATCHES.nelement()*4/1e9:.1f} GB into VRAM")

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
#  EMA
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
        d = self.current_decay
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

    def state_dict(self) -> dict:        return self.shadow.state_dict()
    def load_state_dict(self, sd: dict): self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    _PROFILES = {

        # Phase 1 (structure learning) — strong but not destructive
        "heavy": dict(
            band_drop=0.30,
            cutout=0.20,
            noise=0.15,
            warp=0.10,
            shift=0.10,
            mult=0.10
        ),

        # Phase 2 (robustness shaping)
        "medium": dict(
            band_drop=0.20,
            cutout=0.15,
            noise=0.10,
            warp=0.08,
            shift=0.08,
            mult=0.08
        ),

        # Phase 3 (stability / refinement)
        "light": dict(
            band_drop=0.10,
            cutout=0.08,
            noise=0.05,
            warp=0.05,
            shift=0.05,
            mult=0.05
        ),

        "none": None,
    }

    def __init__(self, indices, aug_strength="none"):
        self.patches          = _GPU_PATCHES
        self.labels           = _GLOBAL_LABELS
        self.indices          = indices
        self.aug_strength     = aug_strength

    def __len__(self): return len(self.indices)

    def _probs(self): return self._PROFILES.get(str(self.aug_strength))

    def _band_dropout(self, x):
        return x * (torch.rand(x.shape[0], device=x.device) > 0.04).float().view(-1,1,1)

    def _band_cutout(self, x):
        x = x.clone(); nb = x.shape[0]
        cut = torch.randint(1, max(2, CONFIG["max_cutout_bands"]), (1,)).item()
        st  = torch.randint(0, max(1, nb - cut), (1,)).item()
        x[st:st+cut] = 0.0; return x

    def _spectral_noise(self, x):
        mask = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        return x + (torch.randn_like(x) * CONFIG["noise_std"]) * mask
    
    def _spectral_warp(self, x):
        C, H, W = x.shape
        scale = 1.0 + random.uniform(-0.10, 0.10); new_C = max(1, int(C * scale))
        if new_C == C: return x
        xp     = x.permute(1,2,0).reshape(-1,1,C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s = (new_C-C)//2; warped = warped[:,:,s:s+C]
        else:
            lo = (C-new_C)//2; warped = F.pad(warped, (lo, C-new_C-lo))
        return warped.reshape(H,W,C).permute(2,0,1)

    def _spectral_shift(self, x): return torch.roll(x, random.randint(-8, 8), dims=0)

    def _mult_noise(self, x):
        mask = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        noise_factor = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device) * 0.05
        return x * noise_factor * mask
    
    def _spatial(self, x):
        if torch.rand(1) < 0.5: x = torch.flip(x, [2])
        if torch.rand(1) < 0.5: x = torch.flip(x, [1])
        return torch.rot90(x, torch.randint(0,4,(1,)).item(), [1,2])

    def __getitem__(self, idx):
        ri = self.indices[idx]
        patch = self.patches[ri].clone()
        label = torch.tensor(int(self.labels[ri]), dtype=torch.long)
        p = self._probs()
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
    """Draws n_cls classes per batch, n_spc samples per class, with optional CDWS weighting."""
    def __init__(self, train_labels, n_cls=16, n_spc=8,
                 class_weights: Optional[Dict[int,float]] = None):
        self.n_cls = n_cls; self.n_spc = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)
        if class_weights is not None:
            raw = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
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
                    rng.choice(pool, self.n_spc, replace=len(pool)<self.n_spc).tolist())
            yield batch

    def __len__(self): return self._n


def build_cdws_weights(class_f1: Dict[int,float], num_classes: int,
                       max_w: float = 3.0, eps: float = 0.05) -> Dict[int,float]:
    raw  = {c: min(1.0/(class_f1.get(c, 0.0)+eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w/mean for c, w in raw.items()}


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def _cutmix(x, y, alpha):
    lam     = float(np.random.beta(alpha, alpha))
    B,C,H,W = x.shape
    idx     = torch.randperm(B, device=x.device)
    r = math.sqrt(1.0 - lam)
    ch, cw  = int(H*r), int(W*r)
    cx, cy  = random.randint(0,W), random.randint(0,H)
    x1=max(cx-cw//2,0); x2=min(cx+cw//2,W)
    y1=max(cy-ch//2,0); y2=min(cy+ch//2,H)
    xm = x.clone(); xm[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    return xm, y, y[idx], 1.0-(x2-x1)*(y2-y1)/(W*H)

def mixed_aug(x, y, alpha=0.4):
    return (_mixup if torch.rand(1)<0.5 else _cutmix)(x, y, alpha)

def mixed_loss(crit, logits, ya, yb, lam):
    return lam*crit(logits, ya) + (1-lam)*crit(logits, yb)


# ══════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss with optional label smoothing.
    Combining both: soft targets from LS, then apply focal modulation (1-pt)^γ.
    This preserves regularisation of LS while sharpening on hard examples.
    """
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C    = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        if self.ls > 0.0:
            with torch.no_grad():
                soft = torch.full_like(logits, self.ls / (C - 1))
                soft.scatter_(1, targets.view(-1,1), 1.0 - self.ls)
            ce = -(soft * logp).sum(1)
        else:
            ce = F.nll_loss(logp, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss. Expects L2-normalised features."""
    def __init__(self, temperature: float = 0.10):
        super().__init__(); self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B   = features.shape[0]
        sim = torch.mm(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        n_pos     = pos_mask.float().sum(1)
        if not (n_pos > 0).any():
            return torch.zeros((), device=features.device, requires_grad=True)
        sim_m    = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim_m - torch.logsumexp(sim_m, dim=1, keepdim=True)
        loss     = -(pos_mask.float() * log_prob.masked_fill(self_mask, 0.0)).sum(1)
        valid    = n_pos > 0
        return (loss[valid] / n_pos[valid]).mean()


class ProtoNCELoss(nn.Module):
    """Class-mean prototype contrastive CE."""
    def __init__(self, temperature: float = 0.10):
        super().__init__(); self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if len(classes) < 2: return (features * 0).sum()
        protos = F.normalize(
            torch.stack([features[labels==c].mean(0) for c in classes]), dim=1)
        sim   = torch.mm(features, protos.T) / self.temperature
        c2l   = {c.item(): i for i,c in enumerate(classes)}
        local = torch.tensor([c2l[y.item()] for y in labels],
                              dtype=torch.long, device=features.device)
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  SAM
# ══════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
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
                if "old_p" in self.state[p]: p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None): raise NotImplementedError("Use first_step/second_step.")

    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        ns  = [p.grad.norm(p=2).to(dev)
               for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(ns), p=2).clamp(min=1e-6) if ns else torch.tensor(0.0)

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups


# ══════════════════════════════════════════════════════════════════════
#  ADAPTIVE SUB-CENTER ARCFACE
# ══════════════════════════════════════════════════════════════════════

class AdaptiveSubcenterArcFaceHead(nn.Module):
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
            self.margins[c] = self.m_base + self.m_delta*(1.0 - min(float(f1), 1.0))
        print(f"[INFO] ArcFace margins  mean={self.margins.mean():.3f}  "
              f"min={self.margins.min():.3f}  max={self.margins.max():.3f}")

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                global_m: Optional[float] = None) -> torch.Tensor:
        x_n    = F.normalize(x, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        cosine = (F.linear(x_n, w_n).clamp(-1+1e-6, 1-1e-6)
                   .view(-1, self.C, self.K).max(dim=2).values)
        if labels is None or not self.training:
            return cosine * self.s
        m_per  = (torch.full((x.shape[0],), global_m, device=x.device)
                  if global_m is not None else self.margins[labels])
        cosm   = torch.cos(m_per); sinm = torch.sin(m_per)
        th     = torch.cos(math.pi - m_per); mm = torch.sin(math.pi - m_per) * m_per
        sine   = torch.sqrt(torch.clamp(1 - cosine**2, min=1e-6))
        tgt_c  = cosine.gather(1, labels.view(-1,1)).squeeze(1)
        tgt_s  = sine.gather(1,   labels.view(-1,1)).squeeze(1)
        phi    = tgt_c*cosm - tgt_s*sinm
        phi    = torch.where(tgt_c > th, phi, tgt_c - mm)
        oh     = torch.zeros_like(cosine).scatter_(1, labels.view(-1,1).long(), 1.0)
        return ((oh*phi.unsqueeze(1)) + ((1-oh)*cosine)) * self.s

    def init_from_linear(self, linear_w: torch.Tensor):
        with torch.no_grad():
            wn = F.normalize(linear_w, dim=1)
            for k in range(self.K):
                noise = torch.randn_like(wn) * 0.01 * k
                self.weight[k::self.K].copy_(wn + noise)
        print(f"[INFO] ArcFace (K={self.K}) bootstrapped from linear head.")


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class SpectralSE(nn.Module):
    """Channel attention using both mean and max pooling (stronger than mean-only)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 16)
        # Input: concat(mean, max) → 2×channels
        self.gate = nn.Sequential(
            nn.Linear(channels*2, mid, bias=False), nn.GELU(),
            nn.Linear(mid, channels, bias=False),   nn.Sigmoid())

    def forward(self, x):
        g = torch.cat([x.mean([2,3]), x.amax([2,3])], dim=1)
        return x * self.gate(g).view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dilation=1):
        super().__init__()
        # Calculate padding to maintain sequence length
        pad = (kernel - 1) * dilation // 2 

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(1, out_ch)

        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + identity)


class CBAM(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(c//r, 8)
        self.ch = nn.Sequential(nn.Conv2d(c,mid,1,bias=False), nn.GELU(),
                                 nn.Conv2d(mid,c,1,bias=False))
        self.sp = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * torch.sigmoid(self.ch(x.mean([2,3],keepdim=True))
                               + self.ch(x.amax([2,3],keepdim=True)))
        return x * self.sp(torch.cat([x.mean(1,keepdim=True),
                                       x.amax(1,keepdim=True)], 1))


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
        return F.gelu(self.n3(self.c3(
            F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) + self.skip(x))

# ══════════════════════════════════════════════════════════════════════
#  SPECTRAL PROFILE BRANCH
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    """
    Advanced Spectral Modeling Branch

    Key Improvements:
    - No hardcoded band count
    - Dynamic positional encoding
    - Optional physical wavelength encoding
    - Learnable derivative filters
    - Cross-scale fusion
    - Attention-based spectral pooling
    - Stable normalization
    """

    def __init__(self, out_dim=256, tower_ch=96, wavelengths=None):
        super().__init__()

        # Optional physical wavelength tensor (1D tensor of size L)
        if wavelengths is not None:
            wl = torch.tensor(wavelengths).float()
            wl = (wl - wl.min()) / (wl.max() - wl.min())
            self.register_buffer("wavelengths", wl)
        else:
            self.wavelengths = None

        # ─────────────────────────────
        # Learnable derivative filters
        # ─────────────────────────────
        self.d1_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.d2_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

        with torch.no_grad():
            self.d1_conv.weight.zero_()
            self.d2_conv.weight.zero_()
            # first derivative init
            self.d1_conv.weight[0, 0, 1] = -1
            self.d1_conv.weight[0, 0, 3] = 1
            # second derivative init
            self.d2_conv.weight[0, 0, 0] = 1
            self.d2_conv.weight[0, 0, 2] = -2
            self.d2_conv.weight[0, 0, 4] = 1

        branch_ch = tower_ch // 3

        def make_proj():
            return nn.Sequential(
                nn.Conv1d(1, branch_ch, 1, bias=False),
                nn.GroupNorm(1, branch_ch),
                nn.GELU()
            )

        self.proj_s  = make_proj()
        self.proj_d1 = make_proj()
        self.proj_d2 = make_proj()

        fused_ch = tower_ch

        # ─────────────────────────────
        # Multi-scale towers
        # ─────────────────────────────
        def make_tower(kernel, dilation=1): # Add dilation
            return nn.Sequential(
                ResBlock1D(fused_ch, fused_ch, kernel, dilation=dilation),
                ResBlock1D(fused_ch, fused_ch, kernel, dilation=dilation)
            )
            
        self.tower_s = make_tower(3, dilation=1)
        self.tower_m = make_tower(5, dilation=2) # Dilated
        self.tower_l = make_tower(5, dilation=4) # Heavily Dilated
        
        # ─────────────────────────────
        # Cross-scale fusion
        # ─────────────────────────────
        self.fusion = nn.Sequential(
            ResBlock1D(fused_ch * 3, fused_ch, 5),
            ResBlock1D(fused_ch, fused_ch, 5)
        )

        # ─────────────────────────────
        # Spectral Attention Pooling
        # ─────────────────────────────
        self.attn_pool = nn.Sequential(
            nn.Conv1d(fused_ch, fused_ch // 4, 1),
            nn.GELU(),
            nn.Conv1d(fused_ch // 4, 1, 1)
        )

        # ─────────────────────────────
        # Output projection
        # ─────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(fused_ch, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self._init_weights()

    # ─────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────
    def add_positional_encoding(self, s):
        """
        Dynamic positional encoding.
        Uses physical wavelength if available,
        otherwise normalized index encoding.
        """
        B, C, L = s.shape

        if self.wavelengths is not None:
            wl = self.wavelengths[:L]
            wl = wl.unsqueeze(0).unsqueeze(0)
            return s + wl

        # fallback: normalized index
        pos = torch.linspace(0, 1, L, device=s.device)
        pos = pos.unsqueeze(0).unsqueeze(0)
        return s + pos

    # ─────────────────────────────
    def spectral_pool(self, x):
        w = torch.softmax(self.attn_pool(x), dim=2)
        return torch.sum(x * w, dim=2)

    # ─────────────────────────────
    def forward(self, ms):
        """
        ms: (B, Bands)
        """
        s = ms.unsqueeze(1)  # (B,1,L)

        # CRITICAL FIX: Do NOT add positional encoding to raw signal here.
        # It ruins the derivative calculations.

        # Learnable derivatives
        d1 = self.d1_conv(s)
        d2 = self.d2_conv(d1)

        # Independent projections
        fs  = self.proj_s(s)
        fd1 = self.proj_d1(d1)
        fd2 = self.proj_d2(d2)

        x = torch.cat([fs, fd1, fd2], dim=1)

        # Add positional encoding safely in the latent space
        x = self.add_positional_encoding(x)

        # Multi-scale extraction
        xs = self.tower_s(x)
        xm = self.tower_m(x)
        xl = self.tower_l(x)

        # Cross-scale fusion
        x_cat = torch.cat([xs, xm, xl], dim=1)
        x_fused = self.fusion(x_cat)

        feat = self.spectral_pool(x_fused)
        return self.proj(feat)    
    
# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS  (mean, std, max across pixels)
# ══════════════════════════════════════════════════════════════════════
class SpectralStatsBranch(nn.Module):
    """
    Enhanced Statistical Spectral Branch (V2)
    
    Now accepts pre-computed, masked 1D statistical vectors to 
    prevent modal collapse and background dilution.
    """

    def __init__(self, num_bands: int, out_dim=256, tower_ch=96):
        super().__init__()

        self.in_channels = 5

        # ─────────────────────────────
        # Statistical Attention
        # ─────────────────────────────
        self.stat_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.in_channels, 16, 1, bias=False), nn.GELU(),
            nn.Conv1d(16, self.in_channels, 1, bias=False), nn.Sigmoid()
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch), nn.GELU()
        )
        
        # ─────────────────────────────
        # Multi-scale towers
        # ─────────────────────────────
        def make_tower(kernel):
            return nn.Sequential(
                ResBlock1D(tower_ch, tower_ch, kernel),
                ResBlock1D(tower_ch, tower_ch, kernel)
            )

        self.tower_s = make_tower(3)
        self.tower_m = make_tower(7)
        self.tower_l = make_tower(15)

        # ─────────────────────────────
        # Cross-scale fusion
        # ─────────────────────────────
        self.fusion = nn.Sequential(
            ResBlock1D(tower_ch * 3, tower_ch, 5),
            ResBlock1D(tower_ch, tower_ch, 5)
        )

        # ─────────────────────────────
        # Attention Pooling
        # ─────────────────────────────
        self.pool_attn = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1, bias=False)
        )

        # ─────────────────────────────
        # Output projection
        # ─────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(tower_ch, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def attention_pool(self, x):
        # Softmax weight over the sequence dimension (dim 2)
        w = torch.softmax(self.pool_attn(x), dim=2)
        return torch.sum(x * w, dim=2)

    
    def forward(self, ms, std, mx, skew, kurt):
        # 1. Shape: (B, 5, 256) -> Now convolutions can actually "see" spectral curves!
        stats = torch.stack([ms, std, mx, skew, kurt], dim=1) 
        stats = stats * self.stat_attn(stats)
        x = self.input_proj(stats)
        
        # 2. Extract multi-scale spectral features
        x_fused = self.fusion(torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1))
        
        # 3. Attention Pool over the 256 bands
        w = torch.softmax(self.pool_attn(x_fused), dim=2)
        pooled = torch.sum(x_fused * w, dim=2)
        return self.proj(pooled)   
     
# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands=256, out_dim=256):
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.GELU())
        self.stages = nn.Sequential(
            ResBlock2D(64,  128,  2), CBAM(128),
            ResBlock2D(128,  192, 2), CBAM(192),
            ResBlock2D(192, 256, 2), CBAM(256),
            ResBlock2D(256, out_dim, 2))
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
        self.ff   = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(d_ff, d), nn.Dropout(drop))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        lx  = self.ln1(x)
        h,_ = self.attn(lx, lx, lx, need_weights=False)
        x   = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))

class SpecFormerBranch(nn.Module):
    def __init__(self, num_bands=256, patch_size=16, stride=8, d_model=128,
                 n_heads=4, n_layers=4, out_dim=256, dropout=0.15):
        super().__init__()
        self.n_patches = (num_bands - patch_size) // stride + 1
        
        # IMPROVEMENT: Use 1D Conv for Overlapping spectral patches
        self.patch_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride, bias=False),
            nn.GroupNorm(1, d_model), nn.GELU()
        )
        
        # Dynamic PE based on patch count
        pe = torch.zeros(self.n_patches, d_model)
        wl_n = torch.linspace(0, 1, self.n_patches)
        half = d_model // 2
        freq = torch.exp(torch.arange(half).float() * -(math.log(1e4) / max(half - 1, 1)))
        pe[:, :half], pe[:, half:] = torch.sin(wl_n.unsqueeze(1) * freq), torch.cos(wl_n.unsqueeze(1) * freq)
        self.register_buffer("wl_pe", pe)
        
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([_PreLNBlock(d_model, n_heads, d_model * 2, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, out_dim), nn.BatchNorm1d(out_dim), nn.GELU())

    def forward(self, ms):
        x = self.patch_proj(ms.unsqueeze(1)).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x + self.wl_pe], dim=1)
        for blk in self.blocks: x = blk(x)
        return self.proj(self.norm(x)[:, 0])

# ══════════════════════════════════════════════════════════════════════
#  BRANCH CROSS-ATTENTION FUSION
# ══════════════════════════════════════════════════════════════════════
class UnifiedModalFusion(nn.Module):
    def __init__(self, d=256, heads=4, drop=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)

        self.ff   = nn.Sequential(
            nn.Linear(d, d*2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d*2, d),
            nn.Dropout(drop)
        )

    def forward(self, branches):
        B = branches[0].shape[0]
        # Stack all 4 branches equally
        tokens = torch.stack(branches, dim=1)  # (B, 4, 256)
        
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B, 5, 256)

        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + h
        x = x + self.ff(self.ln2(x))

        return x[:, 0]

# ══════════════════════════════════════════════════════════════════════
#  SPECTRAL STATISTICS HELPER
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_branch_influence(model, loader, device, max_batches=5):
    model.eval()

    influences = torch.zeros(4, device=device)
    total = 0

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device, non_blocking=True)

        logits_full = model(x)
        p_full = torch.softmax(logits_full, dim=1)

        for b in range(4):
            mask = torch.ones(4, device=device)
            mask[b] = 0.0

            logits_ablate = model(x, branch_mask=mask)
            p_ablate = torch.softmax(logits_ablate, dim=1)

            delta = F.kl_div(
                p_ablate.log(),
                p_full,
                reduction="batchmean"
            )

            influences[b] += delta

        total += 1

    if total == 0:
        return {"A":0, "B":0, "C":0, "D":0}

    influences /= total
    total_inf = influences.sum().clamp(min=1e-8)
    influences = influences / total_inf * 100.0

    return {
        "A": float(influences[0]),
        "B": float(influences[1]),
        "C": float(influences[2]),
        "D": float(influences[3]),
    }
    
def masked_spectral_stats(x: torch.Tensor):
    x32  = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H*W)
    
    # 1. Identify valid seed pixels
    mask = (flat.abs().sum(1, keepdim=True) > 1e-5).float()
    cnt  = mask.sum(2).clamp(min=1.0)
    
    # 2. Mean & Standard Deviation
    mean = (flat * mask).sum(2) / cnt
    centered = (flat - mean.unsqueeze(2)) * mask
    var = (centered ** 2).sum(2) / cnt
    std = torch.sqrt(var + 1e-5)
    
    # 3. Max
    mx = flat.masked_fill(mask.expand_as(flat)==0, -1e4).max(2).values
    mx = mx.masked_fill(mx < -9999.0, 0.0)
    
    # 4. Skewness & Kurtosis (Safe computation)
    m3 = (centered ** 3).sum(2) / cnt
    m4 = (centered ** 4).sum(2) / cnt
    
    skew = torch.clamp(m3 / (std ** 3 + 1e-4), -10.0, 10.0)
    kurt = torch.clamp(m4 / (std ** 4 + 1e-4), 0.0, 20.0)
    
    return (torch.nan_to_num(mean, 0), torch.nan_to_num(std, 0),
            torch.nan_to_num(mx, 0), torch.nan_to_num(skew, 0),
            torch.nan_to_num(kurt, 0))
        
# ══════════════════════════════════════════════════════════════════════
#  SPECTRALQUADNET
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    def __init__(self, num_classes=90, num_bands=256, dropout=0.30,
                 wl_embed_dim=16, cfg=None):
        super().__init__()
        cfg = cfg or CONFIG
        self.branch_drop_prob = cfg.get("branch_drop_prob", 0.0)

        self.se       = SpectralSE(num_bands, 16)
        self.branch_a = SpectralProfileBranch(out_dim=256, tower_ch=96)
        self.branch_b = SpectralStatsBranch(num_bands=num_bands, out_dim=256, tower_ch=96)
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(
            num_bands=num_bands,
            patch_size=cfg["specf_patch"], # 16
            stride=cfg["specf_patch"] // 2, # 8 (Overlapping)
            d_model=cfg["specf_dim"],       # 256
            n_heads=cfg["specf_heads"],     # 8
            n_layers=cfg["specf_layers"],   # 4
            out_dim=256,
            dropout=cfg["specf_drop"]
        )
        
        self.unified_fusion = UnifiedModalFusion(d=256, heads=cfg["fusion_heads"], drop=cfg["fusion_drop"])        
        
        self.aux_head_a = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, num_classes)
        )
        self.aux_head_b = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, num_classes)
        )
        
        self.embed_net  = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,  256), nn.LayerNorm(256))
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
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which: str):
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_embed: bool = False,
                arc_m: Optional[float] = None,
                branch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.se(x)
        ms, std, mx, skew, kurt = masked_spectral_stats(x)

        ba = self.branch_a(ms)                       
        bb = self.branch_b(ms, std, mx, skew, kurt)  
        bc = self.branch_c(x)                        
        bd = self.branch_d(ms)                       
                
        if branch_mask is not None:
            ba, bb, bc, bd = ba*branch_mask[0], bb*branch_mask[1], bc*branch_mask[2], bd*branch_mask[3]
            
        elif self.training:
            drop_probs = torch.tensor([0.05, 0.05, 0.40, 0.15], device=ba.device)
            keeps = (torch.rand(4, device=ba.device) > drop_probs).float()
            if keeps.sum() == 0: keeps[torch.randint(0, 4, ())] = 1.0 
            ba, bb, bc, bd = ba*keeps[0], bb*keeps[1], bc*keeps[2], bd*keeps[3]

        aux_a = self.aux_head_a(ba) if self.training else None
        aux_b = self.aux_head_b(bb) if self.training else None

        joint_token = self.unified_fusion([ba, bb, bc, bd])
        emb = self.embed_net(joint_token)

        if self._use_arcface:
            logits = self.arcface_head(F.normalize(emb, dim=1), labels, global_m=arc_m)
        else:
            logits = self.linear_head(emb)

        if self.training and not return_embed:
            return {"main": logits, "aux_a": aux_a, "aux_b": aux_b}
        
        if return_embed:
            return logits, F.normalize(emb, dim=1)
        return logits
    
# ══════════════════════════════════════════════════════════════════════
#  TTA — 8 spatial + 4 spectral
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor,
                n_spatial: int = 8, n_spectral: int = 4) -> torch.Tensor:
    device = x.device; logits = []
    for k, flip in [(k,f) for k in range(4) for f in (False,True)][:n_spatial]:
        aug = torch.rot90(x, k, [2,3])
        if flip: aug = torch.flip(aug, [3])
        with autocast(device_type=device.type): logits.append(model(aug))
    step   = max(256//(max(n_spectral,1)*2), 1)
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
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va, te  = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    return labels, tr, va, te


def build_loaders(train_idx, val_idx, test_idx, batch_train,
                  balanced=False, all_labels=None,
                  train_aug="none",
                  class_weights=None):

    ds = RiceSeedDataset(train_idx, aug_strength=train_aug)

    if balanced and all_labels is not None:
        samp = ClassBalancedBatchSampler(
            all_labels[train_idx],
            CONFIG["bal_n_cls"],
            CONFIG["bal_n_spc"],
            class_weights=class_weights
        )
        tr_ldr = DataLoader(ds, batch_sampler=samp, num_workers=0)
    else:
        tr_ldr = DataLoader(ds, batch_size=batch_train,
                            shuffle=True, drop_last=True,
                            num_workers=0)

    va_ldr = DataLoader(
        RiceSeedDataset(val_idx),
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    te_ldr = DataLoader(
        RiceSeedDataset(test_idx),
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    return tr_ldr, va_ldr, te_ldr

# ══════════════════════════════════════════════════════════════════════
#  OPTIMISERS & SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def _wd_groups(named_params, lr):
    wd, no_wd = [], []
    for n, p in named_params:
        if not p.requires_grad: continue
        (no_wd if (p.ndim == 1 or n.endswith(".bias")) else wd).append(p)
    return [{"params": wd,    "lr": lr, "weight_decay": CONFIG["weight_decay"]},
            {"params": no_wd, "lr": lr, "weight_decay": 0.0}]

def build_optimizer_s1(model, lr):
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))

def build_optimizer_s2(model, head_lr, back_lr):
    hp, bp = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (hp if n.startswith("arcface_head") else bp).append((n, p))
    return optim.AdamW(_wd_groups(hp, head_lr) + _wd_groups(bp, back_lr))

def build_optimizer_s3(model, lr):
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))

def sgdr_scheduler(optimizer, warmup_ep=5, T_0=10, T_mult=2,
                   eta_min_frac=1e-3) -> optim.lr_scheduler.LambdaLR:
    def _l(ep):
        if ep < warmup_ep:
            return max(ep / max(warmup_ep, 1), 1e-6)
        t = ep - warmup_ep; clen = T_0; elapsed = 0
        while t >= elapsed + clen:
            elapsed += clen; clen = max(int(clen*T_mult), 1)
        ratio = (t - elapsed) / max(clen, 1)
        return eta_min_frac + 0.5*(1 - eta_min_frac)*(1 + math.cos(math.pi*ratio))
    return optim.lr_scheduler.LambdaLR(optimizer, _l)

def arcface_margin(ep, m0, m_target, warmup_ep):
    if ep >= warmup_ep: return m_target
    return m0 + (m_target - m0)*0.5*(1 - math.cos(math.pi*ep/max(warmup_ep,1)))


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

    if model._use_arcface and use_mixup:
        raise ValueError("Mixup cannot be used with ArcFace.")

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        x_in, ya, yb, lam = mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)

        with autocast(device_type=device.type, enabled=use_amp):
            if supcon is not None:
                out = model(x_in, ya, return_embed=True, arc_m=arc_m)
                
                if isinstance(out, dict):
                    logits, emb = out["main"], out["emb"]
                    l_aux_a = criterion(out["aux_a"], ya)
                    l_aux_b = criterion(out["aux_b"], ya)
                    aux_loss = 0.3 * (l_aux_a + l_aux_b)
                else:
                    logits, emb = out
                    aux_loss = 0.0

                cls_l  = criterion(logits, ya)
                sc_l   = supcon(emb, ya)
                pt_l   = proto(emb, ya) if proto is not None else 0.0
                
                loss = ((1 - supcon_weight - proto_weight) * cls_l 
                        + supcon_weight * sc_l 
                        + proto_weight * pt_l 
                        + aux_loss)
            else:
                arc_labels = (ya if model._use_arcface and not use_mixup else None)
                out = model(x_in, labels=arc_labels, arc_m=arc_m)

                if isinstance(out, dict):
                    # Apply mixed_loss to all heads to handle Mixup/Cutmix correctly
                    l_main  = mixed_loss(criterion, out["main"], ya, yb, lam)
                    l_aux_a = mixed_loss(criterion, out["aux_a"], ya, yb, lam)
                    l_aux_b = mixed_loss(criterion, out["aux_b"], ya, yb, lam)
                    
                    # Weighting: 1.0 for main head, 0.3 for each auxiliary spectral head
                    loss   = l_main + 0.3 * (l_aux_a + l_aux_b)
                    logits = out["main"]
                else:
                    logits = out
                    loss   = mixed_loss(criterion, logits, ya, yb, lam)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == ya).float().mean().item()

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n

def train_one_epoch_sam(model, loader, sam_opt, criterion, device,
                        supcon=None, supcon_weight=0.0,
                        proto=None, proto_weight=0.0, arc_m=None):
    torch.set_default_dtype(torch.float32)
    model.train()
    total_loss = total_acc = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        def _compute_loss(lg, em):
            if supcon is not None:
                return ((1-supcon_weight-proto_weight)*criterion(lg, y)
                        + supcon_weight*(supcon(em, y) if supcon else 0)
                        + proto_weight*(proto(em, y)   if proto  else 0))
            return criterion(lg, y)

        sam_opt.zero_grad()
        if supcon is not None:
            logits, emb = model(x, y, return_embed=True, arc_m=arc_m)
        else:
            logits = model(x, labels=y, arc_m=arc_m); emb = None
        loss = _compute_loss(logits, emb)
        if not torch.isfinite(loss): continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.first_step(zero_grad=True)

        if supcon is not None:
            logits2, emb2 = model(x, y, return_embed=True, arc_m=arc_m)
        else:
            logits2 = model(x, labels=y, arc_m=arc_m); emb2 = None
        loss2 = _compute_loss(logits2, emb2)
        if not torch.isfinite(loss2):
            sam_opt.zero_grad()
            for g in sam_opt.param_groups:
                for p in g["params"]:
                    if "old_p" in sam_opt.state.get(p, {}):
                        p.data = sam_opt.state[p]["old_p"]
            continue
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.second_step(zero_grad=True)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.detach().argmax(1) == y).float().mean().item()

    n = max(len(loader), 1)
    return total_loss/n, total_acc/n


@torch.no_grad()
def _run_eval(model, loader, device):
    """Shared evaluation core — returns (preds, targets) numpy arrays."""
    model.eval(); preds, targets = [], []
    with autocast(device_type=device.type, enabled=False):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            preds.append(logits.argmax(1).cpu()); targets.append(y.cpu())
    if device.type == "cuda": torch.cuda.synchronize()
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate(model, loader, device) -> Tuple[float, float]:
    """Returns (macro_f1, accuracy)."""
    p, t = _run_eval(model, loader, device)
    return (f1_score(t, p, average="macro", zero_division=0),
            accuracy_score(t, p))


def evaluate_per_class(model, loader, device, num_classes) -> Dict[int, float]:
    p, t = _run_eval(model, loader, device)
    f1_arr = f1_score(t, p, average=None, zero_division=0, labels=list(range(num_classes)))
    return {i: float(v) for i, v in enumerate(f1_arr)}


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def stage_ckpt_path(s: int) -> str:
    return os.path.join(CONFIG["output_dir"], f"best_stage{s}.pth")

def stage_meta_path(s: int) -> str:
    return os.path.join(CONFIG["output_dir"], f"stage{s}_meta.json")

def stage_exists(s: int) -> bool:
    return (os.path.isfile(stage_ckpt_path(s)) and
            os.path.isfile(stage_meta_path(s)))

def latest_completed_stage() -> int:
    for s in (3, 2, 1):
        if stage_exists(s): return s
    return 0

def save_ckpt(path: str, epoch: int, stage: str,
              model: nn.Module, ema: ModelEMA,
              val_f1: float, val_acc: float, **metadata) -> None:
    bundle = {
        "epoch": epoch, "stage": stage,
        "model": model.state_dict(), "ema": ema.state_dict(),
        "val_f1": val_f1, "val_acc": val_acc,
        "use_arcface": model._use_arcface, **metadata,
    }
    torch.save(bundle, path)
    sidecar = {k: v for k, v in bundle.items()
               if k not in ("model", "ema") and _is_json_serialisable(v)}
    sn = int(stage.split()[-1]) if stage.split()[-1].isdigit() else 0
    with open(stage_meta_path(sn), "w") as f:
        _json.dump(sidecar, f, indent=2)

def _is_json_serialisable(v) -> bool:
    try: _json.dumps(v); return True
    except (TypeError, ValueError): return False

def load_stage_meta(s: int) -> dict:
    p = stage_meta_path(s)
    if not os.path.isfile(p): return {}
    with open(p) as f: raw = _json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            try: out[k] = {int(kk): vv for kk, vv in v.items()}; continue
            except (ValueError, TypeError): pass
        out[k] = v
    return out

def load_ckpt(path: str, model: nn.Module, ema: ModelEMA, device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    flag = ckpt.get("use_arcface", False)
    model.use_arcface(flag); ema.shadow.use_arcface(flag)
    return ckpt

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
#  CLASS DIFFICULTY
# ══════════════════════════════════════════════════════════════════════

def compute_class_difficulty(ema_shadow: nn.Module,
                             val_ldr,
                             device,
                             label: str = "Stage") -> Tuple[Dict[int,float], Dict[int,float]]:

    class_f1 = evaluate_per_class(
        ema_shadow, val_ldr, device, CONFIG["num_classes"])

    cdws_wts = build_cdws_weights(
        class_f1,
        CONFIG["num_classes"],
        CONFIG["cdws_max_weight"],
        CONFIG["cdws_eps"]
    )

    macro = float(np.mean(list(class_f1.values())))
    n_hard = sum(1 for f in class_f1.values() if f < 0.50)

    branch_inf = compute_branch_influence(
        ema_shadow,
        val_ldr,
        device,
        max_batches=3
    )

    print(
        f"[INFO] {label} class difficulty — macro F1={macro:.3f}  "
        f"hard classes (<0.50 F1): {n_hard}/{CONFIG['num_classes']}  |  "
        f"Branch influence % → "
        f"A:{branch_inf['A']:.1f}  "
        f"B:{branch_inf['B']:.1f}  "
        f"C:{branch_inf['C']:.1f}  "
        f"D:{branch_inf['D']:.1f}"
    )

    return class_f1, cdws_wts

# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — 3-PHASE PROGRESSIVE AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def run_stage1(model, ema, loaders_by_phase, val_ldr, device, best_ckpt: str) -> float:
    """
    Phase 1 (0–40%):  heavy aug + mixup + high label-smooth → explore
    Phase 2 (40–70%): medium aug + mixup + decaying label-smooth → consolidate
    Phase 3 (70–100%): light aug + NO mixup + Focal+LS → discriminate hard classes

    Saves and does early stopping on macro-F1 (not accuracy).
    """
    model.use_arcface(False)
    model.unfreeze_head("linear"); model.freeze_head("arcface")

    ep_total = CONFIG["s1_epochs"]
    p1_end   = int(ep_total * CONFIG["s1_phase1_frac"])
    p2_end   = int(ep_total * (CONFIG["s1_phase1_frac"] + CONFIG["s1_phase2_frac"]))

    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"] / 25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=ep_total,
            eta_min=CONFIG["s1_max_lr"] * 1e-3
        )
    scaler       = GradScaler()
    ls_hi        = CONFIG["s1_label_smooth_hi"]
    ls_lo        = CONFIG["s1_label_smooth_lo"]
    best_f1      = 0.0
    no_improve   = 0
    ema_reinited = [False, False]

    w = 66
    print(f"\n{'═'*w}\n  Stage 1 — 3-Phase Progressive Augmentation  [{ep_total} epochs max]\n{'═'*w}")
    print(f"  Phase 1: ep 1–{p1_end}    heavy aug + mixup")
    print(f"  Phase 2: ep {p1_end+1}–{p2_end}  medium aug + mixup")
    print(f"  Phase 3: ep {p2_end+1}–{ep_total}  light aug, NO mixup, Focal+LS")
    print(f"  Label smooth: {ls_hi} → {ls_lo}  |  Primary metric: macro-F1")

    for ep in range(1, ep_total + 1):
        if   ep <= p1_end: phase=1; cur_ldr=loaders_by_phase[1]; use_mx=True
        elif ep <= p2_end: phase=2; cur_ldr=loaders_by_phase[2]; use_mx=True
        else:              phase=3; cur_ldr=loaders_by_phase[3]; use_mx=False

        if phase==2 and not ema_reinited[0] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 2 (ep {ep})")
            ema_reinited[0] = True
        if phase==3 and not ema_reinited[1] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 3 (ep {ep})")
            ema_reinited[1] = True

        t      = (ep - 1) / max(ep_total - 1, 1)
        ls_now = ls_hi*(1 - t) + ls_lo*t

        # Phase 3: Focal loss with label smoothing — keeps regularisation while
        # sharpening focus on hard examples. Phases 1-2 use standard CE+LS.
        if phase == 3:
            crit = FocalLoss(gamma=CONFIG["s1_focal_gamma"], label_smoothing=ls_now)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=ls_now)

        tl, ta = train_one_epoch(
            model, cur_ldr, optimizer, crit, scaler, ema, device,
            scheduler=scheduler, use_mixup=use_mx,
            mixup_alpha=CONFIG["s1_mixup"], accum_steps=CONFIG["s1_accum"])

        # Evaluate both F1 and accuracy for live model and EMA
        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1  = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        lr_now      = optimizer.param_groups[0]["lr"]
        saved       = ""
        
        scheduler.step()

        # Save and track patience on F1 (primary metric)
        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1, _cdws = compute_class_difficulty(ema.shadow, val_ldr, device, "S1")
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema,
                      val_f1=best_ep_f1, val_acc=best_ep_acc,
                      class_f1=_cf1, cdws_weights=_cdws, arcface_init_done=False)
            saved = "  ✓"
        else:
            no_improve += 1

        print(f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
              f"F1 {f1_live:.3f}/{f1_ema:.3f}  Acc {acc_live:.1%}/{acc_ema:.1%} │ "
              f"LR {lr_now:.2e}  LS {ls_now:.3f} [P{phase}]{saved}")

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("arcface")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR
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
        optimizer, warmup_ep=CONFIG["s2_warmup_ep"],
        T_0=CONFIG["s2_sgdr_T0"], T_mult=CONFIG["s2_sgdr_Tmult"],
        eta_min_frac=CONFIG["s2_min_lr"]/CONFIG["s2_head_lr"])

    sc_w = CONFIG["supcon_weight"]; pt_w = CONFIG["proto_weight"]
    ep_total = CONFIG["s2_epochs"]
    best_f1  = 0.0; no_improve = 0

    r1 = CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"]
    r2 = r1 + CONFIG["s2_sgdr_T0"] * CONFIG["s2_sgdr_Tmult"]

    w = 66
    print(f"\n{'═'*w}\n  Stage 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR  [{ep_total} ep]\n{'═'*w}")
    print(f"  hLR={CONFIG['s2_head_lr']:.1e}  bLR={CONFIG['s2_back_lr']:.1e}  "
          f"SGDR T0={CONFIG['s2_sgdr_T0']} Tmult={CONFIG['s2_sgdr_Tmult']} "
          f"→ restarts ep {r1} & {r2}")
    print(f"  ArcFace K={CONFIG['subcenter_K']}  "
          f"m={CONFIG['s2_arcface_m0']}→{CONFIG['s2_arcface_m']}+Δ{CONFIG['s2_arcface_m_delta']}")
    print(f"  Losses: Focal(γ={CONFIG['s2_focal_gamma']}) + SupCon(w={sc_w}) + ProtoNCE(w={pt_w})")
    print(f"  Batch: {CONFIG['bal_n_cls']} cls × {CONFIG['bal_n_spc']} spc = "
          f"{CONFIG['bal_n_cls']*CONFIG['bal_n_spc']} | Primary metric: macro-F1")

    for ep in range(1, ep_total + 1):
        warmup_done = (ep - 1) >= CONFIG["s2_margin_warmup_ep"]
        m_now  = (CONFIG["s2_arcface_m"] if warmup_done
                  else arcface_margin(ep-1, CONFIG["s2_arcface_m0"],
                                      CONFIG["s2_arcface_m"],
                                      CONFIG["s2_margin_warmup_ep"]))
        arc_m  = None if warmup_done else m_now
        ramp   = min(1.0, ep / 10.0)
        sc_now = sc_w * ramp; pt_now = pt_w * ramp

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, focal, scaler=None, ema=ema,
            device=device, scheduler=None,
            use_mixup=False, supcon=supcon, supcon_weight=sc_now,
            proto=proto, proto_weight=pt_now, arc_m=arc_m)
        scheduler.step()

        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1  = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        head_lr     = optimizer.param_groups[0]["lr"]
        back_lr     = optimizer.param_groups[2]["lr"]
        saved       = ""

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1_s2, _cdws_s2 = compute_class_difficulty(ema.shadow, val_ldr, device, "S2")
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema,
                      val_f1=best_ep_f1, val_acc=best_ep_acc,
                      class_f1=_cf1_s2, cdws_weights=_cdws_s2, s2_val_f1=best_ep_f1)
            saved = "  ✓"
        else:
            no_improve += 1

        rf = " ↻R1" if ep==r1 else (" ↻R2" if ep==r2 else "")
        print(f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
              f"F1 {f1_live:.3f}/{f1_ema:.3f}  Acc {acc_live:.1%}/{acc_ema:.1%} │ "
              f"hLR {head_lr:.1e} bLR {back_lr:.1e}  m={m_now:.3f}{saved}{rf}")

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("linear")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 — SAM + GREEDY SWA
# ══════════════════════════════════════════════════════════════════════

def run_stage3_swa(model, ema, train_ldr, val_ldr, device,
                   best_ckpt: str, prev_best_f1: float) -> float:
    if hasattr(torch, "_dynamo"): torch._dynamo.disable()

    model.set_dropout(CONFIG["s2_dropout"])
    model.branch_drop_prob = 0.0
    ema.shadow.branch_drop_prob = 0.0
    model.use_arcface(True); ema.shadow.use_arcface(True)

    params = list(_wd_groups(model.named_parameters(), CONFIG["s3_swa_lr"]))
    sam    = SAM(params, optim.AdamW, rho=CONFIG["s3_sam_rho"],
                 lr=CONFIG["s3_swa_lr"], weight_decay=CONFIG["weight_decay"])

    focal_s3  = FocalLoss(gamma=1.0)
    supcon_s3 = SupConLoss(temperature=0.10)
    proto_s3  = ProtoNCELoss(temperature=0.10)

    swa_state = None
    n_snap = 0; n_rejected = 0; best_live_f1 = 0.0

    w = 66
    print(f"\n{'═'*w}\n  Stage 3 — SAM + Greedy SWA  [{CONFIG['s3_epochs']} epochs]\n{'═'*w}")
    print(f"  SAM ρ={CONFIG['s3_sam_rho']}  Cycle={CONFIG['s3_cycle_len']} ep  "
          f"Peak LR={CONFIG['s3_swa_lr']:.0e}")

    def _s3_margin(ep):
        return 0.25 + 0.05*math.cos(math.pi*ep/CONFIG["s3_epochs"])

    for ep in range(1, CONFIG["s3_epochs"] + 1):
        cycle_ep = (ep - 1) % CONFIG["s3_cycle_len"]
        lr_now   = CONFIG["s3_swa_lr"] * (0.3 + 0.7*0.5*(
            1 + math.cos(math.pi*cycle_ep/CONFIG["s3_cycle_len"])))
        for pg in sam.param_groups: pg["lr"] = lr_now

        tl, ta = train_one_epoch_sam(
            model, train_ldr, sam, focal_s3, device,
            supcon=supcon_s3, supcon_weight=0.02,
            proto=proto_s3,   proto_weight=0.01,
            arc_m=_s3_margin(ep))

        f1_live, acc_live = evaluate(model, val_ldr, device)
        best_live_f1 = max(best_live_f1, f1_live)

        snap_info = ""
        if ep % CONFIG["s3_cycle_len"] == 0:
            if not CONFIG["s3_greedy"] or f1_live >= best_live_f1 * 0.98:
                n_snap += 1
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = copy.deepcopy(sd)
                else:
                    beta = 1.0 / float(n_snap)
                    for k in swa_state:
                        if swa_state[k].is_floating_point():
                            swa_state[k].mul_(1.0 - beta).add_(sd[k], alpha=beta)
                        else:
                            swa_state[k].copy_(sd[k])
                snap_info = f"  ★ snap {n_snap}"
            else:
                n_rejected += 1
                snap_info  = f"  ✗ rejected (F1 {f1_live:.3f} < {best_live_f1*0.98:.3f})"

        print(f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
              f"F1 {f1_live:.3f}  Acc {acc_live:.1%} │ LR {lr_now:.2e}{snap_info}")

    print(f"\nUpdating BN stats ({n_snap} accepted, {n_rejected} rejected) ...")
    if swa_state is None:
        print("[WARN] No snapshots accepted — using final live model.")
        swa_state = copy.deepcopy(model.state_dict())

    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state); swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)
    f1_swa, acc_swa = evaluate(swa_model, val_ldr, device)
    print(f"SWA val: F1={f1_swa:.3f}  Acc={acc_swa:.1%}")

    ema.shadow.load_state_dict(swa_model.state_dict())
    ema.shadow.use_arcface(True)

    note = ""
    if f1_swa <= prev_best_f1:
        note = "val_f1 did not beat Stage 2; Stage 2 ckpt preferred for eval"
        print(f"Stage 3 F1 {f1_swa:.3f} ≤ Stage 2 best {prev_best_f1:.3f} — Stage 2 preferred.")
    else:
        print(f"Stage 3 F1 {f1_swa:.3f} > Stage 2 best {prev_best_f1:.3f} → saving.")

    save_ckpt(best_ckpt, CONFIG["s3_epochs"], "Stage 3",
              swa_model, ema, val_f1=f1_swa, val_acc=acc_swa,
              swa_n_snapshots=n_snap, swa_n_rejected=n_rejected,
              **({"note": note} if note else {}))
    return f1_swa


# ══════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt):
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")
    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow; eval_model.eval()

    print(f"  ArcFace: {eval_model._use_arcface}  |  "
          f"Checkpoint: ep {ckpt['epoch']} | {ckpt['stage']} | "
          f"F1={ckpt.get('val_f1',0):.3f}  Acc={ckpt.get('val_acc',0):.1%}")
    print(f"  TTA: {CONFIG['tta_spatial']} spatial + {CONFIG['tta_spectral']} spectral "
          f"= {CONFIG['tta_spatial']+CONFIG['tta_spectral']} total views")

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x = x.to(device, non_blocking=True)
            logits = (tta_predict(eval_model, x, CONFIG["tta_spatial"], CONFIG["tta_spectral"])
                      if use_tta else eval_model(x))
            preds.append(logits.argmax(1).cpu()); targets.append(y)
        p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p, t)
        print(f"\n  [{tag}]  F1(macro)={f1_score(t,p,average='macro',zero_division=0):.4f}  "
              f"F1(wt)={f1_score(t,p,average='weighted',zero_division=0):.4f}  "
              f"Acc={accuracy_score(t,p):.1%}")

    p_tta, t_tta = results["TTA   "]
    print(f"\nClassification Report (TTA):\n")
    print(classification_report(t_tta, p_tta, zero_division=0))

    out = CONFIG["output_dir"]
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy",   p_tta)
    np.save(f"{out}/test_targets.npy",     t_tta)
    print(f"\nOutputs saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  BEST CHECKPOINT SELECTION
# ══════════════════════════════════════════════════════════════════════

def _pick_best_checkpoint(*ckpt_paths: str) -> str:
    """Select checkpoint with highest val_f1 (primary) across all stages."""
    best_val, best_path = -1.0, ckpt_paths[-1]
    for p in ckpt_paths:
        if not os.path.isfile(p): continue
        try:
            sn   = int(os.path.basename(p).replace("best_stage","").replace(".pth",""))
            meta = load_stage_meta(sn)
            # Prefer val_f1; fall back to val_acc for older checkpoints
            v    = meta.get("val_f1", meta.get("val_acc", None))
        except (ValueError, KeyError):
            v = None
        if v is None:
            try:
                v = torch.load(p, map_location="cpu", weights_only=False).get("val_f1", 0.0)
            except Exception:
                v = 0.0
        if v > best_val:
            best_val, best_path = v, p
    return best_path


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device     = CONFIG["device"]
    ckpt_s1    = stage_ckpt_path(1)
    ckpt_s2    = stage_ckpt_path(2)
    ckpt_s3    = stage_ckpt_path(3)
    done_stage = latest_completed_stage()

    labels = {0:"starting fresh", 1:"Stage 1 done", 2:"Stages 1–2 done", 3:"all done"}
    print(f"\n{'─'*66}")
    print(f"  Auto-resume: {labels.get(done_stage, f'stage {done_stage} done')}")
    print(f"  Output dir : {CONFIG['output_dir']}")
    print(f"{'─'*66}")
    print(f"[INFO] Latest completed stage: {done_stage}")

    _load_data_to_gpu(CONFIG["patches_data"], CONFIG["labels_path"])

    free = torch.cuda.mem_get_info(device)[0] / 1e9
    print(f"[DATA] ✓ GPU mode: {_GPU_PATCHES.nelement()*4/1e9:.1f} GB in VRAM  "
          f"| {free:.1f} GB free | num_workers=0")

    all_labels, train_idx, val_idx, test_idx = build_splits()
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx)//CONFIG['num_classes']}")

    model = SpectralQuadNet(
        num_classes=CONFIG["num_classes"],
        num_bands=CONFIG["num_bands"],
        dropout=CONFIG["s1_dropout"],
        wl_embed_dim=CONFIG["wl_embed_dim"],
        cfg=CONFIG).to(device)

    ema = ModelEMA(model, decay=CONFIG["ema_decay"])

    print(f"Model  : SpectralQuadNet v10")
    print(f"Params : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Device : {device}")

    if hasattr(torch, "compile"):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.recompile_limit        = 64
        warnings.filterwarnings("ignore", message=".*networkx backend.*")
        print("[INFO] Applying torch.compile(mode='default') ...")
        model      = torch.compile(model,      mode="default", fullgraph=False)
        ema.shadow = torch.compile(ema.shadow, mode="default", fullgraph=False)
    else:
        print("[WARN] torch.compile unavailable (PyTorch < 2.0)")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU]  {props.name}  |  VRAM {props.total_memory//1024**3} GB  |  "
              f"TF32={torch.backends.cuda.matmul.allow_tf32}")

    def _s1_ldr(aug_str):
        ds = RiceSeedDataset(train_idx, aug_strength=aug_str)
        return DataLoader(
            ds,
            batch_size=CONFIG["s1_batch"],
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
    if done_stage < 1:
        print("\n[RUN] Stage 1")
        phase_loaders = {1: _s1_ldr("heavy"), 2: _s1_ldr("medium"), 3: _s1_ldr("light")}
        _, val_ldr1, _ = build_loaders(train_idx, val_idx, test_idx,
                                       CONFIG["s1_batch"], train_aug="none")
        run_stage1(model, ema, phase_loaders, val_ldr1, device, ckpt_s1)
        print("[INFO] Reloading best Stage 1 checkpoint ...")
        load_ckpt(ckpt_s1, model, ema, device)
    else:
        print("\n[SKIP] Stage 1 → loading checkpoint")
        load_ckpt(ckpt_s1, model, ema, device)

    meta_s1      = load_stage_meta(1)
    class_f1_s1  = meta_s1.get("class_f1",    {})
    cdws_wts_s1  = meta_s1.get("cdws_weights", {})
    arcface_done = meta_s1.get("arcface_init_done", False)
    s1_best_f1   = meta_s1.get("val_f1", meta_s1.get("val_acc", 0.0))
    print(f"[INFO] Stage 1 → F1={s1_best_f1:.3f}  "
          f"hard classes={sum(1 for f in class_f1_s1.values() if f<0.5)}")

    if done_stage < 2:
        if not arcface_done:
            print("\n[INFO] Bootstrapping ArcFace from linear head")
            lw = model.linear_head[-1].weight.data.clone()
            model.arcface_head.init_from_linear(lw)
            ema.shadow.arcface_head.init_from_linear(lw)

        if not class_f1_s1:
            print("[WARN] No class_f1 in Stage 1 meta — recomputing")
            _, val_cd, _ = build_loaders(train_idx, val_idx, test_idx, 128)
            class_f1_s1, cdws_wts_s1 = compute_class_difficulty(
                ema.shadow, val_cd, device, "Stage 1 (recomputed)")

        print("\n[RUN] Stage 2")
        tr2, va2, _ = build_loaders(train_idx, val_idx, test_idx,
                                    CONFIG["s2_batch"],
                                    balanced=True, all_labels=all_labels,
                                    train_aug="light", class_weights=cdws_wts_s1)
        run_stage2(model, ema, tr2, va2, device, ckpt_s2, class_f1_s1)
        print("[INFO] Reloading best Stage 2 checkpoint ...")
        load_ckpt(ckpt_s2, model, ema, device)
    else:
        print("\n[SKIP] Stage 2 → loading checkpoint")
        load_ckpt(ckpt_s2, model, ema, device)

    meta_s2      = load_stage_meta(2)
    class_f1_s2  = meta_s2.get("class_f1",    {})
    cdws_wts_s2  = meta_s2.get("cdws_weights", {})
    s2_best_f1   = meta_s2.get("val_f1", meta_s2.get("s2_val_f1", meta_s2.get("val_acc", 0.0)))
    print(f"[INFO] Stage 2 → F1={s2_best_f1:.3f}")

    if hasattr(torch, "_dynamo"):
        print("[INFO] Disabling torch.compile for Stage 3 stability")
        torch._dynamo.reset()

    if done_stage < 3:
        if not cdws_wts_s2:
            print("[WARN] No cdws_weights in Stage 2 meta — falling back to Stage 1")
            cdws_wts_s2 = cdws_wts_s1

        print("\n[RUN] Stage 3 (SAM + Greedy SWA)")
        tr3, va3, _ = build_loaders(train_idx, val_idx, test_idx,
                                    CONFIG["s2_batch"],
                                    balanced=True, all_labels=all_labels,
                                    train_aug="light", class_weights=cdws_wts_s2)
        run_stage3_swa(model, ema, tr3, va3, device, ckpt_s3, prev_best_f1=s2_best_f1)
    else:
        print("\n[SKIP] Stage 3 → loading checkpoint")
        load_ckpt(ckpt_s3, model, ema, device)
        meta_s3 = load_stage_meta(3)
        print(f"[INFO] Stage 3 → snaps={meta_s3.get('swa_n_snapshots','?')}  "
              f"rejected={meta_s3.get('swa_n_rejected','?')}  "
              f"F1={meta_s3.get('val_f1', meta_s3.get('val_acc',0)):.3f}")

    best_final_ckpt = _pick_best_checkpoint(ckpt_s1, ckpt_s2, ckpt_s3)
    print(f"\n[INFO] Best checkpoint (by val_f1): {best_final_ckpt}")

    _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx, 256)
    final_evaluation(model, ema, test_ldr, device, best_final_ckpt)

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