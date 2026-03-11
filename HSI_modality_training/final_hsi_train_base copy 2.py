# code2
from __future__ import annotations

import copy, json as _json, math, os, random, warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch._dynamo
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
    # ── Paths ─────────────────────────────────────────────────────────
    "patches_data":    "./dataset/patches.npy",
    "labels_path":     "./dataset/labels.npy",
    "wavelength_path": "./dataset/wavelengths.csv",
    "output_dir":      "./output_v10/",

    # ── Dataset ───────────────────────────────────────────────────────
    "num_bands":       256,
    "num_classes":     90,

    # ── Stage 1 — 3-Phase Progressive Augmentation ────────────────────
    "s1_epochs":            400,
    "s1_phase1_frac":       0.15,
    "s1_phase2_frac":       0.35,
    "s1_warmup_epochs":     10,
                                 
    "s1_batch":             128,
    "s1_max_lr":            1e-3,
    "s1_min_lr":            1e-6,
    "s1_dropout":           0.10,
    "s1_patience":          120,
    "s1_accum":             1,
    "s1_focal_gamma":       1.5,
    "s1_label_smooth_hi":   0.1,
    "s1_label_smooth_lo":   0.05,
    "s1_ema_reinit_phases": True,

    # ── Stage 1 · Phase 3 — Hard-Class Oversampling ───────────────────
    "s1_p3_oversample":         True,
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
    "aux_loss_weight_init":  0.25,
    "aux_loss_weight_final": 0.10,

    # ── Stage 2 ───────────────────────────────────────────────────────
    "s2_epochs":            120,
    "s2_batch":             128,
    "s2_head_lr":           1.5e-4,
    "s2_back_lr":           1.5e-5,
    "s2_min_lr":            1e-7,
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
    "s3_supcon_weight":      0.02,
    "s3_proto_weight":       0.01,

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
}

print(CONFIG)

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.cuda.empty_cache()

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True

_GPU_PATCHES:    Optional[torch.Tensor] = None
_GLOBAL_LABELS:  Optional[np.ndarray]  = None
_PHYSICAL_WL:    Optional[torch.Tensor] = None


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
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

    if _PHYSICAL_WL is not None:
        return

    print("[DATA] Loading physical wavelengths from CSV...")
    try:
        df      = pd.read_csv(csv_path, sep=None, engine="python")
        raw_wl  = df.iloc[:, -1].values.astype(np.float32)
        wl_norm = (raw_wl - raw_wl.min()) / (raw_wl.max() - raw_wl.min())
        _PHYSICAL_WL = torch.from_numpy(wl_norm).to(device)
        print(f"[DATA] ✓ Loaded physical wavelengths: {_PHYSICAL_WL.size(0)} bands.")
    except Exception as e:
        raise RuntimeError(f"Failed to load wavelengths.csv: {e}")


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
        lp = dict(model.named_parameters())
        for n, sp in self.shadow.named_parameters():
            if n in lp:
                sp.copy_(d * sp + (1.0 - d) * lp[n])
        lb = dict(model.named_buffers())
        for n, sb in self.shadow.named_buffers():
            if n in lb and sb.dtype.is_floating_point:
                sb.copy_(lb[n])

    def reinit_from(self, model: nn.Module) -> None:
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    _PROFILES = {
        "heavy":  dict(band_drop=0.08, cutout=0.06, noise=0.04, warp=0.03, mult=0.05),
        "medium": dict(band_drop=0.05, cutout=0.04, noise=0.03, warp=0.02, mult=0.03),
        "light":  dict(band_drop=0.0,  cutout=0.0,  noise=0.0,  warp=0.0,  mult=0.0),
        "none":   None,
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

    def __len__(self) -> int:
        return len(self.indices)

    def _band_dropout(self, x: torch.Tensor, prob: float) -> torch.Tensor:
        C    = x.shape[0]
        mask = (torch.rand(C, device=x.device) > prob).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x: torch.Tensor) -> torch.Tensor:
        x       = x.clone()
        C       = x.shape[0]
        max_cut = max(1, CONFIG["max_cutout_bands"])
        cut     = torch.randint(1, max_cut + 1, (1,)).item()
        start   = torch.randint(0, max(1, C - cut), (1,)).item()
        x[start:start + cut] = 0.0
        return x

    def _spectral_noise(self, x: torch.Tensor) -> torch.Tensor:
        sigma = CONFIG["noise_std"] * self.intensity_scale
        mask  = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        return x + torch.randn_like(x) * sigma * mask

    def _spectral_warp(self, x: torch.Tensor) -> torch.Tensor:
        if self.warp_range <= 0:
            return x
        C, H, W = x.shape
        scale   = 1.0 + random.uniform(-self.warp_range, self.warp_range)
        new_C   = max(1, int(C * scale))
        if new_C == C:
            return x
        xp     = x.permute(1, 2, 0).reshape(-1, 1, C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s      = (new_C - C) // 2
            warped = warped[:, :, s:s + C]
        else:
            pad_l  = (C - new_C) // 2
            pad_r  = C - new_C - pad_l
            warped = F.pad(warped, (pad_l, pad_r))
        return warped.reshape(H, W, C).permute(2, 0, 1)

    def _multiplicative_noise(self, x: torch.Tensor) -> torch.Tensor:
        scale_std = 0.05 * self.intensity_scale
        mask      = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        factor    = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device) * scale_std
        return x * factor * mask

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [1])
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k, [1, 2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ri    = self.indices[idx]
        
        patch_np = np.array(self.patches[ri])
        patch = torch.from_numpy(patch_np).to(CONFIG["device"], non_blocking=True)
        label = torch.tensor(int(self.labels[ri]), dtype=torch.long, device=CONFIG["device"])

        if self.profile is not None:
            p = self.profile
            if torch.rand(1) < p["band_drop"]:
                patch = self._band_dropout(patch, p["band_drop"])
            if torch.rand(1) < p["cutout"]:
                patch = self._band_cutout(patch)
            if torch.rand(1) < p["noise"]:
                patch = self._spectral_noise(patch)
            if torch.rand(1) < p["warp"]:
                patch = self._spectral_warp(patch)
            if torch.rand(1) < p["mult"]:
                patch = self._multiplicative_noise(patch)
            patch = self._spatial(patch)

        return patch, label


# ══════════════════════════════════════════════════════════════════════
#  SAMPLERS
# ══════════════════════════════════════════════════════════════════════

class ClassBalancedBatchSampler(Sampler):
    def __init__(self, train_labels: np.ndarray, n_cls: int = 16, n_spc: int = 8,
                 class_weights: Optional[Dict[int, float]] = None) -> None:
        self.n_cls  = n_cls
        self.n_spc  = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n      = len(train_labels) // (n_cls * n_spc)
        if class_weights is not None:
            raw        = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
            self.probs = raw / raw.sum()
        else:
            self.probs = None

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False, p=self.probs)
            batch  = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(
                    rng.choice(pool, self.n_spc, replace=len(pool) < self.n_spc).tolist()
                )
            yield batch

    def __len__(self) -> int:
        return self._n


class HardClassOversampledSampler(Sampler):
    def __init__(
        self,
        labels:           np.ndarray,
        class_f1:         Dict[int, float],
        num_samples:      int,
        oversample_power: float = 0.75,
        max_weight:       float = 5.0,
        hard_f1_thresh:   float = 0.50,
        eps:              float = 0.05,
    ) -> None:
        self.num_samples = num_samples

        num_classes   = int(np.max(labels)) + 1
        raw_weights: Dict[int, float] = {}
        for c in range(num_classes):
            f1 = float(class_f1.get(c, 0.0))
            w  = (1.0 / (f1 + eps)) ** oversample_power
            raw_weights[c] = min(w, max_weight)

        mean_w = float(np.mean(list(raw_weights.values())))
        norm_weights = {c: w / mean_w for c, w in raw_weights.items()}

        sample_weights         = np.array(
            [norm_weights.get(int(lbl), 1.0) for lbl in labels], dtype=np.float32
        )
        self._weights          = torch.from_numpy(sample_weights)

        n_hard = sum(1 for f in class_f1.values() if f < hard_f1_thresh)
        hard_classes = sorted(
            [c for c, f in class_f1.items() if f < hard_f1_thresh],
            key=lambda c: class_f1[c]
        )
        print(
            f"[INFO] Phase-3 oversampling: {n_hard}/{num_classes} hard classes "
            f"(F1 < {hard_f1_thresh})  |  power={oversample_power:.2f}  "
            f"max_w={max_weight:.1f}  n_samples={num_samples:,}"
        )
        if hard_classes:
            worst5 = [(c, class_f1[c]) for c in hard_classes[:5]]
            print(f"[INFO] Hardest classes (class_id, F1): {worst5}")

    def __iter__(self) -> Iterator[int]:
        return iter(
            torch.multinomial(self._weights, self.num_samples, replacement=True).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples


# ══════════════════════════════════════════════════════════════════════
#  CLASS DIFFICULTY WEIGHTS
# ══════════════════════════════════════════════════════════════════════

def build_cdws_weights(
    class_f1: Dict[int, float],
    num_classes: int,
    max_w: float = 3.0,
    eps:   float = 0.05,
) -> Dict[int, float]:
    raw  = {c: min(1.0 / (class_f1.get(c, 0.0) + eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w / mean for c, w in raw.items()}



# ══════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C    = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        if self.ls > 0.0:
            with torch.no_grad():
                soft = torch.full_like(logits, self.ls / (C - 1))
                soft.scatter_(1, targets.view(-1, 1), 1.0 - self.ls)
            ce = -(soft * logp).sum(1)
        else:
            ce = F.nll_loss(logp, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B         = features.shape[0]
        sim       = torch.mm(features, features.T) / self.temperature
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
    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if len(classes) < 2:
            return (features * 0).sum()
        protos = F.normalize(
            torch.stack([features[labels == c].mean(0) for c in classes]), dim=1
        )
        sim   = torch.mm(features, protos.T) / self.temperature
        c2l   = {c.item(): i for i, c in enumerate(classes)}
        local = torch.tensor(
            [c2l[y.item()] for y in labels], dtype=torch.long, device=features.device
        )
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  SAM — Sharpness-Aware Minimisation
# ══════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs) -> None:
        defaults             = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer  = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups    = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step / second_step.")

    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        ns  = [p.grad.norm(p=2).to(dev)
               for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(ns), p=2).clamp(min=1e-6) if ns else torch.tensor(0.0)

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups


# ══════════════════════════════════════════════════════════════════════
#  ADAPTIVE SUB-CENTER ARCFACE
# ══════════════════════════════════════════════════════════════════════

class AdaptiveSubcenterArcFaceHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        K: int = 2,
        s: float = 32.0,
        m_base: float = 0.35,
        m_delta: float = 0.10,
    ) -> None:
        super().__init__()
        self.K = K; self.C = num_classes
        self.s = s; self.m_base = m_base; self.m_delta = m_delta
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))

    def update_margins_from_f1(self, class_f1: Dict[int, float]) -> None:
        for c, f1 in class_f1.items():
            self.margins[c] = self.m_base + self.m_delta * (1.0 - min(float(f1), 1.0))
        print(
            f"[INFO] ArcFace margins  mean={self.margins.mean():.3f}  "
            f"min={self.margins.min():.3f}  max={self.margins.max():.3f}"
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        global_m: Optional[float] = None,
    ) -> torch.Tensor:
        x_n    = F.normalize(x, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        cosine = (
            F.linear(x_n, w_n)
            .clamp(-1 + 1e-6, 1 - 1e-6)
            .view(-1, self.C, self.K)
            .max(dim=2).values
        )
        if labels is None or not self.training:
            return cosine * self.s
        m_per = (
            torch.full((x.shape[0],), global_m, device=x.device)
            if global_m is not None else self.margins[labels]
        )
        cosm  = torch.cos(m_per); sinm = torch.sin(m_per)
        th    = torch.cos(math.pi - m_per); mm = torch.sin(math.pi - m_per) * m_per
        sine  = torch.sqrt(torch.clamp(1 - cosine ** 2, min=1e-6))
        tgt_c = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        tgt_s = sine.gather(1, labels.view(-1, 1)).squeeze(1)
        phi   = tgt_c * cosm - tgt_s * sinm
        phi   = torch.where(tgt_c > th, phi, tgt_c - mm)
        oh    = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi.unsqueeze(1)) + ((1 - oh) * cosine)) * self.s

    def init_from_linear(self, linear_w: torch.Tensor) -> None:
        with torch.no_grad():
            wn = F.normalize(linear_w, dim=1)
            for k in range(self.K):
                noise = torch.randn_like(wn) * 0.01 * k
                self.weight[k::self.K].copy_(wn + noise)
        print(f"[INFO] ArcFace (K={self.K}) bootstrapped from linear head.")


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class DifferentiableSpectralRouter(nn.Module):
    def __init__(self, channels: int, d_model: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.stat_proj = nn.Linear(3, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        mask = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()
        valid_pixels = mask.sum(dim=[2, 3]).clamp(min=1e-5)
        
        x_masked = x * mask
        x_mean = x_masked.sum(dim=[2, 3]) / valid_pixels
        
        x_max = x.masked_fill(mask == 0, -1e4).amax(dim=[2, 3])
        x_max = x_max.masked_fill(x_max == -1e4, 0.0)
        
        x_var = ((x_masked - x_mean.unsqueeze(-1).unsqueeze(-1))**2 * mask).sum(dim=[2, 3]) / valid_pixels
        x_std = torch.sqrt(x_var + 1e-5)
        
        stats = torch.stack([x_mean, x_max, x_std], dim=-1)
        tokens = self.stat_proj(stats)
        
        t_norm = self.norm1(tokens)
        attn_out, _ = self.self_attn(t_norm, t_norm, t_norm, need_weights=False)
        tokens = tokens + attn_out
        
        logits = self.gate_proj(tokens).squeeze(-1) 
        
        if self.training:
            u1 = torch.rand_like(logits)
            u2 = torch.rand_like(logits)
            gumbel_noise = torch.log(-torch.log(u1 + 1e-8) + 1e-8) - torch.log(-torch.log(u2 + 1e-8) + 1e-8)
            
            tau = torch.clamp(self.tau, min=0.1, max=5.0)
            soft_gate = torch.sigmoid((logits + gumbel_noise) / tau)
            hard_gate = (soft_gate > 0.5).float()
            gate = hard_gate.detach() - soft_gate.detach() + soft_gate
        else:
            gate = (logits > 0.0).float()
        
        safe_gate = 0.9 * gate + 0.1
        return x * safe_gate.view(B, C, 1, 1)
        
class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)
    

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dilation: int = 1) -> None:
        super().__init__()
        pad        = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.se    = SEBlock1D(out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + identity)

class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid     = max(c // r, 8)
        self.ch = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.GELU(),
                                 nn.Conv2d(mid, c, 1, bias=False))
        self.sp = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(
            self.ch(x.mean([2, 3], keepdim=True)) +
            self.ch(x.amax([2, 3], keepdim=True))
        )
        return x * self.sp(
            torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        )


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid     = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid, 1, bias=False);          self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False); self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid, out_ch, 1, bias=False);         self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch)
            )
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(
            self.n3(self.c3(F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) +
            self.skip(x)
        )


class PhysicalWavelengthPE(nn.Module):
    def __init__(self, physical_wl: torch.Tensor, d_model: int) -> None:
        super().__init__()
        dev  = physical_wl.device
        half = d_model // 2
        freq = torch.exp(
            torch.arange(half, device=dev).float() *
            -(math.log(10000.0) / max(half - 1, 1))
        )
        pe         = torch.zeros(physical_wl.size(0), d_model, device=dev)
        pe[:, :half] = torch.sin(physical_wl.unsqueeze(1) * freq.unsqueeze(0))
        pe[:, half:] = torch.cos(physical_wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.transpose(0, 1).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A — SPECTRAL PROFILE
# ══════════════════════════════════════════════════════════════════════

class LargeKernelBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv1d(dim, dim * 4, 1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(dim * 4, dim, 1, bias=False)
        self.se = SEBlock1D(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.se(x)
        return x + res


class SpectralProfileBranch(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        tower_ch: int = 96,
        wl_pe_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.wl_pe_module = wl_pe_module

        self.d1_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.d2_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        
        with torch.no_grad():
            self.d1_conv.weight[0, 0] = torch.tensor([-3, -2, -1, 0, 1, 2, 3]).float() / 28.0
            self.d2_conv.weight[0, 0] = torch.tensor([5, 0, -3, -4, -3, 0, 5]).float() / 42.0

        self.stem = nn.Sequential(
            nn.Conv1d(3, tower_ch, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(1, tower_ch),
            nn.GELU()
        )

        self.tower_s = nn.Sequential(LargeKernelBlock1D(tower_ch, 7), LargeKernelBlock1D(tower_ch, 7))
        self.tower_m = nn.Sequential(LargeKernelBlock1D(tower_ch, 15), LargeKernelBlock1D(tower_ch, 15))
        self.tower_l = nn.Sequential(LargeKernelBlock1D(tower_ch, 31), LargeKernelBlock1D(tower_ch, 31))

        self.fusion = nn.Sequential(
            nn.Conv1d(tower_ch * 3, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch),
            nn.GELU(),
            LargeKernelBlock1D(tower_ch, 7)
        )
        
        self.attn_pool = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1),
            nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and m not in [self.d1_conv, self.d2_conv]:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ms: torch.Tensor) -> torch.Tensor:
        s = ms.unsqueeze(1)
        s_smooth = F.avg_pool1d(s, kernel_size=5, stride=1, padding=2)
        d1 = self.d1_conv(s_smooth)
        d2 = self.d2_conv(s_smooth) 
        x = torch.cat([s, d1, d2], dim=1)
        x = self.stem(x)

        if self.wl_pe_module is not None:
            x = self.wl_pe_module(x)

        x_fused = self.fusion(
            torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1)
        )
        
        w = torch.softmax(self.attn_pool(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))
    

# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(
        self,
        num_bands: int,
        out_dim: int = 256,
        tower_ch: int = 96,
        wl_pe_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_channels  = 9
        self.wl_pe_module = wl_pe_module

        self.stat_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.in_channels, 16, 1, bias=False), nn.GELU(),
            nn.Conv1d(16, self.in_channels, 1, bias=False), nn.Sigmoid()
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch), nn.GELU()
        )

        def _make_tower(kernel: int) -> nn.Sequential:
            return nn.Sequential(
                ResBlock1D(tower_ch, tower_ch, kernel),
                ResBlock1D(tower_ch, tower_ch, kernel)
            )

        self.tower_s = _make_tower(3)
        self.tower_m = _make_tower(7)
        self.tower_l = _make_tower(15)

        self.fusion = nn.Sequential(
            ResBlock1D(tower_ch * 3, tower_ch, 5),
            ResBlock1D(tower_ch, tower_ch, 5)
        )
        self.pool_attn = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1, bias=False)
        )
        self.proj = nn.Sequential(
            nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ms, std, mx, skew, kurt, p10, p25, p75, p90):
        stats    = torch.stack([ms, std, mx, skew, kurt, p10, p25, p75, p90], dim=1)
        stats    = stats * self.stat_attn(stats)
        x        = self.input_proj(stats)

        if self.wl_pe_module is not None:
            x = self.wl_pe_module(x)

        x_fused = self.fusion(
            torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1)
        )
        w = torch.softmax(self.pool_attn(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands: int = 256, out_dim: int = 256) -> None:
        super().__init__()

        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, 1, groups=num_bands, bias=False),
            nn.Conv2d(num_bands, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
                
        self.stages = nn.Sequential(
            ResBlock2D(64,   128, 2), CBAM(128),
            ResBlock2D(128,  192, 2), CBAM(192),
            ResBlock2D(192,  256, 2), CBAM(256),
            ResBlock2D(256,  out_dim, 2)
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    @staticmethod
    def _pn(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().clamp(1e-8).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.band_reduce(x))
        return self.proj(
            F.normalize(torch.cat([self._pn(h.mean([2, 3])), self._pn(h.amax([2, 3]))], 1), dim=1, eps=1e-4)
        )

# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER
# ══════════════════════════════════════════════════════════════════════
class MultiScaleSpectralTokenizer(nn.Module):
    def __init__(self, in_channels: int, d_model: int, stride: int = 8):
        super().__init__()
        # 3 parallel tokenizers with different receptive fields to capture 
        # narrow absorption lines and broad spectral shapes
        out_c = d_model // 3
        rem   = d_model - (out_c * 2)
        
        self.proj_small  = nn.Conv1d(in_channels, out_c, kernel_size=8,  stride=stride, padding=4)
        self.proj_medium = nn.Conv1d(in_channels, out_c, kernel_size=16, stride=stride, padding=8)
        self.proj_large  = nn.Conv1d(in_channels, rem,   kernel_size=32, stride=stride, padding=16)
        
        self.norm = nn.GroupNorm(1, d_model)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Channels, Length]
        t_s = self.proj_small(x)
        t_m = self.proj_medium(x)
        t_l = self.proj_large(x)
        
        # Ensure identical sequence lengths by truncating to the shortest
        min_len = min(t_s.size(2), t_m.size(2), t_l.size(2))
        t_s, t_m, t_l = t_s[..., :min_len], t_m[..., :min_len], t_l[..., :min_len]
        
        tokens = torch.cat([t_s, t_m, t_l], dim=1)
        return self.act(self.norm(tokens))


class _PreLNBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, drop: float) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d_ff, d), nn.Dropout(drop)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lx      = self.ln1(x)
        h, _    = self.attn(lx, lx, lx, need_weights=False)
        x       = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))


class SpecFormerBranch(nn.Module):
    """
    State-of-the-Art Spatial-Spectral Factorised Transformer.
    Processes Multi-Scale Spectral features first, then correlates Spatial grids.
    """
    def __init__(
        self,
        physical_wl: torch.Tensor,
        num_bands:  int = 256,
        patch_size: int = 16, # Kept for API compatibility, handled by MultiScale
        stride:     int = 8,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 4,
        out_dim:    int = 256,
        dropout:    float = 0.15,
    ) -> None:
        super().__init__()
        # Input features: 1 (Raw) + 1 (First Derivative) = 2
        self.tokenizer = MultiScaleSpectralTokenizer(in_channels=2, d_model=d_model, stride=stride)
        
        # Spectral cls token and positional embedding
        self.spec_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spec_cls, std=0.02)
        
        # Estimate number of tokens (approx num_bands // stride)
        n_tokens = (num_bands // stride) + 2 
        self.spec_pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # Factorized Transformer Stages
        # 1. Spectral Attention (Local chemical composition)
        self.spectral_blocks = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers // 2)
        ])
        
        # 2. Spatial Attention (Global seed morphology)
        self.spatial_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spatial_cls, std=0.02)
        
        self.spatial_blocks = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers // 2)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    def forward(self, grid_ms: torch.Tensor) -> torch.Tensor:
        # Expected input: [B, N, C] where N is number of spatial grids (e.g., 16)
        B, N, C = grid_ms.shape
        
        # 1. Infuse 1st Derivative (Gradient of the spectrum)
        # Calculates difference between adjacent bands; pads the last band
        deriv = torch.diff(grid_ms, dim=2)
        deriv = F.pad(deriv, (0, 1), mode='replicate') 
        
        # Combine Raw + Deriv -> [B*N, 2, C]
        x_combo = torch.stack([grid_ms, deriv], dim=2).view(B * N, 2, C)

        # 2. Multi-Scale Tokenization
        tokens = self.tokenizer(x_combo).transpose(1, 2) # [B*N, Seq, d_model]
        
        # 3. Spectral Stage (Within each grid cell)
        cls_tokens = self.spec_cls.expand(B * N, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add Positional Embedding safely
        seq_len = tokens.size(1)
        if seq_len <= self.spec_pos_embed.size(1):
            tokens = tokens + self.spec_pos_embed[:, :seq_len, :]
            
        for blk in self.spectral_blocks:
            tokens = blk(tokens)
            
        # Extract the spectral CLS token to represent the entire grid cell's spectrum
        grid_features = tokens[:, 0, :] # [B*N, d_model]
        
        # 4. Spatial Stage (Across grid cells)
        # Reshape back to separate batch and spatial dimensions
        spatial_tokens = grid_features.view(B, N, -1) # [B, N, d_model]
        
        spatial_cls = self.spatial_cls.expand(B, -1, -1)
        spatial_tokens = torch.cat([spatial_cls, spatial_tokens], dim=1)
        
        for blk in self.spatial_blocks:
            spatial_tokens = blk(spatial_tokens)
            
        # Final classification token representing the whole Spatio-Spectral object
        global_feature = self.norm(spatial_tokens[:, 0, :])
        
        return self.proj(global_feature)
    
# ══════════════════════════════════════════════════════════════════════
#  BRANCH FUSION — Residual Cross-Modal Interaction
# ══════════════════════════════════════════════════════════════════════

class CrossModalInteraction(nn.Module):
    def __init__(self, num_modalities: int = 4, d: int = 256, drop: float = 0.1) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.d              = d

        self.branch_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_modalities)])

        in_dim = d * num_modalities
        mid    = in_dim // 4
        self.se_interaction = nn.Sequential(
            nn.Linear(in_dim, mid, bias=False), nn.GELU(),
            nn.Linear(mid, in_dim, bias=False), nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Linear(in_dim, d * 2), nn.LayerNorm(d * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d * 2, d),     nn.LayerNorm(d)
        )

    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        normed  = [norm(b) for norm, b in zip(self.branch_norms, branches)]
        cat     = torch.cat(normed, dim=1)
        weights = self.se_interaction(cat)
        return self.project(cat + cat * weights)


# ══════════════════════════════════════════════════════════════════════
#  AUXILIARY CLASSIFICATION HEAD
# ══════════════════════════════════════════════════════════════════════

class AuxiliaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.net[1].weight, std=0.02)
        nn.init.zeros_(self.net[1].bias)
        nn.init.trunc_normal_(self.net[3].weight, std=0.02)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════
#  SPECTRAL STATISTICS HELPER
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_branch_influence(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
) -> Dict[str, float]:
    model.eval()
    influences = torch.zeros(4, device=device)
    total      = 0

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x            = x.to(device, non_blocking=True)
        logits_full  = model(x)
        p_full       = torch.softmax(logits_full, dim=1)

        for b in range(4):
            mask       = torch.ones(4, device=device); mask[b] = 0.0
            logits_ab  = model(x, branch_mask=mask)
            p_ab       = torch.softmax(logits_ab, dim=1).clamp(min=1e-10)
            influences[b] += F.kl_div(p_ab.log(), p_full, reduction="batchmean")
        total += 1

    if total == 0:
        return {"A": 0, "B": 0, "C": 0, "D": 0}

    influences  /= total
    total_inf    = influences.sum().clamp(min=1e-8)
    influences   = influences / total_inf * 100.0
    return {k: float(influences[i]) for i, k in enumerate("ABCD")}


def extract_grid_spectra(x: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    B, C, H, W = x.shape
    mask = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()
    grid_sum = F.adaptive_avg_pool2d(x * mask, (grid_size, grid_size))
    grid_mask_sum = F.adaptive_avg_pool2d(mask, (grid_size, grid_size))
    grid_mean = grid_sum / grid_mask_sum.clamp(min=1e-5)
    return grid_mean.view(B, C, -1).transpose(1, 2)

def masked_spectral_stats(x: torch.Tensor):
    x32 = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)

    mask = (flat.abs().sum(1, keepdim=True) > 1e-5).float()
    cnt = mask.sum(2).clamp(min=1.0)

    mean = (flat * mask).sum(2) / cnt
    centered = (flat - mean.unsqueeze(2)) * mask
    var = (centered ** 2).sum(2) / cnt
    std = torch.sqrt(var + 1e-5)
    mx = flat.masked_fill(mask.expand_as(flat) == 0, -1e4).max(2).values
    mx = mx.masked_fill(mx < -9999.0, 0.0)
    skew = torch.clamp(((centered**3).sum(2)/cnt) / (std**3 + 1e-4), -10.0, 10.0)
    kurt = torch.clamp(((centered**4).sum(2)/cnt) / (std**4 + 1e-4), 0.0, 20.0)

    flat_masked = flat.masked_fill(mask.expand_as(flat) == 0, float("inf"))
    sorted_vals, _ = torch.sort(flat_masked, dim=2)

    def gather_percentile(vals, p_frac):
        idx = (cnt * p_frac).long().clamp(max=H * W - 1)
        expanded_idx = idx.unsqueeze(2).expand(-1, C, -1)
        return torch.gather(vals, 2, expanded_idx).squeeze(2)

    p10, p25 = gather_percentile(sorted_vals, 0.10), gather_percentile(sorted_vals, 0.25)
    p75, p90 = gather_percentile(sorted_vals, 0.75), gather_percentile(sorted_vals, 0.90)

    p10 = torch.where(torch.isinf(p10), 0.0, p10)
    p25 = torch.where(torch.isinf(p25), 0.0, p25)
    p75 = torch.where(torch.isinf(p75), 0.0, p75)
    p90 = torch.where(torch.isinf(p90), 0.0, p90)
    
    return (
        torch.nan_to_num(mean, 0), torch.nan_to_num(std, 0),
        torch.nan_to_num(mx, 0), torch.nan_to_num(skew, 0),
        torch.nan_to_num(kurt, 0), torch.nan_to_num(p10, 0),
        torch.nan_to_num(p25, 0), torch.nan_to_num(p75, 0),
        torch.nan_to_num(p90, 0),
    )    

# ══════════════════════════════════════════════════════════════════════
#  GRADIENT ORTHOGONALIZATION AND ADAPTIVE LEVERAGING (GOAL)
# ══════════════════════════════════════════════════════════════════════

class DynamicGradientOrthogonalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b_a, b_b, b_c, b_d, log_a, log_b, log_c, log_d, targets):
        confidences = []
        with torch.no_grad():
            for logits in [log_a, log_b, log_c, log_d]:
                prob = torch.softmax(logits, dim=1)
                conf = prob[torch.arange(prob.size(0), device=prob.device), targets].mean().detach()
                confidences.append(conf)
        
        dom_idx = torch.argmax(torch.stack(confidences))
        ctx.save_for_backward(dom_idx)

        return b_a.clone(), b_b.clone(), b_c.clone(), b_d.clone()

    @staticmethod
    def backward(ctx, g_a, g_b, g_c, g_d):
        dom_idx = ctx.saved_tensors[0].item()
        grads = [g_a, g_b, g_c, g_d]

        g_dom = grads[dom_idx]

        eps = 1e-4
        g_dom_norm = torch.linalg.vector_norm(g_dom, dim=1, keepdim=True)
        valid_mask = (g_dom_norm > 1e-6).float()
        g_dom_unit = F.normalize(g_dom, p=2, dim=1, eps=eps)

        out_grads = []
        for i in range(4):
            g = grads[i]
            if g is None:
                out_grads.append(None)
            elif i == dom_idx:
                out_grads.append(g)
            else:
                dot = (g * g_dom_unit).sum(dim=1, keepdim=True)
                proj = dot * g_dom_unit
                out_grads.append(g - proj * valid_mask)

        return (*out_grads, None, None, None, None, None)
    
@torch._dynamo.disable
def apply_goal(
    ba: torch.Tensor,
    bb: torch.Tensor,
    bc: torch.Tensor,
    bd: torch.Tensor,
    aux_list: List[torch.Tensor],
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not ba.requires_grad or targets is None:
        return ba, bb, bc, bd

    return DynamicGradientOrthogonalization.apply(
        ba, bb, bc, bd,
        aux_list[0], aux_list[1],
        aux_list[2], aux_list[3],
        targets,
    )


# ══════════════════════════════════════════════════════════════════════
#  SPECTRALQUADNET
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    """
    Four-branch hyperspectral classification model.

    Branches
    ────────
    A  SpectralProfileBranch  — raw signal + learnable derivatives
    B  SpectralStatsBranch    — masked band statistics (9 moments)
    C  SpatialCNNBranch       — 2-D spatial texture
    D  SpecFormerBranch       — spectral patch transformer

    GOAL (Gradient Orthogonalization and Adaptive Leveraging)
    ──────────────────────────────────────────────────────────
    Applied during training in Stage 1 (all 3 phases) and Stage 2.
    Disabled for Stage 3 (SAM + SWA — gradient geometry is left intact
    so SAM can find its flat minima without interference).

    Execution order in forward() during training
    ─────────────────────────────────────────────
    1. Compute branch features: ba, bb, bc, bd
    2. Compute aux-head logits on the RAW features → aux_out  (used for loss
       AND as read-only confidence signals inside GOAL)
    3. Apply GOAL ONCE → ba', bb', bc', bd'  (modifies backward graph only;
       forward values are identical to raw features)
    4. Fuse ba'…bd' through CrossModalInteraction → main logits
    5. Return {main, aux_a…aux_d} where aux_* come from step 2
       (their gradients bypass GOAL, giving each branch a direct
        supervision signal independent of the orthogonalisation)

    Heads
    ─────
    Stage 1 : linear_head  (CE / Focal, no margin)
    Stage 2+ : arcface_head (Sub-centre ArcFace)
    """

    def __init__(
        self,
        num_classes: int = 90,
        num_bands:   int = 256,
        dropout:     float = 0.30,
        wl_embed_dim: int = 16,
        cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        global _PHYSICAL_WL

        cfg      = cfg or CONFIG
        tower_ch = 96

        self.branch_drop_prob = cfg.get("branch_drop_prob", 0.0)

        self.se        = DifferentiableSpectralRouter(channels=num_bands)
        self.wl_pe_cnn = PhysicalWavelengthPE(_PHYSICAL_WL, tower_ch)

        self.branch_a = SpectralProfileBranch(
            out_dim=256, tower_ch=tower_ch, wl_pe_module=self.wl_pe_cnn
        )
        self.branch_b = SpectralStatsBranch(
            num_bands=num_bands, out_dim=256, tower_ch=96, wl_pe_module=self.wl_pe_cnn
        )
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(
            physical_wl=_PHYSICAL_WL,
            num_bands=num_bands,
            patch_size=cfg["specf_patch"],
            stride=cfg["specf_patch"] // 2,
            d_model=cfg["specf_dim"],
            n_heads=cfg["specf_heads"],
            n_layers=cfg["specf_layers"],
            out_dim=256,
            dropout=0.10,
        )

        self.cross_interaction = CrossModalInteraction(
            num_modalities=4, d=256, drop=cfg["fusion_drop"]
        )

        aux_hidden = cfg.get("aux_head_hidden", 128)
        self.aux_head_a = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_b = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_c = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_d = AuxiliaryHead(256, aux_hidden, num_classes)

        self.embed_net = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256),
        )

        self.linear_head  = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout * 0.4), nn.Linear(256, num_classes)
        )
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            256, num_classes,
            K=cfg.get("subcenter_K", 2),
            s=cfg["s2_arcface_s"],
            m_base=cfg["s2_arcface_m"],
            m_delta=cfg.get("s2_arcface_m_delta", 0.10),
        )
        self._use_arcface = False
        self._use_goal    = True
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def use_arcface(self, flag: bool) -> None:
        self._use_arcface = flag

    def use_goal(self, flag: bool) -> None:
        self._use_goal = flag

    def freeze_head(self, which: str) -> None:
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which: str) -> None:
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(True)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor]      = None,
        return_embed: bool                  = False,
        arc_m: Optional[float]             = None,
        branch_mask: Optional[torch.Tensor] = None,
    ):
        x = self.se(x)
        
        # Global stats (Branch B)
        ms, std, mx, skew, kurt, p10, p25, p75, p90 = masked_spectral_stats(x)
        # Regional grid spectra (Branches A and D)
        grid_ms = extract_grid_spectra(x, grid_size=4)
        B, N, C = grid_ms.shape
        flat_grid_ms = grid_ms.reshape(B * N, C)

        # ── Branch A (Spectral Profile) ──────────────────────────────
        ba_grid = self.branch_a(flat_grid_ms)
        ba_raw  = ba_grid.view(B, N, -1).mean(dim=1)

        # ── Branch B (Spectral Stats) ────────────────────────────────
        bb_raw = self.branch_b(ms, std, mx, skew, kurt, p10, p25, p75, p90)
        
        # ── Branch C (Spatial CNN) ───────────────────────────────────
        bc_raw = self.branch_c(x)

        # ── Branch D (SpecFormer) ────────────────────────────────────
        bd_raw = self.branch_d(grid_ms)

        ba_raw = F.normalize(ba_raw, dim=1, eps=1e-4).contiguous()
        bb_raw = F.normalize(bb_raw, dim=1, eps=1e-4).contiguous()
        bc_raw = F.normalize(bc_raw, dim=1, eps=1e-4).contiguous()
        bd_raw = F.normalize(bd_raw, dim=1, eps=1e-4).contiguous()
        
        # ── Branch Masking (stochastic branch-drop during training) ──
        if branch_mask is not None:
            ba = ba_raw * branch_mask[0]; bb = bb_raw * branch_mask[1]
            bc = bc_raw * branch_mask[2]; bd = bd_raw * branch_mask[3]
        elif self.training:
            drop_probs = torch.tensor([0.10, 0.10, 0.40, 0.15], device=ba_raw.device)
            keeps      = (torch.rand(4, device=ba_raw.device) > drop_probs).float()
            safe_idx   = torch.randint(0, 4, (), device=ba_raw.device)
            safe_mask  = F.one_hot(safe_idx, num_classes=4).float()
            keeps      = torch.maximum(keeps, safe_mask)
            ba = ba_raw * keeps[0]; bb = bb_raw * keeps[1]
            bc = bc_raw * keeps[2]; bd = bd_raw * keeps[3]
        else:
            ba, bb, bc, bd = ba_raw, bb_raw, bc_raw, bd_raw

        # ── Step 1: Auxiliary heads on RAW (pre-GOAL) branch features ─
        aux_out: dict = {}
        if self.training:
            aux_out["aux_a"] = self.aux_head_a(ba)
            aux_out["aux_b"] = self.aux_head_b(bb)
            aux_out["aux_c"] = self.aux_head_c(bc)
            aux_out["aux_d"] = self.aux_head_d(bd)

        # ── Step 2: GOAL — applied ONCE, only in Stages 1 & 2 ────────
        if self.training and self._use_goal and labels is not None:
            ba, bb, bc, bd = apply_goal(
                ba, bb, bc, bd,
                [aux_out["aux_a"], aux_out["aux_b"],
                 aux_out["aux_c"], aux_out["aux_d"]],
                labels,
            )
        
        # ── Step 3: Cross-modal fusion on (GOAL-modified) features ───
        joint_token = self.cross_interaction([ba, bb, bc, bd])
        emb         = self.embed_net(joint_token)

        # ── Step 4: Classification head ───────────────────────────────
        if self._use_arcface:
            logits = self.arcface_head(F.normalize(emb, dim=1), labels, global_m=arc_m)
        else:
            logits = self.linear_head(emb)

        # ── Training output ───────────────────────────────────────────
        if self.training:
            out = {
                "main":  logits,
                **aux_out,
            }
            if return_embed:
                out["emb"] = F.normalize(emb, dim=1)
            return out

        # ── Inference output ──────────────────────────────────────────
        if return_embed:
            return logits, F.normalize(emb, dim=1)
        return logits

    
# ══════════════════════════════════════════════════════════════════════
#  TTA — 8 spatial + 4 spectral
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_spatial: int = 8,
    n_spectral: int = 4,
) -> torch.Tensor:
    device        = x.device
    logits        = []
    spatial_views = [(k, f) for k in range(4) for f in (False, True)][:n_spatial]

    for k, flip in spatial_views:
        aug = torch.rot90(x, k, [2, 3])
        if flip:
            aug = torch.flip(aug, [3])
        with autocast(device_type=device.type):
            out = model(aug)
            logits.append(out["main"] if isinstance(out, dict) else out)

    scales = torch.linspace(0.95, 1.05, n_spectral, device=device)
    for s in scales:
        if abs(s.item() - 1.0) < 1e-5:
            continue
        mean    = x.mean(dim=[2, 3], keepdim=True)
        aug_sp  = mean + (x - mean) * s
        with autocast(device_type=device.type):
            out = model(aug_sp)
            logits.append(out["main"] if isinstance(out, dict) else out)

    return torch.stack(logits).mean(0)


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels  = _GLOBAL_LABELS
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels,       random_state=42)
    va, te  = train_test_split(tmp,     test_size=0.5, stratify=labels[tmp],  random_state=42)
    return labels, tr, va, te


def build_loaders(
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    test_idx:  np.ndarray,
    batch_train: int,
    balanced: bool = False,
    all_labels: Optional[np.ndarray] = None,
    train_aug:  str = "none",
    class_weights: Optional[Dict[int, float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    ds = RiceSeedDataset(train_idx, aug_strength=train_aug)

    if balanced and all_labels is not None:
        samp   = ClassBalancedBatchSampler(
            all_labels[train_idx],
            CONFIG["bal_n_cls"],
            CONFIG["bal_n_spc"],
            class_weights=class_weights,
        )
        tr_ldr = DataLoader(ds, batch_sampler=samp, num_workers=0)
    else:
        tr_ldr = DataLoader(ds, batch_size=batch_train, shuffle=True, drop_last=True, num_workers=0)

    va_ldr = DataLoader(RiceSeedDataset(val_idx),  batch_size=256, shuffle=False, num_workers=0)
    te_ldr = DataLoader(RiceSeedDataset(test_idx), batch_size=256, shuffle=False, num_workers=0)
    return tr_ldr, va_ldr, te_ldr


def build_phase3_loader(
    train_ds:  Dataset,
    class_f1:  Dict[int, float],
) -> DataLoader:
    if not CONFIG["s1_p3_oversample"] or not class_f1:
        return DataLoader(
            train_ds, batch_size=CONFIG["s1_batch"],
            shuffle=True, drop_last=True, num_workers=0
        )

    train_labels = np.array(
        [int(_GLOBAL_LABELS[train_ds.indices[i]]) for i in range(len(train_ds.indices))]
    )
    sampler = HardClassOversampledSampler(
        labels           = train_labels,
        class_f1         = class_f1,
        num_samples      = len(train_labels),
        oversample_power = CONFIG["s1_p3_oversample_power"],
        max_weight       = CONFIG["s1_p3_oversample_max_w"],
        hard_f1_thresh   = CONFIG["s1_p3_hard_f1_thresh"],
        eps              = CONFIG["s1_p3_oversample_eps"],
    )
    return DataLoader(
        train_ds, batch_size=CONFIG["s1_batch"],
        sampler=sampler, drop_last=True, num_workers=0
    )


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISERS & SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def _wd_groups(named_params, lr: float) -> List[dict]:
    wd, no_wd = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_wd if (p.ndim == 1 or n.endswith(".bias")) else wd).append(p)
    return [
        {"params": wd,    "lr": lr, "weight_decay": CONFIG["weight_decay"]},
        {"params": no_wd, "lr": lr, "weight_decay": 0.0},
    ]


def build_optimizer_s1(model: nn.Module, lr: float) -> optim.AdamW:
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))


def build_optimizer_s2(model: nn.Module, head_lr: float, back_lr: float) -> optim.AdamW:
    hp, bp = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hp if n.startswith("arcface_head") else bp).append((n, p))
    return optim.AdamW(_wd_groups(hp, head_lr) + _wd_groups(bp, back_lr))


def build_optimizer_s3(model: nn.Module, lr: float) -> optim.AdamW:
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))


def sgdr_scheduler(
    optimizer:   optim.Optimizer,
    steps_per_epoch: int,
    warmup_ep:   int   = 5,
    T_0:         int   = 10,
    T_mult:      int   = 2,
    eta_min_frac: float = 1e-3,
) -> optim.lr_scheduler.LambdaLR:
    warmup_steps = warmup_ep * steps_per_epoch
    T_0_steps    = T_0 * steps_per_epoch

    def _l(step: int) -> float:
        if step < warmup_steps:
            return eta_min_frac + (1.0 - eta_min_frac) * (step / max(warmup_steps, 1))
            
        t = step - warmup_steps
        clen = T_0_steps
        elapsed = 0
        while t >= elapsed + clen:
            elapsed += clen
            clen = max(int(clen * T_mult), 1)
            
        ratio = (t - elapsed) / max(clen, 1)
        return eta_min_frac + 0.5 * (1 - eta_min_frac) * (1 + math.cos(math.pi * ratio))
        
    return optim.lr_scheduler.LambdaLR(optimizer, _l)


def arcface_margin(ep: int, m0: float, m_target: float, warmup_ep: int) -> float:
    if ep >= warmup_ep:
        return m_target
    return m0 + (m_target - m0) * 0.5 * (1 - math.cos(math.pi * ep / max(warmup_ep, 1)))


# ══════════════════════════════════════════════════════════════════════
#  AUXILIARY LOSS HELPERS
# ══════════════════════════════════════════════════════════════════════

def _aux_loss_weight(current_ep: int, total_ep: int) -> float:
    progress = current_ep / max(total_ep, 1)
    return (
        CONFIG["aux_loss_weight_init"]
        + (CONFIG["aux_loss_weight_final"] - CONFIG["aux_loss_weight_init"]) * progress
    )


def _compute_aux_loss(
    criterion: nn.Module,
    out: dict,
    ya: torch.Tensor,
) -> torch.Tensor:
    aux_keys = ["aux_a", "aux_b", "aux_c", "aux_d"]
    total    = torch.zeros((), device=ya.device)
    for k in aux_keys:
        if k not in out:
            continue
        total = total + criterion(out[k], ya)
    return total


# ══════════════════════════════════════════════════════════════════════
#  TRAIN ONE EPOCH  (AdamW — Stage 1 and Stage 2)
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    optimizer:     optim.Optimizer,
    criterion:     nn.Module,
    scaler:        Optional[GradScaler],
    ema:           ModelEMA,
    device:        torch.device,
    supcon:        Optional[nn.Module] = None,
    supcon_weight: float = 0.0,
    proto:         Optional[nn.Module] = None,
    proto_weight:  float = 0.0,
    accum_steps:   int   = 1,
    arc_m:         Optional[float] = None,
    current_ep:    int   = 0,
    total_ep:      int   = 100,
    scheduler:     Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    step_scheduler_per_batch: bool = False,
) -> Tuple[float, float]:

    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)

    use_amp   = (supcon is None) and (scaler is not None)
    aux_w     = _aux_loss_weight(current_ep, total_ep)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_in, ya = x, y

        with autocast(device_type=device.type, enabled=use_amp):

            if supcon is not None:
                # Stage 2 path: ArcFace + SupCon + ProtoNCE
                # Labels are passed for both ArcFace margins and GOAL dominance detection.
                out    = model(x_in, ya, return_embed=True, arc_m=arc_m)
                logits = out["main"] if isinstance(out, dict) else out[0]
                emb    = out.get("emb") if isinstance(out, dict) else out[1]

                cls_l  = criterion(logits, ya)
                sc_l   = supcon(emb, ya)
                pt_l   = proto(emb, ya) if proto is not None else 0.0

                aux_l  = (
                    _compute_aux_loss(criterion, out, ya)
                    if isinstance(out, dict) else torch.zeros((), device=device)
                )

                loss = (
                    (1 - supcon_weight - proto_weight) * cls_l
                    + supcon_weight * sc_l
                    + proto_weight * pt_l
                    + aux_w * aux_l
                )

            else:
                out = model(x_in, labels=ya, arc_m=arc_m)

                if isinstance(out, dict):
                    l_main = criterion(out["main"], ya)
                    aux_l  = _compute_aux_loss(criterion, out, ya)
                    loss   = l_main + aux_w * aux_l
                    logits = out["main"]
                else:
                    logits = out
                    loss   = criterion(logits, ya)

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
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)

        if step_scheduler_per_batch and scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.argmax(1) == ya).float().mean().item()

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════════════
#  TRAIN ONE EPOCH (SAM — Stage 3)
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch_sam(
    model:         nn.Module,
    loader:        DataLoader,
    sam_opt:       SAM,
    criterion:     nn.Module,
    device:        torch.device,
    supcon:        Optional[nn.Module] = None,
    supcon_weight: float = 0.0,
    proto:         Optional[nn.Module] = None,
    proto_weight:  float = 0.0,
    arc_m:         Optional[float] = None,
    aux_weight:    float = 0.0,
) -> Tuple[float, float]:

    torch.set_default_dtype(torch.float32)
    model.train()
    total_loss = total_acc = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        sam_opt.zero_grad()
        # Stage 3: GOAL is disabled via model._use_goal=False set in run_stage3_swa.
        # Labels are still passed for ArcFace margin computation.
        out    = model(x, labels=y, arc_m=arc_m, return_embed=(supcon is not None))
        logits = out["main"] if isinstance(out, dict) else out
        emb    = out.get("emb") if isinstance(out, dict) else None

        loss = criterion(logits, y)
        if supcon is not None and emb is not None:
            loss = loss + supcon_weight * supcon(emb, y)
        if isinstance(out, dict) and aux_weight > 0.0:
            aux_l = _compute_aux_loss(criterion, out, y)
            loss  = loss + aux_weight * aux_l

        if not torch.isfinite(loss):
            sam_opt.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.first_step(zero_grad=True)

        out2    = model(x, labels=y, arc_m=arc_m, return_embed=(supcon is not None))
        logits2 = out2["main"] if isinstance(out2, dict) else out2

        loss2 = criterion(logits2, y)
        if not torch.isfinite(loss2):
            sam_opt.zero_grad()
            continue

        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        sam_opt.second_step(zero_grad=True)

        total_loss += loss.item()
        with torch.no_grad():
            total_acc += (logits.detach().argmax(1) == y).float().mean().item()

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_eval(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with autocast(device_type=device.type, enabled=False):
        for x, y in loader:
            x      = x.to(device, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            preds.append(logits.argmax(1).cpu()); targets.append(y.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    p, t = _run_eval(model, loader, device)
    return f1_score(t, p, average="macro", zero_division=0), accuracy_score(t, p)


def evaluate_per_class(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> Dict[int, float]:
    p, t   = _run_eval(model, loader, device)
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
    return os.path.isfile(stage_ckpt_path(s)) and os.path.isfile(stage_meta_path(s))


def latest_completed_stage() -> int:
    for s in (3, 2, 1):
        if stage_exists(s):
            return s
    return 0


def save_ckpt(
    path: str, epoch: int, stage: str,
    model: nn.Module, ema: ModelEMA,
    val_f1: float, val_acc: float, **metadata
) -> None:
    bundle = {
        "epoch": epoch, "stage": stage,
        "model": model.state_dict(), "ema": ema.state_dict(),
        "val_f1": val_f1, "val_acc": val_acc,
        "use_arcface": model._use_arcface,
        **metadata,
    }
    torch.save(bundle, path)
    sidecar = {k: v for k, v in bundle.items()
               if k not in ("model", "ema") and _is_json_serialisable(v)}
    sn = int(stage.split()[-1]) if stage.split()[-1].isdigit() else 0
    with open(stage_meta_path(sn), "w") as f:
        _json.dump(sidecar, f, indent=2)


def _is_json_serialisable(v) -> bool:
    try:
        _json.dumps(v); return True
    except (TypeError, ValueError):
        return False


def load_stage_meta(s: int) -> dict:
    p = stage_meta_path(s)
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        raw = _json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            try:
                out[k] = {int(kk): vv for kk, vv in v.items()}; continue
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out


def load_ckpt(path: str, model: nn.Module, ema: ModelEMA, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    flag = ckpt.get("use_arcface", False)
    model.use_arcface(flag); ema.shadow.use_arcface(flag)
    return ckpt


def update_bn_stats(loader: DataLoader, model: nn.Module, device: torch.device) -> None:
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

def compute_class_difficulty(
    ema_shadow: nn.Module,
    val_ldr:    DataLoader,
    device:     torch.device,
    label:      str = "Stage",
) -> Tuple[Dict[int, float], Dict[int, float]]:

    class_f1 = evaluate_per_class(
        ema_shadow, val_ldr, device, CONFIG["num_classes"]
    )
    cdws_wts = build_cdws_weights(
        class_f1, CONFIG["num_classes"],
        CONFIG["cdws_max_weight"], CONFIG["cdws_eps"]
    )
    macro  = float(np.mean(list(class_f1.values())))
    n_hard = sum(1 for f in class_f1.values() if f < 0.50)

    branch_inf = compute_branch_influence(ema_shadow, val_ldr, device, max_batches=3)

    print(
        f"[INFO] {label} class difficulty — macro F1={macro:.3f}  "
        f"hard classes (<0.50 F1): {n_hard}/{CONFIG['num_classes']}  |  "
        f"Branch influence % → "
        f"A:{branch_inf['A']:.1f}  B:{branch_inf['B']:.1f}  "
        f"C:{branch_inf['C']:.1f}  D:{branch_inf['D']:.1f}"
    )
    return class_f1, cdws_wts


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — 3-Phase Progressive Augmentation
# ══════════════════════════════════════════════════════════════════════

def run_stage1(
    model:           nn.Module,
    ema:             ModelEMA,
    loaders_by_phase: Dict[int, DataLoader],
    val_ldr:         DataLoader,
    device:          torch.device,
    best_ckpt:       str,
) -> float:
    """
    Phase 1 :   heavy aug  + high LS   → explore representation
    Phase 2 : medium aug + decay LS  → robustness consolidation
    Phase 3 :  light aug  + Focal      → discriminate hard classes
                          └─ Uses HardClassOversampledSampler

    GOAL is active throughout all three phases (model._use_goal=True, default).
    Labels (ya) are passed to the model in every batch so GOAL's dominance
    detection always has a valid target signal.
    """
    model.use_arcface(False)
    model.use_goal(True)           # Ensure GOAL is ON for Stage 1
    ema.shadow.use_goal(True)
    model.unfreeze_head("linear")
    model.freeze_head("arcface")

    ep_total = CONFIG["s1_epochs"]
    p1_end   = int(ep_total * CONFIG["s1_phase1_frac"])
    p2_end   = int(ep_total * (CONFIG["s1_phase1_frac"] + CONFIG["s1_phase2_frac"]))

    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, min_lr=CONFIG["s1_min_lr"]
        )
    
    scaler       = GradScaler("cuda")
    ls_hi        = CONFIG["s1_label_smooth_hi"]
    ls_lo        = CONFIG["s1_label_smooth_lo"]
    best_f1      = 0.0
    no_improve   = 0
    ema_reinited = [False, False]

    phase3_ldr:       Optional[DataLoader]   = None
    class_f1_phase2:  Dict[int, float]       = {}

    w = 66
    print(f"\n{'═'*w}")
    print(f"  Stage 1 — 3-Phase Progressive Augmentation  [{ep_total} epochs max]")
    print(f"{'═'*w}")
    print(f"  Phase 1: ep 1–{p1_end}         heavy aug + GOAL")
    print(f"  Phase 2: ep {p1_end+1}–{p2_end}       medium aug + GOAL")
    print(f"  Phase 3: ep {p2_end+1}–{ep_total}      light aug, Focal, class-oversample + GOAL")
    print(f"  Label smooth: {ls_hi} → {ls_lo}  |  Aux w: "
          f"{CONFIG['aux_loss_weight_init']} → {CONFIG['aux_loss_weight_final']}")
    print(f"  Oversample: {CONFIG['s1_p3_oversample']}  "
          f"power={CONFIG['s1_p3_oversample_power']}  "
          f"hard_thresh={CONFIG['s1_p3_hard_f1_thresh']}")

    for ep in range(1, ep_total + 1):
        
        if ep <= CONFIG["s1_warmup_epochs"]:
            lr_warmup = (ep / CONFIG["s1_warmup_epochs"]) * CONFIG["s1_max_lr"]
            for pg in optimizer.param_groups:
                pg['lr'] = max(lr_warmup, 1e-6)

        if   ep <= p1_end: phase = 1
        elif ep <= p2_end: phase = 2
        else:              phase = 3

        if phase == 3:
            model.use_goal(False)
            ema.shadow.use_goal(False)
        else:
            model.use_goal(True)
            ema.shadow.use_goal(True)
            
        if phase == 2 and not ema_reinited[0] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 2 (ep {ep})")
            ema_reinited[0] = True

        if phase == 3 and not ema_reinited[1] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 3 (ep {ep})")
            ema_reinited[1] = True

        if phase == 3 and phase3_ldr is None:
            print(f"\n[INFO] Phase 2→3 boundary: measuring per-class F1 for oversampling ...")
            class_f1_phase2, _ = compute_class_difficulty(
                ema.shadow, val_ldr, device, "Phase2→3"
            )
            phase3_ldr = build_phase3_loader(
                train_ds = loaders_by_phase[3].dataset,
                class_f1 = class_f1_phase2,
            )

        if   phase == 1: cur_ldr = loaders_by_phase[1]
        elif phase == 2: cur_ldr = loaders_by_phase[2]
        else:            cur_ldr = phase3_ldr

        t      = (ep - 1) / max(ep_total - 1, 1)
        ls_now = ls_hi * (1 - t) + ls_lo * t

        if phase == 3:
            crit = FocalLoss(gamma=CONFIG["s1_focal_gamma"], label_smoothing=ls_now)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=ls_now)

        tl, ta = train_one_epoch(
            model, cur_ldr, optimizer, crit, scaler, ema, device,
            accum_steps=CONFIG["s1_accum"],
            current_ep=ep,
            total_ep=ep_total,
        )

        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1        = max(f1_live, f1_ema)
        best_ep_acc       = max(acc_live, acc_ema)
        lr_now            = optimizer.param_groups[0]["lr"]
        aux_w_now         = _aux_loss_weight(ep, ep_total)
        saved             = ""

        if ep > CONFIG["s1_warmup_epochs"]:
            scheduler.step(f1_ema)

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1, _cdws = compute_class_difficulty(ema.shadow, val_ldr, device, "S1")
            save_ckpt(
                best_ckpt, ep, "Stage 1", model, ema,
                val_f1=best_ep_f1, val_acc=best_ep_acc,
                class_f1=_cf1, cdws_weights=_cdws,
                arcface_init_done=False,
                phase3_class_f1=class_f1_phase2,
            )
            saved = "  ✓"
        else:
            no_improve += 1

        print(
            f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
            f"F1 {f1_live:.3f}/{f1_ema:.3f}  Acc {acc_live:.1%}/{acc_ema:.1%} │ "
            f"LR {lr_now:.2e}  LS {ls_now:.3f}  auxW {aux_w_now:.2f} [P{phase}]{saved}"
        )

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("arcface")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR
# ══════════════════════════════════════════════════════════════════════

def run_stage2(
    model:     nn.Module,
    ema:       ModelEMA,
    train_ldr: DataLoader,
    val_ldr:   DataLoader,
    device:    torch.device,
    best_ckpt: str,
    class_f1:  Optional[Dict[int, float]] = None,
) -> float:
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)
    model.use_goal(True)           # Ensure GOAL remains ON for Stage 2
    ema.shadow.use_goal(True)
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
        steps_per_epoch=len(train_ldr),
        warmup_ep=CONFIG["s2_warmup_ep"],
        T_0=CONFIG["s2_sgdr_T0"],
        T_mult=CONFIG["s2_sgdr_Tmult"],
        eta_min_frac=CONFIG["s2_min_lr"] / CONFIG["s2_head_lr"],
    )

    sc_w     = CONFIG["supcon_weight"]; pt_w = CONFIG["proto_weight"]
    ep_total = CONFIG["s2_epochs"]
    best_f1  = 0.0; no_improve = 0

    r1 = CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"]
    r2 = r1 + CONFIG["s2_sgdr_T0"] * CONFIG["s2_sgdr_Tmult"]

    w = 66
    print(f"\n{'═'*w}")
    print(f"  Stage 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR  [{ep_total} ep]")
    print(f"{'═'*w}")
    print(f"  hLR={CONFIG['s2_head_lr']:.1e}  bLR={CONFIG['s2_back_lr']:.1e}  "
          f"SGDR T0={CONFIG['s2_sgdr_T0']} Tmult={CONFIG['s2_sgdr_Tmult']} "
          f"→ restarts ep {r1} & {r2}")
    print(f"  ArcFace K={CONFIG['subcenter_K']}  "
          f"m={CONFIG['s2_arcface_m0']}→{CONFIG['s2_arcface_m']}+Δ{CONFIG['s2_arcface_m_delta']}")
    print(f"  Losses: Focal(γ={CONFIG['s2_focal_gamma']}) + SupCon(w={sc_w}) + ProtoNCE(w={pt_w})")
    print(f"  GOAL: enabled  |  Batch: {CONFIG['bal_n_cls']} cls × {CONFIG['bal_n_spc']} spc = "
          f"{CONFIG['bal_n_cls']*CONFIG['bal_n_spc']} | Primary metric: macro-F1")

    for ep in range(1, ep_total + 1):
        warmup_done = (ep - 1) >= CONFIG["s2_margin_warmup_ep"]
        m_now       = (
            CONFIG["s2_arcface_m"] if warmup_done
            else arcface_margin(ep - 1, CONFIG["s2_arcface_m0"],
                                CONFIG["s2_arcface_m"], CONFIG["s2_margin_warmup_ep"])
        )
        arc_m  = None if warmup_done else m_now
        ramp   = min(1.0, ep / 10.0)
        sc_now = sc_w * ramp; pt_now = pt_w * ramp

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, focal, scaler=None, ema=ema,
            device=device,
            supcon=supcon, supcon_weight=sc_now,
            proto=proto,   proto_weight=pt_now,
            arc_m=arc_m, current_ep=ep, total_ep=ep_total,
            scheduler=scheduler, step_scheduler_per_batch=True
        )

        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1        = max(f1_live, f1_ema)
        best_ep_acc       = max(acc_live, acc_ema)
        head_lr           = optimizer.param_groups[0]["lr"]
        back_lr           = optimizer.param_groups[2]["lr"]
        saved             = ""

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1_s2, _cdws_s2   = compute_class_difficulty(ema.shadow, val_ldr, device, "S2")
            save_ckpt(
                best_ckpt, ep, "Stage 2", model, ema,
                val_f1=best_ep_f1, val_acc=best_ep_acc,
                class_f1=_cf1_s2, cdws_weights=_cdws_s2,
                s2_val_f1=best_ep_f1,
            )
            saved = "  ✓"
        else:
            no_improve += 1

        rf = " ↻R1" if ep == r1 else (" ↻R2" if ep == r2 else "")
        print(
            f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
            f"F1 {f1_live:.3f}/{f1_ema:.3f}  Acc {acc_live:.1%}/{acc_ema:.1%} │ "
            f"hLR {head_lr:.1e} bLR {back_lr:.1e}  m={m_now:.3f}{saved}{rf}"
        )

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("linear")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 — SAM + Greedy SWA
# ══════════════════════════════════════════════════════════════════════

def run_stage3_swa(
    model:        nn.Module,
    ema:          ModelEMA,
    train_ldr:    DataLoader,
    val_ldr:      DataLoader,
    device:       torch.device,
    best_ckpt:    str,
    prev_best_f1: float,
) -> float:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.disable = True

    model.set_dropout(CONFIG["s2_dropout"])
    model.branch_drop_prob = 0.0
    ema.shadow.branch_drop_prob = 0.0
    model.use_arcface(True); ema.shadow.use_arcface(True)

    model.use_goal(False)
    ema.shadow.use_goal(False)

    params = list(_wd_groups(model.named_parameters(), CONFIG["s3_swa_lr"]))
    sam    = SAM(params, optim.AdamW, rho=CONFIG["s3_sam_rho"],
                 lr=CONFIG["s3_swa_lr"], weight_decay=CONFIG["weight_decay"])

    focal_s3  = FocalLoss(gamma=1.0)
    supcon_s3 = SupConLoss(temperature=0.10)
    proto_s3  = ProtoNCELoss(temperature=0.10)

    swa_state    = None
    n_snap       = 0; n_rejected = 0; best_live_f1 = 0.0
    aux_w_s3     = CONFIG["s3_aux_loss_weight"]

    w = 66
    print(f"\n{'═'*w}")
    print(f"  Stage 3 — SAM + Greedy SWA  [{CONFIG['s3_epochs']} epochs]")
    print(f"{'═'*w}")
    print(f"  SAM ρ={CONFIG['s3_sam_rho']}  Cycle={CONFIG['s3_cycle_len']} ep  "
          f"Peak LR={CONFIG['s3_swa_lr']:.0e}  aux_w={aux_w_s3}")
    print(f"  GOAL: disabled (SAM flat-minima search requires unmodified gradients)")

    def _s3_margin(ep: int) -> float:
        return 0.25 + 0.05 * math.cos(math.pi * ep / CONFIG["s3_epochs"])

    for ep in range(1, CONFIG["s3_epochs"] + 1):
        cycle_ep = (ep - 1) % CONFIG["s3_cycle_len"]
        lr_now   = CONFIG["s3_swa_lr"] * (
            0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * cycle_ep / CONFIG["s3_cycle_len"]))
        )
        for pg in sam.param_groups:
            pg["lr"] = lr_now

        tl, ta = train_one_epoch_sam(
            model, train_ldr, sam, focal_s3, device,
            supcon=supcon_s3, supcon_weight=CONFIG["s3_supcon_weight"],
            proto=proto_s3,   proto_weight=CONFIG["s3_proto_weight"],
            arc_m=_s3_margin(ep),
            aux_weight=aux_w_s3,
        )

        f1_live, acc_live = evaluate(model, val_ldr, device)
        best_live_f1      = max(best_live_f1, f1_live)
        snap_info         = ""

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
                snap_info   = f"  ✗ rejected (F1 {f1_live:.3f} < {best_live_f1*0.98:.3f})"

        print(
            f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
            f"F1 {f1_live:.3f}  Acc {acc_live:.1%} │ LR {lr_now:.2e}{snap_info}"
        )

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

    save_ckpt(
        best_ckpt, CONFIG["s3_epochs"], "Stage 3",
        swa_model, ema, val_f1=f1_swa, val_acc=acc_swa,
        swa_n_snapshots=n_snap, swa_n_rejected=n_rejected,
        **({"note": note} if note else {}),
    )
    return f1_swa


# ══════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(
    model:      nn.Module,
    ema:        ModelEMA,
    test_ldr:   DataLoader,
    device:     torch.device,
    best_ckpt:  str,
) -> None:
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
            logits = (
                tta_predict(eval_model, x, CONFIG["tta_spatial"], CONFIG["tta_spectral"])
                if use_tta else eval_model(x)
            )
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
            
        p = torch.cat(preds).cpu().numpy()
        t = torch.cat(targets).cpu().numpy()
        
        results[tag] = (p, t)
        print(
            f"\n  [{tag}]  F1(macro)={f1_score(t,p,average='macro',zero_division=0):.4f}  "
            f"F1(wt)={f1_score(t,p,average='weighted',zero_division=0):.4f}  "
            f"Acc={accuracy_score(t,p):.1%}"
        )

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
    best_val, best_path = -1.0, ckpt_paths[-1]
    for p in ckpt_paths:
        if not os.path.isfile(p):
            continue
        try:
            sn   = int(os.path.basename(p).replace("best_stage", "").replace(".pth", ""))
            meta = load_stage_meta(sn)
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

def main() -> None:
    device     = CONFIG["device"]
    ckpt_s1    = stage_ckpt_path(1)
    ckpt_s2    = stage_ckpt_path(2)
    ckpt_s3    = stage_ckpt_path(3)
    done_stage = latest_completed_stage()

    labels_map = {0: "starting fresh", 1: "Stage 1 done", 2: "Stages 1–2 done", 3: "all done"}
    print(f"\n{'─'*66}")
    print(f"  Auto-resume: {labels_map.get(done_stage, f'stage {done_stage} done')}")
    print(f"  Output dir : {CONFIG['output_dir']}")
    print(f"{'─'*66}")
    print(f"[INFO] Latest completed stage: {done_stage}")

    _load_data_mmap(CONFIG["patches_data"], CONFIG["labels_path"])
    _load_wavelengths_to_gpu(CONFIG["wavelength_path"], device)

    free = torch.cuda.mem_get_info(device)[0] / 1e9

    print(f"[DATA] ✓ Dataset (mmap): {_GPU_PATCHES.size * 4 / 1e9:.1f} GB on disk  "
        f"| {free:.1f} GB VRAM free | num_workers=0")

    all_labels, train_idx, val_idx, test_idx = build_splits()
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx)//CONFIG['num_classes']}")

    model = SpectralQuadNet(
        num_classes=CONFIG["num_classes"],
        num_bands=CONFIG["num_bands"],
        dropout=CONFIG["s1_dropout"],
        wl_embed_dim=CONFIG["wl_embed_dim"],
        cfg=CONFIG,
    ).to(device)

    ema = ModelEMA(model, decay=CONFIG["ema_decay"])

    print(f"Model  : SpectralQuadNet (4-branch + GOAL + AuxHead deep supervision)")
    print(f"Params : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Device : {device}")

    if hasattr(torch, "compile"):
        print("[INFO] Applying torch.compile to heavy modules ...")

        model.branch_a = torch.compile(model.branch_a)
        model.branch_b = torch.compile(model.branch_b)
        model.branch_c = torch.compile(model.branch_c)
        # model.branch_d = torch.compile(model.branch_d)

        model.cross_interaction = torch.compile(model.cross_interaction)
        model.embed_net         = torch.compile(model.embed_net)

    else:
        print("[WARN] torch.compile unavailable")
        
        
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU]  {props.name}  |  VRAM {props.total_memory//1024**3} GB  |  "
              f"TF32={torch.backends.cuda.matmul.allow_tf32}")

    def _s1_ldr(aug_str: str) -> DataLoader:
        ds = RiceSeedDataset(train_idx, aug_strength=aug_str)
        return DataLoader(
            ds, batch_size=CONFIG["s1_batch"],
            shuffle=True, drop_last=True, num_workers=0
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
                ema.shadow, val_cd, device, "Stage 1 (recomputed)"
            )

        print("\n[RUN] Stage 2")
        tr2, va2, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"],
            balanced=True, all_labels=all_labels,
            train_aug="light", class_weights=cdws_wts_s1,
        )
        run_stage2(model, ema, tr2, va2, device, ckpt_s2, class_f1_s1)
        print("[INFO] Reloading best Stage 2 checkpoint ...")
        load_ckpt(ckpt_s2, model, ema, device)
    else:
        print("\n[SKIP] Stage 2 → loading checkpoint")
        load_ckpt(ckpt_s2, model, ema, device)

    meta_s2     = load_stage_meta(2)
    class_f1_s2 = meta_s2.get("class_f1",    {})
    cdws_wts_s2 = meta_s2.get("cdws_weights", {})
    s2_best_f1  = meta_s2.get("val_f1", meta_s2.get("s2_val_f1", meta_s2.get("val_acc", 0.0)))
    print(f"[INFO] Stage 2 → F1={s2_best_f1:.3f}")

    if hasattr(torch, "_dynamo"):
        print("[INFO] Disabling torch.compile for Stage 3 stability")
        torch._dynamo.reset()

    if done_stage < 3:
        if not cdws_wts_s2:
            print("[WARN] No cdws_weights in Stage 2 meta — falling back to Stage 1")
            cdws_wts_s2 = cdws_wts_s1

        print("\n[RUN] Stage 3 (SAM + Greedy SWA)")
        tr3, va3, _ = build_loaders(
            train_idx, val_idx, test_idx, CONFIG["s2_batch"],
            balanced=True, all_labels=all_labels,
            train_aug="light", class_weights=cdws_wts_s2,
        )
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
#  ENTRY POINT
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