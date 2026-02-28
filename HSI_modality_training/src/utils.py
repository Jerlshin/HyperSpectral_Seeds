from __future__ import annotations

import copy, json as _json, math, os, random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

from config import CONFIG
from model import ModelEMA

# ══════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int = CONFIG["seed"]) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

set_seed(CONFIG["seed"])


# ══════════════════════════════════════════════════════════════════════
#  GLOBAL DATA STATE  (3-strategy: GPU VRAM → CPU RAM → mmap)
# ══════════════════════════════════════════════════════════════════════

_GLOBAL_PATCHES: Optional[np.ndarray]   = None
_GLOBAL_LABELS:  Optional[np.ndarray]   = None
_GPU_PATCHES:    Optional[torch.Tensor] = None
_USING_MMAP:     bool = False
_DATA_ON_GPU:    bool = False


def _load_data_into_ram(patches_path: str, labels_path: str) -> None:
    global _GLOBAL_PATCHES, _GLOBAL_LABELS, _GPU_PATCHES, _USING_MMAP, _DATA_ON_GPU
    if _GLOBAL_PATCHES is not None or _GPU_PATCHES is not None:
        return

    import time
    patches_path = str(patches_path); labels_path = str(labels_path)
    if not os.path.isfile(patches_path):
        raise FileNotFoundError(f"patches file not found: {patches_path}")

    _probe        = np.load(patches_path, mmap_mode="r")
    probe_dtype   = _probe.dtype; probe_shape = _probe.shape
    float32_bytes = int(np.prod(probe_shape)) * 4
    disk_gb       = os.path.getsize(patches_path) / 1e9
    f32_gb        = float32_bytes / 1e9
    del _probe

    force_mmap    = CONFIG["force_mmap"]
    force_cpu_ram = CONFIG["force_cpu_ram"]
    print(f"[DATA] patches.npy : {disk_gb:.1f} GB on disk | dtype={probe_dtype} | float32={f32_gb:.1f} GB")

    try:
        import psutil
        avail_ram    = psutil.virtual_memory().available
        avail_ram_gb = avail_ram / 1e9
        print(f"[DATA] Free CPU RAM: {avail_ram_gb:.1f} GB")
    except ImportError:
        avail_ram = None; avail_ram_gb = -1.0

    device = CONFIG["device"]
    if device.type == "cuda" and not force_mmap and not force_cpu_ram:
        torch.cuda.synchronize()
        free_vram = torch.cuda.mem_get_info(device)[0]
        print(f"[DATA] Free GPU VRAM: {free_vram/1e9:.1f} GB")
    else:
        free_vram = 0

    # Strategy 0: GPU VRAM
    if (device.type == "cuda" and free_vram >= float32_bytes * 1.10
            and not force_mmap and not force_cpu_ram):
        print(f"[DATA] ► Strategy 0: Loading {f32_gb:.1f} GB → GPU VRAM ...")
        t0       = time.time()
        mmap_arr = np.load(patches_path, mmap_mode="r")
        gpu_tensor = torch.empty(probe_shape, dtype=torch.float32, device=device)
        for i in range(0, probe_shape[0], 512):
            block = torch.from_numpy(mmap_arr[i:i+512].astype(np.float32)).clone()
            gpu_tensor[i:i+512].copy_(block, non_blocking=True)
        del mmap_arr, block
        torch.cuda.synchronize()
        _GPU_PATCHES   = gpu_tensor; _GLOBAL_LABELS = np.load(labels_path)
        _DATA_ON_GPU   = True;       _USING_MMAP    = False
        print(f"[DATA] ✓ GPU load complete in {time.time()-t0:.1f}s  "
              f"({_GPU_PATCHES.nelement()*4/1e9:.1f} GB | shape={tuple(_GPU_PATCHES.shape)})")
        return

    # Strategy A: CPU RAM (chunked — never OOM)
    if (avail_ram is not None and avail_ram >= float32_bytes * 1.20 and not force_mmap):
        print(f"[DATA] ► Strategy A: Chunked CPU RAM load ...")
        t0 = time.time()
        try:
            mmap_arr = np.load(patches_path, mmap_mode="r")
            out      = np.empty(probe_shape, dtype=np.float32)
            for i in range(0, probe_shape[0], 512):
                out[i:i+512] = mmap_arr[i:i+512].astype(np.float32)
            del mmap_arr
            _GLOBAL_PATCHES = out; _GLOBAL_LABELS = np.load(labels_path)
            _USING_MMAP = False;   _DATA_ON_GPU   = False
            print(f"[DATA] ✓ RAM load complete in {time.time()-t0:.1f}s  "
                  f"({_GLOBAL_PATCHES.nbytes/1e9:.1f} GB)")
            return
        except MemoryError:
            print("[DATA] MemoryError — falling back to mmap")
            _GLOBAL_PATCHES = None

    # Strategy B: mmap + BlockSortedSampler
    shortage = f32_gb - avail_ram_gb if avail_ram_gb >= 0 else 0
    print(f"[DATA] ► Strategy B: mmap fallback  (RAM short by ~{shortage:.1f} GB)")
    _GLOBAL_PATCHES = np.load(patches_path, mmap_mode="r")
    _GLOBAL_LABELS  = np.load(labels_path)
    _USING_MMAP     = True; _DATA_ON_GPU = False
    print(f"[DATA] ✓ mmap ready  shape={_GLOBAL_PATCHES.shape}  dtype={_GLOBAL_PATCHES.dtype}")


# ══════════════════════════════════════════════════════════════════════
#  DATASET
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
                 aug_strength: str = "none",
                 max_cutout_bands: int = CONFIG["aug_max_cutout_bands"],
                 noise_std: float      = CONFIG["aug_noise_std"]):
        global _GLOBAL_PATCHES, _GLOBAL_LABELS, _GPU_PATCHES, _DATA_ON_GPU
        if _GLOBAL_PATCHES is None and _GPU_PATCHES is None:
            _load_data_into_ram(patches_path, labels_path)
        self.patches          = _GPU_PATCHES if _DATA_ON_GPU else _GLOBAL_PATCHES
        self.labels           = _GLOBAL_LABELS
        self.on_gpu           = _DATA_ON_GPU
        self.indices          = indices
        self.aug_strength     = aug_strength
        self.max_cutout_bands = max_cutout_bands
        self.noise_std        = noise_std

    def __len__(self): return len(self.indices)

    def _probs(self): return self._PROFILES.get(str(self.aug_strength))

    def _band_dropout(self, x):
        return x * (torch.rand(x.shape[0], device=x.device) > 0.04).float().view(-1, 1, 1)

    def _band_cutout(self, x):
        x = x.clone(); nb = x.shape[0]
        cut = torch.randint(1, max(2, self.max_cutout_bands), (1,)).item()
        st  = torch.randint(0, max(1, nb - cut), (1,)).item()
        x[st:st+cut] = 0.0; return x

    def _spectral_noise(self, x):
        mask = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        return x + (torch.randn_like(x) * self.noise_std) * mask

    def _spectral_warp(self, x):
        C, H, W = x.shape
        scale = 1.0 + random.uniform(-0.10, 0.10); new_C = max(1, int(C * scale))
        if new_C == C: return x
        xp     = x.permute(1, 2, 0).reshape(-1, 1, C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s = (new_C - C) // 2; warped = warped[:, :, s:s+C]
        else:
            lo = (C - new_C) // 2; warped = F.pad(warped, (lo, C - new_C - lo))
        return warped.reshape(H, W, C).permute(2, 0, 1)

    def _spectral_shift(self, x): return torch.roll(x, random.randint(-8, 8), dims=0)

    def _mult_noise(self, x):
        mask         = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        noise_factor = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device) * 0.05
        return x * noise_factor * mask

    def _spatial(self, x):
        if torch.rand(1) < 0.5: x = torch.flip(x, [2])
        if torch.rand(1) < 0.5: x = torch.flip(x, [1])
        return torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])

    def __getitem__(self, idx):
        ri = self.indices[idx]
        if self.on_gpu:
            patch = self.patches[ri].clone()
        else:
            patch = torch.from_numpy(
                self.patches[ri].astype(np.float32, copy=False).copy())
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
    def __init__(self, train_labels,
                 n_cls: int = CONFIG["bal_n_cls"],
                 n_spc: int = CONFIG["bal_n_spc"],
                 class_weights: Optional[Dict[int, float]] = None):
        self.n_cls = n_cls; self.n_spc = n_spc
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


class BlockSortedSampler(Sampler):
    """Converts random mmap access → block-sequential I/O to reduce page-faults."""
    def __init__(self, indices: np.ndarray,
                 block_size: int = CONFIG["mmap_block_size"]):
        self.indices = np.asarray(indices); self.block_size = block_size

    def __iter__(self):
        sorted_idx = self.indices[np.argsort(self.indices)]
        blocks     = [sorted_idx[s:s+self.block_size]
                      for s in range(0, len(sorted_idx), self.block_size)]
        np.random.shuffle(blocks)
        return iter(np.concatenate(blocks).tolist())

    def __len__(self): return len(self.indices)


def build_cdws_weights(class_f1: Dict[int, float], num_classes: int,
                       max_w: float = CONFIG["cdws_max_weight"],
                       eps: float   = CONFIG["cdws_eps"]) -> Dict[int, float]:
    raw  = {c: min(1.0 / (class_f1.get(c, 0.0) + eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w / mean for c, w in raw.items()}


# ══════════════════════════════════════════════════════════════════════
#  BATCH AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def _cutmix(x, y, alpha):
    lam       = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx       = torch.randperm(B, device=x.device)
    r = math.sqrt(1.0 - lam)
    ch, cw    = int(H * r), int(W * r)
    cx, cy    = random.randint(0, W), random.randint(0, H)
    x1 = max(cx - cw // 2, 0); x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0); y2 = min(cy + ch // 2, H)
    xm = x.clone(); xm[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    return xm, y, y[idx], 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

def mixed_aug(x, y, alpha: float = CONFIG["s1_mixup"]):
    return (_mixup if torch.rand(1) < 0.5 else _cutmix)(x, y, alpha)

def mixed_loss(crit, logits, ya, yb, lam):
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)


# ══════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss with optional label smoothing.
    Combining both: soft targets from LS, then apply focal modulation (1-pt)^γ.
    """
    def __init__(self, gamma: float = CONFIG["s2_focal_gamma"],
                 label_smoothing: float = 0.0):
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
    """Supervised Contrastive Loss. Expects L2-normalised features."""
    def __init__(self, temperature: float = CONFIG["supcon_temp"]):
        super().__init__(); self.temperature = temperature

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
    """Class-mean prototype contrastive CE."""
    def __init__(self, temperature: float = CONFIG["proto_temp"]):
        super().__init__(); self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()
        if len(classes) < 2: return (features * 0).sum()
        protos = F.normalize(
            torch.stack([features[labels == c].mean(0) for c in classes]), dim=1)
        sim   = torch.mm(features, protos.T) / self.temperature
        c2l   = {c.item(): i for i, c in enumerate(classes)}
        local = torch.tensor([c2l[y.item()] for y in labels],
                              dtype=torch.long, device=features.device)
        return F.cross_entropy(sim, local)


# ══════════════════════════════════════════════════════════════════════
#  SAM  (Sharpness-Aware Minimisation)
# ══════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls,
                 rho: float = CONFIG["s3_sam_rho"], **kwargs):
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
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits():
    labels  = np.load(CONFIG["labels_path"])
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=CONFIG["val_split"],
                               stratify=labels, random_state=CONFIG["split_seed"])
    va, te  = train_test_split(tmp, test_size=CONFIG["test_split"],
                               stratify=labels[tmp], random_state=CONFIG["split_seed"])
    return labels, tr, va, te


def build_loaders(train_idx, val_idx, test_idx, batch_train: int,
                  balanced: bool = False,
                  all_labels=None,
                  train_aug: str = "none",
                  class_weights: Optional[Dict[int, float]] = None):
    nw = 0 if _DATA_ON_GPU else CONFIG["num_workers"]
    pf = CONFIG["prefetch_factor"]
    kw = (dict(num_workers=0, pin_memory=False) if _DATA_ON_GPU
          else dict(num_workers=nw, pin_memory=True,
                    persistent_workers=True, prefetch_factor=pf))

    ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                         train_idx, aug_strength=train_aug)

    if balanced and all_labels is not None:
        samp   = ClassBalancedBatchSampler(all_labels[train_idx],
                                           CONFIG["bal_n_cls"], CONFIG["bal_n_spc"],
                                           class_weights=class_weights)
        tr_ldr = DataLoader(ds, batch_sampler=samp, drop_last=False, **kw)
    elif _USING_MMAP:
        bss    = BlockSortedSampler(train_idx, block_size=CONFIG["mmap_block_size"])
        tr_ldr = DataLoader(ds, batch_size=batch_train, sampler=bss, drop_last=True, **kw)
    else:
        tr_ldr = DataLoader(ds, batch_size=batch_train, shuffle=True, drop_last=True, **kw)

    eval_bs = CONFIG["eval_batch_size"]
    va_ldr  = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], val_idx),
        batch_size=eval_bs, shuffle=False, **kw)
    te_ldr  = DataLoader(
        RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], test_idx),
        batch_size=eval_bs, shuffle=False,
        **(dict(num_workers=0, pin_memory=False) if _DATA_ON_GPU
           else dict(num_workers=CONFIG["test_num_workers"], pin_memory=True,
                     persistent_workers=True, prefetch_factor=CONFIG["test_prefetch_factor"])))
    return tr_ldr, va_ldr, te_ldr


# ══════════════════════════════════════════════════════════════════════
#  OPTIMISERS & SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def _wd_groups(named_params, lr: float):
    wd, no_wd = [], []
    for n, p in named_params:
        if not p.requires_grad: continue
        (no_wd if (p.ndim == 1 or n.endswith(".bias")) else wd).append(p)
    return [{"params": wd,    "lr": lr, "weight_decay": CONFIG["weight_decay"]},
            {"params": no_wd, "lr": lr, "weight_decay": 0.0}]

def build_optimizer_s1(model: nn.Module, lr: float):
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))

def build_optimizer_s2(model: nn.Module,
                       head_lr: float = CONFIG["s2_head_lr"],
                       back_lr: float = CONFIG["s2_back_lr"]):
    hp, bp = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (hp if n.startswith("arcface_head") else bp).append((n, p))
    return optim.AdamW(_wd_groups(hp, head_lr) + _wd_groups(bp, back_lr))

def build_optimizer_s3(model: nn.Module, lr: float = CONFIG["s3_swa_lr"]):
    return optim.AdamW(_wd_groups(model.named_parameters(), lr))

def sgdr_scheduler(optimizer,
                   warmup_ep: int    = CONFIG["s2_warmup_ep"],
                   T_0: int          = CONFIG["s2_sgdr_T0"],
                   T_mult: int       = CONFIG["s2_sgdr_Tmult"],
                   eta_min_frac: float = 1e-3) -> optim.lr_scheduler.LambdaLR:
    def _l(ep):
        if ep < warmup_ep:
            return max(ep / max(warmup_ep, 1), 1e-6)
        t = ep - warmup_ep; clen = T_0; elapsed = 0
        while t >= elapsed + clen:
            elapsed += clen; clen = max(int(clen * T_mult), 1)
        ratio = (t - elapsed) / max(clen, 1)
        return eta_min_frac + 0.5 * (1 - eta_min_frac) * (1 + math.cos(math.pi * ratio))
    return optim.lr_scheduler.LambdaLR(optimizer, _l)

def arcface_margin(ep: int,
                   m0: float       = CONFIG["s2_arcface_m0"],
                   m_target: float = CONFIG["s2_arcface_m"],
                   warmup_ep: int  = CONFIG["s2_margin_warmup_ep"]) -> float:
    if ep >= warmup_ep: return m_target
    return m0 + (m_target - m0) * 0.5 * (1 - math.cos(math.pi * ep / max(warmup_ep, 1)))


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler, ema, device,
                    scheduler=None,
                    use_mixup: bool   = True,
                    mixup_alpha: float = CONFIG["s1_mixup"],
                    supcon=None,       supcon_weight: float = 0.0,
                    proto=None,        proto_weight: float  = 0.0,
                    accum_steps: int  = CONFIG["s1_accum"],
                    arc_m=None):
    model.train()
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_amp = (supcon is None) and (scaler is not None)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_in, ya, yb, lam = mixed_aug(x, y, mixup_alpha) if use_mixup else (x, y, y, 1.0)

        with autocast(device_type=device.type, enabled=use_amp):
            if supcon is not None:
                logits, emb = model(x_in, ya, return_embed=True, arc_m=arc_m)
                cls_l = criterion(logits, ya)
                sc_l  = supcon(emb, ya)
                pt_l  = proto(emb, ya) if proto is not None else 0.0
                loss  = ((1 - supcon_weight - proto_weight) * cls_l
                         + supcon_weight * sc_l + proto_weight * pt_l)
            else:
                arc_labels = (ya if model._use_arcface and not use_mixup else None)
                logits     = model(x_in, labels=arc_labels, arc_m=arc_m)
                loss       = mixed_loss(criterion, logits, ya, yb, lam)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True); continue

        (scaler.scale(loss / accum_steps).backward() if use_amp
         else (loss / accum_steps).backward())

        if (step + 1) % accum_steps == 0:
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

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


def train_one_epoch_sam(model, loader, sam_opt, criterion, device,
                        supcon=None,  supcon_weight: float = CONFIG["s3_supcon_weight"],
                        proto=None,   proto_weight: float  = CONFIG["s3_proto_weight"],
                        arc_m=None):
    torch.set_default_dtype(torch.float32)
    model.train()
    total_loss = total_acc = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        def _compute_loss(lg, em):
            if supcon is not None:
                return ((1 - supcon_weight - proto_weight) * criterion(lg, y)
                        + supcon_weight * (supcon(em, y) if supcon else 0)
                        + proto_weight  * (proto(em, y)  if proto  else 0))
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
    return total_loss / n, total_acc / n


@torch.no_grad()
def _run_eval(model, loader, device):
    """Shared evaluation core — returns (preds, targets) numpy arrays."""
    model.eval(); preds, targets = [], []
    with autocast(device_type=device.type, enabled=False):
        for x, y in loader:
            x      = x.to(device, non_blocking=True)
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


def evaluate_per_class(model, loader, device,
                       num_classes: int = CONFIG["num_classes"]) -> Dict[int, float]:
    p, t    = _run_eval(model, loader, device)
    f1_arr  = f1_score(t, p, average=None, zero_division=0,
                       labels=list(range(num_classes)))
    return {i: float(v) for i, v in enumerate(f1_arr)}


# ══════════════════════════════════════════════════════════════════════
#  CLASS DIFFICULTY
# ══════════════════════════════════════════════════════════════════════

def compute_class_difficulty(ema_shadow: nn.Module, val_ldr, device,
                             label: str = "Stage") -> Tuple[Dict[int,float], Dict[int,float]]:
    class_f1 = evaluate_per_class(ema_shadow, val_ldr, device, CONFIG["num_classes"])
    cdws_wts = build_cdws_weights(class_f1, CONFIG["num_classes"],
                                  CONFIG["cdws_max_weight"], CONFIG["cdws_eps"])
    macro    = float(np.mean(list(class_f1.values())))
    n_hard   = sum(1 for f in class_f1.values() if f < 0.50)
    print(f"[INFO] {label} class difficulty — macro F1={macro:.3f}  "
          f"hard classes (<0.50 F1): {n_hard}/{CONFIG['num_classes']}")
    return class_f1, cdws_wts


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

def _is_json_serialisable(v) -> bool:
    try: _json.dumps(v); return True
    except (TypeError, ValueError): return False

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

def update_bn_stats(loader, model: nn.Module, device) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats(); m.momentum = None
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device, non_blocking=True))
    model.eval()