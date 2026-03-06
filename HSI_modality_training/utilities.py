from __future__ import annotations

import copy
import json as _json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Online softmax is disabled.*")

CONFIG: dict = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
}

# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def _load_data_to_gpu(patches_path: str, labels_path: str) -> None:
    global _GPU_PATCHES, _GLOBAL_LABELS

    if _GPU_PATCHES is not None:
        return

    device = CONFIG["device"]
    assert device.type == "cuda", "GPU mode required."

    print("[DATA] Loading full dataset into GPU VRAM...")
    mmap_arr   = np.load(patches_path, mmap_mode="r")
    shape      = mmap_arr.shape
    gpu_tensor = torch.empty(shape, dtype=torch.float32, device=device)

    for i in range(0, shape[0], 512):
        block = torch.from_numpy(
            mmap_arr[i:i+512].astype(np.float32)
        ).to(device, non_blocking=True)
        gpu_tensor[i:i+512].copy_(block)

    del mmap_arr
    torch.cuda.synchronize()

    _GPU_PATCHES  = gpu_tensor
    _GLOBAL_LABELS = np.load(labels_path)
    print(f"[DATA] ✓ Loaded {_GPU_PATCHES.nelement()*4/1e9:.1f} GB into VRAM")


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
#  SAMPLERS
# ══════════════════════════════════════════════════════════════════════

class ClassBalancedBatchSampler(Sampler):
    """Draws n_cls classes per batch, n_spc samples per class, with optional CDWS weighting."""

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
    """
    Stage 1 · Phase 3 — Class-Specific Oversampling Sampler.

    Hard classes (low per-class F1 from Phase 2 evaluation) are assigned
    higher sampling weight via inverse-F1 re-weighting.  The weight is
    raised to ``oversample_power`` to control the aggressiveness of the
    boost, and capped at ``max_weight`` to prevent pathological imbalance.

    Args:
        labels:            Integer class label for every training sample.
        class_f1:          Per-class F1 scores from Phase 2 EMA evaluation.
        num_samples:       Total samples to draw per epoch (with replacement).
        oversample_power:  Exponent on inverse-F1 weight (1.0 = full inverse).
        max_weight:        Ceiling on any single class multiplier.
        hard_f1_thresh:    Classes below this F1 are logged as "hard".
        eps:               Smoothing term in the denominator (avoids div-by-zero).
    """

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

        # ── Build per-class weights ────────────────────────────────────
        num_classes   = int(np.max(labels)) + 1
        raw_weights: Dict[int, float] = {}
        for c in range(num_classes):
            f1 = float(class_f1.get(c, 0.0))
            w  = (1.0 / (f1 + eps)) ** oversample_power
            raw_weights[c] = min(w, max_weight)

        # Normalise so the mean weight stays at 1.0 (no overall rate change)
        mean_w = float(np.mean(list(raw_weights.values())))
        norm_weights = {c: w / mean_w for c, w in raw_weights.items()}

        # ── Assign per-sample weights ──────────────────────────────────
        sample_weights         = np.array(
            [norm_weights.get(int(lbl), 1.0) for lbl in labels], dtype=np.float32
        )
        self._weights          = torch.from_numpy(sample_weights)

        # ── Diagnostics ───────────────────────────────────────────────
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
#  LOSSES
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss with optional label smoothing.
    Soft targets from LS, then focal modulation (1-pt)^γ preserves
    regularisation while sharpening focus on hard examples.
    """
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
    """Supervised Contrastive Loss. Expects L2-normalised features."""

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
    """Class-mean prototype contrastive CE."""

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
#  DATA SPLITS & LOADERS
# ══════════════════════════════════════════════════════════════════════

def build_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels  = np.load(CONFIG["labels_path"])
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
    from training import RiceSeedDataset
    
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
    """
    Build the Phase-3 DataLoader with hard-class oversampling.

    Uses HardClassOversampledSampler to give hard classes (low F1 from
    Phase 2) higher sampling probability.  Falls back to standard
    shuffled loader if oversampling is disabled in CONFIG.
    """
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
    warmup_ep:   int   = 5,
    T_0:         int   = 10,
    T_mult:      int   = 2,
    eta_min_frac: float = 1e-3,
) -> optim.lr_scheduler.LambdaLR:
    def _l(ep: int) -> float:
        if ep < warmup_ep:
            return max(ep / max(warmup_ep, 1), 1e-6)
        t = ep - warmup_ep; clen = T_0; elapsed = 0
        while t >= elapsed + clen:
            elapsed += clen; clen = max(int(clen * T_mult), 1)
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
    """
    Linearly decay the auxiliary branch loss weight from
    ``aux_loss_weight_init`` to ``aux_loss_weight_final`` over training.

    The higher weight early in training forces each branch to be
    independently discriminative; as training matures the weight decays
    so the main fused head dominates.
    """
    progress = current_ep / max(total_ep, 1)
    return max(
        CONFIG["aux_loss_weight_final"],
        CONFIG["aux_loss_weight_init"] * (1.0 - progress),
    )


def _compute_aux_loss(
    criterion: nn.Module,
    out: dict,
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
    use_mixup: bool,
) -> torch.Tensor:
    """
    Compute the summed auxiliary head loss across all four branches.

    Handles both standard CE and Mixup-interpolated targets.
    Branches A, B, C, D each contribute equally.
    """
    
    from training import mixed_loss
    
    aux_keys = ["aux_a", "aux_b", "aux_c", "aux_d"]
    total    = torch.zeros((), device=ya.device)
    for k in aux_keys:
        if k not in out:
            continue
        if use_mixup:
            total = total + mixed_loss(criterion, out[k], ya, yb, lam)
        else:
            total = total + criterion(out[k], ya)
    return total



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
#  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(
    model:      nn.Module,
    ema:        ModelEMA,
    test_ldr:   DataLoader,
    device:     torch.device,
    best_ckpt:  str,
) -> None:
    
    from training import tta_predict
    
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
            preds.append(logits.argmax(1).cpu()); targets.append(y)
        p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
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
    """Select checkpoint with highest val_f1 across all stages."""
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
