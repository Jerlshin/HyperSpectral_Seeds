import os
import math
import random
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    "patches_data": "./dataset/patches.npy",
    "labels_path":  "./dataset/labels.npy",
    "output_dir":   "./output_v2/",

    "num_epochs":   150,
    "batch_size":   64,
    "patience":     30,       # Early stopping patience

    "num_bands":    256,
    "num_classes":  90,

    # Loss
    "label_smoothing": 0.1,

    # Mixup
    "mixup_alpha":  0.4,

    # Optimiser
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "max_lr":       1e-3,     # OneCycleLR peak

    # Regularisation
    "dropout":      0.4,
    "grad_clip":    1.0,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":   42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.cuda.empty_cache()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CONFIG["seed"])

# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """
    Memory-mapped dataset loader with spectral + spatial augmentations.

    Augmentations (training only):
      - Spectral band dropout   : randomly zero individual bands
      - Spectral contiguous cut : zero a random contiguous window of bands
      - Spatial flip / rot90    : rotation-invariant spatial features
    """

    def __init__(
        self,
        patches_path: str,
        labels_path: str,
        indices: np.ndarray,
        augment: bool = False,
        band_drop_prob: float = 0.05,
        max_cutout_bands: int = 24,
    ):
        self.patches = np.load(patches_path, mmap_mode="r")
        self.labels  = np.load(labels_path)
        self.indices = indices
        self.augment         = augment
        self.band_drop_prob  = band_drop_prob
        self.max_cutout_bands = max_cutout_bands

    def __len__(self) -> int:
        return len(self.indices)

    # ── spectral augmentations ──────────────────────────────────────

    def _band_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero individual spectral bands."""
        mask = (torch.rand(x.shape[0]) > self.band_drop_prob).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x: torch.Tensor) -> torch.Tensor:
        """Zero a random contiguous window of spectral bands."""
        x = x.clone()
        nb  = x.shape[0]
        cut = torch.randint(1, max(2, self.max_cutout_bands), (1,)).item()
        # Ensure we do not exceed bounds
        start = torch.randint(0, max(1, nb - cut), (1,)).item()
        x[start: start + cut] = 0.0
        return x

    # ── spatial augmentation ────────────────────────────────────────

    def _spatial_augment(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [1])
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, [1, 2])
        return x

    # ── item ────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        # .copy() required: mmap arrays are read-only
        patch = torch.from_numpy(self.patches[real_idx].copy()).float()
        # FIX: was "dtype=torch.long))" — extra closing parenthesis (syntax error)
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)

        if self.augment:
            patch = self._band_dropout(patch)
            patch = self._band_cutout(patch)
            patch = self._spatial_augment(patch)

        return patch, label


# ═══════════════════════════════════════════════════════════════════
#  MIXUP UTILITY
# ═══════════════════════════════════════════════════════════════════

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Returns mixed inputs and a (y_a, y_b, lam) tuple.
    Loss = lam * CE(pred, y_a) + (1−lam) * CE(pred, y_b)

    Mixup is especially effective for fine-grained classification (90 classes
    that look very similar) because it forces the model to learn smooth
    decision boundaries rather than sharp ones.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    B   = x.size(0)
    idx = torch.randperm(B, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════
#  MODEL BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════

class SpectralSEBlock(nn.Module):
    """
    Squeeze-and-Excitation on the spectral channel dimension.
    Learns which of the 256 bands are most discriminative per sample.
    This is the key attention module for HSI — it adaptively reweights bands.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, _, _ = x.shape
        # Global average pooling over spatial dims
        y = x.mean(dim=[2, 3])            # (B, C)
        w = self.fc(y).view(B, C, 1, 1)   # (B, C, 1, 1)
        return x * w


class ResBlock1D(nn.Module):
    """
    1D Residual block for spectral sequence processing.
    Used in the spectral profile branch (Branch A).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.skip = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel + spatial attention to spatial feature maps.
    """

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(c // r, 8)
        # Channel attention
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_max = nn.AdaptiveMaxPool2d(1)
        self.ch_fc  = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, c, 1, bias=False),
        )
        # Spatial attention
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel
        ca = torch.sigmoid(
            self.ch_fc(self.ch_avg(x)) + self.ch_fc(self.ch_max(x))
        )
        x  = x * ca
        # Spatial
        sp = self.sp_conv(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        )
        return x * sp


class ResBlock2D(nn.Module):
    """
    2D Residual bottleneck block for spatial feature extraction.
    Used in the spatial CNN branch (Branch B).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        mid = out_ch // 2
        self.c1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)

        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.c1(x)))
        h = F.gelu(self.n2(self.c2(h)))
        h = self.n3(self.c3(h))
        return F.gelu(h + self.skip(x))


# ═══════════════════════════════════════════════════════════════════
#  MAIN MODEL : SpectralDualNet
# ═══════════════════════════════════════════════════════════════════

class SpectralDualNet(nn.Module):
    """
    Dual-branch architecture for HSI seed variety classification.

    DESIGN RATIONALE:
      For seed classification, the SPECTRAL PROFILE is the primary
      discriminant — each rice variety has a unique reflectance curve.
      Spatial texture is a secondary cue.  This model reflects that:

      Branch A (Spectral, 70% of fused features):
        Computes the masked mean spectrum (excluding background zeros)
        and processes it with a multi-scale 1D CNN.  Uses three kernel
        sizes (k=3, 7, 15) to capture both sharp absorption features
        and broad spectral shape.  Global avg + max pooling at the end.

      Branch B (Spatial, 30% of fused features):
        Reduces 256 → 16 spectral bands via learned 1×1 conv, then
        applies 2D residual blocks with CBAM attention to extract
        spatial texture.  Lighter than Branch A intentionally.

    The pre-processing SE block learns per-sample band weights, acting
    like an adaptive PCA that highlights discriminative bands.
    """

    def __init__(
        self,
        num_classes: int = 90,
        num_bands:   int = 256,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self.num_bands = num_bands

        # ── 0. Pre-processing: Spectral Channel Attention ───────────
        self.se = SpectralSEBlock(num_bands, reduction=16)

        # ── Branch A: Multi-Scale 1D Spectral CNN ───────────────────
        #   Input: masked mean spectrum (B, 1, 256)
        #   Runs three parallel 1D CNN towers with different kernels,
        #   then fuses for a rich spectral representation.
        self.spec_small  = self._make_spec_tower(1,  64, kernel=3)
        self.spec_medium = self._make_spec_tower(1,  64, kernel=7)
        self.spec_large  = self._make_spec_tower(1,  64, kernel=15)
        # After tower: each → (B, 64, 256) → global pool → (B, 64)
        # Concat 3 towers × 2 pools (avg+max) = 384 → project → 256
        self.spec_proj = nn.Sequential(
            nn.Linear(64 * 3 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Branch B: Spatial 2D CNN ─────────────────────────────────
        #   Reduce spectral dim first (like a learned PCA) then 2D CNN
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 16, 1, bias=False),
            nn.GroupNorm(8, 16),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            ResBlock2D(16,  32, stride=2), CBAM(32),   # 64→32
            ResBlock2D(32,  64, stride=2), CBAM(64),   # 32→16
            ResBlock2D(64, 128, stride=2), CBAM(128),  # 16→8
        )
        self.gap_spatial = nn.AdaptiveAvgPool2d(1)
        # Output: (B, 128)

        # ── Fusion + Classifier ─────────────────────────────────────
        # Branch A: 256, Branch B: 128 → total 384
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _make_spec_tower(in_ch: int, out_ch: int, kernel: int) -> nn.Module:
        """Three stacked 1D ResBlocks, same kernel size throughout."""
        mid = out_ch // 2
        return nn.Sequential(
            ResBlock1D(in_ch,  mid,     kernel),
            ResBlock1D(mid,    out_ch,  kernel),
            ResBlock1D(out_ch, out_ch,  kernel),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _masked_mean_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean spectrum, ignoring zero-padded background pixels.

        After SNV normalisation, background pixels were forced to 0 during
        patch extraction (mask applied per-seed).  Including them in the mean
        would bias the spectral signature toward 0.

        x: (B, 256, H, W)
        returns: (B, 1, 256)  — ready for 1D conv  (sequence length = 256)
        """
        # Pixel is "background" if its L1 norm across bands is near zero
        mask  = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()  # (B,1,H,W)
        count = mask.sum(dim=(2, 3)).clamp(min=1.0)                 # (B, 1)
        spec  = (x * mask).sum(dim=(2, 3)) / count                  # (B, 256)
        return spec.unsqueeze(1)                                      # (B, 1, 256)

    def _pool1d(self, feat: torch.Tensor) -> torch.Tensor:
        """Global avg + max pool over the spectral sequence dimension."""
        avg = feat.mean(dim=2)       # (B, C)
        mx  = feat.max(dim=2).values # (B, C)
        return torch.cat([avg, mx], dim=1)   # (B, 2C)

    # ── forward ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 256, 64, 64)

        # Spectral channel attention
        x = self.se(x)

        # ─ Branch A: Spectral ────────────────────────────────────────
        spec = self._masked_mean_spectrum(x)   # (B, 1, 256)

        f_small  = self._pool1d(self.spec_small(spec))   # (B, 128)
        f_medium = self._pool1d(self.spec_medium(spec))  # (B, 128)
        f_large  = self._pool1d(self.spec_large(spec))   # (B, 128)

        feat_a = self.spec_proj(
            torch.cat([f_small, f_medium, f_large], dim=1)  # (B, 384)
        )  # → (B, 256)

        # ─ Branch B: Spatial ─────────────────────────────────────────
        h = self.band_reduce(x)              # (B, 16, 64, 64)
        h = self.spatial(h)                  # (B, 128, 8, 8)
        feat_b = self.gap_spatial(h).flatten(1)  # (B, 128)

        # ─ Fuse & classify ───────────────────────────────────────────
        fused  = torch.cat([feat_a, feat_b], dim=1)  # (B, 384)
        logits = self.head(fused)                     # (B, 90)
        return logits


# ═══════════════════════════════════════════════════════════════════
#  DATA SETUP
# ═══════════════════════════════════════════════════════════════════

labels  = np.load(CONFIG["labels_path"])
indices = np.arange(len(labels))

train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
)

print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
print(f"Samples/class (train): ~{len(train_idx) // CONFIG['num_classes']}")

train_ds = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], train_idx,
    augment=True, band_drop_prob=0.05, max_cutout_bands=24
)
val_ds = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], val_idx
)
test_ds = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], test_idx
)

train_loader = DataLoader(
    train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=6, pin_memory=True, persistent_workers=True,
    prefetch_factor=2
)
val_loader = DataLoader(
    val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=2, pin_memory=True
)

# ═══════════════════════════════════════════════════════════════════
#  MODEL, OPTIMISER, SCHEDULER
# ═══════════════════════════════════════════════════════════════════

device = CONFIG["device"]
model  = SpectralDualNet(
    num_classes=CONFIG["num_classes"],
    num_bands=CONFIG["num_bands"],
    dropout=CONFIG["dropout"],
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {n_params / 1e6:.2f}M")

# Separate weight-decay groups (no WD on bias / norm params)
wd_params, no_wd_params = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim == 1 or name.endswith(".bias"):
        no_wd_params.append(param)
    else:
        wd_params.append(param)

optimizer = optim.AdamW(
    [
        {"params": wd_params,    "weight_decay": CONFIG["weight_decay"]},
        {"params": no_wd_params, "weight_decay": 0.0},
    ],
    lr=CONFIG["lr"],
)

# CrossEntropyLoss with label smoothing — stable replacement for ArcFace.
# label_smoothing=0.1 prevents overconfident predictions and improves
# calibration on fine-grained tasks (proven effective for 90-class problems).
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

scaler = GradScaler()

# OneCycleLR: ramps up to max_lr then cosine decays.
# pct_start=0.15 → 15% of training is the warm-up phase.
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG["max_lr"],
    epochs=CONFIG["num_epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=0.15,
    div_factor=25,          # start lr = max_lr / 25
    final_div_factor=1e4,   # end   lr = max_lr / 1e4
    anneal_strategy="cos",
)

# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Mixup — interpolates both inputs and labels
        x_mix, y_a, y_b, lam = mixup_batch(x, y, alpha=CONFIG["mixup_alpha"])

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            logits = model(x_mix)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()

        # Gradient clipping prevents instability caused by outlier batches
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        # Accuracy measured on un-mixed logits w.r.t original labels
        # (informational only during mixup training)
        with torch.no_grad():
            total_acc += (logits.argmax(1) == y).float().mean().item()

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)   # No mixup at evaluation
        preds.append(logits.argmax(1).cpu())
        targets.append(y)

    preds   = torch.cat(preds)
    targets = torch.cat(targets)

    acc = accuracy_score(targets, preds)
    f1  = f1_score(targets, preds, average="macro", zero_division=0)
    return f1, acc


# ═══════════════════════════════════════════════════════════════════
#  MAIN TRAINING EXECUTION
# ═══════════════════════════════════════════════════════════════════

best_val_acc = 0.0
best_val_f1  = 0.0
epochs_no_improve = 0
best_ckpt = os.path.join(CONFIG["output_dir"], "best_model.pth")

print("\nStarting Training …")
print(f"Device: {device}  |  Epochs: {CONFIG['num_epochs']}  |  "
      f"Batch: {CONFIG['batch_size']}  |  Loss: CrossEntropy+LS"
      f"(label_smoothing={CONFIG['label_smoothing']})\n")

for ep in range(1, CONFIG["num_epochs"] + 1):

    tl, ta = train_epoch(model, train_loader)
    vf1, va = evaluate(model, val_loader)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Ep {ep:03d}/{CONFIG['num_epochs']} │ "
        f"Train Loss {tl:.4f}  Acc {ta:.1%} │ "
        f"Val F1 {vf1:.4f}  Acc {va:.1%} │ "
        f"LR {current_lr:.2e}",
        end="",
    )

    if va > best_val_acc:
        best_val_acc = va
        best_val_f1  = vf1
        epochs_no_improve = 0
        torch.save(
            {
                "epoch":     ep,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc":   va,
                "val_f1":    vf1,
            },
            best_ckpt,
        )
        print("  ✓ Saved Best")
    else:
        epochs_no_improve += 1
        print()

    if epochs_no_improve >= CONFIG["patience"]:
        print(f"\nEarly stopping triggered at epoch {ep} "
              f"(no improvement for {CONFIG['patience']} epochs).")
        break

# ═══════════════════════════════════════════════════════════════════
#  TEST EVALUATION  (load best checkpoint)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("TEST EVALUATION  (best checkpoint)")
print("═" * 60)

ckpt = torch.load(best_ckpt, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(y)

all_preds   = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

test_acc = accuracy_score(all_targets, all_preds)
test_f1  = f1_score(all_targets, all_preds, average="macro", zero_division=0)
test_f1w = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

print(f"  Best checkpoint: epoch {ckpt['epoch']}")
print(f"  Test Accuracy  : {test_acc:.4f}  ({test_acc:.1%})")
print(f"  Test F1 (macro): {test_f1:.4f}")
print(f"  Test F1 (wt)   : {test_f1w:.4f}")
print(f"\nFull Classification Report:\n")
print(classification_report(all_targets, all_preds, zero_division=0))

# Save predictions
np.save(
    os.path.join(CONFIG["output_dir"], "test_predictions.npy"), all_preds
)
np.save(
    os.path.join(CONFIG["output_dir"], "test_targets.npy"), all_targets
)
print(f"\nPredictions saved to: {CONFIG['output_dir']}")