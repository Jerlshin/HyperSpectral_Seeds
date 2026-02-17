import os
import random
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# =============================================================================
#  CONFIG
# =============================================================================

torch.cuda.empty_cache()
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONFIG = {
    "patches_data": "./dataset/patches.npy",
    "labels_path":  "./dataset/labels.npy",
    "output_dir":   "./output/",

    "num_epochs":   300,
    "batch_size":   128,
    "patience":     60,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":   42,
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(CONFIG["seed"])


# =============================================================================
#  DATASET
# =============================================================================

class RiceSeedDataset(Dataset):
    """
    Loads (N, 256, 64, 64) float16 memory-mapped patches.

    Changes vs original:
      - _spectral_cutout now guards against cut==0 to avoid a randint(0,0) crash
      - augment and spectral_dropout are separate flags (unchanged)
      - Added per-sample instance norm option (helps when SNV wasn't applied
        consistently across sessions during preprocessing)
    """

    def __init__(
        self,
        patches_path,
        labels_path,
        indices,
        augment=False,
        spectral_dropout=False,
        spectral_dropout_prob=0.05,
    ):
        self.patches = np.load(patches_path, mmap_mode="r")
        self.labels  = np.load(labels_path)
        self.indices = indices

        self.augment              = augment
        self.spectral_dropout     = spectral_dropout
        self.spectral_dropout_prob = spectral_dropout_prob

    def __len__(self):
        return len(self.indices)

    # ------------------------------------------------------------------ #
    #  Spectral augmentations                                              #
    # ------------------------------------------------------------------ #

    def _spectral_dropout(self, x):
        """Zero-out individual bands at random."""
        if not self.spectral_dropout:
            return x
        mask = torch.rand(x.shape[0]) > self.spectral_dropout_prob
        return x * mask.view(-1, 1, 1)

    def _spectral_cutout(self, x, max_bands=20):
        """
        Zero-out a contiguous run of bands.
        FIX: guard cut==0 so randint(0, 0) never fires.
        """
        if not self.spectral_dropout:
            return x
        num_bands = x.shape[0]
        cut = torch.randint(1, max(2, max_bands), (1,)).item()  # at least 1
        cut = min(cut, num_bands - 1)
        start = torch.randint(0, num_bands - cut, (1,)).item()
        x = x.clone()
        x[start:start + cut] = 0
        return x

    # ------------------------------------------------------------------ #
    #  Spatial augmentations                                               #
    # ------------------------------------------------------------------ #

    def _spatial_augment(self, x):
        """Flip + 90° rotation — valid for seeds (no canonical orientation)."""
        if not self.augment:
            return x
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1, 2])
        return x

    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        patch = torch.from_numpy(
            self.patches[real_idx].copy()
        ).float()                            # (256, 64, 64)

        label = torch.tensor(
            self.labels[real_idx],
            dtype=torch.long
        )

        patch = self._spectral_dropout(patch)
        patch = self._spectral_cutout(patch)
        patch = self._spatial_augment(patch)

        return patch, label


# =============================================================================
#  DATA SPLIT + LOADERS
# =============================================================================

labels  = np.load(CONFIG["labels_path"])
indices = np.arange(len(labels))

train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
)

print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

train_dataset = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], train_idx,
    augment=True, spectral_dropout=True
)
val_dataset  = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], val_idx
)
test_dataset = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], test_idx
)

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4
)
val_loader = DataLoader(
    val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)
test_loader = DataLoader(
    test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True
)


# =============================================================================
#  MODEL
# =============================================================================

def norm_layer(channels, groups=8):
    """GroupNorm — safe when batch size is small; groups=8 is a good default."""
    return nn.GroupNorm(min(groups, channels), channels)



class SpectralStem(nn.Module):
    """
    256-band → embed_dim feature map.
    Now: 256 → 256 → embed_dim  (no aggressive compression on first conv).
    """
    def __init__(self, in_channels=256, embed_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            # Keep full 256 channels in first conv — learn band mixing first
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels, groups=16),
            nn.GELU(),

            # Light spatial smoothing at full spectral width
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),   # depthwise
            norm_layer(in_channels, groups=16),
            nn.GELU(),

            # Now compress to embed_dim
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            norm_layer(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 2   # was // 4 — doubled

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.norm1 = norm_layer(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.norm2 = norm_layer(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.norm3 = norm_layer(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.gelu(self.norm1(self.conv1(x)))
        out = F.gelu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += identity
        return F.gelu(out)



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # dual pooling
        avg = F.adaptive_avg_pool2d(x, 1).view(B, C)
        mx  = F.adaptive_max_pool2d(x, 1).view(B, C)
        y = F.gelu(self.fc1(avg + mx))          # combine before projection
        y = torch.sigmoid(self.fc2(y)).view(B, C, 1, 1)
        return x * y


class SpectralBranch(nn.Module):
    """
    Processes the mean spectrum (B, 256) as a 1D signal.
    Uses 1D convolutions to detect spectral patterns (absorption features).
    """
    def __init__(self, num_bands=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # treat as (B, 1, 256) signal → 1D convolutions
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(4),   # (B, 128, 4)
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x: (B, 256, H, W) — spatial HSI cube
        # compute mean spectrum across spatial dims
        spec = x.mean(dim=[2, 3])          # (B, 256)
        spec = spec.unsqueeze(1)           # (B, 1, 256)
        spec = self.net(spec)              # (B, 128, 4)
        spec = spec.view(spec.size(0), -1) # (B, 512)
        spec = self.proj(spec)             # (B, out_dim)
        return spec


class RiceHSINet(nn.Module):
    def __init__(self, num_classes=90, in_channels=256):
        super().__init__()

        # ---- Spatial-spectral branch ----------------------------------- #
        self.stem = SpectralStem(in_channels=in_channels, embed_dim=128)

        # stage0: full 64×64 resolution block (no stride) — new
        self.stage0 = BottleneckBlock(128, 128, stride=1)
        self.att0   = SEBlock(128)

        # stage1-3: same as before but wider mid_channels
        self.stage1 = BottleneckBlock(128, 256, stride=2)   # 64→32
        self.att1   = SEBlock(256)

        self.stage2 = BottleneckBlock(256, 384, stride=2)   # 32→16
        self.att2   = SEBlock(384)

        self.stage3 = BottleneckBlock(384, 512, stride=2)   # 16→8
        self.att3   = SEBlock(512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)           # 8→1

        # ---- Pure spectral branch -------------------------------------- #
        self.spectral_branch = SpectralBranch(num_bands=in_channels, out_dim=256)

        # ---- Fusion classifier ---------------------------------------- #
        # 512 (spatial) + 256 (spectral) = 768
        fusion_dim = 512 + 256
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),                # slightly higher than original 0.2
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        # --- spectral branch (operates on raw cube) ---
        spec_feat = self.spectral_branch(x)   # (B, 256)

        # --- spatial branch ---
        x = self.stem(x)

        x = self.stage0(x)
        x = self.att0(x)

        x = self.stage1(x)
        x = self.att1(x)

        x = self.stage2(x)
        x = self.att2(x)

        x = self.stage3(x)
        x = self.att3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)               # (B, 512)

        # --- fuse ---
        fused = torch.cat([x, spec_feat], dim=1)  # (B, 768)
        return self.classifier(fused)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# =============================================================================
#  TRAINING UTILITIES
# =============================================================================

def mixup_data(x, y, alpha=0.2, enabled=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    Returns (x, y, y, 1.0) when disabled (identity).
    """
    if not enabled or alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)           # keep dominant label above 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# =============================================================================
#  MODEL INSTANTIATION
# =============================================================================

model = RiceHSINet(num_classes=90, in_channels=256).to(CONFIG["device"])

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Total parameters: {n_params:.3f} Million")


criterion = nn.CrossEntropyLoss(label_smoothing=0.0)


# Estimate steps_per_epoch
steps_per_epoch = len(train_loader)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=2e-4,    # slightly higher regularisation (was 1e-4)
    betas=(0.9, 0.999),
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    epochs=CONFIG["num_epochs"],
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,         # 10% warm-up, 90% cosine decay
    div_factor=5,          # start_lr = max_lr / 5 = 1e-4
    final_div_factor=1e3,  # end_lr = start_lr / 1000
    anneal_strategy="cos",
)

scaler = GradScaler()

MIXUP_WARMUP_EPOCHS = 15   # start mixup after 15 clean-label epochs


# =============================================================================
#  TRAINING LOOP
# =============================================================================

def compute_metrics(y_true, y_pred):
    acc        = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall     = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1_macro, f1_weighted, precision, recall


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    accumulation_steps=1, mixup_enabled=True):
    model.train()

    total_loss    = 0
    total_correct = 0
    total_samples = 0
    running_preds  = []
    running_labels = []

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, enabled=mixup_enabled)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x_mix)
            loss   = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss   = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # FIX: scheduler.step() is called per-batch for OneCycleLR
        scheduler.step()

        total_loss += loss.item() * x.size(0) * accumulation_steps

        # Accuracy uses unmixed labels (y_a = original y)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()    # compare to true label
        total_samples += y.size(0)

        running_preds.append(preds.detach().cpu())
        running_labels.append(y.detach().cpu())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    all_preds  = torch.cat(running_preds).numpy()
    all_labels = torch.cat(running_labels).numpy()

    f1_macro   = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    precision  = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall     = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, precision, recall


def evaluate(model, loader, criterion, device, phase="Val"):
    model.eval()

    total_loss    = 0
    total_correct = 0
    total_samples = 0
    running_preds  = []
    running_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss   = criterion(logits, y)

            total_loss    += loss.item() * x.size(0)
            preds          = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            running_preds.append(preds.cpu())
            running_labels.append(y.cpu())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    all_preds  = torch.cat(running_preds).numpy()
    all_labels = torch.cat(running_labels).numpy()

    f1_macro    = f1_score(all_labels, all_preds, average="macro")
    f1_weighted  = f1_score(all_labels, all_preds, average="weighted")
    precision   = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall      = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, precision, recall


# =============================================================================
#  MAIN TRAINING LOOP
# =============================================================================

best_val_f1     = 0.0
patience_counter = 0

model_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
os.makedirs(CONFIG["output_dir"], exist_ok=True)

for epoch in range(1, CONFIG["num_epochs"] + 1):

    # FIX 6: disable mixup for first MIXUP_WARMUP_EPOCHS epochs
    mixup_on = epoch > MIXUP_WARMUP_EPOCHS

    train_metrics = train_one_epoch(
        model, train_loader, optimizer, criterion,
        CONFIG["device"], epoch,
        accumulation_steps=1,
        mixup_enabled=mixup_on,
    )

    val_metrics = evaluate(
        model, val_loader, criterion, CONFIG["device"], phase="Val"
    )


    print(
        f"\nEpoch {epoch:03d} | mixup={'on' if mixup_on else 'off'} | "
        f"LR={optimizer.param_groups[0]['lr']:.2e}"
    )
    print(
        f"Train | Loss: {train_metrics[0]:.4f} | "
        f"Acc: {train_metrics[1]:.4f} | "
        f"F1(macro): {train_metrics[2]:.4f}"
    )
    print(
        f"Val   | Loss: {val_metrics[0]:.4f} | "
        f"Acc: {val_metrics[1]:.4f} | "
        f"F1(macro): {val_metrics[2]:.4f}"
    )

    if val_metrics[2] > best_val_f1:
        best_val_f1 = val_metrics[2]
        torch.save(model.state_dict(), model_path)
        patience_counter = 0
        print(f"  ↑ New best val F1: {best_val_f1:.4f} — model saved")
    else:
        patience_counter += 1

    if patience_counter >= CONFIG["patience"]:
        print("Early stopping triggered.")
        break


# =============================================================================
#  EVALUATION
# =============================================================================

# Final (last epoch) model
test_metrics = evaluate(model, test_loader, criterion, CONFIG["device"], phase="Test")
print("\n===== FINAL TEST RESULTS =====")
print(f"Loss:         {test_metrics[0]:.4f}")
print(f"Accuracy:     {test_metrics[1]:.4f}")
print(f"F1 (macro):   {test_metrics[2]:.4f}")
print(f"F1 (weighted):{test_metrics[3]:.4f}")
print(f"Precision:    {test_metrics[4]:.4f}")
print(f"Recall:       {test_metrics[5]:.4f}")

# Best checkpoint model
model.load_state_dict(torch.load(model_path, weights_only=True))
test_metrics = evaluate(model, test_loader, criterion, CONFIG["device"], phase="Test")
print("\n===== BEST CHECKPOINT TEST RESULTS =====")
print(f"Loss:         {test_metrics[0]:.4f}")
print(f"Accuracy:     {test_metrics[1]:.4f}")
print(f"F1 (macro):   {test_metrics[2]:.4f}")
print(f"F1 (weighted):{test_metrics[3]:.4f}")
print(f"Precision:    {test_metrics[4]:.4f}")
print(f"Recall:       {test_metrics[5]:.4f}")