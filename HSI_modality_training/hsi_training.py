import os
import sys
import re
import math
import zipfile
import random
import shutil
from collections import defaultdict, Counter
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

import spectral


"""Config"""

torch.cuda.empty_cache()

CONFIG = {
    "patches_data" :  "./dataset/patches.npy",
    "labels_path" : "./dataset/labels.npy",
    "output_dir" : "./output/",
    
    "num_epochs" : 50,
    "patience" : 20,
     
    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed" : 42,
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(CONFIG["seed"])


"""Dataset"""

class RiceSeedDataset(Dataset):

    def __init__(
        self,
        patches_path,
        labels_path,
        indices,
        augment=False,
        spectral_dropout=False,
        spectral_dropout_prob=0.05
    ):
        self.patches = np.load(patches_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.indices = indices

        self.augment = augment
        self.spectral_dropout = spectral_dropout
        self.spectral_dropout_prob = spectral_dropout_prob

    def __len__(self):
        return len(self.indices)

    def _spectral_dropout(self, x):
        if not self.spectral_dropout:
            return x
        mask = torch.rand(x.shape[0]) > self.spectral_dropout_prob
        return x * mask.view(-1, 1, 1)

    def _spatial_augment(self, x):
        if not self.augment:
            return x
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1, 2])
        return x

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        patch = torch.from_numpy(
            self.patches[real_idx]
        ).float()

        label = torch.tensor(
            self.labels[real_idx],
            dtype=torch.long
        )

        patch = self._spectral_dropout(patch)
        patch = self._spatial_augment(patch)

        return patch, label

labels = np.load(CONFIG["labels_path"])
indices = np.arange(len(labels))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.3,
    stratify=labels,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=labels[temp_idx],
    random_state=42
)

train_dataset = RiceSeedDataset(
    CONFIG["patches_data"],
    CONFIG["labels_path"],
    train_idx,
    augment=True,
    spectral_dropout=True
)

val_dataset = RiceSeedDataset(
    CONFIG["patches_data"],
    CONFIG["labels_path"],
    val_idx
)

test_dataset = RiceSeedDataset(
    CONFIG["patches_data"],
    CONFIG["labels_path"],
    test_idx
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)


"""Model"""

class SpectralProjection(nn.Module):
    def __init__(self, in_channels=256, embed_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)

class SpatialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        return F.relu(out)

class SpectralAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        B, C, H, W = x.shape

        y = F.adaptive_avg_pool2d(x, 1).view(B, C)

        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(B, C, 1, 1)

        return x * y

class RiceHSINet(nn.Module):
    def __init__(self, num_classes=90, in_channels=256):
        super().__init__()

        self.spectral_proj = SpectralProjection(
            in_channels=in_channels,
            embed_dim=64
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SpatialBlock(64)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SpatialBlock(128)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SpatialBlock(256)
        )

        self.attention = SpectralAttention(256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.spectral_proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.attention(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


"""Training"""

model = RiceHSINet(
    num_classes=90,
    in_channels=256
).to(CONFIG["device"])

print("Total parameters:",
      sum(p.numel() for p in model.parameters()) / 1e6,
      "Million")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50
)

scaler = GradScaler()

def compute_metrics(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return acc, f1_macro, f1_weighted, precision, recall


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    accumulation_steps=1
):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    running_preds = []
    running_labels = []

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)


        with autocast(device_type=device.type, enabled=(device.type=="cuda")):
            logits = model(x)
            loss = criterion(logits, y)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * x.size(0) * accumulation_steps

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

        running_preds.append(preds.detach().cpu())
        running_labels.append(y.detach().cpu())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    all_preds = torch.cat(running_preds).numpy()
    all_labels = torch.cat(running_labels).numpy()

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, precision, recall


def evaluate(
    model,
    loader,
    criterion,
    device,
    epoch,
    phase="Val"
):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    running_preds = []
    running_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=(device.type=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            running_preds.append(preds.cpu())
            running_labels.append(y.cpu())


    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    all_preds = torch.cat(running_preds).numpy()
    all_labels = torch.cat(running_labels).numpy()

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, precision, recall



best_val_f1 = 0
patience_counter = 0

model_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
os.makedirs(CONFIG["output_dir"], exist_ok=True)

for epoch in range(1, CONFIG["num_epochs"] + 1):

    train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        CONFIG["device"],
        epoch,
        accumulation_steps=1
    )

    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        CONFIG["device"],
        epoch,
        phase="Val"
    )

    scheduler.step()

    print(f"\nEpoch {epoch:02d}")
    print(f"Train | Loss: {train_metrics[0]:.4f} | "
          f"Acc: {train_metrics[1]:.4f} | "
          f"F1(macro): {train_metrics[2]:.4f}")

    print(f"Val   | Loss: {val_metrics[0]:.4f} | "
          f"Acc: {val_metrics[1]:.4f} | "
          f"F1(macro): {val_metrics[2]:.4f}")

    if val_metrics[2] > best_val_f1:
        best_val_f1 = val_metrics[2]
        torch.save(model.state_dict(), model_path)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= CONFIG["patience"]:
        print("Early stopping triggered.")
        break


# Final model test
test_metrics = evaluate(
    model, test_loader, criterion, CONFIG["device"], epoch=0, phase="Test"
)

print("\n===== FINAL TEST RESULTS =====")
print(f"Loss: {test_metrics[0]:.4f}")
print(f"Accuracy: {test_metrics[1]:.4f}")
print(f"F1 (macro): {test_metrics[2]:.4f}")
print(f"F1 (weighted): {test_metrics[3]:.4f}")
print(f"Precision: {test_metrics[4]:.4f}")
print(f"Recall: {test_metrics[5]:.4f}")

# Best model test
model.load_state_dict(torch.load(model_path))

test_metrics = evaluate(
    model, test_loader, criterion, CONFIG["device"], epoch=0, phase="Test"
)


print("\n===== BEST TEST RESULTS =====")
print(f"Loss: {test_metrics[0]:.4f}")
print(f"Accuracy: {test_metrics[1]:.4f}")
print(f"F1 (macro): {test_metrics[2]:.4f}")
print(f"F1 (weighted): {test_metrics[3]:.4f}")
print(f"Precision: {test_metrics[4]:.4f}")
print(f"Recall: {test_metrics[5]:.4f}")

