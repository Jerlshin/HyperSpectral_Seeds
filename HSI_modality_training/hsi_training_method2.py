import os
import math
import random
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

torch.cuda.empty_cache()
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONFIG = {
    "patches_data": "./dataset/patches.npy",
    "labels_path":  "./dataset/labels.npy",
    "output_dir":   "./output/",

    "num_epochs":   300,
    "batch_size":   128,
    "patience":     60,

    # wavelength range for positional encoding
    # matches the Specim V10E / Hamamatsu CCD: 385–1000 nm, 256 bands
    "wavelength_min": 385.0,
    "wavelength_max": 1000.0,
    "num_bands":       256,
    "num_classes":      90,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed":   42,
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CONFIG["seed"])


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class RiceSeedDataset(Dataset):
    """
    Memory-mapped loader for (N, 256, 64, 64) float16 patches.

    Augmentations (train only)
    ──────────────────────────
      Spatial    : horizontal/vertical flip + 90° rotation
      Spectral   : band dropout (per-band Bernoulli mask)
      Spectral   : spectral cutout (zero-out a contiguous band window)
    """

    def __init__(
        self,
        patches_path: str,
        labels_path: str,
        indices: np.ndarray,
        augment: bool = False,
        spectral_dropout: bool = False,
        spectral_dropout_prob: float = 0.05,
    ):
        self.patches = np.load(patches_path, mmap_mode="r")
        self.labels  = np.load(labels_path)
        self.indices = indices

        self.augment              = augment
        self.spectral_dropout     = spectral_dropout
        self.spectral_dropout_prob = spectral_dropout_prob

    def __len__(self) -> int:
        return len(self.indices)

    # ── spectral augmentations ─────────────────────────────────────────────

    def _spectral_band_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Independently zero-out bands with probability p."""
        if not self.spectral_dropout:
            return x
        mask = torch.rand(x.shape[0]) > self.spectral_dropout_prob
        return x * mask.view(-1, 1, 1)

    def _spectral_cutout(self, x: torch.Tensor, max_bands: int = 24) -> torch.Tensor:
        """Zero-out a contiguous window of [1, max_bands] bands."""
        if not self.spectral_dropout:
            return x
        num_bands = x.shape[0]
        cut   = torch.randint(1, max(2, max_bands), (1,)).item()
        cut   = min(cut, num_bands - 1)
        start = torch.randint(0, num_bands - cut, (1,)).item()
        x = x.clone()
        x[start : start + cut] = 0
        return x

    # ── spatial augmentations ──────────────────────────────────────────────

    def _spatial_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random flip + 90° rotation (seed orientation is arbitrary)."""
        if not self.augment:
            return x
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1, 2])
        return x

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        patch = torch.from_numpy(
            self.patches[real_idx].copy()
        ).float()                             # (256, 64, 64)

        label = torch.tensor(self.labels[real_idx], dtype=torch.long)

        patch = self._spectral_band_dropout(patch)
        patch = self._spectral_cutout(patch)
        patch = self._spatial_augment(patch)

        return patch, label


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA SPLIT + LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

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
    augment=True, spectral_dropout=True,
)
val_dataset = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], val_idx,
)
test_dataset = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], test_idx,
)

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
)
val_loader = DataLoader(
    val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def norm2d(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm — stable with small batch sizes; groups clamped to channels."""
    return nn.GroupNorm(min(groups, channels), channels)


# ── DropPath (Stochastic Depth) ────────────────────────────────────────────────

class DropPath(nn.Module):
    """
    Stochastic depth regularisation (Huang et al., 2016; Larsson et al., 2017).

    During training, entire residual paths are randomly dropped with
    probability `drop_prob`.  At test time the path is always active and
    scaled by (1 - drop_prob) to preserve expected activation magnitude.

    Applied to both Transformer blocks (FFN + attention residuals) and
    CNN BottleneckBlocks, giving a smooth stochastic-depth schedule
    that linearly increases from 0 at the first block to `max_drop_rate`
    at the deepest block.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # sample one Bernoulli value per sample in the batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, device=x.device).floor_().div_(keep)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ── CBAM — Convolutional Block Attention Module ───────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel attention sub-module of CBAM (Woo et al., 2018).

    Both avg-pool and max-pool descriptors share the same MLP weights,
    their outputs are added before the sigmoid gate.  This is strictly
    stronger than the original SE block which uses only avg-pool:
      • avg-pool captures the mean spectral energy per channel
      • max-pool captures peak responses (sharp absorption features)
    The shared-weight design keeps the parameter count the same as SE.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        # shared MLP (applied to both pooled descriptors)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(B, C)
        mx  = F.adaptive_max_pool2d(x, 1).view(B, C)
        gate = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * gate.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial attention sub-module of CBAM.

    Creates a 2D attention map by: pooling across the channel dim
    (avg + max) → concat → 7×7 depthwise-separable conv → sigmoid.

    For 64×64 seed patches the 7×7 receptive field captures the
    elongated grain shape without being too global.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)           # (B,1,H,W)
        mx  = x.max(dim=1, keepdim=True).values     # (B,1,H,W)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    """
    Full CBAM: channel attention then spatial attention, sequentially.

    Replaces the original SE block throughout the CNN backbone.
    Channel attn selects 'which bands matter here', spatial attn
    selects 'where on the seed surface matters'.
    """

    def __init__(self, channels: int, reduction: int = 8, sa_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(sa_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ── Spectral Stem ──────────────────────────────────────────────────────────────

class SpectralStem(nn.Module):
    """
    256-band → 128-channel feature map at full 64×64 resolution.

    Design rationale:
      Step 1 — 1×1 pointwise: learn inter-band combinations (no spatial mixing).
               Keeps all 256 channels so the network sees the full spectrum.
      Step 2 — 3×3 depthwise: build spatial context independently per band.
               `groups=in_channels` means each band processes its own
               spatial neighbourhood — no cross-band mixing yet.
      Step 3 — 1×1 pointwise: compress to embed_dim, now with local spatial
               context already baked in.
    This ordering (mix bands → spread spatially → compress) is more
    information-preserving than compressing first then mixing.
    """

    def __init__(self, in_channels: int = 256, embed_dim: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            # Step 1: inter-band mixing at full spectral width
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm2d(in_channels, groups=16),
            nn.GELU(),
            # Step 2: per-band spatial smoothing (depthwise)
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            norm2d(in_channels, groups=16),
            nn.GELU(),
            # Step 3: compress to embed_dim
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            norm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── CNN Backbone — BottleneckBlock ────────────────────────────────────────────

class BottleneckBlock(nn.Module):
    """
    Pre-activation bottleneck residual block with DropPath.

    mid_channels = out_channels // 2  (vs // 4 in vanilla ResNet-50).
    The wider bottleneck is important here: with only 3–4 stages the
    intermediate representation must be expressive enough on its own —
    there is no depth to compensate for a narrow bottleneck.

    DropPath is applied to the residual path (not the shortcut), following
    the stochastic-depth convention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        mid = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.norm1 = norm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.norm2 = norm2d(mid)

        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.norm3 = norm2d(out_channels)

        self.drop_path = DropPath(drop_path_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.gelu(self.norm1(self.conv1(x)))
        out = F.gelu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out = self.drop_path(out) + identity
        return F.gelu(out)


# ── Spectral Positional Encoding ──────────────────────────────────────────────

class WavelengthPositionalEncoding(nn.Module):
    """
    Wavelength-aware positional encoding for spectral tokens.

    Two components, summed:
      1. Sinusoidal (physics-based, fixed)
         Standard sinusoidal PE but indexed by actual nanometre wavelength
         rather than arbitrary token index.  Wavelength-proportional spacing
         means adjacent tokens in the NIR (wide spectral gap) are farther
         apart in PE space than adjacent tokens in the VIS — matching the
         physical structure of the spectrum.

      2. Learnable (data-driven, trained)
         A small nn.Embedding(n_tokens, token_dim) lets the network shift
         the PE to account for dataset-specific spectral patterns that the
         sinusoidal term cannot capture.

    Usage:
        enc = WavelengthPositionalEncoding(
                  n_tokens=16, token_dim=64,
                  wl_min=385.0, wl_max=1000.0
              )
        tokens = tokens + enc()   # (1, n_tokens, token_dim)
    """

    def __init__(
        self,
        n_tokens: int,
        token_dim: int,
        wl_min: float = 385.0,
        wl_max: float = 1000.0,
    ):
        super().__init__()
        self.n_tokens  = n_tokens
        self.token_dim = token_dim

        # ── sinusoidal component (not a parameter) ──────────────────────
        # centre wavelength of each band group
        wls = torch.linspace(wl_min, wl_max, n_tokens)   # (n_tokens,)

        # normalise to [0, 1] then to [0, 2π]
        wls_norm = (wls - wl_min) / (wl_max - wl_min) * 2 * math.pi

        # standard sin/cos encoding over token_dim dimensions
        dim_idx  = torch.arange(0, token_dim // 2).float()
        div_term = torch.pow(10000.0, 2 * dim_idx / token_dim)  # (token_dim//2,)

        # (n_tokens, token_dim)
        pe = torch.zeros(n_tokens, token_dim)
        pe[:, 0::2] = torch.sin(wls_norm.unsqueeze(1) / div_term.unsqueeze(0))
        pe[:, 1::2] = torch.cos(wls_norm.unsqueeze(1) / div_term.unsqueeze(0))

        # register as buffer — saved with model state, not a trainable param
        self.register_buffer("sin_pe", pe.unsqueeze(0))  # (1, n_tokens, token_dim)

        # ── learnable component ─────────────────────────────────────────
        self.learnable_pe = nn.Embedding(n_tokens, token_dim)
        nn.init.normal_(self.learnable_pe.weight, std=0.02)

    def forward(self) -> torch.Tensor:
        """Returns (1, n_tokens, token_dim) to broadcast over batch."""
        idx = torch.arange(self.n_tokens, device=self.sin_pe.device)
        return self.sin_pe + self.learnable_pe(idx).unsqueeze(0)


# ── Patch-Level Spectral Tokenizer ────────────────────────────────────────────

class SpectralTokenizer(nn.Module):
    """
    Divides the 256-band spectrum into n_tokens contiguous band groups
    and projects each group to a token embedding.

    Why grouping (not individual bands)?
      • 256 individual tokens → 256 × 256 attention = 65 536 pairs.
        With only n_layers=3 and d=64 this is over-parameterised.
      • Adjacent bands are highly correlated (r > 0.95 for neighbours).
        Grouping exploits this: each group captures a coherent spectral
        sub-region (e.g., green 500–560 nm, red 620–700 nm, red-edge
        700–730 nm, NIR 730–1000 nm).
      • n_tokens=16 → 16 × 16 = 256 attention pairs — tractable.

    Steps:
      1. Spatial mean-pool: (B, 256, H, W) → (B, 256)   global spectrum
      2. Reshape into groups: (B, n_tokens, bands_per_token)
      3. Linear project each group: → (B, n_tokens, token_dim)
    """

    def __init__(
        self,
        num_bands: int  = 256,
        n_tokens: int   = 16,
        token_dim: int  = 64,
    ):
        super().__init__()
        assert num_bands % n_tokens == 0, "num_bands must be divisible by n_tokens"
        self.n_tokens       = n_tokens
        self.bands_per_token = num_bands // n_tokens   # 16

        # shared linear projection applied per token group
        self.proj = nn.Linear(self.bands_per_token, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 256, H, W)  raw HSI cube
        Returns:
            tokens: (B, n_tokens, token_dim)
        """
        # global mean spectrum
        spec = x.mean(dim=[2, 3])                          # (B, 256)
        # split into n_tokens groups
        spec = spec.view(x.size(0), self.n_tokens,
                         self.bands_per_token)             # (B, 16, 16)
        # project each group to token_dim
        tokens = self.proj(spec)                           # (B, 16, 64)
        return tokens


# ── Spectral Transformer Block ────────────────────────────────────────────────

class SpectralTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer encoder block with DropPath.

    Pre-LN (norm before attention/FFN) is more stable than Post-LN
    during early training, which matters for small datasets where
    the LR schedule has less room to recover from instabilities.

    Components:
      • Multi-head self-attention (nhead heads)
      • DropPath on the attention residual
      • 2-layer FFN: token_dim → ffn_dim → token_dim, GELU activation
      • DropPath on the FFN residual

    Attention across 16 spectral tokens lets the model learn:
      "How does the reflectance in the NIR (token 14) relate to
       the chlorophyll absorption dip in red (token 8)?"
    This cross-wavelength reasoning is impossible in a pure 1D CNN.
    """

    def __init__(
        self,
        token_dim: int,
        nhead: int,
        ffn_mult: int = 4,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn  = nn.MultiheadAttention(
            token_dim,
            nhead,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(token_dim)
        ffn_dim = token_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(attn_drop),
            nn.Linear(ffn_dim, token_dim),
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_tokens, token_dim)"""
        # ── attention block ─────────────────────────────────────────
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path1(attn_out)

        # ── FFN block ───────────────────────────────────────────────
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


# ── Multiscale 1D Spectral Branch ─────────────────────────────────────────────

class MultiscaleSpectralBranch(nn.Module):
    """
    Parallel 1D convolutions over the mean spectrum with three kernel sizes.

    Why multiscale?
      Spectral features operate at different scales simultaneously:
        k= 3 → narrow absorption bands (e.g., specific pigment peaks, ~5 nm)
        k= 7 → medium features (e.g., red-edge slope, ~20 nm)
        k=15 → broad reflectance plateaus (e.g., NIR plateau, ~50+ nm)

    All three branches receive the same mean spectrum.  Their outputs
    are AdaptiveAvgPooled to the same length then concatenated, giving
    the classifier three complementary views of the spectral shape.

    The 256-band spectrum has known structure:
      • 385–500 nm : blue / UV edge — minimal rice reflectance
      • 500–600 nm : green peak
      • 620–700 nm : red absorption trough (chlorophyll)
      • 700–730 nm : red-edge transition
      • 730–1000 nm: NIR plateau (cell structure / starch / protein)

    Narrow kernels resolve the red-edge; wide kernels capture the
    NIR–VIS contrast. Together they produce a richer feature than
    any single scale.
    """

    def __init__(
        self,
        num_bands: int = 256,
        out_channels: int = 32,
        out_dim: int = 128,
    ):
        super().__init__()
        # three parallel conv branches — all input channels = 1
        self.branch3  = self._make_branch(1, out_channels, kernel=3)
        self.branch7  = self._make_branch(1, out_channels, kernel=7)
        self.branch15 = self._make_branch(1, out_channels, kernel=15)

        # fuse: 3 × out_channels → out_dim
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # per-branch global pool
        )
        self.proj = nn.Sequential(
            nn.Linear(out_channels * 3, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _make_branch(
        in_ch: int, out_ch: int, kernel: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),          # global spectral pooling → (B, out_ch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 256, H, W)"""
        # global mean spectrum → (B, 1, 256)
        spec = x.mean(dim=[2, 3]).unsqueeze(1)

        f3  = self.branch3(spec).squeeze(-1)   # (B, 32)
        f7  = self.branch7(spec).squeeze(-1)   # (B, 32)
        f15 = self.branch15(spec).squeeze(-1)  # (B, 32)

        fused = torch.cat([f3, f7, f15], dim=1)  # (B, 96)
        return self.proj(fused)                   # (B, 128)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN MODEL — SpectralFormerNet
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralFormerNet(nn.Module):
    """
    Tri-branch hyperspectral seed classifier.

    Branch A — SpectralFormer (spectral transformer)
        Captures cross-wavelength relationships that CNNs cannot model:
        self-attention across 16 spectral tokens lets the network ask
        "is this NIR reflectance consistent with this VIS absorption?"

    Branch B — MultiscaleSpectral (1D CNN)
        Extracts spectral shape features at narrow / medium / broad scales.
        Fast, lightweight, complementary to the transformer.

    Branch C — Spatial-Spectral CNN (2D CNN)
        Captures spatial texture on the seed surface: endosperm patterns,
        chalky spots, grain shape, surface roughness — all discriminative
        between varieties.

    The three branches are fused by simple concatenation followed by a
    deep classifier MLP.  No cross-attention fusion was used because:
      • The branches target fundamentally different feature types.
      • With 9 k samples, cross-attention fusion would need careful
        regularisation that simple concatenation avoids.
      • Empirically, concatenation + deep MLP performs competitively.

    DropPath schedule (stochastic depth)
        Linear ramp from 0.0 (first block) to max_drop_path (last block).
        For 4 backbone stages: [0.05, 0.10, 0.15, 0.20] at max=0.20.
        For 3 transformer layers: [0.05, 0.10, 0.15] at max=0.15.
    """

    def __init__(
        self,
        num_classes: int  = 90,
        in_channels: int  = 256,
        n_tokens: int     = 16,
        token_dim: int    = 64,
        n_heads: int      = 4,
        n_tf_layers: int  = 3,
        ffn_mult: int     = 4,
        wl_min: float     = 385.0,
        wl_max: float     = 1000.0,
        max_dp_cnn: float = 0.20,   # max DropPath for CNN backbone
        max_dp_tf: float  = 0.15,   # max DropPath for transformer
    ):
        super().__init__()

        # ── stochastic depth schedules ───────────────────────────────────
        n_cnn_blocks = 4   # stages 0-3
        cnn_dp  = [max_dp_cnn * i / (n_cnn_blocks - 1)
                   for i in range(n_cnn_blocks)]          # [0, .067, .133, .20]
        tf_dp   = [max_dp_tf  * i / max(n_tf_layers - 1, 1)
                   for i in range(n_tf_layers)]           # [0, .075, .15]

        # ── Branch C: Spatial-Spectral CNN ──────────────────────────────
        self.stem = SpectralStem(in_channels=in_channels, embed_dim=128)

        self.stage0 = BottleneckBlock(128, 128, stride=1, drop_path_rate=cnn_dp[0])
        self.cbam0  = CBAM(128)

        self.stage1 = BottleneckBlock(128, 256, stride=2, drop_path_rate=cnn_dp[1])
        self.cbam1  = CBAM(256)

        self.stage2 = BottleneckBlock(256, 384, stride=2, drop_path_rate=cnn_dp[2])
        self.cbam2  = CBAM(384)

        self.stage3 = BottleneckBlock(384, 512, stride=2, drop_path_rate=cnn_dp[3])
        self.cbam3  = CBAM(512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)    # → (B, 512)

        # ── Branch A: SpectralFormer ─────────────────────────────────────
        self.tokenizer  = SpectralTokenizer(
            num_bands=in_channels, n_tokens=n_tokens, token_dim=token_dim
        )
        self.pos_enc    = WavelengthPositionalEncoding(
            n_tokens=n_tokens, token_dim=token_dim,
            wl_min=wl_min, wl_max=wl_max,
        )
        self.tf_blocks  = nn.ModuleList([
            SpectralTransformerBlock(
                token_dim=token_dim, nhead=n_heads,
                ffn_mult=ffn_mult, attn_drop=0.05,
                drop_path_rate=tf_dp[i],
            )
            for i in range(n_tf_layers)
        ])
        self.tf_norm    = nn.LayerNorm(token_dim)
        # flatten 16 tokens × token_dim → single vector
        self.tf_head    = nn.Sequential(
            nn.Linear(n_tokens * token_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ── Branch B: Multiscale 1D Spectral ────────────────────────────
        self.ms_branch = MultiscaleSpectralBranch(
            num_bands=in_channels, out_channels=32, out_dim=128
        )

        # ── Fusion Classifier ────────────────────────────────────────────
        fusion_in = 512 + 256 + 128   # = 896
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    # ── forward pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 256, 64, 64)"""

        # ── Branch A: SpectralFormer ─────────────────────────────────────
        tokens = self.tokenizer(x)                      # (B, 16, 64)
        tokens = tokens + self.pos_enc()                # + positional encoding
        for blk in self.tf_blocks:
            tokens = blk(tokens)                        # (B, 16, 64)
        tokens  = self.tf_norm(tokens)
        tf_feat = self.tf_head(tokens.flatten(1))       # (B, 256)

        # ── Branch B: Multiscale 1D ──────────────────────────────────────
        ms_feat = self.ms_branch(x)                     # (B, 128)

        # ── Branch C: Spatial-Spectral CNN ──────────────────────────────
        h = self.stem(x)                                # (B, 128, 64, 64)

        h = self.cbam0(self.stage0(h))                  # (B, 128, 64, 64)
        h = self.cbam1(self.stage1(h))                  # (B, 256, 32, 32)
        h = self.cbam2(self.stage2(h))                  # (B, 384, 16, 16)
        h = self.cbam3(self.stage3(h))                  # (B, 512,  8,  8)

        sp_feat = self.global_pool(h).flatten(1)        # (B, 512)

        # ── Fusion ───────────────────────────────────────────────────────
        fused = torch.cat([sp_feat, tf_feat, ms_feat], dim=1)  # (B, 896)
        return self.classifier(fused)

    # ── weight initialisation ───────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    enabled: bool = True,
):
    """
    Mixup data augmentation (Zhang et al., 2018).

    Changes vs original:
      • alpha=0.2 (was 0.4) — keeps dominant label above ~0.65 on average.
      • lam = max(lam, 1-lam) — hard-clamp: dominant class always ≥ 0.5.
        With 90 classes and ~67 train samples per class this is critical:
        alpha=0.4 without clamping → mean lam≈0.5 → essentially 50% of
        the signal per class destroyed every step.
      • enabled flag — caller disables for warmup epochs so the model
        first builds a clean class representation (see MIXUP_WARMUP_EPOCHS).
    """
    if not enabled or alpha <= 0.0:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)                       # dominant class ≥ 0.5

    idx     = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


MIXUP_WARMUP_EPOCHS = 15   # train clean-label for this many epochs first


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL + OPTIMISER
# ═══════════════════════════════════════════════════════════════════════════════

model = SpectralFormerNet(
    num_classes=CONFIG["num_classes"],
    in_channels=CONFIG["num_bands"],
    n_tokens=16,
    token_dim=64,
    n_heads=4,
    n_tf_layers=3,
    ffn_mult=4,
    wl_min=CONFIG["wavelength_min"],
    wl_max=CONFIG["wavelength_max"],
    max_dp_cnn=0.20,
    max_dp_tf=0.15,
).to(CONFIG["device"])

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {n_params / 1e6:.3f} M")
print(f"Params / train sample : {n_params / len(train_idx):.0f}")

# ── Loss ──────────────────────────────────────────────────────────────────────
# No label smoothing: model is in the underfitting regime (see diagnostic);
# smoothing + mixup compounds soft-target confusion.  Re-enable at ≥ 0.05
# only if val F1 stops improving while train F1 is high (overfitting signal).
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

# ── Optimiser ─────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,                # OneCycleLR warms up from this
    weight_decay=2e-4,
    betas=(0.9, 0.999),
)

# ── Scheduler: OneCycleLR ─────────────────────────────────────────────────────
# Warms up for pct_start fraction of training then anneals via cosine.
# Called per-batch (inside train_one_epoch), not per-epoch.
# Avoids the early-plateau seen with CosineAnnealingLR (epochs 28-50 in v1).
steps_per_epoch = len(train_loader)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    epochs=CONFIG["num_epochs"],
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,           # 10 % warm-up
    div_factor=5.0,          # start_lr = max_lr / 5 = 1e-4
    final_div_factor=1e3,    # end_lr = start_lr / 1000 = 1e-7
    anneal_strategy="cos",
)

scaler = GradScaler()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mixup_enabled: bool = True,
    accumulation_steps: int = 1,
) -> tuple:

    model.train()

    total_loss     = 0.0
    total_correct  = 0
    total_samples  = 0
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

        # OneCycleLR steps every batch
        scheduler.step()

        total_loss    += loss.item() * x.size(0) * accumulation_steps
        preds          = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()   # accuracy vs true label
        total_samples += y.size(0)

        running_preds.append(preds.detach().cpu())
        running_labels.append(y.detach().cpu())

    avg_loss    = total_loss / total_samples
    accuracy    = total_correct / total_samples
    all_preds   = torch.cat(running_preds).numpy()
    all_labels  = torch.cat(running_labels).numpy()

    f1_macro    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision   = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall      = recall_score(all_labels, all_preds, average="macro",    zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, precision, recall


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "Val",
) -> tuple:

    model.eval()

    total_loss     = 0.0
    total_correct  = 0
    total_samples  = 0
    running_preds  = []
    running_labels = []

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

    avg_loss    = total_loss / total_samples
    accuracy    = total_correct / total_samples
    all_preds   = torch.cat(running_preds).numpy()
    all_labels  = torch.cat(running_labels).numpy()

    f1_macro    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision   = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall      = recall_score(all_labels, all_preds, average="macro",    zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted, precision, recall


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

best_val_f1      = 0.0
patience_counter = 0

os.makedirs(CONFIG["output_dir"], exist_ok=True)
model_path = os.path.join(CONFIG["output_dir"], "best_spectralformer.pth")

for epoch in range(1, CONFIG["num_epochs"] + 1):

    mixup_on = epoch > MIXUP_WARMUP_EPOCHS

    train_m = train_one_epoch(
        model, train_loader, optimizer, criterion,
        CONFIG["device"],
        mixup_enabled=mixup_on,
        accumulation_steps=1,
    )
    val_m = evaluate(model, val_loader, criterion, CONFIG["device"], phase="Val")

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch:03d}/{CONFIG['num_epochs']} "
        f"[mixup={'on ' if mixup_on else 'off'}] "
        f"lr={current_lr:.2e}"
    )
    print(
        f"  Train | Loss: {train_m[0]:.4f} | "
        f"Acc: {train_m[1]:.4f} | F1-mac: {train_m[2]:.4f}"
    )
    print(
        f"  Val   | Loss: {val_m[0]:.4f} | "
        f"Acc: {val_m[1]:.4f} | F1-mac: {val_m[2]:.4f}"
    )

    if val_m[2] > best_val_f1:
        best_val_f1 = val_m[2]
        torch.save(model.state_dict(), model_path)
        patience_counter = 0
        print(f"  ✓ Best val F1 = {best_val_f1:.4f}  →  model saved")
    else:
        patience_counter += 1

    if patience_counter >= CONFIG["patience"]:
        print(f"Early stopping at epoch {epoch}.")
        break


# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(tag: str, m: tuple) -> None:
    print(f"\n{'='*50}")
    print(f" {tag}")
    print(f"{'='*50}")
    print(f"  Loss         : {m[0]:.4f}")
    print(f"  Accuracy     : {m[1]:.4f}  ({m[1]*100:.1f} %)")
    print(f"  F1 (macro)   : {m[2]:.4f}")
    print(f"  F1 (weighted): {m[3]:.4f}")
    print(f"  Precision    : {m[4]:.4f}")
    print(f"  Recall       : {m[5]:.4f}")


# Results for the last-epoch model
test_m = evaluate(model, test_loader, criterion, CONFIG["device"], phase="Test")
print_results("FINAL (last epoch) TEST RESULTS", test_m)

# Results for the best-checkpoint model
model.load_state_dict(torch.load(model_path, weights_only=True))
test_m = evaluate(model, test_loader, criterion, CONFIG["device"], phase="Test")
print_results("BEST CHECKPOINT TEST RESULTS", test_m)