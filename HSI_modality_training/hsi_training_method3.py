"""
SpectralFormerNet v2 — HSI Rice Variety Classification
=======================================================
Dataset : 90 rice varieties × 96 kernels = ~8 640 seeds
Input   : (B, 256, 64, 64) float32  channel-first
Classes : 90

Key fixes over v1
─────────────────
  Issue 1 fixed │ SpatialPatchTokenizer replaces global-mean tokenizer.
                │ The transformer now sees LOCAL spectral variation:
                │ embryo vs endosperm vs seed edge, each as a separate token.

  Issue 2 fixed │ Branch A (transformer) uses SPATIAL patch spectra.
                │ Branch B (multiscale 1D) uses GLOBAL mean spectrum.
                │ Zero redundancy — they process entirely different signals.

  Issue 3 fixed │ Model reduced to 1.42 M params (235/sample, was 522).
                │ Regularization significantly strengthened.

  LR fixed      │ max_lr 5e-4 → 3e-4, div_factor 5 → 10, pct_start 10% → 15%.
                │ Eliminates the val-loss oscillation seen in v1.

  WD fixed      │ Weight decay now EXCLUDED from bias + norm parameters.
                │ Applying WD to LayerNorm/GroupNorm harms gradient flow.

Architecture
────────────
  Branch A │ SpatialPatchTokenizer (64×64 → 4×4 grid → 16 spatial tokens)
  (128-d)  │ + Learnable 2D Spatial Positional Encoding
           │ + SpectralTransformerBlock × 2  (d=48, h=4, DropPath)
           │ + TokenAggregation (mean → linear → 128-d)
           │
  Branch B │ GlobalMeanSpectrum  (B,1,256)
  ( 96-d)  │ + MultiscaleConv1D  k = 3 / 7 / 15  in parallel
           │ + Projection → 96-d
           │
  Branch C │ SpectralStem        256 → 128  (PW → DW → PW)
  (384-d)  │ BottleneckBlock     128 → 128  @ 64×64  + CBAM
           │ BottleneckBlock     128 → 256  @ 32×32  + CBAM
           │ BottleneckBlock     256 → 384  @ 16×16  + CBAM
           │ GlobalAvgPool       → 384-d
           │
  Fusion   │ cat[384 + 128 + 96] = 608
           │ Linear(384) → GELU → Drop(0.50)
           │ Linear(192) → GELU → Drop(0.40)
           │ Linear( 90)

  Total    │ 1.42 M  │  235 params / train-sample
"""

import os
import math
import random
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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

    "num_epochs": 300,
    "batch_size": 128,
    "patience":   60,

    "num_bands":   256,
    "num_classes":  90,

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
    Memory-mapped (N, 256, 64, 64) float16 patch loader.

    Augmentations applied only during training:
        Spatial   — random horizontal/vertical flip + 90° rotation
        Spectral  — per-band Bernoulli dropout
        Spectral  — contiguous band cutout  (window of 1–24 bands zeroed)
    """

    def __init__(
        self,
        patches_path: str,
        labels_path: str,
        indices: np.ndarray,
        augment: bool = False,
        spectral_aug: bool = False,
        band_drop_prob: float = 0.05,
        max_cutout_bands: int = 24,
    ):
        self.patches  = np.load(patches_path, mmap_mode="r")
        self.labels   = np.load(labels_path)
        self.indices  = indices

        self.augment          = augment
        self.spectral_aug     = spectral_aug
        self.band_drop_prob   = band_drop_prob
        self.max_cutout_bands = max_cutout_bands

    def __len__(self) -> int:
        return len(self.indices)

    def _band_dropout(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand(x.shape[0]) > self.band_drop_prob).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x: torch.Tensor) -> torch.Tensor:
        # must clone: x came from mmap, in-place on view is undefined
        x   = x.clone()
        nb  = x.shape[0]
        cut = torch.randint(1, max(2, self.max_cutout_bands), (1,)).item()
        cut = min(cut, nb - 1)
        s   = torch.randint(0, nb - cut, (1,)).item()
        x[s : s + cut] = 0.0
        return x

    def _spatial_augment(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [1])
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, [1, 2])
        return x

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        patch = torch.from_numpy(self.patches[real_idx].copy()).float()
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)

        if self.spectral_aug:
            patch = self._band_dropout(patch)
            patch = self._band_cutout(patch)
        if self.augment:
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

train_ds = RiceSeedDataset(
    CONFIG["patches_data"], CONFIG["labels_path"], train_idx,
    augment=True, spectral_aug=True,
)
val_ds  = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], val_idx)
test_ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"], test_idx)

_loader_kw = dict(pin_memory=True, persistent_workers=True)
train_loader = DataLoader(
    train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=8, prefetch_factor=4, **_loader_kw,
)
val_loader = DataLoader(
    val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=4, **_loader_kw,
)
test_loader = DataLoader(
    test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=2, **_loader_kw,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with automatic group clamp — safe for any channel count."""
    return nn.GroupNorm(min(groups, channels), channels)


# ── DropPath (Stochastic Depth) ────────────────────────────────────────────────
class DropPath(nn.Module):
    """
    Randomly drops entire residual branches during training (Huang et al., 2016).

    Applied with a linear depth schedule: earlier (shallower) blocks have
    lower drop probability, deeper blocks have higher.  This creates an
    implicit ensemble over sub-networks of varying depth and is empirically
    more effective than uniform dropout for residual networks.

    At inference the path is always active; output is scaled by (1 - p) to
    match the expected training activation magnitude.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, device=x.device).floor_().div_(keep)
        return x * mask

    def extra_repr(self) -> str:
        return f"p={self.p:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL-SPECTRAL CNN
# ═══════════════════════════════════════════════════════════════════════════════

# ── SpectralStem ──────────────────────────────────────────────────────────────
class SpectralStem(nn.Module):
    """
    256-band → 128-channel feature map, full 64×64 resolution.

    Three-step design (information-preserving order):
      PW → DW → PW

      Step 1 — Pointwise (1×1):  mix information across all 256 bands.
               Keeps the full spectral width so cross-band combinations
               are learned before any spatial reasoning.

      Step 2 — Depthwise (3×3):  build LOCAL spatial context for each
               of the 256 band channels independently.  groups=in_channels
               means zero cross-band mixing at this step.

      Step 3 — Pointwise (1×1):  compress 256 → 128.  Now each output
               channel has seen both inter-band combinations (step 1) and
               local spatial structure (step 2) before compression.

    Why not compress first then spatially convolve?
    Compressing 256→128 before step 2 would discard half the spectral
    information before the network has built any spatial context to decide
    what to keep.
    """

    def __init__(self, in_ch: int = 256, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            gn(in_ch, 16), nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            gn(in_ch, 16), nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            gn(out_ch),    nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── CBAM ──────────────────────────────────────────────────────────────────────
class ChannelAttn(nn.Module):
    """
    CBAM channel attention with avg + max dual pooling (Woo et al., 2018).

    The MLP is SHARED between the avg-pool and max-pool descriptors —
    their outputs are SUMMED before the sigmoid gate.  This means:
      • avg-pool captures 'mean band energy'   (background illumination)
      • max-pool captures 'peak band response' (sharp absorption features)
    The shared weights force a single coherent gating that respects both,
    without doubling the parameter count vs SE.
    """

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(c // r, 8)
        self.mlp = nn.Sequential(nn.Linear(c, mid, bias=False), nn.GELU(),
                                 nn.Linear(mid, c, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(B, C)
        mx  = F.adaptive_max_pool2d(x, 1).view(B, C)
        g   = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(B, C, 1, 1)
        return x * g


class SpatialAttn(nn.Module):
    """
    CBAM spatial attention (7×7 conv on avg+max channel-descriptor maps).

    Produces a (B,1,H,W) attention mask that highlights 'where on the seed
    surface' to focus.  The 7×7 receptive field was chosen to capture the
    elongated grain profile without being too global at 64×64 resolution.
    """

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        mx  = x.max(1, keepdim=True).values
        g   = torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))
        return x * g


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.ca = ChannelAttn(c, r)
        self.sa = SpatialAttn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ── BottleneckBlock ───────────────────────────────────────────────────────────
class BottleneckBlock(nn.Module):
    """
    Pre-activation bottleneck (1×1 → 3×3 → 1×1) with DropPath.

    mid_channels = out_channels // 2  (wider than ResNet-50's //4).
    For shallow networks (3 stages), each bottleneck carries more load;
    a narrower bottleneck of //4 would over-compress the representation.

    DropPath is applied to the RESIDUAL path only (not the shortcut),
    following the stochastic-depth convention.
    """

    def __init__(self, ic: int, oc: int, stride: int = 1, drop_path: float = 0.0):
        super().__init__()
        mid = oc // 2

        self.conv1 = nn.Conv2d(ic, mid, 1, bias=False);   self.n1 = gn(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride,
                               padding=1, bias=False);     self.n2 = gn(mid)
        self.conv3 = nn.Conv2d(mid, oc, 1, bias=False);   self.n3 = gn(oc)

        self.dp = DropPath(drop_path)

        self.skip = nn.Sequential()
        if stride != 1 or ic != oc:
            self.skip = nn.Sequential(
                nn.Conv2d(ic, oc, 1, stride=stride, bias=False), gn(oc)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.conv1(x)))
        h = F.gelu(self.n2(self.conv2(h)))
        h = self.n3(self.conv3(h))
        return F.gelu(self.dp(h) + self.skip(x))


# ═══════════════════════════════════════════════════════════════════════════════
#  BRANCH A — SPATIAL-SPECTRAL TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

# ── SpatialPatchTokenizer ─────────────────────────────────────────────────────
class SpatialPatchTokenizer(nn.Module):
    """
    Converts a full HSI cube into a sequence of SPATIAL spectral tokens.

    This fixes Issue 1 of v1 which collapsed all spatial information into
    a single global spectrum before tokenizing.  Here every token carries
    the spectral signature of a LOCAL REGION of the seed:

      Step 1 — adaptive_avg_pool2d(x, (P, P)):
               Divides the 64×64 seed image into a P×P spatial grid.
               P=4 → 16 non-overlapping 16×16 regions.
               Each region is avg-pooled to a single spectral vector (B,256,P,P).

      Step 2 — reshape to (B, P², 256):
               Each of the 16 spatial positions becomes a 256-dim spectral
               vector (its average spectrum over the 16×16 crop).

      Step 3 — shared linear (256 → token_dim):
               Project every spatial token to the transformer embedding dim.
               Weights are shared across all 16 positions (translation-equivariant
               spectral projection).

    Why P=4 (16 tokens)?
      • At 64×64, P=4 gives 16×16 pixel patches — large enough to cover
        distinct seed regions (embryo, central endosperm, chalky edge).
      • 16 tokens × 16 tokens = 256 attention pairs — tractable.
      • P=8 (64 tokens) is computationally fine but overfit-prone at 6k samples.

    Why this matters over global mean:
      Rice seeds have spatial heterogeneity:
        - Embryo germ (corner, ~10% area): higher protein, lower starch
        - Central endosperm (60%):         dominant starch region
        - Dorsal chalky region (variable): affects NIR scatter
      These regions have DIFFERENT spectra.  A global mean washes them out.
      The transformer can now learn "the embryo spectrum of variety X is
      more consistent with its endosperm than variety Y."
    """

    def __init__(self, num_bands: int = 256, patch_grid: int = 4, token_dim: int = 48):
        super().__init__()
        self.pg  = patch_grid           # P
        self.n   = patch_grid ** 2      # number of tokens = P²

        # Shared linear projection (applied to each spatial token independently)
        self.proj = nn.Linear(num_bands, token_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 256, 64, 64)
        → tokens : (B, 16, token_dim)
        """
        # (B, 256, 4, 4)  — each cell is the mean spectrum of a 16×16 crop
        pooled = F.adaptive_avg_pool2d(x, (self.pg, self.pg))

        # (B, 4, 4, 256) → (B, 16, 256)
        B, C, H, W = pooled.shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(B, self.n, C)

        # (B, 16, token_dim)
        return self.proj(tokens)


# ── Spatial 2D Positional Encoding ────────────────────────────────────────────
class SpatialGridPE(nn.Module):
    """
    Learnable 2D positional encoding for a P×P spatial grid of tokens.

    Two components summed (analogous to the wavelength PE in v1):

      Sinusoidal (fixed, physics-grounded):
        Standard 2D sinusoidal encoding indexed by (row, col).
        Provides a smooth prior that nearby spatial positions are similar —
        which is true for seed images (adjacent regions share gradual
        spectral transitions, not sudden jumps).

      Learnable (data-driven):
        nn.Embedding(P², token_dim).  Lets the network override the
        sinusoidal prior for positions that have dataset-specific importance
        (e.g., the embryo corner always at position 0 in a consistently
        oriented scan).

    Usage:  tokens = tokens + self.pe()   # broadcasts over batch
    """

    def __init__(self, patch_grid: int = 4, token_dim: int = 48):
        super().__init__()
        n = patch_grid ** 2

        # ── sinusoidal component ────────────────────────────────────────
        # encode row and col positions separately, interleave dimensions
        half = token_dim // 2
        rows = torch.arange(patch_grid).float()
        cols = torch.arange(patch_grid).float()
        div  = torch.pow(10000.0, 2 * torch.arange(half // 2).float() / half)

        pe = torch.zeros(patch_grid, patch_grid, token_dim)
        pe[:, :, 0 : half : 2] = torch.sin(rows.view(-1,1,1) / div)        # row even
        pe[:, :, 1 : half : 2] = torch.cos(rows.view(-1,1,1) / div)        # row odd
        pe[:, :, half     : : 2] = torch.sin(cols.view(1,-1,1) / div)      # col even
        pe[:, :, half + 1 : : 2] = torch.cos(cols.view(1,-1,1) / div)      # col odd

        # (1, n, token_dim) — ready to broadcast over batch
        self.register_buffer("sin_pe", pe.view(1, n, token_dim))

        # ── learnable component ─────────────────────────────────────────
        self.learned = nn.Embedding(n, token_dim)
        nn.init.normal_(self.learned.weight, std=0.02)

        self.n = n

    def forward(self) -> torch.Tensor:
        idx = torch.arange(self.n, device=self.sin_pe.device)
        return self.sin_pe + self.learned(idx).unsqueeze(0)   # (1, n, token_dim)


# ── Spectral Transformer Block (Pre-LN, DropPath) ─────────────────────────────
class SpectralTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer encoder block with DropPath on both residuals.

    Pre-LN (norm before attention and FFN, not after) chosen for:
      • Training stability on small datasets — gradients at early layers
        are far more stable than Post-LN.
      • No need for LR warmup tricks to avoid gradient explosion.

    Self-attention across spatial tokens lets the model reason about
    global spatial relationships on the seed surface:
      "Is the reflectance pattern at the top-left (embryo) region consistent
       with the center endosperm region, given this is variety X?"

    FFN multiplier = 2 (not the standard 4) because:
      • token_dim=48 is already compact.
      • 4× FFN = 4×48=192 hidden units for 16 tokens across 6k samples
        is overparameterised.  2× provides sufficient non-linear capacity.
    """

    def __init__(
        self, d: int, heads: int, ffn_mult: int = 2,
        attn_drop: float = 0.0, drop_path: float = 0.0,
    ):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=attn_drop, batch_first=True)
        self.dp1  = DropPath(drop_path)

        self.ln2  = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d * ffn_mult), nn.GELU(),
            nn.Dropout(attn_drop),
            nn.Linear(d * ffn_mult, d),
        )
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention residual
        n = self.ln1(x)
        a, _ = self.attn(n, n, n)
        x = x + self.dp1(a)
        # FFN residual
        x = x + self.dp2(self.ffn(self.ln2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  BRANCH B — GLOBAL MULTISCALE 1D SPECTRAL
# ═══════════════════════════════════════════════════════════════════════════════

class MultiscaleSpectralBranch(nn.Module):
    """
    Three parallel Conv1D branches over the GLOBAL mean spectrum.

    Why global mean here (while Branch A uses spatial patches)?
      Branch A already extracts LOCAL spatial-spectral tokens.
      This branch provides the COMPLEMENTARY GLOBAL spectral fingerprint:
      the overall spectral shape of the seed without spatial bias.

      The global mean is precisely what a chemist measures with a
      spectrophotometer — it is the variety's canonical spectral signature.

    Kernel sizes and their physical meaning for HSI rice spectra:
      k= 3 → ~3 bands ≈  5 nm — narrow absorption/reflection peaks
              (pigment peaks, specific protein bands)
      k= 7 → ~7 bands ≈ 15 nm — medium features (red-edge slope,
              water absorption bands at ~970 nm)
      k=15 → ~15 bands ≈ 30 nm — broad reflectance regions (NIR plateau
              shape, starch/protein coarse absorption patterns)

    Each branch has TWO conv layers (not one) so it can learn both the
    feature and its context before global pooling.

    Branch outputs are global-pooled then concatenated: (B, 32×3=96).
    A final linear(96, 96) projects to the branch output dimension.
    """

    def __init__(self, num_bands: int = 256, out_ch: int = 32, out_dim: int = 96):
        super().__init__()
        self.b3  = self._branch(out_ch, 3)
        self.b7  = self._branch(out_ch, 7)
        self.b15 = self._branch(out_ch, 15)
        self.proj = nn.Sequential(
            nn.Linear(out_ch * 3, out_dim), nn.GELU(), nn.Dropout(0.1)
        )

    @staticmethod
    def _branch(out_ch: int, k: int) -> nn.Sequential:
        pad = k // 2
        return nn.Sequential(
            nn.Conv1d(1, out_ch, k, padding=pad, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),              # → (B, out_ch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 256, H, W)"""
        spec = x.mean([2, 3]).unsqueeze(1)           # (B, 1, 256)
        f3   = self.b3(spec).squeeze(-1)             # (B, 32)
        f7   = self.b7(spec).squeeze(-1)
        f15  = self.b15(spec).squeeze(-1)
        return self.proj(torch.cat([f3, f7, f15], 1))  # (B, 96)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN MODEL — SpectralFormerNet v2
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralFormerNet(nn.Module):
    """
    Three-branch hyperspectral seed classifier.

    The three branches are complementary by design — they process different
    aspects of the same input:

      Branch A (spatial transformer, 128-d):
        Q: "How do LOCAL spectral signatures at different seed regions
            relate to each other?"
        Self-attention over 16 spatial patches of 16×16 pixels each.
        Captures embryo/endosperm spectral relationships.

      Branch B (multiscale global 1D, 96-d):
        Q: "What does the OVERALL spectral shape of this seed look like
            at narrow, medium, and broad spectral scales?"
        Pure global spectral fingerprint at k=3/7/15 nm resolution.

      Branch C (CNN, 384-d):
        Q: "What local spatial texture and morphological features are
            present on this seed's surface?"
        Hierarchical 2D convolutions with CBAM spatial attention.

    None of these questions overlap — every branch contributes unique
    information to the final classifier.

    DropPath schedules (linear depth ramp):
      CNN  [s0, s1, s2] : [0.00, 0.10, 0.20]  max_p=0.20
      TF   [L0, L1]     : [0.00, 0.15]         max_p=0.15
    """

    def __init__(
        self,
        num_classes: int = 90,
        in_ch: int = 256,
        patch_grid: int = 4,         # P; tokens = P² = 16
        token_dim: int = 48,
        n_heads: int = 4,
        n_tf: int = 2,
        ffn_mult: int = 2,
        max_dp_cnn: float = 0.20,
        max_dp_tf: float  = 0.15,
    ):
        super().__init__()

        # ── stochastic-depth schedules ────────────────────────────────────
        n_cnn = 3
        dp_cnn = [max_dp_cnn * i / max(n_cnn - 1, 1) for i in range(n_cnn)]
        dp_tf  = [max_dp_tf  * i / max(n_tf  - 1, 1) for i in range(n_tf)]

        # ── Branch C: CNN ─────────────────────────────────────────────────
        self.stem   = SpectralStem(in_ch, 128)
        self.s0     = BottleneckBlock(128, 128, stride=1, drop_path=dp_cnn[0])
        self.cbam0  = CBAM(128)
        self.s1     = BottleneckBlock(128, 256, stride=2, drop_path=dp_cnn[1])
        self.cbam1  = CBAM(256)
        self.s2     = BottleneckBlock(256, 384, stride=2, drop_path=dp_cnn[2])
        self.cbam2  = CBAM(384)
        self.gap    = nn.AdaptiveAvgPool2d(1)     # → (B, 384)

        # ── Branch A: Spatial-Spectral Transformer ────────────────────────
        self.tokenizer = SpatialPatchTokenizer(in_ch, patch_grid, token_dim)
        self.pos_enc   = SpatialGridPE(patch_grid, token_dim)
        self.tf_blocks = nn.ModuleList([
            SpectralTransformerBlock(token_dim, n_heads, ffn_mult,
                                     attn_drop=0.05, drop_path=dp_tf[i])
            for i in range(n_tf)
        ])
        self.tf_norm = nn.LayerNorm(token_dim)
        # Token aggregation: mean-pool across tokens → linear(token_dim, 128)
        self.tf_head = nn.Sequential(
            nn.Linear(token_dim, 128), nn.GELU(), nn.Dropout(0.1),
        )

        # ── Branch B: Multiscale 1D ───────────────────────────────────────
        self.ms = MultiscaleSpectralBranch(in_ch, out_ch=32, out_dim=96)

        # ── Fusion Classifier ─────────────────────────────────────────────
        #   [384 (CNN) + 128 (TF) + 96 (MS)] = 608
        #   Stronger dropout (0.50, 0.40) given 235 params/sample.
        self.clf = nn.Sequential(
            nn.Linear(608, 384), nn.GELU(), nn.Dropout(0.50),
            nn.Linear(384, 192), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(192, num_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 256, 64, 64)"""

        # ── Branch A ─────────────────────────────────────────────────────
        tok = self.tokenizer(x)                     # (B, 16, 48)
        tok = tok + self.pos_enc()                  # + spatial PE
        for blk in self.tf_blocks:
            tok = blk(tok)
        tok  = self.tf_norm(tok)
        feat_a = self.tf_head(tok.mean(dim=1))      # (B, 128)

        # ── Branch B ─────────────────────────────────────────────────────
        feat_b = self.ms(x)                         # (B, 96)

        # ── Branch C ─────────────────────────────────────────────────────
        h      = self.stem(x)                       # (B, 128, 64, 64)
        h      = self.cbam0(self.s0(h))             # (B, 128, 64, 64)
        h      = self.cbam1(self.s1(h))             # (B, 256, 32, 32)
        h      = self.cbam2(self.s2(h))             # (B, 384, 16, 16)
        feat_c = self.gap(h).flatten(1)             # (B, 384)

        # ── Fusion ───────────────────────────────────────────────────────
        return self.clf(torch.cat([feat_c, feat_a, feat_b], dim=1))

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm,)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_data(
    x: torch.Tensor, y: torch.Tensor,
    alpha: float = 0.2, enabled: bool = True,
):
    """
    Mixup (Zhang et al., 2018) with two corrections for the 90-class small-data regime:

      1. alpha=0.2  (original used 0.4).
         Beta(0.2, 0.2) has mean=0.5 but mode at extremes — most draws
         are near 0 or 1, so one class dominates the mixed sample.
         Beta(0.4, 0.4) has a flatter distribution, spending much more
         time near 0.5 (equal mix), which destroys per-class signal.

      2. lam = max(lam, 1 - lam)  — dominant-class clamp.
         Forces the primary class to always contribute ≥ 50% of the signal.
         Without this, even with alpha=0.2, some draws would produce a 30/70
         split — catastrophic when a class has only ~67 training samples.

      3. enabled=False during warmup.
         The first MIXUP_WARMUP_EPOCHS epochs train on clean labels so the
         model builds a basic class representation before interpolation starts.
         This was the root cause of train_acc << val_acc in v1: the model
         NEVER saw a clean example during training.
    """
    if not enabled or alpha <= 0.0:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)                           # dominant class ≥ 50%
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


MIXUP_WARMUP = 20   # clean-label epochs before mixup activates


def get_param_groups(model: nn.Module, wd: float):
    """
    Exclude bias and normalization parameters from weight decay.

    Applying L2 penalty to:
      • bias terms  — incorrect: biases should be free to translate activations
      • norm weights/biases (LayerNorm, GroupNorm) — harmful: forces the
        learned scale/shift parameters toward zero, damaging batch statistics

    This is a commonly missed bug that measurably hurts convergence on
    small datasets where every parameter update matters.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # no weight decay for: bias, norm weights (1D), norm biases (1D)
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL INSTANTIATION
# ═══════════════════════════════════════════════════════════════════════════════

model = SpectralFormerNet(
    num_classes=CONFIG["num_classes"],
    in_ch=CONFIG["num_bands"],
    patch_grid=4,           # 4×4 = 16 spatial tokens
    token_dim=48,
    n_heads=4,
    n_tf=2,
    ffn_mult=2,
    max_dp_cnn=0.20,
    max_dp_tf=0.15,
).to(CONFIG["device"])

n_p = sum(p.numel() for p in model.parameters())
print(f"Total parameters      : {n_p / 1e6:.3f} M")
print(f"Params / train sample : {n_p / len(train_idx):.0f}")

# ── Loss ──────────────────────────────────────────────────────────────────────
# No label smoothing:
#   With mixup (soft targets) already applied, adding label smoothing creates
#   a second source of target softening simultaneously.  On an underfitting
#   model this compounds confusion — the correct class receives even less
#   gradient signal.  Re-enable at 0.03–0.05 ONLY if val-train gap closes
#   and overfitting appears (train F1 >> val F1 for many consecutive epochs).
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

# ── Optimiser ─────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(
    get_param_groups(model, wd=5e-4),  # WD=5e-4 only on non-norm, non-bias
    lr=3e-5,                           # OneCycleLR will warm up from this
    betas=(0.9, 0.999),
)

# ── Scheduler: OneCycleLR (per-batch) ─────────────────────────────────────────
#
# max_lr=3e-4 (down from 5e-4 in v1):
#   The val-loss oscillation in v1 was caused by the learning rate being too
#   high relative to the sharpness of the loss landscape for 90-class seeds.
#   Reducing by 40% while widening the warmup phase (15% vs 10%) gives the
#   model time to stabilise before hitting peak LR.
#
# div_factor=10 → start_lr = max_lr / 10 = 3e-5:
#   A gentler ramp than v1's div_factor=5.  The first epoch(s) now use a
#   very small LR, ensuring the randomly-initialised model doesn't make
#   large destructive updates on the first pass through the data.
#
# final_div_factor=1000 → end_lr = start_lr / 1000 = 3e-8:
#   Allows the cosine tail to fully anneal and find the loss minimum.
#   At 3e-8 the model is effectively frozen — stable final convergence.
#
# pct_start=0.15 → warmup over 15% × 300 epochs = 45 epochs:
#   45 epochs of warmup is intentionally longer than the 20-epoch mixup
#   warmup — during the first 20 epochs (clean labels, low LR) the model
#   builds a basic class separation.  Then mixup activates at epoch 21,
#   and the LR continues rising until epoch 45.  This ordering avoids the
#   early plateau seen in v1 (loss flat at epochs 28–50).

steps_per_epoch = len(train_loader)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    epochs=CONFIG["num_epochs"],
    steps_per_epoch=steps_per_epoch,
    pct_start=0.15,
    div_factor=10.0,
    final_div_factor=1000.0,
    anneal_strategy="cos",
)

scaler = GradScaler()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, loader, optimizer, criterion, device,
    mixup_on: bool = True, accum: int = 1,
) -> tuple:
    model.train()

    tot_loss = tot_correct = tot_n = 0
    preds_buf, labels_buf = [], []
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        xm, ya, yb, lam = mixup_data(x, y, alpha=0.2, enabled=mixup_on)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(xm)
            loss   = (lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)) / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()   # per-batch for OneCycleLR

        tot_loss    += loss.item() * x.size(0) * accum
        preds        = logits.argmax(1)
        tot_correct += (preds == y).sum().item()   # accuracy vs true labels
        tot_n       += y.size(0)

        preds_buf.append(preds.detach().cpu())
        labels_buf.append(y.detach().cpu())

    all_p = torch.cat(preds_buf).numpy()
    all_l = torch.cat(labels_buf).numpy()
    return (
        tot_loss / tot_n,
        tot_correct / tot_n,
        f1_score(all_l, all_p, average="macro",    zero_division=0),
        f1_score(all_l, all_p, average="weighted", zero_division=0),
        precision_score(all_l, all_p, average="macro", zero_division=0),
        recall_score(all_l, all_p, average="macro",    zero_division=0),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, phase: str = "Val") -> tuple:
    model.eval()

    tot_loss = tot_correct = tot_n = 0
    preds_buf, labels_buf = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
            loss   = criterion(logits, y)

        tot_loss    += loss.item() * x.size(0)
        preds        = logits.argmax(1)
        tot_correct += (preds == y).sum().item()
        tot_n       += y.size(0)

        preds_buf.append(preds.cpu())
        labels_buf.append(y.cpu())

    all_p = torch.cat(preds_buf).numpy()
    all_l = torch.cat(labels_buf).numpy()
    return (
        tot_loss / tot_n,
        tot_correct / tot_n,
        f1_score(all_l, all_p, average="macro",    zero_division=0),
        f1_score(all_l, all_p, average="weighted", zero_division=0),
        precision_score(all_l, all_p, average="macro", zero_division=0),
        recall_score(all_l, all_p, average="macro",    zero_division=0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

best_f1 = 0.0
patience_ctr = 0

os.makedirs(CONFIG["output_dir"], exist_ok=True)
ckpt_path = os.path.join(CONFIG["output_dir"], "best_spectralformer_v2.pth")

for epoch in range(1, CONFIG["num_epochs"] + 1):

    mixup_on = epoch > MIXUP_WARMUP

    trn = train_one_epoch(
        model, train_loader, optimizer, criterion,
        CONFIG["device"], mixup_on=mixup_on,
    )
    val = evaluate(model, val_loader, criterion, CONFIG["device"])

    lr_now = optimizer.param_groups[0]["lr"]
    print(
        f"Ep {epoch:03d}/{CONFIG['num_epochs']} "
        f"[mx={'on ' if mixup_on else 'off'}] lr={lr_now:.2e}  |  "
        f"Train loss={trn[0]:.4f} acc={trn[1]:.4f} F1={trn[2]:.4f}  |  "
        f"Val   loss={val[0]:.4f} acc={val[1]:.4f} F1={val[2]:.4f}"
    )

    if val[2] > best_f1:
        best_f1 = val[2]
        torch.save(model.state_dict(), ckpt_path)
        patience_ctr = 0
        print(f"  ✓ New best val F1 = {best_f1:.4f}")
    else:
        patience_ctr += 1
        if patience_ctr >= CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch}.")
            break


# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def report(tag: str, m: tuple) -> None:
    print(f"\n{'═'*52}")
    print(f"  {tag}")
    print(f"{'═'*52}")
    print(f"  Loss          {m[0]:.4f}")
    print(f"  Accuracy      {m[1]:.4f}   ({m[1]*100:.1f}%)")
    print(f"  F1  macro     {m[2]:.4f}")
    print(f"  F1  weighted  {m[3]:.4f}")
    print(f"  Precision     {m[4]:.4f}")
    print(f"  Recall        {m[5]:.4f}")


# Last-epoch model
report("FINAL (last epoch) — TEST", evaluate(
    model, test_loader, criterion, CONFIG["device"]
))

# Best-checkpoint model
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
report("BEST CHECKPOINT — TEST", evaluate(
    model, test_loader, criterion, CONFIG["device"]
))