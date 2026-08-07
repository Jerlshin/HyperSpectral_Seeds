"""Branch C — a joint spectral–spatial stem over the full cube (BR-3 / T3-2).

C-3, in one line: the network contained **no joint spectral–spatial operator**.
``band_reduce`` was ``Conv2d(C, C, 1, groups=C)`` followed by ``Conv2d(C, 64, 1)``
— two 1×1 convolutions, so the band axis was collapsed to 64 channels *before*
any spatial kernel ran, and the very first 3×3 saw a spectrally-mixed feature
map. The function "this absorption feature, in this part of the seed" was not in
the hypothesis class of any module in the model.

BR-3 replaces the two 1×1s with a factorised 3-D stem that keeps the spectral
axis alive for three stages before folding it::

    x  (B, 1, 40, 64, 64)
     ├─ Conv3d(1→16,  k=(7,3,3), s=(2,1,1))  → (B,16,20,64,64)
     ├─ Conv3d(16→32, k=(5,3,3), s=(2,2,2))  → (B,32,10,32,32)
     ├─ Conv3d(32→64, k=(5,3,3), s=(2,2,2))  → (B,64, 5,16,16)
     └─ reshape (B, 320, 16, 16) → 1×1 → (B,192,16,16)
          └─ the existing ResBlock2D / CBAM tail

The spectral axis is *folded* into the channel axis at the end, not deleted at
the start: each of the 320 channels entering the 1×1 is a (spectral position ×
learned feature) pair, so the tail's 3×3 kernels operate on features that still
carry where in the spectrum they came from.

The stem costs ≈ 116 k parameters, comfortably inside the ≈ 600 k BR-1 freed
from Branch B, and the branch as a whole moves from 1.69 M to ≈ 2.23 M. That is
the reallocation §3.8 is about: Branch C is the only branch whose input is not
reconstructible from any other, and it was the one starved of capacity.

The persisted mask (FE-2 / T3-7) is applied after every stage, so a padded
region stays exactly zero however deep the stack goes and the CNN can never
learn the frame instead of the seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralquadnet.models.blocks.attention import CBAM
from spectralquadnet.models.blocks.conv_blocks import ResBlock2D
from spectralquadnet.models.stats_ops import foreground_mask


class SpectralSpatialStem3D(nn.Module):
    """Three strided 3-D convolutions, then a 1×1 fold of the spectral axis.

    Only the first stage leaves the spatial resolution alone: the spectral axis
    is halved first, so the expensive 3-D kernels run on a shrinking cube while
    the 64×64 spatial detail is still intact when the widest spectral kernel
    (``k = 7`` bands) passes over it.
    """

    def __init__(self, num_bands: int = 40, out_channels: int = 192) -> None:
        super().__init__()
        self.num_bands = int(num_bands)

        self.stage1 = nn.Sequential(
            nn.Conv3d(1, 16, (7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1), bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(16, 32, (5, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(32, 64, (5, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1), bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # Spectral depth after three stride-2 spectral steps, e.g. 40 → 20 → 10 → 5.
        depth = self.num_bands
        for _ in range(3):
            depth = (depth + 1) // 2
        self.folded_depth = depth

        self.fold = nn.Sequential(
            nn.Conv2d(64 * depth, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _apply_mask(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Re-zero the padded region at this stage's spatial resolution.

        ``mask`` is ``(B, 1, H, W)`` at input resolution; it is area-pooled to
        whatever ``h``'s spatial size now is, which is the same operation the
        stride performed on the signal.
        """
        if mask is None:
            return h
        m = F.adaptive_avg_pool2d(mask, (int(h.shape[-2]), int(h.shape[-1])))
        return h * m.unsqueeze(2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, C, H, W) -> (B, out_channels, H // 4, W // 4)``."""
        h = self._apply_mask(self.stage1(x.unsqueeze(1)), mask)
        h = self._apply_mask(self.stage2(h), mask)
        h = self._apply_mask(self.stage3(h), mask)
        n_batch = h.shape[0]
        folded = h.reshape(n_batch, -1, h.shape[-2], h.shape[-1])
        return self.fold(folded)  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any


class SpatialCNNBranch(nn.Module):
    """3-D spectral–spatial stem into a CBAM-gated ResNet-2D stack over spatial texture.

    Pools the fused feature map with concatenated signed-power-normalised mean
    and max statistics, which is more stable than raw mean/max pooling under
    the heavy-tailed activations a ResNet stack produces.
    """

    def __init__(self, num_bands: int = 256, out_dim: int = 256, stem_channels: int = 192) -> None:
        super().__init__()

        self.stem = SpectralSpatialStem3D(num_bands, stem_channels)

        self.stages = nn.Sequential(
            ResBlock2D(stem_channels, 128, 2),
            CBAM(128),
            ResBlock2D(128, 192, 2),
            CBAM(192),
            ResBlock2D(192, 256, 2),
            CBAM(256),
            ResBlock2D(256, out_dim, 2),
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    @staticmethod
    def _pn(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().clamp(1e-8).sqrt()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stages(self.stem(x, foreground_mask(x, mask)))
        return self.proj(  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
            F.normalize(
                torch.cat([self._pn(h.mean([2, 3])), self._pn(h.amax([2, 3]))], 1), dim=1, eps=1e-4
            )
        )
