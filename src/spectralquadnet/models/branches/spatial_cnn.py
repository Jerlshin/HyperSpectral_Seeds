"""Branch C — 2-D spatial texture CNN over the full band cube.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

==============================  ==============
Symbol                          Baseline lines
==============================  ==============
:class:`SpatialCNNBranch`       995-1024
==============================  ==============
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralquadnet.models.blocks.attention import CBAM
from spectralquadnet.models.blocks.conv_blocks import ResBlock2D


class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands: int = 256, out_dim: int = 256) -> None:
        super().__init__()

        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, 1, groups=num_bands, bias=False),
            nn.Conv2d(num_bands, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.stages = nn.Sequential(
            ResBlock2D(64, 128, 2),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.band_reduce(x))
        return self.proj(  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
            F.normalize(
                torch.cat([self._pn(h.mean([2, 3])), self._pn(h.amax([2, 3]))], 1), dim=1, eps=1e-4
            )
        )
