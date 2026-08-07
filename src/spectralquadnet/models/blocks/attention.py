"""Channel/spatial attention blocks shared across the model's branches and fusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MaskedSpectralECA(nn.Module):
    """
    Residual Channel Attention for Hyperspectral data.
    Prevents band suppression using background masking, local cross-band
    convolutions (ECA), and a residual connection.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Dynamically calculate ECA 1D-Conv kernel size based on channel count
        t = int(abs(math.log2(channels) / 2.0 + 1.0))
        k_size = t if t % 2 != 0 else t + 1

        # 1D Conv processes adjacent spectral bands together to find continuous features
        self.conv = nn.Conv1d(2, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Background-Aware Masking
        mask = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()
        valid_pixels = mask.sum(dim=[2, 3]).clamp(min=1e-5)

        # 2. Extract accurate physical statistics (strictly ignoring the black background)
        x_mean = (x * mask).sum(dim=[2, 3]) / valid_pixels
        x_max = x.masked_fill(mask == 0, -1e4).amax(dim=[2, 3])
        x_max = x_max.masked_fill(x_max == -1e4, 0.0)

        # 3. Stack for 1D Convolution: Shape (Batch, 2, Channels)
        y = torch.stack([x_mean, x_max], dim=1)

        # 4. Local Cross-Band Interaction
        # Output shape: (Batch, 1, Channels) -> permute to (Batch, Channels, 1, 1)
        gate = torch.sigmoid(self.conv(y)).permute(0, 2, 1).unsqueeze(-1)

        # 5. Residual Excitation (Enhance, do not suppress)
        # Weights range from 1.0x to 2.0x, ensuring no band is ever deleted.
        return x + (x * gate)


class SEBlock1D(nn.Module):
    """1D Squeeze-and-Excitation to dynamically re-weight feature channels."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any


class CBAM(nn.Module):
    """Convolutional Block Attention Module: sequential channel then spatial gating."""

    def __init__(self, c: int, r: int = 8) -> None:
        super().__init__()
        mid = max(c // r, 8)
        self.ch = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False), nn.GELU(), nn.Conv2d(mid, c, 1, bias=False)
        )
        self.sp = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(
            self.ch(x.mean([2, 3], keepdim=True)) + self.ch(x.amax([2, 3], keepdim=True))
        )
        return x * self.sp(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1))  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
