"""Residual convolution blocks (1-D spectral and 2-D spatial).

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

===============================  ==============
Symbol                           Baseline lines
===============================  ==============
:class:`ResBlock1D`              738-754
:class:`ResBlock2D`              774-793
:class:`LargeKernelBlock1D`      817-843
===============================  ==============

Bodies are byte-identical to the baseline; ``SEBlock1D`` is now imported from
:mod:`spectralquadnet.models.blocks.attention` instead of being a module-level
neighbour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralquadnet.models.blocks.attention import SEBlock1D


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.se = SEBlock1D(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + identity)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(
            self.n3(self.c3(F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) + self.skip(x)
        )


class LargeKernelBlock1D(nn.Module):
    """
    Modern ConvNeXt-inspired 1D block.
    Uses Depthwise Large Kernels to continuously capture wide absorption valleys
    without the 'blind spots' of dilated convolutions.
    """

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        # Depthwise convolution (groups=dim) makes large kernels highly parameter-efficient
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False
        )
        self.norm = nn.GroupNorm(1, dim)

        # Pointwise Feed-Forward Network (Inverted Bottleneck)
        self.pw1 = nn.Conv1d(dim, dim * 4, 1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(dim * 4, dim, 1, bias=False)
        self.se = SEBlock1D(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.se(x)
        return x + res
