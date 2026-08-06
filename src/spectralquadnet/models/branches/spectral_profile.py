"""Branch A — spectral profile (raw signal through multi-scale large kernels).

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

===================================  ==============
Symbol                               Baseline lines
===================================  ==============
:class:`SpectralProfileBranch`       846-907
===================================  ==============
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectralquadnet.models.blocks.conv_blocks import LargeKernelBlock1D


class SpectralProfileBranch(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        tower_ch: int = 96,
        wl_pe_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.wl_pe_module = wl_pe_module

        # Removed Savitzky-Golay; Stem now takes 1 channel (raw signal)
        self.stem = nn.Sequential(
            nn.Conv1d(1, tower_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, tower_ch),
            nn.GELU(),
        )

        # Scaled down Large Kernels for a 40-band input
        self.tower_s = nn.Sequential(
            LargeKernelBlock1D(tower_ch, 3), LargeKernelBlock1D(tower_ch, 3)
        )
        self.tower_m = nn.Sequential(
            LargeKernelBlock1D(tower_ch, 5), LargeKernelBlock1D(tower_ch, 5)
        )
        self.tower_l = nn.Sequential(
            LargeKernelBlock1D(tower_ch, 7), LargeKernelBlock1D(tower_ch, 7)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(tower_ch * 3, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch),
            nn.GELU(),
            LargeKernelBlock1D(tower_ch, 3),
        )

        self.attn_pool = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1),
            nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1),
        )

        self.proj = nn.Sequential(
            nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ms: torch.Tensor) -> torch.Tensor:
        s = ms.unsqueeze(1)
        x = self.stem(s)

        if self.wl_pe_module is not None:
            x = self.wl_pe_module(x)

        x_fused = self.fusion(torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1))

        w = torch.softmax(self.attn_pool(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))
