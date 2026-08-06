"""Branch B — masked spectral statistics (9 moments/percentiles per band).

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=================================  ==============
Symbol                             Baseline lines
=================================  ==============
:class:`SpectralStatsBranch`       914-988
=================================  ==============

The ``num_bands`` parameter is accepted but unused in the body (the branch is
band-count agnostic because every stat is pooled over the spatial dimensions);
it is carried across verbatim rather than dropped — removing dead parameters is
an explicit non-goal of this refactor (REFACTOR_PLAN.md §6).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectralquadnet.models.blocks.conv_blocks import ResBlock1D


class SpectralStatsBranch(nn.Module):
    """
    Masked statistical spectral branch.  Pre-computed masked stats
    prevent modal collapse and background dilution.
    """

    def __init__(
        self,
        num_bands: int,
        out_dim: int = 256,
        tower_ch: int = 96,
        wl_pe_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = 9
        self.wl_pe_module = wl_pe_module

        self.stat_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.in_channels, 16, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(16, self.in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch),
            nn.GELU(),
        )

        def _make_tower(kernel: int) -> nn.Sequential:
            return nn.Sequential(
                ResBlock1D(tower_ch, tower_ch, kernel), ResBlock1D(tower_ch, tower_ch, kernel)
            )

        self.tower_s = _make_tower(1)
        self.tower_m = _make_tower(3)
        self.tower_l = _make_tower(5)

        self.fusion = nn.Sequential(
            ResBlock1D(tower_ch * 3, tower_ch, 5), ResBlock1D(tower_ch, tower_ch, 5)
        )
        self.pool_attn = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1, bias=False),
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
                if m.bias is not None:  # FIX-4: guard against bias=False linears
                    nn.init.zeros_(m.bias)

    def forward(self, ms, std, mx, skew, kurt, p10, p25, p75, p90):
        stats = torch.stack([ms, std, mx, skew, kurt, p10, p25, p75, p90], dim=1)
        stats = stats * self.stat_attn(stats)
        x = self.input_proj(stats)

        if self.wl_pe_module is not None:
            x = self.wl_pe_module(x)

        x_fused = self.fusion(torch.cat([self.tower_s(x), self.tower_m(x), self.tower_l(x)], dim=1))
        w = torch.softmax(self.pool_attn(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))
