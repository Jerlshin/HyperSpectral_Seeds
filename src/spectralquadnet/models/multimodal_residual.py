"""S23 additive residual head over frozen HSI/RGB representations.

At initialization the predictor equals its calibrated probability-fusion anchor.
Each modality is mapped independently; no cross-attention or kernel interaction is
assumed. Encoders live outside this module and receive no gradient.
"""
from __future__ import annotations

import torch
from torch import nn


class FrozenResidualFusion(nn.Module):
    """Small additive correction to log equal-fusion probabilities."""

    def __init__(self, hsi_dim: int = 256, rgb_dim: int = 384,
                 hidden: int = 32, classes: int = 90, dropout: float = 0.2) -> None:
        super().__init__()
        self.hsi = nn.Sequential(nn.Linear(hsi_dim, hidden), nn.GELU())
        self.rgb = nn.Sequential(nn.Linear(rgb_dim, hidden), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, classes)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, hsi: torch.Tensor, rgb: torch.Tensor,
                anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.readout(self.dropout(self.hsi(hsi) + self.rgb(rgb)))
        return anchor.clamp_min(1e-8).log() + residual, residual
