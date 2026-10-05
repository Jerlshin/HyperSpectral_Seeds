"""S24: remove one learned correction branch while retaining both anchor modalities."""
from __future__ import annotations

from typing import Literal

import torch

from spectralquadnet.models.multimodal_residual import FrozenResidualFusion


class FrozenBranchCorrection(FrozenResidualFusion):
    """Same initialization draw order and head recipe; one active feature branch."""

    def __init__(self, mode: Literal["hsi", "rgb"], hsi_dim: int = 256,
                 rgb_dim: int = 384, hidden: int = 32, classes: int = 90,
                 dropout: float = .2) -> None:
        super().__init__(hsi_dim, rgb_dim, hidden, classes, dropout)
        self.mode = mode
        (self.rgb if mode == "hsi" else self.hsi).requires_grad_(False)

    def forward(self, hsi: torch.Tensor, rgb: torch.Tensor,
                anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hsi(hsi) if self.mode == "hsi" else self.rgb(rgb)
        residual = self.readout(self.dropout(hidden))
        return anchor.clamp_min(1e-8).log() + residual, residual
