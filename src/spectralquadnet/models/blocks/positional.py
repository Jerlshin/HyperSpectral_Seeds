"""Physical-wavelength positional encoding.

The ``pe`` buffer is registered (not a parameter) but **is** part of
``state_dict()`` — it appears in the trained checkpoints as ``wl_pe_cnn.pe`` and
``branch_{a,b}.wl_pe_module.pe`` with shape ``(num_bands, tower_ch)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhysicalWavelengthPE(nn.Module):
    """Sinusoidal positional encoding keyed to actual sensor wavelengths, not band index.

    Unlike a standard Transformer position embedding, the encoding frequency is
    driven by the physical wavelength of each band, so bands that are close in
    wavelength (and thus physically correlated) receive similar encodings even
    if band-selection has made them non-adjacent in index order.
    """

    #: Declares the `register_buffer("pe", ...)` below for the type checker; see
    #: the same note on `AdaptiveSubcenterArcFaceHead.margins`.
    pe: torch.Tensor

    def __init__(self, physical_wl: torch.Tensor, d_model: int) -> None:
        super().__init__()
        dev = physical_wl.device
        half = d_model // 2
        freq = torch.exp(
            torch.arange(half, device=dev).float() * -(math.log(10000.0) / max(half - 1, 1))
        )
        pe = torch.zeros(physical_wl.size(0), d_model, device=dev)
        pe[:, :half] = torch.sin(physical_wl.unsqueeze(1) * freq.unsqueeze(0))
        pe[:, half:] = torch.cos(physical_wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.transpose(0, 1).unsqueeze(0)
