"""Physical-wavelength positional encoding.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

==================================  ==============
Symbol                              Baseline lines
==================================  ==============
:class:`PhysicalWavelengthPE`       796-811
==================================  ==============

The ``pe`` buffer is registered (not a parameter) but **is** part of
``state_dict()`` — it appears in the trained checkpoints as ``wl_pe_cnn.pe`` and
``branch_{a,b}.wl_pe_module.pe`` with shape ``(num_bands, tower_ch)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhysicalWavelengthPE(nn.Module):
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
