"""Physical-wavelength positional encoding.

The ``pe`` buffer is registered (not a parameter) but **is** part of
``state_dict()`` — it appears in the trained checkpoints as ``wl_pe_cnn.pe`` and
``branch_a.wl_pe_module.pe`` with shape ``(num_bands, tower_ch)``.

BR-4 / T3-3 promoted the encoding itself to a free function,
:func:`sinusoidal_wavelength_encoding`. Branch D's tokens are *not* bands — each
covers a λ-uniform window and has its own centre wavelength — so it cannot share
the band-indexed ``pe`` buffer, but it must share the **encoding**, or "token 3"
and "band 3" would carry positional codes on two different scales. §3.3 BR-4(i)
asks for the module to be shared four ways; sharing the function it is a thin
buffer around is the form of that which survives BR-1 deleting Branch B's band
axis entirely.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_wavelength_encoding(wavelengths: torch.Tensor, d_model: int) -> torch.Tensor:
    """``(n,) -> (n, d_model)`` sinusoidal code keyed to wavelength, not to index.

    The single definition of the encoding. Frequencies are geometric over the
    first half of the feature axis and the two halves hold sine and cosine, so
    two positions that are close in :math:`\\lambda` receive similar codes
    whatever their index distance — which is the whole point when band selection
    has made index adjacency meaningless.
    """
    half = d_model // 2
    freq = torch.exp(
        torch.arange(half, device=wavelengths.device).float()
        * -(math.log(10000.0) / max(half - 1, 1))
    )
    pe = torch.zeros(wavelengths.size(0), d_model, device=wavelengths.device)
    pe[:, :half] = torch.sin(wavelengths.unsqueeze(1) * freq.unsqueeze(0))
    pe[:, half:] = torch.cos(wavelengths.unsqueeze(1) * freq.unsqueeze(0))
    return pe


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
        self.register_buffer("pe", sinusoidal_wavelength_encoding(physical_wl, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.transpose(0, 1).unsqueeze(0)
