"""Models: blocks, branches, fusion, heads, and the two composition roots.

Two architectures ship and :func:`~spectralquadnet.models.registry.build_model`
is the only thing that should choose between them — see that module for why
both are load-bearing.

* :class:`~spectralquadnet.models.spectral_seed_net.SpectralSeedNet` — the
  default. Two pathways, ≈2.82 M parameters (CHANGES §16.2).
* :class:`~spectralquadnet.models.spectral_quadnet.SpectralQuadNet` — the
  audited four-branch model, ≈5.19 M. Retained unmodified as the control arm
  for ablations A3 and A8.
"""

from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import (
    ARCHITECTURES,
    build_model,
    count_parameters,
    describe,
    parameter_breakdown,
)
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet

__all__ = [
    "ARCHITECTURES",
    "ModelEMA",
    "SpectralQuadNet",
    "SpectralSeedNet",
    "build_model",
    "count_parameters",
    "describe",
    "parameter_breakdown",
]
