"""SpectralQuadNet model: blocks, branches, fusion, heads and the composition root."""

from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet

__all__ = ["ModelEMA", "SpectralQuadNet"]
