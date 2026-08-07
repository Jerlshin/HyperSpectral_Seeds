"""Stateless spectral feature extractors used by :class:`SpectralQuadNet.forward`.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`extract_grid_spectra`           1367-1386
:func:`masked_spectral_stats`          1388-1422
=====================================  ==============

Both are pure functions of their input tensor (no config, no globals, no RNG),
so they relocate with zero translation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_grid_spectra(x: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    """
    Splits the 64x64 spatial dimensions into a grid and extracts the mean
    spectrum for each grid cell, actively ignoring zero-padded background pixels.
    """
    B, C, H, W = x.shape

    # Create a spatial mask (1 if seed, 0 if background)
    mask = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()  # (B, 1, H, W)

    # Pool the valid signal and the mask area separately
    grid_sum = F.adaptive_avg_pool2d(x * mask, (grid_size, grid_size))
    grid_mask_sum = F.adaptive_avg_pool2d(mask, (grid_size, grid_size))

    # Divide to get the true mean of valid pixels in that specific cell
    # Clamp avoids division by zero for cells that are entirely background
    grid_mean = grid_sum / grid_mask_sum.clamp(min=1e-5)

    # Reshape and transpose to (Batch, Regions, Channels)
    return grid_mean.view(B, C, -1).transpose(1, 2)


def masked_spectral_stats(
    x: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Nine background-masked per-band statistics, each ``(B, C)``.

    Returns:
        ``(mean, std, max, skew, kurtosis, p10, p25, p75, p90)`` — the order
        :class:`~spectralquadnet.models.branches.spectral_stats.SpectralStatsBranch`
        concatenates them in.
    """
    x32 = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)

    mask = (flat.abs().sum(1, keepdim=True) > 1e-5).float()  # (B, 1, HW)
    cnt = mask.sum(2).clamp(min=1.0)  # (B, 1)

    mean = (flat * mask).sum(2) / cnt
    centered = (flat - mean.unsqueeze(2)) * mask
    var = (centered**2).sum(2) / cnt
    std = torch.sqrt(var + 1e-5)
    mx = flat.masked_fill(mask.expand_as(flat) == 0, -1e4).max(2).values
    mx = mx.masked_fill(mx < -9999.0, 0.0)
    skew = torch.clamp(((centered**3).sum(2) / cnt) / (std**3 + 1e-4), -10.0, 10.0)
    kurt = torch.clamp(((centered**4).sum(2) / cnt) / (std**4 + 1e-4), 0.0, 20.0)

    flat_masked = flat.masked_fill(mask.expand_as(flat) == 0, float("inf"))
    sorted_vals, _ = torch.sort(flat_masked, dim=2)

    def gather_percentile(vals: torch.Tensor, p_frac: float) -> torch.Tensor:
        idx = (cnt * p_frac).long().clamp(max=H * W - 1)
        expanded_idx = idx.unsqueeze(2).expand(-1, C, -1)
        return torch.gather(vals, 2, expanded_idx).squeeze(2)

    p10, p25 = gather_percentile(sorted_vals, 0.10), gather_percentile(sorted_vals, 0.25)
    p75, p90 = gather_percentile(sorted_vals, 0.75), gather_percentile(sorted_vals, 0.90)

    return (
        torch.nan_to_num(mean, 0),
        torch.nan_to_num(std, 0),
        torch.nan_to_num(mx, 0),
        torch.nan_to_num(skew, 0),
        torch.nan_to_num(kurt, 0),
        torch.nan_to_num(p10, 0),
        torch.nan_to_num(p25, 0),
        torch.nan_to_num(p75, 0),
        torch.nan_to_num(p90, 0),
    )
