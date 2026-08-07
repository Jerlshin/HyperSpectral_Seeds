"""Stateless spectral feature extractors used by :class:`SpectralQuadNet.forward`.

Both functions are pure — they depend only on their arguments (no config, no
module state, no RNG) — and operate on background-masked hyperspectral cubes.

FE-2 / T3-7 — the mask is an input
──────────────────────────────────
Every masked operator here used to *re-derive* the foreground from
:math:`\\mathbb{1}[\\sum_c |x_c| > 10^{-5}]`. That threshold is a proxy, and it
is wrong in three separate places:

* it breaks under any brightness transform that shifts the background off zero,
  which is what made the pre-Tier-1 spectral TTA view corrupt all four masked
  modules (§2.6.1, C-8);
* it is binary, so a resized boundary pixel that is 40 % seed counts as a whole
  seed pixel and dilutes the cell mean it lands in (§2.2.9, M-12);
* it cannot distinguish "background" from "a band this seed genuinely reflects
  nothing in".

P-3 / T4-3 writes the true fill map :math:`\\alpha \\in [0, 1]^{S \\times S}` to
``masks.npy``. Both functions now take it as ``mask`` and use it as a **weight**,
not a predicate, so partial-coverage pixels contribute in proportion to their
coverage. ``mask=None`` keeps the old threshold, verbatim, so a run against the
arrays that predate T4-3 reproduces bit-for-bit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def foreground_mask(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Resolve the ``(B, 1, H, W)`` foreground weight for a cube.

    The single place the FE-2 fallback lives, so the four masked modules cannot
    disagree about what "background" means.

    Args:
        x: ``(B, C, H, W)`` cube.
        mask: ``(B, H, W)`` or ``(B, 1, H, W)`` fill map in ``[0, 1]``. When
            ``None``, the pre-Tier-3 band-sum threshold is used.
    """
    if mask is None:
        return (x.abs().sum(dim=1, keepdim=True) > 1e-5).to(x.dtype)
    m = mask if mask.dim() == 4 else mask.unsqueeze(1)
    return m.to(dtype=x.dtype, device=x.device)


def extract_grid_spectra(
    x: torch.Tensor,
    grid_size: int = 4,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cell mean spectra on a ``grid_size × grid_size`` grid, plus each cell's mass.

    Background pixels are excluded from each cell's mean so that padding never
    dilutes the seed's spectral signal.

    The second return value is BR-2's :math:`\\omega_n` — the cell's mean
    foreground coverage, which is the normaliser this function has always
    computed and, before T3-6, always discarded. A corner cell of a 64×64 patch
    holding a rice seed is mostly background, so its mean spectrum is estimated
    from a handful of pixels; weighting the cells by :math:`\\omega_n` when they
    are pooled is what stops those estimates from counting as much as a cell
    that is entirely seed (§2.2.9, M-12).

    Args:
        x: Hyperspectral cube, shape ``(B, C, H, W)``.
        grid_size: Number of grid cells per spatial axis.
        mask: Optional explicit fill map — see :func:`foreground_mask`.

    Returns:
        ``(grid_mean, cell_mass)`` of shapes ``(B, grid_size ** 2, C)`` and
        ``(B, grid_size ** 2)``.
    """
    n_batch, n_bands = x.shape[0], x.shape[1]
    m = foreground_mask(x, mask)

    # Pool the valid signal and the mask area separately
    grid_sum = F.adaptive_avg_pool2d(x * m, (grid_size, grid_size))
    grid_mask_sum = F.adaptive_avg_pool2d(m, (grid_size, grid_size))

    # Divide to get the true mean of valid pixels in that specific cell
    # Clamp avoids division by zero for cells that are entirely background
    grid_mean = grid_sum / grid_mask_sum.clamp(min=1e-5)

    # Reshape and transpose to (Batch, Regions, Channels)
    spectra = grid_mean.view(n_batch, n_bands, -1).transpose(1, 2)
    mass = grid_mask_sum.view(n_batch, -1)
    return spectra, mass


def masked_spectral_stats(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
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

    §2.2.5 proves this tensor has rank :math:`\\le 2` under the per-pixel gain
    model, which is why BR-1 / T3-1 stopped feeding it to Branch B. It is kept —
    and kept correct — because ``masked_spectral_stats(x)[0]``, the foreground
    mean spectrum, is the input the new Branch B's index bank and continuum
    hull are computed from, and because the diagnostic scripts of Phase 0
    measure the rank collapse against exactly this function.

    Returns:
        ``(mean, std, max, skew, kurtosis, p10, p25, p75, p90)``.
    """
    x32 = x.float()
    B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)

    m = foreground_mask(x32, mask).reshape(B, 1, H * W)
    cnt = m.sum(2).clamp(min=1.0)  # (B, 1)

    mean = (flat * m).sum(2) / cnt
    centered = (flat - mean.unsqueeze(2)) * m
    var = (centered**2).sum(2) / cnt
    std = torch.sqrt(var + 1e-5)
    solid = m.expand_as(flat) <= 0.0
    mx = flat.masked_fill(solid, -1e4).max(2).values
    mx = mx.masked_fill(mx < -9999.0, 0.0)
    skew = torch.clamp(((centered**3).sum(2) / cnt) / (std**3 + 1e-4), -10.0, 10.0)
    kurt = torch.clamp(((centered**4).sum(2) / cnt) / (std**4 + 1e-4), 0.0, 20.0)

    flat_masked = flat.masked_fill(solid, float("inf"))
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


def masked_mean_spectrum(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The foreground mean spectrum, ``(B, C)`` — Branch B's whole input signal.

    Separated from :func:`masked_spectral_stats` because after BR-1 that is the
    only one of the nine the network still consumes, and computing the other
    eight (two sorts over 4,096 pixels among them) to throw them away was a
    measurable share of the forward pass.
    """
    n_batch, n_bands = x.shape[0], x.shape[1]
    flat = x.float().reshape(n_batch, n_bands, -1)
    m = foreground_mask(x.float(), mask).reshape(n_batch, 1, -1)
    return torch.nan_to_num((flat * m).sum(2) / m.sum(2).clamp(min=1.0), 0.0)
