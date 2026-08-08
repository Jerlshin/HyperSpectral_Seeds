"""FE-1 — a wavelength-aware, spacing-normalised band axis (T3-5).

Every 1-D operator in the pre-Tier-3 model was a ``Conv1d`` over the *band
index*, and the 40 SPA-selected bands are not uniformly spaced: Phase 0's 0-F
measured a :math:`\\max\\Delta\\lambda / \\min\\Delta\\lambda` ratio far from 1, so
"adjacent in index" and "adjacent in nanometres" are different relations and an
index-space kernel is a finite difference on an irregular grid divided by the
wrong step (IMPROVEMENT_PLAN §2.2.6, C-5).

This module supplies the two mechanisms §3.2 FE-1 asks for.

**(b) Explicit derivative channels** — :class:`SpectralDerivatives`. First and
second derivatives with respect to :math:`\\lambda` obtained from a local
weighted least-squares polynomial fit (Savitzky–Golay) whose design matrix is
built from the *actual* wavelength offsets, so the estimator is exact for
polynomials up to ``poly`` regardless of how the bands are spaced. These are the
workhorse features of VIS–NIR chemometrics and the architecture previously could
not compute them correctly.

**(a) Continuous (implicit) kernels** — :class:`LambdaConv1d`. The convolution
weight for the pair *(band i, neighbour j)* is *generated* by a small MLP from a
Fourier featurisation of :math:`\\lambda_j - \\lambda_i`, and the neighbourhood
:math:`\\mathcal{N}_k(i)` is the ``k`` nearest bands **in** :math:`\\lambda`, not
in index. One kernel function is learned once and shared by every band, so the
:math:`\\Delta\\lambda`-dependence is a property of the operator rather than
something each band position has to rediscover.

``model.wl_embed_dim`` — dead since the reference implementation (N-1c) — is the
Fourier-feature width of both, which is the home §3.2 nominates for it.

Scaling convention
──────────────────
``normalise=True`` (the default) multiplies the first-derivative operator by the
median band spacing and the second by its square, so the derivative channels sit
at the same order of magnitude as the reflectance channel instead of
:math:`39\\times` and :math:`1520\\times` above it on the min-max-normalised
:math:`\\lambda` axis. That factor is a **single global constant**: it rescales
every band identically and therefore cannot restore the defect it is applied on
top of — the response ratio between a 2.4 nm gap and a 100 nm gap is untouched,
which is the property ``tests/unit/test_front_end.py`` pins.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def band_spacing_scale(wavelengths: torch.Tensor) -> float:
    """Median spacing between :math:`\\lambda`-consecutive bands.

    The median rather than the mean because the selected subset is irregular by
    construction and a handful of wide gaps would otherwise set the scale for
    the other 35 bands.
    """
    lam, _ = torch.sort(wavelengths.detach().flatten().to(torch.float64))
    if lam.numel() < 2:
        return 1.0
    return float(torch.median(lam[1:] - lam[:-1]).clamp(min=1e-12))


def savitzky_golay_operators(
    wavelengths: torch.Tensor,
    window: int = 7,
    poly: int = 2,
    normalise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(D1, D2)``: dense ``(C, C)`` first/second derivative operators.

    For each band ``i`` the ``window`` nearest bands **in wavelength** form a
    local neighbourhood, a degree-``poly`` polynomial is fitted to the samples
    there in the shifted coordinate :math:`\\lambda_j - \\lambda_i` by ordinary
    least squares, and the fitted derivatives at :math:`\\lambda_i` are read off
    the coefficients: :math:`a_1` and :math:`2a_2`. Because the design matrix
    carries the true offsets, the operator is *exact* on polynomials of degree
    ``poly`` no matter how uneven the grid is — which is precisely what a
    ``Conv1d`` on the index axis cannot be.

    Args:
        wavelengths: ``(C,)`` band wavelengths, in any unit and any order. The
            returned operators differentiate with respect to **this** axis.
        window: Neighbourhood size; clamped to ``[3, C]``.
        poly: Fitted polynomial degree; must be at least 2 for a second
            derivative and is clamped below ``window``.
        normalise: Multiply ``D1`` by the median band spacing and ``D2`` by its
            square — see the module docstring.

    Returns:
        ``(D1, D2)``, each ``(C, C)`` float32, applied as ``F @ D.transpose(0, 1)``
        for ``F`` of shape ``(..., C)``.

    Raises:
        ValueError: ``poly`` resolves below 2, so no second derivative exists.
    """
    lam = wavelengths.detach().cpu().to(torch.float64).flatten()
    
    n_bands = int(lam.numel())
    k = max(3, min(int(window), n_bands))
    degree = min(int(poly), k - 1)
    if degree < 2:
        raise ValueError(
            f"a second derivative needs poly >= 2 and window >= 3; got window={window}, "
            f"poly={poly} on {n_bands} bands"
        )

    # Neighbourhood in λ, not in index. `stable=True` makes ties resolve by band
    # order so the operator is a deterministic function of the wavelength vector.
    distance = (lam[:, None] - lam[None, :]).abs()
    nbr = torch.argsort(distance, dim=1, stable=True)[:, :k]  # (C, k)
    offset = lam[nbr] - lam[:, None]  # (C, k) — Δλ, signed

    powers = torch.arange(degree + 1, dtype=torch.float64)
    vander = offset[..., None] ** powers  # (C, k, degree+1)
    coeffs = torch.linalg.pinv(vander)  # (C, degree+1, k)

    d1 = torch.zeros(n_bands, n_bands, dtype=torch.float64)
    d2 = torch.zeros(n_bands, n_bands, dtype=torch.float64)
    d1.scatter_add_(1, nbr, coeffs[:, 1, :])
    d2.scatter_add_(1, nbr, 2.0 * coeffs[:, 2, :])

    if normalise:
        scale = band_spacing_scale(lam)
        d1 = d1 * scale
        d2 = d2 * (scale**2)

    return d1.to(device=wavelengths.device, dtype=torch.float32), d2.to(device=wavelengths.device, dtype=torch.float32)



class SpectralDerivatives(nn.Module):
    """FE-1(b) — appends :math:`\\partial_\\lambda F` and :math:`\\partial^2_\\lambda F`.

    The two operators are buffers derived once from the wavelength vector, so
    the module has **no parameters** and adds nothing to the budget of §3.8.
    They are persistent (they ride in the checkpoint, like
    ``PhysicalWavelengthPE.pe``) because they are the record of *which*
    wavelength grid a checkpoint was trained on, and a silent change to that
    grid is exactly the kind of drift the gate suite exists to catch.
    """

    d1_op: torch.Tensor
    d2_op: torch.Tensor

    def __init__(
        self,
        wavelengths: torch.Tensor,
        window: int = 7,
        poly: int = 2,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        d1, d2 = savitzky_golay_operators(
            wavelengths, window=window, poly=poly, normalise=normalise
        )
        self.register_buffer("d1_op", d1.to(wavelengths.device))
        self.register_buffer("d2_op", d2.to(wavelengths.device))

    def derivatives(self, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(d1, d2)`` for ``spectra`` of shape ``(..., C)``."""
        d1 = torch.nn.functional.linear(spectra, self.d1_op)
        d2 = torch.nn.functional.linear(spectra, self.d2_op)
        return d1, d2

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """Stack ``[F, d1, d2]`` on a new channel axis: ``(..., C) -> (..., 3, C)``."""
        d1, d2 = self.derivatives(spectra)
        return torch.stack([spectra, d1, d2], dim=-2)


def snv(spectra: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    """Standard Normal Variate along ``dim`` — the per-spectrum gain/offset removal.

    :math:`(\\mathbf{r} - \\bar r) / \\sigma_r` is invariant to
    :math:`\\mathbf{r} \\mapsto a\\mathbf{r} + b` for any :math:`a > 0`, which is
    the multiplicative session gain of §2.1.1 and the per-pixel illumination
    term of §2.2.5 in one operation. Applying it before Branch A's towers is
    what makes that branch's input a *shape* rather than a scaled radiance —
    and what stops it from being reconstructible from Branch D's raw spectra by
    a pointwise map (BR-2).
    """
    mean = spectra.mean(dim=dim, keepdim=True)
    std = spectra.std(dim=dim, keepdim=True, unbiased=False)
    return (spectra - mean) / (std + eps)


class FourierFeatures(nn.Module):
    """Fixed sin/cos featurisation of a scalar offset, ``2 * n_freq`` wide.

    Frequencies are log-spaced from half a period across the offset range up to
    ``2 ** (n_freq - 1)`` periods, so the lowest resolves the overall sign of
    :math:`\\Delta\\lambda` and the highest resolves neighbouring bands. Nothing
    here is learned; the width is ``model.wl_embed_dim``.
    """

    omega: torch.Tensor

    def __init__(self, n_freq: int = 16, span: float = 1.0) -> None:
        super().__init__()
        self.n_freq = int(n_freq)
        base = math.pi / max(float(span), 1e-12)
        omega = base * (2.0 ** torch.arange(self.n_freq, dtype=torch.float32))
        self.register_buffer("omega", omega)

    @property
    def out_features(self) -> int:
        return 2 * self.n_freq

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """``(...,) -> (..., 2 * n_freq)``."""
        phase = offsets.unsqueeze(-1) * self.omega
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class LambdaConv1d(nn.Module):
    """FE-1(a) — a 1-D band-axis convolution whose kernel is a function of :math:`\\Delta\\lambda`.

    .. math::
        (\\mathrm{Conv}_\\lambda F)_{o,i}
        = \\sum_{j \\in \\mathcal{N}_k(i)} \\big[\\kappa_\\phi(\\lambda_j - \\lambda_i)\\big]_{o,c}
          \\, F_{c,j}

    :math:`\\mathcal{N}_k(i)` is the ``k`` nearest bands in wavelength and
    :math:`\\kappa_\\phi` is a two-layer MLP over the Fourier features of the
    signed offset. Two consequences matter:

    * the same physical spacing produces the same weight everywhere on the axis,
      so the operator is a genuine spectral filter rather than 40 unrelated
      index-space taps;
    * the parameter count is independent of the band count — 10.6 k here, against
      the ``Conv1d(1, 96, 3)`` stem's 288, which is the whole of §3.8's ``+7 k``
      line for Branch A.

    The ``LayerNorm`` inside :math:`\\kappa_\\phi` is load-bearing. Every
    ``nn.Linear`` in the model is re-drawn by ``SpectralQuadNet._init_weights``
    at ``trunc_normal_(std=0.02)``, so without a unit-variance hidden state the
    generated kernel's scale would depend on the accident of the feature
    magnitudes; with it, the kernel's initial RMS is
    ``sqrt(hidden) * 0.02``, a known constant.
    """

    nbr: torch.Tensor
    offsets: torch.Tensor

    def __init__(
        self,
        wavelengths: torch.Tensor,
        in_channels: int,
        out_channels: int,
        k: int = 5,
        n_freq: int = 16,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        lam = wavelengths.detach().flatten().float()
        n_bands = int(lam.numel())
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.k = max(1, min(int(k), n_bands))

        distance = (lam[:, None] - lam[None, :]).abs()
        nbr = torch.argsort(distance, dim=1, stable=True)[:, : self.k]
        offsets = lam[nbr] - lam[:, None]

        self.register_buffer("nbr", nbr)
        self.register_buffer("offsets", offsets)

        span = float(offsets.abs().max().clamp(min=1e-6))
        self.features = FourierFeatures(n_freq=n_freq, span=span)
        self.kernel_mlp = nn.Sequential(
            nn.Linear(self.features.out_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.out_channels * self.in_channels),
        )

    def kernel(self) -> torch.Tensor:
        """The generated weight, ``(C, k, out_channels, in_channels)``.

        Depends only on the wavelength grid and :math:`\\phi`, so it is computed
        once per forward and shared across the batch.
        """
        weights: torch.Tensor = self.kernel_mlp(self.features(self.offsets))
        return weights.view(-1, self.k, self.out_channels, self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_channels, C) -> (B, out_channels, C)``."""
        gathered = x[..., self.nbr]  # (B, in, C, k)
        out: torch.Tensor = torch.einsum("ikoc,bcik->boi", self.kernel(), gathered)
        return out
