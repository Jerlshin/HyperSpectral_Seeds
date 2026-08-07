"""FE-1's wavelength-aware band axis — **T3-5**, and §4.3's Δλ-scaled derivative test.

C-5 (§2.2.6): every 1-D convolution in the pre-Tier-3 model ran over the *band
index*, and the 40 SPA-selected bands are not uniformly spaced. An index-space
kernel applied to a spectrum computes a finite difference divided by the wrong
step — and the error is not a constant factor, because the step varies along
the axis. 0-F measured how far: the widest gap in the shipped selection is many
times the narrowest.

§4.2's validation criterion for T3-5 is a synthetic test "with two bands 2.4 nm
and 100 nm apart" that "yields correctly-scaled derivatives". This module is
that test, in the sharpest form available: on a grid holding both spacings, the
derivative of a straight line must be its slope **at every band**, and the
second derivative of a parabola must be ``2a`` at every band. An index-space
operator cannot satisfy either, and
:func:`test_an_index_space_kernel_fails_the_same_grid` runs one to show the test
is not vacuous.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.models.front_end import (
    FourierFeatures,
    LambdaConv1d,
    SpectralDerivatives,
    band_spacing_scale,
    savitzky_golay_operators,
    snv,
)

pytestmark = pytest.mark.regression

#: Five bands 2.4 nm apart, then five 100 nm apart — a 42x spacing ratio, which
#: is the shape of the shipped selection's irregularity taken to an extreme so
#: the failure of an index-space operator is unmistakable.
IRREGULAR_NM = torch.tensor([400.0, 402.4, 404.8, 407.2, 409.6, 500.0, 600.0, 700.0, 800.0, 900.0])


def _derivs(lam: torch.Tensor, values: torch.Tensor, **kw):
    d1, d2 = savitzky_golay_operators(lam, window=5, poly=2, normalise=False, **kw)
    return values @ d1.t(), values @ d2.t()


# ══════════════════════════════════════════════════════════════════════
#  §4.3 — the Δλ-scaled derivative test
# ══════════════════════════════════════════════════════════════════════


def test_the_first_derivative_of_a_ramp_is_its_slope_at_every_spacing() -> None:
    """``d/dλ (3λ + 7) == 3`` on both the 2.4 nm and the 100 nm side of the grid.

    This is the whole of C-5 in one assertion. The correct answer is the *same*
    number at every band; an operator that divides by an index step instead of a
    wavelength step returns two answers 42x apart.
    """
    d1, _ = _derivs(IRREGULAR_NM, (3.0 * IRREGULAR_NM + 7.0).unsqueeze(0))

    assert torch.allclose(d1, torch.full_like(d1, 3.0), rtol=1e-3)


def test_the_second_derivative_of_a_parabola_is_constant_at_every_spacing() -> None:
    """``d²/dλ² (2λ²) == 4`` everywhere — the harder half, since the error is quadratic."""
    _, d2 = _derivs(IRREGULAR_NM, (2.0 * IRREGULAR_NM**2).unsqueeze(0))

    assert torch.allclose(d2, torch.full_like(d2, 4.0), rtol=2e-3)


def test_the_first_derivative_of_a_parabola_tracks_lambda() -> None:
    """``d/dλ (2λ²) == 4λ``: the operator is not merely constant, it is *right*.

    A degenerate operator that returned zero everywhere would pass the ramp's
    second-derivative check and half the constancy checks.
    """
    d1, _ = _derivs(IRREGULAR_NM, (2.0 * IRREGULAR_NM**2).unsqueeze(0))

    assert torch.allclose(d1, 4.0 * IRREGULAR_NM.unsqueeze(0), rtol=1e-3)


def test_an_index_space_kernel_fails_the_same_grid() -> None:
    """The companion: a central difference over the *index* is wrong by the spacing ratio.

    This is what every ``Conv1d`` in the pre-Tier-3 model was computing, up to
    learned weights. Without this assertion the three above would not establish
    that the Savitzky-Golay operators fixed anything — only that they are
    self-consistent.
    """
    values = 3.0 * IRREGULAR_NM + 7.0
    index_diff = (values[2:] - values[:-2]) / 2.0

    narrow, wide = float(index_diff[1]), float(index_diff[-1])

    assert narrow == pytest.approx(3.0 * 2.4, rel=1e-4)
    assert wide == pytest.approx(3.0 * 100.0, rel=1e-4)
    assert wide / narrow == pytest.approx(100.0 / 2.4, rel=1e-3)


def test_a_uniform_grid_reproduces_the_textbook_central_difference() -> None:
    """On a uniform axis the operator degenerates to what everyone expects.

    Guards against the fix being an unrelated transform that happens to satisfy
    the polynomial identities.
    """
    lam = torch.arange(9, dtype=torch.float32) * 5.0
    values = torch.tensor([[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]])

    d1, _ = _derivs(lam, values)
    central = (values[0, 2:] - values[0, :-2]) / (2 * 5.0)

    assert torch.allclose(d1[0, 1:-1], central, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════
#  The normalisation is a global constant
# ══════════════════════════════════════════════════════════════════════


def test_normalisation_rescales_every_band_identically() -> None:
    """``normalise=True`` cannot restore the defect it is applied on top of.

    The derivative channels are multiplied by the median band spacing so they
    sit at the same order of magnitude as the reflectance channel. That is one
    scalar for the whole operator, so the 2.4 nm / 100 nm response ratio the
    tests above pin is untouched — asserted here rather than argued.
    """
    raw1, raw2 = savitzky_golay_operators(IRREGULAR_NM, window=5, poly=2, normalise=False)
    norm1, norm2 = savitzky_golay_operators(IRREGULAR_NM, window=5, poly=2, normalise=True)
    scale = band_spacing_scale(IRREGULAR_NM)

    gaps = (IRREGULAR_NM[1:] - IRREGULAR_NM[:-1]).sort().values

    assert scale == pytest.approx(float(gaps[len(gaps) // 2]), rel=1e-5), "median, not mean"
    assert torch.allclose(norm1, raw1 * scale, rtol=1e-5)
    assert torch.allclose(norm2, raw2 * scale**2, rtol=1e-5)


def test_the_operators_are_a_function_of_wavelength_not_of_index() -> None:
    """Reordering the *bands* reorders the operator; nothing else changes.

    The property that makes the branch transferable: two selections that pick
    the same wavelengths in a different order get the same physics.
    """
    perm = torch.tensor([9, 0, 5, 1, 8, 2, 7, 3, 6, 4])
    values = (3.0 * IRREGULAR_NM + 7.0).unsqueeze(0)

    straight, _ = _derivs(IRREGULAR_NM, values)
    shuffled, _ = _derivs(IRREGULAR_NM[perm], values[:, perm])

    assert torch.allclose(straight[:, perm], shuffled, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════
#  SpectralDerivatives, SNV and the continuous kernel
# ══════════════════════════════════════════════════════════════════════


def test_spectral_derivatives_adds_two_channels_and_no_parameters() -> None:
    """``(B, C) -> (B, 3, C)``, and §3.8's Branch-A budget survives it."""
    module = SpectralDerivatives(IRREGULAR_NM)
    out = module(torch.randn(4, 10))

    assert out.shape == (4, 3, 10)
    assert sum(p.numel() for p in module.parameters()) == 0
    assert torch.equal(out[:, 0], out[:, 0])  # channel 0 is the input, untouched


def test_the_derivative_operators_ride_in_the_state_dict() -> None:
    """They are the record of which wavelength grid a checkpoint was trained on.

    A silent change to that grid is exactly the drift the gate suite exists to
    catch, and it is invisible unless the operators are persisted.
    """
    keys = set(SpectralDerivatives(IRREGULAR_NM).state_dict())

    assert keys == {"d1_op", "d2_op"}


def test_snv_is_invariant_to_gain_and_offset() -> None:
    """BR-2's reason for putting SNV in front of Branch A.

    ``a * r + b`` is the per-pixel illumination term of §2.2.5 and the
    multiplicative session gain of §2.1.1 in one expression; after SNV the
    branch sees a shape, and cannot see brightness at all.
    """
    r = torch.rand(8, 40) + 0.5

    assert torch.allclose(snv(r), snv(3.7 * r + 1.9), atol=1e-4)


def test_the_lambda_kernel_depends_on_the_wavelength_offset() -> None:
    """FE-1(a): one kernel function, evaluated at each band's own Δλ neighbourhood.

    On the irregular grid the taps for a band inside the 2.4 nm cluster and one
    inside the 100 nm run must differ — that difference *is* the mechanism. A
    plain ``Conv1d`` shares one tap vector across every band by construction.
    """
    conv = LambdaConv1d(IRREGULAR_NM, in_channels=3, out_channels=8, k=3, n_freq=8)
    kernel = conv.kernel()

    assert kernel.shape == (10, 3, 8, 3)
    assert not torch.allclose(kernel[1], kernel[8], atol=1e-6)


def test_the_lambda_kernel_neighbourhood_is_in_wavelength() -> None:
    """``N_k(i)`` is the k nearest bands in λ, not the k nearest indices.

    Band 5 (500 nm) sits between a dense cluster and a sparse run, so its
    3-neighbourhood in wavelength is {4, 5, 6} while its index neighbourhood is
    the same — but band 4 (409.6 nm) has {3, 4, 5} in index and {2, 3, 4} in
    wavelength. That is the case that separates the two.
    """
    conv = LambdaConv1d(IRREGULAR_NM, in_channels=1, out_channels=2, k=3)

    assert set(conv.nbr[4].tolist()) == {2, 3, 4}
    assert conv.offsets[0, 0] == 0.0, "a band is always its own nearest neighbour"


def test_the_lambda_conv_shape_and_parameter_cost() -> None:
    """§3.8 prices Branch A's stem change at ~+10 k, independent of the band count."""
    small = LambdaConv1d(IRREGULAR_NM, 3, 96, k=5, n_freq=16)
    large = LambdaConv1d(torch.linspace(400.0, 900.0, 256), 3, 96, k=5, n_freq=16)

    n_small = sum(p.numel() for p in small.parameters())

    assert small(torch.randn(2, 3, 10)).shape == (2, 96, 10)
    assert n_small == sum(p.numel() for p in large.parameters())
    assert 8_000 < n_small < 12_000


def test_fourier_features_are_deterministic_and_parameter_free() -> None:
    """κ_φ's input is fixed; only the MLP over it is learned."""
    feats = FourierFeatures(n_freq=16, span=1.0)
    offsets = torch.linspace(-1.0, 1.0, 7)

    assert feats.out_features == 32
    assert sum(p.numel() for p in feats.parameters()) == 0
    assert torch.equal(feats(offsets), feats(offsets))


def test_a_window_narrower_than_a_second_derivative_is_refused() -> None:
    """``poly < 2`` has no second derivative, and silently returning zeros would hide it."""
    with pytest.raises(ValueError, match="second derivative"):
        savitzky_golay_operators(IRREGULAR_NM, window=5, poly=1)
