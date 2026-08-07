"""The mask is an input, not a threshold — **T3-7 / FE-2**, and BR-1's rank claim.

Every masked operator in the model used to re-derive the foreground from
:math:`\\mathbb{1}[\\sum_c |x_c| > 10^{-5}]`. §3.2 FE-2's validation criterion
for T3-7 is that "masked statistics become invariant to any global brightness
transform", and this module is that criterion: with the persisted fill map
passed explicitly, adding a constant offset to the whole cube must leave the
foreground *support* untouched, where the threshold silently reclassifies every
background pixel as seed.

The second half carries T3-1's criterion. §4.2 asks for 0-D repeated on the new
Branch-B input, with :math:`\\sigma_3/\\sigma_1 > 0.3` where the nine moments
measured far below it. The rank collapse is proved analytically in Appendix A.1
from the gain model :math:`x_{c,p} = a_p r_c`, so the comparison here is run on
data generated from exactly that model — which is what makes the two numbers
comparable rather than two unrelated SVDs.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.models.blocks.attention import MaskedSpectralECA
from spectralquadnet.models.branches.spectral_stats import (
    ContinuumDepths,
    SoftIndexBank,
    SpectralStatsBranch,
)
from spectralquadnet.models.stats_ops import (
    extract_grid_spectra,
    foreground_mask,
    masked_mean_spectrum,
    masked_spectral_stats,
)

pytestmark = pytest.mark.regression

BANDS = 40


@pytest.fixture(scope="module")
def cube_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    """A cube with a zero background and the fill map that describes it."""
    gen = torch.Generator().manual_seed(5)
    x = (torch.rand(4, BANDS, 64, 64, generator=gen) + 0.5).float()
    mask = torch.zeros(4, 1, 64, 64)
    mask[..., 10:50, 20:44] = 1.0
    return x * mask, mask


# ══════════════════════════════════════════════════════════════════════
#  T3-7's validation criterion
# ══════════════════════════════════════════════════════════════════════


def test_the_explicit_mask_survives_a_global_brightness_offset(cube_and_mask) -> None:
    """FE-2's criterion. The threshold does not survive it; the persisted map does.

    Adding a constant lifts the background off zero, so ``sum_c |x_c| > 1e-5``
    now fires on all 4,096 pixels and every masked statistic is computed over
    the frame. This is the mechanism that made the pre-Tier-1 spectral TTA view
    corrupt all four masked modules (§2.6.1, C-8) — the fix there was to
    re-mask, and the fix here is to stop inferring in the first place.
    """
    x, mask = cube_and_mask
    lifted = x + 0.1

    explicit = foreground_mask(lifted, mask)
    inferred = foreground_mask(lifted, None)

    assert torch.equal(explicit, mask)
    assert float(inferred.mean()) == pytest.approx(1.0), "the threshold sees no background"
    assert float(explicit.mean()) < 0.25


def test_masked_statistics_are_invariant_to_the_offset_when_the_mask_is_given(
    cube_and_mask,
) -> None:
    """The consequence, on the statistic Branch B actually reads.

    Under an explicit mask the foreground mean of ``x + c`` is the foreground
    mean of ``x`` plus ``c`` — exactly, on the seed's own pixels. Under the
    threshold it is a mean over the whole frame, which is a different quantity
    diluted by the background fraction.
    """
    x, mask = cube_and_mask
    offset = 0.1

    exact = masked_mean_spectrum(x + offset, mask)
    inferred = masked_mean_spectrum(x + offset, None)

    assert torch.allclose(exact, masked_mean_spectrum(x, mask) + offset, atol=1e-5)
    assert not torch.allclose(exact, inferred, atol=1e-2)


def test_every_masked_operator_takes_the_mask(cube_and_mask) -> None:
    """All four sites §3.2 FE-2 names accept it, and all four react to it."""
    x, mask = cube_and_mask
    half = mask.clone()
    half[..., 30:, :] = 0.0

    eca = MaskedSpectralECA(BANDS)
    grid_full, mass_full = extract_grid_spectra(x, 4, mask)
    grid_half, mass_half = extract_grid_spectra(x, 4, half)

    assert not torch.equal(eca(x, mask), eca(x, half))
    assert not torch.equal(grid_full, grid_half)
    assert not torch.equal(mass_full, mass_half)
    assert not torch.equal(masked_spectral_stats(x, mask)[0], masked_spectral_stats(x, half)[0])


def test_the_fallback_is_the_pre_tier3_behaviour_exactly(cube_and_mask) -> None:
    """``mask=None`` reproduces the threshold bit-for-bit.

    This is what lets ``configs/data/spa40_90class.yaml`` keep reproducing the
    reference run: the arrays that predate T4-3 have no fill map, and a
    "compatible" path that was merely close would make that claim false.
    """
    x, mask = cube_and_mask
    threshold = (x.abs().sum(dim=1, keepdim=True) > 1e-5).float()

    assert torch.equal(foreground_mask(x, None), threshold)
    # On a genuinely zero-background cube the two agree, which is why the
    # fallback is safe *here* and unsafe under any brightness transform.
    assert torch.equal(foreground_mask(x, None), mask)


def test_a_soft_fill_map_is_used_as_a_weight_not_a_predicate(cube_and_mask) -> None:
    """P-3 writes coverage fractions, and M-12 is about what happens if they are rounded.

    A boundary pixel that is 40% seed must contribute 40% of a pixel to the cell
    mean, not a whole one. Halving the coverage of the seed's rows must move the
    cell mass, which a binarising implementation would not do.
    """
    x, mask = cube_and_mask
    soft = mask * 0.4

    _, full_mass = extract_grid_spectra(x, 4, mask)
    _, soft_mass = extract_grid_spectra(x, 4, soft)

    assert torch.allclose(soft_mass, full_mass * 0.4, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
#  T3-1's validation criterion — 0-D repeated on the new input
# ══════════════════════════════════════════════════════════════════════


def _gain_model(n: int = 128, pixels: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Appendix A.1's model: ``x[c, p] = a_p * r_c``, a per-*pixel* illumination gain.

    Returns ``(reflectance, cube)`` — the ``(n, C)`` seed reflectances and the
    ``(n, C, pixels, pixels)`` cube they were observed through. The gain varies
    across pixels within a seed, which is the whole content of §2.2.5: it is
    what makes all nine spatial statistics scalar multiples of the same
    ``r_c``.
    """
    gen = torch.Generator().manual_seed(31)
    reflectance = torch.rand(n, BANDS, generator=gen) * 0.6 + 0.3
    gain = torch.rand(n, 1, pixels, pixels, generator=gen) * 1.5 + 0.5
    return reflectance, reflectance[:, :, None, None] * gain


def _sigma_ratio(matrix: torch.Tensor, k: int = 3) -> float:
    centred = matrix - matrix.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(centred.detach())
    return float(sv[k - 1] / sv[0])


def test_the_nine_moments_collapse_within_every_sample(cube_and_mask) -> None:
    """§2.2.5 / 0-D, reproduced analytically. The companion that stops T3-1 passing vacuously.

    The redundancy is **within a sample**: for one seed under the gain model,
    every one of the nine statistics is ``r_c`` times a moment of that seed's
    gain distribution, so the ``(9, C)`` matrix is rank 1 up to its own mean
    row. That is why 686 k parameters were reading a two-dimensional signal, and
    it is measured here as the mean ``sigma_3 / sigma_1`` over the per-sample
    matrices — the quantity 0-D reported.
    """
    _, cube = _gain_model()
    moments = torch.stack(masked_spectral_stats(cube), dim=1)  # (n, 9, C)

    ratios = [_sigma_ratio(moments[i], k=3) for i in range(moments.shape[0])]

    assert sum(ratios) / len(ratios) < 0.05


def test_the_new_branch_b_input_is_full_rank_across_the_split() -> None:
    """T3-1's criterion: ``sigma_3 / sigma_1 > 0.3`` on the index-bank descriptor.

    The new input is an 88-vector rather than a ``(k, C)`` tensor, so there is
    no within-sample matrix to collapse — which is itself the fix — and the
    rank question moves to where 0-D put it, the design matrix over the split.
    The indices are ratios and survive the gain; the depths are hull-normalised
    and survive it; the morphometrics are not spectral at all.
    """
    torch.manual_seed(0)
    _, cube = _gain_model()
    branch = SpectralStatsBranch(BANDS, torch.linspace(0.0, 1.0, BANDS))
    observed = masked_mean_spectrum(cube)
    morph = torch.randn(observed.shape[0], branch.n_morph)

    features = branch.features(observed, morph)

    assert _sigma_ratio(features) > 0.3


def test_the_old_moments_do_not_reach_that_ratio_within_a_sample(cube_and_mask) -> None:
    """The two numbers side by side, which is what makes the criterion a comparison.

    Reported the way §4.2 asks — "0-D repeated on the new input" — so the claim
    is "0.00x became > 0.3", not "> 0.3 in isolation".
    """
    _, cube = _gain_model()
    moments = torch.stack(masked_spectral_stats(cube), dim=1)
    worst = max(_sigma_ratio(moments[i], k=3) for i in range(moments.shape[0]))

    assert worst < 0.3


def test_the_index_bank_is_invariant_to_the_gain() -> None:
    """BR-1(i)'s defining property, at the operator level."""
    torch.manual_seed(0)
    bank = SoftIndexBank(BANDS, n_indices=64)
    r = torch.rand(16, BANDS) + 0.5

    assert torch.allclose(bank(r), bank(4.2 * r), atol=1e-5)
    assert bank(r).abs().max() <= 1.0


def test_the_continuum_hull_is_exact_on_a_concave_spectrum() -> None:
    """The envelope of a concave function is the function — the sharpest hull check.

    A "rolling ball" approximation with a fixed iteration count fails this by
    however far it has not converged; the pairwise-chord maximum is exact by
    Carathéodory in one dimension.
    """
    lam = torch.linspace(0.0, 1.0, BANDS)
    depths = ContinuumDepths(lam, n_depths=5)
    concave = (-((lam - 0.5) ** 2) + 1.0).unsqueeze(0)

    assert torch.allclose(depths.envelope(concave), concave, atol=1e-6)
    assert torch.allclose(depths(concave), torch.zeros(1, 5), atol=1e-6)


def test_the_hull_finds_an_absorption_feature_and_is_gain_free() -> None:
    """A notch in a flat spectrum has depth ``1 - r/hull``, at any brightness."""
    lam = torch.linspace(0.0, 1.0, BANDS)
    depths = ContinuumDepths(lam, n_depths=3)
    spectrum = torch.ones(1, BANDS)
    spectrum[0, 20] = 0.4

    out = depths(spectrum)

    assert float(out[0, 0]) == pytest.approx(0.6, abs=1e-4)
    assert float(out[0, 1]) == pytest.approx(0.0, abs=1e-4)
    assert torch.allclose(out, depths(spectrum * 7.3), atol=1e-5)


def test_the_hull_buffers_stay_out_of_the_checkpoint() -> None:
    """``O(C^3)`` interpolation weights are a function of λ, not a learned quantity.

    Persisting them would put half a megabyte of derivable constants into every
    checkpoint, and the wavelength vector they derive from is already carried by
    ``wl_pe_cnn.pe``.
    """
    depths = ContinuumDepths(torch.linspace(0.0, 1.0, BANDS), n_depths=8)

    assert depths.state_dict() == {}
    assert depths.chord_w.shape == (BANDS, BANDS, BANDS)
