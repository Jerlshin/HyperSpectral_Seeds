"""T4-2 / P-2 — radiometric normalisation.

The claim SNV has to earn is algebraic, so the tests are algebraic too: under
the gain model :math:`x_{c,p} = a_p r_c` of §2.2.5 and Appendix A.1, SNV
returns :math:`(r_c - \\bar r)/\\mathrm{sd}(r)` — **independent of** :math:`a_p`
— and the session gain :math:`S_c` of §2.1.1 cancels the same way. Those two
identities are C-1's radiometric half, and
:func:`test_snv_removes_a_per_pixel_gain_exactly` /
:func:`test_snv_removes_a_session_gain_exactly` measure them to float32.

Each is paired with a companion showing the **radiance domain** — what the
shipped arrays are in — failing the same test, so the fix is not asserted
against itself: :func:`test_the_radiance_domain_leaks_the_session`
demonstrates the leak channel by recovering the session id from untouched
spectra with a one-line rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectralquadnet.data.prep.radiometry import (
    RADIOMETRY_EPS,
    RADIOMETRY_MODES,
    apply_radiometry,
    find_white_reference,
    resolve_radiometry,
    snv_normalise,
    white_gain,
    white_reference_correct,
)

C = 40
H = W = 8


@pytest.fixture
def reflectance() -> np.ndarray:
    """One "true" spectrum r_c, strictly positive, with structure across bands."""
    rng = np.random.default_rng(0)
    return (1.0 + rng.random(C)).astype(np.float32)


@pytest.fixture
def gains() -> np.ndarray:
    """Per-pixel geometric gain a_p, spanning an order of magnitude."""
    rng = np.random.default_rng(1)
    return (0.2 + 2.0 * rng.random((H, W))).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  SNV — the two identities
# ══════════════════════════════════════════════════════════════════════


def test_snv_removes_a_per_pixel_gain_exactly(reflectance, gains) -> None:
    """``x_{c,p} = a_p r_c`` ⟹ SNV gives ``(r - mean r)/sd r`` at every pixel.

    This is the whole of P-2(b): the pixel-to-pixel brightness variation that
    Appendix A.1 shows is what collapses the ``(9, 40)`` statistics tensor to
    rank 2 is divided out identically, not approximately.
    """
    patch = gains[..., None] * reflectance[None, None, :]
    out, mean, sd = snv_normalise(patch)

    expected = (reflectance - reflectance.mean()) / reflectance.std()
    for i, j in ((0, 0), (3, 5), (H - 1, W - 1)):
        np.testing.assert_allclose(out[i, j], expected, rtol=2e-4, atol=2e-4)

    # And the gain it removed is exactly what it reports.
    np.testing.assert_allclose(mean, gains * reflectance.mean(), rtol=1e-5)
    np.testing.assert_allclose(sd, gains * reflectance.std(), rtol=1e-5)


def test_snv_removes_a_session_gain_exactly(reflectance, gains) -> None:
    """A per-session multiplier ``S`` leaves the SNV spectrum untouched (C-1).

    §2.1.1's leak is that the illumination of a session multiplies every
    spectrum captured in it, and — with an ungrouped split — that multiplier is
    perfectly predictive of the label. After SNV, two acquisitions of the same
    seed under different illumination are the *same vector*.
    """
    patch = gains[..., None] * reflectance[None, None, :]
    day_one, _, _ = snv_normalise(patch)
    day_two, _, _ = snv_normalise(3.7 * patch)
    # Exact in real arithmetic; 1e-5 is float32's accumulation over the 40-band
    # mean and standard deviation, against a signal of order 1.
    np.testing.assert_allclose(day_one, day_two, rtol=2e-5, atol=1e-5)


def test_the_radiance_domain_leaks_the_session(reflectance, gains) -> None:
    """The companion: without P-2, the session is recoverable from the spectra.

    A rule as crude as "is the mean radiance above 2?" separates two sessions
    of the *same* seed perfectly. That is the channel §2.1.1 describes, and it
    is what a network is free to use when the split does not forbid it.
    """
    patch = gains[..., None] * reflectance[None, None, :]
    bright, dim = 3.7 * patch, 0.4 * patch

    assert bright.mean() > dim.mean() * 5  # trivially separable in radiance
    snv_bright, _, _ = snv_normalise(bright)
    snv_dim, _, _ = snv_normalise(dim)
    assert abs(snv_bright.mean() - snv_dim.mean()) < 1e-5  # and not, after SNV


# ══════════════════════════════════════════════════════════════════════
#  SNV — the background invariant
# ══════════════════════════════════════════════════════════════════════


def test_the_background_stays_exactly_zero(reflectance, gains) -> None:
    """SNV of a zero spectrum would be an arbitrary constant; the mask forbids it.

    Every masked operator downstream — ``masked_spectral_stats``, the ECA
    gate, all 12 TTA views — infers the foreground by testing for zero (§2.6),
    so a non-zero background is not a cosmetic defect.
    """
    patch = (gains[..., None] * reflectance[None, None, :]).copy()
    mask = np.zeros((H, W), dtype=bool)
    mask[2:5, 3:7] = True
    patch[~mask] = 0.0

    out, mean, sd = snv_normalise(patch, mask)
    assert np.all(out[~mask] == 0.0)
    assert np.all(mean[~mask] == 0.0)
    assert np.all(sd[~mask] == 0.0)
    assert np.any(out[mask] != 0.0)


def test_snv_amplifies_a_background_residual_to_full_scale(reflectance, gains) -> None:
    """The companion, and the reason P-2 needs P-3.

    SNV divides by a pixel's own spread, so it is *scale-blind by
    construction*: a background pixel left at a thousandth of a real spectrum —
    which is what the pre-T4-3 resize order produced on partial-coverage pixels
    (M-11) — comes out of SNV as a unit-variance spectrum **numerically
    indistinguishable from a seed**. An exactly-zero background survives
    untouched, because ``0/eps`` is 0, and residuals below ~1e-6 are damped by
    ``eps``; anything above that is amplified to full scale. That is why the
    mask is an argument here, and why T4-3 has to make the background exact
    rather than small.
    """
    patch = (gains[..., None] * reflectance[None, None, :]).copy()
    patch[0, 0] = 1e-3 * reflectance  # partial-coverage residual
    mask = np.ones((H, W), dtype=bool)
    mask[0, 0] = False

    unmasked, _, _ = snv_normalise(patch)
    assert np.abs(unmasked[0, 0]).max() > 0.5  # amplified to full scale
    np.testing.assert_allclose(unmasked[0, 0], unmasked[1, 1], rtol=1e-2, atol=1e-2)

    masked, _, _ = snv_normalise(patch, mask)
    assert np.all(masked[0, 0] == 0.0)


def test_a_constant_spectrum_does_not_divide_by_zero() -> None:
    """A flat pixel has zero spread; ``eps`` keeps it finite rather than NaN."""
    patch = np.full((2, 2, C), 5.0, dtype=np.float32)
    out, _, sd = snv_normalise(patch)
    assert np.all(np.isfinite(out))
    assert np.allclose(sd, 0.0)


# ══════════════════════════════════════════════════════════════════════
#  White reference — P-2(a)
# ══════════════════════════════════════════════════════════════════════


def test_white_reference_division_recovers_reflectance() -> None:
    """``(R - D) / (W - D)`` returns the reflectance a synthetic sensor encoded."""
    rng = np.random.default_rng(2)
    rho = rng.random((6, W, C)).astype(np.float32)
    dark = np.full((4, W, C), 7.0, dtype=np.float32)
    illum = (100.0 + 50.0 * rng.random((W, C))).astype(np.float32)

    white = (7.0 + illum)[None].repeat(4, axis=0).astype(np.float32)
    raw = 7.0 + rho * illum[None]

    np.testing.assert_allclose(white_reference_correct(raw, white, dark), rho, rtol=1e-4, atol=1e-4)


def test_white_gain_is_the_denominator_alone() -> None:
    """Split out because ``preprocess_raw`` has already subtracted the dark mean."""
    white = np.full((3, W, C), 12.0, dtype=np.float32)
    dark = np.full((3, W, C), 2.0, dtype=np.float32)
    gain = white_gain(white, dark)
    assert gain.shape == (1, W, C)
    np.testing.assert_allclose(gain, 10.0 + RADIOMETRY_EPS, rtol=1e-6)


def test_the_archive_has_no_white_panel() -> None:
    """P-2(a) is unavailable here, which is *why* the shipped resolution is SNV.

    The Zenodo rice archive's reference cubes are ``black.hdr`` — dark current,
    which ``preprocess_raw`` already consumes — plus a ``chessboard`` geometric
    target. Neither is a reflectance standard.
    """
    members = [
        "rice/Data-VIS-20170111-1/BC15-01.hdr",
        "rice/Data-VIS-20170111-1/black.hdr",
        "rice/chessboard/chessboard.hdr",
        "rice/wavelengths.csv",
    ]
    assert find_white_reference(members) is None
    assert find_white_reference(members + ["rice/white_panel.hdr"]) == "rice/white_panel.hdr"


def test_a_dark_reference_is_never_mistaken_for_a_white_one() -> None:
    assert find_white_reference(["a/whiteblack.hdr"]) is None


# ══════════════════════════════════════════════════════════════════════
#  Mode resolution
# ══════════════════════════════════════════════════════════════════════


def test_auto_falls_back_to_snv_when_there_is_no_panel() -> None:
    mode, why = resolve_radiometry("auto", None)
    assert mode == "snv"
    assert "no white panel" in why


def test_auto_prefers_the_white_panel_when_one_exists() -> None:
    mode, why = resolve_radiometry("auto", "a/white.hdr")
    assert mode == "white"
    assert "white.hdr" in why


def test_requesting_white_without_a_panel_is_an_error_not_a_silent_fallback() -> None:
    """A radiometric domain must never be decided by a silent fallback."""
    with pytest.raises(ValueError, match="requires a white panel"):
        resolve_radiometry("white", None)


def test_none_is_available_so_the_leak_stays_reproducible() -> None:
    """The pre-Tier-4 radiance domain has to remain expressible to be compared."""
    mode, why = resolve_radiometry("none", None)
    assert mode == "none"
    assert "radiance" in why


@pytest.mark.parametrize("mode", RADIOMETRY_MODES)
def test_every_declared_mode_resolves(mode: str) -> None:
    resolved, _ = resolve_radiometry(mode, "a/white.hdr")
    assert resolved in ("white", "snv", "none")


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="radiometry must be one of"):
        resolve_radiometry("zscore", None)


# ══════════════════════════════════════════════════════════════════════
#  apply_radiometry — dispatch
# ══════════════════════════════════════════════════════════════════════


def test_apply_radiometry_normalises_under_snv_and_not_otherwise(reflectance, gains) -> None:
    patch = gains[..., None] * reflectance[None, None, :]
    mask = np.ones((H, W), dtype=bool)

    normalised, _ = apply_radiometry(patch, mask, "snv")
    passthrough, _ = apply_radiometry(patch, mask, "none")

    assert not np.allclose(normalised, patch)
    np.testing.assert_array_equal(passthrough, patch.astype(np.float32))


def test_the_gain_is_persisted_under_every_mode(reflectance, gains) -> None:
    """§3.1 P-2: brightness stays available as an explicit input, not a default.

    "Store *both* the SNV cube and the per-pixel gain … so the network can
    still use brightness if it is genuinely varietal — but must do so
    explicitly rather than by default."
    """
    patch = gains[..., None] * reflectance[None, None, :]
    mask = np.ones((H, W), dtype=bool)
    for mode in ("snv", "white", "none"):
        _, gain = apply_radiometry(patch, mask, mode)
        assert gain.shape == (2, H, W)
        np.testing.assert_allclose(gain[0], gains * reflectance.mean(), rtol=1e-4)
