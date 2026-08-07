"""T4-3 / P-3 — the resize order, and the mask it persists.

M-11: ``INTER_AREA`` on an **already-masked** patch averages seed pixels with
background zeros, so a boundary output pixel covering a fraction
:math:`\\alpha_p` of seed came out at :math:`\\alpha_p` times its true
radiance, and a pixel covering 0.001 % of a seed came out as a *non-zero
background*. The resize was introduced to serve the zero-background invariant
and is what breaks it.

T4-3's criterion is "foreground pixel count matches the resized mask exactly;
no partial-coverage pixels remain", and both halves are asserted here against
a synthetic cube whose true value is known everywhere — a constant seed
spectrum, so any deviation in the interior *or* on the boundary is
attenuation and nothing else.

Every test has a companion running the **old** order (``_old_order``, a
transcription of the pre-Tier-4 three lines) and showing it fail, so none of
them can pass by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectralquadnet.data.prep.patch_extraction import (
    FILL_THRESHOLD,
    extract_patch,
    pad_to_square,
    resize_patch,
)

BANDS = 6
SIZE = 16


class _Region:
    """The two attributes ``extract_patch`` reads off a ``RegionProperties``."""

    def __init__(self, bbox, label: int = 1) -> None:
        self.bbox = bbox
        self.label = label


@pytest.fixture
def disc():
    """A round "seed" of constant spectrum on a larger background.

    Round on purpose: a rectangle aligned to the resize grid has no
    partial-coverage pixels at all, which is the case M-11 does not arise in.
    """
    h = w = 40
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy - 19.5) ** 2 + (xx - 19.5) ** 2) < 14.0**2

    spectrum = np.arange(1.0, BANDS + 1.0, dtype=np.float32)
    cube = np.zeros((h, w, BANDS), dtype=np.float32)
    cube[mask] = spectrum

    labeled = mask.astype(np.int32)
    return cube, labeled, _Region((0, 0, h, w)), spectrum


def _old_order(cube, labeled, region, size):
    """The pre-Tier-4 three lines: mask, pad, resize. Verbatim from the baseline."""
    r0, c0, r1, c1 = region.bbox
    p = cube[r0:r1, c0:c1, :].copy()
    mask = labeled[r0:r1, c0:c1] == region.label
    p *= mask[..., None]
    p = pad_to_square(p)
    p = resize_patch(p, size)
    return np.transpose(p, (2, 0, 1))


# ══════════════════════════════════════════════════════════════════════
#  The two halves of T4-3's criterion
# ══════════════════════════════════════════════════════════════════════


def test_the_foreground_count_matches_the_resized_mask_exactly(disc) -> None:
    """T4-3's criterion, first half: the mask *is* the foreground, exactly."""
    cube, labeled, region, _ = disc
    patch, alpha, _ = extract_patch(cube, labeled, region, SIZE, radiometry="none")

    nonzero = np.abs(patch).sum(axis=0) > 0
    assert np.array_equal(nonzero, alpha > 0)
    assert int(nonzero.sum()) == int((alpha > 0).sum())
    assert 0 < nonzero.sum() < SIZE * SIZE  # the test would be vacuous otherwise


def test_no_partial_coverage_pixel_survives(disc) -> None:
    """T4-3's criterion, second half — measured against a known true value.

    The seed's spectrum is constant, so *every* foreground pixel must carry it
    exactly. Under the old order the boundary pixels carry ``alpha`` times it.
    """
    cube, labeled, region, spectrum = disc
    patch, alpha, _ = extract_patch(cube, labeled, region, SIZE, radiometry="none")

    keep = alpha > 0
    recovered = patch[:, keep]
    for band in range(BANDS):
        np.testing.assert_allclose(recovered[band], spectrum[band], rtol=1e-5)


def test_the_old_order_attenuated_the_boundary(disc) -> None:
    """The companion: M-11, measured.

    The old order leaves boundary pixels scaled by their fill fraction, so the
    same constant-spectrum seed comes out with a rim of darker pixels — a
    gradient that is an artifact of the resize and not of the seed.
    """
    cube, labeled, region, spectrum = disc
    old = _old_order(cube, labeled, region, SIZE)
    _, alpha, _ = extract_patch(cube, labeled, region, SIZE, radiometry="none")

    partial = (alpha > FILL_THRESHOLD) & (alpha < 0.999)
    assert partial.sum() > 0, "no partial-coverage pixels — the fixture is too coarse"

    attenuated = old[0][partial]
    assert np.all(attenuated < spectrum[0] * 0.999)
    # …and the new order recovers them.
    new, _, _ = extract_patch(cube, labeled, region, SIZE, radiometry="none")
    np.testing.assert_allclose(new[0][partial], spectrum[0], rtol=1e-5)


def test_the_old_order_left_a_non_zero_background(disc) -> None:
    """The companion to the invariant: sub-threshold coverage is not zero.

    This is what makes the downstream ``> 1e-5`` foreground test fragile
    (§2.6): "background" was a small number, not zero, so any transform that
    rescales the patch can push it over the threshold.
    """
    cube, labeled, region, _ = disc
    old = _old_order(cube, labeled, region, SIZE)
    _, alpha, _ = extract_patch(cube, labeled, region, SIZE, radiometry="none")

    background = alpha == 0
    old_background = np.abs(old).sum(axis=0)[background]
    assert old_background.max() > 0.0, "fixture has no sub-threshold pixels"


def test_the_background_is_exactly_zero(disc) -> None:
    """Not "small": exactly ``0.0``, in every band, for every masked-out pixel."""
    cube, labeled, region, _ = disc
    patch, alpha, gain = extract_patch(cube, labeled, region, SIZE, radiometry="snv")

    background = alpha == 0
    assert np.all(patch[:, background] == 0.0)
    assert np.all(gain[:, background] == 0.0)


# ══════════════════════════════════════════════════════════════════════
#  Shapes, dtypes and the alpha map's contract
# ══════════════════════════════════════════════════════════════════════


def test_the_outputs_have_the_persisted_shapes(disc) -> None:
    cube, labeled, region, _ = disc
    patch, alpha, gain = extract_patch(cube, labeled, region, SIZE)
    assert patch.shape == (BANDS, SIZE, SIZE)
    assert alpha.shape == (SIZE, SIZE)
    assert gain.shape == (2, SIZE, SIZE)
    assert patch.dtype == alpha.dtype == gain.dtype == np.float32


def test_alpha_is_a_fill_fraction_in_the_kept_range(disc) -> None:
    """Every non-zero alpha is a coverage fraction above the threshold."""
    cube, labeled, region, _ = disc
    _, alpha, _ = extract_patch(cube, labeled, region, SIZE, radiometry="none")
    kept = alpha[alpha > 0]
    assert kept.min() > FILL_THRESHOLD
    assert kept.max() <= 1.0 + 1e-6


def test_a_grid_aligned_seed_needs_no_correction() -> None:
    """A seed whose mask tiles the resize grid exactly is unchanged by T4-3.

    The control: P-3 is a *boundary* correction, so on a patch with no partial
    coverage the old and new orders must agree bit for bit. If they did not,
    the fix would be changing something it has no business changing.
    """
    size = 4
    h = w = 8
    mask = np.zeros((h, w), dtype=bool)
    mask[2:6, 2:6] = True  # 4x4 of 8x8 → each output pixel is fully in or out
    cube = np.zeros((h, w, BANDS), dtype=np.float32)
    cube[mask] = np.arange(1.0, BANDS + 1.0, dtype=np.float32)
    region = _Region((0, 0, h, w))

    new, alpha, _ = extract_patch(cube, np.ones((h, w), np.int32) * mask, region, size, "none")
    old = _old_order(cube, np.ones((h, w), np.int32) * mask, region, size)

    assert set(np.unique(alpha).tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal(new, old)


def test_the_seed_survives_a_bounding_box_crop(disc) -> None:
    """``extract_patch`` crops to ``region.bbox``, as the writer's loop does."""
    cube, labeled, _, spectrum = disc
    tight = _Region((6, 6, 34, 34))
    patch, alpha, _ = extract_patch(cube, labeled, tight, SIZE, radiometry="none")
    assert alpha.sum() > 0
    np.testing.assert_allclose(patch[0][alpha > 0], spectrum[0], rtol=1e-5)


def test_two_regions_in_one_cube_do_not_bleed() -> None:
    """The mask is ``labeled == region.label``, so a neighbouring seed is excluded."""
    h = w = 12
    labeled = np.zeros((h, w), dtype=np.int32)
    labeled[2:6, 2:6] = 1
    labeled[2:6, 7:11] = 2
    cube = np.zeros((h, w, BANDS), dtype=np.float32)
    cube[labeled == 1] = 1.0
    cube[labeled == 2] = 9.0

    patch, alpha, _ = extract_patch(cube, labeled, _Region((0, 0, h, w), label=1), 8, "none")
    assert patch.max() <= 1.0 + 1e-6
    assert alpha.sum() > 0


# ══════════════════════════════════════════════════════════════════════
#  P-2 composes with P-3
# ══════════════════════════════════════════════════════════════════════


def test_snv_is_applied_after_masking(disc) -> None:
    """§3.1 P-2: "applied along lambda *after* masking".

    Order matters: SNV before masking would give every background pixel a
    unit-variance spectrum (see ``test_radiometry.py``), so the two protocol
    items would fight rather than compose.
    """
    cube, labeled, region, _ = disc
    patch, alpha, gain = extract_patch(cube, labeled, region, SIZE, radiometry="snv")

    assert np.all(patch[:, alpha == 0] == 0.0)
    foreground = patch[:, alpha > 0]
    # A constant spectrum is the same after SNV at every foreground pixel.
    assert np.allclose(foreground, foreground[:, :1], atol=1e-5)
    # And the brightness it removed is retained, per pixel.
    assert np.all(gain[0][alpha > 0] > 0.0)
