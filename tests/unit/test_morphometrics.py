"""T4-4 / P-4 — the eight morphometrics, and where they are standardised.

M-13: ``segment`` computes area, eccentricity and solidity, *gates the region
filter on all three*, and then discards them — after which ``resize_patch``
destroys the absolute scale they measured. Grain length and width are defining
cultivar descriptors, so the pipeline was throwing away a feature it had
already paid for.

T4-4's criterion is "``morphology.npy`` exists; standardised on train only".
The second clause is the one with teeth and it is a property of the *consumer*,
not the writer: standardisation cannot happen at extraction time because no
split exists yet, so the array is persisted raw and
:func:`~spectralquadnet.data.morphometrics.standardise_morphometrics` fits the
statistics on the training indices alone.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skimage.measure import label, regionprops

from spectralquadnet.data.morphometrics import (
    MorphometricStats,
    fit_morphometric_stats,
    load_morphometrics,
    standardise_morphometrics,
)
from spectralquadnet.data.prep.segmentation import MORPHOMETRIC_NAMES, morphometrics


@pytest.fixture
def ellipse_region():
    """A filled ellipse: known major/minor axes, solid, non-circular."""
    yy, xx = np.mgrid[0:60, 0:60]
    mask = ((yy - 30) / 18.0) ** 2 + ((xx - 30) / 9.0) ** 2 < 1.0
    return regionprops(label(mask))[0]


# ══════════════════════════════════════════════════════════════════════
#  The writer
# ══════════════════════════════════════════════════════════════════════


def test_the_eight_descriptors_are_in_the_documented_order(ellipse_region) -> None:
    """§3.1 P-4 fixes the column order; the persisted array is unreadable without it."""
    assert MORPHOMETRIC_NAMES == (
        "area",
        "major_axis",
        "minor_axis",
        "axis_ratio",
        "eccentricity",
        "solidity",
        "equivalent_diameter",
        "perimeter_over_sqrt_area",
    )
    values = morphometrics(ellipse_region)
    assert values.shape == (8,)
    assert values.dtype == np.float32


def test_the_descriptors_are_the_region_s_own(ellipse_region) -> None:
    """Each column is the ``regionprops`` value, not a re-derivation of it."""
    v = morphometrics(ellipse_region)
    r = ellipse_region

    assert v[0] == pytest.approx(r.area, rel=1e-5)
    assert v[1] == pytest.approx(r.axis_major_length, rel=1e-5)
    assert v[2] == pytest.approx(r.axis_minor_length, rel=1e-5)
    assert v[3] == pytest.approx(r.axis_major_length / r.axis_minor_length, rel=1e-5)
    assert v[4] == pytest.approx(r.eccentricity, rel=1e-5)
    assert v[5] == pytest.approx(r.solidity, rel=1e-5)
    assert v[6] == pytest.approx(r.equivalent_diameter_area, rel=1e-5)
    assert v[7] == pytest.approx(r.perimeter / np.sqrt(r.area), rel=1e-5)


def test_the_shape_features_discriminate_shape(ellipse_region) -> None:
    """A long grain and a round one differ in the columns that describe elongation."""
    yy, xx = np.mgrid[0:60, 0:60]
    circle = regionprops(label((yy - 30) ** 2 + (xx - 30) ** 2 < 13**2))[0]

    long_grain = morphometrics(ellipse_region)
    round_grain = morphometrics(circle)

    assert long_grain[3] > 1.8 > round_grain[3]  # axis ratio
    assert long_grain[4] > round_grain[4]  # eccentricity


def test_they_are_in_physical_pixel_units_the_resize_destroys(ellipse_region) -> None:
    """P-4's point: these carry the absolute scale the 64x64 resize removes.

    Two seeds differing only in size are identical after ``resize_patch`` and
    distinguishable here — which is why the morphometrics have to be persisted
    separately rather than recovered from the patch.
    """
    yy, xx = np.mgrid[0:60, 0:60]
    small = regionprops(label(((yy - 30) / 9.0) ** 2 + ((xx - 30) / 4.5) ** 2 < 1.0))[0]

    big_values, small_values = morphometrics(ellipse_region), morphometrics(small)
    assert big_values[0] > small_values[0] * 3  # area
    assert big_values[1] > small_values[1] * 1.8  # major axis
    # …while the *shape* ratios agree, since one is a scaled copy of the other.
    # 10 % because the small ellipse is 9 x 4.5 px and its axes are pixelated.
    assert big_values[3] == pytest.approx(small_values[3], rel=0.10)


def test_a_degenerate_region_does_not_divide_by_zero() -> None:
    """A one-pixel region has zero minor axis; the ratio stays finite."""
    single = regionprops(label(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])))[0]
    assert np.all(np.isfinite(morphometrics(single)))


# ══════════════════════════════════════════════════════════════════════
#  The consumer — "standardised on train only"
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def morph_and_split():
    """Morphometrics whose val/test rows are deliberately off-distribution."""
    rng = np.random.default_rng(3)
    morph = rng.normal(50.0, 5.0, size=(200, 8)).astype(np.float32)
    morph[150:] += 40.0  # the held-out half is much larger
    train = np.arange(150)
    held_out = np.arange(150, 200)
    return morph, train, held_out


def test_the_statistics_are_fitted_on_train_alone(morph_and_split) -> None:
    """T4-4's criterion. The train rows standardise to mean 0, sd 1 — nothing else does."""
    morph, train, held_out = morph_and_split
    out, stats = standardise_morphometrics(morph, train)

    np.testing.assert_allclose(out[train].mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(out[train].std(axis=0), 1.0, atol=1e-4)
    assert stats.n_fit == len(train)

    # The held-out rows are transformed by the *train* statistics, so their
    # own mean is free to be anything — and here it is far from zero, because
    # they were drawn from a different distribution.
    assert abs(out[held_out].mean()) > 5.0


def test_fitting_on_everything_would_move_the_origin(morph_and_split) -> None:
    """The companion: what the leak would have been worth.

    Standardising over all rows lets the held-out grain sizes set the origin of
    a feature the network then classifies with. Small, and the same *kind* of
    leak as C-1 — and it costs one index argument to avoid.
    """
    morph, train, _ = morph_and_split
    train_only = fit_morphometric_stats(morph, train)
    everything = fit_morphometric_stats(morph, np.arange(len(morph)))
    assert np.abs(train_only.mean - everything.mean).max() > 1.0


def test_the_fitted_statistics_can_be_reapplied(morph_and_split) -> None:
    """``MorphometricStats`` is the object a run records so a later batch matches."""
    morph, train, held_out = morph_and_split
    out, stats = standardise_morphometrics(morph, train)
    np.testing.assert_allclose(stats.apply(morph[held_out]), out[held_out], rtol=1e-5)
    assert stats.names == MORPHOMETRIC_NAMES


def test_a_constant_column_does_not_produce_nans() -> None:
    """``solidity`` is ~1.0 for every seed segmentation admits; its spread can be 0."""
    morph = np.ones((10, 8), dtype=np.float32)
    out, stats = standardise_morphometrics(morph, np.arange(10))
    assert np.all(np.isfinite(out))
    assert np.all(stats.std > 0)


def test_an_empty_training_split_is_refused(morph_and_split) -> None:
    """Silently falling back to all rows is the leak this function exists to stop."""
    morph, _, _ = morph_and_split
    with pytest.raises(ValueError, match="train_idx is empty"):
        fit_morphometric_stats(morph, np.array([], dtype=np.int64))


# ══════════════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════════════


def test_loading_checks_the_column_count(tmp_path) -> None:
    path = tmp_path / "morphology.npy"
    np.save(path, np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        load_morphometrics(path)

    np.save(path, np.zeros((4, 8), dtype=np.float32))
    assert load_morphometrics(path).shape == (4, 8)


def test_a_missing_file_says_how_to_produce_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_dataset"):
        load_morphometrics(tmp_path / "absent.npy")


def test_the_stats_object_is_frozen() -> None:
    """Fitted statistics are a record of one split; mutating them in place would
    silently re-describe what a persisted array was standardised against."""
    stats = MorphometricStats(mean=np.zeros(8, np.float32), std=np.ones(8, np.float32), n_fit=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.n_fit = 2  # type: ignore[misc]
