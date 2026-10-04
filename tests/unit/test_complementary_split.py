"""Regression for RNG drift with missing kernels in two-acquisition CV."""
import numpy as np
import pytest

from spectralquadnet.experiments.complementary_split import complementary_split


def fixture_arrays():
    sizes = [48, 45, 47, 48, 48, 48, 41, 47, 46, 48]
    groups = np.repeat(np.arange(10), sizes)
    return groups // 2, groups


def test_complementary_folds_cover_unequal_bundles_once():
    y, g = fixture_arrays()
    seen = []
    for fold in (0, 1):
        split = complementary_split(y, g, fold=fold)
        held = np.r_[split["val"], split["test"]]
        assert not set(g[split["train"]]) & set(g[held])
        assert not set(split["train"]) & set(split["calib"])
        assert set(y[held]) == set(y)
        seen.extend(held)
    np.testing.assert_array_equal(np.sort(seen), np.arange(len(y)))


def test_group_choices_ignore_patch_counts_calibration_and_other_classes():
    y, g = fixture_arrays()
    for fold in (0, 1):
        a = complementary_split(y, g, fold=fold)
        keep = np.arange(len(y))[7:]
        b = complementary_split(y[keep], g[keep], fold=fold, calib_frac=0.3)
        np.testing.assert_array_equal(
            np.unique(g[np.r_[a["val"], a["test"]]]),
            np.unique(g[keep][np.r_[b["val"], b["test"]]]),
        )
        c = complementary_split(y[y > 1], g[y > 1], fold=fold)
        np.testing.assert_array_equal(
            np.unique(g[np.r_[a["val"], a["test"]]][y[np.r_[a["val"], a["test"]]] > 1]),
            np.unique(g[y > 1][np.r_[c["val"], c["test"]]]),
        )


def test_split_rejects_invalid_support_and_settings():
    y, g = fixture_arrays()
    for fold in [-1, 2]:
        with pytest.raises(ValueError, match="fold"):
            complementary_split(y, g, fold=fold)
    with pytest.raises(ValueError, match="exactly two"):
        complementary_split(y[g != 0], g[g != 0], fold=0)
    bad = y.copy()
    bad[0] = 4
    with pytest.raises(ValueError, match="multiple classes"):
        complementary_split(bad, g, fold=0)
