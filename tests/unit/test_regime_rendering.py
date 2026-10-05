"""S35 class-conditional acquisition rendering: rule and class-regime assignment."""

from __future__ import annotations

import numpy as np

from run_regime_rendering import ccar, class_regimes
from spectralquadnet.experiments.screen_metrics import Cohort


def test_ccar_renders_only_sharp_rows_for_soft_classes() -> None:
    plain = np.zeros((3, 4))
    rendered = np.ones((3, 4))
    row_soft = np.array([False, True, False])
    class_soft = np.array([True, False, True, False])
    mixed, share = ccar(plain, rendered, row_soft, class_soft)
    expected = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]], dtype=float)
    assert np.array_equal(mixed, expected)
    assert share == 4 / 12


def test_class_regime_is_the_majority_over_outer_training_rows_ties_sharp() -> None:
    y = np.repeat(np.arange(90), 4)
    groups = np.arange(len(y)) // 2  # two rows per scan
    scan_soft = np.zeros(groups.max() + 1, dtype=bool)
    scan_soft[0] = True                  # class 0: rows 0,1 soft, rows 2,3 sharp -> tie -> sharp
    scan_soft[2] = scan_soft[3] = True   # class 1: all soft
    splits = {0: {"train": np.arange(len(y)), "calib": np.array([], dtype=int),
                  "val": np.array([], dtype=int), "test": np.array([], dtype=int)}}
    cohort = Cohort(y, np.zeros(len(y), dtype=int), np.zeros(90, dtype=bool), splits)
    soft = class_regimes(cohort, 0, scan_soft, groups)
    assert not soft[0] and soft[1] and not soft[2:].any()
