"""S36 inference composition: calibration, CCAR switch, equal fusion and lot pooling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from spectralquadnet.models.acquisition_aware import AcquisitionAwareFusion, pool_lots


def test_without_rendering_it_is_equal_calibrated_fusion() -> None:
    rng = np.random.default_rng(0)
    h, r = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    fuse = AcquisitionAwareFusion(t_hsi=2.0, t_rgb=0.5, class_soft=np.zeros(4, dtype=bool))
    expected = (softmax(h / 2.0, axis=1) + softmax(r / 0.5, axis=1)) / 2
    assert np.allclose(fuse(h, r), expected)
    assert np.allclose(fuse(h, r).sum(1), 1)


def test_ccar_swaps_only_sharp_rows_for_soft_classes() -> None:
    fuse = AcquisitionAwareFusion(1.0, 1.0, class_soft=np.array([True, False]))
    plain, rendered = np.zeros((2, 2)), np.ones((2, 2))
    out = fuse.rgb_logits(plain, rendered, np.array([False, True]))
    assert np.array_equal(out, [[1, 0], [0, 0]])


def test_lot_pooling_gives_every_kernel_of_a_lot_the_same_evidence() -> None:
    prob = softmax(np.random.default_rng(1).normal(size=(6, 3)), axis=1)
    pooled = pool_lots(prob, np.array([0, 0, 0, 1, 1, 2]))
    assert np.allclose(pooled[0], pooled[2]) and np.allclose(pooled[3], pooled[4])
    assert np.allclose(pooled[5], prob[5])


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        AcquisitionAwareFusion(0.0, 1.0, class_soft=np.zeros(2, dtype=bool))
