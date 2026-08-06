"""Class-Difficulty-Weighted Sampling bit-exactness — REFACTOR_PLAN.md §3.2.3.

``build_cdws_weights`` turns per-class validation F1 into the sampling weights
``ClassBalancedBatchSampler`` uses for Stage 2 and Stage 3 batch composition, and
every checkpoint's meta sidecar carries a frozen copy under ``cdws_weights``. A
drift here would silently re-weight training without changing a single shape or
raising a single error, so it is compared with ``==`` against the pre-refactor
implementation across the F1 distributions a real run produces — including the
pathological ones (all-zero F1, missing classes) that pin the clamp and the
``.get(c, 0.0)`` fallback.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from spectralquadnet.losses.cdws import build_cdws_weights

pytestmark = pytest.mark.regression

NUM_CLASSES = 90


def _f1_profiles(num_classes: int = NUM_CLASSES) -> dict[str, dict[int, float]]:
    """Per-class F1 dicts spanning the shapes Stage 1/2 hand to CDWS."""
    rng = np.random.default_rng(0)
    return {
        "realistic": {c: float(v) for c, v in enumerate(rng.uniform(0.3, 1.0, num_classes))},
        "all_zero": dict.fromkeys(range(num_classes), 0.0),
        "all_perfect": dict.fromkeys(range(num_classes), 1.0),
        "bimodal": {c: (0.05 if c % 7 == 0 else 0.95) for c in range(num_classes)},
        # Stage 1 can hand over a partial dict; `.get(c, 0.0)` must fill the gaps.
        "sparse": {c: 0.6 for c in range(0, num_classes, 3)},
        "empty": {},
    }


@pytest.mark.parametrize("profile", sorted(_f1_profiles()))
@pytest.mark.parametrize(("max_w", "eps"), [(3.0, 0.05), (1.0, 0.5), (7.0, 1e-6), (20.0, 0.05)])
def test_cdws_weights_match_baseline(baseline, profile, max_w, eps):
    class_f1 = _f1_profiles()[profile]
    expected = baseline.build_cdws_weights(class_f1, NUM_CLASSES, max_w, eps)
    actual = build_cdws_weights(class_f1, NUM_CLASSES, max_w, eps)

    assert actual.keys() == expected.keys()
    for c in expected:
        assert actual[c] == expected[c], f"class {c}: {actual[c]!r} != {expected[c]!r}"


def test_cdws_defaults_match_baseline(baseline):
    """The default ``max_w``/``eps`` in the signature are part of the contract."""
    class_f1 = _f1_profiles()["realistic"]
    assert build_cdws_weights(class_f1, NUM_CLASSES) == baseline.build_cdws_weights(
        class_f1, NUM_CLASSES
    )


def test_cdws_shipped_config_values_match_baseline(baseline, cfg):
    """The exact call ``compute_class_difficulty`` makes, with the composed config."""
    class_f1 = _f1_profiles()["bimodal"]
    kwargs = (cfg.data.num_classes, cfg.stage2.cdws_max_weight, cfg.stage2.cdws_eps)
    assert build_cdws_weights(class_f1, *kwargs) == baseline.build_cdws_weights(class_f1, *kwargs)


def test_cdws_invariants(cfg):
    """Properties the samplers rely on, independent of the baseline comparison."""
    weights = build_cdws_weights(
        _f1_profiles()["realistic"],
        cfg.data.num_classes,
        cfg.stage2.cdws_max_weight,
        cfg.stage2.cdws_eps,
    )

    assert sorted(weights) == list(range(cfg.data.num_classes))
    assert all(w > 0 for w in weights.values())
    # Normalised by the mean of the raw weights, so the mean weight is exactly 1.
    assert float(np.mean(list(weights.values()))) == pytest.approx(1.0)
    # Harder classes (lower F1) must never be sampled less than easier ones.
    f1 = _f1_profiles()["realistic"]
    ranked = sorted(range(cfg.data.num_classes), key=lambda c: f1[c])
    assert all(weights[a] >= weights[b] for a, b in pairwise(ranked))
