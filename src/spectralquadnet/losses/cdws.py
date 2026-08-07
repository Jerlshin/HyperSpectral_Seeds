"""Class-Difficulty-Weighted Sampling weights.

A pure function of primitives — no config, no module state, no RNG. The
returned weights are consumed by
:class:`~spectralquadnet.data.samplers.ClassBalancedBatchSampler` (Stage 2/3
batch composition) and are persisted into every checkpoint's meta sidecar
under ``cdws_weights``.
"""

from __future__ import annotations

import numpy as np


def build_cdws_weights(
    class_f1: dict[int, float],
    num_classes: int,
    max_w: float = 3.0,
    eps: float = 0.05,
) -> dict[int, float]:
    """Turn per-class validation F1 into inverse-difficulty sampling weights.

    Weight is ``1 / (f1 + eps)``, clamped to ``max_w`` and renormalised so the
    mean weight is exactly 1.0 — harder (lower-F1) classes get sampled more
    often without any single class dominating a batch.

    Args:
        class_f1: Per-class validation F1, keyed by class index. A class
            missing from the dict is treated as F1=0.0 (maximally hard).
        num_classes: Total number of classes; every index in ``range(num_classes)``
            gets a weight regardless of whether it appears in ``class_f1``.
        max_w: Upper clamp on the raw (pre-normalisation) weight.
        eps: Denominator floor that keeps a zero-F1 class's weight finite.

    Returns:
        Weight per class index, mean-normalised to 1.0.
    """
    raw = {c: min(1.0 / (class_f1.get(c, 0.0) + eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w / mean for c, w in raw.items()}
