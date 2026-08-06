"""Class-Difficulty-Weighted Sampling weights.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`build_cdws_weights`             453-461
=====================================  ==============

A pure function of primitives — no config, no globals, no RNG — so it relocates
with zero translation. ``tests/unit/test_cdws.py`` asserts bit-exact equality
against the baseline implementation across the full weight/eps grid
(REFACTOR_PLAN.md §3.2.3).

The returned weights are consumed by
:class:`~spectralquadnet.data.samplers.ClassBalancedBatchSampler` (Stage 2/3
batch composition) and are persisted into every checkpoint's meta sidecar under
``cdws_weights``.
"""

from __future__ import annotations

import numpy as np


def build_cdws_weights(
    class_f1: dict[int, float],
    num_classes: int,
    max_w: float = 3.0,
    eps: float = 0.05,
) -> dict[int, float]:
    raw = {c: min(1.0 / (class_f1.get(c, 0.0) + eps), max_w) for c in range(num_classes)}
    mean = float(np.mean(list(raw.values())))
    return {c: w / mean for c, w in raw.items()}
