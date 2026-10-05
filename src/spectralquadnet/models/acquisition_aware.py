"""Inference composition of the proposed acquisition-aware RGB–HSI system (S36).

The two branches stay independent up to calibrated class evidence (D56: learned or kernel-level
fusion is not identifiable on this design, F121–F123). This module combines them:

1. **Per-modality calibration:** a temperature per branch, fitted on calib only.
2. **Class-conditional acquisition rendering (CCAR, S35):** for a test kernel imaged in the sharp
   regime, the RGB evidence for a variety trained in the soft regime is taken from the kernel
   rendered soft (blur is one-way). Regimes are measured label-free (plate background + sharpness).
3. **Fixed equal probability fusion** (S20/S21 calib chose equal; S34 equal beats every learned form).
4. **Optional lot-level pooling:** mean log-probability over kernels known to share a lot. This is a
   separate inference regime with its own reporting (F122); it is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import softmax

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class AcquisitionAwareFusion:
    t_hsi: float
    t_rgb: float
    class_soft: Array  # bool per class: majority of its training rows were imaged in the soft regime
    rgb_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.t_hsi <= 0 or self.t_rgb <= 0 or not 0 <= self.rgb_weight <= 1:
            raise ValueError("Temperatures must be positive and the RGB weight in [0, 1]")

    def rgb_logits(self, plain: Array, rendered: Array | None, row_soft: Array | None) -> Array:
        """RGB logits with CCAR when the rendered pass and test regimes are given."""
        if rendered is None or row_soft is None:
            return plain
        if plain.shape != rendered.shape or len(row_soft) != len(plain) or plain.shape[1] != len(self.class_soft):
            raise ValueError("Misaligned RGB evidence")
        use = (~np.asarray(row_soft, dtype=bool))[:, None] & np.asarray(self.class_soft, dtype=bool)[None, :]
        return np.where(use, rendered, plain)

    def __call__(self, hsi_logits: Array, rgb_plain: Array, rgb_rendered: Array | None = None,
                 row_soft: Array | None = None) -> Array:
        p_hsi = softmax(np.asarray(hsi_logits, dtype=np.float64) / self.t_hsi, axis=1)
        p_rgb = softmax(self.rgb_logits(rgb_plain, rgb_rendered, row_soft).astype(np.float64) / self.t_rgb, axis=1)
        return np.asarray((1 - self.rgb_weight) * p_hsi + self.rgb_weight * p_rgb)


def pool_lots(prob: Array, lots: Array) -> Array:
    """Lot-level regime: every kernel receives its lot's mean log-probability (renormalized)."""
    logp = np.log(np.clip(prob, 1e-12, None))
    out = np.empty_like(logp)
    for lot in np.unique(lots):
        idx = lots == lot
        out[idx] = logp[idx].mean(0)
    return np.asarray(softmax(out, axis=1))
