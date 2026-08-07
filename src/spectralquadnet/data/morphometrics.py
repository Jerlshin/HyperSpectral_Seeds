"""Loading and train-only standardisation of the persisted morphometrics — P-4 / T4-4.

``data/prep/segmentation.py::morphometrics`` writes ``morphology.npy`` in
physical pixel units, unstandardised, because standardisation is a property of
the *split* and no split exists at extraction time.

This module is the consumer side, and the whole of it exists to enforce one
rule: **the mean and scale are fitted on the training indices alone.** Fitting
them over all 8,624 rows would let the val and test grain sizes set the origin
of the feature the network then classifies with — a small leak, but the same
kind as C-1's and pointless to accept when the fix is an index argument.

The features are heavy-tailed in the two ratio columns and are all strictly
positive, so :func:`standardise_morphometrics` also reports the fitted
statistics; a caller that persists them can standardise a later batch without
re-reading the training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spectralquadnet.data.prep.segmentation import MORPHOMETRIC_NAMES

#: Floor on the fitted standard deviation. A morphometric column can be
#: constant on a small training subset (``solidity`` is ~1.0 for every seed
#: that passed segmentation's ``> 0.85`` filter), and dividing by its zero
#: spread would produce NaNs the network cannot recover from.
STD_FLOOR: float = 1e-6


@dataclass(frozen=True)
class MorphometricStats:
    """Per-column mean and standard deviation, fitted on the training split only."""

    mean: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]
    n_fit: int

    @property
    def names(self) -> tuple[str, ...]:
        """Column order — :data:`MORPHOMETRIC_NAMES`."""
        return MORPHOMETRIC_NAMES

    def apply(self, morph: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Standardise ``(N, 8)`` rows with these statistics."""
        return ((np.asarray(morph, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


def load_morphometrics(path: str | Path) -> npt.NDArray[np.float32]:
    """Read ``morphology.npy`` and check its shape against :data:`MORPHOMETRIC_NAMES`.

    Raises:
        FileNotFoundError: The file does not exist — it is written by
            ``scripts/prepare_dataset.py``, which has to be re-run for T4-4.
        ValueError: The array is not ``(N, 8)``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Morphometrics (P-4 / T4-4) are written by "
            "`python scripts/prepare_dataset.py`, which must be re-run to produce them."
        )
    morph = np.load(p)
    if morph.ndim != 2 or morph.shape[1] != len(MORPHOMETRIC_NAMES):
        raise ValueError(
            f"{p} has shape {morph.shape}; expected (N, {len(MORPHOMETRIC_NAMES)}) — "
            f"{', '.join(MORPHOMETRIC_NAMES)}"
        )
    loaded: npt.NDArray[np.float32] = morph.astype(np.float32)
    return loaded


def fit_morphometric_stats(
    morph: npt.NDArray[Any], train_idx: npt.NDArray[Any]
) -> MorphometricStats:
    """Fit per-column mean/std on ``train_idx`` rows **only**.

    Args:
        morph: ``(N, 8)`` raw morphometrics.
        train_idx: Row indices of the training split. Must be non-empty —
            silently falling back to all rows is the leak this function exists
            to prevent.

    Raises:
        ValueError: ``train_idx`` is empty.
    """
    idx = np.asarray(train_idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("train_idx is empty — refusing to fit morphometric stats on no data.")
    fit = np.asarray(morph, dtype=np.float32)[idx]
    return MorphometricStats(
        mean=fit.mean(axis=0).astype(np.float32),
        std=np.maximum(fit.std(axis=0), STD_FLOOR).astype(np.float32),
        n_fit=int(idx.size),
    )


def standardise_morphometrics(
    morph: npt.NDArray[Any], train_idx: npt.NDArray[Any]
) -> tuple[npt.NDArray[np.float32], MorphometricStats]:
    """Standardise every row using statistics fitted on ``train_idx`` alone.

    Returns:
        ``(standardised, stats)`` — the full ``(N, 8)`` array, and the
        statistics used, so they can be recorded in the run's sidecar.
    """
    stats = fit_morphometric_stats(morph, train_idx)
    return stats.apply(morph), stats
