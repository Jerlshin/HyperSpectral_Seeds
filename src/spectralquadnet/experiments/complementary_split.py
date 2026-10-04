"""Explicit two-acquisition CV with independent group and patch random streams.

The legacy splitter is preserved for frozen historical replay. This opt-in splitter
requires exactly two class-pure groups per class and guarantees that each physical
row is held out once across folds 0 and 1, including unequal bundle sizes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]


def complementary_split(
    labels: Array, groups: Array, *, fold: int, calib_frac: float = 0.15, seed: int = 42
) -> dict[str, Array]:
    labels, groups = np.asarray(labels), np.asarray(groups)
    if labels.ndim != 1 or labels.shape != groups.shape or not len(labels):
        raise ValueError("Labels/groups must be nonempty aligned vectors")
    if labels.dtype.kind not in "iu" or groups.dtype.kind not in "iu" or labels.min() < 0:
        raise ValueError("Integer identities and nonnegative labels required")
    if fold not in (0, 1) or not 0 < calib_frac < 1 or seed < 0:
        raise ValueError("Require fold 0/1, calibration fraction in (0,1), nonnegative seed")
    for group in np.unique(groups):
        if np.unique(labels[groups == group]).size != 1:
            raise ValueError("A physical group contains multiple classes")
    parts: dict[str, list[Array]] = {key: [] for key in ["train", "calib", "val", "test"]}
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        pair = np.unique(groups[ids])
        if len(pair) != 2:
            raise ValueError(f"Class {label} requires exactly two physical groups")
        ordering = np.random.default_rng(np.random.SeedSequence([seed, int(label), 0]))
        eval_group = pair[ordering.permutation(2)][fold]
        held = ids[groups[ids] == eval_group]
        train = ids[groups[ids] != eval_group]
        if min(len(train), len(held)) < 2:
            raise ValueError("Each bundle needs at least two rows")
        patch_rng = np.random.default_rng(np.random.SeedSequence([seed, int(label), 1, fold]))
        held = patch_rng.permutation(held)
        train = patch_rng.permutation(train)
        ncal = min(max(1, int(round(len(train) * calib_frac))), len(train) - 1)
        parts["train"].append(train[ncal:])
        parts["calib"].append(train[:ncal])
        parts["val"].append(held[: len(held) // 2])
        parts["test"].append(held[len(held) // 2 :])
    result = {key: np.sort(np.concatenate(values)) for key, values in parts.items()}
    all_rows = np.concatenate(list(result.values()))
    if not np.array_equal(np.sort(all_rows), np.arange(len(labels))):
        raise ValueError("Split does not partition every row exactly once")
    if set(groups[result["train"]]) & set(groups[np.r_[result["val"], result["test"]]]):
        raise ValueError("Physical group leakage")
    return result
