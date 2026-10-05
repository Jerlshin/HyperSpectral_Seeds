"""S39 seed-averaged contrast and seed SD."""

from __future__ import annotations

import numpy as np
import pandas as pd

from run_final_confirmation import seeded_contrast


def test_seed_average_and_seed_sd() -> None:
    rows = []
    for seed, gain in ((0, .10), (1, .20), (2, .30)):
        for fold in (0, 1):
            for label in range(4):
                for arm, f1 in (("a", .5 + gain), ("b", .5)):
                    rows.append({"seed": seed, "fold": fold, "arm": arm, "label": label, "cross": label == 0,
                                 "destination": 8, "f1": f1, "recall": f1})
    out = seeded_contrast(pd.DataFrame(rows), "a", "b", [0, 1, 2])
    assert np.isclose(out["delta_f1"], .20)
    assert np.allclose(out["seed_deltas"], [.10, .20, .30])
    assert np.isclose(out["seed_sd"], .10)
    assert np.isclose(out["delta_cross"], .20)
