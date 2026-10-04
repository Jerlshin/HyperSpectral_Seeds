"""Recompute coverage from immutable row lists; never retrain or resplit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]


def audit(study: str) -> dict:
    evidence = ROOT / "docs/research/evidence" / study
    plan = json.loads((evidence / "preregistration.json").read_text())
    y = np.load(ROOT / "dataset_u430k32/labels.npy")
    groups = np.load(ROOT / "dataset_u430k32/groups.npy")
    scans = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv")
    cross = set(scans.groupby("label").session_id.nunique().loc[lambda x: x > 1].index)
    held = [np.sort(np.r_[s["val"], s["test"]]) for s in plan["splits"].values()]
    counts = np.bincount(np.concatenate(held), minlength=len(y))
    repeated = []
    for label in np.unique(y):
        pair = [np.unique(groups[x[y[x] == label]]) for x in held]
        assert all(len(g) == 1 for g in pair)
        if np.array_equal(pair[0], pair[1]):
            repeated.append(int(label))
    for split in plan["splits"].values():
        train = np.r_[split["train"], split["calib"]]
        test = np.r_[split["val"], split["test"]]
        assert not set(groups[train]) & set(groups[test])
    result = {
        "fold_sizes": [len(x) for x in held],
        "unique_held_out_rows": int((counts > 0).sum()),
        "repeated_held_out_rows": int((counts > 1).sum()),
        "never_held_out_rows": int((counts == 0).sum()),
        "repeated_group_classes": repeated,
        "cross_repeated": [label for label in repeated if label in cross],
        "per_fold_group_disjoint": True,
    }
    saved = json.loads((evidence / "fold_coverage_audit.json").read_text())
    for key in result.keys() & saved.keys():
        assert result[key] == saved[key], (study, key)
    if study.startswith("S21"):
        assert (counts == 1).all() and not repeated
    (evidence / "fold_coverage_recomputed.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    for study in ["S20_rgb_pathway", "S21_complementary_rgb"]:
        print(study, audit(study))
