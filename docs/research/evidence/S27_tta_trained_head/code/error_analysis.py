#!/usr/bin/env python
"""Descriptive S27 error accounting vs equal TTA: rescues/harms and cross-error session attraction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
EVID = ROOT / "docs/research/evidence/S27_tta_trained_head"


def main() -> None:
    y = np.load(ROOT / "dataset_u430k32/labels.npy")
    groups = np.load(ROOT / "dataset_u430k32/groups.npy")
    scans = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv").set_index("scan_id")
    session = scans.session_id.loc[groups].to_numpy()
    cross = scans.groupby("label").session_id.nunique().sort_index().to_numpy() > 1
    splits = json.loads((ROOT / "docs/research/evidence/S21_complementary_rgb/preregistration.json").read_text())["splits"]
    pred = pd.read_csv(EVID / "screen_results/predictions.csv.gz")
    out = []
    for fold in (0, 1):
        tr = np.r_[splits[str(fold)]["train"], splits[str(fold)]["calib"]]
        trained_in = {c: set(session[tr][y[tr] == c]) for c in range(90)}
        p = pred[pred.fold == fold].pivot(index="index", columns="arm", values="prediction")
        ids = p.index.to_numpy()
        a, h = p.equal_tta.to_numpy(), p.tta_trained_residual.to_numpy()
        row = {"fold": fold, "rescued": int(np.sum((a != y[ids]) & (h == y[ids]))),
               "harmed": int(np.sum((a == y[ids]) & (h != y[ids]))), "changed": int(np.sum(a != h))}
        for name, v in (("equal_tta", a), ("tta_trained_residual", h)):
            wrong = np.flatnonzero((v != y[ids]) & cross[y[ids]])
            row[f"{name}_cross_errors"] = len(wrong)
            row[f"{name}_cross_attraction"] = float(np.mean([session[ids[i]] in trained_in[v[i]] for i in wrong]))
        out.append(row)
    (EVID / "error_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
