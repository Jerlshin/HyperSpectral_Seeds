#!/usr/bin/env python
"""Descriptive bottleneck diagnostic on ALREADY-SCORED S21/S22 corrected-fold predictions.

No model is fitted or selected. It locates the remaining error of the fixed
equal v5-TTA/RGB fusion: same- vs cross-session macro-recall deficit, the
either-modality oracle headroom left for any fusion rule, and training-session
attraction of cross-session errors. Output: evidence/S27_tta_trained_head/bottleneck_diagnostic.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "docs/research/evidence/S27_tta_trained_head/bottleneck_diagnostic.json"


def main() -> None:
    labels = np.load(ROOT / "dataset_u430k32/labels.npy")
    groups = np.load(ROOT / "dataset_u430k32/groups.npy")
    scans = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv").set_index("scan_id")
    session = scans.session_id.loc[groups].to_numpy()
    cross = scans.groupby("label").session_id.nunique().sort_index().to_numpy() > 1
    plan = json.loads((ROOT / "docs/research/evidence/S21_complementary_rgb/preregistration.json").read_text())
    s22 = pd.read_csv(ROOT / "outputs/s22_fusion_analysis/predictions.csv.gz")
    s21 = pd.read_csv(ROOT / "outputs/s21_rgb_study/predictions.csv.gz").query("arm == 'dino_rgb'")
    result: dict[str, object] = {"scope": "Descriptive, saved development predictions; no selection.", "folds": []}
    pooled = []
    for fold in (0, 1):
        split = plan["splits"][str(fold)]
        train = np.r_[split["train"], split["calib"]]
        te = np.sort(np.r_[split["val"], split["test"]])
        def pick(df: pd.DataFrame, fold: int = fold, te: np.ndarray = te) -> np.ndarray:
            return df[df.fold == fold].set_index("index").loc[te].prediction.to_numpy()
        hsi, fused, rgb = pick(s22[s22.arm == "v5_tta"]), pick(s22[s22.arm == "v5_rgb_equal"]), pick(s21)
        y = labels[te]
        trained_in = {c: set(session[train][labels[train] == c]) for c in range(90)}
        is_cross = cross[y]
        row = {"fold": fold}
        for name, pred in (("hsi_tta", hsi), ("rgb", rgb), ("equal_fusion", fused)):
            rec = np.array([np.mean(pred[y == c] == c) for c in range(90)])
            deficit_same, deficit_cross = float(np.sum(1 - rec[~cross])), float(np.sum(1 - rec[cross]))
            row[name] = {"same_recall": float(rec[~cross].mean()), "cross_recall": float(rec[cross].mean()),
                         "macro_recall_deficit_share_same": deficit_same / (deficit_same + deficit_cross)}
        either = (hsi == y) | (rgb == y)
        row["oracle"] = {k: {"either_correct": float(either[m].mean()), "fused_correct": float((fused == y)[m].mean()),
                             "both_wrong": float((~either)[m].mean())}
                         for k, m in (("all", np.ones_like(is_cross)), ("same", ~is_cross), ("cross", is_cross))}
        wrong = (fused != y) & is_cross
        attracted = np.array([session[te][i] in trained_in[fused[i]] for i in np.flatnonzero(wrong)])
        base = np.array([np.mean([session[te][i] in trained_in[c] for c in range(90) if c != y[i]])
                         for i in np.flatnonzero(wrong)])
        row["cross_error_attraction"] = {"errors": int(wrong.sum()),
                                         "pred_class_trained_in_test_session": float(attracted.mean()),
                                         "chance_if_uniform_over_wrong_classes": float(base.mean())}
        result["folds"].append(row)  # type: ignore[union-attr]
        pooled.append(row)
    keys = ("hsi_tta", "rgb", "equal_fusion")
    result["mean"] = {k: {m: float(np.mean([r[k][m] for r in pooled])) for m in pooled[0][k]} for k in keys}
    result["mean"]["oracle"] = {s: {m: float(np.mean([r["oracle"][s][m] for r in pooled])) for m in pooled[0]["oracle"][s]}
                                for s in ("all", "same", "cross")}
    result["mean"]["cross_error_attraction"] = {m: float(np.mean([r["cross_error_attraction"][m] for r in pooled]))
                                                for m in ("pred_class_trained_in_test_session", "chance_if_uniform_over_wrong_classes")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["mean"], indent=1))


if __name__ == "__main__":
    main()
