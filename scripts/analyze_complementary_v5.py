#!/usr/bin/env python
"""Once-only S22 v5/RGB fusion analysis after all six frozen GPU cells finish."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import (
    aligned_logits,
    calibrate_temperature,
    class_metrics,
    cluster_interval,
    verify_plan,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("configs/research/s22_complementary_v5.json"))
    p.add_argument("--runs", type=Path, default=Path("outputs/s22_complementary_v5"))
    p.add_argument("--output", type=Path, default=Path("outputs/s22_fusion_analysis"))
    args = p.parse_args()
    plan = verify_plan(args.plan)
    for name, digest in plan["analysis_inputs"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Analysis input changed: {name}")
    if args.output.exists():
        raise FileExistsError("Analysis already started/completed")
    for cell in plan["cells"]:
        root = args.runs / f"f{cell['fold']}_s{cell['seed']}"
        provenance = json.loads((root / "provenance.json").read_text())
        if (provenance["training_plan_sha256"], provenance["partition_plan_sha256"],
            provenance["fold"], provenance["seed"], provenance["action"]) != (
            sha256(args.plan), sha256(Path(plan["partition_plan"])), cell["fold"], cell["seed"], "train"
        ):
            raise ValueError("GPU cell identity or plan differs from the frozen cell")
        if not (root / "results/run.json").exists():
            raise ValueError(f"Incomplete GPU cell: {root}")
    args.output.mkdir(parents=True)
    partitions = json.loads(Path(plan["partition_plan"]).read_text())
    y = np.load("dataset_u430k32/labels.npy")
    scans = pd.read_csv("dataset_u430k32/scan_table.csv")
    cross = scans.groupby("label").session_id.nunique().to_numpy() > 1
    records, selections = [], []
    predictions: list[dict[str, int | str]] = []
    deltas, cross_deltas = [], []
    for cell in plan["cells"]:
        fold, seed = cell["fold"], cell["seed"]
        split = partitions["splits"][str(fold)]
        ca = np.array(split["calib"])
        te = np.sort(np.r_[split["val"], split["test"]])
        root = args.runs / f"f{fold}_s{seed}" / "results"
        cal = aligned_logits(root / "logits_calib_tta.npz", ca, y)
        logits = aligned_logits(root / "logits_val_test_tta.npz", te, y)
        temperature = calibrate_temperature(cal, y[ca], partitions["temperatures"])
        with np.load(f"outputs/s21_rgb_study/probabilities_f{fold}.npz") as rgb:
            if not np.array_equal(rgb["indices"], te):
                raise ValueError("S21 RGB rows differ from the GPU partition")
            fusion = (softmax(logits / temperature, axis=1) + rgb["dino_rgb"]) / 2
        rows = np.load(root / "rows_val_test_tta.npy")
        original = np.load(root / "preds_val_test_tta.npy")
        lookup = {int(r): int(v) for r, v in zip(rows, original, strict=True)}
        baseline = np.array([lookup[int(r)] for r in te])
        pair = {}
        for arm, pred in [("v5_tta", baseline), ("v5_rgb_equal", fusion.argmax(1))]:
            stats = class_metrics(y[te], pred, 90)
            pair[arm] = stats
            records.append({"fold": fold, "seed": seed, "arm": arm,
                            "f1": float(stats["f1"].mean()),
                            "same_recall": float(stats["recall"][~cross].mean()),
                            "cross_recall": float(stats["recall"][cross].mean())})
            predictions.extend({"fold": fold, "seed": seed, "arm": arm,
                                "index": int(i), "target": int(t), "prediction": int(v)}
                               for i, t, v in zip(te, y[te], pred, strict=True))
        deltas.append(pair["v5_rgb_equal"]["f1"] - pair["v5_tta"]["f1"])
        cross_deltas.append((pair["v5_rgb_equal"]["recall"] - pair["v5_tta"]["recall"])[cross])
        selections.append({"fold": fold, "seed": seed, "temperature": temperature,
                           "float16_argmax_changes": int(np.count_nonzero(logits.argmax(1) != baseline))})
    df, dr = np.mean(deltas, axis=0), np.mean(cross_deltas, axis=0)
    ci, cr = cluster_interval(df), cluster_interval(dr)
    pd.DataFrame(records).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "predictions.csv.gz", index=False)
    write_json(args.output / "calibration.json", selections)
    write_json(args.output / "hypothesis.json", {"H40": bool(df.mean() >= .02 and ci[0] > 0 and dr.mean() >= 0),
        "delta_f1": float(df.mean()), "ci": ci, "delta_cross": float(dr.mean()), "cross_ci": cr,
        "transfer_gain_supported": bool(cr[0] > 0),
        "scope": "Three retrained HSI seeds, shared deterministic RGB per fold; existing acquisitions."})
    write_json(args.output / "COMPLETED.json", {"plan_sha256": sha256(args.plan),
        "files": {f.name: sha256(f) for f in args.output.iterdir() if f.is_file()}})


if __name__ == "__main__":
    main()
