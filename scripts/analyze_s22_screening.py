#!/usr/bin/env python
"""Once-only S22 v5/RGB fusion analysis after the amended two-cell screen finishes."""
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
    p.add_argument("--plan", type=Path, default=Path("configs/research/s22_screening_amendment03.json"))
    p.add_argument("--runs", type=Path, default=Path("outputs/s22_complementary_v5"))
    p.add_argument("--output", type=Path, default=Path("outputs/s22_fusion_analysis"))
    args = p.parse_args()
    code_receipt = verify_plan(Path("configs/research/s22_analysis_code04.json"))
    if code_receipt["training_plan_sha256"] != sha256(args.plan):
        raise ValueError("Analysis-code receipt belongs to another training plan")
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
    per_class: list[dict[str, int | str | float | bool]] = []
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
                            "accuracy": float(np.mean(pred == y[te])),
                            "f1_ci": cluster_interval(stats["f1"]),
                            "same_ci": cluster_interval(stats["recall"][~cross]),
                            "cross_ci": cluster_interval(stats["recall"][cross]),
                            "same_recall": float(stats["recall"][~cross].mean()),
                            "cross_recall": float(stats["recall"][cross].mean())})
            per_class.extend({"fold": fold, "seed": seed, "arm": arm, "label": c,
                              "f1": float(stats["f1"][c]), "recall": float(stats["recall"][c]),
                              "cross": bool(cross[c])} for c in range(90))
            predictions.extend({"fold": fold, "seed": seed, "arm": arm,
                                "index": int(i), "target": int(t), "prediction": int(v)}
                               for i, t, v in zip(te, y[te], pred, strict=True))
        deltas.append(pair["v5_rgb_equal"]["f1"] - pair["v5_tta"]["f1"])
        cross_deltas.append((pair["v5_rgb_equal"]["recall"] - pair["v5_tta"]["recall"])[cross])
        selections.append({"fold": fold, "seed": seed, "temperature": temperature,
                           "float16_argmax_changes": int(np.count_nonzero(logits.argmax(1) != baseline))})
    df, dr = np.mean(deltas, axis=0), np.mean(cross_deltas, axis=0)
    ci, cr = cluster_interval(df), cluster_interval(dr)
    pd.DataFrame(per_class).to_csv(args.output / "per_class.csv", index=False)
    pd.DataFrame(records).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "predictions.csv.gz", index=False)
    write_json(args.output / "calibration.json", selections)
    fold_gains = [float(v.mean()) for v in deltas]
    fold_cross = [float(v.mean()) for v in cross_deltas]
    passes = bool(df.mean() >= .02 and ci[0] > 0 and dr.mean() >= 0 and min(fold_gains) > 0)
    write_json(args.output / "hypothesis.json", {"H40_screen": passes,
        "H40_original_confirmatory": "not evaluated: original three-seed design amended before training",
        "fold_f1_deltas": fold_gains, "fold_cross_deltas": fold_cross,
        "development_gate": passes,
        "seed_replication": "defer to final learned candidate and matched controls" if passes else "no automatic replication; diagnose direction disagreement or small gain",
        "confirmation_required_before_paper_claim": True,
        "delta_f1": float(df.mean()), "ci": ci, "delta_cross": float(dr.mean()), "cross_ci": cr,
        "transfer_gain_supported": bool(cr[0] > 0),
        "scope": "One retrained HSI seed on both corrected folds; shared frozen RGB probes; existing acquisitions. Class intervals exclude seed uncertainty."})
    write_json(args.output / "COMPLETED.json", {"plan_sha256": sha256(args.plan),
        "analysis_code_receipt_sha256": sha256(Path("configs/research/s22_analysis_code04.json")),
        "files": {f.name: sha256(f) for f in args.output.iterdir() if f.is_file()}})


if __name__ == "__main__":
    main()
