"""Archive and replay the fixed S23 screen; no model fitting or selection."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import class_metrics, cluster_interval, probe_logits, verify_plan

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "outputs/s23_frozen_multimodal"
PLAN = ROOT / "configs/research/s23_frozen_multimodal.json"
ARCHIVE = ROOT / "docs/research/evidence/S23_frozen_multimodal/screen_results"


def main() -> None:
    verify_plan(PLAN)
    complete = json.loads((OUTPUT / "COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(PLAN):
        raise ValueError("S23 completion plan mismatch")
    for name, digest in complete["files"].items():
        if sha256(OUTPUT / name) != digest:
            raise ValueError("S23 completed artifact changed")
    # Keep the reference evidence fixed, including its saved arithmetic.
    s22 = ROOT / "docs/research/evidence/S22_complementary_v5/screen_results"
    manifest = json.loads((s22 / "ARCHIVED.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(s22 / name) != digest:
            raise ValueError("S22 reference archive changed")
    labels = np.load(ROOT / "dataset_u430k32/labels.npy")
    scans = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv")
    cross = scans.groupby("label").session_id.nunique().to_numpy() > 1
    meta = pd.read_csv(ROOT / "docs/research/evidence/S20_rgb_pathway/kernel_manifest.csv").set_index("index")
    predictions = pd.read_csv(OUTPUT / "predictions.csv.gz")
    # Audit exported RGB inference and warning flags without fitting anything.
    rgb = np.load(ROOT / "outputs/s20_rgb_features_v3/rgb.npy")
    prior_rgb = pd.read_csv(ROOT / "docs/research/evidence/S21_complementary_rgb/predictions.csv.gz")
    prior_rgb = prior_rgb[prior_rgb.arm == "dino_rgb"].set_index(["fold", "index"]).sort_index()
    current_rgb = predictions[predictions.arm == "rgb"].set_index(["fold", "index"]).sort_index()
    if not prior_rgb.index.equals(current_rgb.index):
        raise ValueError("RGB control uses different S21 rows")
    rgb_audit = {"prediction_disagreements_vs_s21": int((prior_rgb.prediction != current_rgb.prediction).sum()),
                 "scope": "Post-run arithmetic audit; no fitting or checkpoint selection", "folds": []}
    for fold in (0, 1):
        logits = probe_logits(ROOT / f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz", rgb)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite exported RGB probe logits")
        rgb_audit["folds"].append({"fold": fold, "rows": len(rgb), "finite_logits": True,
                                   "max_abs_logit": float(np.abs(logits).max())})
    saved = pd.read_csv(OUTPUT / "metrics.csv").set_index(["arm", "fold"])
    per_class, coverage = [], []
    for arm, rows in predictions.groupby("arm"):
        if not np.array_equal(np.sort(rows["index"].to_numpy()), np.arange(len(labels))):
            raise ValueError("S23 does not hold every kernel out exactly once")
        coverage.append({"arm": arm, "rows": len(rows), "unique_rows": rows["index"].nunique()})
        for fold, block in rows.groupby("fold"):
            ids = block["index"].to_numpy()
            if not np.array_equal(labels[ids], block.target.to_numpy()):
                raise ValueError("S23 target identities differ")
            stat = class_metrics(block.target.to_numpy(), block.prediction.to_numpy(), 90)
            for metric, value in (("f1", stat["f1"].mean()),
                                  ("accuracy", (block.target == block.prediction).mean()),
                                  ("same_recall", stat["recall"][~cross].mean()),
                                  ("cross_recall", stat["recall"][cross].mean())):
                if not np.isclose(saved.loc[(arm, fold), metric], value, atol=1e-12, rtol=0):
                    raise ValueError("S23 prediction/metric arithmetic mismatch")
            for label in range(90):
                session = meta.loc[ids[block.target.to_numpy() == label], "session_id"].unique()
                if len(session) != 1:
                    raise ValueError("Held-out variety spans destinations")
                per_class.append({"arm": arm, "fold": int(fold), "label": label,
                                  "cross": bool(cross[label]), "destination_session": int(session[0]),
                                  "f1": float(stat["f1"][label]), "recall": float(stat["recall"][label])})
    classes = pd.DataFrame(per_class)
    reference = pd.read_csv(s22 / "all_per_class.csv")
    reference = reference[reference.arm == "v5_rgb_equal"]
    classes = pd.concat([classes, reference], ignore_index=True)
    summary = []
    for arm, block in classes.groupby("arm"):
        f1 = block.pivot(index="label", columns="fold", values="f1").to_numpy()
        recall = block.pivot(index="label", columns="fold", values="recall").to_numpy()
        summary.append({"arm": arm, "f1": float(f1.mean()), "f1_ci": cluster_interval(f1),
                        "same_recall": float(recall[~cross].mean()), "same_ci": cluster_interval(recall[~cross]),
                        "cross_recall": float(recall[cross].mean()), "cross_ci": cluster_interval(recall[cross])})
    deltas = []
    learned = classes[classes.arm == "learned_residual"].set_index(["label", "fold"]).sort_index()
    for arm in ("equal_single", "v5_rgb_equal"):
        control = classes[classes.arm == arm].set_index(["label", "fold"]).sort_index()
        if not learned.index.equals(control.index):
            raise ValueError("S23 paired comparator identities differ")
        for metric, subset in (("f1", "all"), ("recall", "same"), ("recall", "cross")):
            delta = (learned[metric] - control[metric]).unstack("fold").to_numpy()
            if subset != "all":
                delta = delta[cross if subset == "cross" else ~cross]
            deltas.append({"control": arm, "metric": metric, "subset": subset,
                           "mean": float(delta.mean()), "ci": cluster_interval(delta),
                           "fold_deltas": delta.mean(0).tolist()})
    direction = classes[classes.cross].copy()
    direction["direction"] = np.where(direction.destination_session == 8, "to_session8", "from_session8")
    directions = []
    for (arm, dest), block in direction.groupby(["arm", "direction"]):
        if len(block) != 17 or block.label.nunique() != 17:
            raise ValueError("Bridge direction omitted/repeated")
        directions.append({"arm": arm, "direction": dest, "recall": float(block.recall.mean()),
                           "ci": cluster_interval(block.recall.to_numpy())})
    ARCHIVE.mkdir(exist_ok=False)
    for name in ("metrics.csv", "per_class.csv", "predictions.csv.gz", "learning_curves.csv",
                 "selection.json", "selection_f0.json", "selection_f1.json", "inference_audit.json",
                 "hypothesis.json", "COMPLETED.json"):
        shutil.copy2(OUTPUT / name, ARCHIVE / name)
    classes.to_csv(ARCHIVE / "all_per_class.csv", index=False)
    pd.DataFrame(directions).to_csv(ARCHIVE / "acquisition_directions.csv", index=False)
    write_json(ARCHIVE / "coverage.json", coverage)
    write_json(ARCHIVE / "summary.json", summary)
    write_json(ARCHIVE / "paired_deltas.json", deltas)
    write_json(ARCHIVE / "rgb_inference_audit.json", rgb_audit)
    write_json(ARCHIVE / "ARCHIVED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in ARCHIVE.iterdir() if p.is_file()}})
    print(pd.DataFrame(summary)[["arm", "f1", "same_recall", "cross_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
