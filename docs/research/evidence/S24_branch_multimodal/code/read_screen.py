"""Replay and archive the S24 branch-removal screen; no fitting or selection."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import class_metrics, cluster_interval, verify_plan

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs/s24_branch_multimodal"
PLAN = ROOT / "configs/research/s24_branch_multimodal.json"
ARCHIVE = ROOT / "docs/research/evidence/S24_branch_multimodal/screen_results"


def checked_output(path, plan):
    complete = json.loads((path / "COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(plan):
        raise ValueError("Completed plan differs")
    for name, digest in complete["files"].items():
        if sha256(path / name) != digest:
            raise ValueError("Completed artifact changed")
    return complete


def main():
    spec = verify_plan(PLAN)
    checked_output(OUT, PLAN)
    s23 = ROOT / "outputs/s23_frozen_multimodal"
    checked_output(s23, ROOT / spec["parent_plan"])
    s22 = ROOT / "outputs/s22_fusion_analysis"
    checked_output(s22, ROOT / "configs/research/s22_screening_amendment05.json")
    labels = np.load(ROOT / "dataset_u430k32/labels.npy")
    scans = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv")
    cross = scans.groupby("label").session_id.nunique().to_numpy() > 1
    meta_path = ROOT / "docs/research/evidence/S20_rgb_pathway/kernel_manifest.csv"
    meta = pd.read_csv(meta_path).set_index("index")
    if not np.array_equal(meta.sort_index().label.to_numpy(), labels):
        raise ValueError("Kernel metadata targets differ")
    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    saved = pd.read_csv(OUT / "metrics.csv").set_index(["arm", "fold"])
    coverage, contributions = [], []
    for arm, rows in predictions.groupby("arm"):
        if not np.array_equal(np.sort(rows["index"].to_numpy()), np.arange(len(labels))):
            raise ValueError("A branch control omits or repeats held-out rows")
        coverage.append({"arm": arm, "rows": len(rows), "unique_rows": rows["index"].nunique()})
        for fold, block in rows.groupby("fold"):
            ids = block["index"].to_numpy()
            if not np.array_equal(labels[ids], block.target.to_numpy()):
                raise ValueError("Saved branch-control targets differ")
            stat = class_metrics(block.target.to_numpy(), block.prediction.to_numpy(), 90)
            for metric, value in (("f1", stat["f1"].mean()), ("accuracy", (block.target == block.prediction).mean()),
                                  ("same_recall", stat["recall"][~cross].mean()),
                                  ("cross_recall", stat["recall"][cross].mean())):
                if not np.isclose(saved.loc[(arm, fold), metric], value, atol=1e-12, rtol=0):
                    raise ValueError("Branch-control metric arithmetic differs")
            for label in range(90):
                dest = meta.loc[ids[block.target.to_numpy() == label], "session_id"].unique()
                if len(dest) != 1:
                    raise ValueError("A held-out class spans destinations")
                contributions.append({"fold": int(fold), "arm": arm, "label": label,
                                      "cross": bool(cross[label]), "destination_session": int(dest[0]),
                                      "f1": float(stat["f1"][label]), "recall": float(stat["recall"][label])})
    reference_root = ROOT / "docs/research/evidence/S23_frozen_multimodal/screen_results"
    reference_manifest = json.loads((reference_root / "ARCHIVED.json").read_text())
    for name, digest in reference_manifest["files"].items():
        if sha256(reference_root / name) != digest:
            raise ValueError("S23 reference archive changed")
    reference = pd.read_csv(reference_root / "all_per_class.csv")
    reference = reference[reference.arm.isin(["equal_single", "learned_residual", "v5_rgb_equal"])]
    classes = pd.concat([pd.DataFrame(contributions), reference], ignore_index=True)
    summaries = []
    for arm, block in classes.groupby("arm"):
        f1 = block.pivot(index="label", columns="fold", values="f1").to_numpy()
        recall = block.pivot(index="label", columns="fold", values="recall").to_numpy()
        summaries.append({"arm": arm, "f1": float(f1.mean()), "f1_ci": cluster_interval(f1),
                          "same_recall": float(recall[~cross].mean()), "same_ci": cluster_interval(recall[~cross]),
                          "cross_recall": float(recall[cross].mean()), "cross_ci": cluster_interval(recall[cross])})
    pairs = []
    for arm in spec["arms"]:
        a = classes[classes.arm == arm].set_index(["label", "fold"]).sort_index()
        for control in ("equal_single", "learned_residual", "v5_rgb_equal"):
            b = classes[classes.arm == control].set_index(["label", "fold"]).sort_index()
            if not a.index.equals(b.index):
                raise ValueError("Paired reference identities differ")
            for metric, subset in (("f1", "all"), ("recall", "same"), ("recall", "cross")):
                delta = a[metric]-b[metric]
                if subset != "all":
                    delta = delta[a.cross if subset == "cross" else ~a.cross]
                matrix = delta.unstack("fold").to_numpy()
                pairs.append({"arm": arm, "control": control, "metric": metric, "subset": subset,
                              "mean": float(matrix.mean()), "ci": cluster_interval(matrix),
                              "fold_deltas": matrix.mean(0).tolist()})
    direction = classes[classes.cross].copy()
    direction["direction"] = np.where(direction.destination_session == 8, "to_session8", "from_session8")
    directions = []
    for (arm, dest), block in direction.groupby(["arm", "direction"]):
        if len(block) != 17 or block.label.nunique() != 17:
            raise ValueError("Direction coverage differs")
        directions.append({"arm": arm, "direction": dest, "recall": float(block.recall.mean()),
                           "ci": cluster_interval(block.recall.to_numpy())})
    ARCHIVE.mkdir(exist_ok=False)
    for name in ("metrics.csv", "per_class.csv", "predictions.csv.gz", "learning_curves.csv",
                 "selection.json", "selection_f0.json", "selection_f1.json", "hypothesis.json", "COMPLETED.json"):
        shutil.copy2(OUT / name, ARCHIVE / name)
    shutil.copy2(ROOT / "outputs/s24_execution.json", ARCHIVE / "execution.json")
    classes.to_csv(ARCHIVE / "all_per_class.csv", index=False)
    pd.DataFrame(directions).to_csv(ARCHIVE / "acquisition_directions.csv", index=False)
    write_json(ARCHIVE / "coverage.json", coverage)
    write_json(ARCHIVE / "summary.json", summaries)
    write_json(ARCHIVE / "paired_deltas.json", pairs)
    write_json(ARCHIVE / "ARCHIVED.json", {"plan_sha256": sha256(PLAN), "metadata_sha256": sha256(meta_path),
               "source_sha256": sha256(Path(__file__)),
               "files": {p.name: sha256(p) for p in ARCHIVE.iterdir() if p.is_file()}})
    print(pd.DataFrame(summaries)[["arm", "f1", "same_recall", "cross_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
