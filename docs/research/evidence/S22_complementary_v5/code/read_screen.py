"""Read-only S22/S21 synthesis; archive compact artifacts without fitting models."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import class_metrics, cluster_interval, verify_plan

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = ROOT / "docs/research/evidence/S22_complementary_v5"
OUTPUT = ROOT / "outputs/s22_fusion_analysis"
PLAN = ROOT / "configs/research/s22_screening_amendment05.json"


def main() -> None:
    verify_plan(PLAN)
    complete = json.loads((OUTPUT / "COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(PLAN):
        raise ValueError("Analysis did not run the active amendment")
    for name, digest in complete["files"].items():
        if sha256(OUTPUT / name) != digest:
            raise ValueError("Analysis artifact hash mismatch")
    archive = EVIDENCE / "screen_results"
    archive.mkdir(exist_ok=False)
    for name in [*complete["files"], "COMPLETED.json"]:
        shutil.copy2(OUTPUT / name, archive / name)
    neural = pd.read_csv(OUTPUT / "predictions.csv.gz")
    prior = pd.read_csv(ROOT / "docs/research/evidence/S21_complementary_rgb/predictions.csv.gz")
    prior = prior[prior.arm.isin(["dino_rgb", "hsi_q32", "fusion32_equal"])].copy()
    prior["seed"] = "deterministic"
    predictions = pd.concat([neural, prior], ignore_index=True)
    meta = pd.read_csv(ROOT / "docs/research/evidence/S20_rgb_pathway/kernel_manifest.csv").set_index("index")
    scans = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv")
    cross = scans.groupby("label").session_id.nunique().to_numpy() > 1
    labels = np.load(ROOT / "dataset_u430k32/labels.npy")
    coverage, per_class, fold_metrics = [], [], []
    for arm, all_rows in predictions.groupby("arm"):
        if not np.array_equal(np.sort(all_rows["index"].to_numpy()), np.arange(len(labels))):
            raise ValueError("Candidate does not hold every row out exactly once")
        coverage.append({"arm": arm, "held_out_rows": len(all_rows), "unique_rows": all_rows["index"].nunique()})
        for fold, block in all_rows.groupby("fold"):
            rows = block["index"].to_numpy()
            target = block.target.to_numpy()
            if not np.array_equal(labels[rows], target):
                raise ValueError("Saved target identity mismatch")
            stat = class_metrics(target, block.prediction.to_numpy(), 90)
            fold_metrics.append({"arm": arm, "fold": int(fold), "f1": float(stat["f1"].mean()),
                                 "accuracy": float(np.mean(target == block.prediction.to_numpy())),
                                 "same_recall": float(stat["recall"][~cross].mean()),
                                 "cross_recall": float(stat["recall"][cross].mean())})
            for c in range(90):
                session = meta.loc[rows[target == c], "session_id"].unique()
                if len(session) != 1:
                    raise ValueError("A held-out class has multiple sessions")
                per_class.append({"arm": arm, "fold": int(fold), "label": c, "cross": bool(cross[c]),
                                  "destination_session": int(session[0]), "f1": float(stat["f1"][c]),
                                  "recall": float(stat["recall"][c]), "support": int(stat["support"][c]),
                                  "correct": int(stat["correct"][c])})
    classes = pd.DataFrame(per_class)
    summary = []
    for arm, block in classes.groupby("arm"):
        f1 = block.pivot(index="label", columns="fold", values="f1").to_numpy()
        recall = block.pivot(index="label", columns="fold", values="recall").to_numpy()
        means = pd.DataFrame(fold_metrics).query("arm == @arm")
        correct = block.pivot(index="label", columns="fold", values="correct").to_numpy()
        support = block.pivot(index="label", columns="fold", values="support").to_numpy()
        draws = np.random.default_rng(20261004).integers(90, size=(2000, 90))
        accuracy_boot = (correct[draws].sum(1) / support[draws].sum(1)).mean(1)
        summary.append({"arm": arm, "f1": float(f1.mean()), "f1_ci": cluster_interval(f1),
                        "accuracy": float(means.accuracy.mean()), "accuracy_ci": np.quantile(accuracy_boot, [.025, .975]).tolist(),
                        "same_recall": float(recall[~cross].mean()), "same_ci": cluster_interval(recall[~cross]),
                        "cross_recall": float(recall[cross].mean()), "cross_ci": cluster_interval(recall[cross])})
    complements = []
    for fold in (0, 1):
        rows = predictions[predictions.fold == fold]
        rgb = rows[rows.arm == "dino_rgb"].set_index("index").sort_index()
        hsi = rows[rows.arm == "v5_tta"].set_index("index").sort_index()
        fusion = rows[rows.arm == "v5_rgb_equal"].set_index("index").sort_index()
        if not rgb.index.equals(hsi.index) or not rgb.index.equals(fusion.index):
            raise ValueError("Fusion comparator row identities differ")
        r, h, f = [(v.target == v.prediction).to_numpy() for v in (rgb, hsi, fusion)]
        for subset, mask in (("all", np.ones(len(r), bool)), ("same", ~cross[rgb.target.to_numpy()]),
                             ("cross", cross[rgb.target.to_numpy()])):
            complements.append({"fold": fold, "subset": subset, "n": int(mask.sum()),
                "both_correct": float((r[mask] & h[mask]).mean()),
                "rgb_only_correct": float((r[mask] & ~h[mask]).mean()),
                "hsi_only_correct": float((h[mask] & ~r[mask]).mean()),
                "both_wrong": float((~r[mask] & ~h[mask]).mean()),
                "oracle_accuracy": float((r[mask] | h[mask]).mean()),
                "fusion_rescues_hsi": float((f[mask] & ~h[mask]).mean()),
                "fusion_harms_hsi": float((~f[mask] & h[mask]).mean())})
    direction = classes[classes.cross].copy()
    direction["direction"] = np.where(direction.destination_session == 8, "to_session8", "from_session8")
    # Within each direction, a bridge appears once across the two folds.
    direction_summary = []
    for (arm, d), block in direction.groupby(["arm", "direction"]):
        if block.label.nunique() != 17 or len(block) != 17:
            raise ValueError("A bridge direction is absent or repeated")
        direction_summary.append({"arm": arm, "direction": d, "classes": 17,
                                  "recall": float(block.recall.mean()), "ci": cluster_interval(block.recall.to_numpy())})
    pd.DataFrame(fold_metrics).to_csv(archive / "all_fold_metrics.csv", index=False)
    classes.to_csv(archive / "all_per_class.csv", index=False)
    pd.DataFrame(complements).to_csv(archive / "complementarity.csv", index=False)
    pd.DataFrame(direction_summary).to_csv(archive / "acquisition_directions.csv", index=False)
    write_json(archive / "summary.json", summary)
    write_json(archive / "coverage.json", coverage)
    write_json(archive / "ARCHIVED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in archive.iterdir() if p.is_file()}})
    print(pd.DataFrame(summary)[["arm", "f1", "same_recall", "cross_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
