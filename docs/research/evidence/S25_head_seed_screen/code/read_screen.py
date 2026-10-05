"""Archive S25 head sensitivity and replay saved prediction arithmetic."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import class_metrics, cluster_interval, verify_plan

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs/s25_head_seed_screen"
PLAN = ROOT / "configs/research/s25_head_seed_screen.json"
ARCHIVE = ROOT / "docs/research/evidence/S25_head_seed_screen/screen_results"


def main():
    verify_plan(PLAN)
    complete = json.loads((OUT / "COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(PLAN):
        raise ValueError("Completion plan differs")
    for name, digest in complete["files"].items():
        if sha256(OUT / name) != digest:
            raise ValueError("Completed artifact changed")
    reference_root = ROOT / "docs/research/evidence/S23_frozen_multimodal/screen_results"
    manifest = json.loads((reference_root / "ARCHIVED.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(reference_root / name) != digest:
            raise ValueError("S23 reference archive changed")
    reference = pd.read_csv(reference_root / "all_per_class.csv")
    labels = np.load(ROOT / "dataset_u430k32/labels.npy")
    scans = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv")
    cross = scans.groupby("label").session_id.nunique().to_numpy() > 1
    meta_path = ROOT / "docs/research/evidence/S20_rgb_pathway/kernel_manifest.csv"
    meta = pd.read_csv(meta_path).set_index("index")
    pred = pd.read_csv(OUT / "predictions.csv.gz")
    saved = pd.read_csv(OUT / "metrics.csv").set_index(["seed", "fold"])
    coverage, classes = [], []
    for seed, all_rows in pred.groupby("seed"):
        if not np.array_equal(np.sort(all_rows["index"].to_numpy()), np.arange(len(labels))):
            raise ValueError("Head seed has incomplete or repeated row coverage")
        coverage.append({"head_seed": int(seed), "encoder_seed": 0, "held_out_rows": len(all_rows),
                         "unique_rows": all_rows["index"].nunique()})
        for fold, rows in all_rows.groupby("fold"):
            ids = rows["index"].to_numpy()
            if not np.array_equal(rows.target.to_numpy(), labels[ids]):
                raise ValueError("Head seed targets differ")
            stat = class_metrics(rows.target.to_numpy(), rows.prediction.to_numpy(), 90)
            for metric, value in (("f1", stat["f1"].mean()), ("accuracy", (rows.target == rows.prediction).mean()),
                                  ("same_recall", stat["recall"][~cross].mean()),
                                  ("cross_recall", stat["recall"][cross].mean())):
                if not np.isclose(saved.loc[(seed, fold), metric], value, atol=1e-12, rtol=0):
                    raise ValueError("Head seed metric arithmetic differs")
            for c in range(90):
                dest = meta.loc[ids[rows.target.to_numpy() == c], "session_id"].unique()
                if len(dest) != 1:
                    raise ValueError("Held-out class spans destinations")
                classes.append({"seed": int(seed), "fold": int(fold), "label": c, "cross": bool(cross[c]),
                                "destination_session": int(dest[0]), "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])})
    table = pd.DataFrame(classes)
    summary = []
    for metric, mask in (("f1", np.ones(90, bool)), ("recall", ~cross), ("recall", cross)):
        matrix = table.pivot(index="label", columns=["seed", "fold"], values=metric).to_numpy()[mask]
        summary.append({"metric": metric, "subset": "all" if metric == "f1" else "same" if mask.sum() == 73 else "cross",
                        "mean": float(matrix.mean()), "ci": cluster_interval(matrix)})
    paired = []
    for control in ("equal_single", "v5_rgb_equal"):
        b = reference[reference.arm == control].set_index(["label", "fold"]).sort_index()
        for metric, subset in (("f1", "all"), ("recall", "same"), ("recall", "cross")):
            deltas = []
            for _, block in table.groupby("seed"):
                a = block.set_index(["label", "fold"]).sort_index()
                if not a.index.equals(b.index):
                    raise ValueError("Head/reference class identities differ")
                delta = a[metric]-b[metric]
                if subset != "all":
                    delta = delta[a.cross if subset == "cross" else ~a.cross]
                deltas.append(delta.unstack("fold").to_numpy())
            matrix = np.mean(deltas, axis=0)
            paired.append({"control": control, "metric": metric, "subset": subset,
                           "mean": float(matrix.mean()), "ci": cluster_interval(matrix),
                           "fold_mean_deltas": matrix.mean(0).tolist()})
    direction = table[table.cross].copy()
    direction["direction"] = np.where(direction.destination_session == 8, "to_session8", "from_session8")
    directions = []
    for dest, block in direction.groupby("direction"):
        matrix = block.pivot(index="label", columns="seed", values="recall").to_numpy()
        if matrix.shape != (17, 3):
            raise ValueError("Head-seed acquisition direction coverage differs")
        directions.append({"direction": dest, "recall": float(matrix.mean()), "ci": cluster_interval(matrix)})
    ARCHIVE.mkdir(exist_ok=False)
    for name in ("metrics.csv", "per_class.csv", "predictions.csv.gz", "learning_curves.csv", "selection.json",
                 "selection_f0.json", "selection_f1.json", "cache_replay.json", "hypothesis.json", "COMPLETED.json"):
        shutil.copy2(OUT / name, ARCHIVE / name)
    shutil.copy2(ROOT / "outputs/s25_execution.json", ARCHIVE / "execution.json")
    table.to_csv(ARCHIVE / "all_per_class.csv", index=False)
    pd.DataFrame(directions).to_csv(ARCHIVE / "acquisition_directions.csv", index=False)
    write_json(ARCHIVE / "coverage.json", coverage)
    write_json(ARCHIVE / "summary.json", summary)
    write_json(ARCHIVE / "paired_deltas.json", paired)
    write_json(ARCHIVE / "ARCHIVED.json", {"plan_sha256": sha256(PLAN), "metadata_sha256": sha256(meta_path),
               "source_sha256": sha256(Path(__file__)),
               "files": {p.name: sha256(p) for p in ARCHIVE.iterdir() if p.is_file()}})
    print(json.dumps({"summary": summary, "paired": paired, "directions": directions}, indent=2))


if __name__ == "__main__":
    main()
