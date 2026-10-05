#!/usr/bin/env python
"""Archive a completed S27/S28 screen: verify hashes, replay metric arithmetic from
saved predictions, add acquisition-direction recall, copy compact evidence.

Usage: python archive_screen.py <outputs dir> <frozen plan> <evidence dir> [extra files...]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import class_metrics, cluster_interval, verify_plan

ROOT = Path(__file__).resolve().parents[5]


def main(out: Path, plan: Path, archive: Path, extra: list[Path]) -> None:
    verify_plan(plan)
    complete = json.loads((out / "COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(plan):
        raise ValueError("Completion plan differs")
    for name, digest in complete["files"].items():
        if sha256(out / name) != digest:
            raise ValueError(f"Completed artifact changed: {name}")
    labels = np.load(ROOT / "dataset_u430k32/labels.npy")
    cross = pd.read_csv(ROOT / "dataset_u430k32/scan_table.csv").groupby("label").session_id.nunique().sort_index().to_numpy() > 1
    meta_path = ROOT / "docs/research/evidence/S20_rgb_pathway/kernel_manifest.csv"
    meta = pd.read_csv(meta_path).set_index("index")
    archive.mkdir(parents=True, exist_ok=False)
    for p in sorted(out.iterdir()):
        if p.is_file() and p.suffix in (".json", ".csv", ".gz"):
            shutil.copy2(p, archive / p.name)
    for p in extra:
        shutil.copy2(p, archive / p.name)
    directions: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    if (out / "predictions.csv.gz").exists():
        pred = pd.read_csv(out / "predictions.csv.gz")
        saved = pd.read_csv(out / "metrics.csv").set_index(["arm", "fold"])
        rows = []
        for arm, block in pred.groupby("arm"):
            if not np.array_equal(np.sort(block["index"].to_numpy()), np.arange(len(labels))):
                raise ValueError(f"{arm}: incomplete or repeated held-out coverage")
            coverage.append({"arm": arm, "held_out_rows": len(block), "unique_rows": int(block["index"].nunique())})
            for fold, part in block.groupby("fold"):
                ids = part["index"].to_numpy()
                if not np.array_equal(part.target.to_numpy(), labels[ids]):
                    raise ValueError("Target identity differs")
                stat = class_metrics(part.target.to_numpy(), part.prediction.to_numpy(), 90)
                for metric, value in (("f1", stat["f1"].mean()), ("cross_recall", stat["recall"][cross].mean()),
                                      ("same_recall", stat["recall"][~cross].mean())):
                    if not np.isclose(saved.loc[(arm, fold), metric], value, atol=1e-12, rtol=0):
                        raise ValueError(f"{arm}/{fold}: metric arithmetic differs")
                for c in np.flatnonzero(cross):
                    dest = meta.loc[ids[part.target.to_numpy() == c], "session_id"].unique()
                    if len(dest) != 1:
                        raise ValueError("Held-out class spans destinations")
                    rows.append({"arm": arm, "fold": int(fold), "label": int(c), "recall": float(stat["recall"][c]),
                                 "direction": "to_session8" if dest[0] == 8 else "from_session8"})
        table = pd.DataFrame(rows)
        for (arm, d), block in table.groupby(["arm", "direction"]):
            directions.append({"arm": arm, "direction": d, "cells": len(block), "recall": float(block.recall.mean()),
                               "ci": cluster_interval(block.recall.to_numpy())})
        pd.DataFrame(directions).to_csv(archive / "acquisition_directions.csv", index=False)
        write_json(archive / "coverage.json", coverage)
    write_json(archive / "ARCHIVED.json", {"plan_sha256": sha256(plan), "metadata_sha256": sha256(meta_path),
               "source_sha256": sha256(Path(__file__)),
               "files": {p.name: sha256(p) for p in archive.iterdir() if p.is_file()}})
    print(json.dumps({"coverage": coverage, "directions": directions}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(ROOT / a[0], ROOT / a[1], ROOT / a[2], [ROOT / p for p in a[3:]])
