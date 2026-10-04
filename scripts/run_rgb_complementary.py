#!/usr/bin/env python
"""Freeze/execute S21 complementary-fold CPU screen. Every confirmation input is hashed.

Outputs are immutable per invocation: never silently overwrite a completed run.
A new question requires a new study/manifest; descriptive reanalysis uses saved
predictions and does not refit models or rescore feature arrays.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.complementary_split import complementary_split
from spectralquadnet.experiments.rgb_probe import (
    calibrate_temperature,
    class_metrics,
    cluster_interval,
    fit_probe,
    make_features,
    select_fusion,
    verify_plan,
)


def freeze(args: argparse.Namespace) -> None:
    if args.plan.exists():
        raise FileExistsError("Frozen plan already exists")
    data = args.data
    emb = args.embeddings
    manifest = json.loads((data / "MANIFEST.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete dataset")
    for name, meta in manifest["files"].items():
        if sha256(data / name) != meta["sha256"]:
            raise ValueError(f"Asset integrity failure: {name}")
    wl = pd.read_csv(data / "wavelengths.csv").iloc[:, -1].to_numpy()
    white = np.load(data / "white_spectra.npz")
    keep = pd.read_csv(data / "wavelengths.csv").iloc[:, 0].to_numpy().astype(int) - 1
    own = np.flatnonzero(np.all(white["source"][:, keep] == 0, axis=0)).tolist()
    if len(own) != 214:
        raise ValueError("Strict own-reference axis changed; redesign before freezing")
    ids195 = np.flatnonzero(wl >= 430)
    nested32 = ids195[np.round(np.linspace(0, len(ids195) - 1, 32)).astype(int)]
    # Add 32 uniformly placed unused bands, preserving all nested32 measurements.
    remaining = np.setdiff1d(ids195, nested32)
    nested64 = np.sort(
        np.r_[nested32, remaining[np.round(np.linspace(0, len(remaining) - 1, 32)).astype(int)]]
    )
    axes = {
        "32": np.load("outputs/band_finalists/uniform430_k32.npy").tolist(),
        "64": np.load("outputs/band_finalists/uniform430_k64.npy").tolist(),
        "195": ids195.tolist(),
        "215": list(range(215)),
        "214_own": own,
        "nested32": nested32.tolist(),
        "nested64": nested64.tolist(),
    }
    arms = ["rgb_shape", "rgb_color", "rgb_texture_shape", "rgb_all"] + ["hsi_q" + a for a in axes]
    arms += ["hsi_mean32", "hsi_mean214_own", "hsi_snvmean32", "hsi_snvmean214_own"] + [
        "dino_" + v for v in ["rgb", "gray", "rgb32", "silhouette"]
    ]
    arms += [
        "concat32_dino",
        "fusion32_equal",
        "fusion32_calib",
        "fusion214_equal",
        "fusion32_shuffled",
    ]
    y = np.load(data / "labels.npy")
    g = np.load(data / "groups.npy")
    splits = {}
    for fold in [0, 1]:
        s = complementary_split(y, g, calib_frac=0.15, fold=fold)
        splits[str(fold)] = {k: v.tolist() for k, v in s.items()}
    paths = [data / "MANIFEST.json", *[data / n for n in manifest["files"]]]
    paths += [
        Path(__file__),
        Path("src/spectralquadnet/experiments/rgb_probe.py"),
        Path("src/spectralquadnet/experiments/complementary_split.py"),
    ]
    for view in ["rgb", "gray", "rgb32", "silhouette"]:
        meta = json.loads((emb / (view + ".json")).read_text())
        if sha256(emb / (view + ".npy")) != meta["sha256"] or meta[
            "asset_manifest_sha256"
        ] != sha256(data / "MANIFEST.json"):
            raise ValueError("Feature cache provenance mismatch")
        paths += [emb / (view + ".json"), emb / (view + ".npy")]
    plan = {
        "repo_commit": json.loads(
            Path("docs/research/evidence/S20_rgb_pathway/starting_state.json").read_text()
        )["head"],
        "working_tree": "S21 additions identified by exact source hashes; pre-existing research edits preserved",
        "study": "S21_complementary_rgb",
        "frozen_at": "2026-10-04",
        "phase": "post-S20 protocol repair; complementary folds; historical acquisitions reused, not fresh validation",
        "data": str(data),
        "embeddings": str(emb),
        "input_hashes": {str(p): sha256(p) for p in paths},
        "axes": axes,
        "arms": arms,
        "splits": splits,
        "classifier": "training-only StandardScaler + LDA(lsqr, shrinkage=auto)",
        "temperatures": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
        "hsi_weights": [0.5, 0.25, 0.75, 0.0, 1.0],
        "fusion": "temperature-calibrated probability average; equal=.5; calib chooses alpha by macro-F1; no refit",
        "null": "fusion32_shuffled deterministically permutes RGB within each held-out scan; labels/marginals preserved; seed 20261004+fold",
        "parent_s20_sha256": "c2d4f18394bd031530ba438b4ef2cad04a5b56a97f2eee2cad6a96b67951607a",
        "hypotheses": {
            "H36": ["dino_rgb", "dino_silhouette", 0.02],
            "H37": ["fusion32_equal", "hsi_q32", 0.02],
            "H38": ["fusion32_equal", "fusion32_shuffled", 0.01],
            "H39a": ["hsi_qnested64", "hsi_qnested32", 0.01],
            "H39b": ["hsi_mean214_own", "hsi_mean32", 0.01],
        },
        "additional_deltas": [
            ["fusion32_equal", "dino_rgb"],
            ["fusion32_calib", "fusion32_equal"],
            ["concat32_dino", "fusion32_equal"],
            ["dino_rgb", "dino_gray"],
            ["dino_rgb", "dino_rgb32"],
            ["dino_rgb", "rgb_shape"],
            ["fusion214_equal", "fusion32_equal"],
            ["hsi_q215", "hsi_q214_own"],
            ["fusion32_equal", "fusion32_shuffled"],
        ],
        "decision_rule": "Threshold plus paired class-contribution CI above zero. H37/H39 additionally require nonnegative mean cross-session recall delta; CPU screen is provisional, not neural replication/G3.",
        "bootstrap": "2000 paired variety resamples, retain both fold directions; fixed per-class F1 contributions (precision denominators fixed); cross subgroup resampled separately; no session-population inference",
        "selection_policy": "No held-out tuning, reruns, or new arms. Use evidence to choose next separately frozen study. All rows reported; no model winner asserted from a maximum.",
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.plan, plan)
    args.plan.with_suffix(".sha256").write_text(sha256(args.plan) + "\n")
    print(f"Frozen {len(arms)} arms: {sha256(args.plan)}")


def execute(args: argparse.Namespace) -> None:
    plan = verify_plan(args.plan)
    out = args.output
    if out.exists():
        raise FileExistsError("Result directory already exists; do not repeat confirmation")
    out.mkdir(parents=True)
    write_json(
        out / "STARTED.json",
        {
            "plan_sha256": sha256(args.plan),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
    )
    data = Path(plan["data"])
    features = make_features(data, Path(plan["embeddings"]), plan["axes"])
    y = np.load(data / "labels.npy")
    groups = np.load(data / "groups.npy")
    scans = pd.read_csv(data / "scan_table.csv").set_index("scan_id")
    sessions = scans.loc[groups, "session_id"].to_numpy()
    cross_classes = np.array([c for c in range(90) if len(np.unique(sessions[y == c])) > 1])
    rows = []
    per_class = []
    pred_rows: list[dict[str, int | str]] = []
    selections = []
    fold_stats: dict[int, dict[str, Any]] = {}
    for fold in [0, 1]:
        split = {k: np.array(v, dtype=int) for k, v in plan["splits"][str(fold)].items()}
        tr, ca = split["train"], split["calib"]
        te = np.sort(np.r_[split["val"], split["test"]])
        if set(groups[tr]) & set(groups[te]):
            raise ValueError("Group leakage")
        train_session = np.array([np.unique(sessions[tr][y[tr] == c]).item() for c in range(90)])
        probabilities = {}
        calib_probabilities = {}
        times = {}
        for arm in plan["arms"]:
            if arm.startswith(("fusion", "v5")):
                continue
            t = time.monotonic()
            cal, logits = fit_probe(
                features[arm], y, tr, ca, te, export=out / f"probe_f{fold}_{arm}.npz"
            )
            temp = calibrate_temperature(cal, y[ca], plan["temperatures"])
            probabilities[arm] = softmax(logits / temp, axis=1)
            calib_probabilities[arm] = softmax(cal / temp, axis=1)
            times[arm] = time.monotonic() - t
            selections.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "temperature": temp,
                    "calib_f1": f1_score(y[ca], cal.argmax(1), average="macro"),
                    "dimensions": features[arm].shape[1],
                }
            )
            print(f"fold {fold} fitted {arm} ({times[arm]:.1f}s)", flush=True)
        for arm, hsi in [
            ("fusion32_equal", "hsi_q32"),
            ("fusion32_calib", "hsi_q32"),
            ("fusion214_equal", "hsi_q214_own"),
            ("fusion32_shuffled", "hsi_q32"),
        ]:
            alpha = (
                select_fusion(
                    calib_probabilities[hsi],
                    calib_probabilities["dino_rgb"],
                    y[ca],
                    plan["hsi_weights"],
                )
                if arm == "fusion32_calib"
                else 0.5
            )
            rgb = probabilities["dino_rgb"].copy()
            if arm.endswith("shuffled"):
                rng = np.random.default_rng(20261004 + fold)
                for g in np.unique(groups[te]):
                    ids = np.flatnonzero(groups[te] == g)
                    rgb[ids] = rgb[rng.permutation(ids)]
            probabilities[arm] = alpha * probabilities[hsi] + (1 - alpha) * rgb
            times[arm] = 0.0
            selections.append({"fold": fold, "arm": arm, "hsi_weight": alpha})
        np.savez_compressed(out / f"probabilities_f{fold}.npz", indices=te, **probabilities)
        fold_stats[fold] = {}
        for arm in plan["arms"]:
            pred = probabilities[arm].argmax(1)
            stats = class_metrics(y[te], pred, 90)
            fold_stats[fold][arm] = stats
            same = np.setdiff1d(np.arange(90), cross_classes)
            mask = np.isin(y[te], cross_classes)
            rows.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "f1": stats["f1"].mean(),
                    "accuracy": accuracy_score(y[te], pred),
                    "same_recall": stats["recall"][same].mean(),
                    "cross_recall": stats["recall"][cross_classes].mean(),
                    "session_attraction": float(
                        (train_session[pred[mask]] == sessions[te][mask]).mean()
                    ),
                    "fit_seconds": times[arm],
                }
            )
            for c in range(90):
                per_class.append(
                    {
                        "fold": fold,
                        "arm": arm,
                        "label": c,
                        "session": int(sessions[te][y[te] == c][0]),
                        "cross": c in cross_classes,
                        **{k: float(v[c]) for k, v in stats.items()},
                    }
                )
            pred_rows.extend(
                {
                    "fold": fold,
                    "arm": arm,
                    "index": int(i),
                    "target": int(label),
                    "prediction": int(p),
                }
                for i, label, p in zip(te, y[te], pred, strict=True)
            )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv.gz", index=False)
    pd.DataFrame(selections).to_csv(out / "calibration.csv", index=False)
    aggregates = []
    for arm in plan["arms"]:
        f1 = np.stack([fold_stats[f][arm]["f1"] for f in [0, 1]], 1)
        recall = np.stack([fold_stats[f][arm]["recall"] for f in [0, 1]], 1)
        row = {
            "arm": arm,
            "f1": float(f1.mean()),
            "f1_ci": cluster_interval(f1),
            "accuracy": float(np.mean([r["accuracy"] for r in rows if r["arm"] == arm])),
            "f1_fold_min": float(f1.mean(0).min()),
            "f1_fold_max": float(f1.mean(0).max()),
        }
        for name, idx in [
            ("same", np.setdiff1d(np.arange(90), cross_classes)),
            ("cross", cross_classes),
        ]:
            row[name + "_recall"] = float(recall[idx].mean())
            row[name + "_ci"] = cluster_interval(recall[idx])
        aggregates.append(row)
    write_json(out / "summary.json", aggregates)
    deltas = []
    comparisons = [(h, *spec) for h, spec in plan["hypotheses"].items()] + [
        ("diagnostic", a, b, 0.0) for a, b in plan["additional_deltas"]
    ]
    for hypothesis, a, b, threshold in comparisons:
        df = np.stack([fold_stats[f][a]["f1"] - fold_stats[f][b]["f1"] for f in [0, 1]], 1)
        dr = np.stack([fold_stats[f][a]["recall"] - fold_stats[f][b]["recall"] for f in [0, 1]], 1)[
            cross_classes
        ]
        ci = cluster_interval(df)
        passed = float(df.mean()) >= threshold and ci[0] > 0
        if hypothesis.startswith(("H37", "H39")):
            passed = passed and float(dr.mean()) >= 0
        deltas.append(
            {
                "hypothesis": hypothesis,
                "a": a,
                "b": b,
                "delta_f1": float(df.mean()),
                "ci": ci,
                "delta_cross": float(dr.mean()),
                "cross_ci": cluster_interval(dr),
                "threshold": threshold,
                "screen_pass": bool(passed),
            }
        )
    write_json(out / "deltas.json", deltas)
    write_json(
        out / "COMPLETED.json",
        {
            "plan_sha256": sha256(args.plan),
            "arms": len(plan["arms"]),
            "folds": [0, 1],
            "no_gpu_training": True,
            "files": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        },
    )
    print(
        pd.DataFrame(aggregates)[["arm", "f1", "same_recall", "cross_recall"]].to_string(
            index=False
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "run"])
    p.add_argument("--data", type=Path, default=Path("dataset_rgb_hsi_v3"))
    p.add_argument("--embeddings", type=Path, default=Path("outputs/s20_rgb_features_v3"))
    p.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/research/evidence/S21_complementary_rgb/preregistration.json"),
    )
    p.add_argument("--output", type=Path, default=Path("outputs/s21_rgb_study"))
    args = p.parse_args()
    (freeze if args.action == "freeze" else execute)(args)


if __name__ == "__main__":
    main()
