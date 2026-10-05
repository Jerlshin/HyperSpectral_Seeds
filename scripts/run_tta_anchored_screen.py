#!/usr/bin/env python
"""S26 fixed inference intervention: S23 correction with the stronger TTA anchor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.special import softmax

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import (
    aligned_logits,
    class_metrics,
    cluster_interval,
    probe_logits,
    verify_plan,
)
from spectralquadnet.models.multimodal_residual import FrozenResidualFusion

PLAN = Path("configs/research/s26_tta_anchor.json")
PARENT = Path("configs/research/s25_head_seed_screen.json")
OUT = Path("outputs/s26_tta_anchor")


def freeze() -> None:
    parent = verify_plan(PARENT)
    complete = json.loads(Path("outputs/s25_head_seed_screen/COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(PARENT):
        raise ValueError("S25 completion plan differs")
    for name, digest in complete["files"].items():
        if sha256(Path("outputs/s25_head_seed_screen") / name) != digest:
            raise ValueError("S25 completed artifact changed")
    if not json.loads(Path("outputs/s25_head_seed_screen/hypothesis.json").read_text())["H43_screen"]:
        raise ValueError("Selected head did not survive its sensitivity gate")
    inputs = dict(parent["input_hashes"])
    paths = [PARENT, Path(__file__), Path("outputs/s25_head_seed_screen/COMPLETED.json"),
             Path("outputs/s25_head_seed_screen/hypothesis.json"),
             Path("outputs/s22_fusion_analysis/calibration.json"),
             Path("outputs/s22_fusion_analysis/per_class.csv"), Path("outputs/s22_fusion_analysis/COMPLETED.json")]
    for f in (0, 1):
        paths += [Path(f"outputs/s22_complementary_v5/f{f}_s0/results/logits_{split}_tta.npz") for split in ("calib", "val_test")]
        paths += [Path(f"outputs/s25_head_seed_screen/head_f{f}_s{s}.pth") for s in (1, 2)]
    inputs.update({str(p.relative_to(Path.cwd()) if p.is_absolute() else p): sha256(p) for p in paths})
    spec = {"study": "S26_tta_anchor", "frozen_at": "2026-10-05", "parent_plan": str(PARENT),
            "input_hashes": inputs, "scope": "Fixed deployment-recipe development on reused acquisitions; no training",
            "model": parent["model"], "cells": [{"fold": f, "head_seed": 0, "encoder_seed": 0} for f in (0, 1)],
            "intervention": "Use the frozen S23 additive residual, with unchanged single-view HSI/RGB features and weights; replace only the log equal-single anchor by saved calibrated equal-TTA probabilities. No TTA feature averaging or retraining is claimed.",
            "calib_gate": "Evaluate the fixed seed0 intervention on calib first. Require F1 strictly above the saved S23 calib score in BOTH folds before scoring any new held-out predictor. Otherwise reject without held-out scoring.",
            "H44_screen": "After calib gate, seed0 mean F1 gain over S22 equal-TTA >=.01, paired variety CI>0, positive F1 each fold and nonnegative mean cross delta. Positive cross interval additionally required for transfer support.",
            "conditional_head_sensitivity": "Only if H44 passes, evaluate the SAME fixed intervention with already trained head seeds1/2, both folds; no new head fits. No anchor weights or scales are selected on held-out outcomes.",
            "stop": "One fixed inference intervention; no new encoder/head fit or GPU job. Preserve all prior plans/results."}
    for p in (PLAN, Path("docs/research/evidence/S26_tta_anchor/preregistration.json")):
        if p.exists():
            raise FileExistsError("S26 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def run() -> None:
    spec = verify_plan(PLAN)
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    labels = np.load("dataset_u430k32/labels.npy")
    rgb = np.load("outputs/s20_rgb_features_v3/rgb.npy")
    cross = pd.read_csv("dataset_u430k32/scan_table.csv").groupby("label").session_id.nunique().to_numpy() > 1
    choices = json.loads(Path("outputs/s23_frozen_multimodal/selection.json").read_text())
    temperatures = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())

    def inputs(fold: int, split: str) -> tuple[npt.NDArray[Any], list[torch.Tensor]]:
        choice = next(v for v in choices if v["fold"] == fold)
        temp = next(v["temperature"] for v in temperatures if v["fold"] == fold)
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_{'traincal' if split == 'calib' else 'heldout'}.npz") as cache:
            rows = cache["calib"] if split == "calib" else cache["rows"]
            h = cache["hsi_calib"] if split == "calib" else cache["hsi"]
        with np.load(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz") as scale:
            hs = (h-scale["hsi_mean"])/scale["hsi_scale"]
            rs = (rgb[rows]-scale["rgb_mean"])/scale["rgb_scale"]
        logits = aligned_logits(Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_{'calib' if split == 'calib' else 'val_test'}_tta.npz"), rows, labels)
        rz = probe_logits(Path(f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz"), rgb[rows])
        anchor = (softmax(logits/temp, axis=1) + softmax(rz/choice["rgb_temperature"], axis=1))/2
        if not all(np.isfinite(v).all() for v in (hs, rs, anchor)):
            raise ValueError("Nonfinite fixed TTA intervention inputs")
        return rows, [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in (hs, rs, anchor)]

    def head(fold: int, seed: int) -> FrozenResidualFusion:
        model = FrozenResidualFusion(**spec["model"])
        path = f"outputs/s23_frozen_multimodal/head_f{fold}.pth" if seed == 0 else f"outputs/s25_head_seed_screen/head_f{fold}_s{seed}.pth"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["head"])
        model.eval().requires_grad_(False)
        return model

    calibration = []
    for fold in (0, 1):
        rows, values = inputs(fold, "calib")
        with torch.no_grad():
            z, _ = head(fold, 0)(*values)
        score = float(class_metrics(labels[rows], z.argmax(1).numpy(), 90)["f1"].mean())
        prior = next(v["calib_f1"] for v in choices if v["fold"] == fold)
        calibration.append({"fold": fold, "head_seed": 0, "tta_anchor_calib_f1": score,
                            "single_anchor_calib_f1": prior, "passes": score > prior})
    passed = all(v["passes"] for v in calibration)
    write_json(OUT / "calibration_gate.json", {"passes": passed, "folds": calibration})
    if not passed:
        write_json(OUT / "hypothesis.json", {"H44_screen": "not evaluated: calib gate failed",
                   "held_out_rows_scored": 0, "new_encoder_fits": 0, "new_head_fits": 0,
                   "decision": "Reject this fixed anchor substitution; retain prior S23/S25 outcomes."})
    else:
        metrics: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        reference = pd.read_csv("outputs/s22_fusion_analysis/per_class.csv").query("arm == 'v5_rgb_equal'").set_index(["label", "fold"]).sort_index()
        def evaluate(seed: int) -> None:
            for fold in (0, 1):
                rows, values = inputs(fold, "heldout")
                with torch.no_grad():
                    z, _ = head(fold, seed)(*values)
                pred = z.argmax(1).numpy()
                stat = class_metrics(labels[rows], pred, 90)
                metrics.append({"fold": fold, "head_seed": seed, "encoder_seed": 0, "arm": "tta_anchored_residual",
                                "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(pred == labels[rows])),
                                "same_recall": float(stat["recall"][~cross].mean()), "cross_recall": float(stat["recall"][cross].mean())})
                classes.extend({"fold": fold, "head_seed": seed, "label": c, "cross": bool(cross[c]),
                                "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])} for c in range(90))
                predictions.extend({"fold": fold, "head_seed": seed, "encoder_seed": 0, "index": int(i), "target": int(t), "prediction": int(p)}
                                   for i, t, p in zip(rows, labels[rows], pred, strict=True))
        evaluate(0)
        block = pd.DataFrame(classes).set_index(["label", "fold"]).sort_index()
        df = (block.f1-reference.f1).unstack("fold").to_numpy()
        dr = (block.recall-reference.recall)[block.cross].unstack("fold").to_numpy()
        ci, cr = cluster_interval(df), cluster_interval(dr)
        gate = bool(df.mean() >= .01 and ci[0] > 0 and min(df.mean(0)) > 0 and dr.mean() >= 0)
        if gate:
            for seed in (1, 2):
                evaluate(seed)
        write_json(OUT / "hypothesis.json", {"H44_screen": gate, "delta_f1_tta_seed0": float(df.mean()),
                   "f1_ci": ci, "fold_f1_deltas": df.mean(0).tolist(), "delta_cross_tta_seed0": float(dr.mean()),
                   "cross_ci": cr, "transfer_gain_supported": bool(cr[0] > 0),
                   "head_seeds_evaluated": [0, 1, 2] if gate else [0], "new_encoder_fits": 0, "new_head_fits": 0,
                   "scope": spec["scope"]})
        pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
        pd.DataFrame(classes).to_csv(OUT / "per_class.csv", index=False)
        pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
