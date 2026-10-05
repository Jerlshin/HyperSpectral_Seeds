#!/usr/bin/env python
"""S28: head-initialization sensitivity of the S27 TTA-trained head (seeds 1/2, both folds).

Pre-declared by S27's pass path. Uses only S27/S24 caches: no encoder or TTA inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.special import softmax

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.residual_head import fit_residual_head
from spectralquadnet.experiments.rgb_probe import (
    aligned_logits,
    class_metrics,
    cluster_interval,
    probe_logits,
    verify_plan,
)
from spectralquadnet.models.multimodal_residual import FrozenResidualFusion

PLAN = Path("configs/research/s28_tta_head_seeds.json")
PARENT = Path("configs/research/s27_tta_trained_head.json")
S27 = Path("outputs/s27_tta_trained_head")
OUT = Path("outputs/s28_tta_head_seeds")


def head_inputs(h: npt.NDArray[Any], z_hsi: npt.NDArray[Any], rgb_rows: npt.NDArray[Any], probe: Path,
                scales: Path, hsi_temp: float, rgb_temp: float) -> list[torch.Tensor]:
    """S27 inputs: train-scaled HSI/RGB features and the equal calibrated TTA anchor."""
    with np.load(scales) as s:
        hs, rs = (h - s["hsi_mean"]) / s["hsi_scale"], (rgb_rows - s["rgb_mean"]) / s["rgb_scale"]
    anchor = (softmax(z_hsi / hsi_temp, axis=1) + softmax(probe_logits(probe, rgb_rows) / rgb_temp, axis=1)) / 2
    if not all(np.isfinite(v).all() for v in (hs, rs, anchor)):
        raise ValueError("Nonfinite head inputs")
    return [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in (hs, rs, anchor)]


def fold_inputs(fold: int, rgb: npt.NDArray[Any], labels: npt.NDArray[Any], split: str) -> tuple[npt.NDArray[Any], list[torch.Tensor]]:
    choice = next(v for v in json.loads(Path("outputs/s23_frozen_multimodal/selection.json").read_text()) if v["fold"] == fold)
    temp = next(v["temperature"] for v in json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text()) if v["fold"] == fold)
    root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
    if split == "heldout":
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz") as c:
            rows, h = c["rows"], c["hsi"]
        z = aligned_logits(root / "logits_val_test_tta.npz", rows, labels)
    else:
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz") as c:
            rows, h = c[split], c[f"hsi_{split}"]
        if split == "train":
            with np.load(f"outputs/s27_tta_cache/tta_train_f{fold}.npz") as t:
                if not np.array_equal(t["rows"], rows):
                    raise ValueError("Train TTA cache rows differ")
                z = t["logits"].astype(np.float64)
        else:
            z = aligned_logits(root / "logits_calib_tta.npz", rows, labels)
    return rows, head_inputs(h, z, rgb[rows], Path(f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz"),
                             Path(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz"), temp, choice["rgb_temperature"])


def completed(path: Path, plan: Path) -> None:
    state = json.loads((path / "COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(plan):
        raise ValueError(f"{path} completion plan differs")
    for name, digest in state["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"{path}/{name} changed")


def freeze() -> None:
    parent = verify_plan(PARENT)
    completed(S27, PARENT)
    if not json.loads((S27 / "hypothesis.json").read_text())["H45_screen"]:
        raise ValueError("S27 failed; its pass path does not apply")
    inputs = dict(parent["input_hashes"])
    paths = [PARENT, Path(__file__), Path("src/spectralquadnet/experiments/residual_head.py"),
             S27 / "COMPLETED.json", S27 / "hypothesis.json", S27 / "per_class.csv", S27 / "predictions.csv.gz"]
    paths += [S27 / f"head_f{f}.pth" for f in (0, 1)]
    inputs.update({str(p): sha256(p) for p in paths})
    spec = {"study": "S28_tta_head_seeds", "frozen_at": "2026-10-05", "parent_plan": str(PARENT),
            "scope": "Head-initialization sensitivity only; fixed S22 seed0 encoders; reused acquisitions.",
            "input_hashes": inputs, "model": parent["model"], "training": parent["training"],
            "cells": [{"fold": f, "head_seed": s, "encoder_seed": 0} for f in (0, 1) for s in (1, 2)],
            "equivalence_check": "Before new fits: refit head seed0 with the shared helper on train/calib only; selected epoch and calib F1 must equal S27 exactly.",
            "replay_check": "After selection: saved S27 seed0 heads and anchors must reproduce S27 held-out predictions exactly.",
            "H46_screen": "Over head seeds 0/1/2 vs equal TTA: mean F1 gain >= .01, paired variety CI lower > 0 (class contributions averaged over seeds), positive fold-mean gains, every seed's two-fold mean gain > 0, mean cross delta >= 0. Report seed SD.",
            "stop": "Exactly four head fits; no encoder fit, sweep or recipe change."}
    for p in (PLAN, Path("docs/research/evidence/S28_tta_head_seeds/preregistration.json")):
        if p.exists():
            raise FileExistsError("S28 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def run() -> None:
    spec = verify_plan(PLAN)
    OUT.mkdir(exist_ok=False)
    write_json(OUT / "STARTED.json", {"plan_sha256": sha256(PLAN)})
    torch.set_num_threads(4)
    start = time.perf_counter()
    labels = np.load("dataset_u430k32/labels.npy")
    rgb = np.load("outputs/s20_rgb_features_v3/rgb.npy")
    cross = pd.read_csv("dataset_u430k32/scan_table.csv").groupby("label").session_id.nunique().sort_index().to_numpy() > 1
    prior = json.loads((S27 / "selection.json").read_text())
    heads: dict[tuple[int, int], FrozenResidualFusion] = {}
    selections, traces, checks = [], [], []
    for fold in (0, 1):
        tr, train = fold_inputs(fold, rgb, labels, "train")
        ca, calib = fold_inputs(fold, rgb, labels, "calib")
        _, sel0, _ = fit_residual_head(train, calib, labels[tr], labels[ca], spec["model"], spec["training"], 0)
        ref = next(v for v in prior if v["fold"] == fold)
        same = sel0["selected_epoch"] == ref["selected_epoch"] and sel0["calib_f1"] == ref["calib_f1"]
        checks.append({"fold": fold, "refit_seed0": sel0, "s27": ref, "identical": same})
        if not same:
            raise ValueError("Shared helper does not reproduce S27 seed0 selection")
        for seed in (1, 2):
            head, sel, trace = fit_residual_head(train, calib, labels[tr], labels[ca], spec["model"], spec["training"], seed)
            heads[(fold, seed)] = head
            selections.append({"fold": fold, **sel})
            traces += [{"fold": fold, "head_seed": seed, **t} for t in trace]
            torch.save({"head": head.state_dict(), "selection": selections[-1], "plan_sha256": sha256(PLAN)},
                       OUT / f"head_f{fold}_s{seed}.pth")
    write_json(OUT / "selection.json", selections)
    write_json(OUT / "equivalence_check.json", checks)
    # Held-out only after all four checkpoints are saved.
    s27pred = pd.read_csv(S27 / "predictions.csv.gz")
    metrics, classes, predictions, replay = [], [], [], []
    for fold in (0, 1):
        te, held = fold_inputs(fold, rgb, labels, "heldout")
        saved = FrozenResidualFusion(**spec["model"])
        saved.load_state_dict(torch.load(S27 / f"head_f{fold}.pth", map_location="cpu", weights_only=True)["head"])
        saved.eval()
        with torch.no_grad():
            z0, _ = saved(*held)
        for arm, pred in (("tta_trained_residual", z0.argmax(1).numpy()), ("equal_tta", held[2].argmax(1).numpy())):
            ref = s27pred[(s27pred.fold == fold) & (s27pred.arm == arm)].set_index("index").loc[te].prediction.to_numpy()
            replay.append({"fold": fold, "arm": arm, "disagreements": int(np.sum(pred != ref))})
            if replay[-1]["disagreements"]:
                raise ValueError("Cached inputs do not reproduce S27")
        for seed in (1, 2):
            with torch.no_grad():
                z, _ = heads[(fold, seed)](*held)
            pred = z.argmax(1).numpy()
            stat = class_metrics(labels[te], pred, 90)
            metrics.append({"fold": fold, "head_seed": seed, "arm": f"tta_trained_residual_s{seed}",
                            "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(pred == labels[te])),
                            "same_recall": float(stat["recall"][~cross].mean()), "cross_recall": float(stat["recall"][cross].mean())})
            classes.extend({"fold": fold, "head_seed": seed, "label": c, "cross": bool(cross[c]),
                            "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])} for c in range(90))
            predictions.extend({"fold": fold, "arm": f"tta_trained_residual_s{seed}", "index": int(i), "target": int(t),
                                "prediction": int(p)} for i, t, p in zip(te, labels[te], pred, strict=True))
    s27c = pd.read_csv(S27 / "per_class.csv")
    base = s27c[s27c.arm == "equal_tta"].set_index(["label", "fold"]).sort_index()
    all_seeds = pd.concat([s27c[s27c.arm == "tta_trained_residual"].assign(head_seed=0), pd.DataFrame(classes)])
    f1d, crd, per_seed = [], [], []
    for seed, block in all_seeds.groupby("head_seed"):
        a = block.set_index(["label", "fold"]).sort_index()
        d = (a.f1 - base.f1).unstack("fold").to_numpy()
        r = (a.recall - base.recall)[a.cross].unstack("fold").to_numpy()
        f1d.append(d)
        crd.append(r)
        per_seed.append({"head_seed": int(seed), "delta_f1": float(d.mean()), "fold_deltas": d.mean(0).tolist(),
                         "delta_cross": float(r.mean())})
    df, dr = np.mean(f1d, axis=0), np.mean(crd, axis=0)
    ci = cluster_interval(df)
    gate = bool(df.mean() >= .01 and ci[0] > 0 and min(df.mean(0)) > 0 and all(v["delta_f1"] > 0 for v in per_seed)
                and dr.mean() >= 0)
    write_json(OUT / "hypothesis.json", {"H46_screen": gate, "delta_f1": float(df.mean()), "f1_ci": ci,
               "fold_mean_deltas": df.mean(0).tolist(), "delta_cross": float(dr.mean()), "cross_ci": cluster_interval(dr),
               "per_seed": per_seed, "seed_sd_delta_f1": float(np.std([v["delta_f1"] for v in per_seed], ddof=1)),
               "replay": replay, "new_head_fits": 4, "new_encoder_fits": 0, "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    pd.DataFrame(traces).to_csv(OUT / "learning_curves.csv", index=False)
    write_json(OUT / "execution.json", {"wall_seconds": time.perf_counter() - start})
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
