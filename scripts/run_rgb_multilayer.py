#!/usr/bin/env python
"""S37: does the trained RGB branch gain from S31's two transferable readout findings?

One arm, ``mlm_vitb``: the S32 ViT-B recipe (foreground tokens, same augmentation, schedule and
selection) with (i) a multi-layer readout ([class, fg-mean] after each of the last 4 blocks,
projected and averaged; F115) and (ii) the 8 metric morphometrics as an explicit input (F117).
Control: S32's ``vitb`` cells (same seed, folds, recipe). Factorial with S33 (blur): frozen before
S33 is scored, so neither design depends on the other's outcome.

Stages: ``freeze`` · ``train --fold f`` · ``run``.
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

from run_rgb_finetune import OUT as S32_OUT
from run_rgb_finetune import PLAN as S32_PLAN
from run_rgb_finetune import RGB, device
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.multilayer_finetune import finetune, morph_features, predict
from spectralquadnet.experiments.rgb_finetune import Recipe, macro_f1
from spectralquadnet.experiments.rgb_probe import aligned_logits, calibrate_temperature, verify_plan
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, gate, score, summarise
from spectralquadnet.models.rgb_branch import load_dinov2
from spectralquadnet.models.rgb_multilayer import MultiLayerRGBBranch

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s37_rgb_multilayer.json")
EVIDENCE = Path("docs/research/evidence/S37_rgb_multilayer")
OUT = Path("outputs/s37_rgb_multilayer")
MORPH = Path("dataset_rgb_hsi_v3/rgb_morphology.npy")
MODEL = {"layers": 4, "width": 768, "morph_dim": 8, "keep": 128}


def freeze() -> None:
    s32 = verify_plan(S32_PLAN)
    if Path("outputs/s33_rgb_acquisition/screen").exists():
        raise ValueError("Freeze S37 before S33 scoring (factorial design)")
    paths = [Path(__file__), S32_PLAN, Path("scripts/run_rgb_finetune.py"), RGB, MORPH,
             Path("src/spectralquadnet/models/rgb_multilayer.py"), Path("src/spectralquadnet/models/rgb_branch.py"),
             Path("src/spectralquadnet/experiments/multilayer_finetune.py"), Path("src/spectralquadnet/experiments/rgb_finetune.py"),
             Path("src/spectralquadnet/experiments/screen_metrics.py"), Path("outputs/s22_fusion_analysis/calibration.json"),
             Path("dataset_u430k32/labels.npy"), Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv"),
             Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
             S32_OUT / "screen" / "COMPLETED.json"]
    paths += [S32_OUT / f"vitb_f{f}" / n for f in (0, 1) for n in ("COMPLETED.json", "outputs.npz", "selection.json")]
    for fold in (0, 1):
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        paths += [root / "logits_calib_tta.npz", root / "logits_val_test_tta.npz"]
    spec = {"study": "S37_rgb_multilayer", "frozen_at": "2026-10-05", "parent_plan": str(S32_PLAN),
            "scope": "Trained-RGB component screen on reused acquisitions; S21 corrected folds; seed 0; ViT-B matched to S32 vitb.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "temperatures": s32["temperatures"],
            "cells": [{"arm": "mlm_vitb", "fold": f, "seed": 0} for f in (0, 1)],
            "model": MODEL, "recipe": s32["recipes"]["vitb"],
            "morphometrics": "8 metric RGB morphometrics; log of area/major/minor/equivalent diameter; z-scored on outer-training rows; not augmented",
            "head_parameters": "per-layer projections, morphometric map and classifier at head_lr; backbone layer decay as S32",
            "selection_within_cell": s32["selection_within_cell"],
            "control": "S32 vitb cells (identical recipe and seed)",
            "H57_screen": "RGB: mlm_vitb_tta minus S32 ft_vitb_tta: mean F1 >= .01, paired variety CI lower > 0, positive on each fold, cross delta >= 0.",
            "H58_descriptive": "System: equal fusion with v5 TTA, same contrast.",
            "decision": "H57 pass: the multi-layer + metric-morphometric readout enters the proposed architecture's RGB branch. "
                        "Fail: the trained single-layer readout already captures it.",
            "stop": "One arm, two cells; no width/layer search; attribution between the two changes is deferred to confirmation ablations."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S37 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def train(fold: int) -> None:
    spec = verify_plan(PLAN)
    cell = OUT / f"mlm_vitb_f{fold}"
    cell.mkdir(parents=True, exist_ok=False)
    r = Recipe(**spec["recipe"])
    cohort = Cohort.load()
    split = cohort.splits[fold]
    images = np.load(RGB)
    morph = morph_features(np.load(MORPH), split["train"])
    dev = device()
    write_json(cell / "STARTED.json", {"plan_sha256": sha256(PLAN), "device": str(dev), "torch": torch.__version__})
    start = time.perf_counter()
    m = spec["model"]
    model = MultiLayerRGBBranch(load_dinov2("dinov2_vitb14", r.drop_path), 90, keep=m["keep"], layers=m["layers"],
                                width=m["width"], morph_dim=m["morph_dim"])

    def write(row: dict[str, Any]) -> None:
        with (cell / "trace.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("mlm_vitb", fold, row, flush=True)

    model, best, _ = finetune(model, images, morph, cohort.y, split["train"], split["calib"], r, 0, dev, write)
    train_seconds = time.perf_counter() - start
    torch.save({"model": model.state_dict(), "best": best, "plan_sha256": sha256(PLAN)}, cell / "model.pth")
    rows = np.arange(len(images))
    logits, embeddings = predict(model, images, morph, rows, dev, r)
    np.savez_compressed(cell / "outputs.npz", rows=rows, logits=logits.astype(np.float32), embeddings=embeddings.astype(np.float32))
    cal = logits[:, split["calib"]].mean(0)
    write_json(cell / "selection.json", {"best": best, "temperature": calibrate_temperature(cal, cohort.y[split["calib"]], spec["temperatures"]),
                                         "calib_f1_tta": macro_f1(cohort.y[split["calib"]], cal),
                                         "calib_f1_id": macro_f1(cohort.y[split["calib"]], logits[0, split["calib"]]),
                                         "train_seconds": train_seconds, "total_seconds": time.perf_counter() - start})
    write_json(cell / "COMPLETED.json", {"plan_sha256": sha256(PLAN), "files": {p.name: sha256(p) for p in cell.iterdir() if p.is_file()}})


def run() -> None:
    spec = verify_plan(PLAN)
    report = OUT / "screen"
    report.mkdir(exist_ok=False)
    cohort = Cohort.load()
    y = cohort.y
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    metrics, classes, predictions = [], [], []
    for fold in (0, 1):
        te = cohort.heldout(fold)
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        hsi = softmax(aligned_logits(Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_val_test_tta.npz"), te, y) / t_hsi, axis=1)
        probs: dict[str, Array] = {"hsi_tta": hsi}
        for name, cell in (("ft_vitb", S32_OUT / f"vitb_f{fold}"), ("mlm_vitb", OUT / f"mlm_vitb_f{fold}")):
            state = json.loads((cell / "COMPLETED.json").read_text())
            if any(sha256(cell / n) != d for n, d in state["files"].items()):
                raise ValueError(f"{cell} changed")
            t = json.loads((cell / "selection.json").read_text())["temperature"]
            with np.load(cell / "outputs.npz") as o:
                z = o["logits"][:, te]
            probs[f"{name}_id"] = softmax(z[0] / t, axis=1)
            probs[f"{name}_tta"] = softmax(z.mean(0) / t, axis=1)
            probs[f"equal_{name}"] = (hsi + probs[f"{name}_tta"]) / 2
        for arm, prob in probs.items():
            m, c, p = score(cohort, fold, arm, te, prob)
            metrics.append(m)
            classes += c
            predictions += p
    per_class = pd.DataFrame(classes)
    h57 = contrast(per_class, "mlm_vitb_tta", "ft_vitb_tta")
    write_json(report / "hypothesis.json", {"H57_screen": gate(h57, .01), "H57": h57,
                                            "H58_descriptive": contrast(per_class, "equal_mlm_vitb", "equal_ft_vitb"),
                                            "single_view": contrast(per_class, "mlm_vitb_id", "ft_vitb_id"),
                                            "network_training_runs": len(spec["cells"]), "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(report / "metrics.csv", index=False)
    per_class.to_csv(report / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(report / "predictions.csv.gz", index=False)
    summarise(pd.DataFrame(metrics), per_class).to_csv(report / "summary.csv")
    write_json(report / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in report.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "train", "run"))
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    a = parser.parse_args()
    if a.action == "train":
        train(a.fold)
    else:
        {"freeze": freeze, "run": run}[a.action]()
