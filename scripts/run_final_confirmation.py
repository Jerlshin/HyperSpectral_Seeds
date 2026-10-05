#!/usr/bin/env python
"""S39: matched final confirmation of SeedNet-MX (S36) over seeds 0/1/2 x both corrected folds.

Replaces the S30 brief (D55). Seed-0 cells are reused (S22 v5; S32 or S37 RGB). New fits:
* RGB, local MPS: the S36 RGB branch at seeds 1 and 2 on folds 0 and 1. The branch is resolved by a
  rule fixed before S37 was scored: S37's multi-layer + morphometric arm if H57-screen passed,
  else S32's ViT-B arm.
* HSI, GPU: v5 seeds 1 and 2 on folds 0 and 1 under the sealed S22 amendment06
  (``configs/research/s39_hsi_seeds_amendment06.json``); needs owner authorization.

Contrasts (frozen here, before any new fit exists), each a mean over 3 seeds x 2 folds; the paired
variety interval is computed on seed-averaged per-class scores; G3 needs Δ > 2 x the seed SD of the
fold-averaged delta:
  M1  MX − equal fusion of v5 with the frozen ViT-L probe (matched HSI seed)   ≥ .02
  M2  MX − HSI v5 alone                                                       ≥ .05
  M3  trained RGB − frozen ViT-L probe (RGB alone)                            ≥ .05
  M4  MX − trained RGB alone (multimodal value)                               ≥ .01
Each also needs CI > 0, a positive mean on each fold, and cross Δ ≥ 0; transfer is reported per
direction. Stages: ``freeze`` · ``train-rgb --seed s --fold f`` · ``run [--partial]``.
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

from run_rgb_finetune import RGB, device
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments import multilayer_finetune, rgb_finetune
from spectralquadnet.experiments.rgb_probe import aligned_logits, calibrate_temperature, verify_plan
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, score, summarise
from spectralquadnet.models.rgb_branch import RGBBranch, load_dinov2
from spectralquadnet.models.rgb_multilayer import MultiLayerRGBBranch

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s39_final_confirmation.json")
EVIDENCE = Path("docs/research/evidence/S39_final_confirmation")
OUT = Path("outputs/s39_final_confirmation")
S32_PLAN, S37_PLAN = Path("configs/research/s32_rgb_finetune.json"), Path("configs/research/s37_rgb_multilayer.json")
HSI_PLAN = Path("configs/research/s39_hsi_seeds_amendment06.json")
S37_HYP = Path("outputs/s37_rgb_multilayer/screen/hypothesis.json")
MORPH = Path("dataset_rgb_hsi_v3/rgb_morphology.npy")
THRESHOLDS = {"M1": .02, "M2": .05, "M3": .05, "M4": .01}


def freeze() -> None:
    for plan in (S32_PLAN, S37_PLAN, HSI_PLAN):
        verify_plan(plan)
    if S37_HYP.exists():
        raise ValueError("Freeze S39 before S37 is scored so the RGB-branch rule cannot depend on its outcome")
    paths = [Path(__file__), S32_PLAN, S37_PLAN, HSI_PLAN, Path("scripts/run_rgb_finetune.py"), Path("scripts/run_rgb_multilayer.py"),
             Path("src/spectralquadnet/experiments/rgb_finetune.py"), Path("src/spectralquadnet/experiments/multilayer_finetune.py"),
             Path("src/spectralquadnet/models/rgb_branch.py"), Path("src/spectralquadnet/models/rgb_multilayer.py"),
             Path("src/spectralquadnet/experiments/screen_metrics.py"), RGB, MORPH,
             Path("outputs/s22_fusion_analysis/calibration.json"), Path("outputs/s31_rgb_readout_audit/probabilities_f0.npz"),
             Path("outputs/s31_rgb_readout_audit/probabilities_f1.npz")]
    spec = {"study": "S39_final_confirmation", "frozen_at": "2026-10-05",
            "scope": "Matched confirmation on reused acquisitions (S21 corrected folds); seeds 0/1/2 for both trained encoders.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "temperatures": json.loads(S32_PLAN.read_text())["temperatures"],
            "rgb_branch_rule": "S37 H57-screen pass -> S37 mlm_vitb (model + recipe of its plan); else S32 vitb. Resolved from S37's sealed hypothesis.json.",
            "rgb_cells": [{"seed": s, "fold": f} for s in (1, 2) for f in (0, 1)],
            "hsi_cells": "S22 amendment06 (seeds 1/2 x folds 0/1), GPU; seed-0 cells are the S22 fits",
            "hsi_temperature": "per seed and fold: calib log-loss of the v5 calib TTA logits on the S21 grid (seed 0 reproduces S22's 1.0)",
            "frozen_rgb_probe": "S29 dino_l probe (deterministic; no seeds): S31 'cls' held-out probabilities, which replay S29 exactly",
            "contrasts": {k: f"threshold {v}" for k, v in THRESHOLDS.items()},
            "gate": "mean delta >= threshold, paired variety CI lower > 0 (seed-averaged per-class), positive mean each fold, cross delta >= 0, "
                    "and delta > 2 x seed SD of the fold-averaged delta (G3). M3 may be scored before the HSI seeds exist (--partial).",
            "decision": "All of M1, M2, M4 pass: SeedNet-MX is the paper's system on these acquisitions. Any fail: report it and keep the "
                        "simplest passing system. Transfer claims additionally need away-from-session-8 gains; broad claims need FW-42.",
            "stop": "Exactly 4 RGB + 4 HSI new fits; no further seeds, recipes or components."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S39 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def rgb_arm() -> str:
    hyp = json.loads(S37_HYP.read_text())
    state = json.loads((S37_HYP.parent / "COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(S37_PLAN) or sha256(S37_HYP) != state["files"]["hypothesis.json"]:
        raise ValueError("S37 is not sealed")
    return "mlm_vitb" if hyp["H57_screen"] else "vitb"


def seed0_cell(arm: str, fold: int) -> Path:
    return Path(f"outputs/s37_rgb_multilayer/mlm_vitb_f{fold}") if arm == "mlm_vitb" else Path(f"outputs/s32_rgb_finetune/vitb_f{fold}")


def train_rgb(seed: int, fold: int) -> None:
    spec = verify_plan(PLAN)
    if {"seed": seed, "fold": fold} not in spec["rgb_cells"]:
        raise ValueError("Cell not authorized by the frozen plan")
    arm = rgb_arm()
    cell = OUT / f"rgb_{arm}_s{seed}_f{fold}"
    cell.mkdir(parents=True, exist_ok=False)
    cohort = Cohort.load()
    split = cohort.splits[fold]
    images = np.load(RGB)
    dev = device()
    write_json(cell / "STARTED.json", {"plan_sha256": sha256(PLAN), "arm": arm, "device": str(dev), "torch": torch.__version__})
    start = time.perf_counter()

    def log(row: dict[str, Any]) -> None:
        with (cell / "trace.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(arm, seed, fold, row, flush=True)

    rows = np.arange(len(images))
    if arm == "mlm_vitb":
        s37 = json.loads(S37_PLAN.read_text())
        r, m = rgb_finetune.Recipe(**s37["recipe"]), s37["model"]
        morph = multilayer_finetune.morph_features(np.load(MORPH), split["train"])
        model: Any = MultiLayerRGBBranch(load_dinov2("dinov2_vitb14", r.drop_path), 90, keep=m["keep"], layers=m["layers"],
                                         width=m["width"], morph_dim=m["morph_dim"])
        model, best, _ = multilayer_finetune.finetune(model, images, morph, cohort.y, split["train"], split["calib"], r, seed, dev, log)
        logits, emb = multilayer_finetune.predict(model, images, morph, rows, dev, r)
    else:
        r = rgb_finetune.Recipe(**json.loads(S32_PLAN.read_text())["recipes"]["vitb"])
        model = RGBBranch(load_dinov2("dinov2_vitb14", r.drop_path), 90, r.keep_tokens, train_blocks=r.train_blocks)
        model, best, _ = rgb_finetune.finetune(model, images, cohort.y, split["train"], split["calib"], r, seed, dev, log)
        logits, emb = rgb_finetune.predict(model, images, rows, dev, r)
    torch.save({"model": model.state_dict(), "best": best, "plan_sha256": sha256(PLAN)}, cell / "model.pth")
    np.savez_compressed(cell / "outputs.npz", rows=rows, logits=logits.astype(np.float32), embeddings=emb.astype(np.float32))
    cal = logits[:, split["calib"]].mean(0)
    write_json(cell / "selection.json", {"best": best, "temperature": calibrate_temperature(cal, cohort.y[split["calib"]], spec["temperatures"]),
                                         "total_seconds": time.perf_counter() - start})
    write_json(cell / "COMPLETED.json", {"plan_sha256": sha256(PLAN), "files": {p.name: sha256(p) for p in cell.iterdir() if p.is_file()}})


def sealed(cell: Path) -> Path:
    state = json.loads((cell / "COMPLETED.json").read_text())
    if any(sha256(cell / n) != d for n, d in state["files"].items()):
        raise ValueError(f"{cell} changed")
    return cell


def seeded_contrast(per_class: pd.DataFrame, a: str, b: str, seeds: list[int]) -> dict[str, Any]:
    """Contrast on seed-averaged per-class scores, plus the seed SD of the fold-averaged delta."""
    mean = per_class.groupby(["arm", "label", "fold", "cross", "destination"], as_index=False)[["f1", "recall"]].mean()
    out = contrast(mean, a, b)
    per_seed = [contrast(per_class[per_class.seed == s], a, b)["delta_f1"] for s in seeds]
    out["seed_deltas"] = per_seed
    out["seed_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")
    return out


def run(partial: bool) -> None:
    spec = verify_plan(PLAN)
    arm = rgb_arm()
    seeds = [0, 1, 2]
    cohort = Cohort.load()
    y = cohort.y
    temps22 = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    metrics, classes = [], []
    for seed in seeds:
        for fold in (0, 1):
            te, ca = cohort.heldout(fold), cohort.splits[fold]["calib"]
            cell = seed0_cell(arm, fold) if seed == 0 else OUT / f"rgb_{arm}_s{seed}_f{fold}"
            t = json.loads((sealed(cell) / "selection.json").read_text())["temperature"]
            with np.load(cell / "outputs.npz") as o:
                rgb = softmax(o["logits"].mean(0)[te] / t, axis=1)
            with np.load(f"outputs/s31_rgb_readout_audit/probabilities_f{fold}.npz") as d:
                if not np.array_equal(d["rows"], te):
                    raise ValueError("S31 held-out rows differ")
                frozen = d["cls"].astype(np.float64)  # = the S29 frozen ViT-L probe (replayed exactly in S31)
            probs: dict[str, Array] = {"rgb_trained": rgb, "rgb_frozen": frozen}
            root = Path(f"outputs/s22_complementary_v5/f{fold}_s{seed}/results")
            if not partial:
                t_hsi = (next(v["temperature"] for v in temps22 if v["fold"] == fold) if seed == 0 else
                         calibrate_temperature(aligned_logits(root / "logits_calib_tta.npz", ca, y), y[ca], spec["temperatures"]))
                hsi = softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)
                probs.update({"hsi_v5": hsi, "mx": (hsi + rgb) / 2, "equal_frozen": (hsi + frozen) / 2})
            for name, prob in probs.items():
                m, c, _ = score(cohort, fold, name, te, prob)
                metrics.append({**m, "seed": seed})
                classes += [{**r, "seed": seed} for r in c]
    per_class = pd.DataFrame(classes)
    pairs = {"M3": ("rgb_trained", "rgb_frozen")}
    if not partial:
        pairs.update({"M1": ("mx", "equal_frozen"), "M2": ("mx", "hsi_v5"), "M4": ("mx", "rgb_trained")})
    result = {}
    for key, (a, b) in pairs.items():
        res = seeded_contrast(per_class, a, b, seeds)
        res["pass"] = bool(res["delta_f1"] >= THRESHOLDS[key] and res["f1_ci"][0] > 0 and min(res["fold_f1_deltas"]) > 0
                           and res["delta_cross"] >= 0 and res["delta_f1"] > 2 * res["seed_sd"])
        result[key] = res
    report = OUT / ("partial_rgb" if partial else "screen")
    report.mkdir(parents=True, exist_ok=False)
    write_json(report / "hypothesis.json", {"rgb_arm": arm, "partial": partial, "contrasts": result, "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(report / "metrics.csv", index=False)
    per_class.to_csv(report / "per_class.csv", index=False)
    summarise(pd.DataFrame(metrics), per_class.groupby(["arm", "label", "fold", "cross", "destination"], as_index=False)[["f1", "recall"]].mean()
              ).to_csv(report / "summary.csv")
    write_json(report / "COMPLETED.json", {"plan_sha256": sha256(PLAN), "files": {p.name: sha256(p) for p in report.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "train-rgb", "run"))
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--partial", action="store_true", help="score M3 (RGB only) before the GPU HSI seeds exist")
    a = parser.parse_args()
    if a.action == "freeze":
        freeze()
    elif a.action == "train-rgb":
        train_rgb(a.seed, a.fold)
    else:
        run(a.partial)
