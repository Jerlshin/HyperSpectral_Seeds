#!/usr/bin/env python
"""S33: does a measured-nuisance RGB branch transfer away from session 8?

Session 8's RGB acquisition loses fine texture (σ ≈ 0.5 px Gaussian equivalent on the crops) and
clips red far less (S31 §3). Training scans contain no within-variety acquisition contrast, so the
nuisance family is injected from measurement, never estimated from class-confounded means:

* ``vitb_acq`` - the S32 ViT-B recipe plus Gaussian blur σ ~ U(0, 1.0) px on half the training crops;
* test-time rendering - every test crop is blurred at the measured σ = 0.5 before the 4 views, so all
  classes are compared in the softer regime (one-way: a soft image cannot be sharpened).

Stages: ``freeze`` (plan; training then runs via ``run_rgb_finetune.py train --plan``), ``render``
(rendered 4-view logits for the S32 and S33 ViT-B models, calib + held-out rows), ``run`` (one scoring).
"""
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

from run_rgb_finetune import OUT as S32_OUT
from run_rgb_finetune import PLAN as S32_PLAN
from run_rgb_finetune import RGB, device
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_finetune import Recipe, macro_f1, predict
from spectralquadnet.experiments.rgb_probe import aligned_logits, calibrate_temperature, verify_plan
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, score, summarise
from spectralquadnet.models.rgb_branch import RGBBranch, load_dinov2

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s33_rgb_acquisition.json")
EVIDENCE = Path("docs/research/evidence/S33_rgb_acquisition")
OUT = Path("outputs/s33_rgb_acquisition")
SIGMA = 0.5  # measured session-8 equivalent on the crops (S31 §3)
MODELS = {"ft_vitb": (S32_OUT, "vitb"), "acq_vitb": (OUT, "vitb_acq")}


def freeze() -> None:
    s32 = verify_plan(S32_PLAN)
    base = s32["recipes"]["vitb"]
    acq = Recipe(**{**base, "blur_sigma": 1.0, "blur_p": 0.5})
    paths = [Path(__file__), S32_PLAN, Path("scripts/run_rgb_finetune.py"), RGB,
             Path("src/spectralquadnet/experiments/rgb_finetune.py"), Path("src/spectralquadnet/models/rgb_branch.py"),
             Path("src/spectralquadnet/experiments/screen_metrics.py"),
             Path("docs/research/evidence/S31_rgb_readout_audit/acquisition_audit.csv"),
             Path("dataset_u430k32/labels.npy"), Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv"),
             Path("outputs/s22_fusion_analysis/calibration.json"), Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"]
    paths += [S32_OUT / f"vitb_f{f}" / name for f in (0, 1) for name in ("COMPLETED.json", "model.pth", "selection.json")]
    for fold in (0, 1):
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        paths += [root / "logits_calib_tta.npz", root / "logits_val_test_tta.npz"]
    spec = {"study": "S33_rgb_acquisition", "frozen_at": "2026-10-05", "parent_plan": str(S32_PLAN),
            "scope": "Mechanism screen on reused acquisitions; S21 corrected folds; seed 0; ViT-B matched to S32's ft_vitb. "
                     "The nuisance family is measured label-free (S31 §3); no class-session mean is used.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "temperatures": s32["temperatures"],
            "cells": [{"arm": "vitb_acq", "backbone": "vitb", "fold": f, "seed": 0} for f in (0, 1)],
            "recipes": {"vitb_acq": acq.as_dict()},
            "selection_within_cell": s32["selection_within_cell"],
            "render_sigma": SIGMA,
            "arms": {"rgb": ["ft_vitb_tta (S32)", "ft_vitb_render", "acq_vitb_tta", "acq_vitb_render"],
                     "fusion": ["equal_<rgb arm> with the saved S22 v5 TTA"]},
            "temperature_rule": "Each RGB arm: calib log-loss temperature of its own 4-view mean logits (rendered arms use rendered calib logits).",
            "primary": "acq_vitb_render minus ft_vitb_tta (pre-declared; calib cannot measure transfer, so no calib choice among variants).",
            "H52_screen": "RGB transfer: mean bridge recall in the from_session8 direction (17 class cells) improves by >= .03 with "
                          "a cell-bootstrap CI lower > 0, AND mean F1 delta >= -.01.",
            "H53_screen": "System: equal_acq_vitb_render minus equal_ft_vitb: from_session8 recall delta >= .03 with CI lower > 0, "
                          "AND mean F1 delta >= -.005.",
            "descriptive": ["training augmentation alone (acq_vitb_tta - ft_vitb_tta)", "rendering alone (ft_vitb_render - ft_vitb_tta)",
                            "to_session8 direction; same-session recall; session attraction"],
            "decision": "H52 pass: the measured-nuisance treatment becomes part of the RGB branch of the next architecture, and its "
                        "replication joins the final confirmation. H52 fail: away-from-session-8 transfer is not limited by this "
                        "nuisance at the RGB branch; the treatment is not adopted.",
            "stop": "One augmentation range, one rendering σ, two cells; no σ search."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S33 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def render() -> None:
    """Rendered 4-view logits for both ViT-B models on calib and held-out rows (no fitting)."""
    spec = verify_plan(PLAN)
    cohort, dev = Cohort.load(), device()
    images = np.load(RGB)
    target = OUT / "rendered"
    target.mkdir(parents=True, exist_ok=False)
    for name, (root, arm) in MODELS.items():
        for fold in (0, 1):
            cell = root / f"{arm}_f{fold}"
            if json.loads((cell / "COMPLETED.json").read_text())["files"]["model.pth"] != sha256(cell / "model.pth"):
                raise ValueError(f"{cell} checkpoint changed")
            recipe = Recipe(**(spec["recipes"][arm] if arm in spec["recipes"] else json.loads(S32_PLAN.read_text())["recipes"][arm]))
            model = RGBBranch(load_dinov2("dinov2_vitb14"), 90, recipe.keep_tokens, train_blocks=recipe.train_blocks)
            model.load_state_dict(torch.load(cell / "model.pth", map_location="cpu", weights_only=True)["model"])
            model.to(dev)
            rows = np.sort(np.r_[cohort.splits[fold]["calib"], cohort.heldout(fold)])
            logits, _ = predict(model, images, rows, dev, recipe, sigma=spec["render_sigma"])
            np.savez_compressed(target / f"{name}_f{fold}.npz", rows=rows, logits=logits.astype(np.float32))
            print(name, fold, "rendered", flush=True)
    write_json(target / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in target.iterdir() if p.is_file()}})


def cell_bootstrap(values: Array, n_boot: int = 2000, seed: int = 20261005) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(n_boot, len(values)))].mean(1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def direction_delta(per_class: pd.DataFrame, a: str, b: str, direction: str) -> dict[str, Any]:
    bridge = per_class[per_class.cross].assign(to8=lambda d: d.destination == 8)
    sel = bridge[bridge.to8 == (direction == "to_session8")].set_index(["label", "fold"])
    delta = (sel[sel.arm == a].recall - sel[sel.arm == b].recall).to_numpy()
    return {"cells": len(delta), "delta": float(delta.mean()), "ci": cell_bootstrap(delta)}


def run() -> None:
    spec = verify_plan(PLAN)
    report = OUT / "screen"
    report.mkdir(exist_ok=False)
    cohort = Cohort.load()
    y = cohort.y
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    metrics, classes, predictions, temperatures = [], [], [], []
    for fold in (0, 1):
        te, ca = cohort.heldout(fold), cohort.splits[fold]["calib"]
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        hsi = softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)
        probs: dict[str, Array] = {"hsi_tta": hsi}
        for name, (src, arm) in MODELS.items():
            with np.load(src / f"{arm}_f{fold}" / "outputs.npz") as o:
                plain = {"calib": o["logits"][:, ca].mean(0), "te": o["logits"][:, te].mean(0)}
            with np.load(OUT / "rendered" / f"{name}_f{fold}.npz") as r:
                pos = {int(v): i for i, v in enumerate(r["rows"])}
                logits = r["logits"].mean(0)
                rendered = {"calib": logits[[pos[int(i)] for i in ca]], "te": logits[[pos[int(i)] for i in te]]}
            for variant, z in (("tta", plain), ("render", rendered)):
                t = calibrate_temperature(z["calib"], y[ca], spec["temperatures"])
                temperatures.append({"fold": fold, "arm": f"{name}_{variant}", "temperature": t,
                                     "calib_f1": macro_f1(y[ca], z["calib"])})
                probs[f"{name}_{variant}"] = softmax(z["te"] / t, axis=1)
                probs[f"equal_{name}_{variant}"] = (hsi + probs[f"{name}_{variant}"]) / 2
        for arm, prob in probs.items():
            m, c, p = score(cohort, fold, arm, te, prob)
            metrics.append(m)
            classes += c
            predictions += p
    per_class = pd.DataFrame(classes)
    out: dict[str, Any] = {}
    for label, a, b in (("rgb", "acq_vitb_render", "ft_vitb_tta"), ("system", "equal_acq_vitb_render", "equal_ft_vitb_tta"),
                        ("augmentation_only", "acq_vitb_tta", "ft_vitb_tta"), ("render_only", "ft_vitb_render", "ft_vitb_tta"),
                        ("system_augmentation_only", "equal_acq_vitb_tta", "equal_ft_vitb_tta")):
        out[label] = {**contrast(per_class, a, b), "from_session8": direction_delta(per_class, a, b, "from_session8"),
                      "to_session8": direction_delta(per_class, a, b, "to_session8")}
    rgb, system = out["rgb"], out["system"]
    h52 = bool(rgb["from_session8"]["delta"] >= .03 and rgb["from_session8"]["ci"][0] > 0 and rgb["delta_f1"] >= -.01)
    h53 = bool(system["from_session8"]["delta"] >= .03 and system["from_session8"]["ci"][0] > 0 and system["delta_f1"] >= -.005)
    write_json(report / "hypothesis.json", {"H52_screen": h52, "H53_screen": h53, "contrasts": out,
                                            "network_training_runs": len(spec["cells"]), "scope": spec["scope"]})
    pd.DataFrame(temperatures).to_csv(report / "calibration.csv", index=False)
    pd.DataFrame(metrics).to_csv(report / "metrics.csv", index=False)
    per_class.to_csv(report / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(report / "predictions.csv.gz", index=False)
    summarise(pd.DataFrame(metrics), per_class).to_csv(report / "summary.csv")
    write_json(report / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in report.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "render", "run"))
    {"freeze": freeze, "render": render, "run": run}[parser.parse_args().action]()
