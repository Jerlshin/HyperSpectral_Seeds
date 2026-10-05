#!/usr/bin/env python
"""S35: class-conditional acquisition rendering (CCAR) for the trained RGB branch.

Session attraction (F113) means a test kernel is pulled toward classes trained under its own
acquisition. S31 measured the RGB part of that acquisition difference (session-8 crops are softer,
σ ≈ 0.5 px) and found two label-free regimes from plate background and sharpness (``sharp``/``soft``,
acquisition_regimes.csv). Blur is one-way: a sharp image can be rendered soft, never the reverse.

CCAR scores class c on the test kernel *as it would look under c's training regime*:
logit_c(x) = z_c(render(x)) if regime(x) = sharp and regime(c) = soft, else z_c(x),
where regime(c) is the majority regime of c's outer-training scans and regime(x) is the regime of
the test kernel's own scan (both measured from images, never from labels or session ids). It uses
the identity and σ = 0.5 rendered 4-view logits that S33 exports; no network is trained or refit.

Frozen before S33 is scored. Stages: ``freeze``, ``run``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import softmax

from run_rgb_acquisition import MODELS, direction_delta
from run_rgb_acquisition import OUT as S33_OUT
from run_rgb_acquisition import PLAN as S33_PLAN
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_finetune import macro_f1
from spectralquadnet.experiments.rgb_probe import aligned_logits, calibrate_temperature, verify_plan
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, score, summarise

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s35_regime_rendering.json")
EVIDENCE = Path("docs/research/evidence/S35_regime_rendering")
OUT = Path("outputs/s35_regime_rendering")
REGIMES = Path("docs/research/evidence/S31_rgb_readout_audit/acquisition_regimes.csv")


def freeze() -> None:
    verify_plan(S33_PLAN)
    if (S33_OUT / "screen").exists():
        raise ValueError("Freeze S35 before S33 scoring so its design cannot depend on S33's outcome")
    paths = [Path(__file__), S33_PLAN, Path("scripts/run_rgb_acquisition.py"), REGIMES,
             Path("docs/research/evidence/S31_rgb_readout_audit/code/acquisition_regimes.py"),
             Path("src/spectralquadnet/experiments/screen_metrics.py"), Path("outputs/s22_fusion_analysis/calibration.json"),
             Path("dataset_u430k32/labels.npy"), Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv")]
    spec = {"study": "S35_regime_rendering", "frozen_at": "2026-10-05", "parent_plan": str(S33_PLAN),
            "scope": "CPU mechanism screen on reused acquisitions; S21 corrected folds; S32/S33 seed-0 ViT-B branches; S22 seed-0 v5 TTA.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "temperatures": json.loads(S33_PLAN.read_text())["temperatures"],
            "rule": "logit_c(x) = rendered (σ = 0.5, S33) 4-view mean logit if regime(scan(x)) = sharp and regime(c) = soft, else "
                    "the identity 4-view mean logit. regime(c) = majority regime over c's outer-training rows (ties -> sharp). "
                    "Temperature: calib log-loss of the CCAR logits on the S21 grid.",
            "arms": {"rgb": ["ccar_ft_vitb", "ccar_acq_vitb", "ft_vitb_tta", "acq_vitb_tta"],
                     "fusion": ["equal_<rgb arm> with the saved S22 v5 TTA"]},
            "primary": "ccar_acq_vitb minus acq_vitb_tta (the measured-nuisance branch with and without class-conditional rendering).",
            "H55_screen": "RGB: from_session8 bridge recall delta >= .03 with cell-bootstrap CI lower > 0, AND mean F1 delta >= -.005.",
            "H56_screen": "System (equal fusion with v5 TTA): from_session8 delta >= .03 with CI lower > 0, AND mean F1 delta >= -.005.",
            "descriptive": ["ccar_ft_vitb - ft_vitb_tta (rendering without augmentation)", "session attraction; to_session8; same recall",
                            "share of test kernels x classes that are re-rendered"],
            "decision": "H55 pass: CCAR enters the proposed architecture's RGB scoring rule. Fail: class-conditional rendering does "
                        "not reach the away-from-session-8 bottleneck; acquisition-limited transfer stands (F113).",
            "stop": "One rule, one σ, two branches; no σ or regime-threshold search."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S35 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def class_regimes(cohort: Cohort, fold: int, scan_soft: Array, groups: Array) -> Array:
    """Boolean per class: majority of its outer-training rows come from soft-regime scans (ties sharp)."""
    tr = cohort.train_rows(fold)
    soft = scan_soft[groups[tr]]
    return np.array([soft[cohort.y[tr] == c].mean() > .5 for c in range(90)])


def ccar(plain: Array, rendered: Array, row_soft: Array, class_soft: Array) -> tuple[Array, float]:
    """Mix identity and rendered logits per (row, class); return logits and the re-rendered share."""
    use = (~row_soft)[:, None] & class_soft[None, :]
    return np.where(use, rendered, plain), float(use.mean())


def run() -> None:
    spec = verify_plan(PLAN)
    state = json.loads((S33_OUT / "rendered" / "COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(S33_PLAN):
        raise ValueError("S33 renders were produced under another plan")
    OUT.mkdir(exist_ok=False)
    cohort = Cohort.load()
    y = cohort.y
    groups = np.load("dataset_u430k32/groups.npy")
    regimes = pd.read_csv(REGIMES).set_index("scan_id").sort_index()
    scan_soft = (regimes.regime == "soft").to_numpy()
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    metrics, classes, predictions, notes = [], [], [], []
    for fold in (0, 1):
        te, ca = cohort.heldout(fold), cohort.splits[fold]["calib"]
        class_soft = class_regimes(cohort, fold, scan_soft, groups)
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        hsi = softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)
        probs: dict[str, Array] = {"hsi_tta": hsi}
        for name, (src, arm) in MODELS.items():
            with np.load(src / f"{arm}_f{fold}" / "outputs.npz") as o:
                plain = {"calib": o["logits"][:, ca].mean(0), "te": o["logits"][:, te].mean(0)}
            with np.load(S33_OUT / "rendered" / f"{name}_f{fold}.npz") as r:
                pos = {int(v): i for i, v in enumerate(r["rows"])}
                z = r["logits"].mean(0)
                rendered = {"calib": z[[pos[int(i)] for i in ca]], "te": z[[pos[int(i)] for i in te]]}
            for variant in ("tta", "ccar"):
                if variant == "tta":
                    logits = plain
                    share = 0.0
                else:
                    cal, _ = ccar(plain["calib"], rendered["calib"], scan_soft[groups[ca]], class_soft)
                    held, share = ccar(plain["te"], rendered["te"], scan_soft[groups[te]], class_soft)
                    logits = {"calib": cal, "te": held}
                t = calibrate_temperature(logits["calib"], y[ca], spec["temperatures"])
                arm_name = f"{name}_tta" if variant == "tta" else f"ccar_{name}"
                notes.append({"fold": fold, "arm": arm_name, "temperature": t, "calib_f1": macro_f1(y[ca], logits["calib"]),
                              "held_out_rerendered_share": share, "soft_classes": int(class_soft.sum())})
                probs[arm_name] = softmax(logits["te"] / t, axis=1)
                probs["equal_" + arm_name] = (hsi + probs[arm_name]) / 2
        for arm, prob in probs.items():
            m, c, p = score(cohort, fold, arm, te, prob)
            metrics.append(m)
            classes += c
            predictions += p
    per_class = pd.DataFrame(classes)
    out: dict[str, Any] = {}
    for label, a, b in (("rgb", "ccar_acq_vitb", "acq_vitb_tta"), ("system", "equal_ccar_acq_vitb", "equal_acq_vitb_tta"),
                        ("rgb_without_augmentation", "ccar_ft_vitb", "ft_vitb_tta"),
                        ("system_without_augmentation", "equal_ccar_ft_vitb", "equal_ft_vitb_tta")):
        out[label] = {**contrast(per_class, a, b), "from_session8": direction_delta(per_class, a, b, "from_session8"),
                      "to_session8": direction_delta(per_class, a, b, "to_session8")}
    rgb, system = out["rgb"], out["system"]
    write_json(OUT / "hypothesis.json", {
        "H55_screen": bool(rgb["from_session8"]["delta"] >= .03 and rgb["from_session8"]["ci"][0] > 0 and rgb["delta_f1"] >= -.005),
        "H56_screen": bool(system["from_session8"]["delta"] >= .03 and system["from_session8"]["ci"][0] > 0
                           and system["delta_f1"] >= -.005),
        "contrasts": out, "network_training_runs": 0, "scope": spec["scope"]})
    pd.DataFrame(notes).to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    per_class.to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    summarise(pd.DataFrame(metrics), per_class).to_csv(OUT / "summary.csv")
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
