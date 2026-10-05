#!/usr/bin/env python
"""S38: with a strong trained RGB branch, which HSI information should the HSI branch contribute?

Evidence behind the question:
* F123/S12: the v5 HSI representation encodes acquisition (session) more than trained RGB; v5's
  spatial pathway is the session channel (S12), and within-kernel spread statistics carry the
  session (F67).
* RGB now supplies spatial/texture information strongly (F118).
* S21: spectral *shape* (SNV mean spectrum) transfers better per unit of F1 than quantile/spread
  features (snvmean32 cross .186; q32 cross .078).

Role-specialization hypothesis: let each modality contribute the information whose transfer is
supported. RGB contributes trained object-centric appearance; HSI contributes calibrated spectral
shape only. Components are saved, already-calibrated predictions: S32 trained ViT-B (4-view, calib
temperature), S22 v5 TTA, S21 HSI probes. No fitting except the decodability probes.

Stages: ``freeze`` · ``run``.
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

from run_multimodal_reassessment import acquisition_decodability
from run_rgb_acquisition import direction_delta
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import aligned_logits, make_features, verify_plan
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, score, summarise

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s38_modality_roles.json")
EVIDENCE = Path("docs/research/evidence/S38_modality_roles")
OUT = Path("outputs/s38_modality_roles")
S21_PLAN = Path("docs/research/evidence/S21_complementary_rgb/preregistration.json")
S32 = Path("outputs/s32_rgb_finetune")
HSI_PROBES = ("hsi_snvmean214_own", "hsi_snvmean32", "hsi_mean214_own", "hsi_q214_own", "hsi_q32")
PRIMARY = "hsi_snvmean214_own"


def freeze() -> None:
    paths = [Path(__file__), S21_PLAN, Path("scripts/run_multimodal_reassessment.py"), Path("scripts/run_rgb_acquisition.py"),
             Path("src/spectralquadnet/experiments/screen_metrics.py"), Path("src/spectralquadnet/experiments/rgb_probe.py"),
             Path("outputs/s22_fusion_analysis/calibration.json"), S32 / "screen" / "COMPLETED.json",
             Path("dataset_u430k32/labels.npy"), Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv"),
             Path("dataset_rgb_hsi_v3/hsi_summary.npy"), Path("dataset_rgb_hsi_v3/morphology.npy")]
    for fold in (0, 1):
        paths += [Path(f"outputs/s21_rgb_study/probabilities_f{fold}.npz"), S32 / f"vitb_f{fold}" / "outputs.npz",
                  S32 / f"vitb_f{fold}" / "selection.json",
                  Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_val_test_tta.npz"),
                  Path(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz"),
                  Path(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz")]
    spec = {"study": "S38_modality_roles", "frozen_at": "2026-10-05",
            "scope": "CPU role screen on reused acquisitions; S21 corrected folds; S32 seed-0 trained ViT-B; S22 seed-0 v5 TTA; S21 HSI probes.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "arms": {"baseline": "equal(rgb_trained, v5 TTA) = S34 equal_trained",
                     "roles": [f"equal(rgb_trained, {p})" for p in HSI_PROBES],
                     "tri": f"0.5 rgb_trained + 0.25 v5 + 0.25 {PRIMARY}"},
            "primary": f"equal(rgb_trained, {PRIMARY}) minus equal(rgb_trained, v5 TTA): spectral shape only vs the full v5 HSI encoder.",
            "H59_screen": "cross recall delta >= .02 with paired variety CI lower > 0, AND mean F1 delta >= -.01.",
            "descriptive": ["all roles and tri vs baseline, with from/to-session-8 cell deltas",
                            "acquisition decodability (S34 method) of v5 embedding vs HSI spectral-shape and quantile features"],
            "decision": "H59 pass: the architecture assigns HSI the spectral-shape role (no HSI spatial/spread pathway). Fail: the "
                        "full v5 HSI encoder stays; role specialization by measured transfer is not supported.",
            "stop": "Saved components only; one pre-declared primary; no weight search."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S38 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def run() -> None:
    spec = verify_plan(PLAN)
    OUT.mkdir(exist_ok=False)
    cohort = Cohort.load()
    y = cohort.y
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    s21 = json.loads(S21_PLAN.read_text())
    feats = make_features(Path(s21["data"]), Path(s21["embeddings"]), s21["axes"])
    metrics, classes, predictions, decode = [], [], [], []
    for fold in (0, 1):
        te = cohort.heldout(fold)
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        v5 = softmax(aligned_logits(Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_val_test_tta.npz"), te, y) / t_hsi, axis=1)
        sel = json.loads((S32 / f"vitb_f{fold}" / "selection.json").read_text())
        with np.load(S32 / f"vitb_f{fold}" / "outputs.npz") as o:
            rgb = softmax(o["logits"].mean(0)[te] / sel["temperature"], axis=1)
        with np.load(f"outputs/s21_rgb_study/probabilities_f{fold}.npz") as d:
            if not np.array_equal(d["indices"], te):
                raise ValueError("S21 held-out rows differ")
            hsi = {p: d[p].astype(np.float64) for p in HSI_PROBES}
        probs: dict[str, Array] = {"rgb_trained": rgb, "v5_tta": v5, "baseline": (rgb + v5) / 2,
                                   "tri": .5 * rgb + .25 * v5 + .25 * hsi[PRIMARY]}
        for p, prob in hsi.items():
            probs[p] = prob
            probs["role_" + p] = (rgb + prob) / 2
        for arm, prob in probs.items():
            m, c, pr = score(cohort, fold, arm, te, prob)
            metrics.append(m)
            classes += c
            predictions += pr
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz") as c, \
                np.load(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz") as h:
            emb = np.zeros((len(y), c["hsi_train"].shape[1]))
            emb[c["train"]], emb[c["calib"]], emb[h["rows"]] = c["hsi_train"], c["hsi_calib"], h["hsi"]
        decode += acquisition_decodability(cohort, fold, {"v5_embedding": emb, "snvmean214_own": feats["hsi_snvmean214_own"],
                                                          "mean214_own": feats["hsi_mean214_own"], "q214_own": feats["hsi_q214_own"]})
    per_class = pd.DataFrame(classes)
    out = {}
    for arm in ["role_" + p for p in HSI_PROBES] + ["tri"]:
        out[arm] = {**contrast(per_class, arm, "baseline"),
                    "from_session8": direction_delta(per_class, arm, "baseline", "from_session8"),
                    "to_session8": direction_delta(per_class, arm, "baseline", "to_session8")}
    primary = out["role_" + PRIMARY]
    write_json(OUT / "hypothesis.json", {
        "H59_screen": bool(primary["delta_cross"] >= .02 and primary["cross_ci"][0] > 0 and primary["delta_f1"] >= -.01),
        "primary": primary, "contrasts": out, "network_training_runs": 0, "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    per_class.to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    pd.DataFrame(decode).to_csv(OUT / "acquisition_decodability.csv", index=False)
    summarise(pd.DataFrame(metrics), per_class).to_csv(OUT / "summary.csv")
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
