#!/usr/bin/env python
"""S34: with a strong RGB branch, are the current multimodal baselines still competitive, and what
does the evidence say the next architecture must do?

CPU only, no training. Components are resolved from sealed outputs:
* HSI: saved S22 seed-0 v5 TTA (calib / held-out);
* RGB trained: S32's calib-selected fine-tuned branch (4-view TTA);
* RGB frozen: S31's calib-selected ViT-L readout (or ``cls`` if H48 failed).

Arms (frozen before S32 is scored): fixed fusions of the three, an RGB-only trained+frozen ensemble,
and a within-scan shuffled-pairing control. The S29 selected system (TTA-trained head on ViT-L) is
the incumbent baseline. Diagnostics: kernel-level complementarity and acquisition (session)
decodability of each branch's representation on held-out bridge kernels.
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

from run_rgb_readout_audit import READOUTS, readout_features
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import (
    aligned_logits,
    class_metrics,
    fit_probe,
    probe_logits,
    verify_plan,
)
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, gate, score, summarise

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s34_multimodal_reassessment.json")
EVIDENCE = Path("docs/research/evidence/S34_multimodal_reassessment")
OUT = Path("outputs/s34_multimodal_reassessment")
S31_PLAN, S31_OUT = Path("configs/research/s31_rgb_readout_audit.json"), Path("outputs/s31_rgb_readout_audit")
S32_PLAN, S32_OUT = Path("configs/research/s32_rgb_finetune.json"), Path("outputs/s32_rgb_finetune")
S29_PRED = Path("docs/research/evidence/S29_rgb_backbone_screen/screen_results/predictions.csv.gz")
FUSIONS = {  # weights over (hsi, rgb_trained, rgb_frozen)
    "equal_trained": (0.5, 0.5, 0.0),
    "equal_frozen": (0.5, 0.0, 0.5),
    "tri": (0.5, 0.25, 0.25),
    "rgb_ensemble": (0.0, 0.5, 0.5),
}
CANDIDATES = ("equal_trained", "tri")


def sealed(out: Path, plan: Path) -> None:
    state = json.loads((out / "COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(plan) or any(sha256(out / n) != d for n, d in state["files"].items()):
        raise ValueError(f"{out} is not sealed under {plan}")


def freeze() -> None:
    verify_plan(S31_PLAN)
    verify_plan(S32_PLAN)
    if (S32_OUT / "screen").exists():
        raise ValueError("Freeze S34 before S32 scoring so its design cannot depend on S32's outcome")
    paths = [Path(__file__), S31_PLAN, S32_PLAN, S29_PRED, Path("scripts/run_rgb_readout_audit.py"),
             Path("src/spectralquadnet/experiments/screen_metrics.py"), Path("src/spectralquadnet/experiments/rgb_probe.py"),
             Path("outputs/s22_fusion_analysis/calibration.json"), Path("outputs/s20_rgb_features_v3/rgb.npy"),
             Path("outputs/s28_rgb_features/dino_l.npy"), Path("dataset_u430k32/labels.npy"),
             Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv")]
    for fold in (0, 1):
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        paths += [root / "logits_calib_tta.npz", root / "logits_val_test_tta.npz",
                  Path(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz"),
                  Path(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz")]
    spec = {"study": "S34_multimodal_reassessment", "frozen_at": "2026-10-05",
            "scope": "CPU reassessment on reused acquisitions; S21 corrected folds; seed-0 HSI v5 and seed-0 S32 RGB. No training.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "components": {"hsi": "S22 v5 TTA at S22 calib temperatures",
                           "rgb_trained": "S32 calib-selected backbone, 4-view mean logits at its calib temperature",
                           "rgb_frozen": "S31 calib-selected readout if H48-screen passed, else cls; S31 probe and temperature"},
            "fusions": {k: list(v) for k, v in FUSIONS.items()},
            "incumbent": "S29 head_l (TTA-trained head on frozen ViT-L), saved predictions",
            "candidate_selection": "Before held-out scoring: argmax over CANDIDATES of two-fold mean calib macro-F1; tie -> equal_trained.",
            "H54_screen": "candidate minus S29 head_l: mean F1 >= .01, paired variety CI lower > 0, positive on each fold, mean cross delta >= 0.",
            "descriptive": ["tri - equal_trained (frozen readout adds to trained RGB?)", "rgb_ensemble - rgb_trained",
                            "equal_trained - equal_trained_shuffled (kernel-level pairing; shuffle within held-out scan, seed 20261005)",
                            "equal_trained - hsi_tta; equal_trained - S29 equal_l", "directions, session attraction",
                            "complementarity (either-oracle, both-wrong) per pair", "acquisition decodability: S21 probe "
                            "(train-only LDA) predicting session from each representation, accuracy on held-out bridge kernels"],
            "decision": "H54 pass: the S29 system is superseded as the development baseline by the strong-RGB fusion; the "
                        "next architecture is built around the trained RGB branch. Fail: the S29 system stays competitive.",
            "stop": "Fixed weights, one calib choice between two fusions; no weight search."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S34 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def components(cohort: Cohort, fold: int) -> tuple[dict[str, dict[str, Array]], str, str]:
    """Calib and held-out probabilities of the three components for one fold."""
    y, ca, te = cohort.y, cohort.splits[fold]["calib"], cohort.heldout(fold)
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
    root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
    hsi = {"calib": softmax(aligned_logits(root / "logits_calib_tta.npz", ca, y) / t_hsi, axis=1),
           "te": softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)}
    s32 = json.loads((S32_OUT / "screen" / "hypothesis.json").read_text())
    trained_arm = s32["candidate"]
    sel = json.loads((S32_OUT / f"{trained_arm}_f{fold}" / "selection.json").read_text())
    with np.load(S32_OUT / f"{trained_arm}_f{fold}" / "outputs.npz") as o:
        z = o["logits"].mean(0)
    trained = {"calib": softmax(z[ca] / sel["temperature"], axis=1), "te": softmax(z[te] / sel["temperature"], axis=1)}
    s31 = json.loads((S31_OUT / "hypothesis.json").read_text())
    frozen_arm = s31["candidate"] if s31["H48_screen"] else "cls"
    cache = {v: np.load(Path("outputs/s31_rgb_readouts") / f"readout_{v}.npy", mmap_mode="r") for v in READOUTS[frozen_arm][2]}
    x = readout_features(frozen_arm, cache)
    t_rgb = next(c["temperature"] for c in json.loads((S31_OUT / "selection.json").read_text())["calib"]
                 if c["fold"] == fold and c["arm"] == frozen_arm)
    export = S31_OUT / f"probe_f{fold}_{frozen_arm}.npz"
    frozen = {"calib": softmax(probe_logits(export, x[ca]) / t_rgb, axis=1), "te": softmax(probe_logits(export, x[te]) / t_rgb, axis=1)}
    with np.load(S31_OUT / f"probabilities_f{fold}.npz") as saved:
        if np.any(saved[frozen_arm].argmax(1) != frozen["te"].argmax(1)):
            raise ValueError("Frozen component does not replay S31")
    return {"hsi": hsi, "rgb_trained": trained, "rgb_frozen": frozen}, trained_arm, frozen_arm


def shuffled_within_scan(prob: Array, scans: Array, seed: int = 20261005) -> Array:
    """Permute rows among kernels of the same held-out scan (class-pure), breaking kernel pairing."""
    rng = np.random.default_rng(seed)
    out = prob.copy()
    for scan in np.unique(scans):
        idx = np.flatnonzero(scans == scan)
        out[idx] = prob[rng.permutation(idx)]
    return out


def acquisition_decodability(cohort: Cohort, fold: int, features: dict[str, Array]) -> list[dict[str, Any]]:
    """Train-only LDA predicting acquisition session; accuracy on held-out bridge kernels, whose session
    differs from every training session of their own variety (so class identity cannot supply it)."""
    tr, ca, te = cohort.splits[fold]["train"], cohort.splits[fold]["calib"], cohort.heldout(fold)
    bridge = te[cohort.cross[cohort.y[te]]]
    rows = []
    for name, x in features.items():
        _, logits = fit_probe(x, cohort.session, tr, ca, bridge)
        pred = logits.argmax(1)
        own = np.array([set(cohort.session[tr][cohort.y[tr] == c]) for c in cohort.y[bridge]], dtype=object)
        rows.append({"fold": fold, "representation": name, "bridge_kernels": len(bridge),
                     "session_accuracy": float(np.mean(pred == cohort.session[bridge])),
                     "predicts_own_class_training_session": float(np.mean([p in o for p, o in zip(pred, own, strict=True)]))})
    return rows


def run() -> None:
    spec = verify_plan(PLAN)
    sealed(S31_OUT, S31_PLAN)
    sealed(S32_OUT / "screen", S32_PLAN)
    OUT.mkdir(exist_ok=False)
    cohort = Cohort.load()
    y = cohort.y
    groups = np.load("dataset_u430k32/groups.npy")
    s29 = pd.read_csv(S29_PRED)
    held, calib_rows, comp, decode = {}, [], [], []
    resolved: dict[str, str] = {}
    for fold in (0, 1):
        parts, trained_arm, frozen_arm = components(cohort, fold)
        resolved = {"rgb_trained": trained_arm, "rgb_frozen": frozen_arm}
        te, ca = cohort.heldout(fold), cohort.splits[fold]["calib"]
        probs = {"hsi_tta": parts["hsi"]["te"], "rgb_trained": parts["rgb_trained"]["te"], "rgb_frozen": parts["rgb_frozen"]["te"]}
        for name, (wh, wt, wf) in FUSIONS.items():
            mix = {k: wh * parts["hsi"][k] + wt * parts["rgb_trained"][k] + wf * parts["rgb_frozen"][k] for k in ("calib", "te")}
            probs[name] = mix["te"]
            calib_rows.append({"fold": fold, "arm": name, "calib_f1": float(class_metrics(y[ca], mix["calib"].argmax(1), 90)["f1"].mean())})
        probs["equal_trained_shuffled"] = 0.5 * parts["hsi"]["te"] + 0.5 * shuffled_within_scan(parts["rgb_trained"]["te"], groups[te])
        for arm in ("head_l", "equal_l"):
            pred = s29[(s29.fold == fold) & (s29.arm == arm)].set_index("index").prediction.loc[te].to_numpy()
            probs["s29_" + arm] = np.eye(90)[pred]
        held[fold] = probs
        wrong = {k: probs[k].argmax(1) != y[te] for k in ("hsi_tta", "rgb_trained", "rgb_frozen")}
        bridge = cohort.cross[y[te]]
        for a, b in (("hsi_tta", "rgb_trained"), ("hsi_tta", "rgb_frozen"), ("rgb_trained", "rgb_frozen")):
            comp.append({"fold": fold, "pair": f"{a}+{b}", "either_oracle_accuracy": float(np.mean(~wrong[a] | ~wrong[b])),
                         "both_wrong_same": float(np.mean(wrong[a][~bridge] & wrong[b][~bridge])),
                         "both_wrong_cross": float(np.mean(wrong[a][bridge] & wrong[b][bridge])),
                         "error_phi": float(np.corrcoef(wrong[a], wrong[b])[0, 1])})
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz") as c, \
                np.load(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz") as h:
            hsi_emb = np.zeros((len(y), c["hsi_train"].shape[1]))
            hsi_emb[c["train"]], hsi_emb[c["calib"]], hsi_emb[h["rows"]] = c["hsi_train"], c["hsi_calib"], h["hsi"]
        with np.load(S32_OUT / f"{trained_arm}_f{fold}" / "outputs.npz") as o:
            trained_emb = o["embeddings"]
        decode += acquisition_decodability(cohort, fold, {
            "hsi_v5_embedding": hsi_emb, "rgb_trained_embedding": trained_emb,
            "rgb_frozen_vitl_cls": np.load("outputs/s28_rgb_features/dino_l.npy"),
            "rgb_frozen_vits_cls": np.load("outputs/s20_rgb_features_v3/rgb.npy")})
        print("fold", fold, "components resolved", resolved, flush=True)
    calib = pd.DataFrame(calib_rows)
    choice = calib[calib.arm.isin(CANDIDATES)].groupby("arm").calib_f1.mean().reindex(CANDIDATES)
    candidate = str(choice.idxmax())
    write_json(OUT / "selection.json", {"candidate": candidate, "two_fold_calib_f1": choice.to_dict(), "calib": calib_rows,
                                        "resolved_components": resolved, "selected_before_held_out_scoring": True})
    metrics, classes, predictions = [], [], []
    for fold, probs in held.items():
        for arm, prob in probs.items():
            m, c, p = score(cohort, fold, arm, cohort.heldout(fold), prob)
            metrics.append(m)
            classes += c
            predictions += p
    per_class = pd.DataFrame(classes)
    primary = contrast(per_class, candidate, "s29_head_l")
    pairs = [("tri", "equal_trained"), ("rgb_ensemble", "rgb_trained"), ("equal_trained", "equal_trained_shuffled"),
             ("equal_trained", "hsi_tta"), ("equal_trained", "s29_equal_l"), ("equal_trained", "s29_head_l"),
             ("tri", "s29_head_l"), ("equal_frozen", "s29_equal_l"), ("rgb_trained", "rgb_frozen")]
    write_json(OUT / "hypothesis.json", {"H54_screen": gate(primary, .01), "candidate": candidate, "primary": primary,
                                         "transfer_gain_supported": bool(primary["cross_ci"][0] > 0),
                                         "descriptive": {f"{a}_vs_{b}": contrast(per_class, a, b) for a, b in pairs},
                                         "resolved_components": resolved, "network_training_runs": 0, "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    per_class.to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    pd.DataFrame(comp).to_csv(OUT / "complementarity.csv", index=False)
    pd.DataFrame(decode).to_csv(OUT / "acquisition_decodability.csv", index=False)
    summarise(pd.DataFrame(metrics), per_class).to_csv(OUT / "summary.csv")
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
