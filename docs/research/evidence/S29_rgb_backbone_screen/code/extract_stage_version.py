#!/usr/bin/env python
"""S28: does frozen RGB backbone capacity strengthen the fixed multimodal system?

Stages: ``extract`` (label-free frozen DINOv2 ViT-B/14 and ViT-L/14 class tokens on
the unchanged S20 masked 224 crops, plus a ViT-S/14 numerical re-extraction audit),
``freeze`` (pins features, checkpoints, source and gates), ``run`` (S21 shrinkage-LDA
probes on outer-training rows, calib temperatures, calib-only backbone choice,
then one held-out scoring of the fixed arm list with fixed equal v5-TTA fusion).
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

from cache_rgb_features import transform_rgb
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import (
    aligned_logits,
    calibrate_temperature,
    class_metrics,
    cluster_interval,
    fit_probe,
    verify_plan,
)

PLAN = Path("configs/research/s28_rgb_backbone_screen.json")
PARENT = Path("configs/research/s27_tta_trained_head.json")
S21 = Path("docs/research/evidence/S21_complementary_rgb/preregistration.json")
FEATURES = Path("outputs/s28_rgb_features")
OUT = Path("outputs/s28_rgb_backbone_screen")
HUB = Path.home() / ".cache/torch/hub"
BACKBONES = {"dino_b": ("dinov2_vitb14", 768), "dino_l": ("dinov2_vitl14", 1024)}


def load_backbone(name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(str(HUB / "facebookresearch_dinov2_main"), name,  # type: ignore[no-untyped-call]
                           source="local", pretrained=False)
    model.load_state_dict(torch.load(HUB / f"checkpoints/{name}_pretrain.pth", map_location="cpu",
                                     weights_only=True), strict=True)
    return model.eval().to(device)


def embed(model: torch.nn.Module, images: npt.NDArray[Any], device: torch.device, batch: int = 32) -> npt.NDArray[Any]:
    out = []
    with torch.inference_mode():
        for begin in range(0, len(images), batch):
            x = torch.from_numpy(np.array(images[begin:begin + batch], copy=True)).to(device)
            out.append(model(transform_rgb(x, "rgb")).float().cpu().numpy())
    result = np.concatenate(out)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite embedding")
    return result


def extract() -> None:
    """Label-free: every crop is embedded identically; no row partition is read."""
    FEATURES.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    images = np.load("dataset_rgb_hsi_v3/rgb.npy", mmap_mode="r")
    manifest = json.loads(Path("dataset_rgb_hsi_v3/MANIFEST.json").read_text())
    if sha256(Path("dataset_rgb_hsi_v3/rgb.npy")) != manifest["files"]["rgb.npy"]["sha256"]:
        raise ValueError("RGB asset checksum mismatch")
    receipts: dict[str, Any] = {"device": str(device), "torch": torch.__version__,
                                "transform": "S20 cache_rgb_features.transform_rgb(view='rgb'): 224 masked crop, ImageNet normalization; class token; eval; no TTA",
                                "asset_manifest_sha256": sha256(Path("dataset_rgb_hsi_v3/MANIFEST.json")),
                                "source_sha256": {p: sha256(Path(p)) for p in ("scripts/cache_rgb_features.py", __file__)}}
    # Numerical audit: re-extract ViT-S/14 on the first 512 crops on this device.
    small = embed(load_backbone("dinov2_vits14", device), images[:512], device)
    reference = np.load("outputs/s20_rgb_features_v3/rgb.npy")[:512]
    receipts["vits14_audit"] = {"rows": 512, "max_abs_difference_vs_s20_cpu": float(np.abs(small - reference).max()),
                                "max_relative_norm_difference": float((np.linalg.norm(small - reference, axis=1) /
                                                                       np.linalg.norm(reference, axis=1)).max())}
    for arm, (name, dim) in BACKBONES.items():
        start = time.perf_counter()
        features = embed(load_backbone(name, device), images, device)
        if features.shape != (len(images), dim):
            raise ValueError("Unexpected embedding shape")
        np.save(FEATURES / f"{arm}.npy", features.astype(np.float32))
        receipts[arm] = {"model": name, "dimensions": dim, "seconds": time.perf_counter() - start,
                         "checkpoint_sha256": sha256(HUB / f"checkpoints/{name}_pretrain.pth"),
                         "checkpoint_url": f"https://dl.fbaipublicfiles.com/dinov2/{name}/{name}_pretrain.pth",
                         "sha256": sha256(FEATURES / f"{arm}.npy")}
        print(arm, receipts[arm], flush=True)
    write_json(FEATURES / "provenance.json", receipts)


def freeze() -> None:
    parent = verify_plan(PARENT)
    state = json.loads(Path("outputs/s27_tta_trained_head/COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(PARENT):
        raise ValueError("S27 completion plan differs")
    h27 = json.loads(Path("outputs/s27_tta_trained_head/hypothesis.json").read_text())
    if h27["H45_screen"]:
        raise ValueError("S27 passed; this successor assumes the learned-head line closed")
    inputs = dict(parent["input_hashes"])
    paths = [PARENT, Path(__file__), S21, Path("scripts/cache_rgb_features.py"),
             Path("outputs/s27_tta_trained_head/COMPLETED.json"), Path("outputs/s27_tta_trained_head/hypothesis.json"),
             Path("outputs/s20_rgb_features_v3/rgb.npy"), FEATURES / "provenance.json",
             Path("dataset_u430k32/labels.npy"), Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv"),
             Path("outputs/s22_fusion_analysis/calibration.json"), Path("outputs/s22_fusion_analysis/predictions.csv.gz")]
    paths += [FEATURES / f"{arm}.npy" for arm in BACKBONES]
    paths += [HUB / f"checkpoints/{name}_pretrain.pth" for name, _ in BACKBONES.values()]
    for f in (0, 1):
        paths += [Path(f"outputs/s21_rgb_study/probabilities_f{f}.npz")]
        paths += [Path(f"outputs/s22_complementary_v5/f{f}_s0/results/logits_{s}_tta.npz") for s in ("calib", "val_test")]
    inputs.update({str(p): sha256(p) for p in paths})
    spec = {"study": "S28_rgb_backbone_screen", "frozen_at": "2026-10-05", "parent_plan": str(PARENT),
            "scope": "Frozen-feature development screen on reused acquisitions; S21 corrected folds; S22 seed0 v5 TTA; no training of any network.",
            "input_hashes": inputs, "partition_plan": str(S21), "temperatures": json.loads(S21.read_text())["temperatures"],
            "arms": {"rgb": ["dino_s", "dino_b", "dino_l"], "fusion": ["equal_s", "equal_b", "equal_l"]},
            "probe": "S21 fit_probe: train-only StandardScaler + shrinkage LDA (lsqr, auto); calib log-loss temperature from the S21 grid; no refit on calib.",
            "fusion": "Fixed equal mean of softmax(S22 v5 TTA / S22 calib temperature) and calibrated RGB probabilities. No learned weight or head.",
            "replay_audit": "dino_s probabilities must reproduce S21 dino_rgb probabilities and equal_s must reproduce S22 v5_rgb_equal predictions exactly; otherwise stop.",
            "candidate_selection": "Before held-out scoring: candidate = argmax over {dino_b, dino_l} of the two-fold mean CALIB macro-F1 of its equal fusion; tie -> smaller model. Written to selection.json first.",
            "H46_screen": "Candidate equal fusion minus equal_s (current system): mean F1 >= .01, paired variety CI lower > 0, positive F1 on each fold, mean cross-recall delta >= 0. Transfer support additionally requires cross CI lower > 0. Non-candidate backbone and RGB-only contrasts are descriptive.",
            "diagnostic": "Cross-error session attraction per fusion arm (fraction of cross-session errors predicting a class trained in the test kernel's session), descriptive.",
            "decision": "Pass: the candidate backbone replaces ViT-S/14 in the fixed multimodal system; confirmation joins the final matched allocation. Fail: RGB backbone capacity is not the bottleneck; keep ViT-S/14 and redirect to acquisition-limited transfer work.",
            "stop": "Two new backbones, one fixed probe recipe, no sweep, no fine-tuning, no learned fusion."}
    for p in (PLAN, Path("docs/research/evidence/S28_rgb_backbone_screen/preregistration.json")):
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
    start = time.perf_counter()
    y = np.load("dataset_u430k32/labels.npy")
    groups = np.load("dataset_u430k32/groups.npy")
    scans = pd.read_csv("dataset_u430k32/scan_table.csv").set_index("scan_id")
    session = scans.session_id.loc[groups].to_numpy()
    cross = scans.groupby("label").session_id.nunique().sort_index().to_numpy() > 1
    splits = json.loads(S21.read_text())["splits"]
    features = {"dino_s": np.load("outputs/s20_rgb_features_v3/rgb.npy"),
                **{arm: np.load(FEATURES / f"{arm}.npy") for arm in BACKBONES}}
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    s22 = pd.read_csv("outputs/s22_fusion_analysis/predictions.csv.gz")
    held: dict[int, dict[str, Any]] = {}
    calib_rows: list[dict[str, Any]] = []
    for fold in (0, 1):
        split = {k: np.asarray(v, dtype=int) for k, v in splits[str(fold)].items()}
        tr, ca = split["train"], split["calib"]
        te = np.sort(np.r_[split["val"], split["test"]])
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        hsi_ca = softmax(aligned_logits(root / "logits_calib_tta.npz", ca, y) / t_hsi, axis=1)
        hsi_te = softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)
        probs: dict[str, Any] = {"hsi_tta": hsi_te}
        for arm, x in features.items():
            cal, logits = fit_probe(x, y, tr, ca, te, export=OUT / f"probe_f{fold}_{arm}.npz")
            temp = calibrate_temperature(cal, y[ca], spec["temperatures"])
            probs[arm] = softmax(logits / temp, axis=1)
            tag = "equal_" + arm[-1]
            probs[tag] = (hsi_te + probs[arm]) / 2
            fused_ca = (hsi_ca + softmax(cal / temp, axis=1)) / 2
            calib_rows.append({"fold": fold, "arm": arm, "temperature": temp, "dimensions": x.shape[1],
                               "rgb_calib_f1": float(class_metrics(y[ca], cal.argmax(1), 90)["f1"].mean()),
                               "fusion_calib_f1": float(class_metrics(y[ca], fused_ca.argmax(1), 90)["f1"].mean())})
        held[fold] = {"te": te, "probs": probs, "train_rows": np.r_[tr, ca]}
        print("Fitted probes fold", fold, flush=True)
    calib = pd.DataFrame(calib_rows)
    choice = calib[calib.arm != "dino_s"].groupby("arm").fusion_calib_f1.mean()
    candidate = str(choice.idxmax()) if choice.nunique() > 1 else "dino_b"
    write_json(OUT / "selection.json", {"candidate": candidate, "calib": calib_rows,
               "rule": spec["candidate_selection"], "selected_before_held_out_scoring": True})
    print("Calib-selected candidate", candidate, choice.to_dict(), flush=True)
    # Held-out scoring of the frozen arm list, once.
    metrics: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    audits = []
    for fold, block in held.items():
        te, probs = block["te"], block["probs"]
        with np.load(f"outputs/s21_rgb_study/probabilities_f{fold}.npz") as old:
            if not np.array_equal(old["indices"], te):
                raise ValueError("S21 rows differ")
            diff = float(np.abs(old["dino_rgb"] - probs["dino_s"]).max())
            flips = int(np.sum(old["dino_rgb"].argmax(1) != probs["dino_s"].argmax(1)))
        ref = s22[(s22.fold == fold) & (s22.arm == "v5_rgb_equal")].set_index("index").loc[te].prediction.to_numpy()
        mismatch = int(np.sum(probs["equal_s"].argmax(1) != ref))
        audits.append({"fold": fold, "dino_s_max_prob_difference_vs_s21": diff, "dino_s_vs_s21_disagreements": flips,
                       "equal_s_vs_s22_disagreements": mismatch})
        if diff > 1e-6 or flips or mismatch:
            raise ValueError("Replay audit failed: current system not reproduced")
        trained_in = {c: set(session[block["train_rows"]][y[block["train_rows"]] == c]) for c in range(90)}
        for arm in ("hsi_tta", "dino_s", "dino_b", "dino_l", "equal_s", "equal_b", "equal_l"):
            pred = probs[arm].argmax(1)
            stat = class_metrics(y[te], pred, 90)
            wrong = np.flatnonzero((pred != y[te]) & cross[y[te]])
            attraction = float(np.mean([session[te][i] in trained_in[pred[i]] for i in wrong])) if len(wrong) else float("nan")
            metrics.append({"fold": fold, "arm": arm, "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(pred == y[te])),
                            "same_recall": float(stat["recall"][~cross].mean()), "cross_recall": float(stat["recall"][cross].mean()),
                            "cross_error_session_attraction": attraction})
            classes.extend({"fold": fold, "arm": arm, "label": c, "cross": bool(cross[c]),
                            "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])} for c in range(90))
            predictions.extend({"fold": fold, "arm": arm, "index": int(i), "target": int(t), "prediction": int(p)}
                               for i, t, p in zip(te, y[te], pred, strict=True))
    cls = pd.DataFrame(classes).set_index(["arm", "label", "fold"]).sort_index()

    def contrast(a: str, b: str) -> dict[str, Any]:
        df = (cls.loc[a].f1 - cls.loc[b].f1).unstack("fold").to_numpy()
        dr = (cls.loc[a].recall - cls.loc[b].recall)[cls.loc[a].cross].unstack("fold").to_numpy()
        return {"delta_f1": float(df.mean()), "f1_ci": cluster_interval(df), "fold_f1_deltas": df.mean(0).tolist(),
                "delta_cross": float(dr.mean()), "cross_ci": cluster_interval(dr), "fold_cross_deltas": dr.mean(0).tolist()}
    tag = "equal_" + candidate[-1]
    primary = contrast(tag, "equal_s")
    gate = bool(primary["delta_f1"] >= .01 and primary["f1_ci"][0] > 0 and min(primary["fold_f1_deltas"]) > 0
                and primary["delta_cross"] >= 0)
    other = "dino_l" if candidate == "dino_b" else "dino_b"
    write_json(OUT / "hypothesis.json", {"H46_screen": gate, "candidate": candidate, "primary": primary,
               "transfer_gain_supported": bool(primary["cross_ci"][0] > 0),
               "descriptive": {f"equal_{other[-1]}_vs_equal_s": contrast("equal_" + other[-1], "equal_s"),
                               "dino_b_vs_dino_s": contrast("dino_b", "dino_s"), "dino_l_vs_dino_s": contrast("dino_l", "dino_s"),
                               f"{tag}_vs_hsi_tta": contrast(tag, "hsi_tta")},
               "replay_audit": audits, "network_training_runs": 0, "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    write_json(OUT / "execution.json", {"wall_seconds": time.perf_counter() - start})
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "freeze", "run"))
    {"extract": extract, "freeze": freeze, "run": run}[parser.parse_args().action]()
