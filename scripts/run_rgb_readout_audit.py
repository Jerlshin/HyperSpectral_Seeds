#!/usr/bin/env python
"""S31: what RGB information does the frozen ViT-L class-token probe leave unused?

Stages:
``extract`` (label-free): frozen DINOv2 ViT-L/14 on the unchanged S20 masked crops under four
orientation views; for the last four blocks, the class token and the foreground-weighted mean
patch token; for the identity view, the final-layer patch tokens (float16) for trained readouts.
``freeze``: pins features, plan, gates and the readout arm list.
``run``: the S21 probe recipe (train-only scaling + shrinkage LDA, calib temperature) on every
readout, equal fusion with the saved S22 v5 TTA, calib-only candidate choice, one held-out scoring.
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
    fit_probe,
    verify_plan,
)
from spectralquadnet.experiments.rgb_readout import VIEWS, foreground_weights, layer_readouts, view
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, gate, score, summarise

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s31_rgb_readout_audit.json")
EVIDENCE = Path("docs/research/evidence/S31_rgb_readout_audit")
S21 = Path("docs/research/evidence/S21_complementary_rgb/preregistration.json")
DATA = Path("dataset_rgb_hsi_v3")
FEATURES = Path("outputs/s31_rgb_readouts")
OUT = Path("outputs/s31_rgb_readout_audit")
HUB = Path.home() / ".cache/torch/hub"
NAME, LAYERS, BATCH = "dinov2_vitl14", 4, 16


def load_backbone(device: torch.device) -> torch.nn.Module:
    model: torch.nn.Module = torch.hub.load(  # type: ignore[no-untyped-call]
        str(HUB / "facebookresearch_dinov2_main"), NAME, source="local", pretrained=False)
    model.load_state_dict(torch.load(HUB / f"checkpoints/{NAME}_pretrain.pth", map_location="cpu",
                                     weights_only=True), strict=True)
    return model.eval().to(device)


def extract() -> None:
    """Label-free: every crop is embedded identically; no partition or label is read."""
    FEATURES.mkdir(parents=True, exist_ok=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    images = np.load(DATA / "rgb.npy", mmap_mode="r")
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    if sha256(DATA / "rgb.npy") != manifest["files"]["rgb.npy"]["sha256"]:
        raise ValueError("RGB asset checksum mismatch")
    model = load_backbone(device)
    n = len(images)
    open_memmap = np.lib.format.open_memmap
    tokens = open_memmap(FEATURES / "tokens_id.npy", mode="w+", dtype=np.float16, shape=(n, 256, 1024))  # type: ignore[no-untyped-call]
    weights = np.zeros((n, 256), dtype=np.float32)
    receipts: dict[str, Any] = {"device": str(device), "torch": torch.__version__, "model": NAME,
                                "layers": "last 4 blocks (21-24), DINOv2 final norm applied",
                                "readout": "[class token, foreground-weighted mean patch token] per layer",
                                "views": list(VIEWS), "precision": "float32 compute",
                                "checkpoint_sha256": sha256(HUB / f"checkpoints/{NAME}_pretrain.pth"),
                                "asset_manifest_sha256": sha256(DATA / "MANIFEST.json"),
                                "source_sha256": {p: sha256(Path(p)) for p in (
                                    __file__, "scripts/cache_rgb_features.py",
                                    "src/spectralquadnet/experiments/rgb_readout.py")}}
    for name in VIEWS:
        start = time.perf_counter()
        out = np.zeros((n, LAYERS, 2048), dtype=np.float32)
        with torch.inference_mode():
            for begin in range(0, n, BATCH):
                raw = view(torch.from_numpy(np.array(images[begin:begin + BATCH], copy=True)), name)
                w = foreground_weights(raw).to(device)
                x = transform_rgb(raw.to(device), "rgb")
                readout, final = layer_readouts(model, x, w, LAYERS)
                out[begin:begin + BATCH] = readout.float().cpu().numpy()
                if name == "id":
                    tokens[begin:begin + BATCH] = final.half().cpu().numpy()
                    weights[begin:begin + BATCH] = w.cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite readout")
        np.save(FEATURES / f"readout_{name}.npy", out)
        receipts[name] = {"seconds": time.perf_counter() - start, "sha256": sha256(FEATURES / f"readout_{name}.npy")}
        print(name, receipts[name], flush=True)
    tokens.flush()
    np.save(FEATURES / "foreground_weights.npy", weights)
    s29 = np.load("outputs/s28_rgb_features/dino_l.npy")
    cls = np.load(FEATURES / "readout_id.npy")[:, -1, :1024]
    receipts["s29_class_token_audit"] = {"max_abs_difference": float(np.abs(cls - s29).max()),
                                         "max_relative_norm_difference": float(
                                             (np.linalg.norm(cls - s29, axis=1) / np.linalg.norm(s29, axis=1)).max())}
    receipts["tokens_sha256"] = sha256(FEATURES / "tokens_id.npy")
    receipts["foreground_weights_sha256"] = sha256(FEATURES / "foreground_weights.npy")
    write_json(FEATURES / "provenance.json", receipts)
    print("audit", receipts["s29_class_token_audit"], flush=True)


# Readout arms: name -> (layers, parts, views). Layers index the saved last-4 axis (-1 = block 24);
# parts: "cls" = class token only, "both" = [class, foreground mean]; views: orientation views averaged.
READOUTS: dict[str, tuple[list[int], str, tuple[str, ...]]] = {
    "cls": ([-1], "cls", ("id",)),                 # = S29 dino_l (replayed)
    "cls_fg": ([-1], "both", ("id",)),
    "last4": ([0, 1, 2, 3], "both", ("id",)),
    "cls_fg_tta": ([-1], "both", VIEWS),
    "last4_tta": ([0, 1, 2, 3], "both", VIEWS),
}
CANDIDATES = ("cls_fg", "last4", "cls_fg_tta", "last4_tta")


def readout_features(name: str, cache: dict[str, Array]) -> Array:
    layers, parts, views = READOUTS[name]
    width = 1024 if parts == "cls" else 2048
    stacked = np.mean([cache[v][:, layers, :width] for v in views], axis=0)
    return np.asarray(stacked.reshape(len(stacked), -1), dtype=np.float64)


def diagnostic_features() -> dict[str, Array]:
    """Explicit RGB cues the crop/backbone may discard: metric morphometrics and Lab colour."""
    morph = np.load(DATA / "rgb_morphology.npy").astype(np.float64)
    morph[:, [0, 1, 2, 6]] = np.log(morph[:, [0, 1, 2, 6]])  # positive size terms -> log scale
    return {"morph": morph, "colour": np.load(DATA / "rgb_descriptors.npy")[:, :15].astype(np.float64)}


def freeze() -> None:
    receipts = json.loads((FEATURES / "provenance.json").read_text())
    if receipts["s29_class_token_audit"]["max_abs_difference"] > 1e-4:
        raise ValueError("Identity class token does not reproduce S29")
    paths = [Path(__file__), S21, Path("scripts/cache_rgb_features.py"), FEATURES / "provenance.json",
             Path("src/spectralquadnet/experiments/rgb_readout.py"), Path("src/spectralquadnet/experiments/screen_metrics.py"),
             Path("src/spectralquadnet/experiments/rgb_probe.py"), DATA / "rgb_morphology.npy", DATA / "rgb_descriptors.npy",
             Path("dataset_u430k32/labels.npy"), Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv"),
             Path("outputs/s22_fusion_analysis/calibration.json"),
             Path("docs/research/evidence/S29_rgb_backbone_screen/screen_results/predictions.csv.gz")]
    paths += [FEATURES / f"readout_{v}.npy" for v in VIEWS]
    for fold in (0, 1):
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        paths += [root / "logits_calib_tta.npz", root / "logits_val_test_tta.npz"]
    spec = {"study": "S31_rgb_readout_audit", "frozen_at": "2026-10-05",
            "scope": "Frozen DINOv2 ViT-L/14 readout development screen on reused acquisitions; S21 corrected folds; S22 seed-0 v5 TTA for fusion. No network training.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "temperatures": json.loads(S21.read_text())["temperatures"],
            "readouts": {k: {"layers_from_last4": v[0], "parts": v[1], "views": list(v[2])} for k, v in READOUTS.items()},
            "candidates": list(CANDIDATES), "diagnostics": ["morph (8 metric morphometrics, log size terms)", "colour (15 Lab statistics)",
                                                             "cls_fg_tta_morph (candidate-free concatenation)"],
            "probe": "S21 fit_probe on outer-training rows (train-only scaling + shrinkage LDA); calib log-loss temperature on the S21 grid. View-averaged features are averaged before the probe.",
            "fusion": "Equal probability fusion with the saved S22 seed-0 v5 TTA logits at the S22 calib temperatures (equal_<arm>).",
            "candidate_selection": "Before held-out scoring: candidate = argmax over CANDIDATES of two-fold mean calib macro-F1 of the RGB probe; tie -> earlier in list. Written to selection.json first.",
            "replay_audit": "cls RGB-only and equal_cls predictions reproduce S29 dino_l / equal_l exactly; any disagreement stops.",
            "H48_screen": "RGB-only candidate minus cls: mean F1 >= .02, paired variety CI lower > 0, positive on each fold, mean cross delta >= 0.",
            "H49_descriptive": "System: equal_candidate minus equal_cls (no gate; the system decision is reserved for the trained-branch study).",
            "decision": "Pass: the candidate readout becomes the frozen-ViT-L reference and the trained branch's head readout. Fail: the class token already carries the frozen backbone's usable RGB information; any gain must come from adapting the backbone.",
            "stop": "Five readouts, two diagnostics, one probe recipe; no classifier, resolution or temperature search."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S31 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def run() -> None:
    spec = verify_plan(PLAN)
    OUT.mkdir(exist_ok=False)
    write_json(OUT / "STARTED.json", {"plan_sha256": sha256(PLAN)})
    start = time.perf_counter()
    cohort = Cohort.load()
    y = cohort.y
    cache = {v: np.load(FEATURES / f"readout_{v}.npy") for v in VIEWS}
    features = {name: readout_features(name, cache) for name in READOUTS}
    features.update(diagnostic_features())
    features["cls_fg_tta_morph"] = np.c_[features["cls_fg_tta"], features["morph"]]
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    calib_rows, probs = [], {}
    for fold in (0, 1):
        split = cohort.splits[fold]
        tr, ca, te = split["train"], split["calib"], cohort.heldout(fold)
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        hsi_te = softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)
        hsi_ca = softmax(aligned_logits(root / "logits_calib_tta.npz", ca, y) / t_hsi, axis=1)
        probs[fold] = {"hsi_tta": hsi_te}
        for arm, x in features.items():
            cal, logits = fit_probe(x, y, tr, ca, te, export=OUT / f"probe_f{fold}_{arm}.npz")
            temp = calibrate_temperature(cal, y[ca], spec["temperatures"])
            p_ca, p_te = softmax(cal / temp, axis=1), softmax(logits / temp, axis=1)
            probs[fold][arm], probs[fold]["equal_" + arm] = p_te, (hsi_te + p_te) / 2
            calib_rows.append({"fold": fold, "arm": arm, "dimensions": x.shape[1], "temperature": temp,
                               "rgb_calib_f1": float(class_metrics(y[ca], cal.argmax(1), 90)["f1"].mean()),
                               "equal_calib_f1": float(class_metrics(y[ca], ((hsi_ca + p_ca) / 2).argmax(1), 90)["f1"].mean())})
            print(fold, calib_rows[-1], flush=True)
    calib = pd.DataFrame(calib_rows)
    choice = calib[calib.arm.isin(CANDIDATES)].groupby("arm", sort=False).rgb_calib_f1.mean().reindex(CANDIDATES)
    candidate = str(choice.idxmax())
    write_json(OUT / "selection.json", {"candidate": candidate, "calib": calib_rows, "two_fold_calib_f1": choice.to_dict(),
                                        "rule": spec["candidate_selection"], "selected_before_held_out_scoring": True})
    print("Calib-selected candidate", candidate, choice.to_dict(), flush=True)
    # Held-out scoring of the frozen arm list, once.
    s29 = pd.read_csv("docs/research/evidence/S29_rgb_backbone_screen/screen_results/predictions.csv.gz")
    metrics, classes, predictions, audits = [], [], [], []
    for fold, block in probs.items():
        te = cohort.heldout(fold)
        np.savez_compressed(OUT / f"probabilities_f{fold}.npz", rows=te, **{k: v.astype(np.float32) for k, v in block.items()})
        ref = s29[s29.fold == fold].set_index(["arm", "index"]).prediction
        audit = {"fold": fold, "cls_vs_s29_dino_l": int(np.sum(block["cls"].argmax(1) != ref.loc["dino_l"].loc[te].to_numpy())),
                 "equal_cls_vs_s29_equal_l": int(np.sum(block["equal_cls"].argmax(1) != ref.loc["equal_l"].loc[te].to_numpy()))}
        audits.append(audit)
        if any(v for k, v in audit.items() if k != "fold"):
            raise ValueError(f"Replay audit failed: {audit}")
        for arm, prob in block.items():
            m, c, p = score(cohort, fold, arm, te, prob)
            metrics.append(m)
            classes += c
            predictions += p
    per_class = pd.DataFrame(classes)
    primary = contrast(per_class, candidate, "cls")
    descriptive = {f"{a}_vs_{b}": contrast(per_class, a, b) for a, b in
                   [(c, "cls") for c in CANDIDATES if c != candidate] +
                   [("equal_" + c, "equal_cls") for c in CANDIDATES] +
                   [("cls_fg_tta_morph", "cls_fg_tta"), ("equal_cls_fg_tta_morph", "equal_cls_fg_tta"),
                    ("equal_morph", "hsi_tta"), ("equal_colour", "hsi_tta")]}
    write_json(OUT / "hypothesis.json", {"H48_screen": gate(primary, .02), "candidate": candidate, "primary": primary,
                                         "transfer_gain_supported": bool(primary["cross_ci"][0] > 0),
                                         "descriptive": descriptive, "replay_audit": audits, "network_training_runs": 0,
                                         "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    per_class.to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    summarise(pd.DataFrame(metrics), per_class).to_csv(OUT / "summary.csv")
    write_json(OUT / "execution.json", {"wall_seconds": time.perf_counter() - start})
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "freeze", "run"))
    {"extract": extract, "freeze": freeze, "run": run}[parser.parse_args().action]()
