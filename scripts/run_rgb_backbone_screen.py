#!/usr/bin/env python
"""S29: does frozen RGB backbone capacity strengthen the selected multimodal system?

Stages: ``extract`` (label-free frozen DINOv2 ViT-B/14 and ViT-L/14 class tokens on
the unchanged S20 masked 224 crops, plus a ViT-S/14 numerical re-extraction audit;
run before renumbering, so features live in ``outputs/s28_rgb_features``),
``freeze`` (pins features, checkpoints, source and gates), ``run`` (S21 shrinkage-LDA
probes on outer-training rows, calib temperatures, the S27 TTA-trained head refit on
each new backbone, calib-only backbone choice, then one held-out scoring).
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
from spectralquadnet.experiments.residual_head import fit_residual_head
from spectralquadnet.experiments.rgb_probe import (
    aligned_logits,
    calibrate_temperature,
    class_metrics,
    cluster_interval,
    fit_probe,
    probe_logits,
    verify_plan,
)
from spectralquadnet.models.multimodal_residual import FrozenResidualFusion

PLAN = Path("configs/research/s29_rgb_backbone_screen.json")
PARENT = Path("configs/research/s28_tta_head_seeds.json")
S27 = Path("outputs/s27_tta_trained_head")
S21 = Path("docs/research/evidence/S21_complementary_rgb/preregistration.json")
FEATURES = Path("outputs/s28_rgb_features")
OUT = Path("outputs/s29_rgb_backbone_screen")
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


def completed(path: Path, plan: Path) -> None:
    state = json.loads((path / "COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(plan):
        raise ValueError(f"{path} completion plan differs")
    for name, digest in state["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"{path}/{name} changed")


def freeze() -> None:
    parent = verify_plan(PARENT)
    completed(Path("outputs/s28_tta_head_seeds"), PARENT)
    if not json.loads(Path("outputs/s28_tta_head_seeds/hypothesis.json").read_text())["H46_screen"]:
        raise ValueError("S28 failed; the S27 head is not the retained system")
    inputs = dict(parent["input_hashes"])
    paths = [PARENT, Path(__file__), S21, Path("scripts/cache_rgb_features.py"),
             Path("src/spectralquadnet/experiments/residual_head.py"),
             Path("outputs/s28_tta_head_seeds/COMPLETED.json"), Path("outputs/s28_tta_head_seeds/hypothesis.json"),
             FEATURES / "provenance.json", Path("docs/research/evidence/S29_rgb_backbone_screen/code/extract_stage_version.py")]
    paths += [FEATURES / f"{arm}.npy" for arm in BACKBONES]
    paths += [HUB / f"checkpoints/{name}_pretrain.pth" for name, _ in BACKBONES.values()]
    paths += [Path(f"outputs/s21_rgb_study/probabilities_f{f}.npz") for f in (0, 1)]
    inputs.update({str(p): sha256(p) for p in paths})
    s27 = json.loads(PARENT.read_text())
    spec = {"study": "S29_rgb_backbone_screen", "frozen_at": "2026-10-05", "parent_plan": str(PARENT),
            "scope": "Frozen-feature development screen on reused acquisitions; S21 corrected folds; S22 seed0 v5 encoders; head seed 0. No encoder training.",
            "input_hashes": inputs, "partition_plan": str(S21), "temperatures": json.loads(S21.read_text())["temperatures"],
            "model": s27["model"], "training": s27["training"],
            "arms": {"rgb": ["dino_s", "dino_b", "dino_l"], "anchor": ["equal_s", "equal_b", "equal_l"],
                     "system": ["head_s (saved S27 seed0 head, reused)", "head_b", "head_l"]},
            "probe": "S21 fit_probe on outer-training rows (train-only scaling + shrinkage LDA); calib log-loss temperature on the S21 grid.",
            "head": "S27 recipe/architecture via the shared helper, head seed 0; rgb_dim = backbone width (768/1024); HSI scales from S23, RGB scales fitted on outer-training rows; equal-TTA anchor built with the backbone's own probe/temperature. Train anchor: S27 CPU TTA cache; calib/held-out: saved S22 TTA.",
            "candidate_selection": "Before held-out scoring: candidate = argmax over {dino_b, dino_l} of the two-fold mean calib macro-F1 of its selected head; tie -> smaller. Written to selection.json first.",
            "replay_audit": "dino_s probabilities reproduce S21 (<=1e-6, no argmax change); equal_s reproduces S22 v5_rgb_equal; head_s reproduces S27 predictions; any mismatch stops.",
            "H47_screen": "System level: head_candidate minus head_s: mean F1 >= .01, paired variety CI lower > 0, positive F1 on each fold, mean cross-recall delta >= 0. Transfer support needs cross CI lower > 0. Anchor-level (equal_c - equal_s) and RGB-only contrasts are descriptive.",
            "decision": "Pass: candidate backbone joins the selected system (TTA-trained head); its head-seed and encoder confirmation join the final allocation. Fail: keep DINOv2-S; RGB backbone capacity is not the binding constraint.",
            "stop": "Two backbones, four head fits, one probe recipe; no sweep, fine-tuning, resolution change or new fusion form."}
    for p in (PLAN, Path("docs/research/evidence/S29_rgb_backbone_screen/preregistration.json")):
        if p.exists():
            raise FileExistsError("S29 already frozen")
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
    y = np.load("dataset_u430k32/labels.npy")
    groups = np.load("dataset_u430k32/groups.npy")
    scans = pd.read_csv("dataset_u430k32/scan_table.csv").set_index("scan_id")
    session = scans.session_id.loc[groups].to_numpy()
    cross = scans.groupby("label").session_id.nunique().sort_index().to_numpy() > 1
    splits = json.loads(S21.read_text())["splits"]
    features = {"dino_s": np.load("outputs/s20_rgb_features_v3/rgb.npy"),
                **{arm: np.load(FEATURES / f"{arm}.npy") for arm in BACKBONES}}
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    s23 = json.loads(Path("outputs/s23_frozen_multimodal/selection.json").read_text())
    held: dict[int, dict[str, Any]] = {}
    calib_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for fold in (0, 1):
        split = {k: np.asarray(v, dtype=int) for k, v in splits[str(fold)].items()}
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz") as c:
            tr, ca, htr, hca = c["train"], c["calib"], c["hsi_train"], c["hsi_calib"]
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz") as c:
            te, hte = c["rows"], c["hsi"]
        if set(tr) != set(split["train"]) or set(ca) != set(split["calib"]) or set(te) != set(np.r_[split["val"], split["test"]]):
            raise ValueError("Cached rows differ from the S21 partition")
        if not np.array_equal(te, np.sort(te)):
            raise ValueError("Held-out cache is not sorted")
        with np.load(f"outputs/s27_tta_cache/tta_train_f{fold}.npz") as t:
            if not np.array_equal(t["rows"], tr):
                raise ValueError("Train TTA rows differ")
            ztr = t["logits"].astype(np.float64)
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        hsi_p = {"train": softmax(ztr / t_hsi, axis=1),
                 "calib": softmax(aligned_logits(root / "logits_calib_tta.npz", ca, y) / t_hsi, axis=1),
                 "te": softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)}
        with np.load(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz") as s:
            hm, hs = s["hsi_mean"], s["hsi_scale"]
        hsi_x = {"train": (htr - hm) / hs, "calib": (hca - hm) / hs, "te": (hte - hm) / hs}
        block: dict[str, Any] = {"te": te, "train_rows": np.r_[tr, ca], "probs": {"hsi_tta": hsi_p["te"]}, "heads": {}, "inputs": {}}
        for arm, x in features.items():
            export = OUT / f"probe_f{fold}_{arm}.npz"
            cal, logits = fit_probe(x, y, tr, ca, te, export=export)
            temp = calibrate_temperature(cal, y[ca], spec["temperatures"])
            if arm == "dino_s" and temp != next(v["rgb_temperature"] for v in s23 if v["fold"] == fold):
                raise ValueError("dino_s temperature differs from the S23/S27 system")
            rgb_p = {"train": softmax(probe_logits(export, x[tr]) / temp, axis=1),
                     "calib": softmax(cal / temp, axis=1), "te": softmax(logits / temp, axis=1)}
            anchor = {k: (hsi_p[k] + rgb_p[k]) / 2 for k in hsi_p}
            block["probs"][arm] = rgb_p["te"]
            block["probs"]["equal_" + arm[-1]] = anchor["te"]
            rm, rs = x[tr].mean(0), np.maximum(x[tr].std(0), 1e-6)
            rows = {"train": tr, "calib": ca, "te": te}
            tensors = {k: [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in
                           (hsi_x[k], (x[rows[k]] - rm) / rs, anchor[k])] for k in rows}
            if arm == "dino_s":
                # The selected system: S27's own scales (identical recipe) and saved seed0 head.
                with np.load(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz") as s:
                    tensors = {k: [tensors[k][0], torch.from_numpy(np.asarray((x[rows[k]] - s["rgb_mean"]) / s["rgb_scale"], dtype=np.float32)),
                                   tensors[k][2]] for k in rows}
                head = FrozenResidualFusion(**spec["model"])
                head.load_state_dict(torch.load(S27 / f"head_f{fold}.pth", map_location="cpu", weights_only=True)["head"])
                head.eval()
                calib_f1 = next(v["calib_f1"] for v in json.loads((S27 / "selection.json").read_text()) if v["fold"] == fold)
                sel = {"head_seed": 0, "reused": f"S27 head_f{fold}.pth", "calib_f1": calib_f1}
            else:
                head, sel, trace = fit_residual_head(tensors["train"], tensors["calib"], y[tr], y[ca],
                                                     {**spec["model"], "rgb_dim": x.shape[1]}, spec["training"], 0)
                traces += [{"fold": fold, "arm": arm, **t} for t in trace]
                torch.save({"head": head.state_dict(), "selection": sel, "plan_sha256": sha256(PLAN)}, OUT / f"head_f{fold}_{arm}.pth")
            block["heads"][arm] = head
            block["inputs"][arm] = tensors["te"]
            calib_rows.append({"fold": fold, "arm": arm, "dimensions": x.shape[1], "rgb_temperature": temp,
                               "rgb_calib_f1": float(class_metrics(y[ca], cal.argmax(1), 90)["f1"].mean()),
                               "anchor_calib_f1": float(class_metrics(y[ca], anchor["calib"].argmax(1), 90)["f1"].mean()),
                               "head": sel})
        held[fold] = block
        print("Fitted probes/heads fold", fold, flush=True)
    calib = pd.DataFrame([{"fold": r["fold"], "arm": r["arm"], "head_calib_f1": r["head"]["calib_f1"]} for r in calib_rows])
    choice = calib[calib.arm != "dino_s"].groupby("arm").head_calib_f1.mean()
    candidate = str(choice.idxmax()) if choice.nunique() > 1 else "dino_b"
    write_json(OUT / "selection.json", {"candidate": candidate, "calib": calib_rows, "rule": spec["candidate_selection"],
                                        "selected_before_held_out_scoring": True})
    pd.DataFrame(traces).to_csv(OUT / "learning_curves.csv", index=False)
    print("Calib-selected candidate", candidate, choice.to_dict(), flush=True)
    # Held-out scoring of the frozen arm list, once.
    s22 = pd.read_csv("outputs/s22_fusion_analysis/predictions.csv.gz")
    s27 = pd.read_csv(S27 / "predictions.csv.gz")
    metrics: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    audits = []
    for fold, block in held.items():
        te, probs = block["te"], block["probs"]
        for arm, head in block["heads"].items():
            with torch.no_grad():
                z, _ = head(*block["inputs"][arm])
            probs["head_" + arm[-1]] = z.numpy()
        with np.load(f"outputs/s21_rgb_study/probabilities_f{fold}.npz") as old:
            if not np.array_equal(old["indices"], te):
                raise ValueError("S21 rows differ")
            diff = float(np.abs(old["dino_rgb"] - probs["dino_s"]).max())
            flips = int(np.sum(old["dino_rgb"].argmax(1) != probs["dino_s"].argmax(1)))
        def ref(df: pd.DataFrame, arm: str, fold: int = fold, te: npt.NDArray[Any] = te) -> npt.NDArray[Any]:
            return np.asarray(df[(df.fold == fold) & (df.arm == arm)].set_index("index").loc[te].prediction.to_numpy())
        audit = {"fold": fold, "dino_s_max_prob_difference_vs_s21": diff, "dino_s_vs_s21_disagreements": flips,
                 "equal_s_vs_s22_disagreements": int(np.sum(probs["equal_s"].argmax(1) != ref(s22, "v5_rgb_equal"))),
                 "head_s_vs_s27_disagreements": int(np.sum(probs["head_s"].argmax(1) != ref(s27, "tta_trained_residual")))}
        audits.append(audit)
        if diff > 1e-6 or flips or audit["equal_s_vs_s22_disagreements"] or audit["head_s_vs_s27_disagreements"]:
            raise ValueError(f"Replay audit failed: {audit}")
        trained_in = {c: set(session[block["train_rows"]][y[block["train_rows"]] == c]) for c in range(90)}
        for arm in ("hsi_tta", "dino_s", "dino_b", "dino_l", "equal_s", "equal_b", "equal_l", "head_s", "head_b", "head_l"):
            pred = probs[arm].argmax(1)
            stat = class_metrics(y[te], pred, 90)
            wrong = np.flatnonzero((pred != y[te]) & cross[y[te]])
            metrics.append({"fold": fold, "arm": arm, "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(pred == y[te])),
                            "same_recall": float(stat["recall"][~cross].mean()), "cross_recall": float(stat["recall"][cross].mean()),
                            "cross_error_session_attraction": float(np.mean([session[te][i] in trained_in[pred[i]] for i in wrong]))})
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
    c = candidate[-1]
    primary = contrast("head_" + c, "head_s")
    gate = bool(primary["delta_f1"] >= .01 and primary["f1_ci"][0] > 0 and min(primary["fold_f1_deltas"]) > 0
                and primary["delta_cross"] >= 0)
    descriptive = {f"{a}_vs_{b}": contrast(a, b) for a, b in
                   (("head_b", "head_s"), ("head_l", "head_s"), ("equal_b", "equal_s"), ("equal_l", "equal_s"),
                    ("dino_b", "dino_s"), ("dino_l", "dino_s"), ("head_" + c, "equal_" + c))}
    write_json(OUT / "hypothesis.json", {"H47_screen": gate, "candidate": candidate, "primary": primary,
               "transfer_gain_supported": bool(primary["cross_ci"][0] > 0), "descriptive": descriptive,
               "replay_audit": audits, "new_head_fits": 4, "network_training_runs": 0, "scope": spec["scope"]})
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
