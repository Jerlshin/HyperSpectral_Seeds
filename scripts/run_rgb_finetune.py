#!/usr/bin/env python
"""S32: how much does a properly trained RGB branch extract beyond frozen DINOv2 ViT-L?

Stages:
``profile``  label-free throughput/memory on synthetic crops (no data row is read);
``freeze``   pins the plan (recipes, cells, reference readout from S31, gates, inputs);
``train``    one frozen cell (backbone x fold, seed 0): fine-tune on outer-training rows, choose
             the epoch on calib, export 4-view logits and identity embeddings for all rows;
``run``      calib-only candidate choice over backbones, then one held-out scoring of the frozen
             arm list (RGB-only, equal fusion with the saved S22 v5 TTA, frozen references).
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
from spectralquadnet.experiments.rgb_finetune import Recipe, finetune, macro_f1, predict
from spectralquadnet.experiments.rgb_probe import aligned_logits, calibrate_temperature, verify_plan
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, gate, score, summarise
from spectralquadnet.models.rgb_branch import RGBBranch, load_dinov2

Array = npt.NDArray[Any]
PLAN = Path("configs/research/s32_rgb_finetune.json")
EVIDENCE = Path("docs/research/evidence/S32_rgb_finetune")
S31_PLAN = Path("configs/research/s31_rgb_readout_audit.json")
S31_OUT = Path("outputs/s31_rgb_readout_audit")
S29_PRED = Path("docs/research/evidence/S29_rgb_backbone_screen/screen_results/predictions.csv.gz")
OUT = Path("outputs/s32_rgb_finetune")
RGB = Path("dataset_rgb_hsi_v3/rgb.npy")
# (checkpoint, layer decay, trainable top blocks): ViT-B is fully fine-tuned; ViT-L full fine-tuning exceeds
# the 16 GB local memory (profiled), so its top 8 of 24 blocks are tuned (the frozen stem's rates would be
# <= 3e-5 * 0.9**8 anyway under layer decay).
BACKBONES = {"vitb": ("dinov2_vitb14", 0.8, 0), "vitl": ("dinov2_vitl14", 0.9, 8)}
SEED = 0


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def recipe(backbone: str) -> Recipe:
    return Recipe(layer_decay=BACKBONES[backbone][1], train_blocks=BACKBONES[backbone][2])


def profile(backbone: str, steps: int = 6) -> None:
    """Synthetic crops only: measures training images/s and that one step fits in memory."""
    dev, r = device(), recipe(backbone)
    model = RGBBranch(load_dinov2(BACKBONES[backbone][0], r.drop_path), 90, r.keep_tokens, train_blocks=r.train_blocks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-5)
    x = torch.zeros(r.batch, 3, 224, 224, device=dev)
    x[:, :, 20:204, 85:139] = 0.5
    target = torch.zeros(r.batch, dtype=torch.long, device=dev)
    for i in range(steps):
        if i == 2:
            torch.mps.synchronize() if dev.type == "mps" else None
            start = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x)[0].float(), target)
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    torch.mps.synchronize() if dev.type == "mps" else None
    rate = (steps - 2) * r.batch / (time.perf_counter() - start)
    print(json.dumps({"backbone": backbone, "device": str(dev), "train_images_per_s": rate,
                      "epoch_minutes_3681_rows": 3681 / rate / 60}), flush=True)


def freeze() -> None:
    s31 = verify_plan(S31_PLAN)
    if (S31_OUT / "STARTED.json").exists():
        raise ValueError("Freeze S32 before S31 scoring so its design cannot depend on S31's outcome")
    paths = [Path(__file__), S31_PLAN, S29_PRED, RGB,
             Path("src/spectralquadnet/models/rgb_branch.py"), Path("src/spectralquadnet/experiments/rgb_finetune.py"),
             Path("src/spectralquadnet/experiments/rgb_readout.py"), Path("src/spectralquadnet/experiments/screen_metrics.py"),
             Path("dataset_u430k32/labels.npy"), Path("dataset_u430k32/groups.npy"), Path("dataset_u430k32/scan_table.csv"),
             Path("outputs/s22_fusion_analysis/calibration.json")]
    paths += [Path.home() / f".cache/torch/hub/checkpoints/{name}_pretrain.pth" for name, _, _ in BACKBONES.values()]
    for fold in (0, 1):
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        paths += [root / "logits_calib_tta.npz", root / "logits_val_test_tta.npz"]
    spec = {"study": "S32_rgb_finetune", "frozen_at": "2026-10-05", "parent_plan": str(S31_PLAN),
            "scope": "Trained RGB branch development screen on reused acquisitions; S21 corrected folds; seed 0; both folds. "
                     "Fusion uses the saved S22 seed-0 v5 TTA. No HSI training.",
            "input_hashes": {str(p): sha256(p) for p in paths},
            "temperatures": s31["temperatures"],
            "cells": [{"arm": b, "backbone": b, "fold": f, "seed": SEED} for b in BACKBONES for f in (0, 1)],
            "model": {b: {"checkpoint": name, "readout": "[class, foreground-weighted mean] of final tokens",
                          "foreground_tokens": recipe(b).keep_tokens, "trainable_top_blocks": top or "all"}
                      for b, (name, _, top) in BACKBONES.items()},
            "recipes": {b: recipe(b).as_dict() for b in BACKBONES},
            "selection_within_cell": "Epoch with best identity-view calib macro-F1 (first on ties). Temperature: calib log-loss of the 4-view mean logits on the S21 grid.",
            "candidate_selection": "Before held-out scoring: candidate = argmax over backbones of two-fold mean calib macro-F1 of the 4-view mean logits; tie -> vitb. Written to selection.json first.",
            "frozen_reference": "Rule fixed before S31 scoring: S31's calib-selected readout if H48-screen passes, else 'cls' "
                                "(= S29 ViT-L probe). Resolved at scoring from S31's sealed outputs (COMPLETED.json hashes verified).",
            "arms": {"rgb": ["ft_<b>_id", "ft_<b>_tta", "frozen_ref", "s29_dino_b", "s29_dino_l"],
                     "fusion": ["equal_ft_<b>", "equal_frozen_ref", "hsi_tta"]},
            "H50_screen": "RGB: ft_candidate_tta minus frozen_ref: mean F1 >= .02, paired variety CI lower > 0, positive on each fold, mean cross delta >= 0.",
            "H51_screen": "System: equal_ft_candidate minus equal_frozen_ref: mean F1 >= .01, CI lower > 0, positive on each fold, mean cross delta >= 0.",
            "descriptive": ["ft_vitb_tta - s29_dino_b (training effect at matched backbone)", "ft_vitl_tta - s29_dino_l",
                            "ft_<b>_tta - ft_<b>_id (orientation TTA)", "acquisition directions; session attraction",
                            "complementarity: either-modality oracle and both-wrong cross fraction vs HSI"],
            "decision": "H50+H51 pass: the trained RGB branch replaces frozen DINOv2 in the multimodal baseline and anchors the next "
                        "architecture study. H50 pass, H51 fail: RGB gains are redundant with HSI; the system keeps frozen RGB. "
                        "H50 fail: frozen ViT-L already extracts the usable RGB information under this recipe.",
            "stop": "Two backbones, one fixed recipe, one seed, both folds; no hyperparameter search. Multi-seed confirmation is reserved for the final system."}
    for p in (PLAN, EVIDENCE / "preregistration.json"):
        if p.exists():
            raise FileExistsError("S32 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def train(arm: str, fold: int, plan: Path = PLAN, out: Path = OUT) -> None:
    """One frozen cell of any plan with this schema (S32, and later arms such as S33's)."""
    spec = verify_plan(plan)
    cell_spec = next((c for c in spec["cells"] if c["arm"] == arm and c["fold"] == fold), None)
    if cell_spec is None:
        raise ValueError("Cell not in the frozen plan")
    backbone = cell_spec["backbone"]
    cell = out / f"{arm}_f{fold}"
    cell.mkdir(parents=True, exist_ok=False)
    r = Recipe(**spec["recipes"][arm])
    cohort = Cohort.load()
    split = cohort.splits[fold]
    images = np.load(RGB)  # 1.3 GB in memory
    dev = device()
    write_json(cell / "STARTED.json", {"plan_sha256": sha256(plan), "device": str(dev), "torch": torch.__version__})
    start = time.perf_counter()
    model = RGBBranch(load_dinov2(BACKBONES[backbone][0], r.drop_path), 90, r.keep_tokens, train_blocks=r.train_blocks)
    log = cell / "trace.jsonl"

    def write(row: dict[str, Any]) -> None:
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(arm, fold, row, flush=True)

    model, best, trace = finetune(model, images, cohort.y, split["train"], split["calib"], r, cell_spec["seed"], dev, write)
    train_seconds = time.perf_counter() - start
    torch.save({"model": model.state_dict(), "best": best, "plan_sha256": sha256(plan)}, cell / "model.pth")
    rows = np.arange(len(images))
    logits, embeddings = predict(model, images, rows, dev, r)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    np.savez_compressed(cell / "outputs.npz", rows=rows, logits=logits.astype(np.float32),
                        embeddings=embeddings.astype(np.float32))
    cal = logits[:, split["calib"]].mean(0)
    temperature = calibrate_temperature(cal, cohort.y[split["calib"]], spec["temperatures"])
    write_json(cell / "selection.json", {"best": best, "temperature": temperature,
                                         "calib_f1_tta": macro_f1(cohort.y[split["calib"]], cal),
                                         "calib_f1_id": macro_f1(cohort.y[split["calib"]], logits[0, split["calib"]]),
                                         "train_seconds": train_seconds, "total_seconds": time.perf_counter() - start})
    write_json(cell / "COMPLETED.json", {"plan_sha256": sha256(plan),
               "files": {p.name: sha256(p) for p in cell.iterdir() if p.is_file()}})


def run() -> None:
    spec = verify_plan(PLAN)
    for cell in spec["cells"]:
        state = json.loads((OUT / f"{cell['arm']}_f{cell['fold']}" / "COMPLETED.json").read_text())
        if state["plan_sha256"] != sha256(PLAN):
            raise ValueError("Cell trained under another plan")
    report = OUT / "screen"
    report.mkdir(exist_ok=False)
    cohort = Cohort.load()
    y = cohort.y
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    s31_state = json.loads((S31_OUT / "COMPLETED.json").read_text())
    if s31_state["plan_sha256"] != sha256(S31_PLAN) or any(
            sha256(S31_OUT / name) != digest for name, digest in s31_state["files"].items()):
        raise ValueError("S31 outputs changed since sealing")
    hyp31 = json.loads((S31_OUT / "hypothesis.json").read_text())
    reference = hyp31["candidate"] if hyp31["H48_screen"] else "cls"
    s31 = pd.read_csv(S31_OUT / "predictions.csv.gz")
    s29 = pd.read_csv(S29_PRED)
    sel = {b: [json.loads((OUT / f"{b}_f{f}" / "selection.json").read_text()) for f in (0, 1)] for b in BACKBONES}
    choice = {b: float(np.mean([s["calib_f1_tta"] for s in v])) for b, v in sel.items()}
    candidate = max(BACKBONES, key=lambda b: (choice[b], b == "vitb"))
    write_json(report / "selection.json", {"candidate": candidate, "two_fold_calib_f1_tta": choice, "cells": sel,
                                           "rule": spec["candidate_selection"], "selected_before_held_out_scoring": True})
    metrics, classes, predictions, complementarity = [], [], [], []
    for fold in (0, 1):
        te = cohort.heldout(fold)
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0/results")
        t_hsi = next(v["temperature"] for v in temps if v["fold"] == fold)
        hsi = softmax(aligned_logits(root / "logits_val_test_tta.npz", te, y) / t_hsi, axis=1)
        probs: dict[str, Array] = {"hsi_tta": hsi}
        with np.load(S31_OUT / f"probabilities_f{fold}.npz") as saved:
            if not np.array_equal(saved["rows"], te):
                raise ValueError("S31 held-out rows differ")
            probs["frozen_ref"] = saved[reference].astype(np.float64)
        ref31 = s31[(s31.fold == fold) & (s31.arm == reference)].set_index("index").prediction.loc[te].to_numpy()
        if np.any(probs["frozen_ref"].argmax(1) != ref31):
            raise ValueError("Frozen reference does not replay S31")
        probs["equal_frozen_ref"] = (hsi + probs["frozen_ref"]) / 2
        for b in BACKBONES:
            with np.load(OUT / f"{b}_f{fold}" / "outputs.npz") as o:
                logits = o["logits"][:, te]
            t = sel[b][fold]["temperature"]
            probs[f"ft_{b}_id"] = softmax(logits[0] / t, axis=1)
            probs[f"ft_{b}_tta"] = softmax(logits.mean(0) / t, axis=1)
            probs[f"equal_ft_{b}"] = (hsi + probs[f"ft_{b}_tta"]) / 2
            wrong_h, wrong_r = hsi.argmax(1) != y[te], probs[f"ft_{b}_tta"].argmax(1) != y[te]
            bridge = cohort.cross[y[te]]
            complementarity.append({"fold": fold, "rgb": f"ft_{b}_tta", "either_oracle_accuracy": float(np.mean(~wrong_h | ~wrong_r)),
                                    "both_wrong_cross_fraction": float(np.mean(wrong_h[bridge] & wrong_r[bridge])),
                                    "both_wrong_same_fraction": float(np.mean(wrong_h[~bridge] & wrong_r[~bridge]))})
        for arm in ("dino_b", "dino_l"):
            pred = s29[(s29.fold == fold) & (s29.arm == arm)].set_index("index").prediction.loc[te].to_numpy()
            probs["s29_" + arm] = np.eye(90)[pred]
        for arm, prob in probs.items():
            m, c, p = score(cohort, fold, arm, te, prob)
            metrics.append(m)
            classes += c
            predictions += p
    per_class = pd.DataFrame(classes)
    h50 = contrast(per_class, f"ft_{candidate}_tta", "frozen_ref")
    h51 = contrast(per_class, f"equal_ft_{candidate}", "equal_frozen_ref")
    descriptive = {f"{a}_vs_{b}": contrast(per_class, a, b) for a, b in
                   (("ft_vitb_tta", "s29_dino_b"), ("ft_vitl_tta", "s29_dino_l"), ("ft_vitb_tta", "ft_vitb_id"),
                    ("ft_vitl_tta", "ft_vitl_id"), ("ft_vitl_tta", "ft_vitb_tta"), ("equal_ft_vitb", "equal_frozen_ref"),
                    ("equal_ft_vitl", "equal_frozen_ref"), (f"equal_ft_{candidate}", "hsi_tta"),
                    (f"ft_{candidate}_tta", "hsi_tta"))}
    write_json(report / "hypothesis.json", {"H50_screen": gate(h50, .02), "H51_screen": gate(h51, .01), "candidate": candidate,
                                            "reference": reference, "H50": h50, "H51": h51, "descriptive": descriptive,
                                            "network_training_runs": len(spec["cells"]), "scope": spec["scope"]})
    pd.DataFrame(metrics).to_csv(report / "metrics.csv", index=False)
    per_class.to_csv(report / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(report / "predictions.csv.gz", index=False)
    pd.DataFrame(complementarity).to_csv(report / "complementarity.csv", index=False)
    summarise(pd.DataFrame(metrics), per_class).to_csv(report / "summary.csv")
    write_json(report / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in report.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "freeze", "train", "run"))
    parser.add_argument("--backbone", choices=tuple(BACKBONES), default="vitb", help="profile only")
    parser.add_argument("--arm", default="vitb", help="train: arm name in the plan's cells")
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--out", type=Path, default=OUT)
    a = parser.parse_args()
    if a.action == "profile":
        profile(a.backbone)
    elif a.action == "train":
        train(a.arm, a.fold, a.plan, a.out)
    else:
        {"freeze": freeze, "run": run}[a.action]()
