#!/usr/bin/env python
"""S24: bounded learned-correction branch removal; no new encoder fits."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.special import softmax

from run_frozen_multimodal import extract
from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.pipelines import build_run_context
from spectralquadnet.experiments.frozen_rows import load_frozen_rows
from spectralquadnet.experiments.rgb_probe import (
    class_metrics,
    cluster_interval,
    probe_logits,
    verify_plan,
)
from spectralquadnet.models.branch_residual import FrozenBranchCorrection
from spectralquadnet.tracking.base import NullTracker
from spectralquadnet.utils.distributed import DistContext

PARENT = Path("configs/research/s23_frozen_multimodal.json")
PLAN = Path("configs/research/s24_branch_multimodal.json")
OUT = Path("outputs/s24_branch_multimodal")


def require_parent() -> dict[str, Any]:
    parent = verify_plan(PARENT)
    complete = json.loads(Path("outputs/s23_frozen_multimodal/COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(PARENT):
        raise ValueError("S23 completion plan differs")
    for name, digest in complete["files"].items():
        if sha256(Path("outputs/s23_frozen_multimodal") / name) != digest:
            raise ValueError("S23 artifact changed")
    if not json.loads(Path("outputs/s23_frozen_multimodal/hypothesis.json").read_text())["adoption_gate"]:
        raise ValueError("S23 did not pass its screen/practical gate")
    return parent


def freeze() -> None:
    parent = require_parent()
    inputs = dict(parent["input_hashes"])
    paths = [PARENT, Path(__file__), Path("src/spectralquadnet/models/branch_residual.py"),
             Path("outputs/s23_frozen_multimodal/COMPLETED.json"),
             Path("outputs/s23_frozen_multimodal/hypothesis.json"),
             Path("outputs/s23_frozen_multimodal/selection.json"),
             Path("outputs/s23_frozen_multimodal/per_class.csv"),
             Path("outputs/s23_frozen_multimodal/predictions.csv.gz")]
    paths += [Path(f"outputs/s23_frozen_multimodal/scales_f{f}.npz") for f in (0, 1)]
    # Persist relative paths, as with the preceding frozen plans.
    inputs.update({str(p.relative_to(Path.cwd()) if p.is_absolute() else p): sha256(p) for p in paths})
    spec = {"study": "S24_branch_multimodal", "frozen_at": "2026-10-05",
            "scope": "S23-informed development ablation on reused acquisitions, not final validation",
            "parent_plan": str(PARENT), "partition_plan": parent["partition_plan"],
            "cells": [{"fold": f, "seed": 0} for f in (0, 1)],
            "arms": ["hsi_correction", "rgb_correction"], "input_hashes": inputs,
            "model": parent["model"], "training": parent["training"],
            "anchor": "Both HSI/RGB modalities remain in the fixed S23 equal single-view anchor. Only a learned feature-correction branch is removed; this is not a unimodal predictor.",
            "design": "Same initialization draw order, feature scales, temperatures, head recipe and epoch0 eligibility as S23. Encoders/probes frozen. One seed, both folds; no sweep. Cache frozen features for later cheap head work.",
            "H42_screen": "Both-feature correction is necessary only if it beats EACH removed-branch control by mean F1 >= .005, paired variety CI >0, positive F1 each fold, and nonnegative mean cross delta. These are branch-removal, not capacity-matched controls.",
            "simplification_gate": "Each branch-only candidate must meet S23's H41/practical criteria versus equal-single and S22 TTA. Also require mean F1 within .005 of S23 both-feature head and cross recall within .01. If several qualify, prefer fewer active trainable parameters. This provisional development choice requires eventual matched confirmation.",
            "stop": "Exactly four cheap head fits, two per corrected fold, with existing seed0 encoders. No new GPU training and no automatic extra seeds."}
    for p in (PLAN, Path("docs/research/evidence/S24_branch_multimodal/preregistration.json")):
        if p.exists():
            raise FileExistsError("S24 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def fit(mode: Literal["hsi", "rgb"], spec: dict[str, Any], fold: int,
        train: list[torch.Tensor], calib: list[torch.Tensor], yt: torch.Tensor,
        yc: npt.NDArray[Any]) -> tuple[FrozenBranchCorrection, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(0)
    head = FrozenBranchCorrection(mode=mode, **spec["model"])
    recipe = spec["training"]
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad),
        lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    best = float(class_metrics(yc, calib[2].argmax(1).numpy(), 90)["f1"].mean())
    anchor_calib = best
    state, selected, stale = copy.deepcopy(head.state_dict()), 0, 0
    traces = [{"fold": fold, "arm": mode + "_correction", "epoch": 0, "calib_f1": best,
               "train_accuracy": float((train[2].argmax(1) == yt).float().mean())}]
    for epoch in range(1, recipe["epochs"] + 1):
        head.train()
        order = torch.randperm(len(yt))
        for start in range(0, len(yt), recipe["batch"]):
            batch = order[start:start + recipe["batch"]]
            z, residual = head(*(v[batch] for v in train))
            loss = torch.nn.functional.cross_entropy(z, yt[batch]) + recipe["residual_penalty"] * residual.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite S24 loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            cz, _ = head(*calib)
            tz, _ = head(*train)
        score = float(class_metrics(yc, cz.argmax(1).numpy(), 90)["f1"].mean())
        traces.append({"fold": fold, "arm": mode + "_correction", "epoch": epoch,
                       "calib_f1": score, "train_accuracy": float((tz.argmax(1) == yt).float().mean())})
        if score > best:
            best, state, selected, stale = score, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= recipe["patience"]:
            break
    head.load_state_dict(state)
    head.eval()
    selection = {"fold": fold, "arm": mode + "_correction", "selected_epoch": selected,
                 "epochs_completed": epoch, "calib_f1": best, "anchor_calib_f1": anchor_calib,
                 "active_trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad)}
    torch.save({"state": state, "selection": selection, "plan_sha256": sha256(PLAN)}, OUT / f"head_f{fold}_{mode}.pth")
    return head, selection, traces


def run() -> None:
    spec = verify_plan(PLAN)
    parent = require_parent()
    OUT.mkdir(exist_ok=False)
    write_json(OUT / "STARTED.json", {"plan_sha256": sha256(PLAN)})
    torch.set_num_threads(4)
    s22 = json.loads(Path(parent["s22_plan"]).read_text())
    labels = np.load("dataset_u430k32/labels.npy")
    rgb = np.load("outputs/s20_rgb_features_v3/rgb.npy")
    cross = pd.read_csv("dataset_u430k32/scan_table.csv").groupby("label").session_id.nunique().to_numpy() > 1
    selected = json.loads(Path("outputs/s23_frozen_multimodal/selection.json").read_text())
    metrics: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for fold in (0, 1):
        cfg = load_experiment_config("experiment/seednet_full256", overrides=s22["v5_overrides"] +
            [f"data.split_fold={fold}", "seed=0", "device=cpu", "runtime=default", "runtime.amp_dtype=off",
             "runtime.multi_gpu=off", "runtime.num_workers=0", "runtime.eval_num_workers=0",
             "runtime.compile=off", "runtime.prewarm_cache=off", "tracking=none"])
        parts = load_frozen_rows(Path(spec["partition_plan"]), Path(cfg.data.labels_path), Path(cfg.data.groups_path), fold)
        ctx = build_run_context(cfg, NullTracker(), DistContext(device=torch.device("cpu")), split_override=parts)
        ck = load_ckpt(f"outputs/s22_complementary_v5/f{fold}_s0/best_stage1.pth", ctx.model, ctx.ema, ctx.device)
        prior = next(v for v in selected if v["fold"] == fold)
        if ck.get("best_source", "ema") != prior["encoder_source"]:
            raise ValueError("Selected encoder source changed")
        encoder = ctx.model if prior["encoder_source"] == "live" else ctx.ema.shadow
        tr, ca = parts.train, parts.calib
        ztr, htr = extract(ctx, encoder, tr)
        zca, hca = extract(ctx, encoder, ca)
        np.savez_compressed(OUT / f"features_f{fold}_traincal.npz", train=tr, calib=ca,
                            logits_train=ztr, hsi_train=htr, logits_calib=zca, hsi_calib=hca)
        with np.load(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz") as scale:
            if not np.array_equal(scale["train_rows"], tr):
                raise ValueError("Feature scales use different training rows")
            hm, hs, rm, rs = (scale[n] for n in ("hsi_mean", "hsi_scale", "rgb_mean", "rgb_scale"))
        probe = Path(f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz")
        def tensors(h: npt.NDArray[Any], z: npt.NDArray[Any], ids: npt.NDArray[Any],
                    hm: npt.NDArray[Any] = hm, hs: npt.NDArray[Any] = hs,
                    rm: npt.NDArray[Any] = rm, rs: npt.NDArray[Any] = rs,
                    probe: Path = probe, prior: dict[str, Any] = prior) -> list[torch.Tensor]:
            anchor = (softmax(z / prior["hsi_temperature"], axis=1) +
                      softmax(probe_logits(probe, rgb[ids]) / prior["rgb_temperature"], axis=1)) / 2
            if not np.isfinite(anchor).all():
                raise ValueError("Nonfinite frozen probability anchor")
            return [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in
                    ((h-hm)/hs, (rgb[ids]-rm)/rs, anchor)]
        train, calib = tensors(htr, ztr, tr), tensors(hca, zca, ca)
        fitted = {}
        for arm in spec["arms"]:
            mode = cast(Literal["hsi", "rgb"], arm.split("_")[0])
            head, choice, curve = fit(mode, spec, fold, train, calib,
                                     torch.from_numpy(labels[tr].astype(np.int64)), labels[ca])
            fitted[arm] = head
            selections.append(choice)
            traces.extend(curve)
        write_json(OUT / f"selection_f{fold}.json", [v for v in selections if v["fold"] == fold])
        # Both branch checkpoints are selected before this held-out extraction.
        te = np.sort(np.r_[parts.val, parts.test])
        zte, hte = extract(ctx, encoder, te)
        np.savez_compressed(OUT / f"features_f{fold}_heldout.npz", rows=te, logits=zte, hsi=hte)
        held = tensors(hte, zte, te)
        for arm, head in fitted.items():
            with torch.no_grad():
                scores, _ = head(*held)
            pred = scores.argmax(1).numpy()
            stat = class_metrics(labels[te], pred, 90)
            metrics.append({"fold": fold, "seed": 0, "arm": arm, "f1": float(stat["f1"].mean()),
                            "accuracy": float(np.mean(pred == labels[te])),
                            "same_recall": float(stat["recall"][~cross].mean()),
                            "cross_recall": float(stat["recall"][cross].mean())})
            classes.extend({"fold": fold, "arm": arm, "label": c, "cross": bool(cross[c]),
                            "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])} for c in range(90))
            predictions.extend({"fold": fold, "arm": arm, "index": int(i), "target": int(t), "prediction": int(p)}
                               for i, t, p in zip(te, labels[te], pred, strict=True))
        print("Completed S24 fold", fold, selections[-2:], flush=True)
    table = pd.DataFrame(classes)
    reference = pd.read_csv("outputs/s23_frozen_multimodal/per_class.csv")
    tta = pd.read_csv("outputs/s22_fusion_analysis/metrics.csv").query("arm == 'v5_rgb_equal'")
    gates, comparisons = {}, []
    for arm in spec["arms"]:
        a = table[table.arm == arm].set_index(["label", "fold"]).sort_index()
        blocks = {}
        for control in ("equal_single", "learned_residual"):
            b = reference[reference.arm == control].set_index(["label", "fold"]).sort_index()
            delta = (a.f1-b.f1).unstack("fold").to_numpy()
            cr = (a.recall-b.recall)[a.cross].unstack("fold").to_numpy()
            record = {"arm": arm, "control": control, "delta_f1": float(delta.mean()),
                      "f1_ci": cluster_interval(delta), "fold_f1_deltas": delta.mean(0).tolist(),
                      "delta_cross": float(cr.mean()), "cross_ci": cluster_interval(cr)}
            blocks[control] = record
            comparisons.append(record)
        anchor, both = blocks["equal_single"], blocks["learned_residual"]
        means = pd.DataFrame(metrics).query("arm == @arm")
        head_gate = bool(anchor["delta_f1"] >= .01 and anchor["f1_ci"][0] > 0 and
                         min(anchor["fold_f1_deltas"]) > 0 and anchor["delta_cross"] >= 0)
        practical = bool(head_gate and means.f1.mean() >= tta.f1.mean() and means.cross_recall.mean() >= tta.cross_recall.mean())
        gates[arm] = {"H41_style_screen": head_gate, "practical_gate": practical,
                      "simplification_gate": bool(practical and both["delta_f1"] >= -.005 and both["delta_cross"] >= -.01),
                      "both_better_than_this_branch": bool(-both["delta_f1"] >= .005 and both["f1_ci"][1] < 0 and
                                                          max(both["fold_f1_deltas"]) < 0 and both["delta_cross"] <= 0)}
    write_json(OUT / "hypothesis.json", {"H42_screen": all(v["both_better_than_this_branch"] for v in gates.values()),
               "branch_gates": gates, "comparisons": comparisons,
               "scope": spec["scope"], "new_encoder_fits": 0, "head_fits": 4})
    for name, records in (("metrics", metrics), ("per_class", classes), ("learning_curves", traces)):
        pd.DataFrame(records).to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    write_json(OUT / "selection.json", selections)
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
