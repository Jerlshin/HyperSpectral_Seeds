#!/usr/bin/env python
"""S23: one small additive learned head, conditional on corrected-fold S22 fusion."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.special import softmax

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.pipelines import build_run_context
from spectralquadnet.engine.pipelines.context import RunContext
from spectralquadnet.experiments.frozen_rows import load_frozen_rows
from spectralquadnet.experiments.rgb_probe import (
    calibrate_temperature,
    class_metrics,
    cluster_interval,
    probe_logits,
    verify_plan,
)
from spectralquadnet.models.multimodal_residual import FrozenResidualFusion
from spectralquadnet.tracking.base import NullTracker
from spectralquadnet.utils.distributed import DistContext

S22 = Path("configs/research/s22_screening_amendment05.json")
PLAN = Path("configs/research/s23_frozen_multimodal.json")
OUT = Path("outputs/s23_frozen_multimodal")


def require_gate() -> None:
    complete = json.loads(Path("outputs/s22_fusion_analysis/COMPLETED.json").read_text())
    if complete["plan_sha256"] != sha256(S22):
        raise ValueError("S22 completion uses a different plan")
    for name, digest in complete["files"].items():
        if sha256(Path("outputs/s22_fusion_analysis") / name) != digest:
            raise ValueError("S22 analysis evidence changed")
    if not json.loads(Path("outputs/s22_fusion_analysis/hypothesis.json").read_text())["development_gate"]:
        raise ValueError("S22 development gate did not pass; do not launch S23 automatically")


def freeze() -> None:
    require_gate()
    parent = verify_plan(S22)
    inputs = dict(parent["input_hashes"])
    paths = [S22, Path("scripts/run_frozen_multimodal.py"),
             Path("src/spectralquadnet/models/multimodal_residual.py"),
             Path("outputs/s22_fusion_analysis/COMPLETED.json"),
             Path("outputs/s22_fusion_analysis/hypothesis.json"),
             Path("outputs/s20_rgb_features_v3/rgb.npy"), Path("outputs/s20_rgb_features_v3/rgb.json"),
             Path("outputs/s21_rgb_study/calibration.csv")]
    for fold in (0, 1):
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0")
        paths.extend([root / "best_stage1.pth", root / "provenance.json", root / "results/run.json",
                      root / "results/logits_calib_no_tta.npz", root / "results/logits_val_test_no_tta.npz",
                      Path(f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz")])
        if json.loads((root / "provenance.json").read_text())["training_plan_sha256"] != sha256(S22):
            raise ValueError("HSI checkpoint belongs to a different study")
    inputs.update({str(p): sha256(p) for p in paths})
    spec = {"study": "S23_frozen_multimodal", "frozen_at": "2026-10-05",
            "scope": "S22-informed architecture development on reused acquisitions; seed0, both directions",
            "partition_plan": parent["partition_plan"], "s22_plan": str(S22),
            "cells": [{"fold": f, "seed": 0} for f in (0, 1)], "input_hashes": inputs,
            "model": {"hsi_dim": 256, "rgb_dim": 384, "hidden": 32, "classes": 90, "dropout": .2},
            "training": {"epochs": 100, "patience": 15, "batch": 256, "lr": .001,
                         "weight_decay": .01, "residual_penalty": .1, "seed": 0},
            "features": "Selected S22 HSI live/EMA checkpoint, unaugmented single-view normalized embedding; frozen DINO RGB. Train-only feature scales; train-only head gradients; calib-only checkpoint/temperature.",
            "anchor": "Equal calibrated single-view HSI/RGB probabilities; readout initialized zero; additive per-modality residuals; no encoder tuning or attention.",
            "controls": ["hsi_single", "rgb", "equal_single", "learned_residual", "S22 TTA equal (external reference)"],
            "inference_audit": "Record CPU fp32 single-view logits against saved CUDA/fp16 single-view logits on identical rows. Calibration audit precedes head training; held-out audit follows checkpoint selection. Differences are reported, not used to tune or select the head.",
            "H41_screen": "Learned-minus-equal-single mean F1 >= .01, paired variety CI > 0, positive F1 each fold, mean cross delta >= 0. Also report against S22 TTA fusion; do not adopt if inferior in F1 or cross recall there.",
            "stop": "One fixed head at seed0 on both folds. No hidden-width/LR sweep, no automatic seeds1/2. Replicate finalists with matched encoder and head seeds before a paper claim."}
    for p in (PLAN, Path("docs/research/evidence/S23_frozen_multimodal/preregistration.json")):
        if p.exists():
            raise FileExistsError("S23 plan already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def extract(ctx: RunContext, model: torch.nn.Module, rows: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Frozen single-view features, exact masks and train-standardized morphology."""
    logits, embeddings = [], []
    assert ctx.store.masks is not None and ctx.morph is not None
    model.eval().requires_grad_(False)
    with torch.no_grad():
        for chunk in np.array_split(rows, max(1, (len(rows) + 63) // 64)):
            x = torch.from_numpy(np.array(ctx.store.require_patches()[chunk], dtype=np.float32))
            mask = torch.from_numpy(np.array(ctx.store.masks[chunk], dtype=np.float32))
            morph = torch.from_numpy(np.asarray(ctx.morph[chunk]))
            z, h = model(x, return_embed=True, mask=mask, morph=morph)
            logits.append(z.numpy())
            embeddings.append(h.numpy())
    return np.concatenate(logits), np.concatenate(embeddings)


def audit_logits(path: Path, rows: npt.NDArray[Any], labels: npt.NDArray[Any], logits: npt.NDArray[Any]) -> dict[str, Any]:
    """Separate numerical inference changes from a learned-head gain."""
    with np.load(path) as saved:
        order = np.argsort(saved["rows"])
        reference_rows = saved["rows"][order]
        if not np.array_equal(reference_rows, np.sort(rows)):
            raise ValueError("Inference audit row identities differ")
        if not np.array_equal(saved["targets"][order], labels[reference_rows]):
            raise ValueError("Inference audit target identities differ")
        reference = saved["logits"][order].astype(np.float32)
    current = logits[np.argsort(rows)]
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite single-view inference logits")
    delta = np.abs(current - reference)
    return {"rows": len(rows), "source_sha256": sha256(path),
            "max_abs_logit_difference": float(delta.max()),
            "mean_abs_logit_difference": float(delta.mean()),
            "argmax_disagreements": int(np.sum(current.argmax(1) != reference.argmax(1))),
            "scope": "CPU fp32 versus saved CUDA/fp16 single-view; diagnostic only"}


def run() -> None:
    spec = verify_plan(PLAN)
    require_gate()
    OUT.mkdir(exist_ok=False)
    write_json(OUT / "STARTED.json", {"plan_sha256": sha256(PLAN)})
    torch.set_num_threads(4)
    parent = json.loads(S22.read_text())
    rgb = np.load("outputs/s20_rgb_features_v3/rgb.npy")
    rgb_cal = pd.read_csv("outputs/s21_rgb_study/calibration.csv")
    labels = np.load("dataset_u430k32/labels.npy")
    scans = pd.read_csv("dataset_u430k32/scan_table.csv")
    cross = scans.groupby("label").session_id.nunique().to_numpy() > 1
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    deltas, cross_deltas = [], []
    for cell in spec["cells"]:
        fold = cell["fold"]
        torch.manual_seed(cell["seed"])
        root = Path(f"outputs/s22_complementary_v5/f{fold}_s0")
        cfg = load_experiment_config("experiment/seednet_full256", overrides=parent["v5_overrides"] +
            [f"data.split_fold={fold}", "seed=0", "device=cpu", "runtime=default",
             "runtime.amp_dtype=off", "runtime.multi_gpu=off", "runtime.num_workers=0",
             "runtime.eval_num_workers=0", "runtime.compile=off", "runtime.prewarm_cache=off", "tracking=none"])
        parts = load_frozen_rows(Path(spec["partition_plan"]), Path(cfg.data.labels_path),
                                 Path(cfg.data.groups_path), fold)
        ctx = build_run_context(cfg, NullTracker(), DistContext(device=torch.device("cpu")), split_override=parts)
        ck = load_ckpt(str(root / "best_stage1.pth"), ctx.model, ctx.ema, ctx.device)
        source = ck.get("best_source", "ema")
        if source not in ("live", "ema"):
            raise ValueError("Unsupported checkpoint source")
        encoder = ctx.model if source == "live" else ctx.ema.shadow
        train, calib = parts.train, parts.calib
        ztr, htr = extract(ctx, encoder, train)
        zca, hca = extract(ctx, encoder, calib)
        audits.append({"fold": fold, "split": "calib", **audit_logits(
            root / "results/logits_calib_no_tta.npz", calib, labels, zca)})
        temperature = calibrate_temperature(zca, labels[calib],
            json.loads(Path(spec["partition_plan"]).read_text())["temperatures"])
        rgb_temp = float(rgb_cal[(rgb_cal.fold == fold) & (rgb_cal.arm == "dino_rgb")].temperature.iloc[0])
        probe = Path(f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz")
        def anchor(z: npt.NDArray[Any], ids: npt.NDArray[Any], temperature: float = temperature, probe: Path = probe, rgb_temp: float = rgb_temp) -> npt.NDArray[Any]:
            return np.asarray((softmax(z / temperature, axis=1) +
                               softmax(probe_logits(probe, rgb[ids]) / rgb_temp, axis=1)) / 2, dtype=np.float32)
        # The two scales only see outer training rows; record them for inference.
        hm, hs = htr.mean(0), np.maximum(htr.std(0), 1e-6)
        rm, rs = rgb[train].mean(0), np.maximum(rgb[train].std(0), 1e-6)
        def tensors(h: npt.NDArray[Any], z: npt.NDArray[Any], ids: npt.NDArray[Any], hm: npt.NDArray[Any] = hm, hs: npt.NDArray[Any] = hs, rm: npt.NDArray[Any] = rm, rs: npt.NDArray[Any] = rs, anchor: Any = anchor) -> list[torch.Tensor]:
            return [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in
                    ((h - hm) / hs, (rgb[ids] - rm) / rs, anchor(z, ids))]
        ht, rt, at = tensors(htr, ztr, train)
        hc, rc, ac = tensors(hca, zca, calib)
        yt = torch.from_numpy(labels[train].astype(np.int64))
        torch.manual_seed(cell["seed"])
        head = FrozenResidualFusion(**spec["model"])
        recipe = spec["training"]
        optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
        # Anchor itself is eligible at epoch0, so calib can reject all corrections.
        best = float(class_metrics(labels[calib], ac.argmax(1).numpy(), 90)["f1"].mean())
        anchor_calib = best
        traces.append({"fold": fold, "epoch": 0, "calib_f1": best,
                       "train_accuracy": float((at.argmax(1) == yt).float().mean()),
                       "calib_accuracy": float(np.mean(ac.argmax(1).numpy() == labels[calib]))})
        state, chosen, stale = copy.deepcopy(head.state_dict()), 0, 0
        for epoch in range(1, recipe["epochs"] + 1):
            head.train()
            for start in range(0, len(train), recipe["batch"]):
                if start == 0:
                    order = torch.randperm(len(train))
                batch = order[start:start + recipe["batch"]]
                z, residual = head(ht[batch], rt[batch], at[batch])
                loss = torch.nn.functional.cross_entropy(z, yt[batch]) + recipe["residual_penalty"] * residual.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite learned-head loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                cz, _ = head(hc, rc, ac)
                tz, _ = head(ht, rt, at)
                score = float(class_metrics(labels[calib], cz.argmax(1).numpy(), 90)["f1"].mean())
            traces.append({"fold": fold, "epoch": epoch, "calib_f1": score,
                           "train_accuracy": float((tz.argmax(1) == yt).float().mean()),
                           "calib_accuracy": float(np.mean(cz.argmax(1).numpy() == labels[calib]))})
            if score > best:
                best, state, chosen, stale = score, copy.deepcopy(head.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= recipe["patience"]:
                break
        head.load_state_dict(state)
        head.eval()
        torch.save({"head": state, "plan_sha256": sha256(PLAN), "fold": fold,
                    "seed": cell["seed"], "selected_epoch": chosen, "encoder_source": source}, OUT / f"head_f{fold}.pth")
        np.savez(OUT / f"scales_f{fold}.npz", hsi_mean=hm, hsi_scale=hs, rgb_mean=rm, rgb_scale=rs, train_rows=train)
        exports.append({"fold": fold, "chosen_epoch": chosen, "calib_f1": best,
                        "anchor_calib_f1": anchor_calib,
                        "hsi_temperature": temperature, "rgb_temperature": rgb_temp,
                        "parameters": sum(p.numel() for p in head.parameters()), "encoder_source": source})
        write_json(OUT / f"selection_f{fold}.json", exports[-1])
        # Only after checkpoint selection do we extract/score the held-out bundle.
        te = np.sort(np.r_[parts.val, parts.test])
        zte, hte = extract(ctx, encoder, te)
        audits.append({"fold": fold, "split": "val_test", **audit_logits(
            root / "results/logits_val_test_no_tta.npz", te, labels, zte)})
        he, re, ae = tensors(hte, zte, te)
        with torch.no_grad():
            learned, _ = head(he, re, ae)
        arms = {"hsi_single": zte.argmax(1),
                "rgb": probe_logits(probe, rgb[te]).argmax(1),
                "equal_single": ae.argmax(1).numpy(), "learned_residual": learned.argmax(1).numpy()}
        paired = {}
        for arm, pred in arms.items():
            stat = class_metrics(labels[te], pred, 90)
            paired[arm] = stat
            metrics.append({"fold": fold, "seed": cell["seed"], "arm": arm,
                            "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(labels[te] == pred)),
                            "same_recall": float(stat["recall"][~cross].mean()),
                            "cross_recall": float(stat["recall"][cross].mean())})
            classes.extend({"fold": fold, "arm": arm, "label": i, "cross": bool(cross[i]),
                            "f1": float(stat["f1"][i]), "recall": float(stat["recall"][i])} for i in range(90))
            predictions.extend({"fold": fold, "arm": arm, "index": int(i), "target": int(t), "prediction": int(p)}
                               for i, t, p in zip(te, labels[te], pred, strict=True))
        deltas.append(paired["learned_residual"]["f1"] - paired["equal_single"]["f1"])
        cross_deltas.append((paired["learned_residual"]["recall"] - paired["equal_single"]["recall"])[cross])
        print("Completed head fold", fold, exports[-1], flush=True)
    df, dr = np.mean(deltas, axis=0), np.mean(cross_deltas, axis=0)
    ci = cluster_interval(df)
    s22 = pd.read_csv("outputs/s22_fusion_analysis/metrics.csv")
    learned_mean = np.mean([m["f1"] for m in metrics if m["arm"] == "learned_residual"])
    tta_mean = s22[s22.arm == "v5_rgb_equal"].f1.mean()
    learned_cross = np.mean([m["cross_recall"] for m in metrics if m["arm"] == "learned_residual"])
    tta_cross = s22[s22.arm == "v5_rgb_equal"].cross_recall.mean()
    gate = bool(df.mean() >= .01 and ci[0] > 0 and min(v.mean() for v in deltas) > 0 and dr.mean() >= 0)
    write_json(OUT / "hypothesis.json", {"H41_screen": gate, "delta_f1": float(df.mean()), "ci": ci,
               "fold_f1_deltas": [float(v.mean()) for v in deltas], "delta_cross": float(dr.mean()),
               "cross_ci": cluster_interval(dr), "delta_vs_s22_tta": float(learned_mean - tta_mean),
               "delta_cross_vs_s22_tta": float(learned_cross - tta_cross),
               "adoption_gate": bool(gate and learned_mean >= tta_mean and learned_cross >= tta_cross),
               "scope": "One learned head seed, fixed seed0 HSI encoders, both corrected folds; same acquisitions."})
    for name, records in (("metrics", metrics), ("per_class", classes), ("learning_curves", traces)):
        pd.DataFrame(records).to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    write_json(OUT / "selection.json", exports)
    write_json(OUT / "inference_audit.json", audits)
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
        "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
