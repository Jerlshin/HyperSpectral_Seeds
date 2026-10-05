#!/usr/bin/env python
"""S27: the S23 additive head trained against its intended equal-TTA anchor.

Stages, in order: ``profile`` (train-row TTA timing, CPU vs available MPS),
``cache`` (outer-training-only frozen TTA logits plus a calibration-row audit
against the saved CUDA TTA logits), ``freeze`` (pins every input including the
cache), ``run`` (head seed 0 on both folds; calib selection before held-out).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
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
from spectralquadnet.engine.tta import tta_predict
from spectralquadnet.experiments.frozen_rows import load_frozen_rows
from spectralquadnet.experiments.rgb_probe import (
    aligned_logits,
    class_metrics,
    cluster_interval,
    probe_logits,
    verify_plan,
)
from spectralquadnet.models.multimodal_residual import FrozenResidualFusion
from spectralquadnet.tracking.base import NullTracker
from spectralquadnet.utils.distributed import DistContext

PLAN = Path("configs/research/s27_tta_trained_head.json")
PARENT = Path("configs/research/s26_tta_anchor.json")
S22 = Path("configs/research/s22_screening_amendment05.json")
PROFILE = Path("outputs/s27_tta_profile")
CACHE = Path("outputs/s27_tta_cache")
OUT = Path("outputs/s27_tta_trained_head")


def completed(path: Path, plan: Path) -> None:
    state = json.loads((path / "COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(plan):
        raise ValueError(f"{path} completion plan differs")
    for name, digest in state["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"{path}/{name} changed after completion")


def encoder_for(fold: int) -> tuple[RunContext, torch.nn.Module, npt.NDArray[Any]]:
    """Selected S22 seed-0 encoder on CPU, exactly as S23/S24 built it."""
    s22 = json.loads(S22.read_text())
    cfg = load_experiment_config("experiment/seednet_full256", overrides=s22["v5_overrides"] +
        [f"data.split_fold={fold}", "seed=0", "device=cpu", "runtime=default", "runtime.amp_dtype=off",
         "runtime.multi_gpu=off", "runtime.num_workers=0", "runtime.eval_num_workers=0",
         "runtime.compile=off", "runtime.prewarm_cache=off", "tracking=none"])
    partition = Path(json.loads(Path("configs/research/s23_frozen_multimodal.json").read_text())["partition_plan"])
    parts = load_frozen_rows(partition, Path(cfg.data.labels_path), Path(cfg.data.groups_path), fold)
    ctx = build_run_context(cfg, NullTracker(), DistContext(device=torch.device("cpu")), split_override=parts)
    ck = load_ckpt(f"outputs/s22_complementary_v5/f{fold}_s0/best_stage1.pth", ctx.model, ctx.ema, ctx.device)
    chosen = next(v for v in json.loads(Path("outputs/s23_frozen_multimodal/selection.json").read_text())
                  if v["fold"] == fold)
    if ck.get("best_source", "ema") != chosen["encoder_source"]:
        raise ValueError("Selected encoder source changed")
    encoder = ctx.model if chosen["encoder_source"] == "live" else ctx.ema.shadow
    encoder.eval().requires_grad_(False)
    return ctx, encoder, np.asarray(parts.calib)


def tta_logits(ctx: RunContext, model: torch.nn.Module, rows: npt.NDArray[Any],
               device: torch.device, chunk: int = 64) -> npt.NDArray[Any]:
    """R1 TTA (8 dihedral + 4 foreground spectral-gain views), fp32, logits averaged."""
    assert ctx.store.masks is not None and ctx.morph is not None
    out = []
    for part in np.array_split(rows, max(1, (len(rows) + chunk - 1) // chunk)):
        x = torch.from_numpy(np.array(ctx.store.require_patches()[part], dtype=np.float32)).to(device)
        mask = torch.from_numpy(np.array(ctx.store.masks[part], dtype=np.float32)).to(device)
        morph = torch.from_numpy(np.asarray(ctx.morph[part], dtype=np.float32)).to(device)
        out.append(tta_predict(model, x, 8, 4, mask=mask, morph=morph).float().cpu().numpy())
    return np.concatenate(out)


def profile() -> None:
    """Time train-row TTA only; no calibration or held-out row is touched."""
    PROFILE.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ctx, encoder, _ = encoder_for(0)
    with np.load("outputs/s24_branch_multimodal/features_f0_traincal.npz") as cache:
        rows = cache["train"][:256]
    timings: dict[str, Any] = {}
    reference = None
    for name in ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",):
        device = torch.device(name)
        model = copy.deepcopy(encoder).to(device)
        tta_logits(ctx, model, rows[:64], device)  # warm-up, not timed
        start = time.perf_counter()
        logits = tta_logits(ctx, model, rows, device)
        elapsed = time.perf_counter() - start
        reference = logits if reference is None else reference
        timings[name] = {"rows": len(rows), "seconds": elapsed, "rows_per_second": len(rows) / elapsed,
                         "max_abs_vs_cpu": float(np.abs(logits - reference).max()),
                         "argmax_disagreements_vs_cpu": int(np.sum(logits.argmax(1) != reference.argmax(1)))}
    train_rows = sum(len(np.load(f"outputs/s24_branch_multimodal/features_f{f}_traincal.npz")["train"]) for f in (0, 1))
    calib_rows = 2 * 630
    write_json(PROFILE / "profile.json", {
        "scope": "Fold-0 outer-training rows only (first 256 of the S24 train cache), 12-view fp32 TTA. No calib/held-out rows.",
        "torch": torch.__version__, "threads": torch.get_num_threads(), "timings": timings,
        "projected_rows": {"train_both_folds": train_rows, "calib_audit_both_folds": calib_rows},
        "projected_seconds": {k: (train_rows + calib_rows) / v["rows_per_second"] for k, v in timings.items()},
        "note": "Projection = measured throughput x row count; wall time, not a billing estimate."})
    print(json.dumps(timings, indent=1), flush=True)


def cache() -> None:
    """Outer-training-only TTA logits, plus a calib audit against saved CUDA TTA."""
    receipt = json.loads((PROFILE / "profile.json").read_text())
    CACHE.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    labels = np.load("dataset_u430k32/labels.npy")
    audits = []
    start = time.perf_counter()
    for fold in (0, 1):
        ctx, encoder, calib = encoder_for(fold)
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz") as feats:
            train, cached_calib = feats["train"], feats["calib"]
        if not np.array_equal(np.sort(calib), np.sort(cached_calib)):
            raise ValueError("Calibration rows differ from the S24 cache")
        with np.load(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz") as scale:
            if not np.array_equal(scale["train_rows"], train):
                raise ValueError("Train rows differ from the S23 scale fit")
        t0 = time.perf_counter()
        z_train = tta_logits(ctx, encoder, train, torch.device("cpu"))
        t1 = time.perf_counter()
        z_calib = tta_logits(ctx, encoder, cached_calib, torch.device("cpu"))
        t2 = time.perf_counter()
        if not (np.isfinite(z_train).all() and np.isfinite(z_calib).all()):
            raise ValueError("Nonfinite TTA logits")
        saved = aligned_logits(Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_calib_tta.npz"),
                               cached_calib, labels)
        delta = np.abs(z_calib - saved)
        audits.append({"fold": fold, "split": "calib", "rows": len(cached_calib),
                       "max_abs_logit_difference": float(delta.max()),
                       "mean_abs_logit_difference": float(delta.mean()),
                       "argmax_disagreements": int(np.sum(z_calib.argmax(1) != saved.argmax(1))),
                       "train_rows": len(train), "train_seconds": t1 - t0, "calib_seconds": t2 - t1,
                       "train_tta_accuracy": float(np.mean(z_train.argmax(1) == labels[train])),
                       "scope": "CPU fp32 TTA versus saved CUDA fp32 TTA (float16-serialized); diagnostic only."})
        np.savez(CACHE / f"tta_train_f{fold}.npz", rows=train, logits=z_train.astype(np.float32),
                 targets=labels[train])
        print("Cached train TTA fold", fold, audits[-1], flush=True)
    write_json(CACHE / "calib_audit.json", {"runtime": "cpu", "profile": receipt["projected_seconds"],
               "total_seconds": time.perf_counter() - start, "folds": audits})
    write_json(CACHE / "COMPLETED.json", {"files": {p.name: sha256(p) for p in CACHE.iterdir() if p.is_file()}})


def freeze() -> None:
    parent = verify_plan(PARENT)
    completed(Path("outputs/s26_tta_anchor"), PARENT)
    state = json.loads((CACHE / "COMPLETED.json").read_text())
    for name, digest in state["files"].items():
        if sha256(CACHE / name) != digest:
            raise ValueError("S27 train-TTA cache changed")
    audit = json.loads((CACHE / "calib_audit.json").read_text())
    if any(v["argmax_disagreements"] > 3 or v["max_abs_logit_difference"] > .05 for v in audit["folds"]):
        raise ValueError("CPU TTA does not reproduce the saved CUDA calibration TTA closely enough")
    inputs = dict(parent["input_hashes"])
    paths = [PARENT, Path(__file__), Path("outputs/s26_tta_anchor/COMPLETED.json"),
             Path("outputs/s23_frozen_multimodal/selection.json"), Path("outputs/s23_frozen_multimodal/per_class.csv"),
             Path("outputs/s23_frozen_multimodal/metrics.csv"),
             Path("outputs/s22_fusion_analysis/predictions.csv.gz"), PROFILE / "profile.json",
             CACHE / "COMPLETED.json", CACHE / "calib_audit.json"]
    for f in (0, 1):
        paths += [CACHE / f"tta_train_f{f}.npz", Path(f"outputs/s23_frozen_multimodal/scales_f{f}.npz"),
                  Path(f"outputs/s21_rgb_study/probabilities_f{f}.npz")]
    inputs.update({str(p): sha256(p) for p in paths})
    spec = {"study": "S27_tta_trained_head", "frozen_at": "2026-10-05", "parent_plan": str(PARENT),
            "scope": "Architecture-development screen on reused acquisitions; fixed S22 seed0 encoders, head seed0, both corrected folds.",
            "input_hashes": inputs, "model": parent["model"],
            "training": json.loads(Path("configs/research/s23_frozen_multimodal.json").read_text())["training"],
            "cells": [{"fold": f, "head_seed": 0, "encoder_seed": 0} for f in (0, 1)],
            "runtime": "Train-row TTA logits: CPU fp32, 12 views (8 dihedral + 4 foreground spectral gains), cached and hashed before freezing. Calib/held-out anchors use the saved S22 CUDA fp32 TTA logits (float16-serialized), i.e. exactly S22's equal-TTA comparator.",
            "anchor": "Equal mean of softmax(HSI TTA logits / S22 calib temperature) and softmax(S21 DINO-RGB probe / S23 RGB temperature). Train rows: cached CPU TTA. Calib/held-out: saved S22 TTA. Readout zero-initialized, so epoch 0 equals the anchor.",
            "features": "Unchanged S23 inputs: cached single-view HSI256 (S24 cache) and DINO RGB384, S23 train-only scales. No TTA feature averaging is claimed.",
            "selection": "Train-row gradients; calib macro-F1 selects the checkpoint, epoch 0 eligible. Held-out cache loaded only after both folds' checkpoints are saved. If both folds select epoch 0, no new predictor exists.",
            "replay_audit": "Before scoring the head, the held-out anchor argmax must exactly equal S22's saved v5_rgb_equal predictions; otherwise stop.",
            "controls": ["S22 equal TTA (matched anchor, primary)", "S23 single-anchor head seed0 (descriptive)", "S22 v5 TTA"],
            "H45_screen": "Learned minus matched equal-TTA: mean F1 >= .01, paired variety CI lower > 0, positive F1 gain on each fold, mean cross-recall delta >= 0. Transfer support additionally requires cross CI lower > 0.",
            "decision": "Pass: retain the TTA-trained head; next only cheap head seeds1/2 then matched encoder-seed confirmation in the final allocation. Fail: close the learned fusion-head line; fixed equal-TTA fusion stays the multimodal system; no rescue by recipe change.",
            "stop": "Exactly two head fits. No sweep, no new encoder fit, no automatic seeds."}
    for p in (PLAN, Path("docs/research/evidence/S27_tta_trained_head/preregistration.json")):
        if p.exists():
            raise FileExistsError("S27 already frozen")
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
    labels = np.load("dataset_u430k32/labels.npy")
    rgb = np.load("outputs/s20_rgb_features_v3/rgb.npy")
    cross = pd.read_csv("dataset_u430k32/scan_table.csv").groupby("label").session_id.nunique().to_numpy() > 1
    choices = json.loads(Path("outputs/s23_frozen_multimodal/selection.json").read_text())
    temps = json.loads(Path("outputs/s22_fusion_analysis/calibration.json").read_text())
    recipe = spec["training"]

    def inputs(fold: int, h: npt.NDArray[Any], z: npt.NDArray[Any], rows: npt.NDArray[Any]) -> list[torch.Tensor]:
        choice = next(v for v in choices if v["fold"] == fold)
        temp = next(v["temperature"] for v in temps if v["fold"] == fold)
        with np.load(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz") as s:
            hs, rs = (h - s["hsi_mean"]) / s["hsi_scale"], (rgb[rows] - s["rgb_mean"]) / s["rgb_scale"]
        rz = probe_logits(Path(f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz"), rgb[rows])
        anchor = (softmax(z / temp, axis=1) + softmax(rz / choice["rgb_temperature"], axis=1)) / 2
        if not all(np.isfinite(v).all() for v in (hs, rs, anchor)):
            raise ValueError("Nonfinite S27 head inputs")
        return [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in (hs, rs, anchor)]

    traces: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    heads = {}
    for cell in spec["cells"]:
        fold = cell["fold"]
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz") as c:
            tr, ca, htr, hca = c["train"], c["calib"], c["hsi_train"], c["hsi_calib"]
        with np.load(CACHE / f"tta_train_f{fold}.npz") as t:
            if not np.array_equal(t["rows"], tr):
                raise ValueError("Train TTA cache rows differ")
            ztr = t["logits"].astype(np.float64)
        zca = aligned_logits(Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_calib_tta.npz"), ca, labels)
        train, calib = inputs(fold, htr, ztr, tr), inputs(fold, hca, zca, ca)
        yt = torch.from_numpy(labels[tr].astype(np.int64))
        torch.manual_seed(cell["head_seed"])
        head = FrozenResidualFusion(**spec["model"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
        best = float(class_metrics(labels[ca], calib[2].argmax(1).numpy(), 90)["f1"].mean())
        anchor_calib = best
        traces.append({"fold": fold, "epoch": 0, "calib_f1": best,
                       "train_accuracy": float((train[2].argmax(1) == yt).float().mean())})
        state, chosen, stale, epoch = copy.deepcopy(head.state_dict()), 0, 0, 0
        for epoch in range(1, recipe["epochs"] + 1):
            head.train()
            order = torch.randperm(len(tr))
            for s in range(0, len(tr), recipe["batch"]):
                batch = order[s:s + recipe["batch"]]
                z, residual = head(*(v[batch] for v in train))
                loss = torch.nn.functional.cross_entropy(z, yt[batch]) + recipe["residual_penalty"] * residual.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite S27 loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                cz, _ = head(*calib)
                tz, _ = head(*train)
            score = float(class_metrics(labels[ca], cz.argmax(1).numpy(), 90)["f1"].mean())
            traces.append({"fold": fold, "epoch": epoch, "calib_f1": score,
                           "train_accuracy": float((tz.argmax(1) == yt).float().mean())})
            if score > best:
                best, state, chosen, stale = score, copy.deepcopy(head.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= recipe["patience"]:
                break
        head.load_state_dict(state)
        head.eval()
        heads[fold] = head
        prior = next(v for v in choices if v["fold"] == fold)
        selections.append({"fold": fold, "head_seed": cell["head_seed"], "selected_epoch": chosen,
                           "epochs_completed": epoch, "calib_f1": best, "anchor_calib_f1": anchor_calib,
                           "s23_single_anchor_head_calib_f1": prior["calib_f1"],
                           "s23_single_anchor_calib_f1": prior["anchor_calib_f1"]})
        torch.save({"head": state, "selection": selections[-1], "plan_sha256": sha256(PLAN)}, OUT / f"head_f{fold}.pth")
        write_json(OUT / f"selection_f{fold}.json", selections[-1])
        print("Selected", selections[-1], flush=True)
    write_json(OUT / "selection.json", selections)
    if all(v["selected_epoch"] == 0 for v in selections):
        write_json(OUT / "hypothesis.json", {"H45_screen": False, "held_out_rows_scored": 0,
                   "reason": "Calibration selected the anchor (epoch 0) on both folds; no new predictor."})
    else:
        # Only now: held-out cache, after both checkpoints are saved.
        s22pred = pd.read_csv("outputs/s22_fusion_analysis/predictions.csv.gz")
        metrics: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        replay = []
        for fold in (0, 1):
            with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz") as c:
                te, hte = c["rows"], c["hsi"]
            zte = aligned_logits(Path(f"outputs/s22_complementary_v5/f{fold}_s0/results/logits_val_test_tta.npz"), te, labels)
            held = inputs(fold, hte, zte, te)
            ref = s22pred[(s22pred.fold == fold) & (s22pred.arm == "v5_rgb_equal")].set_index("index").loc[te]
            mismatch = int(np.sum(held[2].argmax(1).numpy() != ref.prediction.to_numpy()))
            replay.append({"fold": fold, "anchor_vs_s22_equal_tta_disagreements": mismatch})
            if mismatch:
                raise ValueError("Held-out anchor does not reproduce S22 equal-TTA predictions")
            with torch.no_grad():
                z, _ = heads[fold](*held)
            for arm, pred in (("equal_tta", held[2].argmax(1).numpy()), ("tta_trained_residual", z.argmax(1).numpy())):
                stat = class_metrics(labels[te], pred, 90)
                metrics.append({"fold": fold, "head_seed": 0, "encoder_seed": 0, "arm": arm,
                                "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(pred == labels[te])),
                                "same_recall": float(stat["recall"][~cross].mean()),
                                "cross_recall": float(stat["recall"][cross].mean())})
                classes.extend({"fold": fold, "arm": arm, "label": c, "cross": bool(cross[c]),
                                "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])} for c in range(90))
                predictions.extend({"fold": fold, "arm": arm, "index": int(i), "target": int(t), "prediction": int(p)}
                                   for i, t, p in zip(te, labels[te], pred, strict=True))
        cls = pd.DataFrame(classes).set_index(["arm", "label", "fold"]).sort_index()
        a, b = cls.loc["tta_trained_residual"], cls.loc["equal_tta"]
        df = (a.f1 - b.f1).unstack("fold").to_numpy()
        dr = (a.recall - b.recall)[a.cross].unstack("fold").to_numpy()
        s23 = pd.read_csv("outputs/s23_frozen_multimodal/per_class.csv").query("arm == 'learned_residual'").set_index(["label", "fold"]).sort_index()
        d23 = (a.f1 - s23.f1).unstack("fold").to_numpy()
        ci, cr = cluster_interval(df), cluster_interval(dr)
        gate = bool(df.mean() >= .01 and ci[0] > 0 and min(df.mean(0)) > 0 and dr.mean() >= 0)
        write_json(OUT / "hypothesis.json", {"H45_screen": gate, "delta_f1_equal_tta": float(df.mean()), "f1_ci": ci,
                   "fold_f1_deltas": df.mean(0).tolist(), "delta_cross_equal_tta": float(dr.mean()), "cross_ci": cr,
                   "fold_cross_deltas": dr.mean(0).tolist(), "transfer_gain_supported": bool(cr[0] > 0),
                   "descriptive_delta_f1_vs_s23_head_seed0": float(d23.mean()), "descriptive_ci_vs_s23": cluster_interval(d23),
                   "replay_audit": replay, "new_encoder_fits": 0, "new_head_fits": 2, "scope": spec["scope"]})
        pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
        pd.DataFrame(classes).to_csv(OUT / "per_class.csv", index=False)
        pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    pd.DataFrame(traces).to_csv(OUT / "learning_curves.csv", index=False)
    write_json(OUT / "execution.json", {"wall_seconds": time.perf_counter() - start})
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "cache", "freeze", "run"))
    {"profile": profile, "cache": cache, "freeze": freeze, "run": run}[parser.parse_args().action]()
