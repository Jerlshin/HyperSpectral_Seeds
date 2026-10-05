#!/usr/bin/env python
"""S25: selected S23 head initialization sensitivity, using S24's frozen cache."""
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

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import (
    class_metrics,
    cluster_interval,
    probe_logits,
    verify_plan,
)
from spectralquadnet.models.multimodal_residual import FrozenResidualFusion

PLAN = Path("configs/research/s25_head_seed_screen.json")
PARENT = Path("configs/research/s24_branch_multimodal.json")
OUT = Path("outputs/s25_head_seed_screen")


def completed(path: Path, plan: Path) -> None:
    state = json.loads((path / "COMPLETED.json").read_text())
    if state["plan_sha256"] != sha256(plan):
        raise ValueError("Completion plan differs")
    for name, digest in state["files"].items():
        if sha256(path / name) != digest:
            raise ValueError("Completed artifact changed")


def freeze() -> None:
    parent = verify_plan(PARENT)
    completed(Path("outputs/s24_branch_multimodal"), PARENT)
    s23 = Path(parent["parent_plan"])
    spec23 = verify_plan(s23)
    completed(Path("outputs/s23_frozen_multimodal"), s23)
    h24 = json.loads(Path("outputs/s24_branch_multimodal/hypothesis.json").read_text())
    if any(v["simplification_gate"] for v in h24["branch_gates"].values()):
        raise ValueError("A simpler head qualified; do not auto-replicate the larger candidate")
    inputs = dict(parent["input_hashes"])
    paths = [PARENT, Path(__file__), Path("outputs/s24_branch_multimodal/COMPLETED.json"),
             Path("outputs/s24_branch_multimodal/hypothesis.json"),
             Path("outputs/s23_frozen_multimodal/metrics.csv")]
    for f in (0, 1):
        paths += [Path(f"outputs/s24_branch_multimodal/features_f{f}_{split}.npz") for split in ("traincal", "heldout")]
        paths.append(Path(f"outputs/s23_frozen_multimodal/head_f{f}.pth"))
    inputs.update({str(p.relative_to(Path.cwd()) if p.is_absolute() else p): sha256(p) for p in paths})
    spec = {"study": "S25_head_seed_screen", "frozen_at": "2026-10-05", "parent_plan": str(PARENT),
            "scope": "Selective head-initialization sensitivity on fixed S22 seed0 encoders and reused acquisitions. Not full-system multi-seed confirmation.",
            "allocation_extension": "S23's fixed stop is preserved. After its >=.01 matched gain and S24's rejected simplifications, add only head seeds1/2 for that selected candidate. Use cached features; no encoder or RGB replication, no ablation-seed expansion.",
            "cells": [{"fold": f, "head_seed": s, "encoder_seed": 0} for f in (0, 1) for s in (1, 2)],
            "reference_head_seed": 0, "model": spec23["model"], "training": spec23["training"],
            "input_hashes": inputs,
            "cache_audit": "After selecting new heads, replay the existing seed0 head and equal anchor on cached held-out features; require exact saved S23 predictions. Never retrain seed0 or change the recipe.",
            "H43_screen": "Mean head gain versus equal-single >=.01 and paired variety CI>0, positive fold-mean gains and positive two-fold gains each head seed, nonnegative mean cross delta. Also require each head seed's two-fold mean F1 and cross recall >= S22 equal-TTA. Report seed SD and all per-fold outcomes; intervals exclude encoder/new-session variance.",
            "stop": "Exactly four cached head fits. No hyperparameter selection, new encoder fits, GPU job or automatic full-system confirmation."}
    for p in (PLAN, Path("docs/research/evidence/S25_head_seed_screen/preregistration.json")):
        if p.exists():
            raise FileExistsError("S25 already frozen")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, spec)
        p.with_suffix(".sha256").write_text(sha256(p) + "  " + p.name + "\n")
    print("Frozen", sha256(PLAN), flush=True)


def run() -> None:
    spec = verify_plan(PLAN)
    completed(Path("outputs/s24_branch_multimodal"), PARENT)
    OUT.mkdir(exist_ok=False)
    write_json(OUT / "STARTED.json", {"plan_sha256": sha256(PLAN)})
    torch.set_num_threads(4)
    labels = np.load("dataset_u430k32/labels.npy")
    rgb = np.load("outputs/s20_rgb_features_v3/rgb.npy")
    cross = pd.read_csv("dataset_u430k32/scan_table.csv").groupby("label").session_id.nunique().to_numpy() > 1
    prior23 = pd.read_csv("outputs/s23_frozen_multimodal/predictions.csv.gz")
    choices = json.loads(Path("outputs/s23_frozen_multimodal/selection.json").read_text())
    metrics: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    audits = []
    for fold in (0, 1):
        choice = next(v for v in choices if v["fold"] == fold)
        with np.load(f"outputs/s23_frozen_multimodal/scales_f{fold}.npz") as scale:
            hm, hs, rm, rs = (scale[k] for k in ("hsi_mean", "hsi_scale", "rgb_mean", "rgb_scale"))
            train_rows = scale["train_rows"]
        probe = Path(f"outputs/s21_rgb_study/probe_f{fold}_dino_rgb.npz")
        def tensors(h: npt.NDArray[Any], z: npt.NDArray[Any], rows: npt.NDArray[Any],
                    hm: npt.NDArray[Any] = hm, hs: npt.NDArray[Any] = hs,
                    rm: npt.NDArray[Any] = rm, rs: npt.NDArray[Any] = rs,
                    probe: Path = probe, choice: dict[str, Any] = choice) -> list[torch.Tensor]:
            anchor = (softmax(z / choice["hsi_temperature"], axis=1) +
                      softmax(probe_logits(probe, rgb[rows]) / choice["rgb_temperature"], axis=1)) / 2
            arrays = [(h-hm)/hs, (rgb[rows]-rm)/rs, anchor]
            if not all(np.isfinite(v).all() for v in arrays):
                raise ValueError("Nonfinite cached head inputs")
            return [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in arrays]
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_traincal.npz") as cache:
            tr, ca = cache["train"], cache["calib"]
            if not np.array_equal(tr, train_rows):
                raise ValueError("Cached features use different scales/train rows")
            train = tensors(cache["hsi_train"], cache["logits_train"], tr)
            calib = tensors(cache["hsi_calib"], cache["logits_calib"], ca)
        yt = torch.from_numpy(labels[tr].astype(np.int64))
        heads = {}
        for seed in (1, 2):
            torch.manual_seed(seed)
            head = FrozenResidualFusion(**spec["model"])
            recipe = spec["training"]
            optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
            best = float(class_metrics(labels[ca], calib[2].argmax(1).numpy(), 90)["f1"].mean())
            state, selected, stale = copy.deepcopy(head.state_dict()), 0, 0
            for epoch in range(1, recipe["epochs"]+1):
                head.train()
                order = torch.randperm(len(tr))
                for start in range(0, len(tr), recipe["batch"]):
                    batch = order[start:start+recipe["batch"]]
                    z, residual = head(*(v[batch] for v in train))
                    loss = torch.nn.functional.cross_entropy(z, yt[batch]) + recipe["residual_penalty"]*residual.square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite head-seed loss")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                head.eval()
                with torch.no_grad():
                    cz, _ = head(*calib)
                score = float(class_metrics(labels[ca], cz.argmax(1).numpy(), 90)["f1"].mean())
                traces.append({"fold": fold, "head_seed": seed, "epoch": epoch, "calib_f1": score})
                if score > best:
                    best, state, selected, stale = score, copy.deepcopy(head.state_dict()), epoch, 0
                else:
                    stale += 1
                if stale >= recipe["patience"]:
                    break
            head.load_state_dict(state)
            head.eval()
            heads[seed] = head
            entry = {"fold": fold, "head_seed": seed, "encoder_seed": 0, "selected_epoch": selected,
                     "epochs_completed": epoch, "calib_f1": best}
            selections.append(entry)
            torch.save({"head": state, "selection": entry, "plan_sha256": sha256(PLAN)}, OUT / f"head_f{fold}_s{seed}.pth")
        write_json(OUT / f"selection_f{fold}.json", [v for v in selections if v["fold"] == fold])
        # Only after both new checkpoints are selected, read cached held-out inputs.
        with np.load(f"outputs/s24_branch_multimodal/features_f{fold}_heldout.npz") as cache:
            te = cache["rows"]
            held = tensors(cache["hsi"], cache["logits"], te)
        original = FrozenResidualFusion(**spec["model"])
        original.load_state_dict(torch.load(f"outputs/s23_frozen_multimodal/head_f{fold}.pth", map_location="cpu", weights_only=True)["head"])
        original.eval()
        with torch.no_grad():
            replay, _ = original(*held)
        for arm, pred in (("learned_residual", replay.argmax(1).numpy()), ("equal_single", held[2].argmax(1).numpy())):
            saved = prior23[(prior23.fold == fold) & (prior23.arm == arm)].set_index("index").loc[te]
            mismatches = int(np.sum(pred != saved.prediction.to_numpy()))
            audits.append({"fold": fold, "arm": arm, "prediction_disagreements": mismatches})
            if mismatches:
                raise ValueError("Cached inputs do not exactly reproduce the S23 reference")
        for seed, head in heads.items():
            with torch.no_grad():
                scores, _ = head(*held)
            pred = scores.argmax(1).numpy()
            stat = class_metrics(labels[te], pred, 90)
            metrics.append({"fold": fold, "seed": seed, "encoder_seed": 0, "arm": "learned_residual",
                            "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(pred == labels[te])),
                            "same_recall": float(stat["recall"][~cross].mean()), "cross_recall": float(stat["recall"][cross].mean())})
            classes.extend({"fold": fold, "seed": seed, "arm": "learned_residual", "label": c, "cross": bool(cross[c]),
                            "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])} for c in range(90))
            predictions.extend({"fold": fold, "seed": seed, "encoder_seed": 0, "arm": "learned_residual", "index": int(i), "target": int(t), "prediction": int(p)}
                               for i, t, p in zip(te, labels[te], pred, strict=True))
        print("Completed cached head seeds for fold", fold, flush=True)
    m0 = pd.read_csv("outputs/s23_frozen_multimodal/metrics.csv").query("arm == 'learned_residual'").assign(encoder_seed=0)
    c0 = pd.read_csv("outputs/s23_frozen_multimodal/per_class.csv").query("arm == 'learned_residual'").assign(seed=0)
    p0 = prior23[prior23.arm == "learned_residual"].assign(seed=0, encoder_seed=0)
    m = pd.concat([m0, pd.DataFrame(metrics)], ignore_index=True)
    c = pd.concat([c0, pd.DataFrame(classes)], ignore_index=True)
    pred_all = pd.concat([p0, pd.DataFrame(predictions)], ignore_index=True)
    anchor = pd.read_csv("outputs/s23_frozen_multimodal/per_class.csv").query("arm == 'equal_single'").set_index(["label", "fold"]).sort_index()
    tta = pd.read_csv("outputs/s22_fusion_analysis/metrics.csv").query("arm == 'v5_rgb_equal'")
    seed_results, f1_deltas, cross_deltas = [], [], []
    for seed, block in c.groupby("seed"):
        a = block.set_index(["label", "fold"]).sort_index()
        delta = (a.f1-anchor.f1).unstack("fold").to_numpy()
        cr = (a.recall-anchor.recall)[a.cross].unstack("fold").to_numpy()
        f1_deltas.append(delta)
        cross_deltas.append(cr)
        means = m[m.seed == seed]
        seed_results.append({"head_seed": int(seed), "f1": float(means.f1.mean()), "cross_recall": float(means.cross_recall.mean()),
                             "delta_f1_anchor": float(delta.mean()), "delta_cross_anchor": float(cr.mean()),
                             "delta_f1_tta": float(means.f1.mean()-tta.f1.mean()),
                             "delta_cross_tta": float(means.cross_recall.mean()-tta.cross_recall.mean())})
    df, dr = np.mean(f1_deltas, axis=0), np.mean(cross_deltas, axis=0)
    ci = cluster_interval(df)
    gate = bool(df.mean() >= .01 and ci[0] > 0 and min(df.mean(0)) > 0 and
                all(v["delta_f1_anchor"] > 0 for v in seed_results) and dr.mean() >= 0 and
                all(v["delta_f1_tta"] >= 0 and v["delta_cross_tta"] >= 0 for v in seed_results))
    write_json(OUT / "hypothesis.json", {"H43_screen": gate, "head_seed_results": seed_results,
               "mean_delta_f1_anchor": float(df.mean()), "f1_ci": ci,
               "fold_mean_f1_deltas": df.mean(0).tolist(), "mean_delta_cross_anchor": float(dr.mean()),
               "cross_ci": cluster_interval(dr),
               "paired_f1_gain_head_seed_sd": float(np.std([v["delta_f1_anchor"] for v in seed_results], ddof=1)),
               "scope": spec["scope"], "new_encoder_fits": 0, "new_head_fits": 4})
    m.to_csv(OUT / "metrics.csv", index=False)
    c.to_csv(OUT / "per_class.csv", index=False)
    pred_all.to_csv(OUT / "predictions.csv.gz", index=False)
    pd.DataFrame(traces).to_csv(OUT / "learning_curves.csv", index=False)
    write_json(OUT / "selection.json", selections)
    write_json(OUT / "cache_replay.json", audits)
    write_json(OUT / "COMPLETED.json", {"plan_sha256": sha256(PLAN),
               "files": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
