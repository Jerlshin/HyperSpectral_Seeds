"""S12 · extract every S11 frozen cell (X1, X2, X4) into small evidence tables, with integrity checks.

No model is run and nothing is selected on held-out: each cell's held-out score was produced once by
its own final evaluation (S11 part 2, Kaggle T4 × 2); this script only reads the files.

Writes to docs/research/evidence/S12_frozen_arms_reading/:
  cells.csv       one row per cell: provenance, the regime as applied, stop/best epoch, calib selection,
                  clean fit (final and at the selected checkpoint), telemetry guards, held-out ±TTA with the
                  same/cross-session breakdown
  curves.csv      one row per cell × epoch: schedule, calib F1/acc live/EMA, clean fit, losses, grad norms,
                  clip fraction, pathway influence
  integrity.json  per-check verdicts: frozen hashes, commit, dirty-flag cause, regime vs frozen overrides,
                  D22 reversal (clip fraction), D21 guard 4 (stop epoch), the unscored cell and why
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from s12common import S11_COMMIT, cells, save, scalars_by_epoch, verify_preregistrations

# What each arm must have applied (S09 prereg X1/X2, S10 prereg X4; X2 = the shipped regime).
EXPECTED = {
    "X1": dict(epochs=200, patience=40, grad_clip=50.0, mixup_epochs=30, arcface_m=0.0,
               label_smooth=[0.1, 0.04], dropout=0.15, aug_profile="medium", aux_first=0.6477),
    "X2": dict(epochs=150, patience=25, grad_clip=5.0, mixup_epochs=110, arcface_m=0.3,
               label_smooth=[0.1, 0.04], dropout=0.15, aug_profile="medium"),
    "X4": dict(epochs=200, patience=200, grad_clip=50.0, mixup_epochs=0, arcface_m=0.0,
               label_smooth=[0.0, 0.0], dropout=0.0, aug_profile="none", aux_first=0.0),
}
PATHWAYS = {"no_morph": (["spatial", "spectral"], False), "spectral_only": (["spectral"], True),
            "spatial_only": (["spatial"], True)}

CURVE_KEYS = [
    "sched/lr", "sched/mixup", "sched/arcface_m", "sched/label_smooth", "sched/aux_weight_applied",
    "train/loss", "train/loss_main", "train/acc_plain", "train/acc_dominant", "train/skipped_batches",
    "train/steps",
    "val/f1_live", "val/f1_ema", "val/acc_live", "val/acc_ema", "val/f1_best",
    "fit/clean_train_acc_live", "fit/clean_train_acc_ema", "fit/clean_train_ce_live",
    "fit/clean_train_ce_ema", "grad_norm/clip_fraction", "grad_norm/preclip_backbone",
    "grad_norm/nonfinite_steps", "grad_norm/stem", "grad_norm/tail", "grad_norm/spectral",
    "grad_norm/fuse", "grad_norm/head", "grad_norm/aux", "loss/branch_spatial_weighted",
    "influence/branch_spatial", "influence/branch_spectral",
]


def regime_checks(arm: str, variant: str, reg: dict) -> list[str]:
    exp = EXPECTED[arm]
    bad = []
    for k in ("epochs", "patience", "grad_clip", "mixup_epochs", "arcface_m", "dropout", "aug_profile"):
        if reg[k] != exp[k]:
            bad.append(f"{k}={reg[k]} (frozen {exp[k]})")
    if list(reg["label_smooth"]) != exp["label_smooth"]:
        bad.append(f"label_smooth={reg['label_smooth']}")
    if "aux_first" in exp and abs(reg["aux_weight_applied"]["first"] - exp["aux_first"]) > 1e-3:
        bad.append(f"aux first={reg['aux_weight_applied']['first']}")
    if reg["clip_partition"] != "legacy" or reg["aux_weight_schedule"] != "legacy":
        bad.append("partition/aux schedule not legacy (D22)")
    want_p, want_m = PATHWAYS.get(variant, (["spatial", "spectral"], True))
    if list(reg["pathways"]) != want_p or bool(reg["morphometrics_input"]) != want_m:
        bad.append(f"pathways={reg['pathways']} morph={reg['morphometrics_input']}")
    return bad


def main() -> None:
    verify_preregistrations()
    rows, curves, integ = [], [], {"preregistration_hashes": "both match (f896d0e5…, 1c8ae693…)",
                                   "cells": {}}
    for c in cells():
        sc = scalars_by_epoch(c)
        last = json.load(open(c.path / "last_stage1.json"))
        cf = json.load(open(c.path / "clean_fit.json"))
        frozen = json.load(open(c.path / "frozen_cell.json"))
        r: dict = dict(cell=c.name, arm=c.arm, variant=c.variant, protocol=c.protocol, fold=c.fold,
                       seed=c.seed, scored=c.scored, stop_epoch=int(last["epoch"]),
                       finished=bool(last["finished"]))
        checks: dict = {"frozen_overrides": frozen["arm_overrides"]}

        # telemetry guards (whole run)
        r["clip_fraction_max"] = float(sc["grad_norm/clip_fraction"].max())
        r["clip_fraction_mean"] = float(sc["grad_norm/clip_fraction"].mean())
        r["epochs_clip_gt_1pct"] = int((sc["grad_norm/clip_fraction"] > 0.01).sum())
        # clip_fraction = mean over the 3 clip groups of "this group's norm exceeded grad_clip", averaged
        # over the epoch's steps → × 3 × steps = clipped group-steps (an fp16-overflow step counts as clipped)
        r["clipped_group_steps"] = float((sc["grad_norm/clip_fraction"] * 3 * sc["train/steps"]).sum())
        r["group_steps"] = int((3 * sc["train/steps"]).sum())
        hit = sc.loc[sc["grad_norm/clip_fraction"] > 0.01, "epoch"]
        ovf = set(sc.loc[sc["grad_norm/nonfinite_steps"] > 0, "epoch"])
        r["clip_epochs_beyond_ep1_without_overflow"] = int(sum(1 for e in hit if e > 1 and e not in ovf))
        r["nonfinite_steps"] = int(sc["grad_norm/nonfinite_steps"].sum())
        r["skipped_batches"] = int(sc["train/skipped_batches"].sum())
        r["aux_applied_first"] = float(sc["sched/aux_weight_applied"].iloc[0])
        r["aux_applied_last"] = float(sc["sched/aux_weight_applied"].iloc[-1])
        r["aux_loss_weighted_max"] = float(sc.get("loss/branch_spatial_weighted", pd.Series([np.nan])).max())

        # clean fit (D18 subset): final, and at the selected checkpoint (H13's reading, S11 §6.1)
        fin, ab = cf["final"], cf.get("at_best_checkpoint") or {}
        r.update(clean_final_epoch=cf["final_epoch"], clean_final_live=fin["live"]["acc"],
                 clean_final_ema=fin["ema"]["acc"], clean_final_ce_live=fin["live"]["ce"],
                 clean_best_epoch=ab.get("epoch"), clean_best_live=ab.get("live", {}).get("acc"),
                 clean_best_ema=ab.get("ema", {}).get("acc"))

        if c.scored:
            R = json.load(open(c.path / "results" / "run.json"))
            code, reg, ck = R["run"]["code"], R["run"]["regime"], R["checkpoint"]
            src = ck["best_source"]
            r.update(commit=code["commit"][:7], dirty=code["dirty"], best_epoch=ck["epoch"], best_source=src,
                     calib_f1_selected=ck["selection_f1"])
            row_at = sc.loc[sc.epoch == ck["epoch"]].iloc[0]
            r["calib_acc_selected"] = float(row_at[f"val/acc_{src}"])
            r["clean_best_selected"] = ab.get(src, {}).get("acc")
            r["fit_gap_selected"] = r["clean_best_selected"] - r["calib_acc_selected"]
            r["lr_at_best_frac_peak"] = float(row_at["sched/lr"]) / 5e-4
            for v in ("no_tta", "tta"):
                x, s = R["results"][v], R["results"][v]["session"]
                r[f"f1_{v}"] = x["macro_f1"]
                r[f"f1_{v}_lo"], r[f"f1_{v}_hi"] = x["macro_f1_ci"]["lo"], x["macro_f1_ci"]["hi"]
                r[f"acc_{v}"] = x["accuracy"]
                r[f"n_eval_{v}"] = x["n_samples"]
                r[f"same_recall_{v}"] = s["same_session"]["macro_recall"]
                r[f"cross_recall_{v}"] = s["cross_session"]["macro_recall"]
                r[f"attraction_cross_{v}"] = (s.get("attraction") or {}).get("cross")
                r[f"attraction_all_{v}"] = (s.get("attraction") or {}).get("all")
                r[f"entropy_bits_{v}"] = (s.get("entropy") or {}).get("predicted_bits")
            checks["commit_is_s11"] = code["commit"] == S11_COMMIT
            checks["dirty"] = code["dirty"]
            checks["regime_deviations"] = regime_checks(c.arm, c.variant, reg)
            checks["n_eval"] = R["results"]["tta"]["n_samples"]
        else:
            log = (c.path / "sweep.log").read_text(errors="replace")
            m = re.search(r"rank1 \| CRITICAL \| FATAL:.*?(\w+Error: [^\n]*)", log, re.S)
            checks["unscored_reason"] = {
                "rank1": m.group(1) if m else None,
                "rank0": "RuntimeError: Storage size calculation overflowed with sizes=[4575657221408423936]"
                if "4575657221408423936" in log else None,
                "best_epoch_equals_last": int(last["epoch"]) == int(cf["final_epoch"])
                and (cf.get("at_best_checkpoint") or {}).get("epoch") == int(last["epoch"]),
            }
        checks["d22_reversal_clip_fraction_gt_0.01"] = r["epochs_clip_gt_1pct"] if c.arm in ("X1", "X4") else None
        checks["d21_guard4_stop_before_160"] = (c.arm == "X1" and r["stop_epoch"] < 160)
        integ["cells"][c.name] = checks
        rows.append(r)

        keep = [k for k in CURVE_KEYS if k in sc.columns]
        cv = sc[["epoch", *keep]].copy()
        cv.insert(0, "cell", c.name)
        curves.append(cv)

    df = pd.DataFrame(rows)
    integ["dirty_cause"] = (
        "Every scored cell records commit 413a11e (the S11 commit) with dirty=true. provenance.code_revision "
        "sets dirty from `git status --porcelain`, which includes untracked files; README §10's Kaggle cell 1 "
        "runs `ln -sfn /kaggle/input/rice-hsi-u430k32 dataset_u430k32`, and .gitignore's `dataset_*/` "
        "(trailing slash) matches directories only, so the symlink is untracked. Reproduced in a scratch "
        "repository with this .gitignore: `?? dataset_u430k32` is the only porcelain line. The flag is "
        "therefore expected on every Kaggle run and does not by itself indicate modified code."
    )
    save(df, "cells.csv")
    save(pd.concat(curves, ignore_index=True), "curves.csv")
    save(integ, "integrity.json")
    print(df[["cell", "stop_epoch", "best_epoch", "f1_tta", "cross_recall_tta", "clean_final_live",
              "clip_fraction_max", "nonfinite_steps", "skipped_batches"]].to_string())


if __name__ == "__main__":
    main()
