"""S14 · extract every S13 screening cell into small evidence tables, with the frozen guards checked.

No model is run and nothing is selected on held-out: each cell's held-out score was produced once by its own
final evaluation (S13 part 2, Kaggle T4 × 2, 2026-10-03); Y1's fused cells by ``experiments/fusion.py`` with a
calib-chosen weight. This script only reads the files.

Writes to docs/research/evidence/S14_screen_reading/:
  cells.csv       one row per cell: provenance, regime as applied, stop/best epoch, calib selection, clean fit,
                  telemetry guards, wall clock, held-out ±TTA with the same/cross-session breakdown, session κ
  curves.csv      one row per GPU cell × epoch: schedule, calib F1/acc, clean fit, losses, grad norms, clip
                  fraction, pathway influence
  integrity.json  per-check verdicts: frozen hashes, commit, dirty, regime vs R1 + the arm's intent, the
                  guards (clip fraction, stop epoch < 160), the P0.3 re-score, the attraction re-derivation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from s14common import (
    S13_COMMIT,
    SWEEP,
    Cell,
    attraction_cross,
    jsonl,
    load_preds,
    protocol,
    run_json,
    s13_cells,
    save,
    scalars_by_epoch,
    verify_preregistrations,
)

R1 = dict(epochs=200, patience=40, grad_clip=50.0, mixup_epochs=30, arcface_m=0.0, label_smooth=[0.1, 0.04],
          dropout=0.15, aug_profile="medium", clip_partition="legacy", aux_weight_schedule="legacy")
# each arm's intent, stated independently of the runner (preregistration_s12.json → arms.*.change)
INTENT = {
    "spectral_only": dict(pathways=["spectral"]),
    "spatial_only": dict(pathways=["spatial"]),
    "mixstyle": dict(spatial_mixstyle=True),
    "lean_grouped": dict(spectral_descriptor="snv_morph", spatial_tail_strides=[2, 2, 2, 1], cbam_min_hw=3),
    "lean_stratified": dict(spectral_descriptor="snv_morph", spatial_tail_strides=[2, 2, 2, 1], cbam_min_hw=3),
    "within_8020": dict(split_eval_frac=0.2),
}
DEFAULTS = dict(pathways=["spatial", "spectral"], spatial_mixstyle=False, spectral_descriptor="full",
                spatial_tail_strides=[2, 2, 2, 2], cbam_min_hw=0, split_eval_frac=0.3, morphometrics_input=True)
CURVE_KEYS = [
    "sched/lr", "sched/mixup", "sched/label_smooth", "sched/aux_weight_applied", "train/loss", "train/loss_main",
    "train/acc_plain", "train/epoch_s", "val/f1_live", "val/f1_ema", "val/acc_live", "val/acc_ema", "val/f1_best",
    "fit/clean_train_acc_live", "fit/clean_train_acc_ema", "grad_norm/clip_fraction", "grad_norm/preclip_backbone",
    "grad_norm/nonfinite_steps", "grad_norm/stem", "grad_norm/tail", "grad_norm/spectral", "grad_norm/fuse",
    "influence/branch_spatial", "influence/branch_spectral",
]


def regime_checks(c: Cell, reg: dict) -> list[str]:
    bad = []
    for k, v in R1.items():
        got = list(reg[k]) if isinstance(reg[k], (list, tuple)) else reg[k]
        if got != v:
            bad.append(f"{k}={reg[k]} (R1 {v})")
    want = {**DEFAULTS, **INTENT[c.variant]}
    for k, v in want.items():
        got = list(reg[k]) if isinstance(reg[k], (list, tuple)) else reg[k]
        if got != v:
            bad.append(f"{k}={reg[k]} (intent {v})")
    return bad


def heldout_block(R: dict) -> dict:
    out = {}
    for v in ("no_tta", "tta"):
        x, s = R["results"][v], R["results"][v]["session"]
        out.update({f"f1_{v}": x["macro_f1"], f"f1_{v}_lo": x["macro_f1_ci"]["lo"],
                    f"f1_{v}_hi": x["macro_f1_ci"]["hi"], f"acc_{v}": x["accuracy"], f"n_eval_{v}": x["n_samples"],
                    f"same_recall_{v}": s["same_session"]["macro_recall"],
                    f"cross_recall_{v}": (s.get("cross_session") or {}).get("macro_recall"),
                    f"attraction_cross_{v}": (s.get("attraction") or {}).get("cross"),
                    f"attraction_all_{v}": (s.get("attraction") or {}).get("all"),
                    f"entropy_bits_{v}": (s.get("entropy") or {}).get("predicted_bits")})
    return out


def main() -> None:
    verify_preregistrations()
    rows, curves = [], []
    integ: dict = {"preregistration_hashes": "both match (S12 88b377c5…, S13 ef598213…)", "cells": {}}
    for c in s13_cells():
        R = run_json(c)
        code = R["run"]["code"]
        r: dict = dict(cell=c.name, arm=c.arm, variant=c.variant, protocol=protocol(c), fold=c.fold, seed=c.seed,
                       commit=code["commit"][:7], dirty=code["dirty"])
        checks: dict = {"commit_is_s13": code["commit"] == S13_COMMIT, "dirty": code["dirty"]}
        r.update(heldout_block(R))

        if c.variant == "fused":
            fu = R["fusion"]
            eq = fu["equal_weight"]["results"]["tta"]
            r.update(fusion_weight=fu["weight"], calib_f1_selected=fu["calib_macro_f1_tta"][f"{fu['weight']:.2f}"],
                     eq_f1_tta=eq["macro_f1"], eq_same_recall_tta=eq["session"]["same_session"]["macro_recall"],
                     eq_cross_recall_tta=eq["session"]["cross_session"]["macro_recall"],
                     eq_attraction_cross_tta=eq["session"]["attraction"]["cross"])
            checks["inputs"] = R["run"].get("inputs") or fu.get("inputs")
            integ["cells"][c.name] = checks
            rows.append(r)
            continue

        sc = scalars_by_epoch(c)
        last = json.load(open(c.path / "last_stage1.json"))
        cf = json.load(open(c.path / "clean_fit.json"))
        reg, ck = R["run"]["regime"], R["checkpoint"]
        ev = jsonl(c)
        src = ck["best_source"]
        r.update(stop_epoch=int(last["epoch"]), finished=bool(last["finished"]), best_epoch=ck["epoch"],
                 best_source=src, calib_f1_selected=ck["selection_f1"],
                 wall_min=(ev[-1]["t"] - ev[0]["t"]) / 60, epoch_s_median=float(sc["train/epoch_s"].median()),
                 parameters=R["run"]["parameters"], **{f"params_{k}": v for k, v in
                                                      R["run"]["parameter_breakdown"].items() if k != "total"},
                 n_train=R["run"]["split_report"]["sizes"]["train"],
                 n_calib=R["run"]["split_report"]["sizes"]["calib"])
        row_at = sc.loc[sc.epoch == ck["epoch"]].iloc[0]
        r["calib_acc_selected"] = float(row_at[f"val/acc_{src}"])
        ab = cf.get("at_best_checkpoint") or {}
        r.update(clean_final_live=cf["final"]["live"]["acc"], clean_final_ema=cf["final"]["ema"]["acc"],
                 clean_best_selected=ab.get(src, {}).get("acc"))
        r["fit_gap_selected"] = r["clean_best_selected"] - r["calib_acc_selected"]
        r["lr_at_best_frac_peak"] = float(row_at["sched/lr"]) / 5e-4
        # guards (parent preregistration_s12.json → guards)
        clip = sc["grad_norm/clip_fraction"]
        ovf = set(sc.loc[sc["grad_norm/nonfinite_steps"] > 0, "epoch"])
        hit = sc.loc[clip > 0.01, "epoch"]
        r.update(clip_fraction_max=float(clip.max()), epochs_clip_gt_1pct=int((clip > 0.01).sum()),
                 clip_epochs_beyond_ep1_without_overflow=int(sum(1 for e in hit if e > 1 and e not in ovf)),
                 clipped_group_steps=float((clip * 3 * sc["train/steps"]).sum()),
                 nonfinite_steps=int(sc["grad_norm/nonfinite_steps"].sum()),
                 skipped_batches=int(sc["train/skipped_batches"].sum()))
        for k in ("influence/branch_spatial", "influence/branch_spectral"):
            if k in sc:
                r[k.replace("/", "_") + "_at_best"] = float(row_at.get(k, np.nan))
        kp = R.get("session_probe", {}).get("representations", {})
        for rep in ("embedding", "spatial", "spectral"):
            r[f"kappa_{rep}"] = kp.get(rep, {}).get("kappa")
        checks["regime_deviations"] = regime_checks(c, reg)
        checks["guard_clip_fraction_gt_0.01_outside_ep1_and_overflow"] = r["clip_epochs_beyond_ep1_without_overflow"]
        checks["guard_stop_before_160"] = r["stop_epoch"] < 160
        checks["n_eval"] = R["results"]["tta"]["n_samples"]

        # re-derive cross-session attraction from the saved predictions (grouped only): definition check
        if protocol(c) == "grouped":
            rr, p, _ = load_preds(c, "tta")
            checks["attraction_rederived_matches"] = abs(attraction_cross(rr, p, c.fold) - r["attraction_cross_tta"]) < 1e-9
        integ["cells"][c.name] = checks
        rows.append(r)

        keep = [k for k in CURVE_KEYS if k in sc.columns]
        cv = sc[["epoch", *keep]].copy()
        cv.insert(0, "cell", c.name)
        curves.append(cv)

    df = pd.DataFrame(rows)
    rescore = SWEEP / "s11/X2/spatial_only__f1_s0/results/run.json"
    integ["p0_3_rescore_X2_spatial_only_f1_s0"] = (
        "done" if rescore.exists() else
        "NOT done — outputs/experiments_u430k32/s11/X2/spatial_only__f1_s0/results/ does not exist (the S11 "
        "outputs were not attached to the Kaggle session; S13 §8's conditional skipped it). Decides nothing.")
    gpu = df[df.variant != "fused"]
    integ["summary"] = dict(
        n_cells=int(len(df)), n_scored=int(df.f1_tta.notna().sum()), all_commit_s13=bool((df.commit == S13_COMMIT[:7]).all()),
        any_dirty=bool(df.dirty.any()), regime_deviations=int(sum(len(v.get("regime_deviations", []))
                                                                   for v in integ["cells"].values())),
        clip_guard_cells=gpu.loc[gpu.clip_epochs_beyond_ep1_without_overflow > 0, "cell"].tolist(),
        stop_before_160=gpu.loc[gpu.stop_epoch < 160, "cell"].tolist(),
        skipped_batches=int(gpu.skipped_batches.sum()), nonfinite_steps_per_run=gpu.nonfinite_steps.tolist(),
        wall_clock_gpu_min=float(gpu.wall_min.sum()), runtime_estimate_min="266 typical · 287 cap (S13 §8)",
        torch_compile="auto-disabled on T4 (sm_75 pre-Ampere) in every cell — eager mode, as in S11; S13 risk 5 moot",
    )
    save(df, "cells.csv")
    save(pd.concat(curves, ignore_index=True), "curves.csv")
    save(integ, "integrity.json")
    print(json.dumps(integ["summary"], indent=1))
    print(df[["cell", "stop_epoch", "best_epoch", "calib_f1_selected", "f1_tta", "same_recall_tta", "cross_recall_tta",
              "attraction_cross_tta", "clean_best_selected", "kappa_embedding"]].round(4).to_string())


if __name__ == "__main__":
    main()
