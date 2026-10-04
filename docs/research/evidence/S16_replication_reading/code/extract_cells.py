"""S16 · extract every S15 cell into small evidence tables, with the frozen guards checked.

No model is run and nothing is selected on held-out: each cell's held-out score was produced once by its own final
evaluation (S15 part 2, Kaggle T4 × 2, 2026-10-03). This script only reads the files.

Checks (preregistration_s14.json → guards; S15 §7 telemetry; D34):
  * the three frozen hashes; commit = the S15 commit and ``dirty: false``; the training-code digest = ``aed5257``'s;
    the runtime = S11/S13's (torch 2.10.0+cu128, T4 × 2) — guard 3 compares the seed-0 Y3 cells across sessions
  * per-cell seed as frozen (``cells.gpu[*].seed``); parameters as S15 part 1 measured them
  * the regime as applied = R1 + the shipped keys + the arm's intent, stated independently of the runner
  * the guards: clip fraction > 0.01 outside epoch 1 and overflow epochs; stop epoch < 160
  * every held-out kernel once; macro-F1 and cross-session attraction re-derived from the saved predictions
  * the fresh-seed X1 reference (0.528496 / 0.728593) re-derived from the X1 cells it names

Writes: cells.csv · curves.csv · integrity.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from s16common import (
    CODE_DIGEST,
    ENVIRONMENT,
    EXPECTED_PARAMS,
    FRESH_REF,
    INTENT,
    PREREG,
    S14_EXTRACT,
    S15,
    S15_COMMIT,
    Cell,
    Scored,
    conf,
    f1_from_conf,
    jsonl,
    protocol,
    run_json,
    s15_cells,
    save,
    scalars_by_epoch,
    verify_preregistrations,
    x1_cells,
)

R1, DEFAULTS, CURVE_KEYS, heldout_block = (S14_EXTRACT.R1, S14_EXTRACT.DEFAULTS, S14_EXTRACT.CURVE_KEYS,
                                           S14_EXTRACT.heldout_block)


def regime_checks(c: Cell, reg: dict) -> list[str]:
    bad = []
    want = {**R1, **DEFAULTS, **INTENT[c.variant]}
    for k, v in want.items():
        got = list(reg[k]) if isinstance(reg[k], (list, tuple)) else reg[k]
        if got != v:
            bad.append(f"{k}={reg[k]} (want {v})")
    return bad


def main() -> None:
    verify_preregistrations()
    frozen = json.load(open(PREREG["S14"]))["cells"]["gpu"]
    frozen_seed = {g["cell"]: g["seed"] for g in frozen}
    rows, curves = [], []
    integ: dict = {"preregistration_hashes": "all three match (S12 88b377c5…, S13 ef598213…, S14 9e182670…)",
                   "cells": {}}
    for c in s15_cells():
        R = run_json(c)
        fc = json.load(open(c.path / "frozen_cell.json"))
        code, env = R["run"]["code"], R["run"]["environment"]
        r: dict = dict(cell=c.name, arm=c.arm, variant=c.variant, protocol=protocol(c), fold=c.fold, seed=c.seed,
                       commit=code["commit"][:7], dirty=code["dirty"])
        r.update(heldout_block(R))
        sc = scalars_by_epoch(c)
        last = json.load(open(c.path / "last_stage1.json"))
        cf = json.load(open(c.path / "clean_fit.json"))
        reg, ck = R["run"]["regime"], R["checkpoint"]
        ev = jsonl(c)
        src = ck["best_source"]
        r.update(stop_epoch=int(last["epoch"]), finished=bool(last["finished"]), best_epoch=ck["epoch"],
                 best_source=src, calib_f1_selected=ck["selection_f1"],
                 wall_min=(ev[-1]["t"] - ev[0]["t"]) / 60, epoch_s_median=float(sc["train/epoch_s"].median()),
                 parameters=R["run"]["parameters"],
                 **{f"params_{k}": v for k, v in R["run"]["parameter_breakdown"].items() if k != "total"},
                 n_train=R["run"]["split_report"]["sizes"]["train"], n_calib=R["run"]["split_report"]["sizes"]["calib"])
        row_at = sc.loc[sc.epoch == ck["epoch"]].iloc[0]
        r["calib_acc_selected"] = float(row_at[f"val/acc_{src}"])
        ab = cf.get("at_best_checkpoint") or {}
        r.update(clean_final_live=cf["final"]["live"]["acc"], clean_final_ema=cf["final"]["ema"]["acc"],
                 clean_best_selected=ab.get(src, {}).get("acc"))
        r["fit_gap_selected"] = r["clean_best_selected"] - r["calib_acc_selected"]
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

        s = Scored(c)
        st = s.stats()
        checks = {
            "commit_is_s15": code["commit"] == S15_COMMIT, "dirty": code["dirty"],
            "code_identity": bool(fc["code_identity"]["identical"]) and fc["code_identity"]["digest"] == CODE_DIGEST,
            "environment_as_s13": all(str(env.get(k)) == str(v) for k, v in ENVIRONMENT.items()),
            "seed_as_frozen": R["run"]["seed"] == frozen_seed[c.name] == c.seed,
            "parameters_as_measured_in_s15_part1": R["run"]["parameters"] == EXPECTED_PARAMS[c.variant],
            "regime_deviations": regime_checks(c, reg),
            "guard_clip_fraction_gt_0.01_outside_ep1_and_overflow": r["clip_epochs_beyond_ep1_without_overflow"],
            "guard_stop_before_160": r["stop_epoch"] < 160,
            "n_eval": R["results"]["tta"]["n_samples"],
            "each_kernel_once": len(np.unique(s.rows)) == len(s.rows) == R["results"]["tta"]["n_samples"],
            "f1_rederived_matches": abs(f1_from_conf(conf(s.t, s.p)) - r["f1_tta"]) < 1e-9,
        }
        if protocol(c) == "grouped":
            checks["attraction_rederived_matches"] = abs(st["attraction"] - r["attraction_cross_tta"]) < 1e-9
            checks["cross_rederived_matches"] = abs(st["cross"] - r["cross_recall_tta"]) < 1e-9
        integ["cells"][c.name] = checks
        rows.append(r)
        keep = [k for k in CURVE_KEYS if k in sc.columns]
        cv = sc[["epoch", *keep]].copy()
        cv.insert(0, "cell", c.name)
        curves.append(cv)

    df = pd.DataFrame(rows)
    x1g = pd.DataFrame([dict(seed=c.seed, **Scored(c).stats()) for c in x1_cells((1, 2))])
    x1s = pd.DataFrame([dict(seed=c.seed, **Scored(c).stats()) for c in x1_cells((1, 2), "stratified")])
    integ["fresh_reference_rederived"] = dict(
        grouped_f1=float(x1g.f1.mean()), grouped_cross=float(x1g.cross.mean()),
        grouped_attraction=float(x1g.attraction.mean()), stratified_f1=float(x1s.f1.mean()),
        matches_frozen=bool(abs(x1g.f1.mean() - FRESH_REF["f1"]) < 1e-6 and abs(x1s.f1.mean() - FRESH_REF["strat_f1"]) < 1e-6
                            and abs(x1g.cross.mean() - FRESH_REF["cross"]) < 1e-6
                            and abs(x1g.attraction.mean() - FRESH_REF["attraction"]) < 1e-6))
    summ = {d["cell"]: d for d in json.load(open(S15 / "summary.json"))}
    integ["summary_json_agrees"] = all(abs(summ[n]["f1_tta"] - v) < 1e-12 for n, v in zip(df.cell, df.f1_tta, strict=True))
    cc = integ["cells"].values()
    integ["summary"] = dict(
        n_cells=int(len(df)), n_scored=int(df.f1_tta.notna().sum()),
        all_commit_s15=all(v["commit_is_s15"] for v in cc), any_dirty=any(v["dirty"] for v in cc),
        all_code_identical=all(v["code_identity"] for v in cc), all_environment_as_s13=all(v["environment_as_s13"] for v in cc),
        all_seeds_as_frozen=all(v["seed_as_frozen"] for v in cc),
        all_parameters_as_measured=all(v["parameters_as_measured_in_s15_part1"] for v in cc),
        regime_deviations=int(sum(len(v["regime_deviations"]) for v in cc)),
        all_rederivations_match=all(v["f1_rederived_matches"] and v.get("attraction_rederived_matches", True)
                                    and v.get("cross_rederived_matches", True) and v["each_kernel_once"] for v in cc),
        clip_guard_cells=df.loc[df.clip_epochs_beyond_ep1_without_overflow > 0, "cell"].tolist(),
        clipped_group_steps=df.clipped_group_steps.round(1).tolist(),
        stop_before_160=df.loc[df.stop_epoch < 160, "cell"].tolist(),
        best_epoch_is_last=df.loc[df.best_epoch == df.stop_epoch, "cell"].tolist(),
        skipped_batches=int(df.skipped_batches.sum()), nonfinite_steps_per_run=df.nonfinite_steps.tolist(),
        n_eval=sorted(set(int(v["n_eval"]) for v in cc)),
        wall_clock_gpu_min=float(df.wall_min.sum()), runtime_estimate_min="266 typical · 287 cap (S15 §7)",
    )
    save(df, "cells.csv")
    save(pd.concat(curves, ignore_index=True), "curves.csv")
    save(integ, "integrity.json")
    print(json.dumps(integ["summary"], indent=1))
    print(json.dumps(integ["fresh_reference_rederived"], indent=1))
    print(df[["cell", "stop_epoch", "best_epoch", "calib_f1_selected", "f1_tta", "same_recall_tta", "cross_recall_tta",
              "attraction_cross_tta", "clean_best_selected", "kappa_embedding", "wall_min"]].round(4).to_string())


if __name__ == "__main__":
    main()
