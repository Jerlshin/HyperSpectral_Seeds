"""S16 · freeze the S17 round: write ``preregistration_s16.json`` and its SHA-256, from S16's evidence only.

The v5 reference (mean of the 6 grouped / 3 stratified Y3 runs at seeds 0–2) and its run-level sds are read from
``arm_summary.csv`` and ``seed_variance.csv`` (hypotheses.py) — not typed in. Margins follow the S12 convention,
2·max(sd, 0.009). Run once, before any S17 cell or runner exists; re-running refuses if the file already exists with
other content (the frozen file is never edited — an amendment is its own file, WORKFLOW §3.2.5).

Writes: preregistration_s16.json · preregistration_s16.sha256
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
from s16common import EVID, PREREG, PREREG_SHA, REPO, verify_preregistrations

V5_ARCH = ["model.spectral_descriptor=snv_morph", "model.spatial_tail_strides=[2,2,2,1]", "model.cbam_min_hw=3"]


def main() -> None:
    verify_preregistrations()
    r1 = json.load(open(PREREG["S12"]))["regime"]["overrides"]
    y3 = json.load(open(PREREG["S14"]))["arms"]["Y3_replication"]["change"]
    assert all(k in y3 for k in V5_ARCH), "v5 keys differ from the frozen Y3 arm"
    a = pd.read_csv(EVID / "arm_summary.csv").set_index("arm")
    sv = pd.read_csv(EVID / "seed_variance.csv")
    g = a.loc["Y3 grouped, seeds 0–2 (H21a)"]
    s = a.loc["Y3 stratified, seeds 0–2 (H21b)"]
    sd = {m: float(sv[(sv.arm == "Y3") & (sv.protocol == "grouped") & (sv.metric == m)].sd_pooled_6.iloc[0])
          for m in ("f1", "same", "cross", "attraction")}
    sd["strat_f1"] = float(sv[(sv.arm == "Y3") & (sv.protocol == "stratified")].sd_pooled_6.iloc[0])
    margin = {k: round(2 * max(v, 0.009), 6) for k, v in sd.items()}
    ref = dict(f1=round(float(g.f1), 6), same=round(float(g.same), 6), cross=round(float(g.cross), 6),
               attraction=round(float(g.attraction), 6), strat_f1=round(float(s.f1), 6))
    seed0 = {c: round(float(v), 6) for c, v in zip(g.cells.split(), g.f1_runs.split(), strict=True) if c.endswith("_s0")}
    t = dict(
        H23=round(ref["strat_f1"] + 0.030, 6),
        H24a=round(ref["f1"] - margin["f1"], 6), H24b=round(ref["f1"] + margin["f1"], 6),
        H24c_cross=round(ref["cross"] - margin["cross"], 6), H24c_attraction=round(ref["attraction"] + margin["attraction"], 6),
        H25a=round(ref["f1"] - margin["f1"], 6),
        H25b_cross=round(ref["cross"] + margin["cross"], 6), H25b_attraction=round(ref["attraction"] - margin["attraction"], 6),
    )
    v5 = [*r1.split(), *V5_ARCH]
    k64 = ["data.num_bands=64", "data.cutmix_bands=13", "data.max_cutout_bands=5",
           *(f"data.{k}=./dataset_u430k64/{f}" for k, f in (
               ("patches_data", "patches.npy"), ("labels_path", "labels.npy"), ("wavelength_path", "wavelengths.csv"),
               ("groups_path", "groups.npy"), ("scan_table_path", "scan_table.csv"), ("masks_path", "masks.npy"),
               ("morphology_path", "morphology.npy")))]
    gpu = (
        [dict(cell=f"Z1/v5_8020__f0_s{sd_}", arm="Z1", data="ablation/u430k32_stratified", fold=0, seed=sd_,
              arm_overrides="data.split_eval_frac=0.2") for sd_ in (0, 1, 2)]
        + [dict(cell=f"Z3/v5_end4__f{f}_s0", arm="Z3", data="ablation/u430k32_grouped", fold=f, seed=0,
                arm_overrides="model.spatial_tail_strides=[2,2,1,1]") for f in (0, 1)]
        + [dict(cell=f"Z2/v5_k64__f{f}_s0", arm="Z2", data="ablation/u430k32_grouped", fold=f, seed=0,
                arm_overrides=" ".join(k64)) for f in (0, 1)]
    )
    doc = {
        "frozen_at": "2026-10-03",
        "study": "S16 (reading of S15: Y3 replicated → SeedNet v5) → S17 (v5's within-acquisition tier-1 row, and two "
                 "screens on v5: a 64-band input and a 4 × 4 end map)",
        "repo_commit": "52fba4f (the S15 runner; every S15 cell ran on it with training code byte-identical to aed5257, "
                       "digest fade41e5…). S16's analysis, its evidence and this file are uncommitted when it is frozen.",
        "depends_on": {
            k: {"file": str(PREREG[k].relative_to(REPO / "docs/research")), "sha256": PREREG_SHA[k]} for k in PREREG},
        "depends_on_roles": {
            "S12": "R1 (regime.overrides), the margin convention 2·max(sd, 0.009), the guards",
            "S13": "the single-seed screening semantics (D28): a screen that passes is replicated at seeds 1–2 before "
                   "it can replace the reference",
            "S14": "the Y3 arm (= v5's architecture keys) and the reading of H21a–H22b, done in S16",
        },
        "motivating_evidence": "S16 findings F83–F89 (evidence/S16_replication_reading/). Every S15 held-out row was "
            "scored once by its own cell's final evaluation and aggregated in S16; nothing was tuned on it. Y3 replicated "
            "(H21a–H21d supported; H21e rejected) and became SeedNet v5 by the frozen rule (D35). The dissection (H22a, "
            "H22b both supported; H22a marginal) and its post-hoc profile tie the robustness gain to the spatial end-map "
            "repair (F87) — that observation MOTIVATES Z3, which is therefore a screen. Z2 is FW-03 (open since S08) run "
            "on the settled architecture. Z1 is FW-37, scheduled by S14 §9.2 if H21a ∧ H21b.",
        "architecture_and_regime": {
            "name": "SeedNet v5 = R1 + the Y3 keys",
            "overrides": " ".join(v5),
            "source": "preregistration_s12.json → regime.overrides; preregistration_s14.json → arms.Y3_replication.change",
            "note": "every S17 cell carries v5 explicitly, so a cell composes identically whether or not the config "
                    "default switch (D36) has landed when it runs",
        },
        "reference": {
            "v5": {**ref, "n_runs_grouped": 6, "n_runs_stratified": 3,
                   "cells": "s13/Y3/lean_*__f*_s0 + s15/Y3/lean_*__f*_s{1,2}"},
            "v5_run_sd": {k: round(v, 6) for k, v in sd.items()},
            "margins": {**margin, "convention": "2·max(sd, 0.009) (S12)"},
            "seed0_matched": {**seed0, "use": "Z2/Z3 seed-0 deltas on the same kernels, reported beside (decide nothing)"},
            "x1": "preregistration_s12.json → reference (unchanged), for context only",
        },
        "metric": "as the parents: macro-F1 on val ∪ test, TTA, mean over runs (no-TTA beside); same/cross-session "
                  "macro-recall and cross-session attraction (reporting/session.py); checkpoint chosen on calib only; "
                  "every held-out kernel scored once. Means of ≥ 2 runs carry a hierarchical bootstrap CI (S16 "
                  "s16common.hboot); one-seed screens carry S14's screen interval (kernel bootstrap ⊕ v5 run sd / √n).",
        "arms": {
            "Z1_tier1": {"change": "v5 on the 80/20 stratified split (data.split_eval_frac=0.2), as S13 Y4 did for X1",
                         "runs": "stratified fold 0 × seeds 0, 1, 2 (3 runs) — the D16 tier-1 row (FW-37)",
                         "role": "reporting row with one hypothesis (H23, H15's form)"},
            "Z2_k64": {"change": "v5 on uniform430 k = 64 (outputs/band_finalists/uniform430_k64, 432–1006 nm) instead "
                                 "of k = 32: a pre-sliced float16 cube built by `scripts/build_presliced_dataset.py "
                                 "--set uniform430_k64` (→ ./dataset_u430k64, uploaded as a Kaggle dataset); "
                                 "data.num_bands=64 and band-augmentation widths from band_augmentation_widths(64) "
                                 "(cutmix 13, cutout 5) so the augmentation is the same physical operation",
                       "runs": "grouped folds 0, 1 × seed 0 (2 runs; a screen, D28)",
                       "role": "FW-03 on the settled architecture: is k = 32 leaving information on the table?"},
            "Z3_end4": {"change": "v5 with tail strides [2,2,1,1]: the spatial tail ends at 4 × 4 instead of 2 × 2 (v5's "
                                  "rules unchanged: every tap trainable, no CBAM on a map ≤ 2 × 2 — so the gate after "
                                  "block 3 returns, its map now being 4 × 4)",
                        "runs": "grouped folds 0, 1 × seed 0 (2 runs; a screen, D28)",
                        "role": "mechanism: the 1 × 1 → 2 × 2 repair carried the robustness gain (F87). If that gain "
                                "came from pooling spatial statistics over a map (extent), a 4 × 4 map continues it; if "
                                "it came from repairing the last block (trainability), it does not."},
        },
        "hypotheses": {
            "H23": f"Z1: v5 80/20 stratified macro-F1, mean of 3 runs (seeds 0–2) − v5 70/30 stratified mean "
                   f"({ref['strat_f1']}) ≤ +0.030 (H15's form; i.e. mean ≤ {t['H23']})",
            "H24a": f"Z2 (screen): v5 k64 grouped mean (seed 0, 2 runs) ≥ v5 − margin = {t['H24a']} (non-inferior)",
            "H24b": f"Z2 (screen): v5 k64 grouped mean ≥ v5 + margin = {t['H24b']} (superior)",
            "H24c": f"Z2 (screen): v5 k64 cross-session recall ≥ {t['H24c_cross']} AND attraction ≤ "
                    f"{t['H24c_attraction']} (no session cost beyond v5's margin)",
            "H25a": f"Z3 (screen): v5 end-4 grouped mean (seed 0, 2 runs) ≥ {t['H25a']} (non-inferior)",
            "H25b": f"Z3 (screen): v5 end-4 cross-session recall ≥ {t['H25b_cross']} AND attraction ≤ "
                    f"{t['H25b_attraction']} (a further robustness step beyond v5's margin)",
        },
        "thresholds": t,
        "decision_rule": {
            "H23": "supported → the tier-1 (within-acquisition) row is v5's 80/20 mean, ≈ its 70/30 level (as F80 for "
                   "X1); rejected → v5 gains from more same-acquisition data, unlike X1 — the paper reports both "
                   "splits and says so. Either way Z1's mean ± CI is the D16 tier-1 row.",
            "H24b ∧ H24c": "more bands are a score lever without session cost: k64 is replicated at seeds 1–2 and the "
                           "215-band cube is screened in a later frozen round; D04/D11 under review",
            "H24b ∧ ¬H24c": "the extra bands buy session recognition (F70's pattern): k = 32 stays; recorded",
            "H24a ∧ ¬H24b": "k = 32 stays v5's input (D11/D15 hold for the network); FW-03 is closed for k ≤ 64; the "
                            "32-band multispectral claim stands",
            "¬H24a": "k = 32 confirmed; more bands hurt the network (F17's proxy result reproduced)",
            "H25a ∧ H25b": "end-map extent is a robustness lever (dose–response): Z3 is replicated at seeds 1–2 before "
                           "it can replace v5 (D28)",
            "H25a ∧ ¬H25b": "the repair's gain is trainability / the 2 × 2 statistics, not extent: v5 stays; the paper "
                            "attributes the robustness to the repair as one step, not a trend",
            "¬H25a": "v5 stays; a 4 × 4 end map costs accuracy",
            "D28_reversal": "a screen arm whose two fold estimates straddle a threshold by more than its margin is "
                            "replicated before its verdict is read",
            "combination": "Z2 and Z3 are read separately; nothing is combined in this round",
        },
        "guards": [
            "the parents' guards, unchanged: dirty / commit; clip fraction > 0.01 outside epoch 1 and overflow epochs "
            "(reported); stop epoch < 160 (reported)",
            "training code: either byte-identical to aed5257 (S15's digest fade41e5…), or the D36 default switch has "
            "landed with G-neutral passing for (i) the new default vs aed5257 + the v5 overrides and (ii) v5 + each "
            "arm's change vs aed5257 + the same overrides; the digest the cells ran on is recorded in frozen_cell.json",
            "runtime: Kaggle T4 × 2, torch 2.10.0+cu128 as S11/S13/S15; a different environment is reported",
            "Z2: dataset_u430k64/MANIFEST.json verifies (scripts/build_presliced_dataset.py --verify); band_axis.json "
            "names uniform430_k64 and 64 of 215 bands; the held-out row ids of each fold equal the k32 cells' (same "
            "kernels, so the seed-0 deltas are paired)",
        ],
        "cells": {
            "base_overrides": "runtime=kaggle_t4x2 tracking=console_jsonl (torchrun, one directory per cell)",
            "output_root": "outputs/experiments_u430k32/s17/<arm>/<variant>__f<fold>_s<seed>/",
            "order": "Z1 (reporting row) → Z3 → Z2 (needs the k64 dataset attached; if absent, Z2 is skipped and run "
                     "in a later session — cells are independent)",
            "gpu": gpu,
            "run_count": {"gpu_training_runs": len(gpu)},
            "runtime_estimate": "scaled from the S13/S15 cells' wall clock on the same runtime by the per-step cost "
                                "ratios of runtime_probe.json: Z1 ≈ 39 min × 3 (S13 Y4 80/20 39.8 min × v5/X1 0.92, "
                                "+ v5's 70/30 cells 28–36 min scaled by 5,864/5,130 kernels), Z3 ≈ 25.5 min × 2 (v5 "
                                "grouped 25.8 min × 0.99), Z2 ≈ 45 min × 2 (× 1.7) → ≈ 258 min ≈ 4.3 h (≤ 4.6 h if every "
                                "cell runs 200 epochs); one Kaggle session",
        },
        "implementation": "S17 part 1: (1) the D36 default switch behind G-neutral; (2) a runner (experiments/s17.py "
                          "+ scripts/run_s17.py) that builds these cells from this file after verifying it and its "
                          "three parents, composes each as v5 + the arm's change, checks the k64 manifest, records the "
                          "code digest, and offers --check/--cfg-job as S15; (3) the k64 dataset built and uploaded "
                          "(PI action, ≈ 4.6 GB). No model code is needed: every key exists at aed5257.",
        "not_in_this_round": "Replicating the dissection arms (H22's rule: not replicated; v5 keeps both halves); the "
                             "215-band cube (≈ 5–7× compute; only after H24b); the X6 level block (D25); the CPU track "
                             "FW-27/FW-28/FW-29 (own frozen files; bar now v5); RGB morphology FW-18 (raw archive).",
        "heldout_use": "Each cell is scored once on val ∪ test. No arm, epoch, threshold or hyperparameter is chosen by a "
                       "held-out score. S16's post-hoc diagnostics (dissection profile, ensembles, per-session "
                       "rescue) chose nothing and feed only the motivation recorded above.",
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    p = EVID / "preregistration_s16.json"
    if p.exists() and p.read_text() != text:
        raise SystemExit(f"{p} exists with other content — frozen files are never edited (write an amendment)")
    p.write_text(text)
    h = hashlib.sha256(text.encode()).hexdigest()
    (EVID / "preregistration_s16.sha256").write_text(f"{h}  preregistration_s16.json\n")
    print(json.dumps({"reference": doc["reference"]["v5"], "sd": doc["reference"]["v5_run_sd"], "thresholds": t}, indent=1))
    print("sha256", h)


if __name__ == "__main__":
    main()
