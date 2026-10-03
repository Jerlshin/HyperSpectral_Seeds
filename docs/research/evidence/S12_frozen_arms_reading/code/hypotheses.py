"""S12 · read the frozen hypotheses H12a, H12b, H13, H14a, H14b and H16 exactly as frozen.

Quantities and files follow the reading map recorded before any arm ran (S11 §6.1). Thresholds are the
frozen ones (S09 ``preregistration_next.json`` f896d0e5…, S10 ``preregistration_s10.json`` 1c8ae693…);
the script refuses to run if either hash moved. Nothing here is selected on held-out: each arm's held-out
predictions were produced once by its own final evaluation; this script only aggregates them.

Uncertainty: run-level mean ± sd, plus a hierarchical bootstrap of every difference (2,000 resamples:
runs resampled with replacement within each fold and arm, held-out kernels resampled with replacement
within each fold and shared by both arms) — it carries both seed variance and test-set sampling.

Writes:
  arm_summary.csv   one row per arm: n, mean ± sd of every headline metric, Δ vs its reference with CI
  hypotheses.json   each frozen hypothesis: text, quantity, value, threshold, outcome, CI, decision routed
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from s12common import (CROSS, EVID, N_CLASSES, REPO, Cell, cells, load_preds, reference_cells,
                       save, verify_preregistrations)

RNG = np.random.default_rng(12)
B = 2000
S09 = REPO / "docs/research/evidence/S09_post_sweep_forensics"


# ── fast metrics on integer arrays ──────────────────────────────────────────────────────────────────
def conf(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(t * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def f1_from_conf(c: np.ndarray) -> float:
    tp = np.diag(c).astype(float)
    den = c.sum(0) + c.sum(1)
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())


def cross_recall_from_conf(c: np.ndarray) -> float:
    sup = c.sum(1)[CROSS]
    return float((np.diag(c)[CROSS] / np.maximum(sup, 1)).mean())


# ── arms ────────────────────────────────────────────────────────────────────────────────────────────
def arm_runs(cs: list[Cell], arm: str, variant: str, seeds=None) -> list[Cell]:
    return [c for c in cs if c.arm == arm and c.variant == variant and c.scored
            and (seeds is None or c.seed in seeds)]


def preds_by_fold(runs: list[Cell], variant: str = "tta") -> dict[int, tuple[np.ndarray, list[np.ndarray]]]:
    """fold → (targets in row order, [preds per run]); every run of a fold is scored on the same rows."""
    out: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}
    for c in runs:
        r, p, t = load_preds(c, variant)
        if c.fold in out:
            assert np.array_equal(out[c.fold][0][0], r), f"{c.name}: rows differ within fold"
            out[c.fold][1].append(p)
        else:
            out[c.fold] = ((r, t), [p])  # type: ignore[assignment]
    return {f: (rt[1], ps) for f, (rt, ps) in out.items()}  # type: ignore[misc]


def boot_delta(a: dict, b: dict | None, stat=f1_from_conf) -> tuple[float, float, float, np.ndarray]:
    """Mean over runs (pooled over folds) of ``stat``: arm a minus arm b (b None → a alone)."""
    folds = sorted(a)            # a reference with more folds is restricted to a's (X4 is fold 0 only)
    if b is not None:
        assert set(folds) <= set(b), "reference lacks a fold of the arm"

    def arm_mean(arm: dict, idx: dict, pick: dict) -> float:
        vals = []
        for f in folds:
            t, ps = arm[f]
            for j in pick[f]:
                vals.append(stat(conf(t[idx[f]], ps[j][idx[f]])))
        return float(np.mean(vals))

    full = {f: np.arange(len(a[f][0])) for f in folds}
    point_a = arm_mean(a, full, {f: range(len(a[f][1])) for f in folds})
    point = point_a - (arm_mean(b, full, {f: range(len(b[f][1])) for f in folds}) if b else 0.0)
    draws = np.empty(B)
    for i in range(B):
        idx = {f: RNG.integers(0, len(a[f][0]), len(a[f][0])) for f in folds}
        pa = {f: RNG.integers(0, len(a[f][1]), len(a[f][1])) for f in folds}
        d = arm_mean(a, idx, pa)
        if b is not None:
            pb = {f: RNG.integers(0, len(b[f][1]), len(b[f][1])) for f in folds}
            d -= arm_mean(b, idx, pb)
        draws[i] = d
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), draws


def main() -> None:
    verify_preregistrations()
    cs, ref = cells(), reference_cells()
    cdf = pd.read_csv(EVID / "cells.csv")
    s09 = pd.read_csv(S09 / "runs.csv")

    arms = {
        "S08 grouped (shipped, 6)": (arm_runs(ref, "S08", "grouped"), None),
        "S08 grouped matched (f0,f1 × s0,s1)": (arm_runs(ref, "S08", "grouped", {0, 1}), None),
        "S08 stratified (shipped, 6)": (arm_runs(ref, "S08", "stratified"), None),
        "X1 grouped": (arm_runs(cs, "X1", "grouped"), "S08 grouped (shipped, 6)"),
        "X1 stratified": (arm_runs(cs, "X1", "stratified"), "S08 stratified (shipped, 6)"),
        "X2 no_morph": (arm_runs(cs, "X2", "no_morph"), "S08 grouped matched (f0,f1 × s0,s1)"),
        "X2 spectral_only": (arm_runs(cs, "X2", "spectral_only"), "S08 grouped matched (f0,f1 × s0,s1)"),
        "X2 spatial_only": (arm_runs(cs, "X2", "spatial_only"), "S08 grouped matched (f0,f1 × s0,s1)"),
        "X4 grouped": (arm_runs(cs, "X4", "grouped"), "X1 grouped"),          # Δ on fold 0 only
        "X4 stratified": (arm_runs(cs, "X4", "stratified"), "X1 stratified"),
    }
    data = {k: {v: preds_by_fold(runs, v) for v in ("tta", "no_tta")} for k, (runs, _) in arms.items()}

    rows, deltas = [], {}
    for k, (runs, refk) in arms.items():
        names = [c.name for c in runs]
        if runs[0].arm == "S08":
            sub = s09[(s09.arm == runs[0].variant) & s09.apply(
                lambda r: f"S08/{r.arm}__f{r.fold}_s{r.seed}" in names, axis=1)]
            m = dict(f1_tta=sub.f1_tta, f1_no_tta=sub.f1_no_tta, same_recall_tta=sub.same_recall_tta,
                     cross_recall_tta=sub.cross_recall_tta, attraction_cross_tta=sub.attraction_cross,
                     clean_selected=pd.Series(dtype=float), calib_f1_selected=sub.calib_best_f1)
        else:
            sub = cdf[cdf.cell.isin(names)]
            m = dict(f1_tta=sub.f1_tta, f1_no_tta=sub.f1_no_tta, same_recall_tta=sub.same_recall_tta,
                     cross_recall_tta=sub.cross_recall_tta, attraction_cross_tta=sub.attraction_cross_tta,
                     clean_selected=sub.clean_best_selected, calib_f1_selected=sub.calib_f1_selected)
        row = {"arm": k, "n_runs": len(runs), "runs": " ".join(n.split("/", 1)[1] for n in names)}
        for q, s in m.items():
            row[f"{q}_mean"] = float(s.mean()) if len(s) else np.nan
            row[f"{q}_sd"] = float(s.std(ddof=1)) if len(s) > 1 else np.nan
        if refk is not None:
            for v in ("tta", "no_tta"):
                pt, lo, hi, dr = boot_delta(data[k][v], data[refk][v])
                row[f"delta_f1_{v}"], row[f"delta_f1_{v}_lo"], row[f"delta_f1_{v}_hi"] = pt, lo, hi
                deltas[(k, v)] = (pt, lo, hi, dr)
            if runs[0].protocol == "grouped":
                pt, lo, hi, _ = boot_delta(data[k]["tta"], data[refk]["tta"], stat=cross_recall_from_conf)
                row["delta_cross_recall_tta"], row["delta_cross_recall_tta_lo"], row["delta_cross_recall_tta_hi"] = pt, lo, hi
            row["reference"] = refk
        rows.append(row)
    summ = pd.DataFrame(rows)
    save(summ, "arm_summary.csv")

    # ── the frozen hypotheses ────────────────────────────────────────────────────────────────────────
    S = summ.set_index("arm")
    ref_g, ref_s = 0.5300333460397801, 0.7123676427318898          # frozen reference values (S09 prereg)
    x1g, x1s = S.loc["X1 grouped", "f1_tta_mean"], S.loc["X1 stratified", "f1_tta_mean"]
    dg_pt, dg_lo, dg_hi, dg_dr = deltas[("X1 grouped", "tta")]
    ds_pt, ds_lo, ds_hi, ds_dr = deltas[("X1 stratified", "tta")]
    ratio_draws = dg_dr / np.where(np.abs(ds_dr) < 1e-9, np.nan, ds_dr)

    x1 = cdf[cdf.arm == "X1"]
    h13 = {p: float(x1[x1.protocol == p].clean_best_selected.mean()) for p in ("grouped", "stratified")}
    h13_final = {p: float(x1[x1.protocol == p].clean_final_live.mean()) for p in ("grouped", "stratified")}

    nm = cdf[(cdf.arm == "X2") & (cdf.variant == "no_morph")]
    so = S.loc["X2 spectral_only"]
    full_matched = S.loc["S08 grouped matched (f0,f1 × s0,s1)", "f1_tta_mean"]
    so_pt, so_lo, so_hi, _ = deltas[("X2 spectral_only", "tta")]
    x4 = cdf[(cdf.arm == "X4") & (cdf.variant == "grouped")].iloc[0]

    H = {
        "H12a": dict(frozen="X1 raises stratified macro-F1 by ≥ +0.05 over 0.712 (3-seed mean)",
                     value=x1s, reference=ref_s, delta=x1s - ref_s, delta_ci=[ds_lo, ds_hi], threshold=0.05,
                     outcome="supported" if x1s - ref_s >= 0.05 else "rejected"),
        "H12b": dict(frozen="X1 raises grouped macro-F1 by ≥ +0.02 AND Δgrouped/Δstratified ≥ 0.37",
                     value=x1g, reference=ref_g, delta=x1g - ref_g, delta_ci=[dg_lo, dg_hi], threshold=0.02,
                     ratio=(x1g - ref_g) / (x1s - ref_s),
                     ratio_ci=[float(np.nanpercentile(ratio_draws, 2.5)), float(np.nanpercentile(ratio_draws, 97.5))],
                     outcome="supported" if (x1g - ref_g >= 0.02 and (x1g - ref_g) / (x1s - ref_s) >= 0.37) else "rejected"),
        "H13": dict(frozen="X1 reaches clean training accuracy ≥ 0.95 on the D18 subset",
                    reading="S11 §6.1: at_best_checkpoint[best_source].acc per protocol; final-epoch live beside",
                    value=h13, final_live=h13_final, reference_F45={"grouped": 0.900, "stratified": 0.911},
                    per_cell=dict(zip(x1.cell, x1.clean_best_selected)), threshold=0.95,
                    outcome="supported" if min(h13.values()) >= 0.95 else "rejected",
                    note="8 of 9 cells ≥ 0.95 at their selected checkpoint; grouped f0 s1 (0.916) early-stopped at "
                         "epoch 131 with its checkpoint at epoch 91 (D21 guard 4) and reaches 0.968/0.974 at its "
                         "last epoch"),
        "H14a": dict(frozen="no_morph cross-session recall ≤ 0.07 (mean of 4 runs, TTA)",
                     value=float(nm.cross_recall_tta.mean()), per_run=nm.cross_recall_tta.round(4).tolist(),
                     reference_full=0.15205610021786492, threshold=0.07,
                     delta_vs_matched_full=[S.loc["X2 no_morph", "delta_cross_recall_tta"],
                                            S.loc["X2 no_morph", "delta_cross_recall_tta_lo"],
                                            S.loc["X2 no_morph", "delta_cross_recall_tta_hi"]],
                     outcome="supported" if nm.cross_recall_tta.mean() <= 0.07 else "rejected"),
        "H14b": dict(frozen="full − spectral_only grouped macro-F1 ≤ +0.02",
                     full_definitions={"matched four S08 cells (f0,f1 × s0,s1)": full_matched,
                                       "six-run S08 mean": ref_g},
                     spectral_only=float(so.f1_tta_mean),
                     value={"matched": full_matched - so.f1_tta_mean, "six_run": ref_g - so.f1_tta_mean},
                     delta_ci_matched=[-so_hi, -so_lo], threshold=0.02,
                     outcome="supported" if max(full_matched, ref_g) - so.f1_tta_mean <= 0.02 else "rejected",
                     note="S11 §8 risk 6 asked for the choice of 'full' to be recorded before reading X2; it was not. "
                          "Both definitions are reported; the outcome is the same under either (Δ ≈ +0.10 ≫ 0.02)."),
        "H16": dict(frozen="X4 (grouped f0 s0) clean training accuracy ≥ 0.98, live, final epoch",
                    value=float(x4.clean_final_live), ema=float(x4.clean_final_ema), threshold=0.98,
                    stratified_companion=float(cdf[(cdf.arm == "X4") & (cdf.variant == "stratified")].clean_final_live.iloc[0]),
                    outcome="supported" if x4.clean_final_live >= 0.98 else "rejected"),
    }
    H["decision_routes"] = {
        "S09 rule": "H12a rejected → 'the regime is not the limit; read X2: if H14b supported, redesign the "
                    "spatial-spectral pathway; else the representation binds (route A)'. H14b rejected → ROUTE A "
                    "(stop capacity work; next = representation/data: FW-03, FW-18, FW-12)",
        "S09 H14a": "rejected → the network's cross-session recall is NOT carried by the 8 morphometric scalars "
                    "(the clause 'claims must be stated as shape-driven' is not triggered by H14a; see S12 §5.4 "
                    "for what does carry it)",
        "S10 H16": "supported → 'the architecture can fit; the shipped under-fit is the regime and its softeners'",
        "S10 regime_rule": "H12a rejected → X5 and X6, if run, use the SHIPPED regime and the S09 sweep "
                           "(0.530, σ_ref = 0.009) as reference",
        "S10 R2": "H12a rejected → 'the regime is not the limit; X2 decides (D17)'",
    }
    save(H, "hypotheses.json")
    print(json.dumps({k: (v.get("outcome"), v.get("value")) for k, v in H.items() if isinstance(v, dict) and "outcome" in v},
                     indent=1, default=float))
    print(summ[["arm", "n_runs", "f1_tta_mean", "f1_tta_sd", "cross_recall_tta_mean", "delta_f1_tta",
                "delta_f1_tta_lo", "delta_f1_tta_hi"]].to_string())


if __name__ == "__main__":
    main()
