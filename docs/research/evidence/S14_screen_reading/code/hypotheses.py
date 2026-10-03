"""S14 · read the S13 screen's frozen hypotheses H19a, H19b, H20, H21a, H21b and H15 exactly as frozen.

Quantities, files and thresholds are the reading map recorded before any S13 cell ran (S13 §9;
preregistration_s12.json → hypotheses, preregistration_s13.json → reading). The script refuses to run if either
hash moved. Nothing is selected on held-out: every held-out prediction was produced once by its cell's own final
evaluation (Y1's fused cells with a calib-chosen weight); this script only aggregates them.

Uncertainty, stated honestly for a one-seed screen (D28):
  * ``ci_kernels``   — test-set sampling only: held-out kernels resampled with replacement within each fold
                       (2,000 draws); the statistic is recomputed on the resample.
  * ``ci_screen``    — adds seed variance analytically, from X1's measured run-level sd under R1 on the same data
                       and code path (the sd the frozen margins were set from): half-width
                       1.96·sqrt(se_kernels² + sd_X1²/n_runs). This is the interval a screening verdict is read
                       against: a verdict is *clear* when it excludes the threshold, *marginal* otherwise.
  * matched deltas   — each S13 cell minus its fold- and seed-matched X1 cell (seed 0), paired on the same
                       held-out kernels (Y4 is scored on other rows: unpaired, its own CI). Reported beside;
                       decide nothing (preregistration_s13.json → reading.reported_beside_deciding_nothing).

Writes: hypotheses.json · arm_summary.csv · matched_deltas.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from s14common import (
    CROSS,
    REF,
    SAME,
    SESSION,
    THRESH,
    Cell,
    conf,
    f1_from_conf,
    grouped_rows,
    load_preds,
    protocol,
    recall_from_conf,
    s13_cells,
    save,
    train_session_by_class,
    verify_preregistrations,
    x1_matched,
)

RNG = np.random.default_rng(14)
B = 2000


class Scored:
    """Held-out TTA predictions of one cell with every statistic the reading needs, recomputable on a resample."""

    def __init__(self, c: Cell, variant: str = "tta"):
        self.cell = c
        self.rows, self.p, self.t = load_preds(c, variant)
        self.grouped = protocol(c) == "grouped"
        if self.grouped:
            tr, _, _ = grouped_rows(c.fold)
            soc = train_session_by_class(tr)
            self.is_cross = np.isin(self.t, CROSS)
            self.attracted = soc[self.p] == SESSION[self.rows]

    def stats(self, idx: np.ndarray | None = None) -> dict[str, float]:
        i = slice(None) if idx is None else idx
        cm = conf(self.t[i], self.p[i])
        out = {"f1": f1_from_conf(cm), "same": recall_from_conf(cm, SAME if self.grouped else np.arange(90))}
        if self.grouped:
            out["cross"] = recall_from_conf(cm, CROSS)
            m = self.is_cross[i]
            out["attraction"] = float(self.attracted[i][m].mean())
        return out


def boot(cells: list[Scored], ref: list[Scored] | None = None) -> dict[str, dict]:
    """Mean over cells (one per fold) of each statistic, minus the paired reference's; kernel bootstrap."""
    def mean_stats(group, idxs):
        s = [g.stats(ix) for g, ix in zip(group, idxs, strict=True)]
        return {k: float(np.mean([x[k] for x in s])) for k in s[0]}

    full = [None] * len(cells)
    point = mean_stats(cells, full)
    if ref is not None:
        for a, b in zip(cells, ref, strict=True):
            assert np.array_equal(a.rows, b.rows), f"{a.cell.name}: rows differ from its matched reference"
        pr = mean_stats(ref, full)
        point = {k: point[k] - pr[k] for k in point}
    draws = {k: [] for k in point}
    for _ in range(B):
        idxs = [RNG.integers(0, len(g.rows), len(g.rows)) for g in cells]
        d = mean_stats(cells, idxs)
        if ref is not None:
            r = mean_stats(ref, idxs)
            d = {k: d[k] - r[k] for k in d}
        for k in d:
            draws[k].append(d[k])
    out = {}
    for k, v in point.items():
        a = np.asarray(draws[k])
        out[k] = dict(value=v, ci_kernels=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                      se_kernels=float(a.std(ddof=1)))
    return out


SEED_SD = {"f1": REF["f1_sd"], "same": REF["same_sd"], "cross": REF["cross_sd"], "attraction": REF["attraction_sd"]}


def screen_interval(stat: dict, key: str, n_runs: int, stratified: bool = False) -> list[float]:
    sd = REF["strat_f1_sd"] if stratified else SEED_SD[key]
    hw = 1.96 * np.sqrt(stat["se_kernels"] ** 2 + sd ** 2 / n_runs)
    return [stat["value"] - hw, stat["value"] + hw]


def verdict(value: float, interval: list[float], thr: float, direction: str) -> dict:
    ok = value >= thr if direction == ">=" else value <= thr
    clear = (interval[0] >= thr or interval[1] < thr) if direction == ">=" else (interval[1] <= thr or interval[0] > thr)
    return dict(value=value, threshold=thr, direction=direction, met=bool(ok), screen_interval=interval,
                clarity="clear" if clear else "marginal")


def main() -> None:
    verify_preregistrations()
    cs = {c.name: c for c in s13_cells()}
    S = {n: Scored(c) for n, c in cs.items()}
    X = {n: Scored(x1_matched(c)) for n, c in cs.items()}

    arms = {
        "Y1 fused (calib w)": ["Y1/fused__f0_s0", "Y1/fused__f1_s0"],
        "Y1 spectral_only": ["Y1/spectral_only__f0_s0", "Y1/spectral_only__f1_s0"],
        "Y1 spatial_only": ["Y1/spatial_only__f0_s0", "Y1/spatial_only__f1_s0"],
        "Y2 mixstyle": ["Y2/mixstyle__f0_s0", "Y2/mixstyle__f1_s0"],
        "Y3 lean grouped": ["Y3/lean_grouped__f0_s0", "Y3/lean_grouped__f1_s0"],
        "Y3 lean stratified": ["Y3/lean_stratified__f0_s0"],
        "Y4 80/20": ["Y4/within_8020__f0_s0"],
    }
    summary, matched = [], []
    B_ = {}
    for arm, names in arms.items():
        own = boot([S[n] for n in names])
        B_[arm] = own
        row = {"arm": arm, "n_runs": len(names), "cells": " ".join(names)}
        for k, v in own.items():
            row[k] = v["value"]
            row[f"{k}_lo"], row[f"{k}_hi"] = v["ci_kernels"]
            row[f"{k}_per_fold"] = " ".join(f"{S[n].stats()[k]:.4f}" for n in names)
        if arm != "Y4 80/20":
            d = boot([S[n] for n in names], [X[n] for n in names])
            for k, v in d.items():
                row[f"d_{k}_vs_x1s0"] = v["value"]
                row[f"d_{k}_vs_x1s0_lo"], row[f"d_{k}_vs_x1s0_hi"] = v["ci_kernels"]
            for n in names:
                dd = boot([S[n]], [X[n]])
                matched.append({"cell": n, "x1_cell": x1_matched(cs[n]).name,
                                **{f"{k}": S[n].stats()[k] for k in dd}, **{f"x1_{k}": X[n].stats()[k] for k in dd},
                                **{f"d_{k}": v["value"] for k, v in dd.items()},
                                **{f"d_{k}_lo": v["ci_kernels"][0] for k, v in dd.items()},
                                **{f"d_{k}_hi": v["ci_kernels"][1] for k, v in dd.items()}})
        else:
            row["d_f1_vs_x1_strat_ref"] = own["f1"]["value"] - REF["strat_f1"]
        summary.append(row)
    save(pd.DataFrame(summary), "arm_summary.csv")
    save(pd.DataFrame(matched), "matched_deltas.csv")

    fu, y2, y3g = B_["Y1 fused (calib w)"], B_["Y2 mixstyle"], B_["Y3 lean grouped"]
    y3s, y4 = B_["Y3 lean stratified"], B_["Y4 80/20"]
    H = {
        "H19a": dict(frozen="Y1 fused grouped macro-F1 (TTA, mean of fused cells) ≥ 0.530786 − 0.020 = 0.5108",
                     **verdict(fu["f1"]["value"], screen_interval(fu["f1"], "f1", 2), THRESH["H19a"], ">="),
                     ci_kernels=fu["f1"]["ci_kernels"], per_fold=[S[n].stats()["f1"] for n in arms["Y1 fused (calib w)"]]),
        "H19b": dict(frozen="Y1 fused cross-session recall ≥ 0.1666 AND cross-session attraction ≤ 0.4240",
                     cross=verdict(fu["cross"]["value"], screen_interval(fu["cross"], "cross", 2), THRESH["H19b_cross"], ">="),
                     attraction=verdict(fu["attraction"]["value"], screen_interval(fu["attraction"], "attraction", 2),
                                        THRESH["H19b_attraction"], "<="),
                     per_fold_cross=[S[n].stats()["cross"] for n in arms["Y1 fused (calib w)"]],
                     per_fold_attraction=[S[n].stats()["attraction"] for n in arms["Y1 fused (calib w)"]]),
        "H20": dict(frozen="Y2 cross-session recall ≥ 0.1666 AND same-session recall ≥ 0.6285",
                    cross=verdict(y2["cross"]["value"], screen_interval(y2["cross"], "cross", 2), THRESH["H20_cross"], ">="),
                    same=verdict(y2["same"]["value"], screen_interval(y2["same"], "same", 2), THRESH["H20_same"], ">="),
                    per_fold_cross=[S[n].stats()["cross"] for n in arms["Y2 mixstyle"]],
                    per_fold_same=[S[n].stats()["same"] for n in arms["Y2 mixstyle"]],
                    frontier_move_test="cross ≥ ref + 0.020 (0.1666) but same < ref − 0.020 (0.6285) → trade-off point"),
        "H21a": dict(frozen="Y3 grouped macro-F1 ≥ 0.5108 (non-inferior)",
                     **verdict(y3g["f1"]["value"], screen_interval(y3g["f1"], "f1", 2), THRESH["H21a"], ">="),
                     ci_kernels=y3g["f1"]["ci_kernels"], per_fold=[S[n].stats()["f1"] for n in arms["Y3 lean grouped"]]),
        "H21b": dict(frozen="Y3 stratified macro-F1 ≥ 0.726984 − 0.018 = 0.7090 (one run)",
                     **verdict(y3s["f1"]["value"], screen_interval(y3s["f1"], "f1", 1, stratified=True), THRESH["H21b"], ">="),
                     ci_kernels=y3s["f1"]["ci_kernels"]),
        "H15": dict(frozen="(S09, unchanged) Y4 (80/20) − R1 stratified (0.726984) ≤ +0.03 (one run)",
                    **verdict(y4["f1"]["value"] - REF["strat_f1"],
                              [x - REF["strat_f1"] for x in screen_interval(y4["f1"], "f1", 1, stratified=True)],
                              THRESH["H15_delta"], "<="),
                    y4_f1=y4["f1"]["value"], y4_ci_kernels=y4["f1"]["ci_kernels"],
                    note="Y4 is scored on its own 20 % held-out rows (1,725 kernels), not X1's 30 % (2,588): unpaired"),
    }
    for h in ("H19a", "H21a", "H21b"):
        H[h]["outcome"] = "supported" if H[h]["met"] else "rejected"
    H["H19b"]["outcome"] = "supported" if (H["H19b"]["cross"]["met"] and H["H19b"]["attraction"]["met"]) else "rejected"
    H["H20"]["outcome"] = "supported" if (H["H20"]["cross"]["met"] and H["H20"]["same"]["met"]) else "rejected"
    H["H15"]["outcome"] = "supported" if H["H15"]["met"] else "rejected"

    # the frozen decision rule (preregistration_s12.json → decision_rule), read with D28's screening semantics
    y2_cross_gain = y2["cross"]["value"] >= THRESH["H20_cross"]
    y2_same_loss = y2["same"]["value"] < THRESH["H20_same"]
    H["decision_routes"] = {
        "Y1": ("H19a ∧ ¬H19b → 'no adoption; the joint network stays; record that decoupling does not buy session "
               "robustness under R1' — a screening rejection (not replicated unless a later frozen file gives a reason)")
        if H["H19a"]["outcome"] == "supported" and H["H19b"]["outcome"] == "rejected" else "see values",
        "Y2": ("H20 supported → passes the screen" if H["H20"]["outcome"] == "supported" else
               "H20 rejected with cross ≥ +0.020 but same < −0.020 → frontier move only" if (y2_cross_gain and y2_same_loss)
               else "H20 rejected otherwise → 'instance statistics are not separable into session and variety here' — "
                    "a screening rejection"),
        "Y3": ("H21a ∧ H21b → 'adopt the lean architecture as the base of every later arm (SeedNet v5)' read under "
               "D28 as PASSES THE SCREEN: replicate at seeds 1–2 (FW-35) before it becomes the reference, SeedNet v5 "
               "or a paper claim; it may serve as the provisional base of later screening arms")
        if H["H21a"]["outcome"] == "supported" and H["H21b"]["outcome"] == "supported" else "see values",
        "Y4": "H15 supported → the 80/20 tier is the within-acquisition (D16 tier-1) number; no adoption attached",
        "D28_reversal_check": {
            arm: {k: [S[n].stats()[k] for n in arms[arm]] for k in keys}
            for arm, keys in (("Y1 fused (calib w)", ("f1", "cross", "attraction")), ("Y2 mixstyle", ("cross", "same")),
                              ("Y3 lean grouped", ("f1",)))},
        "D28_reversal_note": "D28: replicate before reading if an arm's two fold estimates straddle its threshold by "
                             "more than the margin. No arm's folds straddle any threshold (see per-fold values).",
    }
    save(H, "hypotheses.json")
    for h in ("H19a", "H19b", "H20", "H21a", "H21b", "H15"):
        v = H[h]
        print(h, v["outcome"], {k: (round(x["value"], 4), x["met"], x["clarity"]) for k, x in v.items()
                                if isinstance(x, dict) and "met" in x} or (round(v["value"], 4), v["clarity"]))
    print(pd.DataFrame(summary)[["arm", "f1", "same", "cross", "attraction", "d_f1_vs_x1s0", "d_f1_vs_x1s0_lo",
                                 "d_f1_vs_x1s0_hi", "d_cross_vs_x1s0", "d_cross_vs_x1s0_lo", "d_cross_vs_x1s0_hi"]]
          .round(4).to_string())


if __name__ == "__main__":
    main()
