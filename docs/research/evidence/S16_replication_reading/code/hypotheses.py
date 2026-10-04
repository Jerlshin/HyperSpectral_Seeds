"""S16 · read H21a–H21e and H22a/H22b exactly as frozen (preregistration_s14.json, 9e182670…; reading map S15 §8).

The script refuses to run if any of the three frozen hashes moved. Nothing is selected on held-out: every prediction
was produced once by its cell's own final evaluation; this script only aggregates them.

Uncertainty:
  * ``ci`` — hierarchical bootstrap (2,000 resamples; runs within fold and arm, held-out kernels within fold, shared by
    both arms of a difference), as S12 and as the frozen metric clause asks for H21c. Reported beside every quantity.
  * ``screen_interval`` (the one-seed dissection, D28) — as S14: kernel bootstrap combined analytically with X1's
    run-level sd / √n runs. A verdict is *clear* when the interval excludes the threshold, *marginal* otherwise.
  * D28's reversal clause: an arm whose two fold estimates straddle its threshold by more than the margin is
    replicated before its verdict is read. Checked for every hypothesis read on 2 runs.

Beside the frozen hypotheses (decide nothing): all-seed Y3 vs X1 deltas (G3), the seed-0 cells' rank within Y3
(winner's curse), Y3's run-level sd (the margin basis of any later round), the dissection arms seed-matched to X1 s0
and Y3 s0 on the same kernels.

Writes: hypotheses.json · arm_summary.csv · matched_deltas.csv · seed_variance.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from s16common import (
    FRESH_REF,
    REF,
    THRESH,
    Scored,
    hboot,
    s15_cells,
    save,
    verify_preregistrations,
    x1_cells,
    y3_cells,
)

KEYS = ("f1", "same", "cross", "attraction")
SEED_SD = {"f1": REF["f1_sd"], "same": REF["same_sd"], "cross": REF["cross_sd"], "attraction": REF["attraction_sd"]}


def verdict(value: float, thr: float, direction: str, interval: list[float] | None = None) -> dict:
    ok = value >= thr if direction == ">=" else value <= thr
    out = dict(value=value, threshold=thr, direction=direction, met=bool(ok))
    if interval is not None:
        clear = (interval[0] >= thr or interval[1] < thr) if direction == ">=" else (interval[1] <= thr or interval[0] > thr)
        out.update(interval=interval, clarity="clear" if clear else "marginal")
    return out


def screen_interval(runs: list[Scored], key: str) -> list[float]:
    """S14's screen interval: kernel-only bootstrap of the mean + X1 seed sd / √n (one seed per fold)."""
    b = hboot(runs, n=2000, seed=161)[key]          # one run per fold → runs resampling is the identity
    hw = 1.96 * np.sqrt(b["se"] ** 2 + SEED_SD[key] ** 2 / len(runs))
    return [b["value"] - hw, b["value"] + hw]


def straddle(per_fold: list[float], thr: float, margin: float = 0.020) -> dict:
    lo, hi = min(per_fold), max(per_fold)
    st = lo < thr <= hi
    return dict(per_fold=per_fold, straddles=bool(st), below_by=float(thr - lo) if st else 0.0,
                above_by=float(hi - thr) if st else 0.0, margin=margin,
                fires=bool(st and (thr - lo > margin or hi - thr > margin)))


def row(name: str, runs: list[Scored], ref: list[Scored] | None = None, ref_name: str = "") -> dict:
    own = hboot(runs)
    r = {"arm": name, "n_runs": len(runs), "cells": " ".join(s.cell.name for s in runs)}
    for k, v in own.items():
        r[k], (r[f"{k}_lo"], r[f"{k}_hi"]) = v["value"], v["ci"]
        r[f"{k}_runs"] = " ".join(f"{s.stats()[k]:.4f}" for s in runs)
    if ref is not None:
        d = hboot(runs, ref)
        r["vs"] = ref_name
        for k, v in d.items():
            r[f"d_{k}"], (r[f"d_{k}_lo"], r[f"d_{k}_hi"]) = v["value"], v["ci"]
    return r


def main() -> None:
    verify_preregistrations()
    S = lambda cs: [Scored(c) for c in cs]  # noqa: E731
    y3g_all, y3g_fresh = S(y3_cells((0, 1, 2))), S(y3_cells((1, 2)))
    y3s_all, y3s_fresh = S(y3_cells((0, 1, 2), "lean_stratified")), S(y3_cells((1, 2), "lean_stratified"))
    y3g_s0 = S(y3_cells((0,)))
    x1g_all, x1g_fresh, x1g_s0 = S(x1_cells((0, 1, 2))), S(x1_cells((1, 2))), S(x1_cells((0,)))
    x1s_all, x1s_fresh = S(x1_cells((0, 1, 2), "stratified")), S(x1_cells((1, 2), "stratified"))
    y5 = {v: S(sorted((c for c in s15_cells() if c.variant == v), key=lambda c: c.fold))
          for v in ("desc_only", "spatial_repair")}

    arms = [
        row("Y3 grouped, seeds 0–2 (H21a)", y3g_all, x1g_all, "X1 grouped seeds 0–2"),
        row("Y3 grouped, fresh seeds 1–2 (H21c, H21d)", y3g_fresh, x1g_fresh, "X1 grouped seeds 1–2"),
        row("Y3 grouped, seed 0 (S13)", y3g_s0, x1g_s0, "X1 grouped seed 0"),
        row("Y3 stratified, seeds 0–2 (H21b)", y3s_all, x1s_all, "X1 stratified seeds 0–2"),
        row("Y3 stratified, fresh seeds 1–2 (H21e)", y3s_fresh, x1s_fresh, "X1 stratified seeds 1–2"),
        row("Y5 desc_only, seed 0 (H22a)", y5["desc_only"], x1g_s0, "X1 grouped seed 0"),
        row("Y5 spatial_repair, seed 0 (H22b)", y5["spatial_repair"], x1g_s0, "X1 grouped seed 0"),
        row("X1 grouped, seeds 0–2", x1g_all), row("X1 grouped, fresh seeds 1–2", x1g_fresh),
        row("X1 stratified, seeds 0–2", x1s_all),
    ]
    A = {a["arm"].split(" (")[0]: a for a in arms}
    save(pd.DataFrame(arms), "arm_summary.csv")

    # matched (fold- and seed-0) deltas of the dissection arms against X1 s0 and Y3 s0, same kernels
    matched = []
    for runs in y5.values():
        for s in runs:
            for refname, refs in (("X1 s0", x1g_s0), ("Y3 s0", y3g_s0)):
                ref = next(x for x in refs if x.cell.fold == s.cell.fold)
                d = hboot([s], [ref], seed=162)
                matched.append({"cell": s.cell.name, "vs": f"{refname} f{s.cell.fold}",
                                **{k: s.stats()[k] for k in KEYS}, **{f"ref_{k}": ref.stats()[k] for k in KEYS},
                                **{f"d_{k}": d[k]["value"] for k in KEYS}, **{f"d_{k}_lo": d[k]["ci"][0] for k in KEYS},
                                **{f"d_{k}_hi": d[k]["ci"][1] for k in KEYS}})
    save(pd.DataFrame(matched), "matched_deltas.csv")

    # seed variance (run level) — Y3 vs X1, grouped per fold and pooled, stratified
    sv = []
    for name, runs in (("Y3", y3g_all), ("X1", x1g_all)):
        df = pd.DataFrame([dict(fold=s.cell.fold, seed=s.cell.seed, **s.stats()) for s in runs])
        for k in KEYS:
            within = df.groupby("fold")[k].std(ddof=1)
            sv.append(dict(arm=name, protocol="grouped", metric=k, sd_pooled_6=float(df[k].std(ddof=1)),
                           sd_within_fold_f0=float(within.loc[0]), sd_within_fold_f1=float(within.loc[1]),
                           sd_within_fold_rms=float(np.sqrt((within ** 2).mean())),
                           seed0_rank_f0=int((df[df.fold == 0].sort_values(k, ascending=False).seed.tolist()).index(0)) + 1,
                           seed0_rank_f1=int((df[df.fold == 1].sort_values(k, ascending=False).seed.tolist()).index(0)) + 1))
    for name, runs in (("Y3", y3s_all), ("X1", x1s_all)):
        vals = pd.DataFrame([dict(seed=s.cell.seed, f1=s.stats()["f1"]) for s in runs])
        sv.append(dict(arm=name, protocol="stratified", metric="f1", sd_pooled_6=float(vals.f1.std(ddof=1)),
                       seed0_rank_f0=int(vals.sort_values("f1", ascending=False).seed.tolist().index(0)) + 1))
    sv = pd.DataFrame(sv)
    save(sv, "seed_variance.csv")

    # ── the frozen hypotheses ──
    ga, gf, sa, sf = A["Y3 grouped, seeds 0–2"], A["Y3 grouped, fresh seeds 1–2"], A["Y3 stratified, seeds 0–2"], \
        A["Y3 stratified, fresh seeds 1–2"]
    H: dict = {}
    H["H21a"] = dict(frozen="(parent) Y3 grouped macro-F1, mean of 6 runs (folds 0,1 × seeds 0,1,2) ≥ 0.5108",
                     **verdict(ga["f1"], THRESH["H21a"], ">=", [ga["f1_lo"], ga["f1_hi"]]), runs=ga["f1_runs"])
    H["H21b"] = dict(frozen="(parent) Y3 stratified macro-F1, mean of 3 runs (seeds 0,1,2) ≥ 0.7090",
                     **verdict(sa["f1"], THRESH["H21b"], ">=", [sa["f1_lo"], sa["f1_hi"]]), runs=sa["f1_runs"])
    H["H21c"] = dict(frozen="Y3 grouped mean (seeds 1,2; 4 runs) − X1 (same folds/seeds; 0.528496) ≥ +0.020 AND the "
                            "hierarchical bootstrap 95 % CI of the difference excludes 0",
                     delta=verdict(gf["d_f1"], THRESH["H21c_delta"], ">=", [gf["d_f1_lo"], gf["d_f1_hi"]]),
                     ci_excludes_0=bool(gf["d_f1_lo"] > 0), y3=gf["f1"], x1=gf["f1"] - gf["d_f1"],
                     x1_matches_frozen=abs((gf["f1"] - gf["d_f1"]) - FRESH_REF["f1"]) < 1e-6)
    H["H21d"] = dict(frozen="Y3 grouped (seeds 1,2) cross-session recall mean ≥ 0.1666 AND attraction mean ≤ 0.4240",
                     cross=verdict(gf["cross"], THRESH["H21d_cross"], ">=", [gf["cross_lo"], gf["cross_hi"]]),
                     attraction=verdict(gf["attraction"], THRESH["H21d_attraction"], "<=",
                                        [gf["attraction_lo"], gf["attraction_hi"]]),
                     runs_cross=gf["cross_runs"], runs_attraction=gf["attraction_runs"],
                     beside_vs_x1_fresh=dict(d_cross=[gf["d_cross"], gf["d_cross_lo"], gf["d_cross_hi"]],
                                             d_attraction=[gf["d_attraction"], gf["d_attraction_lo"], gf["d_attraction_hi"]]))
    H["H21e"] = dict(frozen="Y3 stratified mean (seeds 1,2) − X1 stratified (seeds 1,2; 0.728593) ≥ +0.018",
                     **verdict(sf["d_f1"], THRESH["H21e_delta"], ">=", [sf["d_f1_lo"], sf["d_f1_hi"]]),
                     y3=sf["f1"], x1=sf["f1"] - sf["d_f1"], runs=sf["f1_runs"],
                     x1_matches_frozen=abs((sf["f1"] - sf["d_f1"]) - FRESH_REF["strat_f1"]) < 1e-6,
                     note="the interval is the hierarchical bootstrap of the difference (2 runs per arm); the frozen "
                          "rule reads the point value only")
    for h, v in (("H22a", "desc_only"), ("H22b", "spatial_repair")):
        a = A[f"Y5 {v}, seed 0"]
        pf = [s.stats()["f1"] for s in y5[v]]
        H[h] = dict(frozen=f"Dissection (screen): Y5 {v} grouped mean (seed 0, 2 runs) ≥ 0.530786 + 0.020 = 0.5508",
                    **verdict(a["f1"], THRESH["H22"], ">=", screen_interval(y5[v], "f1")),
                    d28_straddle=straddle(pf, THRESH["H22"]),
                    beside=dict(d_f1_vs_x1s0=a["d_f1"], d_cross_vs_x1s0=a["d_cross"],
                                d_attraction_vs_x1s0=a["d_attraction"], cross=a["cross"], attraction=a["attraction"]))
    for h in ("H21a", "H21b", "H21e", "H22a", "H22b"):
        H[h]["outcome"] = "supported" if H[h]["met"] else "rejected"
    H["H21c"]["outcome"] = "supported" if (H["H21c"]["delta"]["met"] and H["H21c"]["ci_excludes_0"]) else "rejected"
    H["H21d"]["outcome"] = "supported" if (H["H21d"]["cross"]["met"] and H["H21d"]["attraction"]["met"]) else "rejected"

    a_, b_ = H["H22a"]["outcome"] == "supported", H["H22b"]["outcome"] == "supported"
    H["decision_routes"] = {
        "H21a ∧ H21b": "adopt: the lean architecture becomes the reference form (SeedNet v5) and the base of every later "
                       "arm; config defaults → R1 + lean keys behind G-neutral (D24, D29 deviation 1); the paper drops "
                       "the index-bank/continuum/derivative claims"
        if H["H21a"]["outcome"] == H["H21b"]["outcome"] == "supported" else "F74 was a false screening pass",
        "H21c": "the paper may claim a grouped gain over X1 (G3 met on fresh seeds)" if H["H21c"]["outcome"] == "supported"
                else "non-inferior; gain not confirmed",
        "H21d": "the paper may claim raised cross-session recall and lowered attraction" if H["H21d"]["outcome"] == "supported"
                else "robustness gain not confirmed",
        "H21e": "within-acquisition gain confirmed" if H["H21e"]["outcome"] == "supported" else "not claimed",
        "H22": ("H22a and H22b → either removal alone suffices (non-additive); both kept; attribution reported as redundant"
                if a_ and b_ else "H22a xor H22b → the passing component carries the gain" if a_ != b_
                else "neither → interaction or favourable seed 0; undetermined"),
        "H22_reading_condition": "read as attribution because H21a ∧ H21b hold" if
        H["H21a"]["outcome"] == H["H21b"]["outcome"] == "supported" else "descriptive only",
        "D28_reversal_note": "D28: replicate before reading if an arm's two fold estimates straddle its threshold by "
                             "more than the margin (0.020). See H22a/H22b → d28_straddle.",
    }
    beside = {
        "G3_all_seeds": dict(d_f1=ga["d_f1"], ci=[ga["d_f1_lo"], ga["d_f1_hi"]],
                             two_sigma=2 * float(sv[(sv.arm == "X1") & (sv.metric == "f1") & (sv.protocol == "grouped")].sd_pooled_6.iloc[0]),
                             d_cross=ga["d_cross"], d_attraction=ga["d_attraction"], d_same=ga["d_same"],
                             strat_d_f1=sa["d_f1"], strat_ci=[sa["d_f1_lo"], sa["d_f1_hi"]]),
        "winners_curse": "seed-0 Y3 F1 rank among Y3 seeds: f0 {} / f1 {} of 3".format(
            *sv[(sv.arm == "Y3") & (sv.metric == "f1") & (sv.protocol == "grouped")][["seed0_rank_f0", "seed0_rank_f1"]].iloc[0]),
        "seed0_vs_fresh_y3": dict(seed0=A["Y3 grouped, seed 0"]["f1"], fresh=gf["f1"]),
    }
    H["beside_deciding_nothing"] = beside
    save(H, "hypotheses.json")

    for h in ("H21a", "H21b", "H21c", "H21d", "H21e", "H22a", "H22b"):
        v = H[h]
        parts = {k: (round(x["value"], 4), x["met"], x.get("clarity")) for k, x in v.items() if isinstance(x, dict) and "met" in x}
        print(h, v["outcome"], parts or (round(v["value"], 4), v.get("clarity")), v.get("d28_straddle", ""))
    print(pd.DataFrame(arms)[["arm", "f1", "f1_lo", "f1_hi", "same", "cross", "attraction", "d_f1", "d_f1_lo", "d_f1_hi",
                              "d_cross", "d_cross_lo", "d_cross_hi", "d_attraction"]].round(4).to_string())
    print(sv.round(4).to_string())
    print(pd.DataFrame(matched)[["cell", "vs", "f1", "d_f1", "d_f1_lo", "d_f1_hi", "d_cross", "d_cross_lo", "d_cross_hi",
                                 "d_attraction"]].round(4).to_string())
    print(beside)


if __name__ == "__main__":
    main()
