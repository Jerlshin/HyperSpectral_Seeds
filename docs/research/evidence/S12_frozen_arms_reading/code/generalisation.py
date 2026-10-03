"""S12 · where accuracy is lost now that the network fits: the generalisation ladder, and F46 retested.

Ladder (per run, selected checkpoint): clean training accuracy (D18 subset; S08: S10's full-training-set
value) → calib accuracy (unseen kernels, same bundle) → held-out same-session recall → held-out
cross-session recall. Each rung is a different kind of generalisation: memorisation → within-acquisition
sampling → bundle shift → session shift.

F46 (S10) said held-out F1 rises ≈ 1 : 1 with clean fit within acquisition (S08 stratified seeds,
r 0.99, slope 1.01). X1 is an *intervention* on fit; this script compares the slope it predicts with what
X1 delivered, and locates when calib F1 stops rising relative to clean fit inside each X1 run.

Writes: ladder.csv · ladder_summary.csv · fit_transfer.json · calib_saturation.csv · decomposition_x1.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from s12common import EVID, REPO, save

S09 = REPO / "docs/research/evidence/S09_post_sweep_forensics"
S10 = REPO / "docs/research/evidence/S10_training_architecture_review"


def main() -> None:
    cdf = pd.read_csv(EVID / "cells.csv")
    cv = pd.read_csv(EVID / "curves.csv")
    s09 = pd.read_csv(S09 / "runs.csv")
    fit = pd.read_csv(S10 / "ckpt_fit.csv")

    # ── ladder ────────────────────────────────────────────────────────────────────────────────────────
    rows = []
    for _, r in s09.iterrows():
        f = fit[(fit.arm == r.arm) & (fit.fold == r.fold) & (fit.seed == r.seed)]
        src = f.best_source.iloc[0]
        tr = f[(f.weights == src) & (f.split == "train")].acc.iloc[0]
        rows.append(dict(arm="S08 shipped", protocol=r.arm, cell=f"S08/{r.arm}__f{r.fold}_s{r.seed}",
                         clean_train=tr, calib_acc=r.calib_best_acc, heldout_f1=r.f1_tta,
                         heldout_acc=r.acc_tta, same_recall=r.same_recall_tta,
                         cross_recall=r.cross_recall_tta if r.arm == "grouped" else np.nan))
    for _, r in cdf[cdf.scored].iterrows():
        rows.append(dict(arm=f"{r.arm} {r.variant}" if r.arm == "X2" else r.arm, protocol=r.protocol,
                         cell=r.cell, clean_train=r.clean_best_selected, calib_acc=r.calib_acc_selected,
                         heldout_f1=r.f1_tta, heldout_acc=r.acc_tta, same_recall=r.same_recall_tta,
                         cross_recall=r.cross_recall_tta))
    lad = pd.DataFrame(rows)
    lad["gap_memorisation"] = lad.clean_train - lad.calib_acc          # train → unseen kernels, same bundle
    lad["gap_bundle"] = lad.calib_acc - lad.same_recall               # → other bundle, same session
    lad["gap_session"] = lad.same_recall - lad.cross_recall           # → other session
    save(lad, "ladder.csv")
    summ = lad.groupby(["arm", "protocol"]).agg(
        n=("cell", "size"), clean_train=("clean_train", "mean"), calib_acc=("calib_acc", "mean"),
        same_recall=("same_recall", "mean"), cross_recall=("cross_recall", "mean"),
        heldout_f1=("heldout_f1", "mean"), gap_memorisation=("gap_memorisation", "mean"),
        gap_bundle=("gap_bundle", "mean"), gap_session=("gap_session", "mean")).reset_index()
    save(summ, "ladder_summary.csv")

    # ── F46 retested: does the S08 fit→held-out slope predict the X1 intervention? ─────────────────────
    out = {}
    for proto in ("stratified", "grouped"):
        s08 = lad[(lad.arm == "S08 shipped") & (lad.protocol == proto)]
        x1 = lad[(lad.arm == "X1") & (lad.protocol == proto)]
        slope, icpt = np.polyfit(s08.clean_train, s08.heldout_f1, 1)
        r = float(np.corrcoef(s08.clean_train, s08.heldout_f1)[0, 1])
        pred = icpt + slope * x1.clean_train.mean()
        both = pd.concat([s08, x1])
        out[proto] = dict(
            s08_slope=slope, s08_r=r, s08_fit_mean=s08.clean_train.mean(), s08_f1_mean=s08.heldout_f1.mean(),
            x1_fit_mean=x1.clean_train.mean(), x1_f1_mean=x1.heldout_f1.mean(),
            predicted_x1_f1_from_s08_line=pred, observed_minus_predicted=x1.heldout_f1.mean() - pred,
            realised_slope=(x1.heldout_f1.mean() - s08.heldout_f1.mean()) / (x1.clean_train.mean() - s08.clean_train.mean()),
            pooled_r=float(np.corrcoef(both.clean_train, both.heldout_f1)[0, 1]),
            x1_within_r=float(np.corrcoef(x1.clean_train, x1.heldout_f1)[0, 1]) if len(x1) > 2 else None,
        )
    out["note"] = ("S08 within-regime slope is the F46 relationship (seeds that happened to fit better). X1 raised "
                   "fit by intervention; 'realised_slope' is Δheld-out / Δfit between the regimes' means.")
    save(out, "fit_transfer.json")

    # ── when does calib stop improving, relative to clean fit? (X1, X4; per run) ───────────────────────
    sat = []
    for cell, g in cv[cv.cell.str.startswith(("X1/", "X4/", "X2/"))].groupby("cell"):
        g = g.sort_values("epoch")
        calib = g[["val/f1_live", "val/f1_ema"]].max(axis=1).rolling(5, min_periods=1).mean()
        fitc = g[["fit/clean_train_acc_live", "fit/clean_train_acc_ema"]].max(axis=1).interpolate()
        top = calib.max()
        e95 = int(g.epoch.iloc[int(np.argmax(calib.values >= top - 0.01))])  # first epoch within 0.01 of the run's best (5-ep mean)
        fit_at = float(fitc[g.epoch == e95].iloc[0])
        sat.append(dict(cell=cell, calib_best_5ep=top, epoch_within_0p01=e95, clean_fit_then=fit_at,
                        clean_fit_final=float(fitc.iloc[-1]), lr_frac_then=float(g.loc[g.epoch == e95, "sched/lr"].iloc[0]) / 5e-4,
                        mixup_then=float(g.loc[g.epoch == e95, "sched/mixup"].iloc[0])))
    save(pd.DataFrame(sat), "calib_saturation.csv")

    # ── S09's shortfall decomposition, recomputed for X1 ───────────────────────────────────────────────
    x1g = lad[(lad.arm == "X1") & (lad.protocol == "grouped")]
    import json
    s09dec = json.loads((S09 / "decomposition.json").read_text())
    # stratified same/cross recall for X1 come from the per-class tables (cross varieties defined by S09)
    from s12common import CROSS, SAME, cells, load_preds, recalls
    strat = [c for c in cells(("X1",)) if c.variant == "stratified"]
    rec = []
    for c in strat:
        _, p, t = load_preds(c, "tta")
        rec.append(recalls(t, p))
    rec = np.mean(rec, axis=0)
    st_same, st_cross = float(rec[SAME].mean()), float(rec[CROSS].mean())
    g_same, g_cross = float(x1g.same_recall.mean()), float(x1g.cross_recall.mean())
    w_same, w_cross = len(SAME) / 90, len(CROSS) / 90
    parts = {
        "same-session · already lost in-distribution": w_same * (1 - st_same),
        "same-session · lost to the bundle shift": w_same * (st_same - g_same),
        "cross-session · already lost in-distribution": w_cross * (1 - st_cross),
        "cross-session · lost to the session shift": w_cross * (st_cross - g_cross),
    }
    tot = sum(parts.values())
    save({"x1_stratified_recall": {"same": st_same, "cross": st_cross},
          "x1_grouped_recall": {"same": g_same, "cross": g_cross},
          "shortfall_total": tot, "parts": {k: {"value": v, "share": v / tot} for k, v in parts.items()},
          "in_distribution_share": (parts["same-session · already lost in-distribution"]
                                    + parts["cross-session · already lost in-distribution"]) / tot,
          "s08_for_comparison": s09dec["parts"]}, "decomposition_x1.json")
    print(summ.round(3).to_string())
    print(pd.DataFrame(sat).round(3).to_string())
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
