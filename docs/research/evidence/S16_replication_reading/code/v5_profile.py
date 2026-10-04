"""S16 · the lean network (Y3) at three seeds against X1 at three seeds — from saved artifacts only (no model is run).

S14 profiled Y3 from one seed; with seeds 1–2 the same questions get seed-level answers. Decides nothing; the frozen
verdicts are hypotheses.py's.

1. Separation: every Y3 run against every X1 run of the same fold/protocol (exact permutation p of "all Y3 above all
   X1" under exchangeability = 1 / C(6, 3) per cell).
2. Ensembles: 3-seed softmax average (TTA logits) per fold, Y3 vs X1 — how much of each arm's error is seed noise.
3. Error structure: Jaccard of error sets between seeds, within and across arms; share of errors made by all 3 seeds.
4. Who gains: per-class recall (3-seed mean) Y3 − X1; cross-session accuracy by acquisition session.
5. The ladder: clean fit → calib F1 → held-out same → cross, 3-seed means; held-out ECE.
6. The linear bar: no-TTA F1 against shrinkage LDA on within-kernel quantiles + morph (S12 F67, 0.518).
7. Pathway use across seeds: spatial influence at the selected epoch and training-rows κ (in-pipeline, S13/S15).

Writes: v5_separation.csv · v5_ensembles.csv · v5_errors.csv · v5_per_class.csv · v5_sessions.csv · v5_profile.json
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from s16common import (
    CROSS,
    EVID,
    S12_EVID,
    S14_EVID,
    S14_LEAN,
    SESSION,
    Scored,
    conf,
    f1_from_conf,
    grouped_rows,
    load_logits,
    recall_from_conf,
    recalls,
    save,
    softmax,
    train_session_by_class,
    verify_preregistrations,
    x1_cells,
    y3_cells,
)


def main() -> None:
    verify_preregistrations()
    cells = pd.concat([pd.read_csv(EVID / "cells.csv"), pd.read_csv(S14_EVID / "cells.csv"),
                       pd.read_csv(S12_EVID / "cells.csv")], ignore_index=True).drop_duplicates("cell").set_index("cell")
    cv12 = pd.read_csv(S12_EVID / "curves.csv")
    G = {"Y3": y3_cells(), "X1": x1_cells()}
    ST = {"Y3": y3_cells(variant="lean_stratified"), "X1": x1_cells(variant="stratified")}
    SC = {a: [Scored(c) for c in cs] for a, cs in {**{f"{k} g": v for k, v in G.items()},
                                                    **{f"{k} s": v for k, v in ST.items()}}.items()}

    # 1 · separation
    sep, p_joint = [], 1.0
    for proto, folds in (("grouped", (0, 1)), ("stratified", (0,))):
        tag = "g" if proto == "grouped" else "s"
        for f in folds:
            y = [s.stats()["f1"] for s in SC[f"Y3 {tag}"] if s.cell.fold == f]
            x = [s.stats()["f1"] for s in SC[f"X1 {tag}"] if s.cell.fold == f]
            above = min(y) > max(x)
            p = 1 / comb(len(x) + len(y), len(y))
            p_joint *= p if above else 1.0
            sep.append(dict(protocol=proto, fold=f, y3=" ".join(f"{v:.4f}" for v in y),
                            x1=" ".join(f"{v:.4f}" for v in x), y3_min=min(y), x1_max=max(x), gap=min(y) - max(x),
                            all_y3_above_all_x1=above, p_exchangeable=p))
    sep = pd.DataFrame(sep)
    save(sep, "v5_separation.csv")

    # 2 · ensembles (grouped f0/f1, stratified f0)
    ens = []
    for proto, tag, folds in (("grouped", "g", (0, 1)), ("stratified", "s", (0,))):
        for arm in ("Y3", "X1"):
            for f in folds:
                cs = [c for c in (G if tag == "g" else ST)[arm] if c.fold == f]
                P, rows, t = None, None, None
                for c in cs:
                    r, lg, tt = load_logits(c, "val_test_tta")
                    pr = softmax(lg)
                    if P is None:
                        P, rows, t = pr, r, tt
                    else:
                        assert np.array_equal(rows, r)
                        P = P + pr
                pred = P.argmax(1)
                cm = conf(t, pred)
                singles = [s.stats() for s in SC[f"{arm} {tag}"] if s.cell.fold == f]
                r_ = dict(protocol=proto, arm=arm, fold=f, n_seeds=len(cs), ens_f1=f1_from_conf(cm),
                          mean_single_f1=float(np.mean([s["f1"] for s in singles])))
                if tag == "g":
                    soc = train_session_by_class(grouped_rows(f)[0])
                    m = np.isin(t, CROSS)
                    r_.update(ens_cross=recall_from_conf(cm, CROSS), ens_attraction=float((soc[pred[m]] == SESSION[rows[m]]).mean()),
                              mean_single_cross=float(np.mean([s["cross"] for s in singles])),
                              mean_single_attraction=float(np.mean([s["attraction"] for s in singles])))
                r_["ens_gain_f1"] = r_["ens_f1"] - r_["mean_single_f1"]
                ens.append(r_)
    ens = pd.DataFrame(ens)
    save(ens, "v5_ensembles.csv")

    # 3 · error structure (grouped)
    err = []
    for f in (0, 1):
        W = {}
        for arm in ("Y3", "X1"):
            for s in SC[f"{arm} g"]:
                if s.cell.fold == f:
                    W[f"{arm} s{s.cell.seed}"] = s.p != s.t
        for a, b in combinations(sorted(W), 2):
            kind = "within Y3" if a[:2] == b[:2] == "Y3" else "within X1" if a[:2] == b[:2] == "X1" else "Y3 vs X1"
            err.append(dict(fold=f, a=a, b=b, kind=kind, jaccard=float((W[a] & W[b]).sum() / (W[a] | W[b]).sum())))
        for arm in ("Y3", "X1"):
            ws = [v for k, v in W.items() if k.startswith(arm)]
            anyw, allw = np.logical_or.reduce(ws), np.logical_and.reduce(ws)
            err.append(dict(fold=f, a=arm, b="all 3 seeds", kind="shared by all seeds", jaccard=float(allw.sum() / anyw.sum()),
                            mean_errors=float(np.mean([w.sum() for w in ws]))))
    err = pd.DataFrame(err)
    save(err, "v5_errors.csv")

    # 4 · per-class recall (3-seed mean per fold, then mean of folds) and cross-session by session
    pc = []
    for arm in ("Y3", "X1"):
        for s in SC[f"{arm} g"]:
            rc = recalls(s.t, s.p)
            pc += [dict(arm=arm, fold=s.cell.fold, seed=s.cell.seed, cls=k, recall=rc[k]) for k in range(90)]
    pc = pd.DataFrame(pc).groupby(["arm", "cls"]).recall.mean().unstack(0).reset_index()
    pc["cross"] = pc.cls.isin(CROSS)
    pc["delta"] = pc.Y3 - pc.X1
    save(pc, "v5_per_class.csv")
    sess = []
    for arm in ("Y3", "X1"):
        for s in SC[f"{arm} g"]:
            m = s.is_cross
            for ss in np.unique(SESSION[s.rows[m]]):
                k = m & (SESSION[s.rows] == ss)
                sess.append(dict(arm=arm, fold=s.cell.fold, seed=s.cell.seed, session=int(ss), n=int(k.sum()),
                                 acc=float((s.p[k] == s.t[k]).mean())))
    sess = pd.DataFrame(sess).groupby(["fold", "session", "n", "arm"]).acc.mean().unstack("arm").reset_index()
    sess["delta"] = sess.Y3 - sess.X1
    save(sess, "v5_sessions.csv")

    # 5 · ladder, ECE · 6 · linear bar · 7 · pathway use
    lad = []
    for arm, cs in (("Y3", G["Y3"] + ST["Y3"]), ("X1", G["X1"] + ST["X1"])):
        for c in cs:
            src = cells.loc[c.name]
            _, lg, t = load_logits(c, "val_test_tta")
            infl = (src.get("influence_branch_spatial_at_best") if arm == "Y3" else
                    cv12[(cv12.cell == c.name) & (cv12.epoch == src.best_epoch)]["influence/branch_spatial"].iloc[0])
            lad.append(dict(arm=arm, cell=c.name, protocol="stratified" if "strat" in c.variant else "grouped",
                            fold=c.fold, seed=c.seed, clean_fit=src.clean_best_selected, calib_f1=src.calib_f1_selected,
                            heldout_same=src.same_recall_tta, heldout_cross=src.cross_recall_tta, f1_tta=src.f1_tta,
                            f1_no_tta=src.f1_no_tta, best_epoch=src.best_epoch, stop_epoch=src.stop_epoch,
                            ece=S14_LEAN.ece(softmax(lg), t), influence_spatial=infl,
                            kappa_embedding=src.get("kappa_embedding"), kappa_spatial=src.get("kappa_spatial"),
                            kappa_spectral=src.get("kappa_spectral")))
    lad = pd.DataFrame(lad)
    summ = lad.groupby(["arm", "protocol"])[["clean_fit", "calib_f1", "heldout_same", "heldout_cross", "f1_tta",
                                             "f1_no_tta", "ece", "influence_spatial", "kappa_embedding",
                                             "kappa_spatial", "kappa_spectral", "best_epoch"]].mean()
    qlda = pd.read_csv(S12_EVID / "pixel_controls.csv")
    q = qlda[(qlda.representation.str.startswith("quantiles")) & (qlda.protocol == "grouped")].heldout_f1.mean()
    out = dict(
        separation_p_joint=p_joint,
        separation_note="exact permutation probability that all 3 Y3 runs exceed all 3 X1 runs in every cell, if the "
                        "6 runs of a cell were exchangeable (1/20 per cell); seed-to-seed spread only",
        ladder=summ.round(4).reset_index().to_dict("records"),
        per_class=dict(n_up_gt_0_10=int((pc.delta > 0.10).sum()), n_down_lt_m0_10=int((pc.delta < -0.10).sum()),
                       n_up=int((pc.delta > 0).sum()), n_down=int((pc.delta < 0).sum()),
                       mean_delta_same=float(pc[~pc.cross].delta.mean()), mean_delta_cross=float(pc[pc.cross].delta.mean()),
                       cross_classes=pc[pc.cross].sort_values("delta")[["cls", "X1", "Y3", "delta"]].round(3).to_dict("records")),
        linear_bar=dict(quantile_lda_grouped=float(q), y3_no_tta=float(lad[(lad.arm == "Y3") & (lad.protocol == "grouped")].f1_no_tta.mean()),
                        x1_no_tta=float(lad[(lad.arm == "X1") & (lad.protocol == "grouped")].f1_no_tta.mean())),
        error_structure=err.groupby("kind").jaccard.mean().round(4).to_dict(),
        ensembles=ens.groupby(["protocol", "arm"])[[c for c in ens.columns if c.startswith(("ens_", "mean_single"))]].mean().round(4).reset_index().to_dict("records"),
    )
    out["linear_bar"]["y3_minus_qlda"] = out["linear_bar"]["y3_no_tta"] - q
    out["linear_bar"]["x1_minus_qlda"] = out["linear_bar"]["x1_no_tta"] - q
    save(lad, "v5_ladder.csv")
    save(out, "v5_profile.json")
    print(sep.round(4).to_string())
    print(ens.round(4).to_string())
    print(err.groupby("kind").jaccard.agg(["mean", "min", "max"]).round(4))
    print(err[err.kind == "shared by all seeds"].round(3).to_string())
    print(summ.round(4).to_string())
    print(sess.round(3).to_string())
    import json
    print(json.dumps({k: v for k, v in out.items() if k not in ("ladder", "ensembles")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
