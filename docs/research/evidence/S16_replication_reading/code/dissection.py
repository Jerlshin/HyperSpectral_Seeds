"""S16 · what each half of the lean network does — the Y5 dissection beyond its frozen F1 bar (decides nothing).

H22a/H22b ask one question (does each half alone reach 0.5508 grouped F1?). The saved artifacts answer more, all
seed 0, seed- and fold-matched against X1 s0 (shipped architecture) and Y3 s0 (both halves) on the same kernels:

1. Profile per cell: held-out F1 / same / cross / attraction; calib F1; clean fit; best and stop epoch; spatial-pathway
   influence at the selected epoch (leave-one-out KL, training telemetry); training-rows session κ of the embedding
   and of each pathway output (in-pipeline probe, validated in S14); held-out ECE.
2. Additivity: Δ(desc_only) + Δ(spatial_repair) against Δ(Y3), each vs X1 s0, for every metric.
3. Who is rescued: cross-session held-out accuracy by acquisition session of the kernel.
4. Whose solution it is: error-set Jaccard against X1 s0 and Y3 s0, and the correlation of per-class recall changes
   (vs X1 s0) with Y3's.

Writes: dissection_profile.csv · dissection_additivity.csv · dissection_sessions.csv · dissection.json
"""
from __future__ import annotations

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
    load_logits,
    recalls,
    s15_cells,
    save,
    softmax,
    verify_preregistrations,
    x1_cells,
    y3_cells,
)

KEYS = ("f1", "same", "cross", "attraction")


def main() -> None:
    verify_preregistrations()
    c16 = pd.read_csv(EVID / "cells.csv").set_index("cell")
    c14 = pd.read_csv(S14_EVID / "cells.csv").set_index("cell")
    c12 = pd.read_csv(S12_EVID / "cells.csv").set_index("cell")
    cv12 = pd.read_csv(S12_EVID / "curves.csv")
    kap = pd.read_csv(S14_EVID / "session_kappa.csv")

    arms = {"X1 s0": {c.fold: c for c in x1_cells((0,))}, "Y3 s0": {c.fold: c for c in y3_cells((0,))}}
    for v in ("desc_only", "spatial_repair"):
        arms[v] = {c.fold: c for c in s15_cells() if c.variant == v}
    S = {a: {f: Scored(c) for f, c in d.items()} for a, d in arms.items()}

    prof = []
    for a, d in arms.items():
        for f, c in d.items():
            st = S[a][f].stats()
            _, lg, t = load_logits(c, "val_test_tta")
            r = dict(arm=a, fold=f, cell=c.name, **st, ece=S14_LEAN.ece(softmax(lg), t))
            if a == "X1 s0":
                src = c12.loc[c.name]
                k = kap[(kap.model == "X1 grouped") & (kap.fold == f) & (kap.seed == 0)].set_index("representation").kappa
                infl = cv12[(cv12.cell == c.name) & (cv12.epoch == src.best_epoch)]["influence/branch_spatial"].iloc[0]
                r.update(kappa_embedding=k["embedding"], kappa_spatial=k["spatial"], kappa_spectral=k["spectral"],
                         influence_spatial=infl)
            else:
                src = (c14 if a == "Y3 s0" else c16).loc[c.name]
                r.update(kappa_embedding=src.kappa_embedding, kappa_spatial=src.kappa_spatial,
                         kappa_spectral=src.kappa_spectral, influence_spatial=src.influence_branch_spatial_at_best)
            r.update(calib_f1=src.calib_f1_selected, clean_fit=src.clean_best_selected, best_epoch=src.best_epoch,
                     stop_epoch=src.stop_epoch, f1_no_tta=src.f1_no_tta)
            prof.append(r)
    prof = pd.DataFrame(prof)
    save(prof, "dissection_profile.csv")

    # additivity (Δ vs X1 s0, mean of folds)
    m = prof.groupby("arm")[[*KEYS, "calib_f1", "ece", "kappa_spectral", "kappa_spatial", "influence_spatial"]].mean()
    add = []
    for k in m.columns:
        dd, ds, dy = (m.loc[a, k] - m.loc["X1 s0", k] for a in ("desc_only", "spatial_repair", "Y3 s0"))
        add.append(dict(metric=k, x1_s0=m.loc["X1 s0", k], d_desc_only=dd, d_spatial_repair=ds, d_sum=dd + ds,
                        d_y3_s0=dy, interaction=dy - (dd + ds), share_spatial_of_y3=ds / dy if abs(dy) > 1e-9 else np.nan))
    add = pd.DataFrame(add)
    save(add, "dissection_additivity.csv")

    # who is rescued: cross-session held-out accuracy by session
    sess = []
    for a in arms:
        for f, s in S[a].items():
            mk = s.is_cross
            for ss in np.unique(SESSION[s.rows[mk]]):
                k = mk & (SESSION[s.rows] == ss)
                sess.append(dict(arm=a, fold=f, session=int(ss), n=int(k.sum()), acc=float((s.p[k] == s.t[k]).mean())))
    sess = pd.DataFrame(sess)
    wide = sess.pivot_table(index=["fold", "session", "n"], columns="arm", values="acc").reset_index()
    save(wide, "dissection_sessions.csv")

    # whose solution: error Jaccard and per-class-change correlation
    sol = {}
    for f in (0, 1):
        wrong = {a: S[a][f].p != S[a][f].t for a in arms}
        rec = {a: recalls(S[a][f].t, S[a][f].p) for a in arms}
        for a in ("desc_only", "spatial_repair", "Y3 s0"):
            for b in ("X1 s0", "Y3 s0"):
                if a == b:
                    continue
                wa, wb = wrong[a], wrong[b]
                sol.setdefault(f"jaccard {a} vs {b}", []).append(float((wa & wb).sum() / (wa | wb).sum()))
        for a in ("desc_only", "spatial_repair"):
            r = np.corrcoef(rec[a] - rec["X1 s0"], rec["Y3 s0"] - rec["X1 s0"])[0, 1]
            rc = np.corrcoef((rec[a] - rec["X1 s0"])[CROSS], (rec["Y3 s0"] - rec["X1 s0"])[CROSS])[0, 1]
            sol.setdefault(f"per-class Δ corr with Y3: {a}", []).append(float(r))
            sol.setdefault(f"per-class Δ corr with Y3 (cross classes): {a}", []).append(float(rc))
    out = dict(
        mean_by_arm=m.round(4).to_dict("index"),
        solution_similarity={k: dict(f0=v[0], f1=v[1], mean=float(np.mean(v))) for k, v in sol.items()},
        cross_by_session_pooled=sess.groupby(["arm", "session"]).apply(
            lambda g: float(np.average(g.acc, weights=g.n)), include_groups=False).unstack(0).round(3).to_dict(),
        note="seed 0 only; the dissection arms and X1 s0 share no initial weights (modules differ), so 'matched' means "
             "same kernels, same seed, not paired initialisation. Decides nothing.",
    )
    save(out, "dissection.json")
    print(prof.round(4).to_string())
    print(add.round(4).to_string())
    print(wide.round(3).to_string())
    import json
    print(json.dumps(out["solution_similarity"], indent=1))


if __name__ == "__main__":
    main()
