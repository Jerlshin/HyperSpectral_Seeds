"""S14 · what the lean architecture (Y3) changed — from saved artifacts only (no model is run).

Y3 removed three things at once (inert spectral descriptor blocks, the 1 × 1 tail end, CBAM on 2 × 2 maps), so
no single-component attribution is possible from these cells (that needs the dissection arms, FW-36). What the
saved artifacts *can* say:

1. Is the gain larger than X1's seed spread? Rank of each Y3 cell among the X1 runs of the same fold/protocol
   (exchangeability: P(a draw from X1's distribution beats all n X1 runs) = 1/(n+1)).
2. Where on the generalisation ladder it appears: clean fit → calib → held-out same-session → cross-session.
3. Who gains: per-class recall deltas vs the seed-matched X1 cell; cross-session classes and kernels rescued.
4. Is it a different solution or a better copy? Error overlap of Y3 with X1 s0 against X1 seed-to-seed overlap.
5. How it uses its pathways: leave-one-pathway-out influence (KL, training telemetry) and the session κ of each
   pathway output, vs X1 s0.
6. Over-confidence on held-out (ECE, 15 bins, TTA logits) — S12 F62's quantity.

Writes: lean_rank.csv · lean_ladder.csv · lean_per_class.csv · lean_errors.csv · lean.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from s14common import (
    CROSS,
    EVID,
    S12_EVID,
    SESSION,
    load_logits,
    load_preds,
    recalls,
    s11_cells,
    s13_cells,
    save,
    softmax,
    verify_preregistrations,
)


def ece(probs: np.ndarray, t: np.ndarray, bins: int = 15) -> float:
    conf_, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf_ > lo) & (conf_ <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == t[m]).mean() - conf_[m].mean())
    return float(e)


def main() -> None:
    verify_preregistrations()
    s13 = {c.name: c for c in s13_cells()}
    x1 = {c.name: c for c in s11_cells(("X1",))}
    c13 = pd.read_csv(EVID / "cells.csv").set_index("cell")
    c12 = pd.read_csv(S12_EVID / "cells.csv").set_index("cell")

    # 1 · rank among X1 runs
    rank, p_joint = [], 1.0
    for y, (variant, fold) in (("Y3/lean_grouped__f0_s0", ("grouped", 0)), ("Y3/lean_grouped__f1_s0", ("grouped", 1)),
                               ("Y3/lean_stratified__f0_s0", ("stratified", 0))):
        xs = c12[(c12.arm == "X1") & (c12.variant == variant) & (c12.fold == fold)]
        for q, col in (("f1", "f1_tta"), ("cross", "cross_recall_tta"), ("same", "same_recall_tta")):
            if pd.isna(c13.loc[y, col]) or xs[col].isna().all():
                continue
            v, ref = c13.loc[y, col], xs[col].to_numpy()
            r = dict(cell=y, quantity=q, y3=v, x1_runs=" ".join(f"{a:.4f}" for a in ref), x1_max=ref.max(),
                     x1_mean=ref.mean(), above_all=bool(v > ref.max()), p_exchangeable=1 / (len(ref) + 1),
                     z_vs_x1_sd=(v - ref.mean()) / (0.009933 if q == "f1" and variant == "grouped" else
                                                    0.005806 if variant == "stratified" else
                                                    0.009823 if q == "cross" else 0.013873))
            rank.append(r)
            if q == "f1":
                p_joint *= r["p_exchangeable"] if r["above_all"] else 1.0
    rank = pd.DataFrame(rank)
    save(rank, "lean_rank.csv")

    # 2 · ladder
    ladder = []
    for name in ("X1/grouped__f0_s0", "X1/grouped__f1_s0", "Y3/lean_grouped__f0_s0", "Y3/lean_grouped__f1_s0",
                 "X1/stratified__f0_s0", "Y3/lean_stratified__f0_s0"):
        src = c12.loc[name] if name.startswith("X1") else c13.loc[name]
        ladder.append(dict(cell=name, clean_fit=src.clean_best_selected, calib_acc=src.calib_acc_selected,
                           calib_f1=src.calib_f1_selected, heldout_same=src.same_recall_tta,
                           heldout_cross=src.cross_recall_tta, heldout_f1=src.f1_tta, best_epoch=src.best_epoch,
                           stop_epoch=src.stop_epoch))
    x1g = c12[(c12.arm == "X1") & (c12.variant == "grouped")]
    ladder.append(dict(cell="X1 grouped (6-run mean)", clean_fit=x1g.clean_best_selected.mean(),
                       calib_acc=x1g.calib_acc_selected.mean(), calib_f1=x1g.calib_f1_selected.mean(),
                       heldout_same=x1g.same_recall_tta.mean(), heldout_cross=x1g.cross_recall_tta.mean(),
                       heldout_f1=x1g.f1_tta.mean()))
    ladder = pd.DataFrame(ladder)
    save(ladder, "lean_ladder.csv")

    # 3 · per-class deltas vs X1 s0 (grouped, both folds pooled by class mean)
    per = []
    for fold in (0, 1):
        ry, py, ty = load_preds(s13[f"Y3/lean_grouped__f{fold}_s0"], "tta")
        rx, px, tx = load_preds(x1[f"X1/grouped__f{fold}_s0"], "tta")
        assert np.array_equal(ry, rx)
        ry_, rx_ = recalls(ty, py), recalls(tx, px)
        for c in range(90):
            per.append(dict(fold=fold, cls=c, cross=bool(c in CROSS), y3=ry_[c], x1=rx_[c], delta=ry_[c] - rx_[c]))
    per = pd.DataFrame(per)
    pc = per.groupby(["cls", "cross"])[["y3", "x1", "delta"]].mean().reset_index()
    save(pc, "lean_per_class.csv")

    # cross-session kernels by session (who is rescued)
    by_sess = []
    for fold in (0, 1):
        ry, py, ty = load_preds(s13[f"Y3/lean_grouped__f{fold}_s0"], "tta")
        _, px, _ = load_preds(x1[f"X1/grouped__f{fold}_s0"], "tta")
        m = np.isin(ty, CROSS)
        for s in np.unique(SESSION[ry[m]]):
            k = m & (SESSION[ry] == s)
            by_sess.append(dict(fold=fold, session=int(s), n=int(k.sum()), y3_acc=float((py[k] == ty[k]).mean()),
                                x1_acc=float((px[k] == ty[k]).mean())))
    by_sess = pd.DataFrame(by_sess)

    # 4 · error overlap: Y3 vs X1 s0, against X1 seed pairs (same fold)
    err = []
    for fold in (0, 1):
        ry, py, ty = load_preds(s13[f"Y3/lean_grouped__f{fold}_s0"], "tta")
        xs = {s: load_preds(x1[f"X1/grouped__f{fold}_s{s}"], "tta")[1] for s in (0, 1, 2)}
        wrong = {f"X1 s{s}": p != ty for s, p in xs.items()}
        wrong["Y3"] = py != ty
        for a, b in (("X1 s0", "X1 s1"), ("X1 s0", "X1 s2"), ("X1 s1", "X1 s2"), ("Y3", "X1 s0"), ("Y3", "X1 s1"),
                     ("Y3", "X1 s2")):
            wa, wb = wrong[a], wrong[b]
            err.append(dict(fold=fold, a=a, b=b, jaccard_errors=float((wa & wb).sum() / (wa | wb).sum()),
                            a_only_wrong=int((wa & ~wb).sum()), b_only_wrong=int((~wa & wb).sum()),
                            both_wrong=int((wa & wb).sum())))
    err = pd.DataFrame(err)
    save(err, "lean_errors.csv")

    # 6 · ECE
    ec = {}
    for name, c in (("Y3 f0", s13["Y3/lean_grouped__f0_s0"]), ("Y3 f1", s13["Y3/lean_grouped__f1_s0"]),
                    ("X1 s0 f0", x1["X1/grouped__f0_s0"]), ("X1 s0 f1", x1["X1/grouped__f1_s0"]),
                    ("Y3 strat", s13["Y3/lean_stratified__f0_s0"]), ("X1 s0 strat", x1["X1/stratified__f0_s0"])):
        _, lg, t = load_logits(c, "val_test_tta")
        ec[name] = ece(softmax(lg), t)

    # 5 · pathway use (telemetry + κ), vs X1 s0
    cv12 = pd.read_csv(S12_EVID / "curves.csv")
    kap = pd.read_csv(EVID / "session_kappa.csv")
    use = {}
    for fold, best in ((0, 139), (1, 175)):
        x = cv12[(cv12.cell == f"X1/grouped__f{fold}_s0") & (cv12.epoch == best)].iloc[0]
        y = c13.loc[f"Y3/lean_grouped__f{fold}_s0"]
        kx = kap[(kap.model == "X1 grouped") & (kap.fold == fold) & (kap.seed == 0)].set_index("representation").kappa
        use[f"f{fold}"] = dict(
            x1_influence_spatial=float(x["influence/branch_spatial"]), y3_influence_spatial=float(y.influence_branch_spatial_at_best),
            x1_kappa={k: float(v) for k, v in kx.items()},
            y3_kappa=dict(embedding=float(y.kappa_embedding), spatial=float(y.kappa_spatial), spectral=float(y.kappa_spectral)))

    cross_pc = pc[pc.cross].sort_values("delta")
    out = dict(
        rank_joint_p_f1_above_all_x1_runs=p_joint,
        rank_note="Y3 beats every X1 run of its fold/protocol on F1 in all three cells; under exchangeability each "
                  "has probability 1/4, jointly 1/64 ≈ 0.016 (seed-to-seed spread only; not a held-out-free test).",
        per_class=dict(n_up_gt_0_10=int((pc.delta > 0.10).sum()), n_down_lt_m0_10=int((pc.delta < -0.10).sum()),
                       mean_delta_same=float(pc[~pc.cross].delta.mean()), mean_delta_cross=float(pc[pc.cross].delta.mean()),
                       sd_delta=float(pc.delta.std()),
                       cross_classes_up=cross_pc[cross_pc.delta > 0.05][["cls", "x1", "y3", "delta"]].round(3).to_dict("records"),
                       cross_classes_down=cross_pc[cross_pc.delta < -0.05][["cls", "x1", "y3", "delta"]].round(3).to_dict("records")),
        cross_by_session=by_sess.round(3).to_dict("records"),
        error_overlap=err.groupby(["a", "b"]).jaccard_errors.mean().round(4).to_dict().__repr__(),
        ece=ec, pathway_use=use,
    )
    save(out, "lean.json")
    print(rank.round(4).to_string())
    print(ladder.round(4).to_string())
    print(err.round(3).to_string())
    print(by_sess.round(3).to_string())
    import json
    print(json.dumps({k: v for k, v in out.items() if k not in ("cross_by_session",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
