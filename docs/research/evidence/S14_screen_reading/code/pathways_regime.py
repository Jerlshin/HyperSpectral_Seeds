"""S14 · why Y1 bought no session robustness: the single-pathway networks under R1 vs the shipped regime.

S12 F65 (shipped regime, post hoc, 3 cells) found late fusion of separately trained pathways keeps more
cross-session recall than the joint network (+0.018) with less attraction. Y1 re-tested it under R1 and H19b
was rejected. This script separates the two candidate explanations — the fusion weight, or the networks being
fused — using saved predictions and logits only (no model is run):

1. Regime effect on each single pathway: Y1 (R1) vs X2 (shipped), seed- and fold-matched where both exist
   (spectral f0 s0, f1 s0; spatial f0 s0 — X2 spatial f1 s0 was never scored, F71a), plus arm means.
2. Joint vs fused under each regime, seed 0: shipped = S08 joint vs X2 equal-weight fusion (S12
   ``pathway_fusion.csv``); R1 = X1 joint vs Y1 fused (calib w and w = 0.5).
3. The fusion weight: calib macro-F1 along the frozen grid (what chose w), and — **diagnostic only, chooses
   nothing** — held-out F1 / cross-session recall / attraction along the same grid, to ask whether *any* weight
   could have met H19b. This reads existing TTA logits; no weight is selected from it.
4. Complementarity on cross-session held-out kernels (as S12 §5.3): who gets which kernels right.

Writes: pathway_regime.csv · fusion_grid.csv · complementarity_r1.csv · pathway_regime.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from s14common import (
    CROSS,
    EVID,
    S12_EVID,
    SAME,
    SESSION,
    THRESH,
    conf,
    f1_from_conf,
    grouped_rows,
    load_logits,
    load_preds,
    recall_from_conf,
    s11_cells,
    s13_cells,
    save,
    train_session_by_class,
    verify_preregistrations,
)


def stats(rows: np.ndarray, p: np.ndarray, t: np.ndarray, fold: int) -> dict:
    cm = conf(t, p)
    tr, _, _ = grouped_rows(fold)
    soc = train_session_by_class(tr)
    m = np.isin(t, CROSS)
    return dict(f1=f1_from_conf(cm), same=recall_from_conf(cm, SAME), cross=recall_from_conf(cm, CROSS),
                attraction=float((soc[p[m]] == SESSION[rows[m]]).mean()))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main() -> None:
    verify_preregistrations()
    s13 = {c.name: c for c in s13_cells()}
    s11 = {c.name: c for c in s11_cells(("X1", "X2"))}
    c13 = pd.read_csv(EVID / "cells.csv").set_index("cell")
    c12 = pd.read_csv(S12_EVID / "cells.csv").set_index("cell")
    kap = pd.read_csv(EVID / "session_kappa.csv")

    def kappa(model: str, fold: int, seed: int) -> float:
        k = kap[(kap.model == model) & (kap.fold == fold) & (kap.seed == seed) & (kap.representation == "embedding")]
        return float(k.kappa.iloc[0]) if len(k) else np.nan

    # 1 · single pathways: R1 (Y1) vs shipped (X2), matched cells
    rows = []
    for pw in ("spectral_only", "spatial_only"):
        for fold in (0, 1):
            y = c13.loc[f"Y1/{pw}__f{fold}_s0"]
            x = c12.loc[f"X2/{pw}__f{fold}_s0"]
            r = dict(pathway=pw, fold=fold, seed=0)
            for reg, src, kmodel in (("R1", y, f"Y1 {pw}"), ("shipped", x, f"X2 {pw}")):
                r.update({f"{reg}_f1": src.f1_tta, f"{reg}_same": src.same_recall_tta,
                          f"{reg}_cross": src.cross_recall_tta, f"{reg}_attraction": src.attraction_cross_tta,
                          f"{reg}_clean_fit": src.clean_best_selected, f"{reg}_calib_f1": src.calib_f1_selected,
                          f"{reg}_kappa": kappa(kmodel, fold, 0)})
            rows.append(r)
    # arm means (all runs) beside
    for pw in ("spectral_only", "spatial_only"):
        y = c13[(c13.arm == "Y1") & (c13.variant == pw)]
        x = c12[(c12.arm == "X2") & (c12.variant == pw) & c12.f1_tta.notna()]
        rows.append(dict(pathway=pw, fold="all", seed=f"Y1 n={len(y)} · X2 n={len(x)}",
                         R1_f1=y.f1_tta.mean(), R1_same=y.same_recall_tta.mean(), R1_cross=y.cross_recall_tta.mean(),
                         R1_attraction=y.attraction_cross_tta.mean(), R1_clean_fit=y.clean_best_selected.mean(),
                         R1_calib_f1=y.calib_f1_selected.mean(),
                         shipped_f1=x.f1_tta.mean(), shipped_same=x.same_recall_tta.mean(),
                         shipped_cross=x.cross_recall_tta.mean(), shipped_attraction=x.attraction_cross_tta.mean(),
                         shipped_clean_fit=x.clean_best_selected.mean(), shipped_calib_f1=x.calib_f1_selected.mean()))
    reg = pd.DataFrame(rows)
    save(reg, "pathway_regime.csv")

    # 2 · joint vs fused, per regime, seed 0
    pf = pd.read_csv(S12_EVID / "pathway_fusion.csv")
    joint_fused = []
    for fold in (0, 1):
        x1 = c12.loc[f"X1/grouped__f{fold}_s0"]
        fu = c13.loc[f"Y1/fused__f{fold}_s0"]
        joint_fused.append(dict(regime="R1", fold=fold, joint_f1=x1.f1_tta, joint_cross=x1.cross_recall_tta,
                                joint_attraction=x1.attraction_cross_tta, fused_calib_w=fu.fusion_weight,
                                fused_f1=fu.f1_tta, fused_cross=fu.cross_recall_tta,
                                fused_attraction=fu.attraction_cross_tta, eq_f1=fu.eq_f1_tta,
                                eq_cross=fu.eq_cross_recall_tta, eq_attraction=fu.eq_attraction_cross_tta))
    for (fold, seed), g in pf[pf.view == "tta"].groupby(["fold", "seed"]):
        j = g[g.model.str.contains("joint|S08", regex=True)]
        e = g[g.model.str.contains("equal")]
        cw = g[g.model.str.contains("calib")]
        if len(j) and len(e):
            joint_fused.append(dict(regime="shipped", fold=fold, seed=seed, joint_f1=j.f1.iloc[0],
                                    joint_cross=j.cross_recall.iloc[0], joint_attraction=j.attraction_cross.iloc[0],
                                    fused_calib_w=cw.w_spectral.iloc[0] if len(cw) else np.nan,
                                    fused_f1=cw.f1.iloc[0] if len(cw) else np.nan,
                                    fused_cross=cw.cross_recall.iloc[0] if len(cw) else np.nan,
                                    fused_attraction=cw.attraction_cross.iloc[0] if len(cw) else np.nan,
                                    eq_f1=e.f1.iloc[0], eq_cross=e.cross_recall.iloc[0],
                                    eq_attraction=e.attraction_cross.iloc[0]))
    jf = pd.DataFrame(joint_fused)

    # 3 · the weight: calib curve (from run.json) and the held-out grid (diagnostic only)
    grid = []
    for fold in (0, 1):
        spec, spat = s13[f"Y1/spectral_only__f{fold}_s0"], s13[f"Y1/spatial_only__f{fold}_s0"]
        rs, ls, ts = load_logits(spec, "val_test_tta")
        rp, lp, tp = load_logits(spat, "val_test_tta")
        assert np.array_equal(rs, rp) and np.array_equal(ts, tp)
        cs_, csl, cst = load_logits(spec, "calib_tta")
        cp_, cpl, cpt = load_logits(spat, "calib_tta")
        assert np.array_equal(cs_, cp_)
        for w in np.round(np.arange(0, 1.0001, 0.05), 2):
            z = w * log_softmax(ls) + (1 - w) * log_softmax(lp)
            zc = w * log_softmax(csl) + (1 - w) * log_softmax(cpl)
            st = stats(rs, z.argmax(1), ts, fold)
            grid.append(dict(fold=fold, w_spectral=w, calib_f1=f1_from_conf(conf(cst, zc.argmax(1))),
                             heldout_f1=st["f1"], heldout_same=st["same"], heldout_cross=st["cross"],
                             heldout_attraction=st["attraction"]))
    g = pd.DataFrame(grid)
    save(g, "fusion_grid.csv")
    gm = g.groupby("w_spectral")[["calib_f1", "heldout_f1", "heldout_cross", "heldout_attraction"]].mean()
    passes = gm[(gm.heldout_f1 >= THRESH["H19a"]) & (gm.heldout_cross >= THRESH["H19b_cross"])
                & (gm.heldout_attraction <= THRESH["H19b_attraction"])]

    # 4 · complementarity on cross-session kernels (R1, seed 0)
    comp = []
    for fold in (0, 1):
        r_, ps, t_ = load_preds(s13[f"Y1/spectral_only__f{fold}_s0"], "tta")
        _, pp, _ = load_preds(s13[f"Y1/spatial_only__f{fold}_s0"], "tta")
        _, pj, _ = load_preds(s11[f"X1/grouped__f{fold}_s0"], "tta")
        _, pfu, _ = load_preds(s13[f"Y1/fused__f{fold}_s0"], "tta")
        m = np.isin(t_, CROSS)
        a, b = ps[m] == t_[m], pp[m] == t_[m]
        for name, sel in (("spectral only right", a & ~b), ("spatial only right", ~a & b), ("both right", a & b),
                          ("neither", ~a & ~b)):
            comp.append(dict(fold=fold, subset=name, n=int(sel.sum()), share=float(sel.mean()),
                             joint_x1_right=float((pj[m][sel] == t_[m][sel]).mean()) if sel.any() else np.nan,
                             fused_right=float((pfu[m][sel] == t_[m][sel]).mean()) if sel.any() else np.nan))
    comp = pd.DataFrame(comp)
    save(comp, "complementarity_r1.csv")

    out = dict(
        joint_vs_fused=jf.to_dict("records"),
        calib_chosen_w={f"f{f}": float(c13.loc[f"Y1/fused__f{f}_s0", "fusion_weight"]) for f in (0, 1)},
        shipped_calib_w_mean=float(jf[jf.regime == "shipped"].fused_calib_w.mean()),
        heldout_grid_mean_over_folds=gm.reset_index().round(4).to_dict("records"),
        any_weight_meets_H19a_and_H19b_on_heldout=bool(len(passes)),
        max_heldout_cross_over_grid=float(gm.heldout_cross.max()),
        w_at_max_cross=float(gm.heldout_cross.idxmax()),
        note="The held-out grid is a post-hoc diagnostic of whether the weight could explain H19b's rejection. "
             "No weight is chosen from it; the frozen reading uses the calib-chosen w only.",
    )
    save(out, "pathway_regime.json")
    print(reg.round(4).to_string())
    print(jf.round(4).to_string())
    print(gm.round(4).to_string())
    print(comp.round(3).to_string())
    print({k: v for k, v in out.items() if k not in ("joint_vs_fused", "heldout_grid_mean_over_folds")})


if __name__ == "__main__":
    main()
