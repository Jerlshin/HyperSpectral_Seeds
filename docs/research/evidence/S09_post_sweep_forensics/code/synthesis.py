"""S09 · derived quantities, computed only from the evidence tables (no outputs/ access).

  decomposition.json   where the grouped macro-recall shortfall sits: in-distribution ceiling vs
                       acquisition shift, for same- and cross-session varieties
  transfer.csv         grouped vs stratified macro-F1 for every model/input measured (network,
                       LDA/tabular controls) and the least-squares transfer line
  network_margin.csv   the network against the best fixed-hyperparameter tabular model on the
                       same scalar inputs (mean spectrum + morphometrics)
  fit_link.json        run-level correlation of training fit with held-out F1, Fisher 95% CI
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CROSS, EVID, N_CLASSES, SAME, save


def fisher_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def main() -> None:
    runs = pd.read_csv(EVID / "runs.csv")
    g, s = runs[runs.arm == "grouped"], runs[runs.arm == "stratified"]
    w_same, w_cross = len(SAME) / N_CLASSES, len(CROSS) / N_CLASSES

    # ── decomposition (network, TTA, mean of runs) ──────────────────────────
    iid_same, iid_cross = s.same_recall_tta.mean(), s.cross_recall_tta.mean()
    ho_same, ho_cross = g.same_recall_tta.mean(), g.cross_recall_tta.mean()
    parts = {
        "same-session · already lost in-distribution": w_same * (1 - iid_same),
        "same-session · lost to the bundle shift": w_same * (iid_same - ho_same),
        "cross-session · already lost in-distribution": w_cross * (1 - iid_cross),
        "cross-session · lost to the session shift": w_cross * (iid_cross - ho_cross),
    }
    total = sum(parts.values())
    dec = dict(
        note="Macro-recall shortfall of the grouped network (1 − recall), split by population. "
             "'In-distribution' = 1 − stratified recall of the same classes (both bundles in training, "
             "59 % train). Stratified has ~1.4× the training kernels of grouped; C1 (LDA) puts the "
             "size effect at ≈0.01–0.02 of macro-F1, so the shift terms are slight over-estimates.",
        grouped_macro_recall=w_same * ho_same + w_cross * ho_cross,
        stratified_recall=dict(same=iid_same, cross=iid_cross),
        grouped_recall=dict(same=ho_same, cross=ho_cross),
        shortfall_total=total,
        parts={k: dict(value=v, share=v / total) for k, v in parts.items()},
        in_distribution_share=(parts["same-session · already lost in-distribution"]
                               + parts["cross-session · already lost in-distribution"]) / total,
        ceiling_if_same_session_perfect=w_same + w_cross * ho_cross,
    )
    save(dec, "decomposition.json")

    # ── transfer line across every model/input pair measured on both protocols ──
    pts = [dict(source="SpectralSeedNet (TTA, mean of runs)", stratified=s.f1_tta.mean(), grouped=g.f1_tta.mean()),
           dict(source="SpectralSeedNet (no TTA)", stratified=s.f1_no_tta.mean(), grouped=g.f1_no_tta.mean())]
    c3 = pd.read_csv(EVID / "c3_representation.csv").pivot(index="representation", columns="protocol", values="macro_f1")
    pts += [dict(source=f"LDA · {k}", stratified=v.stratified, grouped=v.grouped) for k, v in c3.iterrows()]
    c5 = pd.read_csv(EVID / "c5_tabular.csv")
    c5 = c5[c5.input != "morphometrics only"].pivot_table(index=["input", "model"], columns="protocol", values="macro_f1")
    pts += [dict(source=f"{m} · {i}", stratified=v.stratified, grouped=v.grouped)
            for (i, m), v in c5.iterrows() if m != "LDA"]
    t = pd.DataFrame(pts).drop_duplicates(subset=["stratified", "grouped"])
    slope, icept = np.polyfit(t.stratified, t.grouped, 1)
    r = np.corrcoef(t.stratified, t.grouped)[0, 1]
    t["fit_grouped"] = icept + slope * t.stratified
    t["residual"] = t.grouped - t.fit_grouped
    save(t, "transfer.csv")
    save(dict(slope=slope, intercept=icept, r=r, n=len(t), ratio_grouped_over_stratified=dict(
        median=float((t.grouped / t.stratified).median()), network=float(g.f1_tta.mean() / s.f1_tta.mean()))),
        "transfer_fit.json")

    # ── network margin over the best tabular control on its own scalar inputs ──
    c5a = pd.read_csv(EVID / "c5_tabular.csv")
    best = c5a[c5a.input == "k32 spectrum + morphometrics"].sort_values("macro_f1").groupby("protocol").tail(1)
    rows = []
    for proto, net in (("grouped", g), ("stratified", s)):
        b = best[best.protocol == proto].iloc[0]
        for variant in ("tta", "no_tta"):
            rows.append(dict(protocol=proto, variant=variant, network_f1=net[f"f1_{variant}"].mean(),
                             best_tabular=f"{b.model} · {b.input}", tabular_f1=b.macro_f1,
                             margin=net[f"f1_{variant}"].mean() - b.macro_f1,
                             network_cross_recall=net[f"cross_recall_{variant}"].mean(),
                             tabular_cross_recall=b.cross_recall))
    save(pd.DataFrame(rows), "network_margin.csv")

    # ── fit ↔ held-out link per protocol ───────────────────────────────────
    link = {}
    for proto, x in (("grouped", g), ("stratified", s)):
        for col in ("clean_train_acc", "final_train_loss", "calib_best_f1"):
            rr = float(np.corrcoef(x[col], x.f1_tta)[0, 1])
            link[f"{proto}: {col} vs held-out F1"] = dict(r=rr, ci95=fisher_ci(rr, len(x)), n=len(x))
    save(link, "fit_link.json")


if __name__ == "__main__":
    main()
