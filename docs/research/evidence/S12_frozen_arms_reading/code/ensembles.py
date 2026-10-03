"""S12 · variance or bias? Seed ensembles, error sharing and probability calibration from the saved logits.

S09 F41 (shipped regime, predictions only): 80 % of grouped errors are made by every seed and a 3-seed
vote adds ≈ +0.01 — errors are systematic. X1 fits its training set almost perfectly; if what remains
were variance, averaging seeds' logits would recover it. Also measured: whether X1's near-perfect fit
made its held-out probabilities over-confident (ECE, NLL), against X4 (no regularisers) and the X2
arms (shipped regime) — relevant to any later selective-prediction or open-set use.

Diagnostic only: ensembles are formed from runs that already exist, with equal weights, and are not
candidates; nothing is selected on held-out.

Writes: ensembles.csv · ensemble_session.csv · calibration.csv · per_class_x1_vs_s08.csv
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from s12common import (CROSS, cells, load_logits, load_preds, macro_f1, recalls, reference_cells,
                       save, softmax)


def ece(prob: np.ndarray, t: np.ndarray, bins: int = 15) -> float:
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == t[m]).mean() - conf[m].mean())
    return float(e)


def main() -> None:
    x1 = [c for c in cells(("X1",)) if c.scored]
    ens = []
    for proto, fold in (("grouped", 0), ("grouped", 1), ("stratified", 0)):
        runs = [c for c in x1 if c.variant == proto and c.fold == fold]
        for v in ("tta", "no_tta"):
            Z = [load_logits(c, f"val_test_{v}") for c in runs]
            rows, t = Z[0][0], Z[0][2]
            P = [softmax(z) for _, z, _ in Z]
            singles = [macro_f1(t, p.argmax(1)) for p in P]
            correct = np.stack([p.argmax(1) == t for p in P])
            for k in (2, 3):
                for combo in itertools.combinations(range(len(P)), k):
                    pe = np.mean([P[i] for i in combo], 0).argmax(1)
                    ens.append(dict(protocol=proto, fold=fold, view=v, k=k, members=str(combo),
                                    ensemble_f1=macro_f1(t, pe), mean_single_f1=float(np.mean([singles[i] for i in combo]))))
            wrong = ~correct
            ens.append(dict(protocol=proto, fold=fold, view=v, k=0, members="error sharing",
                            ensemble_f1=float(wrong.all(0).sum() / max(wrong.any(0).sum(), 1)),
                            mean_single_f1=float(np.mean(singles)),
                            note_all_wrong_over_single_error=float(wrong.all(0).mean() / wrong.mean()),
                            pairwise_agreement=float(np.mean([(P[i].argmax(1) == P[j].argmax(1)).mean()
                                                              for i, j in itertools.combinations(range(len(P)), 2)]))))
    edf = pd.DataFrame(ens)
    save(edf, "ensembles.csv")

    # does ensembling two FULL networks move cross-session recall / attraction the way fusing the two
    # single-pathway networks does (pathway_fusion.csv)? X1 grouped, 2-member ensembles, TTA.
    from s12common import SESSION, grouped_rows, train_session_by_class
    es = []
    for fold in (0, 1):
        tss = train_session_by_class(grouped_rows(fold)[0])
        runs = [c for c in x1 if c.variant == "grouped" and c.fold == fold]
        Z = [load_logits(c, "val_test_tta") for c in runs]
        rows, t = Z[0][0], Z[0][2]
        P = [softmax(z) for _, z, _ in Z]
        cross = np.isin(t, CROSS)
        def sess(pred):
            rec = recalls(t, pred)
            return dict(f1=macro_f1(t, pred), cross_recall=float(rec[CROSS].mean()),
                        attraction_cross=float((tss[pred[cross]] == SESSION[rows][cross]).mean()))
        for i in range(len(P)):
            es.append(dict(fold=fold, kind="single", members=str(i), **sess(P[i].argmax(1))))
        for combo in itertools.combinations(range(len(P)), 2):
            es.append(dict(fold=fold, kind="2-seed ensemble", members=str(combo),
                           **sess(np.mean([P[i] for i in combo], 0).argmax(1))))
    esd = pd.DataFrame(es)
    save(esd, "ensemble_session.csv")
    print(esd.groupby("kind")[["f1", "cross_recall", "attraction_cross"]].mean().round(4).to_string())

    # calibration of held-out probabilities (no TTA: TTA averages 12 views and is itself an ensemble)
    cal = []
    for c in cells():
        if not c.scored:
            continue
        _, z, t = load_logits(c, "val_test_no_tta")
        p = softmax(z)
        nll = float(-np.log(np.clip(p[np.arange(len(t)), t], 1e-12, None)).mean())
        cal.append(dict(cell=c.name, arm=c.arm, variant=c.variant, acc=float((p.argmax(1) == t).mean()),
                        mean_conf=float(p.max(1).mean()), ece=ece(p, t), nll=nll,
                        conf_when_wrong=float(p.max(1)[p.argmax(1) != t].mean())))
    cdf = pd.DataFrame(cal)
    save(cdf, "calibration.csv")

    # per class: X1 − S08 (grouped, TTA, mean over the 6 runs of each)
    def mean_rec(cs):
        out = []
        for c in cs:
            _, p, t = load_preds(c, "tta")
            out.append(recalls(t, p))
        return np.mean(out, 0)

    r_x1 = mean_rec([c for c in x1 if c.variant == "grouped"])
    r_s8 = mean_rec([c for c in reference_cells() if c.variant == "grouped"])
    pc = pd.DataFrame(dict(cls=np.arange(90), recall_s08=r_s8, recall_x1=r_x1, delta=r_x1 - r_s8,
                           cross=np.isin(np.arange(90), CROSS)))
    save(pc, "per_class_x1_vs_s08.csv")

    print(edf[edf.k > 0].groupby(["protocol", "view", "k"])[["ensemble_f1", "mean_single_f1"]].mean()
          .assign(gain=lambda d: d.ensemble_f1 - d.mean_single_f1).round(4).to_string())
    print(edf[edf.k == 0].round(3).to_string())
    print(cdf.groupby(["arm", "variant"])[["acc", "mean_conf", "ece", "nll", "conf_when_wrong"]].mean().round(3).to_string())
    print("per-class |Δ| > 0.10:", int((pc.delta.abs() > 0.10).sum()), "classes; up", int((pc.delta > 0.10).sum()),
          "down", int((pc.delta < -0.10).sum()), "| same-session mean Δ", round(pc[~pc.cross].delta.mean(), 4),
          "cross", round(pc[pc.cross].delta.mean(), 4), "| sd of Δ", round(pc.delta.std(), 4))


if __name__ == "__main__":
    main()
