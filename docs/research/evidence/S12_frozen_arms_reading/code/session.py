"""S12 · the same-session / cross-session trade-off, and what buys cross-session recall.

1. **Linear controls** (CPU, StandardScaler → LDA as in S09, fit on the fold's ``train`` rows only, scored
   on its held-out bundle, both folds): raw reflectance, SNV, level (log mean reflectance, one scalar), and
   their combinations with the 8 morphometrics. They ask whether the spectral-only network's cross-session
   advantage is a property of its *representation* (SNV + morphometrics, level-blind) or of the network.
   Diagnostic controls in S09's sense — nothing is selected for the network on them.
2. **Frontier** — every trained run and every control as (same-session recall, cross-session recall,
   cross-session attraction).
3. **Per-class cross-session recall** for the 17 cross-session varieties, by model.
4. **Direction** — cross-session recall split by the held-out bundle's session (S09 F39: → session 8).

Writes: linear_controls.csv · frontier.csv · cross_per_class.csv · cross_direction.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s12common import (CROSS, LABELS, REPO, SAME, SESSION, cells, grouped_rows, load_preds,
                       macro_f1, recalls, reference_cells, save, train_session_by_class)

warnings.filterwarnings("ignore")
MEAN = np.load(REPO / "outputs/s09_forensics/mean_k32.npy").astype(np.float64)
MORPH = np.load(REPO / "dataset_u430k32/morphology.npy").astype(np.float64)


def snv(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)


def representations() -> dict[str, np.ndarray]:
    level = np.log(MEAN.mean(1, keepdims=True))
    return {
        "raw reflectance": MEAN,
        "SNV": snv(MEAN),
        "morphometrics": MORPH,
        "raw + morph (S09 best linear)": np.hstack([MEAN, MORPH]),
        "SNV + morph (spectral-path analogue)": np.hstack([snv(MEAN), MORPH]),
        "SNV + level + morph": np.hstack([snv(MEAN), level, MORPH]),
        "level + morph": np.hstack([level, MORPH]),
    }


def metrics(rows, t, p, tr_sess) -> dict:
    rec = recalls(t, p)
    cross = np.isin(t, CROSS)
    return dict(f1=macro_f1(t, p), same_recall=float(rec[SAME].mean()), cross_recall=float(rec[CROSS].mean()),
                attraction_cross=float((tr_sess[p[cross]] == SESSION[rows][cross]).mean()))


def main() -> None:
    folds = {f: grouped_rows(f) for f in (0, 1)}
    tr_sess = {f: train_session_by_class(folds[f][0]) for f in (0, 1)}

    # 1 · linear controls
    lin, lin_preds = [], {}
    for name, X in representations().items():
        for f, (tr, _, te) in folds.items():
            for shrink in (False, True):
                est = LDA(solver="lsqr", shrinkage="auto") if shrink else LDA(solver="svd", tol=1e-4)
                m = make_pipeline(StandardScaler(), est).fit(X[tr], LABELS[tr])
                p = m.predict(X[te])
                lin.append(dict(representation=name, lda="shrinkage" if shrink else "svd", fold=f,
                                **metrics(te, LABELS[te], p, tr_sess[f])))
                if not shrink:
                    lin_preds[(name, f)] = (te, p)
    ldf = pd.DataFrame(lin)
    save(ldf, "linear_controls.csv")

    # 2 · frontier: every network run + the svd-LDA controls (fold means)
    fr = []
    for c in reference_cells() + cells():
        if c.protocol != "grouped" or not c.scored:
            continue
        r, p, t = load_preds(c, "tta")
        label = "S08 full (shipped)" if c.arm == "S08" else (f"X2 {c.variant}" if c.arm == "X2" else f"{c.arm} full")
        fr.append(dict(model=label, kind="network", cell=c.name, fold=c.fold, seed=c.seed,
                       **metrics(r, t, p, tr_sess[c.fold])))
    for (name, f), (te, p) in lin_preds.items():
        fr.append(dict(model=f"LDA {name}", kind="linear", cell="", fold=f, seed=-1,
                       **metrics(te, LABELS[te], p, tr_sess[f])))
    fdf = pd.DataFrame(fr)
    save(fdf, "frontier.csv")

    # 3 · per-class cross-session recall (mean over runs of each model)
    pc = []
    groups = {
        "S08 full": [c for c in reference_cells() if c.variant == "grouped"],
        "X1 full": [c for c in cells(("X1",)) if c.variant == "grouped"],
        "X2 spectral_only": [c for c in cells(("X2",)) if c.variant == "spectral_only"],
        "X2 spatial_only": [c for c in cells(("X2",)) if c.variant == "spatial_only" and c.scored],
        "X2 no_morph": [c for c in cells(("X2",)) if c.variant == "no_morph"],
    }
    for k, runs in groups.items():
        rec = []
        for c in runs:
            r, p, t = load_preds(c, "tta")
            rec.append(recalls(t, p))
        rec = np.mean(rec, axis=0)
        for cl in CROSS:
            pc.append(dict(model=k, cls=int(cl), recall=float(rec[cl])))
    for name in ("raw + morph (S09 best linear)", "SNV + morph (spectral-path analogue)", "morphometrics"):
        rec = np.mean([recalls(LABELS[lin_preds[(name, f)][0]], lin_preds[(name, f)][1]) for f in (0, 1)], axis=0)
        for cl in CROSS:
            pc.append(dict(model=f"LDA {name}", cls=int(cl), recall=float(rec[cl])))
    pdf = pd.DataFrame(pc).pivot(index="cls", columns="model", values="recall").reset_index()
    save(pdf, "cross_per_class.csv")

    # 4 · direction: by the session the held-out cross-session kernel was imaged in
    dr = []
    for k in ("S08 full", "X1 full", "X2 spectral_only", "X2 no_morph"):
        for c in groups[k]:
            r, p, t = load_preds(c, "tta")
            cross = np.isin(t, CROSS)
            for s in np.unique(SESSION[r][cross]):
                m = cross & (SESSION[r] == s)
                dr.append(dict(model=k, cell=c.name, heldout_session=int(s), n=int(m.sum()),
                               acc=float((p[m] == t[m]).mean())))
    ddf = pd.DataFrame(dr).groupby(["model", "heldout_session"]).agg(n=("n", "mean"), acc=("acc", "mean")).reset_index()
    save(ddf, "cross_direction.csv")

    print(ldf.groupby(["representation", "lda"])[["f1", "same_recall", "cross_recall", "attraction_cross"]]
          .mean().round(3).to_string())
    print(fdf.groupby("model")[["f1", "same_recall", "cross_recall", "attraction_cross"]].mean().round(3)
          .sort_values("cross_recall").to_string())
    print(pdf.round(2).to_string())
    print(ddf.pivot(index="heldout_session", columns="model", values="acc").round(3).to_string())


if __name__ == "__main__":
    main()
