"""S09 · CPU controls on mean spectra — cheap experiments that separate data effects from model effects.

All arms use the sweep baseline's own estimator (StandardScaler → LDA, svd solver) on the
foreground mean spectrum, fitted on `train` (calib excluded, as in the sweep), so C0 reproduces
outputs/experiments_u430k32/baselines exactly. These are *exploratory diagnostics*, not
pre-registered arms: nothing here selects a model, band set or hyperparameter for the network.

  C0  sanity          reproduce the sweep's LDA numbers from the cache
  C1  learning curve  macro-F1 vs kernels/class, stratified and grouped, matched n
                      → how much of the leakage gap is training-set size; what 80/20 would score
  C2  acquisition mix grouped + k kernels/class from the held-out bundle (additive and n-matched)
                      → does seeing the second acquisition close the gap?
  C3  representation  k32 vs 215 bands, SNV, log-reflectance, + morphometrics (grouped & stratified)
  C5  tabular       fixed-hyperparameter non-linear models on mean spectrum + morphometrics
                      → how much of the network's score a 40-number summary already carries
  C4  fingerprint     per-band session F-ratio on reflectance (S06/a9 method); the SNV-256 curve
                      it is compared with stays in evidence/S06_session_confound/a9_session_F.json

Needs outputs/s09_forensics/mean_{k32,r215}.npy from spectra.py. ~10–15 min on a laptop CPU (C5's boosting and SVM dominate).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (CACHE, CROSS, DS32, DS215, LABELS, N_CLASSES, REPO, SAME, SESSION,
                    grouped_rows, save, stratified_rows)

warnings.filterwarnings("ignore")
Y = LABELS
REPS = 10


def lda(shrink: bool = False):
    est = LDA(solver="lsqr", shrinkage="auto") if shrink else LDA(solver="svd", tol=1e-4)
    return make_pipeline(StandardScaler(), est)


def score(t, p):
    rec = recall_score(t, p, labels=np.arange(N_CLASSES), average=None, zero_division=0)
    present = np.isin(np.arange(N_CLASSES), np.unique(t))
    return dict(macro_f1=f1_score(t, p, labels=np.unique(t), average="macro", zero_division=0),
                acc=float((t == p).mean()),
                same_recall=float(rec[np.intersect1d(SAME, np.flatnonzero(present))].mean()),
                cross_recall=float(rec[np.intersect1d(CROSS, np.flatnonzero(present))].mean()))


def fit_score(X, tr, te, shrink=False):
    m = lda(shrink).fit(X[tr], Y[tr])
    return score(Y[te], m.predict(X[te]))


def per_class_sample(rows, n, rng):
    out = []
    for c in range(N_CLASSES):
        r = rows[Y[rows] == c]
        out.append(rng.choice(r, size=min(n, len(r)), replace=False))
    return np.concatenate(out)


def snv(x):
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def main() -> None:
    X32 = np.load(CACHE / "mean_k32.npy").astype(np.float64)
    X215 = np.load(CACHE / "mean_r215.npy").astype(np.float64)
    morph = np.load(DS32 / "morphology.npy").astype(np.float64)
    wl215 = pd.read_csv(DS215 / "wavelengths.csv").iloc[:, -1].to_numpy(float)
    G = {f: grouped_rows(f) for f in (0, 1)}
    S = stratified_rows()
    rng = np.random.default_rng(0)

    # ── C0 sanity ─────────────────────────────────────────────────────────────
    sweep = pd.read_csv(REPO / "docs/research/evidence/S09_post_sweep_forensics/baselines.csv")
    c0 = []
    for arm, (tr, ca, held) in [("grouped_f0", G[0]), ("grouped_f1", G[1]), ("stratified_f0", S)]:
        got = fit_score(X32, tr, held)["macro_f1"]
        a, f = arm.split("_f")
        want = sweep[(sweep.arm == a) & (sweep.fold == int(f)) & (sweep.model == "lda")].macro_f1.item()
        c0.append(dict(arm=arm, reproduced=got, sweep=want, abs_diff=abs(got - want)))
    c0 = pd.DataFrame(c0)
    print(c0)
    assert (c0.abs_diff < 1e-6).all(), "cache does not reproduce the sweep baseline"

    # ── C1 learning curves at matched kernels/class ───────────────────────────
    c1 = []
    pool_s = np.concatenate([S[0], S[1]])               # stratified train ∪ calib (~67/class)
    for n in (5, 10, 20, 30, 40, 47, 57, 67):
        for rep in range(REPS):
            tr = per_class_sample(pool_s, n, rng)
            c1.append(dict(protocol="stratified (both bundles)", n_per_class=n, rep=rep,
                           **fit_score(X32, tr, S[2])))
            for f in (0, 1):
                if n > 47:
                    continue
                pool_g = np.concatenate([G[f][0], G[f][1]])  # one bundle, train ∪ calib (~47/class)
                tr = per_class_sample(pool_g, n, rng)
                c1.append(dict(protocol=f"grouped f{f} (one bundle)", n_per_class=n, rep=rep,
                               **fit_score(X32, tr, G[f][2])))
    # 80/20 patch-level 5-fold CV over all 8,624 kernels: the literature-style protocol
    from sklearn.model_selection import StratifiedKFold
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X32, Y)):
        c1.append(dict(protocol="80/20 patch-level 5-fold", n_per_class=len(tr) / N_CLASSES, rep=k,
                       **fit_score(X32, tr, te)))
        c1.append(dict(protocol="80/20 patch-level 5-fold · r215 shrinkage", n_per_class=len(tr) / N_CLASSES,
                       rep=k, **fit_score(X215, tr, te, shrink=True)))
    c1 = pd.DataFrame(c1)
    c1["n_per_class"] = c1.n_per_class.round(0)
    save(c1.groupby(["protocol", "n_per_class"]).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"), acc=("acc", "mean"),
        same_recall=("same_recall", "mean"), cross_recall=("cross_recall", "mean"),
        reps=("rep", "count")).reset_index(), "c1_learning_curve.csv")

    # ── C2 acquisition mixing: k kernels/class of the held-out bundle in training ─
    c2 = []
    for f in (0, 1):
        tr_g, ca_g, held = G[f]
        base = np.concatenate([tr_g, ca_g])
        for k in (0, 1, 2, 4, 8, 16, 24):
            for rep in range(REPS):
                add = per_class_sample(held, k, rng) if k else np.empty(0, int)
                test = np.setdiff1d(held, add)
                # additive: the whole training bundle + k target-acquisition kernels
                c2.append(dict(fold=f, k=k, design="additive", rep=rep,
                               **fit_score(X32, np.concatenate([base, add]), test)))
                # n-matched: drop k training-bundle kernels per class so the total is unchanged
                keep = np.concatenate([rng.choice(base[Y[base] == c], size=(Y[base] == c).sum() - k,
                                                  replace=False) for c in range(N_CLASSES)])
                c2.append(dict(fold=f, k=k, design="n-matched", rep=rep,
                               **fit_score(X32, np.concatenate([keep, add]), test)))
    c2 = pd.DataFrame(c2)
    save(c2.groupby(["design", "k"]).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        same_recall=("same_recall", "mean"), cross_recall=("cross_recall", "mean"),
        n=("rep", "count")).reset_index(), "c2_acquisition_mix.csv")

    # ── C3 representation (grouped f0/f1 and stratified, train only, as the sweep) ─
    ge430 = wl215 >= 430
    reps = {
        "k32 reflectance (sweep input)": (X32, False),
        "k32 SNV": (snv(X32), False),
        "k32 log-reflectance": (np.log(np.clip(X32, 1e-4, None)), False),
        "k32 + morphometrics": (np.c_[X32, morph], False),
        "r215 reflectance (shrinkage)": (X215, True),
        "r215 ≥430 nm (shrinkage)": (X215[:, ge430], True),
        "r215 SNV (shrinkage)": (snv(X215), True),
        "k32 reflectance (shrinkage)": (X32, True),
    }
    c3 = []
    for name, (X, sh) in reps.items():
        for arm, (tr, ca, held) in [("grouped f0", G[0]), ("grouped f1", G[1]), ("stratified", S)]:
            c3.append(dict(representation=name, arm=arm, n_features=X.shape[1], **fit_score(X, tr, held, sh)))
    c3 = pd.DataFrame(c3)
    c3["protocol"] = np.where(c3.arm.str.startswith("grouped"), "grouped", "stratified")
    save(c3.groupby(["representation", "protocol"]).agg(
        n_features=("n_features", "first"), macro_f1=("macro_f1", "mean"),
        same_recall=("same_recall", "mean"), cross_recall=("cross_recall", "mean")).reset_index(),
        "c3_representation.csv")


    # ── C5 tabular models on the scalar inputs the network also receives ──────
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    # Hyperparameters fixed a priori (library defaults or one conventional value); none was
    # tuned, on held-out or otherwise.
    models = {
        "LDA": lambda: lda(),
        "logistic (C=1)": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "RBF-SVM (C=10)": lambda: make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale")),
        "grad. boosting": lambda: HistGradientBoostingClassifier(random_state=0),
        "MLP 256×256": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            (256, 256), alpha=1e-3, early_stopping=True, max_iter=500, random_state=0)),
    }
    inputs = {"morphometrics only": morph, "k32 spectrum": X32, "k32 spectrum + morphometrics": np.c_[X32, morph]}
    c5 = []
    for iname, X in inputs.items():
        for mname, make in models.items():
            for arm, (tr, ca, held) in [("grouped f0", G[0]), ("grouped f1", G[1]), ("stratified", S)]:
                m = make().fit(X[tr], Y[tr])
                c5.append(dict(input=iname, model=mname, arm=arm, **score(Y[held], m.predict(X[held]))))
    c5 = pd.DataFrame(c5)
    c5["protocol"] = np.where(c5.arm.str.startswith("grouped"), "grouped", "stratified")
    save(c5.groupby(["input", "model", "protocol"]).agg(
        macro_f1=("macro_f1", "mean"), same_recall=("same_recall", "mean"),
        cross_recall=("cross_recall", "mean")).reset_index(), "c5_tabular.csv")

    # ── C4 session fingerprint by wavelength (a9 method, training bundle rows) ──
    def session_F(X):
        res = []
        for f in (0, 1):
            rows = np.concatenate([G[f][0], G[f][1]])
            Xr, y, s = X[rows], Y[rows], SESSION[rows]
            cm = np.stack([Xr[y == c].mean(0) for c in range(N_CLASSES)])
            cs = np.array([np.bincount(s[y == c]).argmax() for c in range(N_CLASSES)])
            K = np.unique(cs)
            sm = np.stack([cm[cs == k].mean(0) for k in K]); nk = np.array([(cs == k).sum() for k in K])
            between = (nk[:, None] * (sm - cm.mean(0)) ** 2).sum(0) / (len(K) - 1)
            within = sum(((cm[cs == k] - sm[i]) ** 2).sum(0) for i, k in enumerate(K)) / (N_CLASSES - len(K))
            res.append(between / within)
        return (res[0] + res[1]) / 2

    F_refl, F_snv = session_F(X215), session_F(snv(X215))
    c4 = pd.DataFrame(dict(wavelength_nm=wl215, F_reflectance=F_refl, F_snv_of_reflectance=F_snv))
    save(c4, "c4_session_F_reflectance.csv")
    names = ["area", "major", "minor", "aspect", "eccentricity", "solidity", "eq_diameter", "perim/sqrt(area)"]
    save(pd.DataFrame(dict(feature=names, session_F=session_F(morph))), "c4_session_F_morphometrics.csv")


if __name__ == "__main__":
    main()
