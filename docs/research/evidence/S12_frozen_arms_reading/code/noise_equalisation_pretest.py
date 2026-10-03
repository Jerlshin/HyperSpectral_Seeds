"""S12 · CPU pre-test of a candidate S13 input transform: noise-floor equalisation.

F68: the per-kernel high-frequency noise level steps ≈ +25–30 % between sessions 0–4 and 5–8 and is the most
session-decodable within-kernel statistic. Candidate remedy (applied identically at train and test, so it is a
preprocessing, not an augmentation): for every kernel and band, add zero-mean Gaussian noise to its foreground
pixels with sd √max(σ*_b² − σ̂_b², 0), where σ̂_b is the kernel's own high-frequency residual sd and σ*_b the
99th percentile of σ̂_b over the fold's **training** rows. Every kernel then carries (approximately) the same
noise floor, so noise level stops identifying the session; the price is SNR.

Measured here, with the linear controls and the training-rows session probe used throughout S12, before and
after equalisation: session κ (train rows), calib F1 (selection split), and — diagnostically — held-out F1,
cross-session recall and attraction. Nothing is selected for the network on held-out; the result motivates
(or not) a pre-registered network arm.

Writes: noise_equalisation_pretest.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s12common import (CACHE, CROSS, LABELS, REPO, SAME, SESSION, grouped_rows, macro_f1, recalls, save,
                       train_session_by_class)
from session_probe import probe

warnings.filterwarnings("ignore")
DS = REPO / "dataset_u430k32"


def features(P, M, sigma_star=None, hf_est=None, seed=0):
    """Per-kernel pixel statistics (+ HF residual sd), optionally after noise-floor equalisation."""
    rng = np.random.default_rng(seed)
    n, C = P.shape[:2]
    out = {k: np.zeros((n, C), np.float32) for k in ("mean", "sd", "q10", "q50", "q90", "hf")}
    for s in range(0, n, 512):
        xb = np.asarray(P[s:s + 512], dtype=np.float32)
        mb = np.asarray(M[s:s + 512], dtype=np.float32) > 0
        for j in range(xb.shape[0]):
            i, fg = s + j, mb[j]
            x = xb[j]
            if sigma_star is not None:
                add = np.sqrt(np.clip(sigma_star ** 2 - hf_est[i] ** 2, 0, None))
                x = x + (rng.standard_normal(x.shape).astype(np.float32) * add[:, None, None]) * fg
            px = x[:, fg].T
            out["mean"][i], out["sd"][i] = px.mean(0), px.std(0)
            out["q10"][i], out["q50"][i], out["q90"][i] = np.quantile(px, [0.1, 0.5, 0.9], axis=0)
            f = fg.astype(np.float32)
            core = uniform_filter(f, 3) > 0.999
            den = uniform_filter(f, 3) + 1e-8
            for c in range(C):
                img = x[c] * f
                out["hf"][i, c] = (img - uniform_filter(img, 3) / den)[core].std() * 3 / np.sqrt(8)
    return out


def main() -> None:
    P = np.load(DS / "patches.npy", mmap_mode="r")
    M = np.load(DS / "masks.npy", mmap_mode="r")
    morph = np.load(DS / "morphology.npy").astype(np.float64)
    hf0 = np.load(CACHE / "noise_features.npz")["hf"]
    base = features(P, M)
    rows = []
    for f in (0, 1):
        tr, ca, te = grouped_rows(f)
        tss = train_session_by_class(tr)
        sigma_star = np.percentile(hf0[tr], 99, axis=0)
        eq = features(P, M, sigma_star=sigma_star, hf_est=hf0, seed=f)
        for cond, F in (("as recorded", base), ("noise-floor equalised", eq)):
            reps = {
                "HF residual sd": np.log(F["hf"] + 1e-6),
                "mean + morph": np.hstack([F["mean"], morph]),
                "mean + sd + morph": np.hstack([F["mean"], F["sd"], morph]),
                "quantiles 10/50/90 + morph": np.hstack([F["q10"], F["q50"], F["q90"], morph]),
            }
            for name, X in reps.items():
                X = X.astype(np.float64)
                m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(X[tr], LABELS[tr])
                pc, pt = m.predict(X[ca]), m.predict(X[te])
                rec = recalls(LABELS[te], pt)
                cross = np.isin(LABELS[te], CROSS)
                rows.append(dict(condition=cond, representation=name, fold=f, **probe(X, tr),
                                 calib_f1=macro_f1(LABELS[ca], pc), heldout_f1=macro_f1(LABELS[te], pt),
                                 same_recall=float(rec[SAME].mean()), cross_recall=float(rec[CROSS].mean()),
                                 attraction_cross=float((tss[pt[cross]] == SESSION[te][cross]).mean())))
        print(f"fold {f}: σ* (median over bands) = {np.median(sigma_star):.4f}", flush=True)
    df = pd.DataFrame(rows)
    save(df, "noise_equalisation_pretest.csv")
    print(df.groupby(["representation", "condition"])[["kappa", "calib_f1", "heldout_f1", "same_recall",
                                                      "cross_recall", "attraction_cross"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
