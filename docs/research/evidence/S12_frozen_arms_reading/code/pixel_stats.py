"""S12 · is there variety information in the within-kernel pixel distribution that the mean spectrum discards?

The spectral pathway sees one foreground-mean spectrum per kernel (32 numbers); the spatial pathway sees the
whole cube but must learn from ~41 kernels per class. A set / multiple-instance design would learn from the
~1,000 foreground pixel spectra of every kernel instead. Before proposing it, this CPU control asks whether
pixel-distribution summaries carry *linearly usable* information beyond the mean, and whether that
information is session-robust:

  mean · per-band sd · per-band quantiles (10/50/90) · core vs rim mean (inner/outer half of the kernel by
  distance to its boundary) · per-pixel-SNV mean and sd · log-Euclidean spectral covariance (528 numbers)

each with the 8 morphometrics, through StandardScaler → shrinkage LDA (Ledoit–Wolf), fit on the fold's
``train`` rows. Scored on **calib** (the selection split — the number that may motivate a design) and,
diagnostically, on the held-out bundle with the session breakdown (S09 precedent: nothing is selected on it).

Writes: pixel_controls.csv · (cache) outputs/s12_reading/pixel_features.npz
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s12common import (CACHE, CROSS, LABELS, REPO, SAME, SESSION, grouped_rows, macro_f1, recalls, save,
                       stratified_rows, train_session_by_class)

warnings.filterwarnings("ignore")
DS = REPO / "dataset_u430k32"
FEAT = CACHE / "pixel_features.npz"


def extract() -> dict[str, np.ndarray]:
    if FEAT.exists():
        return dict(np.load(FEAT))
    P = np.load(DS / "patches.npy", mmap_mode="r")
    M = np.load(DS / "masks.npy", mmap_mode="r")
    n, C = P.shape[0], P.shape[1]
    iu = np.triu_indices(C)
    out = {k: np.zeros((n, d), np.float32) for k, d in
           [("mean", C), ("sd", C), ("q10", C), ("q50", C), ("q90", C), ("core", C), ("rim", C),
            ("psnv_mean", C), ("psnv_sd", C), ("logcov", len(iu[0]))]}
    t0 = time.time()
    for s in range(0, n, 256):
        xb = np.asarray(P[s:s + 256], dtype=np.float32)
        mb = np.asarray(M[s:s + 256], dtype=np.float32)
        for j in range(xb.shape[0]):
            fg = mb[j] > 0
            px = xb[j][:, fg].T                       # (n_px, C)
            i = s + j
            out["mean"][i] = px.mean(0)
            out["sd"][i] = px.std(0)
            q = np.quantile(px, [0.1, 0.5, 0.9], axis=0)
            out["q10"][i], out["q50"][i], out["q90"][i] = q
            d = distance_transform_edt(fg)[fg]
            core = d >= np.median(d)
            out["core"][i], out["rim"][i] = px[core].mean(0), px[~core].mean(0)
            z = (px - px.mean(1, keepdims=True)) / (px.std(1, keepdims=True) + 1e-5)
            out["psnv_mean"][i], out["psnv_sd"][i] = z.mean(0), z.std(0)
            lp = np.log(np.clip(px, 1e-4, None))
            cov = np.cov(lp, rowvar=False) + 1e-6 * np.eye(C)
            w, V = np.linalg.eigh(cov)
            out["logcov"][i] = ((V * np.log(np.clip(w, 1e-10, None))) @ V.T)[iu]
        if s % 2048 == 0:
            print(f"  {s}/{n}  {time.time() - t0:.0f}s")
    np.savez(FEAT, **out)
    return out


def main() -> None:
    F = extract()
    morph = np.load(DS / "morphology.npy").astype(np.float64)
    reps = {
        "mean + morph (S09 best linear)": ["mean"],
        "mean + sd + morph": ["mean", "sd"],
        "quantiles 10/50/90 + morph": ["q10", "q50", "q90"],
        "core + rim + morph": ["core", "rim"],
        "pixel-SNV mean + sd + morph": ["psnv_mean", "psnv_sd"],
        "mean + log-cov + morph": ["mean", "logcov"],
        "all pixel statistics + morph": ["mean", "sd", "q10", "q50", "q90", "core", "rim", "psnv_mean", "psnv_sd", "logcov"],
    }
    splits = {("grouped", 0): grouped_rows(0), ("grouped", 1): grouped_rows(1), ("stratified", 0): stratified_rows()}
    rows = []
    for name, keys in reps.items():
        X = np.hstack([F[k].astype(np.float64) for k in keys] + [morph])
        for (proto, f), (tr, ca, te) in splits.items():
            m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(X[tr], LABELS[tr])
            pc, pt = m.predict(X[ca]), m.predict(X[te])
            rec = recalls(LABELS[te], pt)
            r = dict(representation=name, n_features=X.shape[1], protocol=proto, fold=f,
                     calib_f1=macro_f1(LABELS[ca], pc), heldout_f1=macro_f1(LABELS[te], pt),
                     same_recall=float(rec[SAME].mean()))
            if proto == "grouped":
                tss = train_session_by_class(tr)
                cross = np.isin(LABELS[te], CROSS)
                r.update(cross_recall=float(rec[CROSS].mean()),
                         attraction_cross=float((tss[pt[cross]] == SESSION[te][cross]).mean()))
            rows.append(r)
    df = pd.DataFrame(rows)
    save(df, "pixel_controls.csv")
    print(df.groupby(["representation", "protocol"])[["n_features", "calib_f1", "heldout_f1", "same_recall",
                                                       "cross_recall", "attraction_cross"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
