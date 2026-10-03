"""S12 · which within-kernel statistic carries the session: sensor noise or low-frequency spatial variation?

``pixel_controls.csv`` / ``session_probe.csv`` show that per-band pixel spread (sd, quantiles, covariance) adds
same-session accuracy *and* session information. Split that spread into two physically different parts:

* **high-frequency residual** — pixel minus its 3 × 3 foreground neighbourhood mean (mask-normalised box
  filter): dominated by detector/shot noise and fine texture → an acquisition fingerprint if it is session-made;
* **low-frequency spread** — sd of the 3 × 3-smoothed image over the foreground: illumination non-uniformity,
  kernel curvature/geometry, and coarse tissue structure (germ vs endosperm).

Each family (per-band sd, 32 numbers; log-scaled) is scored with the training-rows session probe (class-disjoint
κ) and, diagnostically, as shrinkage-LDA held-out (with morphometrics) like the other controls. Per-session
medians of the noise level are reported so the physical difference is visible.

Writes: noise_channel.csv · noise_by_session.csv · (cache) outputs/s12_reading/noise_features.npz
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s12common import (CACHE, CROSS, LABELS, REPO, SAME, SCANS, SESSION, grouped_rows, macro_f1,
                       recalls, save, train_session_by_class)
from session_probe import probe

warnings.filterwarnings("ignore")
DS = REPO / "dataset_u430k32"
FEAT = CACHE / "noise_features.npz"


def extract() -> dict[str, np.ndarray]:
    if FEAT.exists():
        return dict(np.load(FEAT))
    P = np.load(DS / "patches.npy", mmap_mode="r")
    M = np.load(DS / "masks.npy", mmap_mode="r")
    n, C = P.shape[:2]
    hf, lf, mu = (np.zeros((n, C), np.float32) for _ in range(3))
    for s in range(0, n, 512):
        xb = np.asarray(P[s:s + 512], dtype=np.float32)
        mb = (np.asarray(M[s:s + 512], dtype=np.float32) > 0).astype(np.float32)
        for j in range(xb.shape[0]):
            fg = mb[j]
            core = uniform_filter(fg, 3) > 0.999            # pixels whose whole 3×3 is foreground
            den = uniform_filter(fg, 3) + 1e-8
            for c in range(C):
                img = xb[j, c] * fg
                sm = uniform_filter(img, 3) / den
                r = (img - sm)[core]
                hf[s + j, c] = r.std() * 3 / np.sqrt(8)      # residual sd → per-pixel sd (white-noise scaling)
                lf[s + j, c] = sm[fg > 0].std()
                mu[s + j, c] = img[fg > 0].mean()
    np.savez(FEAT, hf=hf, lf=lf, mu=mu)
    return dict(hf=hf, lf=lf, mu=mu)


def main() -> None:
    F = extract()
    morph = np.load(DS / "morphology.npy").astype(np.float64)
    fams = {
        "high-frequency residual sd": np.log(F["hf"] + 1e-6),
        "low-frequency spread sd": np.log(F["lf"] + 1e-6),
        "relative HF noise (sd / mean)": np.log(F["hf"] / (F["mu"] + 1e-6) + 1e-6),
    }
    rows = []
    for f in (0, 1):
        tr, _, te = grouped_rows(f)
        tss = train_session_by_class(tr)
        for name, X in fams.items():
            pr = probe(X, tr)
            Xm = np.hstack([X, morph])
            m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(Xm[tr], LABELS[tr])
            p = m.predict(Xm[te])
            rec = recalls(LABELS[te], p)
            cross = np.isin(LABELS[te], CROSS)
            rows.append(dict(family=name, fold=f, **pr, heldout_f1_with_morph=macro_f1(LABELS[te], p),
                             same_recall=float(rec[SAME].mean()), cross_recall=float(rec[CROSS].mean()),
                             attraction_cross=float((tss[p[cross]] == SESSION[te][cross]).mean())))
    df = pd.DataFrame(rows)
    save(df, "noise_channel.csv")

    sess = pd.DataFrame(dict(session=SESSION, hf_mid=np.median(F["hf"][:, 8:24], 1),
                             rel_hf=np.median(F["hf"][:, 8:24] / (F["mu"][:, 8:24] + 1e-6), 1),
                             lf_mid=np.median(F["lf"][:, 8:24], 1)))
    by = sess.groupby("session").agg(n=("hf_mid", "size"), hf_median=("hf_mid", "median"),
                                     rel_hf_median=("rel_hf", "median"), lf_median=("lf_mid", "median")).reset_index()
    by["session_name"] = [SCANS.drop_duplicates("session_id").set_index("session_id").loc[s, "session"]
                          if "session" in SCANS.columns else "" for s in by.session]
    save(by, "noise_by_session.csv")
    print(df.groupby("family").mean(numeric_only=True).round(3).drop(columns="fold").to_string())
    print(by.round(5).to_string())


if __name__ == "__main__":
    main()
