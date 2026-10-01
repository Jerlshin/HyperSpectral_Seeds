"""A2 — which proxy is a credible stand-in? Full 256 bands, within-training-bundle only.
Scores: (a) train->calib (the existing study's decision split), (b) 5-fold CV over train∪calib."""
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import fold, load, LABELS
from proxies import lda_svd, lda_shrink, logreg, pixel_lda

ms = load("mean_snv"); sd = load("sd_snv"); px = load("px_snv")
feats = {
    "mean": ms,
    "mean+sd": np.concatenate([ms, sd], 1),
}
for f in (0, 1):
    tr, ca, _ = fold(f)
    pool = np.concatenate([tr, ca])
    y = LABELS
    print(f"=== fold {f} ===")
    rows = []
    for fname, X in feats.items():
        for pname, fn in [("lda_svd", lda_svd), ("lda_shrink", lda_shrink), ("logreg", logreg)]:
            t = time.time()
            cal = fn(X[tr], y[tr], X[ca], y[ca])
            cv = []
            for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(pool, y[pool]):
                cv.append(fn(X[pool[a]], y[pool[a]], X[pool[b]], y[pool[b]]))
            cv = np.array(cv)
            print(f"{fname:8s} {pname:10s} calibF1 {cal[0]:.4f} | CV F1 {cv[:,0].mean():.4f}±{cv[:,0].std():.4f} acc {cv[:,1].mean():.4f}  ({time.time()-t:.0f}s)")
    t = time.time()
    cal = pixel_lda(px[tr], y[tr], px[ca], y[ca])
    cv = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(pool, y[pool]):
        cv.append(pixel_lda(px[pool[a]], y[pool[a]], px[pool[b]], y[pool[b]]))
    cv = np.array(cv)
    print(f"pixel48  pix_lda    calibF1 {cal[0]:.4f} | CV F1 {cv[:,0].mean():.4f}±{cv[:,0].std():.4f} acc {cv[:,1].mean():.4f}  ({time.time()-t:.0f}s)")
