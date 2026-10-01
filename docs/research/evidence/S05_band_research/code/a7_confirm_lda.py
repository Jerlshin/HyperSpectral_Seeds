"""A7 — PRE-REGISTERED held-out confirmation with the linear proxy.

Refuses to run unless preregistration.json exists and its sha256 matches preregistration.sha256
(written when the list was frozen). Scores each frozen arm ONCE per fold: fit on train∪calib
(the fold's training bundles), score on val∪test (the other bundle). Bootstrap CI by kernel.
"""
import hashlib, json, sys
import numpy as np, warnings
from scipy.fft import dct
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
from common import fold, load, LABELS
warnings.filterwarnings("ignore")

pre = open("preregistration.json", "rb").read()
if hashlib.sha256(pre).hexdigest() != open("preregistration.sha256").read().strip():
    sys.exit("preregistration.json changed after it was frozen — refusing.")
P = json.loads(pre)
ms = load("mean_snv").astype(np.float64)
gs = np.load("cache/kernel_gainstats.npy")
D = dct(ms, type=2, norm="ortho", axis=1)
y = LABELS
rng = np.random.default_rng(0)

def features(arm, f):
    a = P["lda_arms"][arm]
    if a["kind"] == "bands":
        X = ms[:, a["bands"][str(f)]]
    elif a["kind"] == "dct":
        X = D[:, : a["m"]]
    else:
        raise ValueError(a)
    extra = a.get("extra", [])
    if extra:
        X = np.c_[X, gs[:, extra]]
    return X

out = []
for f in (0, 1):
    tr, ca, held = fold(f)
    fit_rows = np.concatenate([tr, ca])
    print(f"!! HELD-OUT REVEAL fold {f}: {len(held)} kernels, {len(P['lda_arms'])} frozen arms", flush=True)
    boot = [rng.choice(len(held), len(held)) for _ in range(500)]
    for arm in P["lda_arms"]:
        X = features(arm, f)
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage=P["lda_shrinkage"])).fit(X[fit_rows], y[fit_rows])
        pred = m.predict(X[held]); yt = y[held]
        bs = [f1_score(yt[b], pred[b], average="macro") for b in boot[:200]]
        out.append(dict(arm=arm, fold=f, k=X.shape[1], f1=f1_score(yt, pred, average="macro"),
                        acc=accuracy_score(yt, pred), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
        print(f"fold {f} {arm:28s} k={X.shape[1]:3d}  F1 {out[-1]['f1']:.4f}  acc {out[-1]['acc']:.4f}", flush=True)
json.dump(out, open("confirm_lda.json", "w"), indent=1)
