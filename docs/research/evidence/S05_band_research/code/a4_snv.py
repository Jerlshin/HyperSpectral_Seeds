"""A4 — is a k-band result achievable by a k-band instrument?

patches.npy is per-pixel SNV computed over ALL 256 bands, so any 'k selected bands' taken from
it carry the discarded bands' information through each pixel's mean and sd. A real k-band sensor
can only normalise over its own k bands. Rebuild radiance for 48 sampled pixels per kernel
(raw = snv*sd + mu, from gain.npy) and compare, within training bundles only:
  snv256 : SNV over 256, keep S           (what selecting from patches.npy measures)
  snvS   : SNV over S only                (what a k-band sensor can do)
  rawS   : no SNV, radiance at S          (reference)
"""
import numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from common import fold, load, LABELS, WL, C
import bandsel as sel
warnings.filterwarnings("ignore")

px = load("px_snv")            # (N,48,256)
pg = load("px_gain")           # (N,48,2)
raw = px * pg[:, :, 1:2] + pg[:, :, 0:1]
y = LABELS

def feats(S, mode):
    if mode == "snv256":
        return px[:, :, S].mean(1)
    r = raw[:, :, S]
    if mode == "rawS":
        return r.mean(1)
    mu = r.mean(2, keepdims=True); sd = r.std(2, keepdims=True) + 1e-6
    return ((r - mu) / sd).mean(1)

def cv(X, pool):
    s = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(pool, y[pool]):
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage=1e-3)).fit(X[pool[a]], y[pool[a]])
        s.append(f1_score(y[pool[b]], m.predict(X[pool[b]]), average="macro"))
    return np.mean(s)

for f in (0, 1):
    tr, ca, _ = fold(f); pool = np.concatenate([tr, ca])
    print(f"=== fold {f} ===   (48-pixel kernel means; LDA shrink 1e-3; 5-fold CV inside the training bundles)")
    print(f"{'set':18s} {'k':>4s}  snv256  snvS   rawS")
    for name, fn in [("uniform", lambda k: sel.uniform(k)), ("uniform_430", lambda k: sel.uniform_nm(k, 430.0))]:
        for k in [4, 8, 16, 32, 64, 128, 256]:
            if name == "uniform_430" and k == 256:
                continue
            S = fn(k)
            r = [cv(feats(S, m), pool) for m in ("snv256", "snvS", "rawS")]
            print(f"{name:18s} {k:4d}  {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}")
