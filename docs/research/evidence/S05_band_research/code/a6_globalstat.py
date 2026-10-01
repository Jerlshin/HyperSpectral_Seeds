"""A6 — is the 383-430 nm 'information' just the global SNV statistic? Within training bundles only.
Kernel-level statistics from gain.npy over all foreground pixels: mean of mu, sd, and -mu/sd."""
import numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from common import fold, load, LABELS, WL, REPO
import bandsel as sel
warnings.filterwarnings("ignore")
ms = load("mean_snv").astype(np.float64)
g = np.load(f"{REPO}/dataset/gain.npy", mmap_mode="r")
N = len(ms)
stat = np.zeros((N, 3))
for s in range(0, N, 512):
    gg = np.asarray(g[s:s+512], dtype=np.float64)
    m = gg[:, 1] > 0
    mu = np.where(m, gg[:, 0], 0).sum((1, 2)) / m.sum((1, 2))
    sd = np.where(m, gg[:, 1], 0).sum((1, 2)) / m.sum((1, 2))
    r = np.where(m, -gg[:, 0] / np.where(m, gg[:, 1], 1), 0).sum((1, 2)) / m.sum((1, 2))
    stat[s:s+512] = np.stack([np.log(mu), np.log(sd), r], 1)
np.save("cache/kernel_gainstats.npy", stat)
y = LABELS
def cv(X, pool):
    s = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(pool, y[pool]):
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage=1e-3)).fit(X[pool[a]], y[pool[a]])
        s.append(f1_score(y[pool[b]], m.predict(X[pool[b]]), average="macro"))
    return np.mean(s)
for f in (0, 1):
    tr, ca, _ = fold(f); pool = np.concatenate([tr, ca])
    print(f"=== fold {f} ===  k | uniform(383-1006) | uniform430 | uniform430 + (-mu/sd) | uniform430 + log mu, log sd, -mu/sd")
    for k in [16, 32, 64, 128]:
        U = ms[:, sel.uniform(k)]; U4 = ms[:, sel.uniform_nm(k, 430.0)]
        print(f"  {k:3d} | {cv(U, pool):.3f} | {cv(U4, pool):.3f} | {cv(np.c_[U4, stat[:, 2]], pool):.3f} | {cv(np.c_[U4, stat], pool):.3f}")
