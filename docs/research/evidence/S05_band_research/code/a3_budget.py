"""A3 — within-training-bundle budget curves, nested selection, 2 folds x 5 outer CV splits.
Nothing here reads the held-out bundles."""
import json, sys, time
import numpy as np, warnings
from scipy.fft import dct
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from common import fold, load, LABELS, WL, C
import bandsel as sel
warnings.filterwarnings("ignore")

BUDGETS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
KMAX_GREEDY = 96
ms = load("mean_snv").astype(np.float64)
y = LABELS
SNR_LO_NM = 430.0   # pixel SNR >= 10 from a1

def lda_reg():
    return make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage=1e-3))
def lda_svd():
    return make_pipeline(StandardScaler(), LDA(solver="svd", tol=1e-4))
EVALS = dict(lda_reg=lda_reg, lda_svd=lda_svd)

def score(X, a, b, mk):
    m = mk().fit(X[a], y[a]); return f1_score(y[b], m.predict(X[b]), average="macro")

methods = sys.argv[1].split(",") if len(sys.argv) > 1 else ["uniform", "uniform_snr", "random", "mrmr", "spa", "cluster_ward", "glw", "glw_snr", "dct", "bin"]
rows = []
t0 = time.time()
for f in (0, 1):
    tr, ca, _ = fold(f)
    pool = np.concatenate([tr, ca])
    for oi, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(pool, y[pool])):
        A, B = pool[a], pool[b]
        for meth in methods:
            sets = {}
            feats = None
            if meth == "uniform":
                sets = {k: sel.uniform(k) for k in BUDGETS}
            elif meth == "uniform_snr":
                sets = {k: sel.uniform_nm(k, SNR_LO_NM) for k in BUDGETS if k <= int((WL >= SNR_LO_NM).sum())}
            elif meth == "random":
                rng = np.random.default_rng(1000 * f + oi)
                sets = {k: np.sort(rng.choice(C, k, replace=False)) for k in BUDGETS}
            elif meth in ("mrmr", "spa", "cluster_ward"):
                sets = sel.repo_method(meth, ms[A], y[A], [k for k in BUDGETS])
            elif meth in ("glw", "glw_snr"):
                cand = None if meth == "glw" else np.flatnonzero(WL >= SNR_LO_NM)
                order = sel.greedy_lda_wrapper(ms[A], y[A], KMAX_GREEDY, candidates=cand, seed=oi)
                sets = {k: np.sort(order[:k]) for k in BUDGETS if k <= KMAX_GREEDY}
            elif meth == "dct":
                D = dct(ms, type=2, norm="ortho", axis=1)
                feats = {k: D[:, :k] for k in BUDGETS}
            elif meth == "bin":
                edges = lambda k: np.array_split(np.arange(C), k)
                feats = {k: np.stack([ms[:, e].mean(1) for e in edges(k)], 1) for k in BUDGETS}
            for k in BUDGETS:
                if feats is not None:
                    X = feats[k]
                elif k in sets:
                    X = ms[:, sets[k]]
                else:
                    continue
                for en, mk in EVALS.items():
                    rows.append(dict(fold=f, outer=oi, method=meth, k=k, proxy=en, f1=score(X, A, B, mk),
                                     bands=None if feats is not None else [int(i) for i in sets[k]]))
            print(f"fold {f} outer {oi} {meth:12s} done  {time.time()-t0:.0f}s", flush=True)
        json.dump(rows, open(f"a3_budget_{'_'.join(methods) if len(methods)<4 else 'all'}.json", "w"))
