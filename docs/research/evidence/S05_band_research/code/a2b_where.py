"""Where does the within-bundle signal live? regularisation path + spectral frequency (DCT) + region."""
import numpy as np, warnings
from scipy.fft import dct
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from common import fold, load, LABELS, WL
warnings.filterwarnings("ignore")
ms = load("mean_snv").astype(np.float64)
y = LABELS
def cv(X, pool, model):
    s=[]
    for a,b in StratifiedKFold(5, shuffle=True, random_state=0).split(pool, y[pool]):
        m = model().fit(X[pool[a]], y[pool[a]]); s.append(f1_score(y[pool[b]], m.predict(X[pool[b]]), average="macro"))
    return np.mean(s)
for f in (0,1):
    tr, ca, _ = fold(f); pool = np.concatenate([tr, ca])
    print(f"=== fold {f} ===")
    for sh in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
        mk = (lambda sh=sh: make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage=sh))) if sh>0 else (lambda: make_pipeline(StandardScaler(), LDA(solver="svd", tol=1e-4)))
        print(f"shrinkage {sh:g}: {cv(ms, pool, mk):.4f}")
    # DCT low-pass: keep first m coefficients (spectral frequency truncation == Nyquist-limited uniform sampling)
    D = dct(ms, type=2, norm="ortho", axis=1)
    svd = lambda: make_pipeline(StandardScaler(), LDA(solver="svd", tol=1e-4))
    line=[]
    for m in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
        line.append(f"{m}:{cv(D[:, :m], pool, svd):.3f}")
    print("DCT low-pass m:", " ".join(line))
    # high-pass only: drop the first m coefficients
    line=[]
    for m in [8, 16, 32, 64, 128]:
        line.append(f">{m}:{cv(D[:, m:], pool, svd):.3f}")
    print("DCT high-pass only:", " ".join(line))
    # region ablations (contiguous windows)
    for lo, hi in [(383,430),(430,500),(500,600),(600,700),(700,800),(800,900),(900,1007)]:
        idx = np.flatnonzero((WL>=lo)&(WL<hi))
        print(f"only {lo}-{hi} nm ({len(idx)} b): {cv(ms[:, idx], pool, svd):.3f}   drop it: {cv(np.delete(ms, idx, 1), pool, svd):.3f}")
