"""Proxy estimators. All return macro-F1 on the given eval rows."""
import warnings
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")


def _score(y, p):
    return f1_score(y, p, average="macro"), accuracy_score(y, p)


def lda_svd(Xtr, ytr, Xev, yev):
    m = make_pipeline(StandardScaler(), LDA(solver="svd", tol=1e-4)).fit(Xtr, ytr)
    return _score(yev, m.predict(Xev))


def lda_shrink(Xtr, ytr, Xev, yev):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(Xtr, ytr)
    return _score(yev, m.predict(Xev))


def logreg(Xtr, ytr, Xev, yev, C=1.0):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr, ytr)
    return _score(yev, m.predict(Xev))


def pixel_lda(Ptr, ytr, Pev, yev):
    """Ptr: (n, P, k) pixel spectra per kernel. Fit shrinkage-LDA on pixels, aggregate
    kernel posterior as the sum of pixel log-probabilities (naive-Bayes pooling)."""
    n, P, k = Ptr.shape
    sc = StandardScaler().fit(Ptr.reshape(-1, k))
    m = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(Ptr.reshape(-1, k)), np.repeat(ytr, P))
    ne, Pe, _ = Pev.shape
    lp = m.predict_log_proba(sc.transform(Pev.reshape(-1, k))).reshape(ne, Pe, -1).sum(1)
    return _score(yev, m.classes_[lp.argmax(1)])


PROXIES = dict(lda_svd=lda_svd, lda_shrink=lda_shrink, logreg=logreg)
