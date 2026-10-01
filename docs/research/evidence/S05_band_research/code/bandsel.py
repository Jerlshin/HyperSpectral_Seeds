"""Band selectors. Each returns either a ranking (nested) or a dict {k: bands}."""
import numpy as np
from common import WL, C
import sys
from spectralquadnet.bandstudy import methods as bsm


def uniform(k, lo=0, hi=C - 1):
    return np.unique(np.round(np.linspace(lo, hi, k)).astype(int))


def uniform_nm(k, lo_nm, hi_nm=WL[-1]):
    """k bands evenly spaced in wavelength between lo_nm and hi_nm (nearest band)."""
    targets = np.linspace(lo_nm, hi_nm, k)
    idx = np.abs(WL[None, :] - targets[:, None]).argmin(1)
    return np.unique(idx)


def repo_method(name, X, y, budgets, seed=0):
    ctx = bsm.SelectionContext(x=np.asarray(X, np.float64), y=np.asarray(y), seed=seed)
    out = bsm.run_method(name, ctx, list(budgets), draws=1)
    if out.failure:
        raise RuntimeError(out.failure)
    return {k: np.array(sorted(v[0])) for k, v in out.per_budget.items()}


def greedy_lda_wrapper(X, y, kmax, candidates=None, shrink=1e-3, n_inner=3, seed=0):
    """Greedy forward selection maximising the inner-CV mean log-likelihood of a
    (lightly shrunk) shared-covariance Gaussian (LDA) classifier.

    Exact sequential conditioning: every column of the class-difference tensor is kept
    residualised on the selected set in the Sw metric (Gram-Schmidt), and Sw is kept as
    the Schur complement. Adding band j:
        r_m <- r_m - r_j * Sres[j,m] / Sres[j,j],   Sres <- Sres - Sres[:,j] Sres[j,:] / Sres[j,j]
    and the Mahalanobis distance of candidate m is d + r_m^2 / Sres[m,m].
    Returns the selection order (nested ranking of length kmax).
    """
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, np.float64)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    p = Z.shape[1]
    allowed = np.zeros(p, bool)
    allowed[np.arange(p) if candidates is None else np.asarray(candidates)] = True
    splits = []
    for a, b in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(Z, y):
        Za, ya, Zb, yb = Z[a], y[a], Z[b], y[b]
        cls = np.unique(ya)
        M = np.stack([Za[ya == c].mean(0) for c in cls])
        W = Za - M[np.searchsorted(cls, ya)]
        Sw = W.T @ W / (len(ya) - len(cls))
        Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / p * np.eye(p)
        R = (Zb[:, None, :] - M[None, :, :]).astype(np.float32)
        splits.append(dict(S=Sw.copy(), R=R, yi=np.searchsorted(cls, yb), d=np.zeros(R.shape[:2], np.float32),
                           lp=np.log(np.bincount(np.searchsorted(cls, ya)) / len(ya)).astype(np.float32)))
    order = []
    for _ in range(kmax):
        total = np.zeros(p)
        for s in splits:
            var = np.clip(np.diag(s["S"]), 1e-9, None).astype(np.float32)
            logit = -0.5 * (s["d"][:, :, None] + s["R"] ** 2 / var) + s["lp"][None, :, None]
            lse = np.logaddexp.reduce(logit, axis=1)
            true = logit[np.arange(len(s["yi"])), s["yi"], :]
            total += (true - lse).mean(0)
        total[~allowed] = -np.inf
        total[order] = -np.inf
        j = int(np.argmax(total))
        for s in splits:
            Sj = s["S"][:, j].copy()
            vj = max(Sj[j], 1e-12)
            rj = s["R"][:, :, j].copy()
            s["d"] += rj ** 2 / np.float32(vj)
            s["R"] -= rj[:, :, None] * (Sj / vj).astype(np.float32)[None, None, :]
            s["S"] -= np.outer(Sj, Sj) / vj
        order.append(j)
    return np.array(order)
