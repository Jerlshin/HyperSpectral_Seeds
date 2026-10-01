"""S10 — what input mixup does to a segmented single-kernel patch (train rows only, no model).

Mixup (``losses/mixup.py``) blends two whole 64 × 64 patches pixel-wise, and blends their fill maps
and morphometrics with the same λ. For segmented single objects at different positions and
orientations, the pixel blend is a superimposition of two kernels, and the spectral pathway's
input — the mask-weighted mean spectrum of the blend — is not the λ-blend of the two kernels'
mean spectra, because each kernel's pixels are re-weighted by the *other* kernel's mask.

For 2,000 random training pairs (λ ~ Beta(0.35, 0.35), as shipped) this measures:
  overlap        share of the union foreground where both kernels are present
  mean_err       ‖r(mix) − [λ r₁ + (1−λ) r₂]‖ / ‖λ r₁ + (1−λ) r₂‖, r = masked mean spectrum
  snv_err        the same after SNV (what the spectral MLP actually sees, apart from level)
Writes mixup_geometry.csv (summary quantiles).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from s10common import DS32, save, split_rows

rng = np.random.default_rng(0)
P = np.load(DS32 / "patches.npy", mmap_mode="r")
M = np.load(DS32 / "masks.npy", mmap_mode="r")


def mmean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (x * m[None]).reshape(x.shape[0], -1).sum(1) / max(m.sum(), 1.0)


def snv(r: np.ndarray) -> np.ndarray:
    return (r - r.mean()) / (r.std() + 1e-5)


def main() -> None:
    tr, _ = split_rows("grouped", 0)
    out = []
    for _ in range(2000):
        a, b = rng.choice(tr, 2, replace=False)
        lam = rng.beta(0.35, 0.35)
        xa, xb = np.asarray(P[a], np.float32), np.asarray(P[b], np.float32)
        ma, mb = np.asarray(M[a], np.float32), np.asarray(M[b], np.float32)
        xm, mm = lam * xa + (1 - lam) * xb, lam * ma + (1 - lam) * mb
        ra, rb = mmean(xa, ma), mmean(xb, mb)
        target = lam * ra + (1 - lam) * rb
        got = mmean(xm, mm)
        union = ((ma > 0.5) | (mb > 0.5)).sum()
        both = ((ma > 0.5) & (mb > 0.5)).sum()
        out.append(dict(lam_minor=min(lam, 1 - lam), overlap=both / max(union, 1),
                        mean_err=np.linalg.norm(got - target) / np.linalg.norm(target),
                        snv_err=np.linalg.norm(snv(got) - snv(target)) / np.linalg.norm(snv(target))))
    df = pd.DataFrame(out)
    rows = []
    for label, sub in (("all pairs", df), ("minor share > 0.2", df[df.lam_minor > 0.2])):
        for q in (0.25, 0.5, 0.75):
            rows.append(dict(subset=label, quantile=q, n=len(sub), **sub.quantile(q).to_dict()))
    save(pd.DataFrame(rows), "mixup_geometry.csv")


if __name__ == "__main__":
    main()
