"""A1 — label-free and train-only spectral characterisation.

1. per-band SNR in radiance (spatial-neighbour and spectral-2nd-difference noise estimates)
2. band-band correlation structure (correlation length)
3. intrinsic dimension: PCA, noise-whitened (MNF-style) signal subspace
4. discriminant dimension: eigen-spectrum of Sw^-1 Sb on the training bundle
5. Nyquist argument: PSD along wavelength of (a) the spectra, (b) the class-mean
   (between-class) signal, (c) within-class noise -> the spectral frequency above which
   discriminative power falls under noise -> uniform sampling interval needed.
"""
import json
import numpy as np
from common import fold, load, WL, LABELS, C

out = {}
mr = load("mean_raw")
ms = load("mean_snv")
nsp = load("noise_sp_raw")
nsd = load("noise_sd2_raw")

# ---------- 1. SNR (per kernel pixel level, radiance) ----------
sig = np.median(mr, 0)
noise_sp = np.sqrt(np.median(nsp, 0))
noise_sd = np.sqrt(np.median(nsd, 0))
snr_sp = sig / noise_sp
snr_sd = sig / noise_sd
# kernel-mean SNR: pixel noise / sqrt(n_px) -- ~900 fg px on avg
out["snr_pixel_sd2"] = snr_sd.round(2).tolist()
out["snr_pixel_spatial"] = snr_sp.round(2).tolist()
print("band  nm   radiance  pixSNR(spectral2nd)  pixSNR(spatial-nbr, incl texture)")
for i in list(range(0, 30, 3)) + list(range(30, 220, 20)) + list(range(220, 256, 4)) + [255]:
    print(f"{i:4d} {WL[i]:6.1f} {sig[i]:9.1f} {snr_sd[i]:10.1f} {snr_sp[i]:10.1f}")
for thr in (5, 10, 20, 50):
    good = np.flatnonzero(snr_sd >= thr)
    print(f"pixel SNR>={thr}: bands {good.min()}..{good.max()}  ({WL[good.min()]:.1f}-{WL[good.max()]:.1f} nm), n={len(good)}")

# ---------- 2-5 on training rows of each fold ----------
for f in (0, 1):
    tr, ca, _ = fold(f)
    X = ms[tr].astype(np.float64)
    y = LABELS[tr]
    res = {}
    # correlation length
    R = np.corrcoef(X.T)
    adj = np.array([R[i, i + 1] for i in range(C - 1)])
    res["adjacent_r_median"] = float(np.median(adj))
    res["adjacent_r_min"] = float(adj.min())
    lens = {}
    for thr in (0.99, 0.95, 0.9):
        L = []
        for i in range(C):
            j = i
            while j + 1 < C and R[i, j + 1] >= thr:
                j += 1
            L.append(j - i)
        lens[str(thr)] = float(np.median(L))
    res["corr_length_bands_median"] = lens
    # PCA
    Xc = X - X.mean(0)
    ev = np.linalg.eigvalsh(np.cov(Xc.T))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    res["pca_n_for"] = {str(p): int(np.searchsorted(cum, p) + 1) for p in (0.99, 0.999, 0.9999)}
    res["participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    # noise-whitened: within-class covariance as 'noise' for MNF-like SNR ordering
    classes = np.unique(y)
    mu_c = np.stack([X[y == c].mean(0) for c in classes])
    Xw = X - mu_c[np.searchsorted(classes, y)]
    Sw = np.cov(Xw.T) + 1e-6 * np.eye(C)
    mu = X.mean(0)
    n_c = np.array([(y == c).sum() for c in classes])
    Sb = ((mu_c - mu).T * n_c) @ (mu_c - mu) / len(y)
    # generalized eigen Sb v = l Sw v
    Lw = np.linalg.cholesky(Sw)
    Li = np.linalg.inv(Lw)
    M = Li @ Sb @ Li.T
    gev = np.sort(np.linalg.eigvalsh((M + M.T) / 2))[::-1]
    gev = np.clip(gev, 0, None)
    cumd = np.cumsum(gev) / gev.sum()
    res["fisher_eig_top10"] = gev[:10].round(3).tolist()
    res["discriminant_dirs_for"] = {str(p): int(np.searchsorted(cumd, p) + 1) for p in (0.9, 0.95, 0.99)}
    res["discriminant_dirs_eig_gt_0.1"] = int((gev > 0.1).sum())
    res["discriminant_dirs_eig_gt_1"] = int((gev > 1.0).sum())

    # ---------- 5. PSD / Nyquist on a uniform grid (wavelength is ~uniform, 2.444 nm) ----------
    # restrict to the usable window to avoid the blue-end noise dominating
    band_rng = slice(0, C)
    def psd(A):
        A = A - A.mean(1, keepdims=True)
        w = np.hanning(A.shape[1])
        F = np.fft.rfft(A * w, axis=1)
        return (np.abs(F) ** 2).mean(0)
    dl = float(np.mean(np.diff(WL)))
    freqs = np.fft.rfftfreq(C, d=dl)       # cycles / nm
    P_between = psd((mu_c - mu))           # class-mean deviations: the discriminative signal
    P_within = psd(Xw)                     # within-class variation (kernel-to-kernel + noise)
    ratio = P_between / P_within
    res["psd_freq_cyc_per_nm"] = freqs.round(5).tolist()
    res["psd_between"] = P_between.tolist()
    res["psd_within"] = P_within.tolist()
    # cumulative between-class power
    cb = np.cumsum(P_between) / P_between.sum()
    for p in (0.95, 0.99, 0.999):
        k = int(np.searchsorted(cb, p))
        fc = freqs[k]
        # Nyquist: sample interval 1/(2 fc); number of uniform bands over the range
        n_uniform = int(np.ceil((WL[-1] - WL[0]) * 2 * fc)) + 1
        res[f"between_power_{p}_fc"] = float(fc)
        res[f"between_power_{p}_nyquist_bands"] = n_uniform
    # frequency beyond which between-class power < within-class power per frequency bin (Fisher ratio<1/n_c)
    res["fisher_ratio_by_freq_first10"] = ratio[:10].round(4).tolist()
    out[f"fold{f}"] = res
    print(f"\n=== fold {f} (train n={len(tr)}) ===")
    for k, v in res.items():
        if not k.startswith("psd"):
            print(k, v)

json.dump(out, open("a1_spectral.json", "w"))
