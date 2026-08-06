"""Research-correct hyperspectral band selection (mRMR + SPA + validated elbow).

Relocated from the root script ``band_selection.py`` @ ``886560f``:

===================================  ==============
Symbol                               Baseline lines
===================================  ==============
:func:`extract_mean_spectra`         98-127
:func:`decorrelation_prefilter`      133-159
:func:`fisher_discriminant_ratio`    165-196
:func:`run_mrmr`                     202-273
:func:`run_spa`                      279-361
:func:`validate`                     367-422
:func:`find_elbow`                   428-441
:func:`save_outputs`                 447-472
:func:`select_bands`                 478-563  (was ``main``)
===================================  ==============

Declared deviations, all mechanical:

* the module-level ``CONFIG`` dict → an explicit
  :class:`~spectralquadnet.data.prep.config.BandSelectionConfig` parameter;
* ``np.random.seed(CONFIG["seed"])`` at import (line 92) is gone — it is now
  applied inside :func:`select_bands`, so importing this module no longer
  perturbs the global NumPy RNG (REFACTOR_PLAN.md §3.6);
* the module-level ``warnings.filterwarnings`` calls (lines 58-60) moved into
  :func:`select_bands` for the same reason;
* ``main`` renamed to :func:`select_bands`; the ``if __name__`` guard is dropped
  in favour of Phase 4's ``scripts/select_bands.py``.

This is the pipeline that produced ``dataset/patches_spa_40b.npy`` — the 40-band
SPA subset the ``output_v12_spa40`` experiment trains on.

Why mRMR + SPA for this task
------------------------------
• CARS is a regression method (PLS-R); it requires a continuous scalar
  target and is not designed for 90-class discrimination.
• mRMR (Peng et al., IEEE TPAMI 2005) works directly with class labels
  via mutual information. Its MID criterion simultaneously maximises
  class relevance and penalises inter-band redundancy.
• SPA (Araújo et al., Chemom. Intell. Lab. 2001) selects geometrically
  orthogonal bands (Gram-Schmidt residuals), complementing mRMR's
  statistical approach by minimising multicollinearity.
• Both methods are validated at multiple band counts and the best winner
  + optimal count are determined automatically from a StratifiedKFold
  accuracy curve, not from a proxy regression metric.

References
----------
Peng H, Long F, Ding C. Feature selection based on mutual information:
  criteria of max-dependency, max-relevance, and min-redundancy.
  IEEE Trans Pattern Anal Mach Intell 2005;27(8):1226-38.

Araújo M, Saldanha T, Galvao R, Yoneyama T, Chame H, Visani V.
  The successive projections algorithm for variable selection in
  spectroscopic multicomponent analysis.
  Chemom Intell Lab Syst 2001;57(2):65-73.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from spectralquadnet.data.prep.config import BandSelectionConfig


# =====================================================================
# STEP 1 — Mean spectra extraction
# =====================================================================
def extract_mean_spectra(cfg: BandSelectionConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Load patches via memory-map and compute the per-patch spatially
    averaged spectrum over foreground (non-background) pixels only.

    Background detection: pixels whose band-sum absolute value < 1e-5
    are treated as padded zeros from the segmentation pipeline.

    Returns
    -------
    X : float32 (N, C)  — mean spectrum per patch
    y : int64   (N,)    — class labels
    """
    print("\n[1/6]  Extracting mean spectra...")
    patches = np.load(cfg.patches_path, mmap_mode="r")
    labels = np.load(cfg.labels_path)
    N, C, H, W = patches.shape
    X = np.zeros((N, C), dtype=np.float32)

    for s in tqdm(range(0, N, cfg.chunk_size), desc="       chunks"):
        e = min(s + cfg.chunk_size, N)
        b = patches[s:e].astype(np.float32)  # (B, C, H, W)
        flat = b.reshape(len(b), C, -1)  # (B, C, H*W)
        # Binary foreground mask per pixel (sum across bands)
        mask = (np.abs(flat).sum(axis=1, keepdims=True) > 1e-5).astype(np.float32)
        valid_px = mask.sum(axis=2).clip(min=1e-5)  # (B, 1) pixel counts
        X[s:e] = (flat * mask).sum(axis=2) / valid_px  # masked mean (B, C)

    print(f"         X = {X.shape}   classes = {np.unique(labels).size}")
    return X, labels.astype(np.int64)


# =====================================================================
# STEP 2 — Decorrelation pre-filter
# =====================================================================
def decorrelation_prefilter(X: np.ndarray, cfg: BandSelectionConfig) -> np.ndarray:
    """
    Remove near-duplicate spectral bands by greedy correlation screening.
    Scans bands left-to-right (low → high wavelength); keeps a band only
    if its |Pearson r| with every previously kept band is ≤ threshold.

    This reduces the candidate pool before running mRMR / SPA so that
    both methods operate on a set of already-distinct spectral directions.

    Returns
    -------
    candidates : int64 array — original-space indices of kept bands, sorted.
    """
    thr = cfg.corr_threshold
    print(f"\n[2/6]  Decorrelation pre-filter  (|r| > {thr} → drop)...")

    # Full (P, P) correlation matrix — cheap for P = 256
    Corr = np.corrcoef(X.T)
    keep: list[int] = []
    for i in range(X.shape[1]):
        # Drop band i if it is near-duplicate of any already-kept band
        if not any(abs(Corr[i, j]) > thr for j in keep):
            keep.append(i)

    candidates = np.array(keep, dtype=np.int64)
    print(f"         Kept {len(candidates)} / {X.shape[1]} bands after pre-filter")
    return candidates


# =====================================================================
# STEP 3 — Fisher Discriminant Ratio  (diagnostic)
# =====================================================================
def fisher_discriminant_ratio(
    X: np.ndarray, y: np.ndarray, wl_df: pd.DataFrame
) -> np.ndarray:
    """
    Computes the multiclass Fisher Discriminant Ratio for every band:

        FDR_k = Σ_c  n_c (μ_ck − μ_k)²
                ─────────────────────────
                Σ_c  n_c  σ²_ck

    Higher FDR → band separates classes better relative to within-class
    variance. Used here as a diagnostic; mRMR drives the actual selection.

    Returns fdr : float64 (P,)
    """
    print("\n[3/6]  Fisher Discriminant Ratio  (top-10 for reference)...")
    classes = np.unique(y)
    mu_global = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])

    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        between += nc * (Xc.mean(axis=0) - mu_global) ** 2
        within += nc * Xc.var(axis=0)

    fdr = between / (within + 1e-10)
    top = np.argsort(fdr)[::-1][:10]
    for rank, b in enumerate(top, 1):
        wl = wl_df.iloc[b]["Wavelength (nm)"]
        print(f"         #{rank:2d}  band {b:3d}  ({wl:7.2f} nm)  FDR = {fdr[b]:.4f}")
    return fdr


# =====================================================================
# STEP 4 — mRMR  (Min-Redundancy Max-Relevance)
# =====================================================================
def run_mrmr(
    X: np.ndarray, y: np.ndarray, candidates: np.ndarray, cfg: BandSelectionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy mRMR using the MID (Mutual Information Difference) criterion:

        J(x_k) = MI(x_k ; y)  −  (1/|S|) Σ_{x_j ∈ S} MI(x_k ; x_j)

    Relevance term  MI(band ; class):
        Computed with sklearn's mutual_info_classif (k-NN estimator for
        continuous features vs. discrete labels; no Gaussian assumption).

    Redundancy term  MI(band_i ; band_j):
        Approximated as  −½ log(1 − r²ij + ε)  where r is the Pearson
        correlation. This is exact for bivariate Gaussians and a very
        good approximation for spectrally smooth HSI bands. It avoids a
        costly O(P²) k-NN MI matrix computation.

    Parameters
    ----------
    X          : (N, P_full) mean spectra
    y          : (N,) class labels
    candidates : original-space indices surviving decorrelation

    Returns
    -------
    ordered   : original-space band indices in mRMR selection order
    rel_local : MI relevance score per candidate (local indexing)
    """
    print(f"\n[4/6]  mRMR greedy selection  (up to {cfg.n_select_max} bands)...")

    Xc = X[:, candidates]  # (N, P_cand) — candidate sub-matrix
    P = len(candidates)
    eps = 1e-10

    # ── Relevance: MI(band_i ; y) ──────────────────────────────────
    rel = mutual_info_classif(
        Xc,
        y,
        discrete_features=False,
        n_neighbors=cfg.mi_neighbors,
        random_state=cfg.seed,
    )  # (P_cand,)
    print(f"         MI relevance  min={rel.min():.4f}  max={rel.max():.4f}")

    # ── Redundancy proxy: Pearson → MI ─────────────────────────────
    Corr = np.corrcoef(Xc.T)  # (P, P)
    np.clip(Corr, -1.0 + eps, 1.0 - eps, out=Corr)
    MI_bb = -0.5 * np.log(1.0 - Corr**2 + eps)  # (P, P), symmetric

    # ── Greedy forward selection ────────────────────────────────────
    n_select = min(cfg.n_select_max, P)
    selected = []  # local indices into Xc / candidates
    remaining = list(range(P))

    # Seed with the highest-relevance band
    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    for _ in tqdm(range(n_select - 1), desc="       steps"):
        if not remaining:
            break
        rem = np.array(remaining)
        # Mean MI between each candidate and all already-selected bands
        mean_redundancy = MI_bb[rem][:, selected].mean(axis=1)  # (R,)
        scores = rel[rem] - mean_redundancy  # (R,) MID
        best = int(np.argmax(scores))
        selected.append(remaining[best])
        remaining.pop(best)

    ordered = candidates[np.array(selected)]
    print(f"         First 10 bands (original idx): {ordered[:10].tolist()}")
    return ordered, rel


# =====================================================================
# STEP 5 — SPA  (Successive Projections Algorithm)
# =====================================================================
def run_spa(
    X: np.ndarray, candidates: np.ndarray, init_global: int, cfg: BandSelectionConfig
) -> np.ndarray:
    """
    Successive Projections Algorithm — selects a maximally orthogonal
    (geometrically non-redundant) band subset via sequential Gram-Schmidt
    orthogonalisation.

    Algorithm
    ---------
    1. Column-normalise all candidate bands to unit L2 norm.
    2. Set the seed band as the first selected column.
    3. For each subsequent step:
       a. Project the last selected column out of all remaining columns:
              x_j ← x_j − (x_j · pivot / ‖pivot‖²) pivot
       b. Select the remaining band with the largest residual norm.
    4. Map local indices back to original 256-band space.

    The pivot for each step is the last selected band in its current
    (already orthogonalised) state — equivalent to Gram-Schmidt on the
    data matrix. This guarantees maximum geometric spread in spectral
    space and minimises multicollinearity in any downstream model.

    init_global is set to the highest-MI band so SPA's geometric
    exploration is anchored on a direction already known to be
    discriminative.

    Parameters
    ----------
    X           : (N, P_full) mean spectra
    candidates  : original-space indices surviving decorrelation
    init_global : starting band (original index)

    Returns
    -------
    ordered : original-space band indices in SPA selection order
    """
    print(f"\n[5/6]  SPA selection  (up to {cfg.n_select_max} bands)...")

    Xc = X[:, candidates].astype(np.float64)  # work in float64
    P = len(candidates)
    n_select = min(cfg.n_select_max, P)

    # ── Unit-normalise columns ─────────────────────────────────────
    col_norms = np.linalg.norm(Xc, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    Xn = Xc / col_norms  # (N, P), unit columns

    # ── Determine starting band (local index) ─────────────────────
    local_match = np.where(candidates == init_global)[0]
    init_local = int(local_match[0]) if len(local_match) else int(np.argmax(Xc.var(axis=0)))

    selected = [init_local]
    remaining = list(range(P))
    remaining.remove(init_local)

    for _ in tqdm(range(n_select - 1), desc="       steps"):
        if not remaining:
            break

        pivot = Xn[:, selected[-1]]  # (N,) — current orthog. pivot
        norm_sq = float(pivot @ pivot)
        if norm_sq < 1e-12:
            break  # pivot collapsed; stop

        rem = np.array(remaining)
        Xn_rem = Xn[:, rem]  # (N, R) — copy via fancy idx

        # Gram-Schmidt step: project pivot out of all remaining columns
        # coefs[j] = (x_j · pivot) / ‖pivot‖²
        coefs = (Xn_rem.T @ pivot) / norm_sq  # (R,)
        # x_j ← x_j − coefs[j] * pivot  for all j simultaneously
        # np.outer(pivot, coefs) : (N,) × (R,) → (N, R)  where [i,j] = pivot[i]*coefs[j]
        Xn[:, rem] = Xn_rem - np.outer(pivot, coefs)  # write back projected columns

        # Select remaining band with largest residual norm
        residual_norms = np.linalg.norm(Xn[:, rem], axis=0)  # (R,)
        best = int(np.argmax(residual_norms))
        selected.append(remaining[best])
        remaining.pop(best)

    ordered = candidates[np.array(selected)]
    print(f"         First 10 bands (original idx): {ordered[:10].tolist()}")
    return ordered


# =====================================================================
# STEP 6 — StratifiedKFold validation curve
# =====================================================================
def validate(
    X: np.ndarray, y: np.ndarray, ordered: np.ndarray, label: str, cfg: BandSelectionConfig
) -> dict:
    """
    For each k in cfg.n_candidates, evaluate two classifiers using
    the first k bands from `ordered` (i.e. the top-k by selection priority).

    Classifiers
    -----------
    LDA (solver='svd'): uses pseudo-inverse; handles 90 classes correctly
      without matrix inversion failures. Appropriate when class-conditional
      distributions are roughly Gaussian (common for HSI mean spectra).

    LinearSVC (C=0.1): max-margin linear classifier; more robust than LDA
      when distributions deviate from Gaussian. Conservative C avoids
      overfitting on the relatively small mean-spectra feature space.

    Returns
    -------
    {k: {"lda": float, "svc": float}} — mean accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    counts = [k for k in cfg.n_candidates if k <= len(ordered)]
    result = {}

    print(f"\n       Validating [{label}]  ({cfg.cv_folds}-fold StratifiedKFold)...")

    for k in tqdm(counts, desc=f"       {label}"):
        Xs = X[:, ordered[:k]]

        # LDA
        try:
            lda_acc = float(
                cross_val_score(
                    make_pipeline(
                        StandardScaler(),
                        LinearDiscriminantAnalysis(solver="svd", tol=1e-4),
                    ),
                    Xs,
                    y,
                    cv=skf,
                    scoring="accuracy",
                    n_jobs=-1,
                ).mean()
            )
        except Exception:
            lda_acc = float("nan")

        # LinearSVC
        svc_acc = float(
            cross_val_score(
                make_pipeline(
                    StandardScaler(),
                    LinearSVC(C=cfg.svc_C, max_iter=3000, random_state=cfg.seed),
                ),
                Xs,
                y,
                cv=skf,
                scoring="accuracy",
                n_jobs=-1,
            ).mean()
        )

        result[k] = {"lda": round(lda_acc, 4), "svc": round(svc_acc, 4)}
        print(f"         k = {k:3d}   LDA = {lda_acc:.4f}   SVC = {svc_acc:.4f}")

    return result


# =====================================================================
# ELBOW DETECTION
# =====================================================================
def find_elbow(counts: list, accs: list, cfg: BandSelectionConfig) -> int:
    """
    Returns the smallest band count k that achieves at least
    cfg.elbow_pct × peak_accuracy.

    This avoids over-selecting bands for diminishing returns. The 98%
    threshold is a standard heuristic in HSI band selection literature.
    """
    peak = max(accs)
    threshold = cfg.elbow_pct * peak
    for k, a in zip(counts, accs):  # noqa: B905
        if a >= threshold:
            return k
    return counts[int(np.argmax(accs))]  # fallback: peak


# =====================================================================
# STEP 7 — Save reduced dataset
# =====================================================================
def save_outputs(
    final_bands: np.ndarray, wl_df: pd.DataFrame, tag: str, cfg: BandSelectionConfig
) -> None:
    """
    Writes the reduced patch array and the selected wavelength CSV.
    Reads the full patches via memory-map and writes only the optimal
    band subset in chunks to avoid peak memory usage.
    """
    patches = np.load(cfg.patches_path, mmap_mode="r")
    N, _, H, W = patches.shape
    n = len(final_bands)
    out = Path(cfg.output_dir)

    patch_path = out / f"patches_{tag}_{n}b.npy"
    wl_path = out / f"wavelengths_{tag}_{n}b.csv"

    print(f"\n[6/6]  Saving  {patch_path.name}  shape = ({N}, {n}, {H}, {W})...")
    reduced = np.zeros((N, n, H, W), dtype=np.float32)
    for s in tqdm(range(0, N, cfg.chunk_size), desc="       writing"):
        e = min(s + cfg.chunk_size, N)
        reduced[s:e] = patches[s:e, final_bands, :, :]
    np.save(patch_path, reduced)

    wl_df.iloc[final_bands].reset_index(drop=True).to_csv(wl_path, index=False)
    print(f"         Patches     → {patch_path}")
    print(f"         Wavelengths → {wl_path}")


# =====================================================================
# MAIN
# =====================================================================
def select_bands(cfg: BandSelectionConfig | None = None) -> None:
    cfg = cfg or BandSelectionConfig()

    # Baseline lines 58-60 and 92 ran these at import time; scoping them to the
    # entrypoint keeps `import spectralquadnet.data.prep.band_selection` inert.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.random.seed(cfg.seed)

    bar = "═" * 62
    print(f"\n{bar}")
    print("  HSI Band Selection  ·  mRMR + SPA + Validated Elbow")
    print(bar)

    # ── Load ──────────────────────────────────────────────────────
    wl_df = pd.read_csv(cfg.wavelength_path)
    X, y = extract_mean_spectra(cfg)

    # ── Pre-filter ────────────────────────────────────────────────
    candidates = decorrelation_prefilter(X, cfg)

    # ── Diagnostic ────────────────────────────────────────────────
    _fdr = fisher_discriminant_ratio(X, y, wl_df)

    # ── mRMR ──────────────────────────────────────────────────────
    # run_mrmr also returns the per-candidate MI relevance so we can
    # seed SPA from the same highest-relevance band without recomputing MI.
    mrmr_order, rel_local = run_mrmr(X, y, candidates, cfg)

    # ── SPA — seeded from highest-MI band ────────────────────────
    best_mi_global = candidates[int(np.argmax(rel_local))]
    spa_order = run_spa(X, candidates, init_global=best_mi_global, cfg=cfg)

    # ── Validation ────────────────────────────────────────────────
    print("\n  [5b/6]  Cross-validated accuracy curves...")
    mrmr_val = validate(X, y, mrmr_order, "mRMR", cfg)
    spa_val = validate(X, y, spa_order, "SPA", cfg)

    # ── Compare methods ───────────────────────────────────────────
    shared = [k for k in cfg.n_candidates if k in mrmr_val and k in spa_val]
    mrmr_accs = [mrmr_val[k]["svc"] for k in shared]
    spa_accs = [spa_val[k]["svc"] for k in shared]

    mrmr_peak = max(mrmr_accs)
    spa_peak = max(spa_accs)
    print(f"\n  Peak SVC → mRMR : {mrmr_peak:.4f}   SPA : {spa_peak:.4f}")

    if mrmr_peak >= spa_peak:
        winner, w_order, w_val, w_accs = "mrmr", mrmr_order, mrmr_val, mrmr_accs
    else:
        winner, w_order, w_val, w_accs = "spa", spa_order, spa_val, spa_accs

    optimal_k = find_elbow(shared, w_accs, cfg)
    final_bands = np.sort(w_order[:optimal_k])

    # ── Save comparison report ────────────────────────────────────
    rows = [
        {
            "n_bands": k,
            "mrmr_lda": mrmr_val[k]["lda"],
            "mrmr_svc": mrmr_val[k]["svc"],
            "spa_lda": spa_val[k]["lda"],
            "spa_svc": spa_val[k]["svc"],
        }
        for k in shared
    ]
    report = pd.DataFrame(rows)
    rpath = Path(cfg.output_dir) / "band_selection_report.csv"
    report.to_csv(rpath, index=False)
    print(f"\n  Accuracy table:\n{report.to_string(index=False)}")
    print(f"\n  Report → {rpath}")

    # ── Save reduced dataset ──────────────────────────────────────
    save_outputs(final_bands, wl_df, winner, cfg)

    # ── Final summary ─────────────────────────────────────────────
    sel_wl = wl_df.iloc[final_bands]["Wavelength (nm)"].values
    print(f"\n{bar}")
    print("  FINAL SELECTION SUMMARY")
    print(bar)
    print(f"  Winner method   : {winner.upper()}")
    print(f"  Bands selected  : {optimal_k} of 256  ({(1 - optimal_k/256)*100:.1f}% reduction)")
    print(f"  Wavelength range: {sel_wl.min():.1f} – {sel_wl.max():.1f} nm")
    print(f"  SVC accuracy    : {w_val[optimal_k]['svc']*100:.2f}%  (5-fold, mean spectra)")
    print(f"  LDA accuracy    : {w_val[optimal_k]['lda']*100:.2f}%  (5-fold, mean spectra)")
    print(f"  Band indices    : {sorted(final_bands.tolist())}")
    print(f"  Wavelengths (nm): {[round(float(v), 1) for v in sorted(sel_wl.tolist())]}")
    print(bar)
    print()
    print("  Next step: retrain your CNN using")
    print(f"  ./dataset/patches_{winner}_{optimal_k}b.npy  and compare")
    print("  against the 256-band baseline (86.9% TTA).")
    print(bar)
