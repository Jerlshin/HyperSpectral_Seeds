"""Research-correct hyperspectral band selection (mRMR + SPA + validated elbow).

Importing this module has no side effects: RNG seeding and warning-filter
configuration both happen inside :func:`select_bands`, not at module scope.
The CLI entry point is ``scripts/select_bands.py``.

This pipeline produces reduced-band patch arrays (e.g. a 40-band SPA subset)
that a training config can point ``cfg.data.patches_data`` at.

This module is a **build step**: it materialises one cube at one band count,
taking the method pair and the elbow rule below as given. The experiment that
decides *whether* those are the right choices — twelve methods including two
nulls, budgets to the full 256, selection stability, redundancy and a
recommendation — is :mod:`spectralquadnet.bandstudy`
(``python -m spectralquadnet.bandstudy.cli list``, ``docs/07_BAND_STUDY.md``).
Run that first if k is meant to be an experimental result rather than an
inherited one; run this when k is already settled and a materialised cube is
wanted.

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

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
def resolve_train_indices(cfg: BandSelectionConfig) -> npt.NDArray[np.int64] | None:
    """The rows band selection is allowed to see (IC-4).

    ``None`` when ``cfg.fold`` is ``None``, which reproduces the whole-corpus
    selection — leaky, and retained deliberately as A2's control arm.

    When a fold is named, the training rows come from the *same* split builder
    the training run uses, driven by the same parameters. Approximating the
    partition here would leave the two selections describing different data and
    make A2 uninterpretable.

    Note:
        ``calib`` is excluded along with ``val`` and ``test``. Calib is a
        held-out split — it carries fitted parameters — so a band chosen with
        its labels in scope is chosen with information the training gradient
        never had.

    Raises:
        FileNotFoundError: A fold was requested but ``groups_path`` is absent.
    """
    if cfg.fold is None:
        return None

    labels = np.load(cfg.labels_path)
    if cfg.split_scheme == "stratified":
        from spectralquadnet.data.loaders import _stratified_split

        bundle = _stratified_split(labels, cfg.split_eval_frac, cfg.calib_frac, None)
        return np.asarray(bundle.train, dtype=np.int64)

    groups_path = Path(cfg.groups_path)
    if not groups_path.exists():
        raise FileNotFoundError(
            f"band selection at fold {cfg.fold} needs the scan ids at {groups_path}, which "
            "does not exist. It is written by `python scripts/prepare_dataset.py`. Without "
            "it there is no group-disjoint training set to restrict the selection to, and "
            "the selection would silently fall back to the leaky whole-corpus one."
        )
    from spectralquadnet.data.loaders import grouped_split

    bundle = grouped_split(
        labels,
        np.load(groups_path),
        eval_frac=cfg.split_eval_frac,
        calib_frac=cfg.calib_frac,
        fold=int(cfg.fold),
        seed=cfg.seed,
        single_group_policy="patch_split",
    )
    return np.asarray(bundle.train, dtype=np.int64)


def extract_mean_spectra(
    cfg: BandSelectionConfig,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """
    Load patches via memory-map and compute the per-patch spatially
    averaged spectrum over foreground (non-background) pixels only.

    Background detection: pixels whose band-sum absolute value < 1e-5
    are treated as padded zeros from the segmentation pipeline.

    Every row is computed — the reduced cube written at the end has to cover
    the whole dataset, since the model scores eval patches too. Restricting the
    *selection* to training rows is :func:`resolve_train_indices`'s job and
    happens on the returned arrays, not here.

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
def decorrelation_prefilter(X: npt.NDArray[Any], cfg: BandSelectionConfig) -> npt.NDArray[Any]:
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
    X: npt.NDArray[Any], y: npt.NDArray[Any], wl_df: pd.DataFrame
) -> npt.NDArray[Any]:
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
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    candidates: npt.NDArray[Any],
    cfg: BandSelectionConfig,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
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
    X: npt.NDArray[Any],
    candidates: npt.NDArray[Any],
    init_global: int,
    cfg: BandSelectionConfig,
) -> npt.NDArray[Any]:
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
    return ordered  # type: ignore[no-any-return]  # fancy-index of an `Any`-typed array


# =====================================================================
# STEP 6 — StratifiedKFold validation curve
# =====================================================================
def validate(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    ordered: npt.NDArray[Any],
    label: str,
    cfg: BandSelectionConfig,
) -> dict[int, dict[str, float]]:
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
    result: dict[int, dict[str, float]] = {}

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
def find_elbow(counts: list[int], accs: list[float], cfg: BandSelectionConfig) -> int:
    """
    Returns the smallest band count k that achieves at least
    cfg.elbow_pct × peak_accuracy.

    This avoids over-selecting bands for diminishing returns. The 98%
    threshold is a standard heuristic in HSI band selection literature.

    The result is only meaningful when ``counts`` extends **past** the k it
    returns: ``peak`` is taken over the counts supplied, so a curve truncated
    at its own elbow satisfies the criterion trivially (M-14). Pair every call
    with :func:`verify_elbow`, which measures that.
    """
    peak = max(accs)
    threshold = cfg.elbow_pct * peak
    for k, a in zip(counts, accs):  # noqa: B905
        if a >= threshold:
            return k
    return counts[int(np.argmax(accs))]  # fallback: peak


@dataclass(frozen=True)
class ElbowVerdict:
    """Whether a chosen band count is an elbow or merely the end of the curve.

    T4-6's validation criterion — "the recorded curve extends past the chosen
    k; the elbow is demonstrable, not asserted" — is exactly
    :attr:`demonstrable`, and every field below is one of the measurements it
    is made of.
    """

    k: int
    max_k_recorded: int
    peak_k: int
    peak_acc: float
    acc_at_k: float
    n_points_past_k: int
    #: Best accuracy anywhere past ``k``, minus the accuracy at ``k``. Positive
    #: means more bands still help — which is F-3's prediction.
    headroom_past_k: float
    demonstrable: bool
    reason: str

    def lines(self) -> list[str]:
        """Report lines, for the console and the run log."""
        verdict = "DEMONSTRABLE" if self.demonstrable else "NOT DEMONSTRABLE"
        return [
            f"  Elbow at k = {self.k}: {verdict} — {self.reason}",
            f"    curve recorded to k = {self.max_k_recorded} "
            f"({self.n_points_past_k} points past k)",
            f"    acc(k) = {self.acc_at_k:.4f}   peak = {self.peak_acc:.4f} at k = {self.peak_k}"
            f"   headroom past k = {self.headroom_past_k:+.4f}",
        ]


def verify_elbow(
    counts: list[int], accs: list[float], k: int, cfg: BandSelectionConfig
) -> ElbowVerdict:
    """Measure whether ``k`` is an elbow of ``(counts, accs)`` or its endpoint.

    An elbow claim has two parts and the shipped run could only support the
    first: that accuracy at ``k`` is within ``elbow_pct`` of the peak, and that
    the curve *goes past* ``k`` so the peak is not simply ``acc(k)`` itself.
    ``dataset/band_selection_report.csv`` terminates at k = 40, the value it
    selected, so its own peak is at 40 by construction (M-14, §2.2.11).

    Args:
        counts: Band counts evaluated, ascending.
        accs: Accuracy at each count.
        k: The chosen band count.
        cfg: Supplies ``elbow_pct``.

    Returns:
        The verdict. :attr:`ElbowVerdict.demonstrable` is the criterion T4-6
        states; it is False whenever the curve stops at ``k``, whatever the
        accuracies say.
    """
    pairs = sorted(zip(counts, accs, strict=True))
    ks = [c for c, _ in pairs]
    values = [a for _, a in pairs]
    peak_i = int(np.argmax(values))
    acc_at_k = values[ks.index(k)] if k in ks else float("nan")
    past = [a for c, a in pairs if c > k]
    headroom = (max(past) - acc_at_k) if past else 0.0

    if not past:
        reason = (
            f"the curve terminates at the chosen k — peak is acc({k}) by construction, "
            "so the 98 % criterion is satisfied vacuously (M-14)"
        )
        demonstrable = False
    elif acc_at_k < cfg.elbow_pct * values[peak_i]:
        reason = (
            f"acc({k}) = {acc_at_k:.4f} is below {cfg.elbow_pct:.0%} of the peak "
            f"{values[peak_i]:.4f} at k = {ks[peak_i]}"
        )
        demonstrable = False
    else:
        reason = (
            f"acc({k}) is within {cfg.elbow_pct:.0%} of the peak and "
            f"{len(past)} larger band counts were evaluated"
        )
        demonstrable = True

    return ElbowVerdict(
        k=int(k),
        max_k_recorded=int(ks[-1]),
        peak_k=int(ks[peak_i]),
        peak_acc=float(values[peak_i]),
        acc_at_k=float(acc_at_k),
        n_points_past_k=len(past),
        headroom_past_k=float(headroom),
        demonstrable=demonstrable,
        reason=reason,
    )


def load_deployed_curve(path: str | Path) -> dict[int, float]:
    """Read the deployed estimator's accuracy curve — T4-6 / F-3.

    The proxy classifiers this module cross-validates (LDA and ``LinearSVC``
    on 256-dimensional mean spectra) are not the estimator that gets deployed,
    and F-3 is the prediction that the two curves disagree: a linear model on
    class-mean spectra saturates long before a four-branch network that reads
    spatial structure and band interactions does. When a curve from the real
    estimator is available it replaces theirs for the winner and elbow
    decisions.

    Args:
        path: CSV with an ``n_bands`` column and an ``accuracy`` (or ``acc``,
            or ``macro_f1``) column.

    Returns:
        ``{k: accuracy}``.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The expected columns are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"deployed_curve_path={p} does not exist")
    frame = pd.read_csv(p)
    score_col = next((c for c in ("accuracy", "acc", "macro_f1", "f1") if c in frame.columns), None)
    if "n_bands" not in frame.columns or score_col is None:
        raise ValueError(
            f"{p} must have an 'n_bands' column and one of accuracy/acc/macro_f1/f1; "
            f"found {list(frame.columns)}"
        )
    return {int(r.n_bands): float(getattr(r, score_col)) for r in frame.itertuples()}


# =====================================================================
# STEP 7 — Save reduced dataset
# =====================================================================
def output_paths(cfg: BandSelectionConfig, tag: str, n_bands: int) -> tuple[Path, Path]:
    """Where this selection's cube and wavelength CSV go.

    Whole-corpus selections keep the historical flat names
    (``patches_spa_40b.npy``), so existing configs are unaffected. A per-fold
    selection goes to ``<output_dir>/<fold_subdir>/patches_fold<k>_<n>b.npy`` —
    a *different filename*, because the two arrays are not interchangeable and
    a run that silently picked up the wrong one would be exactly the leak IC-4
    exists to close, wearing the name of the fix.
    """
    out = Path(cfg.output_dir)
    if cfg.fold is None:
        return out / f"patches_{tag}_{n_bands}b.npy", out / f"wavelengths_{tag}_{n_bands}b.csv"
    fold_dir = out / cfg.fold_subdir
    fold_dir.mkdir(parents=True, exist_ok=True)
    return (
        fold_dir / f"patches_fold{cfg.fold}_{n_bands}b.npy",
        fold_dir / f"wavelengths_fold{cfg.fold}_{n_bands}b.csv",
    )


def save_outputs(
    final_bands: npt.NDArray[Any], wl_df: pd.DataFrame, tag: str, cfg: BandSelectionConfig
) -> None:
    """
    Writes the reduced patch array and the selected wavelength CSV.
    Reads the full patches via memory-map and writes only the optimal
    band subset in chunks to avoid peak memory usage.

    **Every row is written**, not only the training ones: the bands were chosen
    from training patches (IC-4), but the model still has to score val and test,
    and slicing the array down to the training split would make that impossible.
    Restricting *selection* and restricting *data* are different things and only
    the first is the leak.
    """
    patches = np.load(cfg.patches_path, mmap_mode="r")
    N, _, H, W = patches.shape
    n = len(final_bands)

    patch_path, wl_path = output_paths(cfg, tag, n)

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
    """Run the full band-selection pipeline: extract, pre-filter, select, validate, save.

    Loads mean spectra, decorrelation-prefilters the candidate pool, runs
    both mRMR and SPA selection, cross-validates each at several band
    counts, picks the winning method and its elbow-point band count, and
    writes the reduced patch array plus a comparison report.

    Args:
        cfg: Band-selection configuration; a default :class:`BandSelectionConfig`
            is used if omitted.
    """
    cfg = cfg or BandSelectionConfig()

    # Scoped to the entrypoint (not module scope) so importing this module
    # has no side effects on global warning filters or the NumPy RNG.
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
    X_all, y_all = extract_mean_spectra(cfg)

    # ── IC-4: restrict the SELECTION to training rows ─────────────
    # Everything below this line — the correlation matrix, the FDR diagnostic,
    # the mutual information, SPA's Gram-Schmidt residuals and the CV curve —
    # sees `X`/`y` and therefore sees training patches only. The reduced cube
    # written at the end still covers every row; what changed is whose labels
    # were allowed to choose the bands.
    train_idx = resolve_train_indices(cfg)
    if train_idx is None:
        X, y = X_all, y_all
        scope = "ALL 8,624 patches — including every eval patch (LEAKY, A2 control arm)"
    else:
        X, y = X_all[train_idx], y_all[train_idx]
        scope = (
            f"{len(train_idx):,} TRAINING patches of fold {cfg.fold} "
            f"({cfg.split_scheme}); val/test/calib labels are out of scope"
        )
    print(f"\n  Selection scope: {scope}")

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

    # ── T4-6 / F-3 — the deployed estimator overrides the proxies ──
    deployed: dict[int, float] = {}
    decision_counts, decision_accs = shared, w_accs
    estimator = f"{winner.upper()} ranking, LinearSVC on mean spectra"
    if cfg.deployed_curve_path:
        deployed = load_deployed_curve(cfg.deployed_curve_path)
        decision_counts = sorted(deployed)
        decision_accs = [deployed[k] for k in decision_counts]
        estimator = f"deployed estimator ({cfg.deployed_curve_path})"
        print(f"\n  Elbow decided on the {estimator}, not the proxy classifiers.")

    optimal_k = find_elbow(decision_counts, decision_accs, cfg)
    verdict = verify_elbow(decision_counts, decision_accs, optimal_k, cfg)
    final_bands = np.sort(w_order[:optimal_k])

    # ── Save comparison report ────────────────────────────────────
    rows = [
        {
            "n_bands": k,
            "mrmr_lda": mrmr_val[k]["lda"],
            "mrmr_svc": mrmr_val[k]["svc"],
            "spa_lda": spa_val[k]["lda"],
            "spa_svc": spa_val[k]["svc"],
            "deployed": deployed.get(k, float("nan")),
        }
        for k in shared
    ]
    report = pd.DataFrame(rows)
    # Per-fold selections write their own report next to their own cube; a
    # single shared filename would let fold 1 silently overwrite fold 0's
    # evidence, and the two curves are the artifact A2 compares.
    suffix = "" if cfg.fold is None else f"_fold{cfg.fold}"
    report_dir = (
        Path(cfg.output_dir) if cfg.fold is None else Path(cfg.output_dir) / cfg.fold_subdir
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    rpath = report_dir / f"band_selection_report{suffix}.csv"
    report.to_csv(rpath, index=False)
    print(f"\n  Accuracy table:\n{report.to_string(index=False)}")
    print(f"\n  Report → {rpath}")

    # ── T4-6 — is the elbow demonstrable? ─────────────────────────
    # Written next to the curve, not only printed: M-14 is a defect of an
    # *artifact* — a report that stops at its own chosen k — so the artifact
    # has to carry the answer.
    vpath = report_dir / f"band_selection_elbow{suffix}.json"
    vpath.write_text(
        json.dumps(
            {
                "estimator": estimator,
                # IC-4: recorded in the artifact, because "were the eval labels
                # in scope when these bands were chosen?" is the first question
                # a reviewer asks and it must be answerable from the file.
                "selection_scope": scope,
                "fold": cfg.fold,
                "n_selection_rows": int(len(X)),
                **verdict.__dict__,
            },
            indent=2,
        )
    )
    print("\n".join(verdict.lines()))
    print(f"  Verdict → {vpath}")
    if not verdict.demonstrable:
        print(
            "  ⚠ The elbow is NOT demonstrable from this curve. §4.5: publish the curve to "
            "k = 256 under the deployed estimator, or withdraw the elbow claim."
        )

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
    # `optimal_k` comes from the decision curve, which under T4-6 may be the
    # deployed estimator's and need not share the proxies' band counts.
    if optimal_k in w_val:
        print(f"  SVC accuracy    : {w_val[optimal_k]['svc']*100:.2f}%  (5-fold, mean spectra)")
        print(f"  LDA accuracy    : {w_val[optimal_k]['lda']*100:.2f}%  (5-fold, mean spectra)")
    print(
        f"  Elbow           : {'demonstrable' if verdict.demonstrable else 'NOT demonstrable'}"
        f"  (curve to k = {verdict.max_k_recorded}, {verdict.n_points_past_k} points past k,"
        f" headroom {verdict.headroom_past_k:+.4f})"
    )
    print(f"  Decided on      : {estimator}")
    print(f"  Band indices    : {sorted(final_bands.tolist())}")
    print(f"  Wavelengths (nm): {[round(float(v), 1) for v in sorted(sel_wl.tolist())]}")
    print(bar)
    print()
    print("  Next step: retrain your CNN using")
    print(f"  ./dataset/patches_{winner}_{optimal_k}b.npy  and compare")
    print("  against the 256-band baseline (86.9% TTA).")
    print(bar)
