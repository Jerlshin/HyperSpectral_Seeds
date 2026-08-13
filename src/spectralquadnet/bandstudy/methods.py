"""Band-selection methods, as a registry of comparable strategies.

Every method here takes the same input — a training-rows-only ``(n, C)`` mean
spectrum matrix and its labels — and returns band indices for each requested
budget. Nothing else differs, which is the property that makes the comparison a
comparison.

The families, and why each is present
─────────────────────────────────────
========================  ===========================================================
``uniform``               **Null.** Evenly spaced across the spectrum, ignoring the
                          data entirely. On a smooth 256-band VIS-NIR cube this is a
                          genuinely strong baseline and the cheapest possible method.
``random``                **Null.** Uniform random subsets, several draws per budget.
                          Its spread is the null distribution every other method is
                          scored against — with neighbour correlations above 0.99, a
                          method that does not beat a random draw of the same size
                          has not been shown to select anything.
``variance``              Unsupervised univariate: the loudest bands. Present because
                          it is what an unlabelled pipeline would do, and because a
                          supervised method that fails to beat it is not using labels.
``fdr``                   Supervised univariate: multiclass Fisher discriminant
                          ratio, between-class over within-class scatter. No
                          redundancy control at all — deliberately, so the cost of
                          omitting it is measurable.
``mi``                    Supervised univariate: k-NN mutual information with the
                          class label, no Gaussian assumption (Kraskov et al. 2004).
``mrmr``                  Supervised, redundancy-aware: the MID criterion, relevance
                          minus mean redundancy against the already-selected set
                          (Peng, Long & Ding, IEEE TPAMI 27(8):1226-38, 2005). The
                          repository's incumbent.
``spa``                   Geometric: successive Gram-Schmidt projections pick maximally
                          orthogonal bands (Araújo et al., Chemom Intell Lab Syst
                          57(2):65-73, 2001). The repository's other incumbent, and
                          the method that produced the shipped 40-band subset.
``cluster_ward``          Supervised, redundancy-aware by construction: Ward-cluster
                          the bands into exactly k groups, take each group's most
                          discriminative member. Guarantees spectral coverage in a way
                          greedy forward selection does not.
``pca_loading``           Unsupervised, decorrelating: take the band with the largest
                          absolute loading on PC1, then PC2, and so on. Immune to
                          label leakage by construction, which makes it the control
                          for "how much does supervision buy here?".
``l1_path``               Embedded, multivariate: order bands by how early they enter
                          an L1-penalised linear model as the penalty relaxes — a
                          regularisation-path ranking rather than one fit's
                          coefficients, so bands zeroed at a single C are still
                          ordered.
``tree_importance``       Embedded, nonlinear, multivariate: ExtraTrees impurity
                          importance. The only method here that can express "this band
                          matters *given* that one" without a linear model.
``pls_vip``               Chemometric: PLS-DA variable-importance-in-projection. The
                          standard in NIR spectroscopy and the method a reviewer from
                          that field will ask why you omitted.
========================  ===========================================================

Nested and non-nested
─────────────────────
Most methods produce a **ranking**, so every budget is a prefix of one ordering
and k does not multiply their cost. ``uniform``, ``cluster_ward`` and ``random``
are **per-budget**: their k = 20 set is not a subset of their k = 40 set. That
distinction is recorded on the spec and reported, because a nested method's
budget curve is a curve through one nested family and a non-nested method's is
not, and comparing their stability without saying so would be comparing two
different quantities.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

_log = logging.getLogger("spectralquadnet.bandstudy.methods")


# ══════════════════════════════════════════════════════════════════════
#  Specs
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MethodSpec:
    """What a method is, for the tables that compare methods rather than scores."""

    name: str
    #: ``ranking`` — one ordering, every budget a prefix. ``per_budget`` — a
    #: fresh set per k.
    kind: str
    supervised: bool
    #: ``null_model`` | ``univariate`` | ``redundancy`` | ``geometric`` |
    #: ``embedded`` | ``chemometric``. Used to group the method-comparison
    #: table, because three univariate filters agreeing is one piece of
    #: evidence, not three.
    #:
    #: Spelled ``null_model`` rather than ``null`` because these values are
    #: round-tripped through CSV, and ``pandas.read_csv`` reads the bare token
    #: ``null`` back as ``NaN`` — which turned the nulls' family into a missing
    #: value in every generated table.
    family: str
    reference: str
    note: str
    #: Number of independent draws per budget. Only ``random`` exceeds 1.
    draws: int = 1

    @property
    def is_null(self) -> bool:
        return self.family == "null_model"


METHODS: dict[str, MethodSpec] = {}
_IMPLS: dict[str, Callable[..., Any]] = {}


def _register(spec: MethodSpec, impl: Callable[..., Any]) -> None:
    METHODS[spec.name] = spec
    _IMPLS[spec.name] = impl


def get(name: str) -> MethodSpec:
    """Look up a method spec.

    Raises:
        KeyError: With the available names, because a typo costs a stage.
    """
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(
            f"Unknown band-selection method {name!r}. Available: {', '.join(sorted(METHODS))}"
        ) from None


# ══════════════════════════════════════════════════════════════════════
#  Shared per-replicate quantities
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SelectionContext:
    """One replicate's training matrix, plus the statistics several methods share.

    The mutual-information estimate is the expensive object here — a k-NN
    estimator over 256 features — and three methods need it (``mi`` for its
    ranking, ``mrmr`` for its relevance term, ``spa`` for its seed). Computing
    it once per replicate rather than once per method is most of the difference
    between this study taking hours and taking a day.

    Everything on this object is derived from ``x``/``y``, which are the
    replicate's **training rows only**. There is no path from here to calib or
    to the held-out split, which is the point.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    seed: int
    corr_threshold: float = 0.995
    mi_neighbors: int = 5

    _mi: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    _fdr: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    _corr: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    _z: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_bands(self) -> int:
        return int(self.x.shape[1])

    def z(self) -> npt.NDArray[np.float64]:
        """Column-standardised spectra.

        The VIS-NIR range spans more than an order of magnitude in raw
        reflectance, so any distance- or variance-based method run on the raw
        matrix is ranking the brightest region of the spectrum rather than the
        most informative one.
        """
        if self._z is None:
            mu = self.x.mean(axis=0, keepdims=True)
            sd = self.x.std(axis=0, keepdims=True)
            sd[sd < 1e-12] = 1.0
            self._z = (self.x - mu) / sd
        return self._z

    def corr(self) -> npt.NDArray[np.float64]:
        """``(C, C)`` Pearson correlation between bands."""
        if self._corr is None:
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = np.corrcoef(self.x.T)
            self._corr = np.nan_to_num(np.atleast_2d(c), nan=0.0)
            self.timings["corr"] = time.perf_counter() - started
        return self._corr

    def mi(self) -> npt.NDArray[np.float64]:
        """``MI(band; class)`` per band, k-NN estimated."""
        if self._mi is None:
            from sklearn.feature_selection import mutual_info_classif

            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._mi = np.asarray(
                    mutual_info_classif(
                        self.x,
                        self.y,
                        discrete_features=False,
                        n_neighbors=self.mi_neighbors,
                        random_state=self.seed,
                    ),
                    dtype=np.float64,
                )
            self.timings["mi"] = time.perf_counter() - started
        return self._mi

    def fdr(self) -> npt.NDArray[np.float64]:
        """Multiclass Fisher discriminant ratio per band.

        ``sum_c n_c (mu_ck - mu_k)^2  /  sum_c n_c sigma^2_ck`` — between-class
        scatter over within-class scatter, the same quantity the shipped
        selector computed as a diagnostic and never used.
        """
        if self._fdr is None:
            started = time.perf_counter()
            mu_global = self.x.mean(axis=0)
            between = np.zeros(self.n_bands)
            within = np.zeros(self.n_bands)
            for c in np.unique(self.y):
                xc = self.x[self.y == c]
                between += len(xc) * (xc.mean(axis=0) - mu_global) ** 2
                within += len(xc) * xc.var(axis=0)
            self._fdr = between / (within + 1e-12)
            self.timings["fdr"] = time.perf_counter() - started
        return self._fdr

    def candidates(self) -> npt.NDArray[np.int64]:
        """Bands surviving the greedy decorrelation pre-filter.

        Scans low to high wavelength and keeps a band only when its ``|r|``
        against every already-kept band is at or below the threshold. Used by
        ``mrmr`` and ``spa``, which are the two methods the repository already
        ran this way — keeping it means the reimplemented arms are the same
        arms, not similar ones.
        """
        corr = self.corr()
        keep: list[int] = []
        for i in range(self.n_bands):
            if not any(abs(corr[i, j]) > self.corr_threshold for j in keep):
                keep.append(i)
        return np.asarray(keep, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════
#  Outcome
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SelectionOutcome:
    """What one (method, replicate) produced.

    Attributes:
        per_budget: ``{k: [band_set, ...]}``. A list because ``random`` draws
            several sets per budget; every other method contributes exactly one.
        ranking: The full ordering for a nested method, ``None`` otherwise.
            Recorded because "what would this method have picked at k = 37?" is
            answerable from a ranking and not from a set.
        failure: A human-readable reason when the method could not run. The
            cell is recorded as failed rather than dropped — a hole in a table
            that nobody can see is worse than one everybody can.
    """

    method: str
    per_budget: dict[int, list[list[int]]] = field(default_factory=dict)
    ranking: list[int] | None = None
    seconds: float = 0.0
    failure: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failure is None and bool(self.per_budget)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "ranking": self.ranking,
            "per_budget": {str(k): v for k, v in sorted(self.per_budget.items())},
            "seconds": round(self.seconds, 3),
            "failure": self.failure,
            "diagnostics": self.diagnostics,
        }


# ══════════════════════════════════════════════════════════════════════
#  Ranking helpers
# ══════════════════════════════════════════════════════════════════════


def _rank_desc(scores: npt.NDArray[Any]) -> list[int]:
    """Band indices ordered by descending score, ties broken by index.

    Deterministic tie-breaking matters more than it looks: several methods
    produce exact ties (a zeroed L1 coefficient, an unvisited tree feature),
    and an ``argsort`` whose tie order depends on the BLAS build would make the
    stability analysis measure the BLAS build.
    """
    s = np.asarray(scores, dtype=np.float64)
    return [int(i) for i in np.lexsort((np.arange(len(s)), -s))]


def _complete(ranking: list[int], n_bands: int, fallback: npt.NDArray[Any]) -> list[int]:
    """Extend a partial ranking to cover every band.

    ``mrmr`` and ``spa`` rank only the decorrelation pre-filter's survivors, so
    their ordering stops short of C and every budget past that point would be
    unreachable. The bands the pre-filter dropped are appended in descending
    ``fallback`` order — they are near-duplicates of bands already ranked, so
    where exactly they land barely matters, but *being unable to evaluate
    k = 256 at all* would make the method's curve incomparable to the others'.
    """
    seen = set(ranking)
    rest = [b for b in _rank_desc(fallback) if b not in seen]
    return ranking + rest


def _sets_from_ranking(ranking: list[int], budgets: list[int]) -> dict[int, list[list[int]]]:
    return {k: [sorted(ranking[:k])] for k in budgets if k <= len(ranking)}


# ══════════════════════════════════════════════════════════════════════
#  The methods
# ══════════════════════════════════════════════════════════════════════


def _uniform(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    out = SelectionOutcome(method="uniform")
    c = ctx.n_bands
    for k in budgets:
        if k > c:
            continue
        # `linspace` over positions rather than over wavelength: the cube's
        # bands are already close to λ-uniform (2.4 nm steps), and spacing by
        # index needs no wavelength file, so this null stays runnable on any
        # cube.
        idx = np.unique(np.round(np.linspace(0, c - 1, k)).astype(int))
        # Rounding can collide at large k; fill from the unused bands nearest
        # the collisions so the set is exactly k.
        if len(idx) < k:
            spare = [b for b in range(c) if b not in set(idx.tolist())]
            idx = np.sort(np.concatenate([idx, np.asarray(spare[: k - len(idx)], dtype=int)]))
        out.per_budget[k] = [sorted(int(b) for b in idx[:k])]
    return out


def _random(
    ctx: SelectionContext, budgets: list[int], draws: int = 10, **_: Any
) -> SelectionOutcome:
    out = SelectionOutcome(method="random")
    rng = np.random.default_rng(ctx.seed)
    c = ctx.n_bands
    for k in budgets:
        if k > c:
            continue
        # At k = C every draw is the same set; drawing ten of them would put
        # ten identical rows into the null distribution and shrink its apparent
        # spread to zero at exactly the budget the study compares against.
        n_draws = 1 if k == c else draws
        out.per_budget[k] = [
            sorted(int(b) for b in rng.choice(c, size=k, replace=False)) for _ in range(n_draws)
        ]
    return out


def _variance(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    ranking = _rank_desc(ctx.x.var(axis=0))
    return SelectionOutcome(
        method="variance", ranking=ranking, per_budget=_sets_from_ranking(ranking, budgets)
    )


def _fdr(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    scores = ctx.fdr()
    ranking = _rank_desc(scores)
    return SelectionOutcome(
        method="fdr",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={"max_fdr": float(scores.max()), "min_fdr": float(scores.min())},
    )


def _mi(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    scores = ctx.mi()
    ranking = _rank_desc(scores)
    return SelectionOutcome(
        method="mi",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={"max_mi": float(scores.max()), "min_mi": float(scores.min())},
    )


def _mrmr(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    """Greedy MID: relevance minus mean redundancy against the selected set.

    Redundancy between two bands is taken as ``-0.5 log(1 - r^2)``, which is
    the exact mutual information of a bivariate Gaussian and a close
    approximation for spectrally smooth bands. The alternative — a full
    ``C × C`` k-NN MI matrix — costs 32,640 estimator calls per replicate for a
    quantity this study never reads on its own.
    """
    candidates = ctx.candidates()
    relevance_full = ctx.mi()
    rel = relevance_full[candidates]
    corr = ctx.corr()[np.ix_(candidates, candidates)]
    eps = 1e-10
    mi_bb = -0.5 * np.log(np.clip(1.0 - np.clip(corr, -1 + eps, 1 - eps) ** 2, eps, None))

    p = len(candidates)
    selected: list[int] = [int(np.argmax(rel))]
    remaining = [i for i in range(p) if i != selected[0]]
    while remaining:
        rem = np.asarray(remaining)
        chosen = np.asarray(selected, dtype=np.int64)
        redundancy = mi_bb[np.ix_(rem, chosen)].mean(axis=1)
        best = int(np.argmax(rel[rem] - redundancy))
        selected.append(remaining.pop(best))

    ranking = _complete([int(candidates[i]) for i in selected], ctx.n_bands, relevance_full)
    return SelectionOutcome(
        method="mrmr",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={"n_candidates": int(len(candidates))},
    )


def _spa(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    """Successive projections: pick maximally orthogonal bands by Gram-Schmidt.

    Seeded from the highest-MI band rather than the highest-variance one, so
    the geometric exploration is anchored on a direction already known to be
    discriminative — the same seeding rule the shipped selector used.
    """
    candidates = ctx.candidates()
    relevance = ctx.mi()
    xc = ctx.x[:, candidates].astype(np.float64, copy=True)
    norms = np.linalg.norm(xc, axis=0)
    norms[norms < 1e-12] = 1.0
    xn = xc / norms

    init_global = int(candidates[int(np.argmax(relevance[candidates]))])
    init_local = int(np.flatnonzero(candidates == init_global)[0])
    selected = [init_local]
    remaining = [i for i in range(len(candidates)) if i != init_local]

    # Gram-Schmidt on 256 bands whose neighbours correlate above 0.99 exhausts
    # the numerical rank of the matrix long before it exhausts the bands. Past
    # that point the residuals are float noise, the projection coefficients
    # divide by an underflowed pivot norm, and the "most orthogonal remaining
    # band" is whichever one rounded largest. Stopping there and letting
    # `_complete` order the rest by relevance is both the numerically sound
    # choice and the honest one: SPA has nothing left to say about bands that
    # already lie in the span of the selected set.
    exhausted_at: int | None = None
    with np.errstate(all="ignore"):
        while remaining:
            pivot = xn[:, selected[-1]]
            norm_sq = float(pivot @ pivot)
            if not np.isfinite(norm_sq) or norm_sq < 1e-12:
                exhausted_at = len(selected)
                break
            rem = np.asarray(remaining)
            block = xn[:, rem]
            coefs = np.nan_to_num((block.T @ pivot) / norm_sq, nan=0.0, posinf=0.0, neginf=0.0)
            xn[:, rem] = block - np.outer(pivot, coefs)
            residual = np.nan_to_num(np.linalg.norm(xn[:, rem], axis=0), nan=0.0)
            if float(residual.max()) < 1e-9:
                exhausted_at = len(selected)
                break
            best = int(np.argmax(residual))
            selected.append(remaining.pop(best))

    ranking = _complete([int(candidates[i]) for i in selected], ctx.n_bands, relevance)
    return SelectionOutcome(
        method="spa",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={
            "n_candidates": int(len(candidates)),
            "seed_band": init_global,
            # How far SPA got before the residuals collapsed. Budgets past this
            # are ordered by relevance, not by orthogonality, and a reader
            # comparing SPA's large-k arm to its small-k arm needs to know.
            "numerical_rank_exhausted_at": exhausted_at,
        },
    )


def _cluster_ward(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    """Ward-cluster the bands into k groups; keep each group's best member.

    Redundancy control by construction rather than by penalty: k clusters
    partition the spectrum, so the selected set covers it whether or not any
    greedy criterion would have got there. The representative is the
    highest-Fisher-ratio band in its cluster, so the method is supervised in
    *which* member it keeps and unsupervised in *how* it groups.

    Non-nested: the k = 20 partition is not a coarsening of the k = 40 one, so
    the two sets need not overlap at all. That is a property worth measuring,
    not a defect — it is why the stability table separates nested from
    non-nested methods.
    """
    from sklearn.cluster import AgglomerativeClustering

    out = SelectionOutcome(method="cluster_ward")
    relevance = ctx.fdr()
    points = ctx.z().T  # (C, n) — a band is a point in sample space
    for k in budgets:
        if k > ctx.n_bands:
            continue
        if k == ctx.n_bands:
            out.per_budget[k] = [list(range(ctx.n_bands))]
            continue
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(points)
        picked = [
            int(np.flatnonzero(labels == c)[int(np.argmax(relevance[labels == c]))])
            for c in range(k)
        ]
        out.per_budget[k] = [sorted(picked)]
    return out


def _pca_loading(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    """One band per principal component, largest absolute loading first.

    Wholly unsupervised, which is what makes it the control for supervision:
    it cannot leak a label because it never sees one, so the gap between it and
    the supervised methods is the value of the labels on this task, measured
    rather than assumed.
    """
    from sklearn.decomposition import PCA

    z = ctx.z()
    n_comp = int(min(z.shape[0], z.shape[1]))
    pca = PCA(n_components=n_comp, random_state=ctx.seed).fit(z)
    loadings = np.abs(np.asarray(pca.components_))  # (n_comp, C)

    ranking: list[int] = []
    taken: set[int] = set()
    for comp in range(n_comp):
        order = np.argsort(-loadings[comp])
        for band in order:
            if int(band) not in taken:
                ranking.append(int(band))
                taken.add(int(band))
                break
        if len(ranking) >= ctx.n_bands:
            break
    ranking = _complete(ranking, ctx.n_bands, ctx.x.var(axis=0))
    return SelectionOutcome(
        method="pca_loading",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={
            "explained_variance_ratio_top10": [
                round(float(v), 5) for v in pca.explained_variance_ratio_[:10]
            ]
        },
    )


#: Inverse regularisation strengths walked by :func:`_l1_path`, weakest penalty
#: last. Six points spanning three orders of magnitude: enough to order the
#: bands by entry time without turning the method into the study's bottleneck.
_L1_CS: tuple[float, ...] = (0.002, 0.01, 0.05, 0.2, 1.0, 5.0)


def _l1_path(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    """Order bands by how early they enter an L1 model as the penalty relaxes.

    A single L1 fit gives most bands an exactly-zero coefficient and therefore
    no usable order among them, which is why the naive "rank by |coef|"
    version of this method silently degenerates into "rank by index" past the
    first few dozen bands. Walking the path fixes that: a band's rank is the
    weakest penalty at which it is still selected, with coefficient magnitude
    at that point breaking ties.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    z = StandardScaler().fit_transform(ctx.x)
    entry = np.full(ctx.n_bands, len(_L1_CS), dtype=np.float64)
    strength = np.zeros(ctx.n_bands, dtype=np.float64)

    for step, c_value in enumerate(_L1_CS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LinearSVC(
                penalty="l1",
                dual=False,
                C=c_value,
                max_iter=2000,
                random_state=ctx.seed,
            ).fit(z, ctx.y)
        magnitude = np.linalg.norm(np.atleast_2d(model.coef_), axis=0)
        alive = magnitude > 0.0
        first = alive & (entry == len(_L1_CS))
        entry[first] = step
        strength = np.maximum(strength, magnitude)

    # Earliest entry wins; within one entry step, the larger coefficient does.
    ranking = [int(i) for i in np.lexsort((np.arange(ctx.n_bands), -strength, entry))]
    return SelectionOutcome(
        method="l1_path",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={"n_never_selected": int((entry == len(_L1_CS)).sum())},
    )


def _tree_importance(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    """ExtraTrees impurity importance — the one nonlinear multivariate ranker.

    Extremely randomised trees rather than a random forest: the split
    thresholds are drawn rather than optimised, which decorrelates the
    importance estimate across 256 near-duplicate bands instead of letting one
    member of each correlated group absorb the whole group's importance.
    """
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=ctx.seed,
    ).fit(ctx.x, ctx.y)
    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    ranking = _rank_desc(importance)
    return SelectionOutcome(
        method="tree_importance",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={"importance_gini_top": round(float(importance.max()), 6)},
    )


#: Latent variables for the PLS-DA whose VIP scores :func:`_pls_vip` ranks by.
#: 20 is the usual working range for NIR discrimination and is well below the
#: rank of a 256-band matrix at ~3,000 samples.
_PLS_COMPONENTS: int = 20


def _pls_vip(ctx: SelectionContext, budgets: list[int], **_: Any) -> SelectionOutcome:
    """PLS-DA variable importance in projection — the chemometric standard.

    ``VIP_j^2 = C * sum_a (w_aj^2 * SSY_a) / sum_a SSY_a``, where ``SSY_a`` is
    the response variance the a-th latent variable explains. The conventional
    ``VIP > 1`` cut is not used: this study needs an ordering at every budget,
    not one threshold's subset, so the ordering is the whole VIP vector.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(ctx.y)
    one_hot = np.zeros((len(ctx.y), len(classes)), dtype=np.float64)
    one_hot[np.arange(len(ctx.y)), np.searchsorted(classes, ctx.y)] = 1.0

    z = StandardScaler().fit_transform(ctx.x)
    n_comp = int(min(_PLS_COMPONENTS, z.shape[1], max(1, z.shape[0] - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pls = PLSRegression(n_components=n_comp, scale=False).fit(z, one_hot)

    t = np.asarray(pls.x_scores_)  # (n, A)
    q = np.asarray(pls.y_loadings_)  # (K, A)
    w = np.asarray(pls.x_weights_)  # (C, A)
    ssy = np.asarray([float((t[:, a] ** 2).sum() * (q[:, a] ** 2).sum()) for a in range(n_comp)])
    total = float(ssy.sum())
    if total <= 0.0:
        raise ValueError("PLS explained zero response variance; VIP is undefined")

    w_norm = w / np.clip(np.linalg.norm(w, axis=0, keepdims=True), 1e-12, None)
    vip = np.sqrt(ctx.n_bands * ((w_norm**2) * ssy).sum(axis=1) / total)
    ranking = _rank_desc(vip)
    return SelectionOutcome(
        method="pls_vip",
        ranking=ranking,
        per_budget=_sets_from_ranking(ranking, budgets),
        diagnostics={
            "n_components": n_comp,
            "n_vip_above_1": int((vip > 1.0).sum()),
        },
    )


# ══════════════════════════════════════════════════════════════════════
#  Registration
# ══════════════════════════════════════════════════════════════════════

_register(
    MethodSpec(
        "uniform",
        "per_budget",
        False,
        "null_model",
        "—",
        "Evenly spaced bands. The cheapest possible method and a strong baseline "
        "on a smooth cube; every other method has to earn its complexity against it.",
    ),
    _uniform,
)
_register(
    MethodSpec(
        "random",
        "per_budget",
        False,
        "null_model",
        "—",
        "Uniform random subsets. Its spread across draws IS the null distribution.",
        draws=10,
    ),
    _random,
)
_register(
    MethodSpec(
        "variance",
        "ranking",
        False,
        "univariate",
        "—",
        "Highest-variance bands. What an unlabelled pipeline would do.",
    ),
    _variance,
)
_register(
    MethodSpec(
        "fdr",
        "ranking",
        True,
        "univariate",
        "Fisher (1936); multiclass form as in Duda, Hart & Stork (2001)",
        "Between-class over within-class scatter, per band. No redundancy control.",
    ),
    _fdr,
)
_register(
    MethodSpec(
        "mi",
        "ranking",
        True,
        "univariate",
        "Kraskov, Stögbauer & Grassberger, Phys Rev E 69:066138 (2004)",
        "k-NN mutual information with the label; no distributional assumption.",
    ),
    _mi,
)
_register(
    MethodSpec(
        "mrmr",
        "ranking",
        True,
        "redundancy",
        "Peng, Long & Ding, IEEE TPAMI 27(8):1226-38 (2005)",
        "Greedy MID: relevance minus mean redundancy. The repository's incumbent.",
    ),
    _mrmr,
)
_register(
    MethodSpec(
        "spa",
        "ranking",
        False,
        "geometric",
        "Araújo et al., Chemom Intell Lab Syst 57(2):65-73 (2001)",
        "Gram-Schmidt orthogonal selection. Produced the shipped 40-band subset.",
    ),
    _spa,
)
_register(
    MethodSpec(
        "cluster_ward",
        "per_budget",
        True,
        "redundancy",
        "Ward, J Am Stat Assoc 58(301):236-44 (1963); band-clustering as in "
        "Martínez-Usó et al., IEEE TGRS 45(12):4158-71 (2007)",
        "k Ward clusters over bands, each represented by its most discriminative member.",
    ),
    _cluster_ward,
)
_register(
    MethodSpec(
        "pca_loading",
        "ranking",
        False,
        "geometric",
        "Chang et al., IEEE TGRS 37(6):2631-41 (1999)",
        "Largest absolute loading per principal component. Cannot leak a label.",
    ),
    _pca_loading,
)
_register(
    MethodSpec(
        "l1_path",
        "ranking",
        True,
        "embedded",
        "Tibshirani, J R Stat Soc B 58(1):267-88 (1996)",
        "Order of entry along an L1 regularisation path, not one fit's coefficients.",
    ),
    _l1_path,
)
_register(
    MethodSpec(
        "tree_importance",
        "ranking",
        True,
        "embedded",
        "Geurts, Ernst & Wehenkel, Mach Learn 63(1):3-42 (2006)",
        "ExtraTrees impurity importance. The only ranker that can express interactions.",
    ),
    _tree_importance,
)
_register(
    MethodSpec(
        "pls_vip",
        "ranking",
        True,
        "chemometric",
        "Wold, Sjöström & Eriksson, Chemom Intell Lab Syst 58(2):109-30 (2001)",
        "PLS-DA variable importance in projection. The NIR-spectroscopy standard.",
    ),
    _pls_vip,
)


def run_method(
    name: str,
    ctx: SelectionContext,
    budgets: list[int],
    draws: int = 10,
) -> SelectionOutcome:
    """Run one method, converting any failure into a recorded outcome.

    A method that raises is a data point — ``pls_vip`` on a replicate whose PLS
    explains no response variance, ``cluster_ward`` on a degenerate distance
    matrix — and dropping the cell would leave a table that silently describes
    fewer methods than its header claims. The exception text is kept.
    """
    get(name)  # raises a named KeyError before any work is done
    started = time.perf_counter()
    try:
        outcome: SelectionOutcome = _IMPLS[name](ctx, budgets, draws=draws)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        _log.error("method %s failed: %s", name, exc)
        outcome = SelectionOutcome(method=name, failure=f"{type(exc).__name__}: {exc}")
    outcome.seconds = time.perf_counter() - started
    return outcome
