"""The study's configuration: one frozen value, recorded next to its results.

Every stage takes this object and nothing else, which is what makes a resumed
run provably the same run: :meth:`BandStudyConfig.fingerprint` hashes the fields
that change a number, the store writes it beside the results, and a resume
against a different fingerprint refuses rather than silently appending cells
from a second experiment to the first one's table.

Fields that do **not** enter the fingerprint are the ones that cannot change a
result — output location, worker count, log verbosity. Putting them in would
make "resume with ``--jobs 4``" look like a different experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

#: Default band budgets. Deliberately dense at the small end, where the curve
#: moves, and terminating at the **full** band count, which is the property
#: CHANGES M-14 says both shipped selections lack: an elbow claim is only
#: falsifiable if the curve extends past the elbow.
DEFAULT_BUDGETS: tuple[int, ...] = (
    1,
    2,
    3,
    5,
    8,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    64,
    80,
    100,
    128,
    160,
    192,
    224,
    256,
)

#: Default selection methods. See :mod:`spectralquadnet.bandstudy.methods` for
#: what each one is and why it is here; the two nulls (``uniform``, ``random``)
#: are not optional in any defensible version of this experiment.
DEFAULT_METHODS: tuple[str, ...] = (
    "uniform",
    "random",
    "variance",
    "fdr",
    "mi",
    "mrmr",
    "spa",
    "cluster_ward",
    "pca_loading",
    "l1_path",
    "tree_importance",
    "pls_vip",
)

#: Default proxy estimators, one per model family: a generative linear model, a
#: discriminative linear one, and a nonlinear ensemble. Three families rather
#: than one because "how many bands are needed" is allowed to depend on what is
#: reading them, and a conclusion that holds for only one hypothesis class is a
#: conclusion about that class.
DEFAULT_PROXIES: tuple[str, ...] = ("lda", "linsvc", "extratrees")

#: Where the study writes, relative to the repo root.
DEFAULT_OUTPUT_ROOT = "outputs/band_study"

#: Fields that cannot change a measured number and are therefore excluded from
#: the fingerprint.
_NON_SEMANTIC: frozenset[str] = frozenset(
    {"output_root", "jobs", "verbose", "force", "dry_run", "progress"}
)


@dataclass(frozen=True)
class BandStudyConfig:
    """Everything the study needs, and nothing about how it is displayed."""

    # ── Inputs ────────────────────────────────────────────────────────
    #: The **full** 256-band cube. Not the 40-band or 100-band reduction: the
    #: whole question is what a budget below 256 costs, and a study run on a
    #: pre-reduced cube can only answer it about bands somebody already chose.
    patches_path: str = "./dataset/patches.npy"
    labels_path: str = "./dataset/labels.npy"
    groups_path: str = "./dataset/groups.npy"
    wavelength_path: str = "./dataset/wavelengths.csv"
    #: ``(N, 8)`` morphometrics, appended to the proxy feature vector when
    #: ``use_morphology`` is on. Off by default: the question is about bands.
    morphology_path: str = "./dataset/morphology.npy"

    # ── Protocol (mirrors configs/data/spa40_90class_pfix.yaml) ───────
    split_scheme: str = "grouped"
    split_eval_frac: float = 0.30
    calib_frac: float = 0.15
    single_group_policy: str = "error"
    folds: tuple[int, ...] = (0, 1)

    # ── The sweep ─────────────────────────────────────────────────────
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    methods: tuple[str, ...] = DEFAULT_METHODS
    proxies: tuple[str, ...] = DEFAULT_PROXIES
    #: Independent selection + evaluation replicates per (fold, method). Each
    #: draws a different stratified subsample of the fold's training rows, so
    #: the spread across replicates measures both metric variance *and*
    #: selection stability — the second of which no single-shot selection can
    #: report at all.
    replicates: int = 5
    #: Share of the fold's training rows each replicate sees. Below 1.0 so that
    #: replicates differ; high enough that each still represents the split.
    replicate_frac: float = 0.8
    #: Draws per budget for the ``random`` null. Its spread *is* the null
    #: distribution every other method is compared against, so one draw would
    #: be useless.
    random_draws: int = 10

    # ── Features ──────────────────────────────────────────────────────
    #: ``mean`` — the foreground-masked mean spectrum, the representation the
    #: project's own most important baseline uses (CHANGES §19.4).
    #: ``mean_sd`` — that, concatenated with the per-band spatial sd, which
    #: carries texture the mean discards and is the cheapest test of whether
    #: the proxies' ceiling is a representation limit rather than a band limit.
    features: str = "mean"
    use_morphology: bool = False

    # ── Numerics ──────────────────────────────────────────────────────
    seed: int = 42
    #: Patches per chunk in the mean-spectrum pass over the mmapped cube.
    chunk_size: int = 512
    #: Bootstrap resamples for the confirm stage's intervals. 0 disables.
    n_boot: int = 2000
    #: Decorrelation pre-filter threshold used by the methods that take one.
    #: 0.995 is the value the shipped selector used; kept so the reimplemented
    #: mRMR/SPA arms are the same arms.
    corr_threshold: float = 0.995
    #: k-NN neighbours for the mutual-information estimator.
    mi_neighbors: int = 5

    # ── Decision thresholds (pre-registered) ──────────────────────────
    #: A budget is "saturated" at the smallest k whose calib score is within
    #: this many points of the best score on the curve. Stated as an absolute
    #: macro-F1 tolerance rather than a percentage of the peak, because a
    #: percentage of a low peak is a tighter test than a percentage of a high
    #: one and the two curves here differ by 30 points.
    plateau_tol: float = 0.01
    #: A selection is "unstable" below this mean pairwise Jaccard over
    #: replicates. 0.5 = the median band in one replicate's set is not in
    #: another's.
    stability_floor: float = 0.5
    #: A method must beat the ``random`` null's mean by at least this much, at
    #: some budget, to be called effective.
    null_margin: float = 0.01

    # ── Non-semantic ──────────────────────────────────────────────────
    output_root: str = DEFAULT_OUTPUT_ROOT
    jobs: int = 1
    verbose: bool = False
    force: bool = False
    dry_run: bool = False
    progress: bool = True

    #: Free-form provenance, written into the manifest and ignored by the
    #: fingerprint. Use it to record why a variant run exists.
    note: str = ""

    # ── Derived paths ─────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return Path(self.output_root)

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def selections_dir(self) -> Path:
        return self.root / "selections"

    @property
    def bands_dir(self) -> Path:
        """Where the per-(method, fold, k) index arrays and λ CSVs go.

        These are the files a training run points ``data.band_indices_path`` at,
        so they are kept out of ``selections/`` — that directory holds the
        study's own bookkeeping, this one holds inputs to other experiments.
        """
        return self.root / "bands"

    @property
    def proxy_dir(self) -> Path:
        return self.root / "proxy"

    @property
    def confirm_dir(self) -> Path:
        return self.root / "confirm"

    @property
    def neural_dir(self) -> Path:
        return self.root / "neural"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def figures_dir(self) -> Path:
        return self.analysis_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.analysis_dir / "tables"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def report_path(self) -> Path:
        return self.root / "REPORT.md"

    # ── Identity ──────────────────────────────────────────────────────

    def semantic_dict(self) -> dict[str, Any]:
        """The fields that can change a number, sorted."""
        payload = {k: v for k, v in asdict(self).items() if k not in _NON_SEMANTIC}
        payload.pop("note", None)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in sorted(payload.items())}

    def fingerprint(self) -> str:
        """A short stable hash of :meth:`semantic_dict`.

        Written next to the results and checked on resume. Two runs with the
        same fingerprint produced cells that belong in the same table; two with
        different fingerprints did not, however similar the command lines look.
        """
        blob = json.dumps(self.semantic_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        """Everything, including the non-semantic fields, for the manifest."""
        out = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        out["fingerprint"] = self.fingerprint()
        return out

    def with_(self, **changes: Any) -> BandStudyConfig:
        """A copy with fields replaced — the CLI's only mutation path."""
        return replace(self, **changes)

    # ── Validation ────────────────────────────────────────────────────

    def validate(self, n_bands_available: int | None = None) -> list[str]:
        """Design problems worth refusing to start on, as readable strings.

        Checked before any compute is spent, because every one of these
        produces a table that looks finished and answers a different question
        from the one asked.
        """
        problems: list[str] = []

        if not self.budgets:
            problems.append("no band budgets requested")
        if any(k < 1 for k in self.budgets):
            problems.append(f"band budgets must be >= 1, got {sorted(self.budgets)}")
        if n_bands_available is not None:
            over = [k for k in self.budgets if k > n_bands_available]
            if over:
                problems.append(
                    f"budgets {sorted(over)} exceed the cube's {n_bands_available} bands"
                )
            if n_bands_available not in self.budgets:
                problems.append(
                    f"the full band count ({n_bands_available}) is not among the budgets. "
                    "Without it there is no reference for 'how much did reduction cost', and "
                    "any elbow found is the endpoint of a truncated curve (CHANGES M-14)."
                )
        if not self.methods:
            problems.append("no selection methods requested")
        if "random" not in self.methods:
            problems.append(
                "the `random` null is not among the methods. With neighbour correlations above "
                "0.99 a random subset is a strong baseline, and a method that does not beat it "
                "has not been shown to select anything."
            )
        if not self.proxies:
            problems.append("no proxy estimators requested")
        if self.replicates < 1:
            problems.append(f"replicates must be >= 1, got {self.replicates}")
        if self.replicates < 2:
            problems.append(
                "replicates < 2 makes selection stability and metric variance unmeasurable; "
                "the study will report neither."
            )
        if not 0.0 < self.replicate_frac <= 1.0:
            problems.append(f"replicate_frac must be in (0, 1], got {self.replicate_frac}")
        if self.features not in ("mean", "mean_sd"):
            problems.append(f"features must be 'mean' or 'mean_sd', got {self.features!r}")
        if self.split_scheme not in ("grouped", "stratified"):
            problems.append(f"unknown split_scheme {self.split_scheme!r}")
        if self.split_scheme == "stratified":
            problems.append(
                "split_scheme=stratified puts every acquisition bundle in train AND eval, so a "
                "band chosen on train is chosen on a tray that is also scored. Every conclusion "
                "would be about acquisition recognition. Use `grouped` unless this arm is "
                "deliberately the leaky contrast."
            )
        return problems


def cost_estimate(cfg: BandStudyConfig) -> dict[str, int]:
    """How many cells each stage will run. Printed by ``list``, before anything.

    Mirrors what the pipeline actually does, which is not the naive product:

    * A **ranking** method produces one ordering per (fold, scope) and every
      budget is a prefix of it, so k does not multiply its selection cost. Only
      the per-budget methods (``uniform``, ``cluster_ward``) pay per k.
    * ``random`` runs at the canonical scope only. Its uncertainty is the spread
      across draws, which it already has; resampling a uniform random choice
      would measure the resampler.
    * At k equal to the full band count every method returns the same set, so
      those cells collapse to one fit per (fold, scope, proxy). The estimate
      subtracts them, which is why it is lower than methods × budgets × proxies.
    """
    n_folds = len(cfg.folds)
    n_budgets = len(cfg.budgets)
    scopes = cfg.replicates + 1  # the replicates, plus the canonical selection
    per_budget = {"uniform", "cluster_ward"}

    named = [m for m in cfg.methods if m != "random"]
    ranking = [m for m in named if m not in per_budget]
    budgeted = [m for m in named if m in per_budget]

    # `random` draws once rather than `random_draws` times at the full budget:
    # there is only one subset of size C, so extra draws would be identical
    # rows padding the null distribution at exactly the budget it is compared
    # against.
    full = max(cfg.budgets)
    random_sets = 0
    if "random" in cfg.methods:
        random_sets = n_folds * (
            (n_budgets - 1) * cfg.random_draws + 1
            if full in cfg.budgets
            else n_budgets * cfg.random_draws
        )

    selections = n_folds * scopes * len(ranking)
    selections += n_folds * scopes * len(budgeted) * n_budgets
    selections += random_sets

    band_sets = n_folds * scopes * len(named) * n_budgets + random_sets

    # The full-budget column: every named method selects every band there, so
    # all of them share one fit per (fold, scope, proxy).
    duplicates = n_folds * scopes * (len(named) - 1) if full in cfg.budgets else 0

    return {
        "folds": n_folds,
        "selection_calls": selections,
        "band_sets": band_sets,
        "proxy_fits": max(band_sets - duplicates, 0) * len(cfg.proxies),
        "confirm_fits": n_folds * (len(cfg.methods) + 3) * len(cfg.proxies),
    }


def default_config(**overrides: Any) -> BandStudyConfig:
    """A :class:`BandStudyConfig` with the shipped defaults and ``overrides``."""
    return BandStudyConfig(**overrides)


#: A cheap preset for verifying the machinery end to end without spending the
#: real budget. Every stage runs, every artifact is written, every table and
#: figure is produced — only the grid shrinks. Selected with ``--quick``.
QUICK_OVERRIDES: dict[str, Any] = {
    "budgets": (5, 20, 50, 256),
    "methods": ("uniform", "random", "mi", "mrmr"),
    "proxies": ("lda",),
    "replicates": 2,
    "random_draws": 3,
    "n_boot": 200,
}


@dataclass(frozen=True)
class StageResult:
    """What one stage did — printed, and merged into the run manifest."""

    stage: str
    n_done: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "n_done": self.n_done,
            "n_skipped": self.n_skipped,
            "n_failed": self.n_failed,
            "seconds": round(self.seconds, 2),
            "notes": list(self.notes),
        }

    def line(self) -> str:
        return (
            f"{self.stage:9s}  {self.n_done:5d} done  {self.n_skipped:5d} resumed  "
            f"{self.n_failed:4d} failed  {self.seconds:8.1f}s"
        )
