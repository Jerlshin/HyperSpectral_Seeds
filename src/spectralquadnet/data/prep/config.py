"""Config objects for the offline dataset-preparation pipeline.

Both dataclasses are passed explicitly into the functions that need them, so
importing :mod:`spectralquadnet.data.prep` creates no directories, touches no
global RNG and reads no disk.

These are **preparation-time** settings and intentionally live outside the
Hydra training config: they describe how a reduced-band patch dataset is
built once, offline, not how a training run behaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Scan rows kept for segmentation. The raw scenes are 931 rows long; the rows
#: beyond this hold the in-scene Spectralon white tile and the bottom stripes —
#: the extraction notebook's own crop comment reads "Remove white calibration +
#: bottom stripes". Seeds are segmented inside ``[0, SEED_ROWS)`` only;
#: :mod:`~spectralquadnet.data.prep.white_tile` searches the rest.
SEED_ROWS: int = 600

DATA_URL = (
    "https://zenodo.org/records/3241923/files/"
    "RGB%20and%20VIS-NIR%20HSI%20Data%20for%2090%20Rice%20Seed%20Varieties.zip?download=1"
)


@dataclass(frozen=True)
class TileConfig:
    """Thresholds for :mod:`~spectralquadnet.data.prep.white_tile`.

    Informed by the scene geometry (931 × 336 × 256 raw scenes, seeds of
    300-800 px, a 12-bit sensor) but not yet fitted to real tile pixels:
    ``scripts/prepare_dataset.py --probe-tiles N`` measures real scans and writes
    the QC table that should confirm or retune them before any extraction.
    """

    #: ``(first, last)`` scan rows searched; ``None`` is the end of the scene.
    #: Default: the rows the seed crop discards.
    search_rows: tuple[int, int | None] = (SEED_ROWS, None)
    #: Band window for the brightness image — high SNR, clear of the noisy blue end.
    brightness_nm: tuple[float, float] = (500.0, 900.0)
    #: Raw digital number at which the 12-bit sensor clips.
    saturation_dn: float = 4095.0
    #: Candidate pixels exceed this fraction of the 99.5th-percentile brightness.
    candidate_frac: float = 0.6
    #: Smallest region accepted as the tile, in pixels. The seed gate stops at
    #: 800 px, so a region this large cannot be a seed.
    min_area_px: int = 1000
    #: Smallest accepted region solidity — a tile is a compact convex shape.
    min_solidity: float = 0.85
    #: Pixels removed from the region's edge before measuring.
    erode_px: int = 3
    #: Smallest core, in pixels, a measurement is taken from.
    min_core_px: int = 200
    #: A band is measured only when at most this share of core pixels is saturated.
    max_saturated_frac: float = 0.01
    #: Largest accepted robust coefficient of variation (MAD / median) of core brightness.
    max_core_cv: float = 0.10
    #: Tile median brightness over the median of the searched rows outside it.
    min_contrast: float = 1.5
    #: Certified reflectance of the panel; the Zenodo record states 100 %.
    reference_reflectance: float = 1.0
    #: What to do with a band whose white level no tile in its session could
    #: measure. ``True`` (the default, and how ``./dataset`` is built) drops
    #: every such band from **every** scan — one common axis, written to
    #: ``<root>/wavelengths.csv`` — rather than filling it. ``False`` refuses to
    #: extract instead. On this archive the tile clips in the same 608-706 nm
    #: bands in every scan of sessions 0-7 (the lamp peak), so no scan in those
    #: sessions can supply them and any fill would be a model of the lamp, not a
    #: measurement: 215 of 256 bands are kept.
    drop_unresolved_bands: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "search_rows": list(self.search_rows),
            "brightness_nm": list(self.brightness_nm),
            "saturation_dn": self.saturation_dn,
            "candidate_frac": self.candidate_frac,
            "min_area_px": self.min_area_px,
            "min_solidity": self.min_solidity,
            "erode_px": self.erode_px,
            "min_core_px": self.min_core_px,
            "max_saturated_frac": self.max_saturated_frac,
            "max_core_cv": self.max_core_cv,
            "min_contrast": self.min_contrast,
            "reference_reflectance": self.reference_reflectance,
            "drop_unresolved_bands": self.drop_unresolved_bands,
        }


@dataclass
class PrepConfig:
    """Settings for download → segmentation → patch extraction.

    Tier 4 adds one setting (``radiometry``) and five output arrays. The
    outputs are not optional: ``groups.npy`` is what makes a grouped split
    constructible at all (P-1), and the other four are the data the model was
    computing badly or throwing away (P-2, P-3, P-4).
    """

    root: Path = Path("./dataset")
    data_url: str = DATA_URL
    patch_size: int = 64
    num_bands: int = 256

    #: P-2 / T4-2. ``tile`` (the default, and the radiometry of ``./dataset``)
    #: divides each scan by its own in-scene Spectralon tile — reflectance. The
    #: others remain selectable: ``snv`` is the previous per-pixel SNV of
    #: radiance (what ``auto`` resolves to on this archive, whose only
    #: reference cubes are ``black.hdr``), and ``none`` the pre-Tier-4 radiance
    #: domain, kept so the domains can be compared rather than argued about.
    radiometry: str = "tile"

    #: The downloaded archive. ``None`` keeps it at ``<root>/rice_hsi.zip``; set it
    #: to build a second dataset (e.g. the reflectance cube) into a new ``root``
    #: from an archive that already exists elsewhere, without re-downloading.
    archive: Path | None = None

    #: Replace an existing ``patches.npy`` in ``root``. Off by default: the patch
    #: cube is 36 GB and hours to rebuild, and a changed radiometry is a new
    #: dataset that belongs in its own ``root`` rather than on top of the old one.
    overwrite: bool = False

    #: In-scene white-tile detection thresholds, read by the ``tile`` modes.
    tile: TileConfig = field(default_factory=TileConfig)

    @property
    def zip_file(self) -> Path:
        return self.archive if self.archive is not None else self.root / "rice_hsi.zip"

    @property
    def patches_path(self) -> Path:
        return self.root / "patches.npy"

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.npy"

    @property
    def wavelengths_path(self) -> Path:
        """The cube's band axis: ``index`` (1-based instrument band) and ``Wavelength (nm)``."""
        return self.root / "wavelengths.csv"

    # ── Tier-4 outputs ────────────────────────────────────────────────

    @property
    def groups_path(self) -> Path:
        """P-1 / T4-1 — ``(N,)`` int64 cube-level ``scan_id`` per patch."""
        return self.root / "groups.npy"

    @property
    def scan_table_path(self) -> Path:
        """P-1 / T4-1 — ``scan_id`` → session, variety, label, member, patch count."""
        return self.root / "scan_table.csv"

    @property
    def masks_path(self) -> Path:
        """P-3 / T4-3 — ``(N, S, S)`` float16 resized mask, i.e. the fill map alpha."""
        return self.root / "masks.npy"

    @property
    def gain_path(self) -> Path:
        """P-2 / T4-2 — ``(N, 2, S, S)`` float32 per-pixel ``(mean, sd)`` along lambda."""
        return self.root / "gain.npy"

    @property
    def morphology_path(self) -> Path:
        """P-4 / T4-4 — ``(N, 8)`` float32 morphometrics, unstandardised."""
        return self.root / "morphology.npy"

    # ── In-scene white-tile QC (``tile`` modes) ──────────────────────

    @property
    def white_tiles_path(self) -> Path:
        """One row per scan: was the tile found, where, how uniform, how saturated."""
        return self.root / "white_tiles.csv"

    @property
    def white_spectra_path(self) -> Path:
        """Measured and resolved white spectra per scan, and each value's source."""
        return self.root / "white_spectra.npz"

    @property
    def radiometry_log_path(self) -> Path:
        """How this dataset's radiometry was decided, with the thresholds used."""
        return self.root / "radiometry.json"

    def ensure_root(self) -> Path:
        """Create the dataset root directory (and parents) if it doesn't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root


@dataclass
class BandSelectionConfig:
    """Settings for mRMR / SPA band selection."""

    # Paths
    patches_path: str = "./dataset/patches.npy"
    labels_path: str = "./dataset/labels.npy"
    wavelength_path: str = "./dataset/wavelengths.csv"
    output_dir: str = "./dataset/"

    # Decorrelation pre-filter
    corr_threshold: float = 0.995  # Drop band if |r| > this with any kept band

    # Band selection
    #
    # T4-6 / M-14. Both of these ran to 100 and were validated at ten counts
    # ending at 100, but the shipped run recorded a curve that stops at its own
    # chosen k = 40 (`dataset/band_selection_report.csv`), so the "elbow at 40"
    # claim is unfalsifiable from the artifact: the peak the 98 % threshold is
    # measured against is the peak of a truncated curve. The curve now runs to
    # the full band count, which is the only thing that makes an elbow
    # demonstrable rather than asserted.
    n_select_max: int = 256  # Rank up to this many bands in mRMR / SPA
    n_candidates: list[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 128, 160, 192, 224, 256]
    )
    #: T4-6 / F-3. Optional CSV of the **deployed** estimator's accuracy curve
    #: (columns ``n_bands`` and ``accuracy``), i.e. SpectralQuadNet itself
    #: rather than LDA/LinearSVC on mean spectra. When set, it — not the proxy
    #: classifiers — decides the winner and the elbow. F-3 predicts the curve
    #: does not plateau at k = 40 under this estimator, and the six runs that
    #: would produce it are the cost of settling that.
    deployed_curve_path: str | None = None

    # mRMR: k-NN for MI estimation (5 is standard)
    mi_neighbors: int = 5

    # Validation
    cv_folds: int = 5
    svc_C: float = 0.1  # Conservative C; avoids overfit on mean spectra
    elbow_pct: float = 0.98  # Fraction of peak accuracy used for elbow

    # ── IC-4 / CHANGES §4.1 — selection inside the resampling loop ────
    #
    # mRMR relevance was `mutual_info_classif(X, y)` over **every** patch,
    # including the 1,294 that become test. The 40 selected bands are a
    # hyperparameter of the input representation, chosen with test labels in
    # scope. Feature selection outside the resampling loop is a known and
    # quantified source of optimism — Ambroise & McLachlan, PNAS
    # 99(10):6562-6566 (2002), who show cross-validated error estimates become
    # severely optimistic when gene selection precedes CV, and obtain near-zero
    # apparent error on *random labels*. This is genuine label leakage,
    # independent of the split protocol, and it contaminates `grouped` too.
    #
    #: Which fold's training rows the selection is restricted to. ``None``
    #: reproduces the leaky whole-corpus selection, kept because A2's control
    #: arm is exactly that. An integer restricts every step — decorrelation,
    #: FDR, mRMR, SPA and the CV curve — to that fold's training patches.
    fold: int | None = None
    #: ``(N,)`` scan ids, required when :attr:`fold` is set: the training rows
    #: have to be the ones the *training run* will use, and under the grouped
    #: protocol that is a group-disjoint selection.
    groups_path: str = "./dataset/groups.npy"
    #: Split parameters, mirrored from ``cfg.data`` so the selector reproduces
    #: the training run's partition exactly rather than approximating it.
    split_scheme: str = "grouped"
    split_eval_frac: float = 0.30
    calib_frac: float = 0.15
    #: Subdirectory of :attr:`output_dir` per-fold artifacts are written to, so
    #: a fold's band array cannot be mistaken for the whole-corpus one.
    fold_subdir: str = "folds"

    # Memory
    chunk_size: int = 2048
    seed: int = 42
