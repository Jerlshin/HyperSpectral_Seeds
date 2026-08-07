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

DATA_URL = (
    "https://zenodo.org/records/3241923/files/"
    "RGB%20and%20VIS-NIR%20HSI%20Data%20for%2090%20Rice%20Seed%20Varieties.zip?download=1"
)


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

    #: P-2 / T4-2. ``auto`` divides by a white panel when the archive has one
    #: and applies per-pixel SNV when it does not — which is this archive's
    #: case, verified: its only reference cubes are ``black.hdr``. ``none``
    #: reproduces the pre-Tier-4 radiance domain, and therefore §2.1.1's leak
    #: channel; it exists so the two can be compared rather than argued about.
    radiometry: str = "auto"

    @property
    def zip_file(self) -> Path:
        return self.root / "rice_hsi.zip"

    @property
    def patches_path(self) -> Path:
        return self.root / "patches.npy"

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.npy"

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

    # Memory
    chunk_size: int = 2048
    seed: int = 42
