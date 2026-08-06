"""Config objects for the offline dataset-preparation pipeline.

The two root scripts this package absorbs each carried their own module-level
constants, applied as import-time side effects:

* ``data_setup_v3.py`` lines 27-40 — ``ROOT``/``ZIP_FILE``/``PATCHES_PATH``/
  ``LABELS_PATH``/``PATCH_SIZE``/``NUM_BANDS``, plus ``ROOT.mkdir(exist_ok=True)``
  executed at import.
* ``band_selection.py`` lines 66-92 — a ``CONFIG`` dict, plus
  ``np.random.seed(CONFIG["seed"])`` executed at import.

Both are now plain dataclasses passed explicitly into the functions that need
them, so importing :mod:`spectralquadnet.data.prep` creates no directories,
touches no global RNG and reads no disk. Every default below is transcribed
verbatim from those scripts — nothing is re-defaulted (REFACTOR_PLAN.md §6).

These are **preparation-time** settings and intentionally live outside the Hydra
training config: they describe how ``dataset/patches_spa_40b.npy`` was built once,
not how a training run behaves.
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

    Source: ``data_setup_v3.py`` lines 27-40.
    """

    root: Path = Path("./dataset")
    data_url: str = DATA_URL
    patch_size: int = 64
    num_bands: int = 256

    @property
    def zip_file(self) -> Path:
        return self.root / "rice_hsi.zip"

    @property
    def patches_path(self) -> Path:
        return self.root / "patches.npy"

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.npy"

    def ensure_root(self) -> Path:
        """Create the dataset root.

        The baseline did this at import time (``ROOT.mkdir(exist_ok=True)``,
        line 28); it is now an explicit call made by the entrypoints.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root


@dataclass
class BandSelectionConfig:
    """Settings for mRMR / SPA band selection.

    Source: ``band_selection.py`` lines 66-91, field-for-field.
    """

    # Paths
    patches_path: str = "./dataset/patches.npy"
    labels_path: str = "./dataset/labels.npy"
    wavelength_path: str = "./dataset/wavelengths.csv"
    output_dir: str = "./dataset/"

    # Decorrelation pre-filter
    corr_threshold: float = 0.995  # Drop band if |r| > this with any kept band

    # Band selection
    n_select_max: int = 100  # Rank up to this many bands in mRMR / SPA
    n_candidates: list[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50, 70, 100]
    )

    # mRMR: k-NN for MI estimation (5 is standard)
    mi_neighbors: int = 5

    # Validation
    cv_folds: int = 5
    svc_C: float = 0.1  # Conservative C; avoids overfit on mean spectra
    elbow_pct: float = 0.98  # Fraction of peak accuracy used for elbow

    # Memory
    chunk_size: int = 2048
    seed: int = 42
