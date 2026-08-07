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
    """Settings for download → segmentation → patch extraction."""

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
