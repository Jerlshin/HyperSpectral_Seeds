"""Zero-RAM memory-mapped access to the patch cube, labels and wavelengths.

Non-negotiable invariants
──────────────────────────
* ``np.load(..., mmap_mode="r")`` — a **read-only** mapping. Never pass
  ``mmap_mode=None`` here; that would page the full multi-GB cube into RAM.
* **Load-once guard.** :meth:`load_patches` returns early when ``patches`` is
  already set. :class:`DataStore` is additionally a process-wide singleton,
  so constructing it twice yields the *same object* and never re-mmaps or
  duplicates the file handle — the property ``tests/unit/test_mmap_store.py``
  asserts.
* **The per-item copy lives in the Dataset, not here.** ``DataStore``
  deliberately exposes the raw ``np.memmap``; only
  :meth:`~spectralquadnet.data.datasets.RiceSeedDataset.__getitem__` touches
  it, and only via ``np.array(self.patches[ri])``, which pages in exactly one
  ``(bands, 64, 64)`` patch. Never add a batched ``.copy()`` over the full
  array anywhere on the loader path.
* **The paths are kept alongside the arrays.** A DataLoader worker is a
  separate process, and on macOS it is started with ``spawn``, so the dataset
  is *pickled* into it. ``np.memmap`` pickles by materialising — a worker that
  received one would silently pull the whole 5.6 GB cube into its own RAM,
  which is the exact failure the zero-RAM invariant exists to prevent. So
  :attr:`DataStore.patches_path` and :attr:`DataStore.masks_path` are recorded
  here and ``RiceSeedDataset.__setstate__`` re-opens the mapping in the worker
  instead of inheriting it. Never remove the paths without also removing
  ``num_workers > 0``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import DataConfig

_log = logging.getLogger(__name__)


class DataStore:
    """Process-wide singleton holding the mmapped patches, labels and wavelengths.

    Construct it directly (``DataStore(patches_path=..., labels_path=...)``) or via
    :meth:`from_config`. Repeat construction is a no-op that returns the existing
    instance, so every caller in the process shares one open mmap handle.
    """

    _instance: DataStore | None = None

    #: The mmapped cube. Typed as a plain array because `np.load(mmap_mode="r")`
    #: returns `np.memmap`, an `ndarray` subclass — nothing on the read path may
    #: depend on which of the two it is.
    patches: npt.NDArray[Any] | None
    labels: npt.NDArray[Any] | None
    wavelengths: torch.Tensor | None
    #: FE-2 / T3-7. ``(N, S, S)`` fill map, mmapped like the patches. ``None``
    #: when ``data.masks_path`` is unset or the file has not been written yet,
    #: in which case every masked module falls back to its threshold.
    masks: npt.NDArray[Any] | None
    #: FU-4 / P-4. ``(N, 8)`` raw morphometrics, small enough to hold in RAM.
    #: Raw, not standardised: the mean and scale are a property of the *split*,
    #: and `data/morphometrics.py` fits them on the training indices alone.
    morphology: npt.NDArray[Any] | None
    #: Where the two mmapped arrays came from, so a DataLoader worker can
    #: re-open them instead of being handed a pickled copy — see the module
    #: docstring. ``None`` for a store built from in-memory arrays (the tests).
    patches_path: str | None
    masks_path: str | None

    def __new__(cls, *args: Any, **kwargs: Any) -> DataStore:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst.patches = None
            inst.labels = None
            inst.wavelengths = None
            inst.masks = None
            inst.morphology = None
            inst.patches_path = None
            inst.masks_path = None
            cls._instance = inst
        return cls._instance

    def __init__(
        self,
        patches_path: str | None = None,
        labels_path: str | None = None,
        wavelength_path: str | None = None,
        device: torch.device | str | None = None,
        masks_path: str | None = None,
        morphology_path: str | None = None,
    ) -> None:
        if patches_path is not None and labels_path is not None:
            self.load_patches(patches_path, labels_path)
        if wavelength_path is not None:
            self.load_wavelengths(wavelength_path, device if device is not None else "cpu")
        self.load_side_arrays(masks_path, morphology_path)

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: DataConfig | Any, device: torch.device | str = "cpu") -> DataStore:
        """Build/fetch the singleton, loading patches, labels and wavelengths from ``cfg``."""
        return cls(
            patches_path=cfg.patches_data,
            labels_path=cfg.labels_path,
            wavelength_path=cfg.wavelength_path,
            device=device,
            masks_path=str(getattr(cfg, "masks_path", "") or "") or None,
            morphology_path=str(getattr(cfg, "morphology_path", "") or "") or None,
        )

    @classmethod
    def instance(cls) -> DataStore:
        """Return the singleton, creating an empty one if it does not exist yet."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton so a fresh one can be built.

        Only for tests — production code loads once per process.
        """
        cls._instance = None

    # ── Typed accessors for the optional side arrays ──────────────────

    def has_masks(self) -> bool:
        """Whether the FE-2 fill map is available for this run."""
        return self.masks is not None

    def has_morphology(self) -> bool:
        """Whether the P-4 morphometrics are available for this run."""
        return self.morphology is not None

    # ── Loading ──────────────────────────────────────────────────────

    def load_patches(self, patches_path: str, labels_path: str) -> None:
        """Memory-map the patch cube and read the labels array into RAM.

        A no-op if patches are already loaded (see the load-once invariant
        in the module docstring).
        """
        if self.patches is not None:
            return

        _log.info("[DATA] Memory-mapping dataset from disk (Zero-RAM footprint)...")
        self.patches = np.load(patches_path, mmap_mode="r")
        self.patches_path = str(patches_path)
        self.labels = np.load(labels_path)
        _log.info("[DATA] ✓ Indexed %d samples via mmap.", self.patches.shape[0])

    def load_side_arrays(
        self, masks_path: str | None = None, morphology_path: str | None = None
    ) -> None:
        """Memory-map the fill map and read the morphometrics — both optional.

        A configured path that does not exist is **not** an error here, and the
        reason is the one T4-5 already settled for the calibration split: these
        arrays are written by an extraction run that takes hours and 36 GB, and
        the two committed data configs differ in whether they expect it. What
        *would* be a defect is a path that silently loads a row-misaligned
        array, so the row count is checked against the labels.

        Raises:
            ValueError: An array exists but is not row-aligned with the labels,
                or the morphometrics are not ``(N, 8)``.
        """
        if masks_path and self.masks is None and Path(masks_path).exists():
            self.masks = np.load(masks_path, mmap_mode="r")
            self.masks_path = str(masks_path)
            self._check_rows(masks_path, self.masks)
            _log.info("[DATA] ✓ Mapped fill maps (FE-2): %s", self.masks.shape)

        if morphology_path and self.morphology is None and Path(morphology_path).exists():
            morph = np.load(morphology_path).astype(np.float32)
            self._check_rows(morphology_path, morph)
            if morph.ndim != 2:
                raise ValueError(f"{morphology_path} must be (N, 8); got {morph.shape}")
            self.morphology = morph
            _log.info("[DATA] ✓ Loaded morphometrics (FU-4): %s", morph.shape)

    def _check_rows(self, path: str, array: npt.NDArray[Any]) -> None:
        if self.labels is not None and len(array) != len(self.labels):
            raise ValueError(
                f"{path} has {len(array)} rows but labels.npy has {len(self.labels)} — "
                "they must be row-aligned; re-run scripts/prepare_dataset.py."
            )

    def load_wavelengths(self, csv_path: str, device: torch.device | str) -> None:
        """Read and min-max normalise the physical wavelengths.

        Uses ``sep=None`` to sniff the CSV's delimiter and takes the last
        column as the wavelength value; a no-op if wavelengths are already
        loaded.

        Raises:
            RuntimeError: The CSV could not be read or parsed as expected.
        """
        if self.wavelengths is not None:
            return

        _log.info("[DATA] Loading physical wavelengths from CSV...")
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")
            raw_wl = df.iloc[:, -1].values.astype(np.float32)
            wl_norm = (raw_wl - raw_wl.min()) / (raw_wl.max() - raw_wl.min())
            self.wavelengths = torch.from_numpy(wl_norm).to(device)
            _log.info("[DATA] ✓ Loaded physical wavelengths: %d bands.", self.wavelengths.size(0))
        except Exception as e:
            raise RuntimeError(f"Failed to load wavelengths.csv: {e}") from e

    # ── Typed accessors ───────────────────────────────────────────────

    def require_patches(self) -> npt.NDArray[Any]:
        """The mmapped patch cube, narrowed to non-``None`` for callers.

        Raises:
            RuntimeError: :meth:`load_patches` has not been called yet.
        """
        if self.patches is None:
            raise RuntimeError("DataStore.load_patches() has not been called.")
        return self.patches

    def require_labels(self) -> npt.NDArray[Any]:
        """The in-RAM label array, narrowed to non-``None`` for callers.

        Raises:
            RuntimeError: :meth:`load_patches` has not been called yet.
        """
        if self.labels is None:
            raise RuntimeError("DataStore.load_patches() has not been called.")
        return self.labels

    def require_wavelengths(self) -> torch.Tensor:
        """The normalised wavelength tensor, narrowed to non-``None`` for callers.

        Raises:
            RuntimeError: :meth:`load_wavelengths` has not been called yet.
        """
        if self.wavelengths is None:
            raise RuntimeError("DataStore.load_wavelengths() has not been called.")
        return self.wavelengths


class BandGeometryError(ValueError):
    """The cube, the wavelength vector and ``data.num_bands`` do not agree."""


def band_geometry(cfg: DataConfig | Any, store: DataStore) -> dict[str, int]:
    """Check that the three descriptions of the spectral axis are one description.

    Four numbers claim to be the band count and nothing used to compare them:

    ==========================  =========================================
    ``patches.npy``'s axis 1    what is on disk
    ``data.band_indices_path``  which of those bands are read, if subsetting
    ``data.wavelength_path``    the λ the model's operators are built from
    ``data.num_bands``          the model's input width
    ==========================  =========================================

    A disagreement between the last two is a network whose Savitzky–Golay
    operators, continuum hull and index bank describe bands the input does not
    contain — silently wrong rather than loudly broken, and the failure surfaces
    as a shape error inside a branch that names none of the four. The primary
    pipeline reads all 256 stored bands, so the check is trivially satisfied
    there; it earns its place on the retained band-selection arms, where the
    three files genuinely differ and are produced by three separate steps.

    Args:
        cfg: The ``data`` config group.
        store: A store with patches and wavelengths already loaded.

    Returns:
        ``{"stored": …, "selected": …, "wavelengths": …, "configured": …}``.

    Raises:
        BandGeometryError: Any two of the four disagree.
    """
    stored = int(store.require_patches().shape[1])
    n_wl = int(store.require_wavelengths().numel())
    configured = int(cfg.num_bands)

    index_path = str(getattr(cfg, "band_indices_path", "") or "")
    selected = stored
    if index_path:
        selected = int(np.asarray(np.load(index_path)).reshape(-1).size)

    if selected != configured:
        detail = (
            f"{Path(index_path).name} selects {selected} of the cube's {stored} bands"
            if index_path
            else f"{Path(str(cfg.patches_data)).name} stores {stored} bands"
        )
        raise BandGeometryError(
            f"data.num_bands={configured} but {detail}. num_bands sets the model's input "
            "width; the two must agree. The primary pipeline is the full cube "
            "(num_bands=256, no band_indices_path); a reduced arm must set both."
        )
    if n_wl != configured:
        raise BandGeometryError(
            f"data.wavelength_path holds {n_wl} wavelengths but data.num_bands={configured}. "
            "Every λ-aware operator — the Savitzky-Golay derivatives, the continuum hull, "
            "Branch D's λ windows — is built from that vector, so a mismatch builds a model "
            "that describes different bands from the ones it reads."
        )
    return {
        "stored": stored,
        "selected": selected,
        "wavelengths": n_wl,
        "configured": configured,
    }
