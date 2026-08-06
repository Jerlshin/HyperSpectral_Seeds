"""Zero-RAM memory-mapped access to the patch cube, labels and wavelengths.

Replaces the three module-level globals in ``HSI_modality_training/hsi_training.py``
@ ``886560f``:

=========================================  ==============  ==========================
Baseline symbol                            Baseline lines  Replacement
=========================================  ==============  ==========================
``_GPU_PATCHES`` / ``_load_data_mmap``     154, 163-170    :attr:`DataStore.patches`
``_GLOBAL_LABELS`` / ``_load_data_mmap``   155, 163-170    :attr:`DataStore.labels`
``_PHYSICAL_WL`` / ``_load_wavelengths_to_gpu``  156, 173-187  :attr:`DataStore.wavelengths`
=========================================  ==============  ==========================

Non-negotiable invariants carried over verbatim (REFACTOR_PLAN.md §3.4)
─────────────────────────────────────────────────────────────────────
* ``np.load(..., mmap_mode="r")`` — a **read-only** mapping. Never pass
  ``mmap_mode=None`` here; that would page the full 5.6 GB cube into RAM.
* **Load-once guard.** :meth:`load_patches` returns early when ``patches`` is
  already set, mirroring the baseline's ``if _GPU_PATCHES is not None: return``.
  :class:`DataStore` is additionally a process-wide singleton, so constructing it
  twice yields the *same object* and never re-mmaps or duplicates the file
  handle — the property ``tests/unit/test_mmap_store.py`` asserts.
* **The per-item copy lives in the Dataset, not here.** ``DataStore`` deliberately
  exposes the raw ``np.memmap``; only
  :meth:`~spectralquadnet.data.datasets.RiceSeedDataset.__getitem__` touches it,
  and only via ``np.array(self.patches[ri])``, which pages in exactly one
  ``(bands, 64, 64)`` patch. Never add a batched ``.copy()`` over the full array
  anywhere on the loader path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import DataConfig


class DataStore:
    """Process-wide singleton holding the mmapped patches, labels and wavelengths.

    Construct it directly (``DataStore(patches_path=..., labels_path=...)``) or via
    :meth:`from_config`. Repeat construction is a no-op that returns the existing
    instance, which is what makes it a drop-in replacement for the baseline's
    lazily-initialised module globals.
    """

    _instance: DataStore | None = None

    patches: np.ndarray | None
    labels: np.ndarray | None
    wavelengths: torch.Tensor | None

    def __new__(cls, *args: Any, **kwargs: Any) -> DataStore:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst.patches = None
            inst.labels = None
            inst.wavelengths = None
            cls._instance = inst
        return cls._instance

    def __init__(
        self,
        patches_path: str | None = None,
        labels_path: str | None = None,
        wavelength_path: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if patches_path is not None and labels_path is not None:
            self.load_patches(patches_path, labels_path)
        if wavelength_path is not None:
            self.load_wavelengths(wavelength_path, device if device is not None else "cpu")

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: DataConfig | Any, device: torch.device | str = "cpu") -> DataStore:
        """Load everything the baseline's ``main()`` loaded (lines 2715-2716)."""
        return cls(
            patches_path=cfg.patches_data,
            labels_path=cfg.labels_path,
            wavelength_path=cfg.wavelength_path,
            device=device,
        )

    @classmethod
    def instance(cls) -> DataStore:
        """Return the singleton, creating an empty one if it does not exist yet."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton so a fresh one can be built.

        Only for tests — production code loads once per process, exactly as the
        baseline's module-level globals did.
        """
        cls._instance = None

    # ── Loading (mirrors the baseline free functions) ──────────────────

    def load_patches(self, patches_path: str, labels_path: str) -> None:
        """Memory-map the patch cube and read the labels array into RAM.

        Verbatim translation of ``_load_data_mmap`` (baseline lines 163-170); the
        ``global`` statement becomes instance state and the ``is not None`` guard
        is preserved exactly.
        """
        if self.patches is not None:
            return

        print("[DATA] Memory-mapping dataset from disk (Zero-RAM footprint)...")
        self.patches = np.load(patches_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        print(f"[DATA] ✓ Indexed {self.patches.shape[0]} samples via mmap.")

    def load_wavelengths(self, csv_path: str, device: torch.device | str) -> None:
        """Read and min-max normalise the physical wavelengths.

        Verbatim translation of ``_load_wavelengths_to_gpu`` (baseline lines
        173-187), including the ``sep=None`` sniffing, the last-column selection
        and the re-raise as ``RuntimeError``.
        """
        if self.wavelengths is not None:
            return

        print("[DATA] Loading physical wavelengths from CSV...")
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")
            raw_wl = df.iloc[:, -1].values.astype(np.float32)
            wl_norm = (raw_wl - raw_wl.min()) / (raw_wl.max() - raw_wl.min())
            self.wavelengths = torch.from_numpy(wl_norm).to(device)
            print(f"[DATA] ✓ Loaded physical wavelengths: {self.wavelengths.size(0)} bands.")
        except Exception as e:
            raise RuntimeError(f"Failed to load wavelengths.csv: {e}") from e

    # ── Typed accessors ───────────────────────────────────────────────

    def require_patches(self) -> np.ndarray:
        if self.patches is None:
            raise RuntimeError("DataStore.load_patches() has not been called.")
        return self.patches

    def require_labels(self) -> np.ndarray:
        if self.labels is None:
            raise RuntimeError("DataStore.load_patches() has not been called.")
        return self.labels

    def require_wavelengths(self) -> torch.Tensor:
        if self.wavelengths is None:
            raise RuntimeError("DataStore.load_wavelengths() has not been called.")
        return self.wavelengths
