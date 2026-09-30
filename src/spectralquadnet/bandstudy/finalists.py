"""The finalist band sets for the neural stage: evenly spaced, 430 nm and up.

What these are and why they are fixed
─────────────────────────────────────
The September 2026 band study (``outputs/band_research``) compared twelve
selectors under nested, acquisition-disjoint evaluation. For the spatial-spectral
proxy, *evenly spaced* bands tied or beat every supervised selector at matched
budget, and 32-64 of them scored above the full 256-band cube on held-out bundles
in every fold × seed cell. So the arms the network is confirmed on are a
deterministic, label-free rule rather than the output of a selector:

* **Candidates are the bands at or above** :data:`LOWER_BOUND_NM`. Below it the
  per-pixel SNR is under 10, up to 15 % of pixels are clipped to zero by the dark
  subtraction at 383 nm, and after per-pixel SNV those "bands" are 85-91 %
  explained by one whole-spectrum statistic — a normalisation artifact, not a
  measurement a k-band instrument could reproduce.
* **Targets are spaced evenly in wavelength** from the first candidate to the
  last, and each target takes the nearest candidate band. A gap in the axis —
  bands removed upstream, e.g. the 608-706 nm window the white tile could not
  measure in the reflectance cube — is collapsed first, so the targets spread
  over the *measured* spectrum instead of piling onto the gap's two edges. On a
  gap-free axis this is exactly the plain rule.
* **No label is read**, so a set is the same for both folds, needs no nested
  selection and cannot leak — one wavelength list per budget for the paper.

The indices address the axis of the wavelength file the sets were cut from —
``dataset/wavelengths.csv``, the 215-band reflectance axis, which drops the
bands the white tile could not measure. Sets cut from the previous 256-band SNV
axis address different bands and must not be reused on it.

Each set is written in the band study's own format — ``<name>.npy`` (int64
indices) plus ``<name>_wavelengths.csv`` (``index,Wavelength (nm)``) — so it is
loadable by ``data.band_indices_path`` / ``data.wavelength_path`` unchanged, and
``manifest.json`` records the exact training overrides for each, including the
band-expressed augmentation widths rescaled by
:func:`~spectralquadnet.data.datasets.band_augmentation_widths`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from spectralquadnet.data.datasets import band_augmentation_widths

#: Shortest wavelength a finalist set may contain, in nm.
LOWER_BOUND_NM: float = 430.0

#: The budgets confirmed on the network: the proxy plateau (24-48, centred on 32)
#: plus one step either side of it.
FINALIST_BUDGETS: tuple[int, ...] = (16, 24, 32, 48, 64)

#: A step between neighbouring candidates wider than this multiple of the median
#: step is a gap (bands removed upstream), not the instrument's spacing, which
#: varies by well under 1 %.
GAP_FACTOR: float = 1.5

#: Prefix of every finalist file name. Encodes the rule, not a fold: the sets
#: are fold-independent by construction.
NAME_PREFIX: str = "uniform430"


def read_wavelengths(path: str | Path) -> npt.NDArray[np.float64]:
    """The wavelength column exactly as training reads it: sniffed delimiter, last column."""
    frame = pd.read_csv(path, sep=None, engine="python")
    return np.asarray(frame.iloc[:, -1].values, dtype=np.float64)


def measured_axis(wavelengths: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Wavelength with every gap collapsed to one median step — the "measured spectrum" coordinate.

    Monotone in wavelength and equal to it (up to a constant) wherever the axis
    has no gap, so spacing evenly in it is spacing evenly in wavelength over the
    bands that exist.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    if wl.size < 2:
        return wl.copy()
    steps = np.diff(wl)
    typical = float(np.median(steps))
    excess = np.where(steps > GAP_FACTOR * typical, steps - typical, 0.0)
    out: npt.NDArray[np.float64] = wl - np.concatenate([[0.0], np.cumsum(excess)])
    return out


def window_uniform_bands(
    wavelengths: npt.NDArray[Any], k: int, lo_nm: float, hi_nm: float | None = None
) -> npt.NDArray[np.int64]:
    """``k`` band indices spaced evenly in wavelength over ``[lo_nm, hi_nm]``.

    Only bands inside the window are candidates, so every returned band honours
    the bound — the nearest band to 430 nm on this instrument is 429.6 nm, which
    the window excludes. Targets run from the first candidate's wavelength to
    the last one's, so both ends of the usable range are always sampled. Gaps in
    the axis are collapsed first (:func:`measured_axis`).

    Raises:
        ValueError: ``k`` is below 1, or the window holds fewer than ``k``
            distinct bands at this spacing.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    upper = float(wl.max()) if hi_nm is None else float(hi_nm)
    candidates = np.flatnonzero((wl >= lo_nm) & (wl <= upper))
    if candidates.size < k:
        raise ValueError(f"only {candidates.size} bands lie in [{lo_nm}, {upper}] nm; k={k}")
    u = measured_axis(wl[candidates])
    targets = np.linspace(u[0], u[-1], k)
    picked = candidates[np.abs(u[None, :] - targets[:, None]).argmin(axis=1)]
    bands: npt.NDArray[np.int64] = np.unique(picked).astype(np.int64)
    if bands.size != k:
        raise ValueError(
            f"k={k} in [{lo_nm}, {upper}] nm maps two targets onto one band; "
            "the spacing is finer than the instrument's"
        )
    return bands


@dataclass(frozen=True)
class BandSet:
    """One finalist set and where it was written."""

    name: str
    bands: npt.NDArray[np.int64]
    wavelengths: npt.NDArray[np.float64]
    band_path: Path
    wavelength_path: Path

    @property
    def k(self) -> int:
        return int(self.bands.size)

    def overrides(self) -> list[str]:
        """The Hydra overrides that make a ``train.py`` run read exactly this set."""
        return [
            f"data.band_indices_path={self.band_path}",
            f"data.wavelength_path={self.wavelength_path}",
            f"data.num_bands={self.k}",
            *[f"data.{key}={value}" for key, value in band_augmentation_widths(self.k).items()],
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "bands": [int(b) for b in self.bands],
            "wavelengths_nm": [round(float(w), 6) for w in self.wavelengths],
            "min_nm": float(self.wavelengths.min()),
            "max_nm": float(self.wavelengths.max()),
            "mean_spacing_nm": float(np.diff(self.wavelengths).mean()) if self.k > 1 else 0.0,
            "band_indices_path": str(self.band_path),
            "wavelength_path": str(self.wavelength_path),
            "overrides": self.overrides(),
        }


def write_band_set(
    out_dir: str | Path, name: str, bands: npt.NDArray[Any], wavelengths: npt.NDArray[Any]
) -> BandSet:
    """Write ``<name>.npy`` and ``<name>_wavelengths.csv`` in the band study's format."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(sorted(int(b) for b in bands), dtype=np.int64)
    wl = np.asarray(wavelengths, dtype=np.float64)[idx]
    band_path = root / f"{name}.npy"
    wl_path = root / f"{name}_wavelengths.csv"
    np.save(band_path, idx)
    lines = ["index,Wavelength (nm)"]
    lines += [f"{int(b)},{float(v):.6f}" for b, v in zip(idx, wl, strict=True)]
    wl_path.write_text("\n".join(lines) + "\n")
    return BandSet(
        name=name, bands=idx, wavelengths=wl, band_path=band_path, wavelength_path=wl_path
    )


def write_finalists(
    wavelength_path: str | Path,
    out_dir: str | Path,
    budgets: tuple[int, ...] = FINALIST_BUDGETS,
    lo_nm: float = LOWER_BOUND_NM,
) -> dict[str, Any]:
    """Write every finalist set plus ``manifest.json``; return the manifest.

    The manifest carries the source CSV's sha256, so a set can be traced to the
    wavelength axis it was cut from, and every set's training overrides.
    """
    source = Path(wavelength_path)
    wl = read_wavelengths(source)
    sets = [
        write_band_set(out_dir, f"{NAME_PREFIX}_k{k}", window_uniform_bands(wl, k, lo_nm), wl)
        for k in budgets
    ]
    manifest: dict[str, Any] = {
        "rule": (
            f"k bands spaced evenly in wavelength over the source axis's bands >= {lo_nm} nm "
            "(first to last candidate, gaps in the axis collapsed), nearest band per "
            "target; label-free and fold-independent"
        ),
        "lower_bound_nm": lo_nm,
        "budgets": list(budgets),
        "source_wavelengths": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n_source_bands": int(wl.size),
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "sets": {s.name: s.as_dict() for s in sets},
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
