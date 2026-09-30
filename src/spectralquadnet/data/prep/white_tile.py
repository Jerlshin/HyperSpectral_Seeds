"""The in-scene Spectralon white tile: find it, measure it, and turn it into a white reference.

Why this exists
───────────────
The Zenodo record (3241923) states that *"each HSI image contains in the scene a
100 % reflective spectralon tile"*. The original extraction never used it:
:func:`~spectralquadnet.data.prep.radiometry.find_white_reference` looks for a
separate white-panel *file*, finds none, and the pipeline falls back to
per-pixel SNV — and ``preprocess_raw`` crops every scene to its first 600 rows,
which is where the extraction notebook notes the white calibration lives
("Remove white calibration + bottom stripes"). SNV removes a scalar gain per
pixel but not the *spectral shape* of the illumination, and the September 2026
band study found exactly that shape acting as a session fingerprint: models on
SNV spectra recognised the acquisition session rather than the variety. Dividing
each scene by its own tile's spectrum removes the lamp-and-detector spectral
shape of that scene, which SNV cannot.

What is measured
────────────────
:func:`detect_white_tile` works on the **uncropped** raw scene (the raw digital
numbers, so saturation is visible) and its dark frame:

1. a brightness image — the dark-corrected mean over a high-SNR band window —
   inside the search rows (by default the rows the seed crop discards);
2. candidate pixels above a fraction of the region's robust maximum, lightly
   opened, and split into connected components;
3. the brightest component that is large (bigger than any seed), compact
   (solidity) and far brighter than its surroundings (contrast);
4. its core — the component eroded by a few pixels, so edge and penumbra pixels
   are excluded — must be spatially uniform;
5. per band, the median dark-corrected value over the core pixels that are
   **not saturated** in that band. A band where more than a small share of core
   pixels hit the 12-bit ceiling (4095 DN) is *not measured* — a clipped white
   value would inflate every reflectance in that band — and is recorded as such.

The raw archive shows the ceiling is reached: the maximum raw value is 4095 in
at least 75 % of the 180 cubes, and the tile is the brightest object in a scene.

From measurements to white references
─────────────────────────────────────
:func:`resolve_white_references` turns one measurement per scan into one usable
white spectrum per scan. A scan keeps its own measured value in every band it
could measure. Bands it could not — saturated, or no tile found at all — are
filled from its **session**: the session's illumination spectral *shape* is the
median of its measured scans' spectra, each normalised by its own gain over the
bands they all measured, and it is scaled back to the scan by the scan's own
gain (or, with no tile at all, by the session's median gain). That rests on one
stated assumption — scans within a session share the lamp's spectral shape and
differ at most by a scalar intensity — and every filled value is marked, so how
much of the dataset rests on it is a number in the QC table rather than a guess.
A band no scan in the session could measure is **unresolved**, and the
extraction refuses to start rather than invent a value.

Limits, stated
──────────────
* The tile spans only some sensor columns, so it corrects the illumination's
  spectral shape per scene, not the push-broom column-to-column gain (flat
  field). That residual is the same in every session and is not a session
  fingerprint.
* The thresholds in :class:`TileConfig` are informed by the scene geometry
  (931 × 336 × 256, seeds 300-800 px) but have not been fitted to real tile
  pixels, which are not on this machine. ``scripts/prepare_dataset.py
  --probe-tiles N`` measures N real scans and writes the QC table **before**
  any extraction, and that table is what should confirm or retune them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_erosion, binary_opening
from skimage.measure import label, regionprops

from spectralquadnet.data.prep.config import TileConfig

#: ``WhiteReferences.source`` codes.
SOURCE_OWN: int = 0
SOURCE_SESSION_SHAPE: int = 1  # band filled from the session shape, scaled by the scan's own gain
SOURCE_SESSION_GAIN: int = 2  # no tile in this scan: session shape × session median gain
SOURCE_UNRESOLVED: int = 3


@dataclass
class TileMeasurement:
    """What :func:`detect_white_tile` found in one scene."""

    found: bool
    reason: str
    n_bands: int
    #: ``(row0, col0, row1, col1)`` in **scene** coordinates, or ``None``.
    bbox: tuple[int, int, int, int] | None = None
    n_region_px: int = 0
    n_core_px: int = 0
    contrast: float = 0.0
    core_cv: float = 0.0
    #: ``(C,)`` dark-corrected median DN per band; NaN where not measured.
    spectrum: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    #: ``(C,)`` share of core pixels at the saturation ceiling, per band.
    saturated_frac: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    @property
    def n_measured_bands(self) -> int:
        return int(np.isfinite(self.spectrum).sum()) if self.found else 0

    @property
    def n_saturated_bands(self) -> int:
        return int((self.saturated_frac > 0).sum()) if self.found else 0

    def as_row(self) -> dict[str, Any]:
        """One QC-table row."""
        r0, c0, r1, c1 = self.bbox if self.bbox else (None, None, None, None)
        return {
            "found": self.found,
            "reason": self.reason,
            "row0": r0,
            "col0": c0,
            "row1": r1,
            "col1": c1,
            "n_region_px": self.n_region_px,
            "n_core_px": self.n_core_px,
            "contrast": round(self.contrast, 4),
            "core_cv": round(self.core_cv, 5),
            "n_measured_bands": self.n_measured_bands,
            "n_bands_with_saturation": self.n_saturated_bands,
            "max_saturated_frac": (
                round(float(self.saturated_frac.max()), 5) if self.saturated_frac.size else 0.0
            ),
        }


def _not_found(reason: str, n_bands: int) -> TileMeasurement:
    nan = np.full(n_bands, np.nan)
    return TileMeasurement(
        found=False, reason=reason, n_bands=n_bands, spectrum=nan, saturated_frac=nan.copy()
    )


def detect_white_tile(
    raw: npt.NDArray[Any],
    dark_level: npt.NDArray[Any],
    wavelengths: npt.NDArray[Any],
    cfg: TileConfig | None = None,
) -> TileMeasurement:
    """Find and measure the white tile in one **uncropped** raw scene.

    Args:
        raw: ``(H, W, C)`` raw digital numbers — *not* dark-corrected, so that
            saturation at :attr:`TileConfig.saturation_dn` is visible.
        dark_level: ``(1, W, C)`` or ``(W, C)`` dark frame, e.g.
            :func:`~spectralquadnet.data.prep.segmentation.dark_frame`.
        wavelengths: ``(C,)`` band centres in nm.
        cfg: Thresholds; defaults to :class:`TileConfig`.

    Returns:
        A :class:`TileMeasurement`. ``found=False`` carries the reason, and a
        found tile may still have unmeasured (NaN) bands where it saturated.
    """
    cfg = cfg or TileConfig()
    raw = np.asarray(raw)
    n_rows, _, n_bands = raw.shape
    wl = np.asarray(wavelengths, dtype=np.float64)
    if wl.size != n_bands:
        raise ValueError(f"{wl.size} wavelengths for a {n_bands}-band scene")

    r_start = max(0, int(cfg.search_rows[0]))
    r_stop = n_rows if cfg.search_rows[1] is None else min(n_rows, int(cfg.search_rows[1]))
    if r_stop - r_start < 3:
        return _not_found(
            f"search rows [{r_start}, {r_stop}) are empty in a {n_rows}-row scene", n_bands
        )

    dark = np.asarray(dark_level, dtype=np.float32).reshape(1, raw.shape[1], n_bands)
    window = raw[r_start:r_stop].astype(np.float32)
    corrected = np.clip(window - dark, 0.0, None)

    in_window = (wl >= cfg.brightness_nm[0]) & (wl <= cfg.brightness_nm[1])
    if not in_window.any():
        in_window = np.ones(n_bands, dtype=bool)
    brightness = corrected[:, :, in_window].mean(axis=2)

    top = float(np.percentile(brightness, 99.5))
    if top <= 0:
        return _not_found("the searched rows are dark", n_bands)
    candidates = binary_opening(brightness > cfg.candidate_frac * top, iterations=1)
    if not candidates.any():
        return _not_found("no candidate pixels", n_bands)

    labeled = label(candidates)  # type: ignore[no-untyped-call]
    components = regionprops(labeled)  # type: ignore[no-untyped-call]
    regions = [
        r for r in components if r.area >= cfg.min_area_px and r.solidity >= cfg.min_solidity
    ]
    if not regions:
        largest = max(int(r.area) for r in components)
        return _not_found(
            f"no compact region of >= {cfg.min_area_px} px (largest candidate {largest} px)",
            n_bands,
        )

    best = max(regions, key=lambda r: float(np.median(brightness[labeled == r.label])))
    region = labeled == best.label
    core = binary_erosion(region, iterations=cfg.erode_px) if cfg.erode_px > 0 else region
    n_core = int(core.sum())
    r0, c0, r1, c1 = (int(v) for v in best.bbox)
    bbox = (r0 + r_start, c0, r1 + r_start, c1)
    if n_core < cfg.min_core_px:
        return TileMeasurement(
            found=False,
            reason=f"core of {n_core} px after {cfg.erode_px}-px erosion < {cfg.min_core_px}",
            n_bands=n_bands,
            bbox=bbox,
            n_region_px=int(best.area),
            n_core_px=n_core,
            spectrum=np.full(n_bands, np.nan),
            saturated_frac=np.full(n_bands, np.nan),
        )

    core_b = brightness[core]
    median_b = float(np.median(core_b))
    cv = float(np.median(np.abs(core_b - median_b)) / max(median_b, 1e-9))
    outside = brightness[~region]
    contrast = median_b / max(float(np.median(outside)), 1e-9) if outside.size else np.inf

    raw_core = window[core]  # (n_core, C) raw DN
    cor_core = corrected[core]
    saturated = raw_core >= cfg.saturation_dn
    sat_frac = saturated.mean(axis=0).astype(np.float64)
    spectrum = np.full(n_bands, np.nan)
    for c in range(n_bands):
        if sat_frac[c] <= cfg.max_saturated_frac:
            ok = ~saturated[:, c]
            value = float(np.median(cor_core[ok, c])) if ok.any() else 0.0
            if value > 0:
                spectrum[c] = value

    region_stats: dict[str, Any] = dict(
        n_bands=n_bands,
        bbox=bbox,
        n_region_px=int(best.area),
        n_core_px=n_core,
        contrast=float(contrast),
        core_cv=cv,
    )
    rejected = None
    if contrast < cfg.min_contrast:
        rejected = f"contrast {contrast:.2f} < {cfg.min_contrast}"
    elif cv > cfg.max_core_cv:
        rejected = f"core not uniform (cv {cv:.3f} > {cfg.max_core_cv})"
    if rejected is not None:
        # The region's statistics stay, for the QC table; its spectrum does not,
        # so nothing downstream can mistake a rejected candidate for a white level.
        return TileMeasurement(
            found=False,
            reason=rejected,
            spectrum=np.full(n_bands, np.nan),
            saturated_frac=np.full(n_bands, np.nan),
            **region_stats,
        )
    return TileMeasurement(
        found=True, reason="ok", spectrum=spectrum, saturated_frac=sat_frac, **region_stats
    )


# ══════════════════════════════════════════════════════════════════════
#  From one measurement per scan to one white reference per scan
# ══════════════════════════════════════════════════════════════════════


@dataclass
class WhiteReferences:
    """One white spectrum per scan, with where every value came from.

    Attributes:
        spectra: ``(n_scans, C)`` white level in dark-corrected DN; NaN where
            unresolved.
        source: ``(n_scans, C)`` int8, one of the ``SOURCE_*`` codes.
        unresolved: ``{scan_index: [bands]}`` that no measurement could supply.
    """

    spectra: npt.NDArray[np.float64]
    source: npt.NDArray[np.int8]
    unresolved: dict[int, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unresolved

    def counts(self) -> dict[str, int]:
        """How many (scan, band) values came from each source."""
        return {
            "own": int((self.source == SOURCE_OWN).sum()),
            "session_shape": int((self.source == SOURCE_SESSION_SHAPE).sum()),
            "session_gain": int((self.source == SOURCE_SESSION_GAIN).sum()),
            "unresolved": int((self.source == SOURCE_UNRESOLVED).sum()),
        }


def resolve_white_references(
    measurements: Sequence[TileMeasurement],
    session_ids: Sequence[int],
    n_bands: int,
) -> WhiteReferences:
    """Fill every scan's unmeasured bands from its session; see the module docstring.

    Args:
        measurements: One per scan, in scan order.
        session_ids: The session of each scan, aligned with ``measurements``.
        n_bands: Bands per spectrum.
    """
    if len(measurements) != len(session_ids):
        raise ValueError("one session id per measurement")
    n = len(measurements)
    spectra = np.full((n, n_bands), np.nan)
    source = np.full((n, n_bands), SOURCE_UNRESOLVED, dtype=np.int8)
    sessions = np.asarray(session_ids)

    for s in np.unique(sessions).tolist():
        members = np.flatnonzero(sessions == s)
        measured = [int(i) for i in members if measurements[i].found]
        if not measured:
            continue
        stack = np.stack([measurements[i].spectrum for i in measured])  # (m, C)
        valid_all = np.isfinite(stack).all(axis=0)
        ref_bands = valid_all if valid_all.sum() >= 3 else np.isfinite(stack).mean(axis=0) >= 0.5
        gains = np.array(
            [
                np.nanmedian(stack[j, ref_bands]) if ref_bands.any() else np.nan
                for j in range(len(measured))
            ]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            shapes = stack / gains[:, None]
        finite_shapes = np.isfinite(shapes)
        shape = np.full(n_bands, np.nan)
        for c in range(n_bands):
            if finite_shapes[:, c].any():
                shape[c] = float(np.median(shapes[finite_shapes[:, c], c]))
        session_gain = float(np.nanmedian(gains)) if np.isfinite(gains).any() else np.nan

        gain_of = dict(zip(measured, gains.tolist(), strict=True))
        for i in members.tolist():
            own = measurements[i].spectrum if measurements[i].found else np.full(n_bands, np.nan)
            g = gain_of.get(i, session_gain)
            fill_code = SOURCE_SESSION_SHAPE if i in gain_of else SOURCE_SESSION_GAIN
            for c in range(n_bands):
                if np.isfinite(own[c]):
                    spectra[i, c], source[i, c] = own[c], SOURCE_OWN
                elif np.isfinite(shape[c]) and np.isfinite(g):
                    spectra[i, c], source[i, c] = g * shape[c], fill_code

    unresolved = {
        i: np.flatnonzero(source[i] == SOURCE_UNRESOLVED).tolist()
        for i in range(n)
        if (source[i] == SOURCE_UNRESOLVED).any()
    }
    return WhiteReferences(spectra=spectra, source=source, unresolved=unresolved)
