"""The in-scene white tile (``spectralquadnet.data.prep.white_tile``).

The real scenes are not on this machine, so every scene here is synthetic and
built from the physics the detector relies on: raw = dark + lamp(λ) × ρ, clipped
at the 12-bit ceiling. Seeds (ρ ≈ 0.3-0.6) sit on a dark background (ρ = 0.05)
in the upper rows; a 100 % tile sits in the rows the seed crop discards. Because
the true lamp spectrum and every true reflectance are known, each measurement is
checked against its exact value rather than for plausibility.

The motivating property gets its own test: the same seed imaged under two lamps
with different spectral *shapes* — two sessions — gives two different SNV
spectra, and one reflectance spectrum after tile division.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectralquadnet.data.prep.config import TileConfig
from spectralquadnet.data.prep.radiometry import snv_normalise, tile_reflectance
from spectralquadnet.data.prep.segmentation import dark_correct, dark_frame
from spectralquadnet.data.prep.white_tile import (
    SOURCE_OWN,
    SOURCE_SESSION_GAIN,
    SOURCE_SESSION_SHAPE,
    SOURCE_UNRESOLVED,
    TileMeasurement,
    detect_white_tile,
    resolve_white_references,
)

WL = np.array([420.0, 460, 500, 550, 600, 650, 700, 780, 860, 950])
C = WL.size
ROWS, COLS, SEARCH_FROM = 120, 64, 80
TILE = (90, 118, 8, 56)  # row0, row1, col0, col1 — 28 × 48 = 1,344 px
SEED_REFLECTANCE = np.linspace(0.30, 0.60, C)
DARK_DN = 190.0

CFG = TileConfig(search_rows=(SEARCH_FROM, None))


def lamp(peak: float = 3000.0, tilt: float = 0.0) -> np.ndarray:
    """A smooth lamp spectrum in DN; ``tilt`` changes its spectral *shape*."""
    base = np.exp(-(((WL - 700.0) / 260.0) ** 2))
    shape = base * (1.0 + tilt * (WL - WL.mean()) / np.ptp(WL))
    return peak * shape / shape.max()


def scene(
    lam: np.ndarray, tile: bool = True, seeds: bool = True, noise: float = 2.0, seed: int = 0
):
    """``(raw, dark)`` for one synthetic scene under lamp ``l``."""
    rng = np.random.default_rng(seed)
    rho = np.full((ROWS, COLS, C), 0.05)
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    if seeds:
        for cy, cx in [(20, 16), (20, 46), (55, 30)]:
            inside = ((yy - cy) / 14.0) ** 2 + ((xx - cx) / 6.0) ** 2 <= 1.0
            rho[inside] = SEED_REFLECTANCE
    if tile:
        r0, r1, c0, c1 = TILE
        rho[r0:r1, c0:c1] = 1.0
    dark_col = DARK_DN + rng.normal(0, 1.5, (1, COLS, C))
    raw = dark_col + lam[None, None, :] * rho + rng.normal(0, noise, (ROWS, COLS, C))
    raw = np.clip(np.round(raw), 0, 4095)
    dark = dark_col + rng.normal(0, noise, (20, COLS, C))
    return raw.astype(np.float32), dark.astype(np.float32)


def measure(lam: np.ndarray, **kw) -> TileMeasurement:
    raw, dark = scene(lam, **kw)
    return detect_white_tile(raw, dark_frame(dark), WL, CFG)


# ══════════════════════════════════════════════════════════════════════
#  Detection and measurement
# ══════════════════════════════════════════════════════════════════════


def test_the_tile_is_found_where_it_is_and_measures_the_lamp() -> None:
    lam = lamp()
    m = measure(lam)
    assert m.found, m.reason
    r0, c0, r1, c1 = m.bbox
    assert abs(r0 - TILE[0]) <= 1 and abs(r1 - TILE[1]) <= 1
    assert abs(c0 - TILE[2]) <= 1 and abs(c1 - TILE[3]) <= 1
    assert m.n_core_px < m.n_region_px, "the edge is eroded before measuring"
    assert np.allclose(m.spectrum, lam, rtol=0.01)
    assert m.n_saturated_bands == 0


def test_saturated_bands_are_left_unmeasured_not_clipped() -> None:
    """A clipped white value would inflate every reflectance in its band."""
    lam = lamp(peak=4500.0)
    m = measure(lam)
    assert m.found, m.reason
    clipped = lam + DARK_DN >= 4095
    assert clipped.any() and not clipped.all()
    assert np.isnan(m.spectrum[clipped]).all()
    assert (m.saturated_frac[clipped] > 0.5).all()
    assert np.allclose(m.spectrum[~clipped], lam[~clipped], rtol=0.01)


def test_no_tile_is_reported_not_invented() -> None:
    m = measure(lamp(), tile=False)
    assert not m.found
    assert np.isnan(m.spectrum).all()


def test_seeds_are_never_mistaken_for_a_tile() -> None:
    """Search the whole scene with no tile in it: the seeds are the brightest thing there."""
    raw, dark = scene(lamp(), tile=False)
    whole = TileConfig(search_rows=(0, None))
    m = detect_white_tile(raw, dark_frame(dark), WL, whole)
    assert not m.found
    assert "compact region" in m.reason


def test_a_non_uniform_bright_patch_is_rejected() -> None:
    """Two abutting bright surfaces (a stripe beside a reflector) are not one Lambertian panel."""
    raw, dark = scene(lamp(), tile=False)
    r0, r1, c0, c1 = TILE
    mid = (c0 + c1) // 2
    raw[r0:r1, c0:mid] = DARK_DN + 0.70 * lamp()[None, None, :]
    raw[r0:r1, mid:c1] = DARK_DN + 1.00 * lamp()[None, None, :]
    m = detect_white_tile(raw, dark_frame(dark), WL, CFG)
    assert not m.found
    assert "uniform" in m.reason
    assert m.bbox is not None and np.isnan(m.spectrum).all(), "stats kept, spectrum withheld"


def test_the_wavelength_axis_must_match_the_scene() -> None:
    raw, dark = scene(lamp())
    with pytest.raises(ValueError):
        detect_white_tile(raw, dark_frame(dark), WL[:-1], CFG)


# ══════════════════════════════════════════════════════════════════════
#  Reflectance — and the session effect it removes
# ══════════════════════════════════════════════════════════════════════


def _seed_pixels(cube: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    inside = ((yy - 20) / 10.0) ** 2 + ((xx - 16) / 4.0) ** 2 <= 1.0
    return cube[inside]


def test_tile_division_recovers_the_seed_reflectance() -> None:
    raw, dark = scene(lamp())
    m = detect_white_tile(raw, dark_frame(dark), WL, CFG)
    rho = tile_reflectance(dark_correct(raw, dark), m.spectrum)
    assert np.allclose(np.median(_seed_pixels(rho), axis=0), SEED_REFLECTANCE, rtol=0.02)


def test_snv_keeps_the_lamp_shape_and_the_tile_removes_it() -> None:
    """Two sessions, one seed. SNV cancels a scalar gain, not a spectral tilt."""
    per_session_snv, per_session_rho = [], []
    for tilt, peak in [(-0.6, 2600.0), (0.6, 3400.0)]:
        raw, dark = scene(lamp(peak=peak, tilt=tilt))
        corrected = dark_correct(raw, dark)
        m = detect_white_tile(raw, dark_frame(dark), WL, CFG)
        snv, _, _ = snv_normalise(corrected)
        per_session_snv.append(np.median(_seed_pixels(snv), axis=0))
        per_session_rho.append(
            np.median(_seed_pixels(tile_reflectance(corrected, m.spectrum)), axis=0)
        )
    snv_gap = np.abs(per_session_snv[0] - per_session_snv[1]).max()
    rho_gap = np.abs(per_session_rho[0] - per_session_rho[1]).max() / SEED_REFLECTANCE.max()
    assert snv_gap > 0.2, "the lamp tilt survives SNV — the session fingerprint"
    assert rho_gap < 0.02, "and does not survive the tile division"


def test_reflectance_refuses_an_unusable_white_spectrum() -> None:
    corrected = np.ones((2, 2, C), dtype=np.float32)
    bad = lamp()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        tile_reflectance(corrected, bad)
    with pytest.raises(ValueError):
        tile_reflectance(corrected, lamp()[:-1])


# ══════════════════════════════════════════════════════════════════════
#  From measurements to one white reference per scan
# ══════════════════════════════════════════════════════════════════════


def _m(spectrum: np.ndarray | None) -> TileMeasurement:
    if spectrum is None:
        return TileMeasurement(
            found=False,
            reason="none",
            n_bands=C,
            spectrum=np.full(C, np.nan),
            saturated_frac=np.full(C, np.nan),
        )
    return TileMeasurement(
        found=True, reason="ok", n_bands=C, spectrum=spectrum, saturated_frac=np.zeros(C)
    )


def test_a_saturated_band_is_filled_from_the_session_at_the_scans_own_gain() -> None:
    lam = lamp()
    bright = 2.0 * lam
    bright[[6, 7]] = np.nan  # saturated in this scan only
    refs = resolve_white_references([_m(bright), _m(lam)], [0, 0], C)
    assert refs.ok
    assert np.allclose(refs.spectra[0], 2.0 * lam)
    assert np.allclose(refs.spectra[1], lam)
    assert (refs.source[0, [6, 7]] == SOURCE_SESSION_SHAPE).all()
    assert (np.delete(refs.source[0], [6, 7]) == SOURCE_OWN).all()


def test_a_scan_without_a_tile_takes_the_session_median() -> None:
    lam = lamp()
    refs = resolve_white_references([_m(lam), _m(3.0 * lam), _m(None)], [0, 0, 0], C)
    assert refs.ok
    assert (refs.source[2] == SOURCE_SESSION_GAIN).all()
    gains = [np.median(lam), np.median(3.0 * lam)]
    assert np.allclose(refs.spectra[2], np.median(gains) * lam / np.median(lam))


def test_sessions_never_borrow_from_each_other() -> None:
    """A session with no tile at all is unresolved, however well another session measured."""
    refs = resolve_white_references([_m(lamp()), _m(None)], [0, 1], C)
    assert not refs.ok
    assert 1 in refs.unresolved and 0 not in refs.unresolved
    assert (refs.source[1] == SOURCE_UNRESOLVED).all()


def test_a_band_nobody_measured_is_unresolved_not_interpolated() -> None:
    a, b = lamp(), 1.5 * lamp()
    a[4] = b[4] = np.nan
    refs = resolve_white_references([_m(a), _m(b)], [0, 0], C)
    assert refs.unresolved == {0: [4], 1: [4]}
    assert refs.counts()["unresolved"] == 2
