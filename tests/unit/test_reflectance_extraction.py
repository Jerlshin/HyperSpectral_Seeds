"""End to end: extraction with ``radiometry="tile"`` from an archive laid out like the real one.

The archive is synthetic but structurally faithful — ``<top>/Data-VIS-…/<VARIETY>-0n``
ENVI cubes, one ``black`` dark cube per session, ``wavelengths.csv`` — and every
scene is 640 rows with the seeds above row 600 and a 100 % tile below it, as in
the real 931-row scenes. Three varieties: one imaged in session 1 only, one in
session 2 only, and one whose two bundles span both sessions (the dataset's 17).
The sessions' lamps differ in spectral *shape*, and one scan's lamp is bright
enough to saturate its tile.

What must hold:

* the ``tile`` and ``snv`` extractions are **row-aligned** — identical labels,
  groups, masks, morphology and scan table — because both segment on radiance;
* ``tile`` patches are the seeds' true reflectance;
* the cross-session variety's two bundles agree after tile division and do not
  after SNV — the session fingerprint, removed;
* the QC table, the spectra archive and ``radiometry.json`` are written, the
  saturated scan's bands are filled from its session and marked so, and an
  existing dataset is not overwritten.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("spectral")
pytest.importorskip("cv2")
pytest.importorskip("skimage")

from scipy.ndimage import gaussian_filter  # noqa: E402
from spectral.io import envi  # noqa: E402

from spectralquadnet.data.prep.config import PrepConfig  # noqa: E402
from spectralquadnet.data.prep.patch_extraction import (  # noqa: E402
    build_patch_dataset,
    probe_white_tiles,
)
from spectralquadnet.data.prep.white_tile import SOURCE_SESSION_SHAPE  # noqa: E402

WL = np.array([420.0, 460, 500, 550, 600, 650, 700, 780, 860, 950])
C = WL.size
ROWS, COLS = 640, 64
TOP = "RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties"
SESSIONS = {1: "Data-VIS-20170111-1-room-light-off", 2: "Data-VIS-20170112-1-room-light-off"}
#: variety -> (session of bundle 01, session of bundle 02)
VARIETIES = {"AAA": (1, 1), "BBB": (1, 2), "CCC": (2, 2)}
REFLECTANCE = {
    "AAA": np.linspace(0.30, 0.55, C),
    "BBB": np.linspace(0.55, 0.35, C),
    "CCC": 0.40 + 0.10 * np.sin(np.linspace(0, 3, C)),
}
SEEDS = [(150, 20), (150, 44), (380, 32)]
TILE = (606, 638, 8, 56)
DARK_DN = 190.0


def lamp(session: int, boost: float = 1.0) -> np.ndarray:
    tilt = -0.5 if session == 1 else 0.5
    base = np.exp(-(((WL - 700.0) / 260.0) ** 2)) * (1 + tilt * (WL - WL.mean()) / np.ptp(WL))
    return boost * 3000.0 * base / base.max()


def cube(variety: str, session: int, boost: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Near-black background, as in the real scenes (seeds are bright on dark);
    # `segment`'s 0.4 × Otsu rule is tuned for that contrast.
    rho = np.full((ROWS, COLS, C), 0.01)
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    for cy, cx in SEEDS:
        rho[((yy - cy) / 16.0) ** 2 + ((xx - cx) / 7.0) ** 2 <= 1.0] = REFLECTANCE[variety]
    r0, r1, c0, c1 = TILE
    rho[r0:r1, c0:c1] = 1.0
    # A 1-px optical blur, as a real lens gives: without intermediate edge values
    # the histogram is two spikes, Otsu lands on top of the background spike and
    # `segment`'s 0.4 × Otsu rule then takes the whole frame as foreground.
    rho = gaussian_filter(rho, sigma=(1.0, 1.0, 0.0))
    raw = DARK_DN + lamp(session, boost)[None, None, :] * rho + rng.normal(0, 2.0, rho.shape)
    return np.clip(np.round(raw), 0, 4095).astype(np.uint16)


@pytest.fixture(scope="module")
def archive(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("archive")
    staging = root / "staging"
    members: list[Path] = []
    rng = np.random.default_rng(0)
    for name in SESSIONS.values():
        folder = staging / TOP / name
        folder.mkdir(parents=True)
        dark = np.clip(np.round(DARK_DN + rng.normal(0, 2.0, (20, COLS, C))), 0, 4095)
        envi.save_image(str(folder / "black.hdr"), dark.astype(np.uint16), dtype=np.uint16,
                        interleave="bil", ext=".raw", force=True)  # fmt: skip
        members += [folder / "black.hdr", folder / "black.raw"]
    seed = 1
    for variety, (s1, s2) in VARIETIES.items():
        for bundle, session in ((1, s1), (2, s2)):
            boost = 1.45 if (variety, bundle) == ("CCC", 2) else 1.0
            folder = staging / TOP / SESSIONS[session]
            stem = folder / f"{variety}-0{bundle}"
            envi.save_image(str(stem) + ".hdr", cube(variety, session, boost, seed),
                            dtype=np.uint16, interleave="bil", ext=".raw", force=True)  # fmt: skip
            members += [Path(str(stem) + ".hdr"), Path(str(stem) + ".raw")]
            seed += 1
    wl_csv = staging / TOP / "wavelengths.csv"
    lines = ["index,Wavelength (nm)"] + [f"{i + 1},{w}" for i, w in enumerate(WL)]
    wl_csv.write_text("\n".join(lines) + "\n")
    members.append(wl_csv)
    zpath = root / "rice_hsi.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for m in members:
            zf.write(m, m.relative_to(staging).as_posix())
    return zpath


@pytest.fixture(scope="module")
def built(archive, tmp_path_factory) -> dict[str, PrepConfig]:
    out: dict[str, PrepConfig] = {}
    for mode in ("tile", "snv"):
        cfg = PrepConfig(
            root=tmp_path_factory.mktemp(mode), archive=archive, num_bands=C, radiometry=mode
        )
        build_patch_dataset(cfg)
        out[mode] = cfg
    return out


def test_tile_and_snv_extractions_are_row_aligned(built) -> None:
    tile, snv = built["tile"], built["snv"]
    for name in ("labels", "groups", "morphology", "masks"):
        a = np.load(getattr(tile, f"{name}_path"))
        b = np.load(getattr(snv, f"{name}_path"))
        assert np.array_equal(a, b), name
    assert np.load(tile.labels_path).size == len(VARIETIES) * 2 * len(SEEDS)
    pd.testing.assert_frame_equal(
        pd.read_csv(tile.scan_table_path), pd.read_csv(snv.scan_table_path)
    )


def _interior_spectra(cfg: PrepConfig) -> np.ndarray:
    """Median of each patch's central 4 × 4 pixels — the seed's interior, clear of the blur."""
    patches = np.load(cfg.patches_path)
    return np.median(patches[:, :, 30:34, 30:34].reshape(len(patches), C, -1), axis=2)


def test_tile_patches_are_the_true_reflectance(built) -> None:
    spectra = _interior_spectra(built["tile"])
    table = pd.read_csv(built["tile"].scan_table_path)
    variety = table.set_index("scan_id")["variety"]
    groups = np.load(built["tile"].groups_path)
    for spectrum, g in zip(spectra, groups, strict=True):
        assert np.allclose(spectrum, REFLECTANCE[variety[g]], rtol=0.03), variety[g]


def test_the_session_fingerprint_survives_snv_and_not_the_tile(built) -> None:
    """BBB's two bundles were imaged under two lamp shapes."""
    table = pd.read_csv(built["tile"].scan_table_path).set_index("scan_id")
    groups = np.load(built["tile"].groups_path)
    session = table.loc[groups, "session_id"].to_numpy()
    is_bbb = table.loc[groups, "variety"].to_numpy() == "BBB"
    gaps = {}
    for mode in ("tile", "snv"):
        spectra = _interior_spectra(built[mode])
        a = spectra[is_bbb & (session == session[is_bbb].min())].mean(axis=0)
        b = spectra[is_bbb & (session == session[is_bbb].max())].mean(axis=0)
        gaps[mode] = np.abs(a - b).max() / np.abs(a).max()
    assert gaps["snv"] > 0.2
    assert gaps["tile"] < 0.03


def test_qc_artifacts_record_every_tile_and_every_filled_band(built) -> None:
    cfg = built["tile"]
    qc = pd.read_csv(cfg.white_tiles_path)
    assert len(qc) == len(VARIETIES) * 2 and qc["found"].all()
    spectra = np.load(cfg.white_spectra_path)
    assert spectra["resolved"].shape == (len(qc), C)
    assert np.isfinite(spectra["resolved"]).all()
    saturated = qc.set_index("scan_key").filter(like="CCC-02", axis=0)
    assert int(saturated["n_bands_session_filled"].iloc[0]) > 0
    ccc2 = int(saturated["scan_id"].iloc[0])
    assert (spectra["source"][ccc2] == SOURCE_SESSION_SHAPE).any()
    log = json.loads(cfg.radiometry_log_path.read_text())
    assert log["applied"] == "tile" and log["tiles_found"] == len(qc)
    assert json.loads(built["snv"].radiometry_log_path.read_text())["tile"] is None


def test_an_existing_dataset_is_not_overwritten(built) -> None:
    before = built["tile"].patches_path.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        build_patch_dataset(built["tile"])
    assert built["tile"].patches_path.stat().st_mtime_ns == before
    assert not list(built["tile"].root.glob("*.partial"))


def test_the_probe_measures_without_extracting(archive, tmp_path) -> None:
    cfg = PrepConfig(root=tmp_path, archive=archive, num_bands=C, radiometry="tile")
    table = probe_white_tiles(cfg, limit=3)
    assert len(table) == 3 and table["found"].all()
    assert set(table["session_id"]) == {0, 1}, "round-robin over sessions"
    assert not cfg.patches_path.exists()
