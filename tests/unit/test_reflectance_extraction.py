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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("spectral")
pytest.importorskip("cv2")
pytest.importorskip("skimage")

from scipy.ndimage import gaussian_filter  # noqa: E402
from spectral.io import envi  # noqa: E402

from spectralquadnet.data.prep.config import PrepConfig, TileConfig  # noqa: E402
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


def _write_archive(root: Path, boost_of: Callable[[str, int, int], float]) -> Path:
    """The synthetic archive; ``boost_of(variety, bundle, session)`` scales that scan's lamp."""
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
            boost = boost_of(variety, bundle, session)
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
def archive(tmp_path_factory) -> Path:
    """One scan (CCC-02) saturates its tile; its session's other scan does not."""
    return _write_archive(
        tmp_path_factory.mktemp("archive"),
        lambda variety, bundle, session: 1.45 if (variety, bundle) == ("CCC", 2) else 1.0,
    )


#: Bands a 1.45x session-1 lamp clips on the tile: 190 + 1.45 * lamp >= 4095 DN
#: at 600, 650 and 700 nm (4267, 4540, 4490 DN) and nowhere else (<= 3791 DN).
CLIPPED_IN_SESSION_1 = [4, 5, 6]


@pytest.fixture(scope="module")
def clipped_archive(tmp_path_factory) -> Path:
    """Every scan of session 1 saturates its tile in the same bands — the real archive's case."""
    return _write_archive(
        tmp_path_factory.mktemp("clipped_archive"),
        lambda variety, bundle, session: 1.45 if session == 1 else 1.0,
    )


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


@pytest.mark.parametrize("n_rows", [17280, 17279, 9999, 1])
def test_a_short_stream_is_truncated_in_place(tmp_path, n_rows) -> None:
    """Trimming uncounted rows rewrites the header and truncates — no second copy of the cube."""
    from spectralquadnet.data.prep.patch_extraction import _finalise_stream, _open_stream

    path = tmp_path / "patches.npy"
    stream = _open_stream(path, (17280, 3, 2, 2), np.float32)
    stream[:] = np.arange(stream.size, dtype=np.float32).reshape(stream.shape)
    expected = np.array(stream[:n_rows])
    inode = (tmp_path / "patches.npy.partial").stat().st_ino

    _finalise_stream(stream, path, n_rows)

    assert not (tmp_path / "patches.npy.partial").exists()
    assert path.stat().st_ino == inode
    out = np.load(path, mmap_mode="r")
    assert out.shape == (n_rows, 3, 2, 2)
    np.testing.assert_array_equal(out, expected)
    assert path.stat().st_size == out.offset + expected.nbytes


def test_a_band_no_session_tile_measured_is_refused_by_default(clipped_archive, tmp_path) -> None:
    cfg = PrepConfig(root=tmp_path, archive=clipped_archive, num_bands=C, radiometry="tile")
    with pytest.raises(RuntimeError, match="--tile-drop-unresolved"):
        build_patch_dataset(cfg)
    assert not cfg.patches_path.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert cfg.white_tiles_path.exists(), "the QC table is written before refusing"


def test_unresolved_bands_are_dropped_from_every_scan_on_request(clipped_archive, tmp_path) -> None:
    cfg = PrepConfig(
        root=tmp_path,
        archive=clipped_archive,
        num_bands=C,
        radiometry="tile",
        tile=TileConfig(drop_unresolved_bands=True),
    )
    build_patch_dataset(cfg)
    keep = np.setdiff1d(np.arange(C), CLIPPED_IN_SESSION_1)

    patches = np.load(cfg.patches_path, mmap_mode="r")
    assert patches.shape == (len(VARIETIES) * 2 * len(SEEDS), keep.size, 64, 64)
    wl = pd.read_csv(cfg.wavelengths_path)
    assert list(wl.columns) == ["index", "Wavelength (nm)"]
    assert wl["index"].tolist() == (keep + 1).tolist()
    np.testing.assert_allclose(wl["Wavelength (nm)"], WL[keep])

    log = json.loads(cfg.radiometry_log_path.read_text())
    assert log["tile"]["drop_unresolved_bands"] is True
    assert log["bands"]["n_kept"] == keep.size
    assert log["bands"]["dropped_instrument_index"] == CLIPPED_IN_SESSION_1
    np.testing.assert_allclose(log["bands"]["dropped_nm"], WL[CLIPPED_IN_SESSION_1])

    # Every scan — session 2's too, whose tile measured these bands — is still
    # true reflectance on the kept axis.
    spectra = np.median(patches[:, :, 30:34, 30:34].reshape(len(patches), keep.size, -1), axis=2)
    variety = pd.read_csv(cfg.scan_table_path).set_index("scan_id")["variety"]
    for spectrum, g in zip(spectra, np.load(cfg.groups_path), strict=True):
        assert np.allclose(spectrum, REFLECTANCE[variety[g]][keep], rtol=0.03), variety[g]


def test_the_full_axis_is_written_when_nothing_is_dropped(built) -> None:
    for cfg in built.values():
        wl = pd.read_csv(cfg.wavelengths_path)
        assert wl["index"].tolist() == list(range(1, C + 1))
        assert (
            json.loads(cfg.radiometry_log_path.read_text())["bands"]["dropped_instrument_index"]
            == []
        )
