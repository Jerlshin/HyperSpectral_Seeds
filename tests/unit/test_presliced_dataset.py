"""The pre-sliced cube: faithful to its source, verifiable, and honest about being reduced."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from spectralquadnet.data.mmap_store import (
    BandGeometryError,
    DataStore,
    acquired_bands,
    band_geometry,
)
from spectralquadnet.data.prep.preslice import (
    BAND_AXIS_FILE,
    MANIFEST_FILE,
    PresliceError,
    build_presliced,
    sha256_file,
    verify_presliced,
)

N, BANDS, SIZE = 12, 10, 8
KEEP = [1, 4, 7]


@pytest.fixture
def fresh_store():
    DataStore.reset()
    yield
    DataStore.reset()


@pytest.fixture
def source(tmp_path):
    """A miniature ``./dataset/`` with every file ``prepare_dataset.py`` writes that matters."""
    root = tmp_path / "dataset"
    root.mkdir()
    rng = np.random.default_rng(0)
    patches = rng.uniform(0.0, 1.2, size=(N, BANDS, SIZE, SIZE)).astype(np.float32)
    patches[:, :, :2, :] = 0.0  # exact-zero background
    np.save(root / "patches.npy", patches)
    np.save(root / "labels.npy", np.arange(N, dtype=np.int64) % 3)
    np.save(root / "groups.npy", np.arange(N, dtype=np.int64) % 6)
    np.save(root / "masks.npy", np.ones((N, SIZE, SIZE), dtype=np.float16))
    np.save(root / "morphology.npy", rng.normal(size=(N, 8)).astype(np.float32))
    (root / "wavelengths.csv").write_text(
        "index,Wavelength (nm)\n"
        + "".join(f"{i + 1},{400.0 + 20.0 * i:.6f}\n" for i in range(BANDS))
    )
    (root / "scan_table.csv").write_text("scan_id,session\n0,a\n")
    (root / "radiometry.json").write_text('{"applied": "tile"}\n')
    return root, patches


def test_a_float16_build_is_the_source_bands_within_the_cast_bound(source, tmp_path) -> None:
    root, patches = source
    out = tmp_path / "sliced"
    manifest = build_presliced(root, out, KEEP, set_name="test_k3")

    cube = np.load(out / "patches.npy", mmap_mode="r")
    assert cube.shape == (N, len(KEEP), SIZE, SIZE) and cube.dtype == np.float16
    expected = patches[:, KEEP]
    np.testing.assert_allclose(cube.astype(np.float32), expected, rtol=2**-11, atol=0)
    assert ((cube == 0) == (expected == 0)).all(), "the zero background stays exactly zero"

    axis = json.loads((out / BAND_AXIS_FILE).read_text())
    assert axis["source_n_bands"] == BANDS
    assert axis["source_band_indices"] == KEEP
    assert axis["source_wavelengths_sha256"] == sha256_file(root / "wavelengths.csv")
    assert axis["cast_error"]["max_rel_normal_range"] <= 2**-11
    assert axis["wavelengths_nm"] == [420.0, 480.0, 540.0]

    wl_rows = (out / "wavelengths.csv").read_text().splitlines()
    assert wl_rows[1:] == ["2,420.0", "5,480.0", "8,540.0"], "instrument band numbers kept"
    for name in ("labels.npy", "groups.npy", "masks.npy", "morphology.npy"):
        np.testing.assert_array_equal(np.load(out / name), np.load(root / name))
    assert set(manifest["files"]) >= {"patches.npy", BAND_AXIS_FILE, "scan_table.csv"}
    assert verify_presliced(out) == []


def test_a_float32_build_is_bit_identical(source, tmp_path) -> None:
    root, patches = source
    build_presliced(root, tmp_path / "f32", KEEP, set_name="t", storage_dtype="float32")
    np.testing.assert_array_equal(np.load(tmp_path / "f32" / "patches.npy"), patches[:, KEEP])


def test_verify_catches_a_corrupted_upload(source, tmp_path) -> None:
    root, _ = source
    out = tmp_path / "sliced"
    build_presliced(root, out, KEEP, set_name="t")
    with (out / "labels.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x7f")
    assert verify_presliced(out, checksums=False) == [], "same size: only the hash can tell"
    assert any("SHA-256" in p for p in verify_presliced(out))


@pytest.mark.parametrize(
    ("indices", "kwargs", "match"),
    [
        ([1, 1, 2], {}, "duplicates"),
        ([0, BANDS], {}, "must lie in"),
        (KEEP, {"expected_source_sha256": "0" * 64}, "different|SHA-256"),
        (KEEP, {"storage_dtype": "int8"}, "storage_dtype"),
    ],
)
def test_a_subset_that_cannot_be_written_faithfully_is_refused(
    source, tmp_path, indices, kwargs, match
) -> None:
    root, _ = source
    with pytest.raises(PresliceError, match=match):
        build_presliced(root, tmp_path / "bad", indices, set_name="t", **kwargs)


def test_a_finished_build_is_never_overwritten(source, tmp_path) -> None:
    root, _ = source
    build_presliced(root, tmp_path / "sliced", KEEP, set_name="t")
    with pytest.raises(PresliceError, match=MANIFEST_FILE):
        build_presliced(root, tmp_path / "sliced", KEEP, set_name="t")


def _store_and_cfg(out):
    store = DataStore(
        patches_path=str(out / "patches.npy"),
        labels_path=str(out / "labels.npy"),
        wavelength_path=str(out / "wavelengths.csv"),
    )
    cfg = SimpleNamespace(
        patches_data=str(out / "patches.npy"),
        wavelength_path=str(out / "wavelengths.csv"),
        band_indices_path="",
        num_bands=len(KEEP),
    )
    return store, cfg


def test_a_presliced_cube_reports_itself_as_a_reduced_arm(fresh_store, source, tmp_path) -> None:
    """All four counts say 3 — only the provenance knows the spectrum had 10 bands."""
    from spectralquadnet.engine.pipelines.context import describe_band_geometry

    root, _ = source
    out = tmp_path / "sliced"
    build_presliced(root, out, KEEP, set_name="t")
    store, cfg = _store_and_cfg(out)

    geometry = band_geometry(cfg, store)
    assert geometry == {
        "stored": 3,
        "selected": 3,
        "wavelengths": 3,
        "configured": 3,
        "acquired": BANDS,
    }
    assert acquired_bands(geometry) == BANDS
    banner = describe_band_geometry(geometry)
    assert "REDUCED" in banner and f"3 of {BANDS}" in banner and "pre-sliced" in banner


def test_provenance_that_does_not_describe_the_cube_fails_the_run(
    fresh_store, source, tmp_path
) -> None:
    root, _ = source
    out = tmp_path / "sliced"
    build_presliced(root, out, KEEP, set_name="t")
    axis = json.loads((out / BAND_AXIS_FILE).read_text())
    axis["source_band_indices"] = [1, 4]
    (out / BAND_AXIS_FILE).write_text(json.dumps(axis))
    store, cfg = _store_and_cfg(out)
    with pytest.raises(BandGeometryError, match="provenance"):
        band_geometry(cfg, store)


def test_the_loader_widens_a_float16_cube_to_float32(fresh_store, source, tmp_path) -> None:
    """Every augmentation and every model input stays float32 whatever the storage."""
    from spectralquadnet.data.datasets import RiceSeedDataset

    root, _ = source
    out = tmp_path / "sliced"
    build_presliced(root, out, KEEP, set_name="t")
    store, _ = _store_and_cfg(out)
    ds = RiceSeedDataset(
        np.arange(N),
        aug_strength="none",
        store=store,
        data_cfg=SimpleNamespace(max_cutout_bands=1, noise_std=0.02),
    )
    cube = np.load(out / "patches.npy")

    patch, _ = ds[5]
    assert patch.dtype.is_floating_point and patch.numpy().dtype == np.float32
    np.testing.assert_array_equal(patch.numpy(), cube[5].astype(np.float32))

    batch = ds.__getitems__([0, 5])
    assert all(item[0].numpy().dtype == np.float32 for item in batch)
    np.testing.assert_array_equal(batch[1][0].numpy(), cube[5].astype(np.float32))


def test_a_batched_band_subset_reads_the_same_values_as_the_per_item_path(
    fresh_store, source, tmp_path
) -> None:
    """The outer-product read must equal slicing each patch."""
    from spectralquadnet.data.datasets import RiceSeedDataset

    root, patches = source
    np.save(tmp_path / "bands.npy", np.array(KEEP, dtype=np.int64))
    store = DataStore(patches_path=str(root / "patches.npy"), labels_path=str(root / "labels.npy"))
    ds = RiceSeedDataset(
        np.arange(N),
        aug_strength="none",
        store=store,
        data_cfg=SimpleNamespace(
            max_cutout_bands=1,
            noise_std=0.02,
            band_indices_path=str(tmp_path / "bands.npy"),
            num_bands=len(KEEP),
        ),
    )
    rows = [7, 2, 9]
    batched = ds.__getitems__(rows)
    for row, item in zip(rows, batched, strict=True):
        np.testing.assert_array_equal(item[0].numpy(), patches[row, KEEP])
        np.testing.assert_array_equal(ds[row][0].numpy(), patches[row, KEEP])
