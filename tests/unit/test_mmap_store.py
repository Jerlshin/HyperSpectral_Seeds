"""DataStore invariants.

These tests pin the properties that keep :class:`DataStore` zero-RAM:

* the mapping is opened read-only (``mmap_mode="r"``) and stays an
  :class:`numpy.memmap` — never a materialised array;
* constructing the store twice is a no-op returning the *same object*, so no
  duplicate file handle and no second mmap;
* indexing pages in one patch, and the ``np.array(...)`` copy in
  ``RiceSeedDataset.__getitem__`` does not drag the whole array in.

Everything here runs against a tiny synthetic ``.npy``; checking RSS against
the real multi-GB cube stays a manual step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectralquadnet.data.mmap_store import DataStore

BANDS, SIZE, N = 4, 8, 16


@pytest.fixture
def fresh_store():
    """Drop the process-wide singleton before and after each test."""
    DataStore.reset()
    yield
    DataStore.reset()


@pytest.fixture
def tiny_dataset(tmp_path):
    rng = np.random.default_rng(0)
    patches = rng.standard_normal((N, BANDS, SIZE, SIZE)).astype(np.float32)
    labels = np.arange(N, dtype=np.int64) % 3

    patches_path = tmp_path / "patches.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(patches_path, patches)
    np.save(labels_path, labels)

    wl_path = tmp_path / "wavelengths.csv"
    wl_path.write_text(
        "Band,Wavelength (nm)\n" + "".join(f"{i},{385.0 + i * 15.0}\n" for i in range(BANDS))
    )
    return patches, labels, str(patches_path), str(labels_path), str(wl_path)


def test_second_construction_is_a_noop(fresh_store, tiny_dataset):
    """Re-instantiating must not re-mmap or duplicate the file handle."""
    _, _, patches_path, labels_path, _ = tiny_dataset

    first = DataStore(patches_path=patches_path, labels_path=labels_path)
    first_array = first.patches

    second = DataStore(patches_path=patches_path, labels_path=labels_path)

    assert second is first, "DataStore must be a process-wide singleton"
    assert second.patches is first_array, "the second construction re-opened the mmap"
    assert DataStore.instance() is first


def test_patches_are_a_readonly_memmap(fresh_store, tiny_dataset):
    """``mmap_mode='r'`` — read-only, and never eagerly materialised."""
    _, _, patches_path, labels_path, _ = tiny_dataset
    store = DataStore(patches_path=patches_path, labels_path=labels_path)

    assert isinstance(store.patches, np.memmap)
    assert not store.patches.flags.writeable

    with pytest.raises(ValueError):
        store.patches[0] = 0.0


def test_contents_match_source(fresh_store, tiny_dataset):
    patches, labels, patches_path, labels_path, _ = tiny_dataset
    store = DataStore(patches_path=patches_path, labels_path=labels_path)

    np.testing.assert_array_equal(np.array(store.patches), patches)
    np.testing.assert_array_equal(store.labels, labels)
    assert store.patches.shape == (N, BANDS, SIZE, SIZE)


def test_wavelengths_are_minmax_normalised(fresh_store, tiny_dataset):
    """Reproduces ``_load_wavelengths_to_gpu``: last CSV column, scaled to [0, 1]."""
    *_, wl_path = tiny_dataset
    store = DataStore(wavelength_path=wl_path, device="cpu")

    wl = store.require_wavelengths()
    assert wl.dtype == torch.float32
    assert wl.shape == (BANDS,)
    assert float(wl.min()) == pytest.approx(0.0)
    assert float(wl.max()) == pytest.approx(1.0)


def test_wavelength_load_is_idempotent(fresh_store, tiny_dataset):
    *_, wl_path = tiny_dataset
    store = DataStore(wavelength_path=wl_path, device="cpu")
    first = store.wavelengths
    store.load_wavelengths(wl_path, "cpu")
    assert store.wavelengths is first


def test_require_accessors_raise_before_load(fresh_store):
    store = DataStore()
    for accessor in ("require_patches", "require_labels", "require_wavelengths"):
        with pytest.raises(RuntimeError, match="has not been called"):
            getattr(store, accessor)()


def test_dataset_getitem_copies_one_patch(fresh_store, tiny_dataset):
    """``RiceSeedDataset.__getitem__`` yields an owned copy, not a view on the mmap.

    This is the invariant that bounds RAM: one ``(bands, 64, 64)`` patch is paged
    in per item. A view would keep the mapping alive and, worse, invite an
    accidental batched materialisation upstream.
    """
    from types import SimpleNamespace

    from spectralquadnet.data.datasets import RiceSeedDataset

    patches, labels, patches_path, labels_path, _ = tiny_dataset
    store = DataStore(patches_path=patches_path, labels_path=labels_path)

    ds = RiceSeedDataset(
        np.arange(N),
        aug_strength="none",
        store=store,
        data_cfg=SimpleNamespace(max_cutout_bands=3, noise_std=0.02),
        device="cpu",
    )

    patch, label = ds[3]

    assert patch.shape == (BANDS, SIZE, SIZE)
    assert int(label) == int(labels[3])
    np.testing.assert_array_equal(patch.numpy(), patches[3])
    assert len(ds) == N

    # The returned tensor owns its storage: mutating it cannot reach the mapping.
    patch += 1.0
    np.testing.assert_array_equal(np.array(store.patches[3]), patches[3])


@pytest.mark.requires_dataset
def test_store_reproduces_golden_wavelengths(fresh_store, cfg, physical_wl):
    """The real CSV, read through ``DataStore``, matches the captured golden vector.

    Ties ``tests/regression/golden/physical_wl_spa40.npy`` back to
    ``dataset/wavelengths_spa_40b.csv``, so the regression gates' independence
    from ``dataset/`` is a convenience rather than an unverified assumption.
    """
    from pathlib import Path

    csv = Path(cfg.data.wavelength_path)
    if not csv.exists():
        pytest.skip(f"{csv} not present")

    store = DataStore(wavelength_path=str(csv), device="cpu")
    torch.testing.assert_close(store.require_wavelengths(), physical_wl, rtol=0, atol=0)


# ══════════════════════════════════════════════════════════════════════
#  The band-geometry contract
# ══════════════════════════════════════════════════════════════════════


def _geometry_cfg(paths, **overrides):
    from types import SimpleNamespace

    _, _, patches_path, _, wl_path = paths
    base = dict(
        patches_data=patches_path,
        wavelength_path=wl_path,
        band_indices_path="",
        num_bands=BANDS,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_band_geometry_accepts_a_consistent_full_cube(fresh_store, tiny_dataset) -> None:
    """The primary path's case: every stored band read, λ vector the same length."""
    from spectralquadnet.data.mmap_store import band_geometry

    _, _, patches_path, labels_path, wl_path = tiny_dataset
    store = DataStore(patches_path=patches_path, labels_path=labels_path, wavelength_path=wl_path)

    geometry = band_geometry(_geometry_cfg(tiny_dataset), store)
    assert geometry == {
        "stored": BANDS,
        "selected": BANDS,
        "wavelengths": BANDS,
        "configured": BANDS,
    }
    assert geometry["selected"] == geometry["stored"], "no reduction on the primary path"


def test_band_geometry_refuses_a_num_bands_that_the_cube_contradicts(
    fresh_store, tiny_dataset
) -> None:
    """The failure that used to surface as a shape error deep inside a branch."""
    from spectralquadnet.data.mmap_store import BandGeometryError, band_geometry

    _, _, patches_path, labels_path, wl_path = tiny_dataset
    store = DataStore(patches_path=patches_path, labels_path=labels_path, wavelength_path=wl_path)

    with pytest.raises(BandGeometryError, match="num_bands"):
        band_geometry(_geometry_cfg(tiny_dataset, num_bands=BANDS + 1), store)


def test_band_geometry_refuses_a_mismatched_wavelength_vector(
    fresh_store, tiny_dataset, tmp_path
) -> None:
    """A λ vector describing different bands from the input is silently wrong.

    Every wavelength-dependent operator — the Savitzky-Golay derivatives, the
    continuum hull, Branch D's λ windows — is built from that vector, so this is
    the mismatch that produces a model about nothing rather than a crash.
    """
    from spectralquadnet.data.mmap_store import BandGeometryError, band_geometry

    _, _, patches_path, labels_path, _ = tiny_dataset
    # The cube and `num_bands` agree; only the λ vector is the odd one out, which
    # is the combination no other check would catch.
    short = tmp_path / "short_wavelengths.csv"
    short.write_text(
        "Band,Wavelength (nm)\n" + "".join(f"{i},{385.0 + i * 15.0}\n" for i in range(BANDS - 1))
    )
    store = DataStore(
        patches_path=patches_path, labels_path=labels_path, wavelength_path=str(short)
    )

    with pytest.raises(BandGeometryError, match="wavelength"):
        band_geometry(_geometry_cfg(tiny_dataset), store)


def test_band_geometry_reports_a_reduced_arm_as_reduced(
    fresh_store, tiny_dataset, tmp_path
) -> None:
    """The band-selection pathway's case, and the banner line that names it."""
    from spectralquadnet.data.mmap_store import band_geometry
    from spectralquadnet.engine.pipelines.context import describe_band_geometry

    _, _, patches_path, labels_path, _ = tiny_dataset
    indices = tmp_path / "bands.npy"
    np.save(indices, np.array([0, 2], dtype=np.int64))
    short = tmp_path / "two_wavelengths.csv"
    short.write_text("Band,Wavelength (nm)\n0,385.0\n2,415.0\n")

    store = DataStore(
        patches_path=patches_path, labels_path=labels_path, wavelength_path=str(short)
    )
    geometry = band_geometry(
        _geometry_cfg(tiny_dataset, band_indices_path=str(indices), num_bands=2), store
    )

    assert geometry == {"stored": BANDS, "selected": 2, "wavelengths": 2, "configured": 2}
    assert "REDUCED" in describe_band_geometry(geometry)
    assert "no band selection" in describe_band_geometry(
        {"stored": BANDS, "selected": BANDS, "wavelengths": BANDS, "configured": BANDS}
    )
