"""The finalist band sets (``spectralquadnet.bandstudy.finalists``).

The properties that matter are the ones a training run would otherwise discover
late: every band honours the 430 nm bound, the set has exactly ``k`` distinct
bands spread over the whole usable range, and the two written files pass the
same validators ``train.py`` runs at startup.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from spectralquadnet.bandstudy.finalists import (
    FINALIST_BUDGETS,
    LOWER_BOUND_NM,
    read_wavelengths,
    window_uniform_bands,
    write_finalists,
)
from spectralquadnet.data.datasets import _band_selection, band_augmentation_widths

#: The instrument's axis: 256 bands, 383.22 → 1006.47 nm at ~2.444 nm spacing.
WL = np.linspace(383.222222, 1006.472223, 256)


@pytest.fixture
def wavelength_csv(tmp_path):
    path = tmp_path / "wavelengths.csv"
    rows = ["index,Wavelength (nm)"] + [f"{i + 1},{w:.6f}" for i, w in enumerate(WL)]
    path.write_text("\n".join(rows) + "\n")
    return path


@pytest.mark.parametrize("k", FINALIST_BUDGETS)
def test_every_band_honours_the_bound_and_the_range_is_covered(k: int) -> None:
    bands = window_uniform_bands(WL, k, LOWER_BOUND_NM)
    assert bands.size == k and np.unique(bands).size == k
    assert WL[bands].min() >= LOWER_BOUND_NM
    assert bands[0] == np.flatnonzero(WL >= LOWER_BOUND_NM)[0]
    assert bands[-1] == WL.size - 1


def test_the_nearest_band_below_the_bound_is_excluded() -> None:
    """On the real axis 429.6 nm is nearer to 430 than 432.0 nm is; the window still refuses it."""
    wl = WL.copy()
    below = int(np.flatnonzero(wl < LOWER_BOUND_NM)[-1])
    wl[below] = LOWER_BOUND_NM - 0.1  # make the out-of-window band the nearest one
    bands = window_uniform_bands(wl, 32, LOWER_BOUND_NM)
    assert below not in bands.tolist()
    assert wl[bands[0]] >= LOWER_BOUND_NM


def test_spacing_is_even_to_within_one_band() -> None:
    bands = window_uniform_bands(WL, 32, LOWER_BOUND_NM)
    gaps = np.diff(WL[bands])
    step = float(np.mean(np.diff(WL)))
    assert gaps.max() - gaps.min() <= step + 1e-9


def test_a_budget_finer_than_the_instrument_is_refused() -> None:
    with pytest.raises(ValueError):
        window_uniform_bands(WL, 300, LOWER_BOUND_NM)
    with pytest.raises(ValueError):
        window_uniform_bands(WL, 0, LOWER_BOUND_NM)


def test_written_files_pass_the_training_validators(wavelength_csv, tmp_path) -> None:
    out = tmp_path / "finalists"
    manifest = write_finalists(wavelength_csv, out)
    assert set(manifest["sets"]) == {f"uniform430_k{k}" for k in FINALIST_BUDGETS}
    for entry in manifest["sets"].values():
        k = entry["k"]
        idx = _band_selection(
            SimpleNamespace(band_indices_path=entry["band_indices_path"], num_bands=k)
        )
        assert idx is not None and idx.dtype == np.int64 and idx.size == k
        wl = read_wavelengths(entry["wavelength_path"])
        assert wl.size == k and np.allclose(wl, WL[idx], atol=1e-5)
        widths = band_augmentation_widths(k)
        assert f"data.num_bands={k}" in entry["overrides"]
        assert f"data.cutmix_bands={widths['cutmix_bands']}" in entry["overrides"]
        assert f"data.max_cutout_bands={widths['max_cutout_bands']}" in entry["overrides"]
    on_disk = json.loads((out / "manifest.json").read_text())
    assert len(on_disk["source_sha256"]) == 64
