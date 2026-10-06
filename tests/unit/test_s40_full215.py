"""S40: SeedNet v5 on all 215 bands — the configuration, the model and the driver's contracts.

The training code is the hash-pinned S22 path, unchanged. What is new and checked here:
the float16 215-band data config, the v5 network that config builds, the plan's cell
list and regime, the S40 driver's cell-state and output-validation rules, and the
float16 stream the dataset builder hands the unchanged extraction.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.datasets import band_augmentation_widths
from spectralquadnet.data.prep import patch_extraction
from spectralquadnet.models.branches.spatial_cnn import (
    SpectralSpatialStem3D,
    spectral_stride_schedule,
)
from spectralquadnet.models.registry import build_model, parameter_breakdown
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/research/s40_full215_v5.json"
PARENT = ROOT / "configs/research/s22_screening_amendment05.json"
BANDS = 215
#: SeedNet v5 at 215 bands (k32: 2,725,700, the S22 log's count).
V5_PARAMETERS_215 = 2_761_782


def _v5_overrides() -> list[str]:
    parent = json.loads(PARENT.read_text())
    return [
        "data=refl215_f16_grouped",
        *(o for o in parent["v5_overrides"] if not o.startswith("data=")),
    ]


@pytest.fixture(scope="module")
def cfg215():
    return load_experiment_config("experiment/seednet_full256", overrides=_v5_overrides())


@pytest.fixture(scope="module")
def model215(cfg215, physical_wl_full):
    torch.manual_seed(0)
    model = build_model(cfg215, physical_wl_full)
    assert isinstance(model, SpectralSeedNet)
    return model


def _batch(
    n: int = 3, side: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, BANDS, side, side, generator=gen) * 0.6 + 0.05
    mask = torch.ones(n, side, side)
    x[:, :, :8] = 0.0
    mask[:, :8] = 0.0
    return x, torch.arange(n) % 90, mask, torch.randn(n, 8, generator=gen)


# ── configuration ────────────────────────────────────────────────────


def test_the_f16_config_is_the_primary_axis_with_its_band_fraction_widths(cfg215) -> None:
    primary = load_experiment_config(
        "experiment/seednet_full256", overrides=["data=refl215_grouped"]
    )
    assert cfg215.data.num_bands == primary.data.num_bands == BANDS
    widths = band_augmentation_widths(BANDS)
    assert cfg215.data.cutmix_bands == primary.data.cutmix_bands == widths["cutmix_bands"] == 43
    assert (
        cfg215.data.max_cutout_bands
        == primary.data.max_cutout_bands
        == widths["max_cutout_bands"]
        == 16
    )
    assert not str(cfg215.data.band_indices_path or "")
    assert str(cfg215.data.patches_data).endswith("dataset_refl215_f16/patches.npy")
    assert str(cfg215.data.gain_path) == ""
    for key in ("noise_std", "cutmix_spatial", "split_scheme", "calib_frac", "single_group_policy"):
        assert cfg215.data[key] == primary.data[key], key


def test_the_v5_regime_is_untouched(cfg215) -> None:
    k32 = load_experiment_config(
        "experiment/seednet_full256", overrides=json.loads(PARENT.read_text())["v5_overrides"]
    )
    for group in ("single", "model", "evaluation"):
        assert cfg215[group] == k32[group], group
    for key in (
        "grad_clip",
        "clip_partition",
        "ema_decay",
        "weight_decay",
        "tta_spatial",
        "tta_spectral",
    ):
        assert cfg215[key] == k32[key], key
    assert cfg215.model.spectral_descriptor == "snv_morph"
    assert list(cfg215.model.spatial_tail_strides) == [2, 2, 2, 1]


# ── model ────────────────────────────────────────────────────────────


def test_the_stem_reduces_215_bands_without_skipping_one() -> None:
    assert spectral_stride_schedule(BANDS, 8) == (8, 2, 2)
    stem = SpectralSpatialStem3D(BANDS, 192).eval()
    assert stem.kernel_depths == (15, 5, 5) and stem.folded_depth == 7
    assert stem.fold[0].in_channels == 64 * 7
    conv = stem.stage1[0]
    torch.nn.init.constant_(conv.weight, 1.0)
    for band in range(BANDS):
        x = torch.zeros(1, 1, BANDS, 4, 4)
        x[0, 0, band] = 1.0
        with torch.no_grad():
            assert float(conv(x).abs().max()) > 0.0, f"band {band} is never read"


def test_v5_at_215_bands_has_the_pinned_parameter_budget(model215) -> None:
    breakdown = parameter_breakdown(model215)
    assert breakdown["total"] == V5_PARAMETERS_215
    # snv_morph: SNV of the 215-band mean spectrum plus the 8 morphometrics.
    assert model215.spectral.mlp[0].in_features == BANDS + 8


def test_v5_at_215_bands_trains_and_evaluates_with_the_expected_shapes(model215) -> None:
    x, y, mask, morph = _batch()
    model215.train()
    out = model215(x, labels=y, mask=mask, morph=morph)
    assert out["main"].shape == (3, 90) and out["aux_spatial"].shape == (3, 90)
    loss = F.cross_entropy(out["main"], y) + 0.4 * F.cross_entropy(out["aux_spatial"], y)
    loss.backward()
    grads = [p.grad for p in model215.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    model215.zero_grad(set_to_none=True)
    model215.eval()
    with torch.no_grad():
        logits = model215(x, mask=mask, morph=morph)
        assert logits.shape == (3, 90) and torch.isfinite(logits).all()
        # Both pathways reach the logits.
        for branch_mask in (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])):
            assert not torch.allclose(
                model215(x, mask=mask, morph=morph, branch_mask=branch_mask), logits
            )


# ── plan ─────────────────────────────────────────────────────────────


def test_the_plan_authorises_exactly_two_cells_at_seed_zero() -> None:
    if not PLAN.exists():
        pytest.skip("S40 plan not frozen")
    plan = json.loads(PLAN.read_text())
    parent = json.loads(PARENT.read_text())
    assert plan["cells"] == [{"fold": 0, "seed": 0}, {"fold": 1, "seed": 0}]
    assert plan["v5_overrides"] == _v5_overrides()
    assert (
        plan["runtime_overrides"]
        == parent["runtime_overrides"]
        == ["runtime=kaggle_t4x2", "tracking=console_jsonl", "device=cuda"]
    )
    assert plan["partition_plan"] == parent["partition_plan"]
    from spectralquadnet.data.prep.multimodal import sha256

    assert PLAN.with_suffix(".sha256").read_text().split()[0] == sha256(PLAN)
    for name in (
        "scripts/run_complementary_v5.py",
        "scripts/run_full215_v5.py",
        "configs/data/refl215_f16_grouped.yaml",
        "dataset_refl215_f16/MANIFEST.json",
    ):
        assert name in plan["input_hashes"], name


# ── driver ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def driver():
    return runpy.run_path(str(ROOT / "scripts/run_full215_v5.py"))


def test_cell_states_separate_done_interrupted_and_claimed_cells(driver, tmp_path) -> None:
    state = driver["cell_state"]
    out = tmp_path / "f0_s0"
    assert state(out) == "new"
    out.mkdir()
    assert state(out) == "empty"
    (out / "stray.txt").write_text("x")
    assert state(out) == "broken"
    (out / "provenance.json").write_text("{}")
    assert state(out) == "interrupted"
    (out / "results").mkdir()
    assert state(out) == "interrupted"
    (out / "results" / "run.json").write_text("{}")
    assert state(out) == "done"


def test_the_float16_stream_survives_the_unchanged_finaliser(tmp_path) -> None:
    builder = runpy.run_path(str(ROOT / "scripts/build_refl215_f16.py"))
    final = tmp_path / "patches.npy"
    sink = builder["Float16Sink"](patch_extraction._partial(final), (3, BANDS, 4, 4))
    rows = np.random.default_rng(0).uniform(0, 1.2, size=(2, BANDS, 4, 4)).astype(np.float32)
    sink[0], sink[1] = rows[0], rows[1]
    # A pass-2 failure leaves a counted row unwritten; the finaliser trims it in place.
    patch_extraction._finalise_stream(sink, final, 2)
    cube = np.load(final)
    assert cube.dtype == np.float16 and cube.shape == (2, BANDS, 4, 4)
    np.testing.assert_array_equal(cube, rows.astype(np.float16))
    stats = sink.stats.as_dict()
    assert stats["max_rel_normal_range"] <= 2.0**-11 and sink.rows_written == 2
