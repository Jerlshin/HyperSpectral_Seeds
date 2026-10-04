"""Identity, leakage and reproducibility gates for the RGB research pathway."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.measure import regionprops

from spectralquadnet.data.prep.multimodal import read_envi, sha256, spectral_summary
from spectralquadnet.data.prep.rgb import decode_rgb, foreground_crop, grid_cells, pair_grids
from spectralquadnet.experiments.rgb_probe import (
    calibrate_temperature,
    cluster_interval,
    fit_probe,
    select_fusion,
    verify_plan,
)


def jitter_grid():
    rng = np.random.default_rng(27)
    return np.array([(r * 60, c * 50) for r in range(8) for c in range(6)], float) + rng.normal(
        0, 4, (48, 2)
    )


def test_grid_missing_component_does_not_shift_identity():
    points = jitter_grid()
    expected = np.array([(r, c) for r in range(8) for c in range(6)])
    keep = np.delete(np.arange(48), [8, 23])
    np.testing.assert_array_equal(grid_cells(points[keep]), expected[keep])
    rgb = points * 5 + np.array([300, 1600])
    indices, qc = pair_grids(points[keep], rgb)
    np.testing.assert_array_equal(indices, keep)
    assert qc["max_hsi_px"] < 1e-3


def test_grid_rejects_duplicate_and_orientation_ambiguity():
    points = jitter_grid()
    points[1] = points[0]
    with pytest.raises(ValueError, match="Duplicate"):
        grid_cells(points)
    regular = np.array([(r * 60, c * 50) for r in range(8) for c in range(6)], float)
    with pytest.raises(ValueError, match="orientation"):
        pair_grids(regular, regular * 5)


def test_pairing_rejects_permuted_physical_cells():
    points = jitter_grid()
    rgb = points * 5
    # Move a seed substantially within its own cell: topology alone cannot certify it.
    rgb[14, 0] += 100
    with pytest.raises(ValueError, match="residual"):
        pair_grids(points, rgb)


def test_exif_does_not_rotate_identity():
    a = np.zeros((20, 30, 3), np.uint8)
    a[:8, :8] = [255, 0, 0]
    image = Image.fromarray(a)
    exif = Image.Exif()
    exif[274] = 3
    data = io.BytesIO()
    image.save(data, format="JPEG", exif=exif)
    actual = decode_rgb(data.getvalue())
    assert actual.shape == a.shape and actual[:6, :6, 0].mean() > 200
    with pytest.raises(ValueError, match="payload"):
        decode_rgb(b"not an image")


def test_crop_excludes_other_objects_and_background():
    labels = np.zeros((40, 50), np.int32)
    labels[10:30, 20:25] = 1
    labels[10:30, 30:34] = 2
    image = np.full((40, 50, 3), 255, np.uint8)
    image[labels == 1] = [80, 120, 160]
    crop = foreground_crop(image, labels, regionprops(labels)[0], size=44)
    assert crop.shape == (44, 44, 3)
    assert crop.max() <= 160 and np.count_nonzero(crop) == 20 * 5 * 4 * 3
    assert np.all(crop[:, 0] == 0)


def test_envi_exact_member_and_payload_validation():
    b = io.BytesIO()
    header = "ENVI\nsamples = 2\nlines = 3\nbands = 4\nheader offset = 0\ndata type = 12\ninterleave = bil\nbyte order = 0\n"
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("scan.hdr", header)
        z.writestr("scan.jpg", b"not cube")
        z.writestr("scan.raw", np.arange(24, dtype="<u2").tobytes())
        z.writestr("bad.hdr", header)
        z.writestr("bad.raw", b"x")
    with zipfile.ZipFile(b) as z:
        assert read_envi(z, "scan.hdr").shape == (3, 2, 4)
        assert read_envi(z, "scan.hdr")[1, 1, 2] == 13
        with pytest.raises(ValueError, match="payload"):
            read_envi(z, "bad.hdr")


def test_spectral_summary_ignores_background_and_empty_regions():
    patch = np.ones((3, 64, 64), np.float32) * 999
    mask = np.zeros((64, 64), np.float32)
    mask[:16, :16] = 1
    patch[:, :16, :16] = np.arange(3)[:, None, None] + 1
    summary, tokens, occupancy = spectral_summary(patch, mask)
    np.testing.assert_allclose(summary[0], [1, 2, 3])
    assert np.all(summary[1] == 0) and occupancy.sum() == 1
    np.testing.assert_allclose(tokens[0], [1, 2, 3])
    assert np.all(tokens[1:] == 0)


def test_frozen_guard_detects_mutation(tmp_path: Path):
    data = tmp_path / "data"
    data.write_text("original")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"input_hashes": {str(data): sha256(data)}}))
    plan.with_suffix(".sha256").write_text(sha256(plan))
    verify_plan(plan)
    data.write_text("changed")
    with pytest.raises(ValueError, match="input hash"):
        verify_plan(plan)
    plan.write_text("{}")
    with pytest.raises(ValueError, match="plan hash"):
        verify_plan(plan)


def test_probe_scaling_and_fitting_are_training_only():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(60, 5))
    y = np.tile(np.arange(3), 20)
    tr = np.arange(30)
    ca = np.arange(30, 45)
    te = np.arange(45, 60)
    a, _ = fit_probe(x, y, tr, ca, te)
    changed = x.copy()
    changed[te] += 10000
    b, _ = fit_probe(changed, y, tr, ca, te)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="Overlapping"):
        fit_probe(x, y, tr, tr, te)


def test_calibration_and_paired_interval():
    logits = np.array([[100.0, 0.0], [0.0, 100.0]])
    assert calibrate_temperature(logits, np.array([1, 0]), [1.0, 16.0]) == 16
    p = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert select_fusion(p, p[:, ::-1], np.array([0, 1]), [0.0, 0.5, 1.0]) == 1
    assert cluster_interval(np.ones((8, 2)) * 0.02) == [0.02, 0.02]
    with pytest.raises(ValueError, match="positive"):
        calibrate_temperature(logits, np.array([0, 1]), [0.0])


def test_historical_logits_require_exact_row_identity(tmp_path: Path):
    from spectralquadnet.experiments.rgb_probe import aligned_logits

    path = tmp_path / "logits.npz"
    labels = np.array([0, 1, 2, 0, 1, 2])
    rows = np.array([5, 3, 1])
    logits = np.arange(9).reshape(3, 3)
    np.savez(path, rows=rows, targets=labels[rows], logits=logits)
    actual = aligned_logits(path, np.array([1, 3, 5]), labels)
    np.testing.assert_array_equal(actual, logits[::-1])
    with pytest.raises(ValueError, match="rows"):
        aligned_logits(path, np.array([0, 3, 5]), labels)
    np.savez(path, rows=rows, targets=np.zeros(3), logits=logits)
    with pytest.raises(ValueError, match="targets"):
        aligned_logits(path, np.array([1, 3, 5]), labels)


def test_frozen_rgb_controls_preserve_geometry_and_remove_color():
    import importlib.util

    import torch

    spec = importlib.util.spec_from_file_location("rgb_cache", "scripts/cache_rgb_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = torch.zeros(1, 224, 224, 3, dtype=torch.uint8)
    image[:, 40:180, 85:125] = torch.tensor([70, 130, 210], dtype=torch.uint8)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    gray = module.transform_rgb(image, "gray") * std + mean
    torch.testing.assert_close(gray[:, 0], gray[:, 1])
    torch.testing.assert_close(gray[:, 1], gray[:, 2])
    binary = module.transform_rgb(image, "silhouette") * std + mean
    torch.testing.assert_close(binary[:, 0], (image.sum(-1) > 0).float())
    for view in ["rgb", "gray", "rgb32", "silhouette"]:
        actual = module.transform_rgb(image, view)
        assert actual.shape == (1, 3, 224, 224) and torch.isfinite(actual).all()
    with pytest.raises(ValueError, match="Unknown"):
        module.transform_rgb(image, "invalid")


def test_exported_probe_matches_fitted_pipeline(tmp_path: Path):
    from spectralquadnet.experiments.rgb_probe import probe_logits

    x = np.random.default_rng(80).normal(size=(90, 8)).astype(np.float32)
    y = np.tile(np.arange(3), 30)
    path = tmp_path / "probe.npz"
    ca, te = fit_probe(x, y, np.arange(60), np.arange(60, 75), np.arange(75, 90), export=path)
    np.testing.assert_allclose(probe_logits(path, x[60:75]), ca, atol=1e-12)
    np.testing.assert_allclose(probe_logits(path, x[75:]), te, atol=1e-12)
    with pytest.raises(ValueError, match="dimensions"):
        probe_logits(path, x[:, :3])
    with pytest.raises(FileExistsError, match="export"):
        fit_probe(x, y, np.arange(60), np.arange(60, 75), np.arange(75, 90), export=path)
