"""The shared head recipe keeps the anchor eligible and is seed-deterministic."""
import numpy as np
import torch

from spectralquadnet.experiments.residual_head import fit_residual_head

MODEL = {"hsi_dim": 6, "rgb_dim": 5, "hidden": 4, "classes": 3, "dropout": 0.0}
RECIPE = {"epochs": 6, "patience": 3, "batch": 16, "lr": 0.01, "weight_decay": 0.0, "residual_penalty": 0.1}


def _data(n: int, seed: int) -> tuple[list[torch.Tensor], np.ndarray]:
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 3
    h = torch.randn(n, 6, generator=g) + y[:, None].float()
    r = torch.randn(n, 5, generator=g)
    anchor = torch.full((n, 3), 1 / 3)
    anchor[torch.arange(n), y] += 0.05  # anchor already right on every row
    return [h, r, anchor / anchor.sum(1, keepdim=True)], y.numpy()


def test_perfect_anchor_is_kept_when_nothing_improves_calibration():
    train, yt = _data(48, 0)
    calib, yc = _data(30, 1)
    head, selection, traces = fit_residual_head(train, calib, yt, yc, MODEL, RECIPE, seed=0)
    assert traces[0]["epoch"] == 0 and selection["anchor_calib_f1"] == 1.0
    assert selection["selected_epoch"] == 0  # no epoch can strictly beat F1 = 1
    with torch.no_grad():
        z, residual = head(*calib)
    assert residual.count_nonzero() == 0


def test_refit_with_same_seed_is_identical():
    train, yt = _data(48, 2)
    train[2] = torch.full((48, 3), 1 / 3)
    calib, yc = _data(30, 3)
    calib[2] = torch.full((30, 3), 1 / 3)
    a = fit_residual_head(train, calib, yt, yc, MODEL, RECIPE, seed=5)
    b = fit_residual_head(train, calib, yt, yc, MODEL, RECIPE, seed=5)
    assert a[1] == b[1] and a[2] == b[2]
    for k, v in a[0].state_dict().items():
        torch.testing.assert_close(v, b[0].state_dict()[k], rtol=0, atol=0)
