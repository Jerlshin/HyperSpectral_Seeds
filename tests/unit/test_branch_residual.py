import pytest
import torch

from spectralquadnet.models.branch_residual import FrozenBranchCorrection


@pytest.mark.parametrize("mode", ["hsi", "rgb"])
def test_disabled_correction_cannot_use_removed_features(mode):
    torch.manual_seed(0)
    head = FrozenBranchCorrection(mode, hsi_dim=3, rgb_dim=4, hidden=5, classes=2)
    head.eval()
    h, r, anchor = torch.randn(6, 3), torch.randn(6, 4), torch.tensor([[.3, .7]]).repeat(6, 1)
    initial, _ = head(h, r, anchor)
    torch.testing.assert_close(initial.softmax(1), anchor)
    with torch.no_grad():
        head.readout.weight.normal_()
    baseline, _ = head(h, r, anchor)
    removed, _ = head(h if mode == "hsi" else h + 10, r + 10 if mode == "hsi" else r, anchor)
    active, _ = head(h + 10 if mode == "hsi" else h, r if mode == "hsi" else r + 10, anchor)
    torch.testing.assert_close(baseline, removed)
    assert not torch.allclose(baseline, active)
    assert all(not p.requires_grad for p in (head.rgb if mode == "hsi" else head.hsi).parameters())
