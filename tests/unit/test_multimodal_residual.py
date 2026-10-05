"""The learned candidate preserves its fusion control and modality additivity."""
import torch

from spectralquadnet.models.multimodal_residual import FrozenResidualFusion


def test_initial_predictor_equals_calibrated_fusion_and_is_trainable():
    model = FrozenResidualFusion(hsi_dim=5, rgb_dim=7, hidden=3, classes=4)
    h, r = torch.randn(8, 5), torch.randn(8, 7)
    p = torch.randn(8, 4).softmax(1)
    z, residual = model(h, r, p)
    torch.testing.assert_close(z.softmax(1), p)
    assert residual.count_nonzero() == 0
    torch.nn.functional.cross_entropy(z, torch.arange(8) % 4).backward()
    assert model.readout.weight.grad.norm() > 0


def test_no_cross_modality_interaction_in_correction():
    model = FrozenResidualFusion(hsi_dim=5, rgb_dim=7, hidden=3, classes=4).eval()
    torch.nn.init.normal_(model.readout.weight)
    h1, h2, r1, r2 = torch.randn(1, 5), torch.randn(1, 5), torch.randn(1, 7), torch.randn(1, 7)
    p = torch.full((1, 4), .25)
    v11 = model(h1, r1, p)[1]
    v12 = model(h1, r2, p)[1]
    v21 = model(h2, r1, p)[1]
    v22 = model(h2, r2, p)[1]
    torch.testing.assert_close(v11 + v22, v12 + v21)
