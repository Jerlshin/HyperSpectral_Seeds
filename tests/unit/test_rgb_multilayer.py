"""S36 multi-layer foreground-token RGB branch."""

from __future__ import annotations

import pytest
import torch
from test_rgb_branch import TinyViT, crops

from spectralquadnet.experiments.rgb_finetune import to_unit
from spectralquadnet.models.rgb_multilayer import MultiLayerRGBBranch


def test_shapes_and_layer_count() -> None:
    torch.manual_seed(0)
    model = MultiLayerRGBBranch(TinyViT(depth=4), classes=5, keep=64, layers=3, width=12).eval()
    logits, z = model(to_unit(crops()))
    assert logits.shape == (2, 5) and z.shape == (2, 12)
    assert len(model.project) == 3


def test_each_read_layer_contributes() -> None:
    torch.manual_seed(0)
    model = MultiLayerRGBBranch(TinyViT(depth=3), classes=5, keep=64, layers=2, width=6).eval()
    x = to_unit(crops())
    base = model.embed(x)
    with torch.no_grad():
        model.project[0][1].weight.zero_()
        model.project[0][1].bias.zero_()
    assert not torch.allclose(model.embed(x), base)


def test_morphometrics_are_required_when_configured_and_shift_the_embedding() -> None:
    torch.manual_seed(0)
    model = MultiLayerRGBBranch(TinyViT(), classes=5, keep=64, layers=2, width=6, morph_dim=8).eval()
    x = to_unit(crops())
    with pytest.raises(ValueError):
        model(x)
    a, _ = model(x, torch.arange(16.0).view(2, 8))
    b, _ = model(x, torch.arange(16.0).view(2, 8).flip(1))
    assert not torch.allclose(a, b)


def test_partial_training_must_cover_every_read_layer() -> None:
    with pytest.raises(ValueError):
        MultiLayerRGBBranch(TinyViT(depth=4), classes=5, layers=3, train_blocks=2)
    model = MultiLayerRGBBranch(TinyViT(depth=4), classes=5, keep=64, layers=2, width=6, train_blocks=2)
    trainable = {n.split(".")[2] for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.blocks.")}
    assert trainable == {"2", "3"}


def test_finetune_loop_heads_at_head_rate_and_morph_standardised_on_train() -> None:
    import numpy as np

    from spectralquadnet.experiments.multilayer_finetune import (
        finetune,
        morph_features,
        parameter_groups,
        predict,
    )
    from spectralquadnet.experiments.rgb_finetune import Recipe

    torch.manual_seed(0)
    raw = np.abs(np.random.default_rng(0).normal(5, 1, (12, 8))) + 1
    morph = morph_features(raw, np.arange(8))
    assert np.allclose(morph[:8].mean(0), 0, atol=1e-5)
    model = MultiLayerRGBBranch(TinyViT(depth=3), classes=3, keep=64, layers=2, width=6, morph_dim=8)
    groups = parameter_groups(model, lr=1.0, head_lr=10.0, decay=0.5, weight_decay=0.05)
    rate = {id(p): g["lr"] for g in groups for p in g["params"]}  # type: ignore[attr-defined]
    named = dict(model.named_parameters())
    assert rate[id(named["project.0.1.weight"])] == rate[id(named["morph.1.weight"])] == 10.0
    images = crops(12).numpy()
    y = np.arange(12) % 3
    recipe = Recipe(epochs=1, batch=4, precision="fp32", keep_tokens=64)
    model, best, trace = finetune(model, images, morph, y, np.arange(8), np.arange(8, 12), recipe, 0,
                                  torch.device("cpu"), lambda row: None)
    logits, z = predict(model, images, morph, np.array([2, 0]), torch.device("cpu"), recipe)
    assert logits.shape == (4, 2, 3) and z.shape == (2, 6) and len(trace) == 1
