"""S31/S32 RGB readout, trainable branch, fine-tuning recipe and shared screen metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from spectralquadnet.experiments.rgb_finetune import Recipe, augment, lr_factor, to_unit
from spectralquadnet.experiments.rgb_readout import VIEWS, foreground_weights, pooled_readout, view
from spectralquadnet.experiments.screen_metrics import Cohort, contrast, score, summarise
from spectralquadnet.models.rgb_branch import RGBBranch, layer_decay_groups, patch_foreground


class TinyViT(nn.Module):
    """Minimal DINOv2-shaped backbone: patch embed + pos embed, blocks, norm."""

    def __init__(self, dim: int = 8, depth: int = 3) -> None:
        super().__init__()
        self.embed_dim = dim
        self.patch_embed = nn.Conv2d(3, dim, 14, 14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 257, dim))
        self.blocks = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def prepare_tokens_with_masks(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(len(x), -1, -1), p], 1) + self.pos_embed


def crops(n: int = 2) -> torch.Tensor:
    """uint8 N x 224 x 224 x 3 with a vertical kernel and exact-zero background."""
    images = torch.zeros(n, 224, 224, 3, dtype=torch.uint8)
    images[:, 20:200, 90:130] = torch.randint(1, 255, (n, 180, 40, 3), dtype=torch.uint8)
    return images


def test_views_are_orientation_group_and_weights_follow_foreground() -> None:
    images = crops()
    for name in VIEWS:
        twice = view(view(images, name), name)
        assert torch.equal(twice, images)  # every view is an involution
        w = foreground_weights(view(images, name))
        assert w.shape == (2, 256) and float(w.sum(1)[0]) == float(foreground_weights(images).sum(1)[0])
    chw = images.permute(0, 3, 1, 2)
    assert torch.equal(view(chw, "hflip"), view(images, "hflip").permute(0, 3, 1, 2))


def test_pooled_readout_ignores_background_tokens() -> None:
    tokens = torch.randn(1, 4, 3)
    weights = torch.tensor([[1.0, 0.0, 0.5, 0.0]])
    out = pooled_readout(tokens, torch.zeros(1, 3), weights)
    expected = (tokens[0, 0] + 0.5 * tokens[0, 2]) / 1.5
    assert torch.allclose(out[0, 3:], expected)
    tokens[0, 1] += 100  # a background token cannot move the readout
    assert torch.allclose(pooled_readout(tokens, torch.zeros(1, 3), weights)[0, 3:], expected)


def test_branch_keeps_all_foreground_tokens_and_ignores_background_content() -> None:
    torch.manual_seed(0)
    model = RGBBranch(TinyViT(), classes=5, keep=64).eval()
    x = to_unit(crops())
    fg = int((patch_foreground(x) > 0).sum(1).max())
    assert fg <= 64
    logits, z = model(x)
    assert logits.shape == (2, 5) and z.shape == (2, 16)
    # Changing pixels in a patch that is never selected leaves the output unchanged.
    y = x.clone()
    y[:, :, 210:224, 0:14] = 0.0  # stays background
    assert torch.allclose(model(y)[0], logits)


def test_layer_decay_orders_rates_and_exempts_norms_from_weight_decay() -> None:
    model = RGBBranch(TinyViT(depth=3), classes=5)
    groups = layer_decay_groups(model, lr=1.0, head_lr=10.0, decay=0.5, weight_decay=0.05)
    rates = {id(p): (g["lr"], g["weight_decay"]) for g in groups for p in g["params"]}  # type: ignore[attr-defined]
    named = dict(model.named_parameters())
    assert rates[id(named["classifier.weight"])][0] == 10.0
    assert rates[id(named["backbone.blocks.2.weight"])][0] == 0.5
    assert rates[id(named["backbone.blocks.0.weight"])][0] == 0.125
    assert rates[id(named["backbone.pos_embed"])] == (0.0625, 0.0)
    assert rates[id(named["backbone.norm.weight"])][1] == 0.0
    assert sum(len(g["params"]) for g in groups) == len(named)  # type: ignore[arg-type]


def test_augmentation_preserves_background_and_foreground_support_size() -> None:
    x = to_unit(crops(4))
    out = augment(x, Recipe(), torch.Generator().manual_seed(0))
    for i in range(4):
        assert int((out[i].amax(0) > 0).sum()) == int((x[i].amax(0) > 0).sum())
        assert float(out[i].min()) >= 0 and float(out[i].max()) <= 1
    blurred = augment(x, Recipe(blur_sigma=1.0, blur_p=1.0), torch.Generator().manual_seed(0))
    for i in range(4):
        assert int((blurred[i].amax(0) > 0).sum()) == int((x[i].amax(0) > 0).sum())
    assert lr_factor(0, 100, 10) == 0.1 and abs(lr_factor(99, 100, 10)) < 1e-3


def test_score_and_contrast_on_a_toy_cohort() -> None:
    y = np.repeat(np.arange(90), 2)
    session = np.where(np.arange(180) % 2 == 0, 8, 1)
    cross = np.zeros(90, dtype=bool)
    cross[:3] = True
    te = np.arange(0, 180, 2)
    splits = {0: {"train": np.arange(1, 180, 2), "calib": np.array([], dtype=int), "val": te, "test": np.array([], dtype=int)}}
    cohort = Cohort(y, session, cross, splits)
    perfect = np.eye(90)[y[te]]
    shifted = np.eye(90)[(y[te] + 1) % 90]
    rows, classes = [], []
    for arm, prob in (("good", perfect), ("bad", shifted)):
        m, c, p = score(cohort, 0, arm, te, prob)
        rows.append(m)
        classes += c
        assert len(p) == len(te)
    table = summarise(pd.DataFrame(rows), pd.DataFrame(classes))
    assert table.loc["good", "f1"] == 1.0 and table.loc["bad", "f1"] == 0.0
    assert table.loc["good", "to_session8"] == 1.0
    assert table.loc["bad", "cross_error_session_attraction"] == 0.0  # predicted classes trained only in session 1
    c = contrast(pd.DataFrame(classes), "good", "bad")
    assert c["delta_f1"] == 1.0 and c["delta_cross"] == 1.0


def test_finetune_selects_on_calib_and_exports_every_view() -> None:
    from spectralquadnet.experiments.rgb_finetune import finetune, predict

    torch.manual_seed(0)
    images = crops(12).numpy()
    y = np.arange(12) % 3
    recipe = Recipe(epochs=2, batch=4, precision="fp32", keep_tokens=64)
    model = RGBBranch(TinyViT(), classes=3, keep=64)
    seen: list[dict[str, float]] = []
    model, best, trace = finetune(model, images, y, np.arange(8), np.arange(8, 12), recipe, 0,
                                  torch.device("cpu"), seen.append)
    assert len(trace) == 2 and best["calib_f1"] == max(t["calib_f1"] for t in trace)
    logits, z = predict(model, images, np.array([3, 1]), torch.device("cpu"), recipe)
    assert logits.shape == (4, 2, 3) and z.shape == (2, 16)
    single, _ = predict(model, images, np.array([1]), torch.device("cpu"), recipe)
    assert np.allclose(single[:, 0], logits[:, 1], atol=1e-5)  # row order is preserved


def test_render_softens_texture_but_keeps_support() -> None:
    from spectralquadnet.experiments.rgb_finetune import render

    x = to_unit(crops(2))
    y = render(x, 0.5)
    assert torch.equal(y.amax(1) > 0, x.amax(1) > 0)
    assert float(y[:, :, 40:180, 95:125].diff(dim=-1).abs().mean()) < float(x[:, :, 40:180, 95:125].diff(dim=-1).abs().mean())
    assert torch.equal(render(x, 0.0), x)


def test_partial_finetuning_freezes_the_stem_and_matches_full_forward() -> None:
    torch.manual_seed(0)
    backbone = TinyViT(depth=3)
    full = RGBBranch(backbone, classes=5, keep=64).eval()
    x = to_unit(crops())
    reference = full(x)[0]
    partial = RGBBranch(backbone, classes=5, keep=64, train_blocks=1).eval()
    partial.classifier.load_state_dict(full.classifier.state_dict())
    assert torch.allclose(partial(x)[0], reference, atol=1e-6)
    trainable = {n for n, p in partial.named_parameters() if p.requires_grad}
    assert trainable == {"backbone.blocks.2.weight", "backbone.blocks.2.bias", "backbone.norm.weight",
                         "backbone.norm.bias", "classifier.weight", "classifier.bias"}
