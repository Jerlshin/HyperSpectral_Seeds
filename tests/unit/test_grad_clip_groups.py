"""Per-group gradient clipping (**T2-5** / OP-3, M-10).

One global ``clip_grad_norm_(model.parameters(), 1.0)`` rescales *every*
parameter by ``max_norm / ||g||`` the moment the total norm crosses the
threshold. The head's gradients are amplified by ``s = 48`` relative to the
backbone's, so the total is dominated by the head, and a single saturated batch
in the head divided the whole model's effective learning rate — including the
branches, whose own gradients were nowhere near the limit (§2.5.8).

Clipping head, fusion and backbone independently decouples them. T2-5's
criterion is that "the backbone's pre/post-clip norm ratio decouples from the
head's", which is what :func:`test_a_saturated_head_no_longer_scales_the_backbone`
measures directly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.optim.param_groups import (
    CLIP_GROUPS,
    clip_grad_norm_by_group,
    split_by_clip_group,
)

pytestmark = pytest.mark.regression

DEVICE = torch.device("cpu")
GROUP_NAMES = {label for label, _ in CLIP_GROUPS}


@pytest.fixture
def model(cfg, physical_wl):
    torch.manual_seed(0)
    return SpectralQuadNet.from_config(cfg, physical_wl).to(DEVICE)


def fill_grads(model: nn.Module, per_group: dict[str, float]) -> None:
    """Give every parameter in each group a constant gradient of the given size."""
    groups = split_by_clip_group(model)
    for label, value in per_group.items():
        for param in groups[label]:
            param.grad = torch.full_like(param, value)


# ══════════════════════════════════════════════════════════════════════
#  The partition
# ══════════════════════════════════════════════════════════════════════


def test_every_trainable_parameter_lands_in_exactly_one_group(model) -> None:
    """The three clips together must cover what the one global clip covered.

    A parameter that fell through would silently stop being clipped at all,
    which is a strictly worse failure than the one this replaces.
    """
    groups = split_by_clip_group(model)
    counted = sum(len(v) for v in groups.values())
    trainable = [p for p in model.parameters() if p.requires_grad]

    assert counted == len(trainable)
    seen = {id(p) for params in groups.values() for p in params}
    assert seen == {id(p) for p in trainable}


def test_the_groups_are_the_ones_the_defect_is_about(model) -> None:
    groups = split_by_clip_group(model)

    assert set(groups) == GROUP_NAMES
    assert set(groups["head"]) == set(model.arcface_head.parameters())
    assert set(groups["fusion"]) == set(model.cross_interaction.parameters()) | set(
        model.embed_net.parameters()
    )


# ══════════════════════════════════════════════════════════════════════
#  T2-5's validation criterion
# ══════════════════════════════════════════════════════════════════════


def test_a_saturated_head_no_longer_scales_the_backbone(model) -> None:
    """The criterion: the backbone's clip ratio is independent of the head's norm.

    The backbone is given a gradient far below the threshold and the head one
    far above it. Under the global clip the backbone came out divided by the
    *head's* excess; under per-group clipping it comes out untouched.
    """
    fill_grads(model, {"head": 10.0, "fusion": 1e-6, "backbone": 1e-6})
    backbone = split_by_clip_group(model)["backbone"]
    before = torch.cat([p.grad.reshape(-1) for p in backbone]).clone()

    norms = clip_grad_norm_by_group(model, max_norm=1.0)

    after = torch.cat([p.grad.reshape(-1) for p in backbone])
    assert torch.equal(before, after), "the backbone was rescaled by the head's saturation"
    assert float(norms["head"]) > 1.0, "the head must actually have been over the threshold"


def test_the_global_clip_would_have_scaled_it(model) -> None:
    """The companion: the same gradients through one global clip *do* couple.

    Without this the test above would pass on a model whose head never
    saturates, which is not the claim.
    """
    fill_grads(model, {"head": 10.0, "fusion": 1e-6, "backbone": 1e-6})
    backbone = split_by_clip_group(model)["backbone"]
    before = torch.cat([p.grad.reshape(-1) for p in backbone]).clone()

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    after = torch.cat([p.grad.reshape(-1) for p in backbone])
    assert not torch.allclose(before, after)
    assert after.abs().max() < before.abs().max()


def test_each_group_is_clipped_to_the_threshold_independently(model) -> None:
    """Every group over the limit comes back at the limit; every group under it is untouched."""
    fill_grads(model, {"head": 10.0, "fusion": 10.0, "backbone": 1e-8})

    clip_grad_norm_by_group(model, max_norm=1.0)

    groups = split_by_clip_group(model)
    for label in ("head", "fusion"):
        # Recomputed in float64. A flat fp32 re-reduction over the fusion's
        # 2.45 M elements disagrees with `clip_grad_norm_`'s per-tensor one by
        # 2.5e-3 — an artefact of the check, not of the clip, which is exact
        # to 4e-8 once the accumulation is wide enough to see it.
        norm = torch.cat([p.grad.reshape(-1) for p in groups[label]]).double().norm()
        assert float(norm) == pytest.approx(1.0, rel=1e-6), label
    backbone_norm = torch.cat([p.grad.reshape(-1) for p in groups["backbone"]]).double().norm()
    assert float(backbone_norm) < 1.0


def test_the_returned_norms_are_pre_clip(model) -> None:
    """Callers log these as ``grad_norm/preclip_*``; they must not describe the clipped state."""
    fill_grads(model, {"head": 10.0, "fusion": 10.0, "backbone": 10.0})

    norms = clip_grad_norm_by_group(model, max_norm=1.0)

    assert set(norms) == GROUP_NAMES
    for label, value in norms.items():
        assert float(value) > 1.0, label


def test_a_group_with_no_gradient_is_still_reported(model) -> None:
    """``clip_grad_norm_`` returns 0 for an all-``None`` group rather than raising.

    Stage 1 used to freeze the head; nothing does now, but a caller that
    freezes anything must not crash the clip.
    """
    fill_grads(model, {"head": 1.0, "fusion": 1.0, "backbone": 1.0})
    for param in split_by_clip_group(model)["head"]:
        param.grad = None

    norms = clip_grad_norm_by_group(model, max_norm=1.0)

    assert float(norms["head"]) == 0.0
