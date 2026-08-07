"""ASAM: the SAM perturbation is normalised by ``|theta|`` (**T2-4** / OP-5).

SAM's ``rho``-ball is defined in raw parameter space and is not scale-invariant.
The ArcFace head *is* scale-invariant by construction — it normalises both its
embedding and its weights, so ``L(cW) = L(W)`` — which means a raw-space
perturbation budget has no defined meaning there at all. Worse, the budget is
allocated in proportion to gradient magnitude, and the head's gradients are
amplified by ``s = 48``: §2.5.8 argues SAM was therefore flattening the
classifier rather than the representation.

ASAM's element-wise ``T_theta = diag(|theta|)`` re-weighting,
``eps = rho * T^2 g / ||T g||``, is invariant to a per-parameter rescaling and
puts the budget where the parameters are large.

**The premise holds at initialisation and not on the weights Stage 3 starts
from.** Measured head perturbation share against its 0.88 % parameter share:

    weights            raw SAM     ASAM       ratio to param share
    fresh init         2.01 %      0.034 %    2.29x  ->  0.038x
    best_stage2.pth    0.30 %      0.0008 %   0.34x  ->  0.0009x

So T2-4's stated criterion is met — comfortably — but the defect it is aimed at
is only visible before Stage 2 has fitted the head. Recorded here so the
+0.002…+0.006 attached to it is not read as measured.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from spectralquadnet.engine.checkpoint import SchemaTooOldError, remap_state_dict
from spectralquadnet.losses.auxiliary import _compute_aux_loss
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.optim.sam import SAM

pytestmark = pytest.mark.regression

DEVICE = torch.device("cpu")
RHO = 0.015
HEAD_PREFIX = "arcface_head"


def head_and_rest(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    head = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    rest = [p for n, p in model.named_parameters() if not n.startswith(HEAD_PREFIX)]
    return head, rest


def perturbation_share(model: nn.Module, x, y, adaptive: bool) -> tuple[float, float]:
    """``(head's share of ||eps||^2, head's share of the parameters)``.

    The objective is Stage 3's own compound one, since that is the gradient the
    real ``first_step`` sees.
    """
    head, rest = head_and_rest(model)
    sam = SAM(
        [{"params": head}, {"params": rest}],
        optim.AdamW,
        rho=RHO,
        adaptive=adaptive,
        lr=1e-5,
        weight_decay=2e-4,
    )
    sam.zero_grad()
    focal = FocalLoss(gamma=1.0)
    out = model(x, labels=y, return_embed=True)
    loss = (
        focal(out["main"], y)
        + 0.02 * SupConLoss(temperature=0.10)(out["emb"], y)
        + 0.01 * ProtoNCELoss(temperature=0.10)(out["emb"], y)
        + 0.10 * _compute_aux_loss(focal, out, y, y, 1.0, use_mixup=False)
    )
    loss.backward()

    mass = sam.perturbation_mass()
    total = sum(float(v) for v in mass.values())
    n_head = sum(p.numel() for p in head)
    n_all = n_head + sum(p.numel() for p in rest)
    return float(mass[0]) / total, n_head / n_all


@pytest.fixture
def sam_model(cfg, physical_wl, silence_dropout):
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl).to(DEVICE)
    silence_dropout(model)
    model.branch_drop_prob = 0.0
    return model.train()


@pytest.fixture(scope="module")
def sam_batch(request) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(11)
    num_bands = request.getfixturevalue("cfg").data.num_bands
    x = torch.randn(8, num_bands, 64, 64, generator=gen)
    return x, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])


# ══════════════════════════════════════════════════════════════════════
#  T2-4's validation criterion
# ══════════════════════════════════════════════════════════════════════


def test_asam_puts_less_than_its_parameter_share_on_the_head(sam_model, sam_batch) -> None:
    """The criterion, verbatim: perturbation mass on ``arcface_head`` falls below its share."""
    x, y = sam_batch

    share, param_share = perturbation_share(sam_model, x, y, adaptive=True)

    assert (
        share < param_share
    ), f"head took {share:.4%} of the budget for {param_share:.4%} of theta"


def test_raw_sam_does_not_overspend_on_the_head_under_the_tier3_architecture(
    sam_model, sam_batch
) -> None:
    """§2.5.8's over-allocation is gone at initialisation too — measured, not assumed.

    Through Tier 2 this test asserted the **defect**: raw SAM gave the head 2.3x
    its parameter share at initialisation, which is the premise T2-4's
    +0.002…+0.006 was reasoned from. Its companion below already found the
    premise absent on ``best_stage2.pth`` (the weights Stage 3 actually starts
    from), and Tier 3 removes it at initialisation as well: raw SAM now puts
    ~0.20% of the perturbation on a head holding 1.33% of theta.

    Two Tier-3 changes drive it. The head is a larger share of a model that lost
    2.66 M parameters, and FU-2 replaced the fusion's per-sample ``LayerNorm``
    with ``BatchNorm1d``, so the embedding handed to the head is no longer
    rescaled to a fixed norm per sample and the head's gradient at
    initialisation is correspondingly smaller.

    The assertion is inverted rather than deleted. ASAM is still *correct* —
    SAM's rho-ball is not scale-invariant and the head is — and the tests above
    still hold, but the mechanism the estimate came from is not present at
    either measurement point, and that has to stay recorded.
    """
    x, y = sam_batch

    share, param_share = perturbation_share(sam_model, x, y, adaptive=False)

    assert share < param_share, (
        f"head took {share:.4%} of the budget for {param_share:.4%} of theta — the "
        "pre-Tier-3 over-allocation is back; §2.5.8's premise would need re-examining"
    )


def test_asam_shrinks_the_head_budget_by_at_least_an_order_of_magnitude(
    sam_model, sam_batch
) -> None:
    x, y = sam_batch

    raw, _ = perturbation_share(sam_model, x, y, adaptive=False)
    adaptive, _ = perturbation_share(sam_model, x, y, adaptive=True)

    assert adaptive * 10 < raw


@pytest.mark.requires_dataset
def test_the_tier2_measurement_on_the_archived_weights_is_now_unreachable(
    cfg, physical_wl, checkpoint_paths
) -> None:
    """The same measurement on ``best_stage2.pth`` — which Tier 3 put out of reach.

    Through Tier 2 this test loaded the archived Stage-2 checkpoint and found
    raw SAM already **under** its parameter share there (adaptive cut it a
    further ~375x). That number is recorded in MIGRATION_PROGRESS under T2-4 and
    is not re-derivable: BR-1/BR-3/BR-4 changed what three branches consume, so
    those weights cannot be loaded into this model at all.

    Kept, and inverted, rather than deleted. A test that silently disappeared
    would leave the impression the measurement still stands behind the current
    architecture; this one states that it does not, and fails if the refusal it
    depends on ever stops being total.
    """
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl).to(DEVICE)
    bundle = torch.load(checkpoint_paths[2], map_location="cpu", weights_only=False)

    with pytest.raises(SchemaTooOldError, match="predates Tier 3"):
        remap_state_dict(bundle["model"], model.state_dict(), int(bundle.get("schema_version", 1)))


# ══════════════════════════════════════════════════════════════════════
#  The property ASAM exists for
# ══════════════════════════════════════════════════════════════════════


def test_asam_is_invariant_to_a_parameter_rescaling() -> None:
    """``theta -> c*theta`` leaves the *relative* perturbation unchanged; raw SAM's does not.

    The definition, on the smallest model that can express it. A network whose
    loss is invariant under such a rescaling — which is exactly what a
    normalised head is — must get the same perturbation either way, and under
    raw SAM it does not.
    """

    def relative_step(scale: float, adaptive: bool) -> torch.Tensor:
        layer = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(1.0, 5.0).view(1, 4) * scale)
        sam = SAM(layer.parameters(), optim.SGD, rho=RHO, adaptive=adaptive, lr=0.0)
        before = layer.weight.detach().clone()
        # A loss whose gradient scales as 1/c, so the *relative* geometry is
        # identical at every c — the scale-invariant regime.
        (layer(torch.ones(1, 4)).squeeze() / scale).backward()
        sam.first_step()
        return (layer.weight.detach() - before) / before

    assert torch.allclose(relative_step(1.0, True), relative_step(7.0, True), atol=1e-6)
    assert not torch.allclose(relative_step(1.0, False), relative_step(7.0, False), atol=1e-6)


def test_a_group_can_opt_out_of_the_perturbation_entirely() -> None:
    """OP-5's stated alternative — "exclude the head" — is available and works."""
    frozen = nn.Linear(3, 1, bias=False)
    moving = nn.Linear(3, 1, bias=False)
    sam = SAM(
        [
            {"params": list(frozen.parameters()), "perturb": False},
            {"params": list(moving.parameters())},
        ],
        optim.SGD,
        rho=0.5,
        lr=0.0,
    )
    before = frozen.weight.detach().clone()
    moved_before = moving.weight.detach().clone()

    (frozen(torch.ones(1, 3)) + moving(torch.ones(1, 3))).squeeze().backward()
    sam.first_step()

    assert torch.equal(frozen.weight.detach(), before)
    assert not torch.equal(moving.weight.detach(), moved_before)


def test_perturbation_mass_does_not_move_the_weights(sam_model, sam_batch) -> None:
    """The measurement above must be a measurement, not a step."""
    x, y = sam_batch
    head, rest = head_and_rest(sam_model)
    sam = SAM([{"params": head}, {"params": rest}], optim.AdamW, rho=RHO, adaptive=True, lr=1e-5)
    out = sam_model(x, labels=y)
    FocalLoss(gamma=1.0)(out["main"], y).backward()
    before = {k: v.detach().clone() for k, v in sam_model.named_parameters()}

    sam.perturbation_mass()

    for key, value in sam_model.named_parameters():
        assert torch.equal(value.detach(), before[key]), key


def test_the_reported_shares_are_a_partition(sam_model, sam_batch) -> None:
    """``perturbation_mass`` accounts for the whole budget, so a share is a share.

    ``||eps||`` is ``rho`` by construction under raw SAM — the normalisation
    is exactly what makes the ball a ball — so the group masses must sum to
    ``rho^2``. If they did not, "below its parameter share" would be measured
    against an incomplete denominator.
    """
    x, y = sam_batch
    head, rest = head_and_rest(sam_model)
    sam = SAM([{"params": head}, {"params": rest}], optim.AdamW, rho=RHO, adaptive=False, lr=1e-5)
    sam.zero_grad()
    out = sam_model(x, labels=y)
    FocalLoss(gamma=1.0)(out["main"], y).backward()

    total = sum(float(v) for v in sam.perturbation_mass().values())

    assert np.sqrt(total) == pytest.approx(RHO, rel=1e-4)
