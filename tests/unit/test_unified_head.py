"""One cosine head for all three stages (**T2-10** / HD-1).

The model used to carry two heads and a boolean. Stage 1 trained
``linear_head`` under cross-entropy; Stage 2 switched to ``arcface_head``,
bootstrapped from the linear head's weights, and changed the loss, the sampler,
the optimiser, the augmentation and the margin at the same moment. §2.4.6 counts
six simultaneous changes at one epoch boundary, and 1.1(i) attributes Stage 2's
epoch-1 collapse to them.

HD-1 removes the head selection. At ``m = 0`` the sub-centre head is a cosine
(NormFace) classifier, which §2.3.8 shows is nearly what the linear head
already was — ``EmbedNet``'s terminal LayerNorm pins ``||e|| ~ 16``, so the
linear head was scoring a near-constant-norm embedding. The curriculum now
differs across stages in the margin, the sampler and the optimiser, and in
nothing else.

T2-10's stated criterion — "Stage 2's epoch-1 val F1 >= Stage 1's final" —
cannot be evaluated without a real two-stage run; what is testable, and what is
tested here, is the property it follows from: **the forward pass at the
Stage-1/Stage-2 boundary is continuous**, because only the margin changes.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.engine.checkpoint import SCHEMA_VERSION
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet

pytestmark = pytest.mark.regression

DEVICE = torch.device("cpu")
BATCH, SPATIAL = 4, 64


@pytest.fixture
def model(cfg, physical_wl, silence_dropout):
    torch.manual_seed(0)
    built = SpectralQuadNet.from_config(cfg, physical_wl).to(DEVICE)
    silence_dropout(built)
    built.branch_drop_prob = 0.0
    return built


@pytest.fixture(scope="module")
def batch(cfg) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(13)
    x = torch.randn(BATCH, cfg.data.num_bands, SPATIAL, SPATIAL, generator=gen)
    return x, torch.tensor([0, 1, 2, 3])


# ══════════════════════════════════════════════════════════════════════
#  The second head is gone
# ══════════════════════════════════════════════════════════════════════


def test_the_model_has_exactly_one_head(model) -> None:
    assert hasattr(model, "arcface_head")
    assert not hasattr(model, "linear_head")


def test_the_head_selection_api_is_gone(model) -> None:
    """``use_arcface``/``freeze_head``/``init_from_linear`` all went with it.

    Not cosmetic: each was a way for a caller to put the model into a state
    that no longer exists, and leaving them as no-ops would make that
    invisible rather than impossible.
    """
    for name in ("use_arcface", "freeze_head", "unfreeze_head", "_use_arcface"):
        assert not hasattr(model, name), name
    assert not hasattr(model.arcface_head, "init_from_linear")
    assert not hasattr(model.arcface_head, "update_margins_from_f1")


def test_the_schema_version_records_the_change() -> None:
    """T2-10 took the bundle schema to v2; Tier 3 took it to v3.

    The head is still HD-1's — one for every stage — and nothing in *this*
    file's subject changed at v3. What changed is the version the head ships
    inside, which is why the assertion is ``>=``: pinning an equality here
    would make every later architecture item fail a head test it has nothing
    to do with.
    """
    assert SCHEMA_VERSION >= 2


# ══════════════════════════════════════════════════════════════════════
#  A zero margin is a cosine classifier
# ══════════════════════════════════════════════════════════════════════


def test_a_zero_margin_is_the_plain_scaled_cosine(model, batch) -> None:
    """Stage 1's head must be exactly ``s * cos(theta)`` — a NormFace classifier.

    Both forwards stay in ``train()``: BatchNorm uses batch statistics there
    and running ones in ``eval()``, so a cross-mode comparison would be
    measuring the buffers, not the head.
    """
    x, y = batch
    model.train()

    with torch.no_grad():
        margined = model(x, labels=y, arc_m=0.0)["main"]
        unmargined = model(x, labels=None)["main"]

    assert torch.allclose(margined, unmargined, atol=1e-5)


def test_the_zero_margin_fast_path_matches_the_margin_algebra(cfg, physical_wl) -> None:
    """The shortcut in ``forward`` must be an optimisation, not a different function.

    At ``m = 0``: ``cos m = 1``, ``sin m = 0``, so
    ``phi = cos(theta)*1 - sin(theta)*0`` is the target cosine unchanged. The
    head skips that algebra; this checks the two agree to fp32.
    """
    from spectralquadnet.models.heads import AdaptiveSubcenterArcFaceHead

    torch.manual_seed(0)
    head = AdaptiveSubcenterArcFaceHead(in_dim=16, num_classes=5, K=3, s=48.0).train()
    x = torch.randn(8, 16)
    y = torch.randint(0, 5, (8,))

    fast = head(x, y, global_m=0.0)
    explicit = head(x, y, global_m=1e-12)

    assert torch.allclose(fast, explicit, atol=1e-4)


def test_a_positive_margin_lowers_only_the_target_logit(model, batch) -> None:
    """The margin is a penalty on the sample's own class and nothing else."""
    x, y = batch
    model.train()

    with torch.no_grad():
        zero = model(x, labels=y, arc_m=0.0)["main"]
        margined = model(x, labels=y, arc_m=0.35)["main"]

    target = torch.zeros_like(zero, dtype=torch.bool).scatter_(1, y.view(-1, 1), True)
    assert (margined[target] < zero[target]).all()
    assert torch.allclose(margined[~target], zero[~target], atol=1e-5)


# ══════════════════════════════════════════════════════════════════════
#  The Stage-1 → Stage-2 transition
# ══════════════════════════════════════════════════════════════════════


def test_the_stage_boundary_changes_the_margin_and_nothing_else(model, batch, cfg) -> None:
    """The property T2-10's criterion follows from: the boundary is continuous.

    Stage 1's last forward and Stage 2's first differ by ``arcface_m0``, the
    warm-up's starting margin — 0.18 rad. Under the two-head design they
    differed by an entire classifier, whose ArcFace half had been bootstrapped
    from the other's weights and never trained.
    """
    x, y = batch
    model.train()

    with torch.no_grad():
        stage1_last = model(x, labels=y, arc_m=cfg.stage1.arcface_m)["main"]
        stage2_first = model(x, labels=y, arc_m=cfg.stage2.arcface_m0)["main"]

    off_target = torch.ones_like(stage1_last, dtype=torch.bool).scatter_(1, y.view(-1, 1), False)
    assert torch.allclose(
        stage1_last[off_target], stage2_first[off_target], atol=1e-5
    ), "89 of the 90 logits must be untouched by the transition"
    assert cfg.stage1.arcface_m == 0.0


def test_eval_mode_is_identical_at_every_stage(model, batch, cfg) -> None:
    """Inference takes no margin, so all three stages evaluate the same function."""
    x, _ = batch
    model.eval()

    with torch.no_grad():
        first = model(x)
        model.arcface_head.margins.fill_(cfg.stage2.arcface_m_max)
        second = model(x)

    assert torch.equal(first, second)


def test_the_head_is_trainable_from_stage_1(model) -> None:
    """Stage 1 used to freeze ``arcface_head``; under one head that would freeze everything."""
    assert all(p.requires_grad for p in model.arcface_head.parameters())


def test_stage_1_config_declares_a_zero_margin(cfg) -> None:
    """``0.0`` is the substance of HD-1, so it is a config value rather than a literal."""
    assert cfg.stage1.arcface_m == 0.0
    assert cfg.stage2.arcface_m > 0.0


# ══════════════════════════════════════════════════════════════════════
#  Gradient flow
# ══════════════════════════════════════════════════════════════════════


def test_a_stage_1_step_trains_the_shared_head(model, batch, cfg) -> None:
    """The head that Stage 2 inherits is the head Stage 1 trained.

    That is what makes the bootstrap unnecessary. Under the old design Stage 2
    began with sub-centres copied from a linear head and never optimised
    through the angular objective.
    """
    x, y = batch
    model.train()
    model.zero_grad()

    out = model(x, labels=y, arc_m=cfg.stage1.arcface_m)
    torch.nn.functional.cross_entropy(out["main"], y).backward()

    assert model.arcface_head.weight.grad is not None
    assert float(model.arcface_head.weight.grad.abs().sum()) > 0.0
