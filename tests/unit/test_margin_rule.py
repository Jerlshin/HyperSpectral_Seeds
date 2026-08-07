"""The signed precision/recall margin rule (**T2-8** / HD-3, M-6, M-7).

Carries §4.3's ``test_margin_rule_sign``.

An additive angular margin is a **penalty on the class it is applied to**. It
requires ``cos(theta_y + m)`` to beat every unmargined ``cos theta_c``, so it
*shrinks* the margined class's decision region. The rule this replaces read
``M(c) = m_base + m_delta (1 - F1_c)``: it raised the margin for every low-``F1``
class, including the ones whose ``F1`` was low because their own samples were
going elsewhere. That is a positive-feedback loop with the wrong sign — the
class loses recall, gets a larger margin, loses more recall (§2.4.1).

``F1`` cannot tell the two failure modes apart because it conflates them. The
signed gap does:

    M(c) = clip(m_base + m_delta (R_c - P_c), m_min, m_max)

    over-claims  (P low, R high)  ->  R - P > 0  ->  margin raised  (region shrinks)
    balanced                      ->  R - P ~ 0  ->  m_base
    under-claims (R low)          ->  R - P < 0  ->  margin lowered (region expands)

The second half of HD-3 is the pairwise term: pushing class ``y`` away from all
89 others uniformly is what shrinks its region in the first place, so the
push is aimed by the row-normalised confusion matrix instead. The third is the
``theta + m < pi/2`` cap, which removes the gradient-decay regime of M-7.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectralquadnet.models.heads import HALF_PI, AdaptiveSubcenterArcFaceHead

pytestmark = pytest.mark.regression

DIM, CLASSES, K = 8, 5, 2
M_BASE, M_DELTA, M_MIN, M_MAX = 0.35, 0.20, 0.20, 0.50


def make_head(**overrides) -> AdaptiveSubcenterArcFaceHead:
    kwargs = dict(
        in_dim=DIM,
        num_classes=CLASSES,
        K=K,
        s=48.0,
        m_base=M_BASE,
        m_delta=M_DELTA,
        m_min=M_MIN,
        m_max=M_MAX,
    )
    torch.manual_seed(0)
    return AdaptiveSubcenterArcFaceHead(**{**kwargs, **overrides})


def legacy_f1_margin(f1: float) -> float:
    """The rule HD-3 replaces, restated so its sign can be compared rather than asserted."""
    return M_BASE + 0.10 * (1.0 - min(f1, 1.0))


# ══════════════════════════════════════════════════════════════════════
#  §4.3 · test_margin_rule_sign
# ══════════════════════════════════════════════════════════════════════


def test_margin_rule_sign() -> None:
    """§4.3's gate: ``M(c)`` **decreases** as ``R_c`` falls at fixed ``P_c``.

    The single statement the whole item turns on. A class losing recall must
    have its region widened, not narrowed.
    """
    head = make_head()
    fixed_precision = {0: 0.9, 1: 0.9, 2: 0.9}

    head.update_margins_from_pr(fixed_precision, {0: 0.9, 1: 0.6, 2: 0.3})

    assert head.margins[0] > head.margins[1] > head.margins[2]


def test_the_f1_rule_it_replaces_has_the_opposite_sign() -> None:
    """The companion: the old rule moved the *same* class the *other* way.

    Three classes with identical precision and falling recall have falling
    ``F1``, so the old rule raised their margins monotonically — the direction
    §2.4.1 identifies as the feedback loop.
    """
    precision, recalls = 0.9, (0.9, 0.6, 0.3)
    f1s = [2 * precision * r / (precision + r) for r in recalls]

    legacy = [legacy_f1_margin(f) for f in f1s]

    assert legacy[0] < legacy[1] < legacy[2], "the old rule raised the margin as recall fell"


def test_an_over_claiming_class_is_penalised() -> None:
    """``R > P`` — absorbing other classes' samples — is the case a raise is correct for."""
    head = make_head()

    head.update_margins_from_pr({0: 0.4}, {0: 0.95})

    assert float(head.margins[0]) > M_BASE


def test_a_balanced_class_keeps_the_base_margin() -> None:
    head = make_head()

    head.update_margins_from_pr({0: 0.8}, {0: 0.8})

    assert float(head.margins[0]) == pytest.approx(M_BASE)


def test_the_margin_is_clipped_to_its_configured_range() -> None:
    """``R - P`` spans [-1, 1], which at ``m_delta = 0.20`` would leave the range."""
    head = make_head()

    head.update_margins_from_pr({0: 1.0, 1: 0.0}, {0: 0.0, 1: 1.0})

    assert float(head.margins[0]) == pytest.approx(M_MIN)
    assert float(head.margins[1]) == pytest.approx(M_MAX)
    assert all(M_MIN <= float(m) <= M_MAX for m in head.margins)


def test_the_margin_is_non_monotone_in_f1() -> None:
    """T2-8's criterion: ``M(c)`` stops being a function of ``F1`` at all.

    Two classes constructed to share an ``F1`` and differ in *which* half of it
    is weak. Any rule driven by ``F1`` gives them the same margin; the point of
    HD-3 is that they need opposite ones.
    """
    head = make_head()
    # Both have F1 = 2*0.9*0.3/(0.9+0.3) = 0.45.
    head.update_margins_from_pr({0: 0.9, 1: 0.3}, {0: 0.3, 1: 0.9})

    f1_0 = 2 * 0.9 * 0.3 / 1.2
    f1_1 = 2 * 0.3 * 0.9 / 1.2
    assert f1_0 == pytest.approx(f1_1)
    assert float(head.margins[0]) < M_BASE < float(head.margins[1])


def test_classes_absent_from_the_measurement_keep_their_margin() -> None:
    head = make_head()
    head.margins[4] = 0.42

    head.update_margins_from_pr({0: 0.5}, {0: 0.5})

    assert float(head.margins[4]) == pytest.approx(0.42)


# ══════════════════════════════════════════════════════════════════════
#  The pairwise confusion margin
# ══════════════════════════════════════════════════════════════════════


def test_the_pairwise_term_is_inert_until_a_confusion_matrix_arrives() -> None:
    """A freshly built head must be a plain sub-centre ArcFace head."""
    head = make_head(pairwise_delta=0.5).train()
    x = torch.randn(4, DIM)
    y = torch.arange(4) % CLASSES

    assert not head.confusion.any()
    plain = make_head(pairwise_delta=0.0).train()
    assert torch.allclose(head(x, y), plain(x, y))


def test_the_confusion_matrix_is_row_normalised_with_a_zero_diagonal() -> None:
    """``Omega[y, c]`` is the fraction of class ``y`` predicted as ``c``; ``Omega[y, y] = 0``."""
    head = make_head(pairwise_delta=0.1)
    counts = torch.tensor([[6.0, 3.0, 1.0, 0.0, 0.0], [2.0, 8.0, 0.0, 0.0, 0.0]] + [[0.0] * 5] * 3)

    head.set_confusion(counts)

    assert float(head.confusion[0, 0]) == 0.0
    assert float(head.confusion[0, 1]) == pytest.approx(0.3)
    assert float(head.confusion[0, 2]) == pytest.approx(0.1)
    assert float(head.confusion[1, 0]) == pytest.approx(0.2)


def test_a_class_with_no_calibration_samples_does_not_divide_by_zero() -> None:
    head = make_head(pairwise_delta=0.1)

    head.set_confusion(torch.zeros(CLASSES, CLASSES))

    assert torch.isfinite(head.confusion).all()
    assert not head.confusion.any()


def test_the_pairwise_term_pushes_only_the_confused_classes() -> None:
    """The point of aiming it: class 1's logit drops, class 3's does not.

    A per-class additive margin lowers a sample's own logit relative to *all*
    89 others, which is the mechanism that shrinks its region. The pairwise
    term lowers the competitors it actually loses to, which does not.
    """
    delta = 0.5
    head = make_head(pairwise_delta=delta).train()
    counts = torch.zeros(CLASSES, CLASSES)
    counts[0, 0], counts[0, 1] = 5.0, 5.0  # class 0 is confused with 1, never with 3
    head.set_confusion(counts)

    x = torch.randn(2, DIM)
    y = torch.zeros(2, dtype=torch.long)
    baseline = make_head(pairwise_delta=0.0).train()
    baseline.load_state_dict({k: v for k, v in head.state_dict().items()})
    baseline.pairwise_delta = 0.0

    with_pairwise, without = head(x, y), baseline(x, y)

    assert (with_pairwise[:, 1] < without[:, 1]).all()
    assert torch.allclose(with_pairwise[:, 3], without[:, 3])
    assert torch.allclose(with_pairwise[:, 0], without[:, 0]), "the target column is not pushed"


def test_the_pairwise_term_never_reaches_inference() -> None:
    """``Omega`` is fitted on labels, so a labelled forward is the only place it belongs."""
    head = make_head(pairwise_delta=0.5)
    head.set_confusion(torch.ones(CLASSES, CLASSES))
    x = torch.randn(4, DIM)

    head.eval()
    with_labels = head(x, torch.zeros(4, dtype=torch.long))
    without = head(x)

    assert torch.equal(with_labels, without)


# ══════════════════════════════════════════════════════════════════════
#  The theta + m < pi/2 cap (M-7)
# ══════════════════════════════════════════════════════════════════════


def test_the_margin_is_capped_below_a_right_angle() -> None:
    """Past ``pi/2`` the gradient ``s*sin(theta + m)`` starts falling with the angle.

    A sample at ``theta = 1.5 rad`` plus a 0.5 rad margin would land at 2.0,
    well into the decay regime. The cap holds ``theta + m`` at ``pi/2``, where
    the gradient magnitude is maximal.
    """
    head = make_head().train()
    head.margins.fill_(0.5)

    # An embedding almost orthogonal to its class's sub-centre: theta ~ pi/2.
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 1.0  # class 0, sub-centre 0
        head.weight[1, 0] = 1.0
    x = torch.zeros(1, DIM)
    x[0, 0], x[0, 1] = math.cos(1.5), math.sin(1.5)
    y = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        logits = head(x, y)

    # cos(theta + m) with m capped at pi/2 - theta is cos(pi/2) = 0.
    assert float(logits[0, 0]) == pytest.approx(0.0, abs=1e-3)


def test_the_cap_leaves_ordinary_samples_alone() -> None:
    """It is a ceiling, not a schedule: a sample at 0.3 rad gets its full margin."""
    head = make_head().train()
    head.margins.fill_(0.4)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 1.0
        head.weight[1, 0] = 1.0
    theta = 0.3
    x = torch.zeros(1, DIM)
    x[0, 0], x[0, 1] = math.cos(theta), math.sin(theta)
    y = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        logits = head(x, y)

    assert float(logits[0, 0]) / head.s == pytest.approx(math.cos(theta + 0.4), abs=1e-3)


def test_the_cap_never_pushes_a_sample_past_a_right_angle() -> None:
    """``theta_y + m <= max(theta_y, pi/2)`` for every sample, at the largest margin.

    Two statements in one. A sample below ``pi/2`` is never pushed past it, so
    ``cos(theta + m) >= 0``. A sample *already* past it — its own class is on
    the far side of the sphere — receives no margin at all, because
    ``max(0, pi/2 - theta)`` is zero there, so its logit is the unmargined
    cosine rather than something even more negative.

    That is what makes the ``phi = cos(theta) - mm`` easy-margin guard the cap
    replaces unreachable: that branch only opened past ``pi - m``, and nothing
    now gets there.
    """
    head = make_head().train()
    head.margins.fill_(M_MAX)
    torch.manual_seed(3)
    x = torch.randn(64, DIM)
    y = torch.randint(0, CLASSES, (64,))

    with torch.no_grad():
        target = head(x, y).gather(1, y.view(-1, 1)).squeeze(1) / head.s
        plain = head.pool_subcentres(head.subcentre_cosines(x))
        bare = plain.gather(1, y.view(-1, 1)).squeeze(1)

    obtuse = bare < 0.0
    assert bool(obtuse.any()), "the batch must contain samples past pi/2 for this to bite"
    assert torch.allclose(
        target[obtuse], bare[obtuse], atol=1e-5
    ), "a sample already past pi/2 must receive no margin"
    assert (
        target[~obtuse] >= -1e-4
    ).all(), f"min acute-sample target cosine {float(target[~obtuse].min()):.4f}"
    assert pytest.approx(math.pi / 2) == HALF_PI
