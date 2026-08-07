"""The ArcFace cosine clamp (**T1-7** / N-4 / HD-4).

``sine = sqrt(1 - cos^2)`` has derivative ``-c / sqrt(1 - c^2)``, which diverges
as the cosine approaches 1. The clamp is what bounds it: at the original
``1e-6`` the bound is **707**, at ``1e-3`` it is **22.4**, and the angular
resolution given up is 2.6° against margins of 20–26°
(IMPROVEMENT_PLAN §2.4.4, §3.5 HD-4).

**T1-7's stated validation criterion — "max observed head grad-norm falls by
≥10×" — does not hold, and cannot.** §2.4.4 reads the 707 off the sine in
isolation, but the head normalises both its embedding and its weights, so the
gradient that actually reaches them is the *tangential* one, and the tangential
factor vanishes at exactly the rate the sine derivative diverges. Written out,
``phi = cos(theta + m)``, so

    d(s * phi) / d(theta) = -s * sin(theta + m),      |.| <= s

with no singularity anywhere. ``test_head_gradient_is_bounded_by_s`` measures
that identity to 1e-4 at *both* clamps; the batch grad-norm ratio is ≈1.15, not
≥10. The change is worth making on the grounds the plan's own Δ column already
gives it — "≈ 0 (stability)" — and those grounds are pinned below: near the old
clamp ``1 - cos^2`` loses 5 significant figures to fp32 cancellation, so the
sine feeding the margin is numerically meaningless.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectralquadnet.models import heads
from spectralquadnet.models.heads import COS_CLAMP_EPS, AdaptiveSubcenterArcFaceHead

IN_DIM, NUM_CLASSES, SUBCENTRES, SCALE = 16, 8, 3, 48.0
M_BASE = 0.35

#: The clamp before Tier 1, kept only as the comparison arm.
LEGACY_EPS = 1e-6

#: ``1 - cos`` for the swept batch: three samples inside both clamps, three
#: between them, three outside both.
GAPS = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def max_sine_derivative(eps: float) -> float:
    """``|d/dc sqrt(1 - c^2)|`` at the clamp boundary ``c = 1 - eps``, in float64."""
    c = torch.tensor(1.0 - eps, dtype=torch.float64, requires_grad=True)
    torch.sqrt(1 - c**2).backward()
    assert c.grad is not None
    return abs(float(c.grad))


def build_head() -> AdaptiveSubcenterArcFaceHead:
    torch.manual_seed(0)
    head = AdaptiveSubcenterArcFaceHead(
        IN_DIM, NUM_CLASSES, K=SUBCENTRES, s=SCALE, m_base=M_BASE, m_delta=0.10
    )
    return head.train()


def saturating_batch(head: AdaptiveSubcenterArcFaceHead) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit embeddings whose cosine to their own sub-centre is ``1 - GAPS[i]``.

    Built as ``w*cos + u*sin`` with ``u`` a unit vector orthogonal to ``w``, so
    the cosine is exact by construction rather than fitted.
    """
    labels = torch.arange(len(GAPS)) % NUM_CLASSES
    w = torch.nn.functional.normalize(head.weight.detach(), dim=1)

    rows = []
    for gap, label in zip(GAPS, labels.tolist(), strict=True):
        target = w[label * head.K]  # sub-centre 0 of that class
        perp = torch.randn(IN_DIM)
        perp = perp - (perp @ target) * target
        perp = perp / perp.norm()
        cos = 1.0 - gap
        rows.append(target * cos + perp * math.sqrt(max(1.0 - cos * cos, 0.0)))
    return torch.stack(rows), labels


def target_logit_grads(eps: float, monkeypatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample ``|dL/dx|`` and the head's total ``|dL/dW|``, for ``L = -sum(target logit)``.

    Fixing ``dL/d(logit) = -1`` isolates the amplification the head's own
    Jacobian manufactures from how confident the model happens to be — under a
    cross-entropy loss a saturated cosine also drives ``dL/d(logit)`` to zero,
    which would mask the effect rather than measure it.
    """
    head = build_head()
    x, y = saturating_batch(head)
    x = x.clone().requires_grad_(True)

    monkeypatch.setattr(heads, "COS_CLAMP_EPS", eps)
    head.zero_grad(set_to_none=True)
    (-head(x, y).gather(1, y.view(-1, 1)).sum()).backward()

    assert x.grad is not None and head.weight.grad is not None
    return x.grad.norm(dim=1), head.weight.grad.norm()


# ══════════════════════════════════════════════════════════════════════
#  The isolated derivative claim — true as stated
# ══════════════════════════════════════════════════════════════════════


def test_the_shipped_clamp_is_the_loosened_one() -> None:
    assert COS_CLAMP_EPS == 1e-3


def test_clamp_bounds_the_sine_derivative_at_the_quoted_values() -> None:
    """§2.4.4's two numbers: 707 at ``1e-6``, 22.4 at ``1e-3``.

    Exactly, the boundary derivatives are 707.106 and 22.344; the plan rounds
    the second up, hence the wider tolerance on it.
    """
    assert max_sine_derivative(LEGACY_EPS) == pytest.approx(707.0, rel=1e-3)
    assert max_sine_derivative(COS_CLAMP_EPS) == pytest.approx(22.4, rel=5e-3)
    assert max_sine_derivative(LEGACY_EPS) / max_sine_derivative(COS_CLAMP_EPS) > 30


def test_the_angular_resolution_given_up_is_immaterial() -> None:
    """2.6° against per-class margins of 20–26°, so no decision boundary moves."""
    lost = math.degrees(math.acos(1 - COS_CLAMP_EPS))
    assert lost == pytest.approx(2.56, abs=0.01)
    assert lost < math.degrees(M_BASE), "smaller than even the smallest margin, m_base"


# ══════════════════════════════════════════════════════════════════════
#  Why the ≥10x grad-norm criterion is refuted
# ══════════════════════════════════════════════════════════════════════


def exact_tangential_grad(gap: float) -> float:
    """``s * sin(theta + m)`` — the analytic gradient magnitude at ``cos theta = 1 - gap``."""
    return SCALE * math.sin(math.acos(1.0 - gap) + M_BASE)


@pytest.mark.parametrize("eps", [LEGACY_EPS, 1e-3])
def test_head_gradient_is_bounded_by_s(eps, monkeypatch) -> None:
    """``|d(s*phi)/dx| = s*sin(theta + m)`` for every unclamped sample, at either clamp.

    The head normalises ``x``, so the gradient reaching it is the tangential
    one; the ``1/sin(theta)`` in the sine derivative is multiplied by the
    ``sin(theta)`` in ``d(cos)/dx`` and cancels exactly. This is the identity
    that makes N-4's ``3.4e4`` amplification unreachable.

    The identity is checked where fp32 can represent it — at gaps below ~1e-4
    the *measured* value drifts off it, which is the ill-conditioning the next
    test isolates, not a failure of the algebra.
    """
    per_sample, _ = target_logit_grads(eps, monkeypatch)

    for gap, observed in zip(GAPS, per_sample.tolist(), strict=True):
        assert observed <= SCALE, f"gap={gap:g}"
        if gap >= max(1e-4, 2 * eps):  # unclamped, and representable in fp32
            assert observed == pytest.approx(exact_tangential_grad(gap), rel=1e-3), f"gap={gap:g}"


def test_the_old_clamp_admitted_numerically_meaningless_gradients(monkeypatch) -> None:
    """Every sample the new clamp admits matches the exact identity; the old one's did not.

    The samples between the two clamps are precisely those whose ``sqrt(1 -
    cos^2)`` fp32 cannot compute — and the margin is built from that sine, so
    their contribution to the update is noise of their own magnitude.
    """

    def worst_deviation(eps: float) -> float:
        per_sample, _ = target_logit_grads(eps, monkeypatch)
        admitted = [
            abs(g - exact_tangential_grad(gap)) / exact_tangential_grad(gap)
            for gap, g in zip(GAPS, per_sample.tolist(), strict=True)
            if g > 0.0
        ]
        assert admitted, "the batch must admit at least one sample"
        return max(admitted)

    assert worst_deviation(LEGACY_EPS) > 5e-3
    assert worst_deviation(COS_CLAMP_EPS) < 1e-3


def test_the_batch_grad_norm_does_not_fall_tenfold(monkeypatch) -> None:
    """T1-7's criterion as written, measured — it lands near 1.15x, not 10x.

    Recorded so the plan's claim and the code's behaviour cannot drift apart
    silently: if a future change *does* buy a large reduction here, this test
    fails and the reason has to be written down.
    """
    _, legacy = target_logit_grads(LEGACY_EPS, monkeypatch)
    _, fixed = target_logit_grads(COS_CLAMP_EPS, monkeypatch)

    assert 1.0 < float(legacy / fixed) < 2.0


def test_samples_inside_the_clamp_contribute_no_gradient(monkeypatch) -> None:
    """What the wider clamp actually changes: the size of the dead zone.

    A sample whose cosine exceeds ``1 - eps`` is clamped, and ``clamp`` has zero
    derivative outside its range — so it contributes nothing at all. Widening
    the clamp moves that boundary from ``1 - 1e-6`` to ``1 - 1e-3``, retiring
    the samples whose ``sqrt(1 - cos^2)`` is numerically meaningless anyway.
    """
    legacy, _ = target_logit_grads(LEGACY_EPS, monkeypatch)
    fixed, _ = target_logit_grads(COS_CLAMP_EPS, monkeypatch)

    # Gaps within a factor of two of a clamp are left unasserted: at those
    # magnitudes which side of the boundary an fp32 cosine lands on is decided
    # by rounding, not by the construction.
    for gap, l_grad, f_grad in zip(GAPS, legacy.tolist(), fixed.tolist(), strict=True):
        if gap <= COS_CLAMP_EPS / 2:
            assert f_grad == 0.0, f"gap={gap:g} must be inside the new clamp"
        elif gap >= COS_CLAMP_EPS * 2:
            assert f_grad > 0.0, f"gap={gap:g} must survive the new clamp"
        if gap >= LEGACY_EPS * 2:
            assert l_grad > 0.0, f"gap={gap:g} survived the old clamp"


def test_the_sine_is_ill_conditioned_at_the_old_clamp() -> None:
    """The numerical case for HD-4, which is the one that survives measurement.

    ``1 - c^2`` at ``c = 1 - 1e-6`` is a difference of two numbers agreeing to
    six digits: fp32 keeps about two of the result's, so the sine — and every
    margin computed from it — carries ~0.7% error. At ``1e-3`` the same
    quantity is accurate to 1e-5.
    """

    def relative_error(eps: float) -> float:
        c64 = torch.tensor(1.0 - eps, dtype=torch.float64)
        exact = float(torch.sqrt(1 - c64**2))
        c32 = torch.tensor(1.0 - eps, dtype=torch.float32)
        fp32 = float(torch.sqrt(torch.clamp(1 - c32**2, min=1e-6)))
        return abs(fp32 - exact) / exact

    assert relative_error(LEGACY_EPS) > 5e-3
    assert relative_error(COS_CLAMP_EPS) < 1e-4
    assert relative_error(LEGACY_EPS) / relative_error(COS_CLAMP_EPS) > 100


# ══════════════════════════════════════════════════════════════════════
#  What must not have changed
# ══════════════════════════════════════════════════════════════════════


def test_clamp_holds_the_cosine_inside_its_bounds() -> None:
    """Whatever the input, inference logits stay within ``+/- s * (1 - eps)``."""
    head = build_head().eval()
    x, _ = saturating_batch(head)

    assert head(x).abs().max().item() <= SCALE * (1 - COS_CLAMP_EPS) + 1e-4


def test_inference_logits_barely_move(monkeypatch) -> None:
    """A stability fix, not a re-scoring: no test-set prediction can change.

    Only cosines beyond ``1 - 1e-3`` are touched at all, and by at most
    ``s * (1e-3 - 1e-6) = 0.048`` of logit.
    """
    head = build_head().eval()
    x, _ = saturating_batch(head)

    monkeypatch.setattr(heads, "COS_CLAMP_EPS", LEGACY_EPS)
    legacy = head(x)
    monkeypatch.setattr(heads, "COS_CLAMP_EPS", 1e-3)
    fixed = head(x)

    assert (fixed - legacy).abs().max().item() <= SCALE * (1e-3 - LEGACY_EPS) + 1e-4
    assert torch.equal(legacy.argmax(1), fixed.argmax(1))
