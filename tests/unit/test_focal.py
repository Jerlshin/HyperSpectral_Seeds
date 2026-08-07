"""Focal × label smoothing — the modulator's range (**T1-3** / M-9 / OP-1).

``FocalLoss`` modulates the smoothed cross-entropy by ``(1 - p_t)**gamma``.
When ``p_t`` was taken as ``exp(-ce)`` and ``ce`` was the *smoothed* loss, the
modulator could not reach zero: smoothed CE is bounded below by the target
distribution's entropy ``H(q)``, so ``p_t <= exp(-H(q)) < 1``. At ``C=90``,
``gamma=1.5``, ``eps=0.10`` the floor is 0.3955 — focal loss degenerates into a
mild monotone rescaling of cross-entropy, and the "hard example mining" is
partly mining the smoothing entropy rather than the model's confidence
(IMPROVEMENT_PLAN §2.4.5, Appendix A.2).

The floors below are quoted from that table and re-derived here rather than
transcribed, so a change to either side is visible.

Stage 1 Phase 3 — the phase that produced the shipped checkpoint — ran at
``eps ≈ 0.051``, i.e. a floor of ≈0.20. Stages 2 and 3 pass ``eps = 0`` and are
unaffected; ``test_no_smoothing_path_is_bit_identical`` pins that.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from spectralquadnet.losses.focal import FocalLoss

NUM_CLASSES = 90

#: Every label-smoothing value the curriculum passes: the Stage-1 schedule's
#: endpoints (0.10 → 0.04), a mid-stage value, and Stage 2/3's zero.
SMOOTHINGS = [0.0, 0.04, 0.051, 0.07, 0.10]
GAMMAS = [1.0, 1.5, 2.0]


def logits_for_confidence(p_y: float, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """A ``(1, C)`` logit row whose softmax puts exactly ``p_y`` on class 0.

    With every non-target logit at 0, ``p_y = e^t / (e^t + C - 1)``. Built in
    float64: at ``p_y = 1 - 1e-6`` the quantity under test is ``1 - p_y``, and
    fp32 cancellation alone costs it three significant figures.
    """
    t = math.log((num_classes - 1) * p_y / (1.0 - p_y))
    z = torch.zeros(1, num_classes, dtype=torch.float64)
    z[0, 0] = t
    return z


def logits_matching_the_smoothed_target(ls: float, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Logits whose softmax *is* the smoothed target ``q`` — where smoothed CE attains ``H(q)``."""
    q = torch.full((1, num_classes), ls / (num_classes - 1), dtype=torch.float64)
    q[0, 0] = 1.0 - ls
    return q.log()


def modulator(logits: torch.Tensor, gamma: float, ls: float) -> float:
    """The factor ``FocalLoss`` applies to the smoothed CE, recovered by division.

    ``gamma = 0`` makes the modulator identically 1, so the same class supplies
    the denominator — no second implementation of the smoothed CE to drift.
    """
    y = torch.zeros(1, dtype=torch.long)
    focal = FocalLoss(gamma=gamma, label_smoothing=ls)(logits, y)
    ce = FocalLoss(gamma=0.0, label_smoothing=ls)(logits, y)
    return float(focal / ce)


def legacy_modulator(logits: torch.Tensor, gamma: float, ls: float) -> float:
    """The pre-Tier-1 modulator, ``(1 - exp(-ce_smoothed))**gamma``."""
    y = torch.zeros(1, dtype=torch.long)
    ce = FocalLoss(gamma=0.0, label_smoothing=ls)(logits, y)
    return float((1.0 - torch.exp(-ce)) ** gamma)


def entropy_floor(gamma: float, ls: float, num_classes: int = NUM_CLASSES) -> float:
    """``(1 - exp(-H(q)))**gamma`` — the bound the old modulator could not pass."""
    if ls == 0.0:
        return 0.0
    h = -(1 - ls) * math.log(1 - ls) - ls * math.log(ls / (num_classes - 1))
    return float((1.0 - math.exp(-h)) ** gamma)


# ══════════════════════════════════════════════════════════════════════
#  The §4.3 gate
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("ls", SMOOTHINGS)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_focal_modulator_reaches_zero(gamma, ls) -> None:
    """``(1 - p_t)**gamma < 1e-3`` at ``p_y = 1 - 1e-6``, for every smoothing used."""
    assert modulator(logits_for_confidence(1 - 1e-6), gamma, ls) < 1e-3


@pytest.mark.parametrize("ls", [0.04, 0.051, 0.07, 0.10])
@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_old_modulator_could_not_reach_zero(gamma, ls) -> None:
    """The defect: a perfectly-confident sample still carried most of its loss.

    Guards the test above from passing vacuously — the old formula fails it by
    two to three orders of magnitude on the very same input.
    """
    assert legacy_modulator(logits_for_confidence(1 - 1e-6), gamma, ls) > 0.1


@pytest.mark.parametrize("ls", [0.04, 0.051, 0.07, 0.10])
@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_old_modulator_bottoms_out_at_the_entropy_floor(gamma, ls) -> None:
    """Where the bound of Appendix A.2 is attained, and that nothing goes below it.

    ``(1 - exp(-l))**gamma >= (1 - exp(-H(q)))**gamma`` because smoothed CE is
    minimised at ``p = q``. The first assertion pins the bound as an identity at
    that point; the second checks it really is a floor across the confidence
    range, which is what makes the modulator's dynamic range finite.
    """
    floor = entropy_floor(gamma, ls)
    at_the_minimum = legacy_modulator(logits_matching_the_smoothed_target(ls), gamma, ls)
    assert at_the_minimum == pytest.approx(floor, rel=1e-9)

    for p_y in (0.05, 0.5, 0.9, 0.99, 1 - 1e-6):
        assert legacy_modulator(logits_for_confidence(p_y), gamma, ls) >= floor - 1e-9


def test_the_quoted_floor_table_still_holds() -> None:
    """§2.4.5's table, at ``C = 90``, ``gamma = 1.5``.

    The ``eps = 0.07`` row is quoted as 0.2857 from ``H(q) = 0.5686``; the exact
    entropy at ``C = 90`` is 0.56784, giving 0.28518. A 5e-4 discrepancy in the
    plan's own arithmetic, not in the loss — hence the 1e-3 tolerance on that
    row alone.
    """
    assert entropy_floor(1.5, 0.10) == pytest.approx(0.3955, abs=5e-4)
    assert entropy_floor(1.5, 0.04) == pytest.approx(0.1591, abs=5e-4)
    assert entropy_floor(1.5, 0.07) == pytest.approx(0.2857, abs=1e-3)
    assert entropy_floor(1.5, 0.0) == 0.0


# ══════════════════════════════════════════════════════════════════════
#  The modulator now means what its name says
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("ls", SMOOTHINGS)
def test_modulator_is_one_minus_p_y_to_the_gamma(ls) -> None:
    """It is a function of the model's unsmoothed confidence, and of nothing else."""
    for p_y in (0.01, 0.25, 0.5, 0.9, 0.999):
        z = logits_for_confidence(p_y)
        assert modulator(z, 1.5, ls) == pytest.approx((1.0 - p_y) ** 1.5, rel=1e-4)


@pytest.mark.parametrize("ls", SMOOTHINGS)
def test_modulator_decreases_monotonically_with_confidence(ls) -> None:
    values = [modulator(logits_for_confidence(p), 1.5, ls) for p in (0.05, 0.3, 0.6, 0.9, 0.99)]
    assert all(a > b for a, b in pairwise(values))


# ══════════════════════════════════════════════════════════════════════
#  What must not have changed
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.regression
@pytest.mark.parametrize("gamma", GAMMAS)
def test_no_smoothing_path_is_bit_identical(gamma) -> None:
    """At ``ls = 0``, ``p_y`` and ``exp(-ce)`` are the same float.

    Stages 2 and 3 pass no smoothing, so this fix must leave their loss
    untouched — not close, identical.
    """
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(64, NUM_CLASSES, generator=gen) * 4.0
    y = torch.randint(0, NUM_CLASSES, (64,), generator=gen)

    logp = torch.log_softmax(z, dim=1)
    ce = torch.nn.functional.nll_loss(logp, y, reduction="none")
    old = ((1.0 - torch.exp(-ce)) ** gamma * ce).mean()

    assert FocalLoss(gamma=gamma)(z, y).item() == old.item()


def test_gamma_zero_is_plain_smoothed_cross_entropy() -> None:
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(32, NUM_CLASSES, generator=gen)
    y = torch.randint(0, NUM_CLASSES, (32,), generator=gen)
    reference = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(z, y)

    assert FocalLoss(gamma=0.0, label_smoothing=0.1)(z, y) == pytest.approx(
        reference.item(), rel=1e-5
    )


def test_loss_stays_finite_at_the_confidence_extremes() -> None:
    """A saturated or a hopeless sample must not produce a NaN the epoch loop skips."""
    y = torch.zeros(1, dtype=torch.long)
    for p_y in (1e-12, 1 - 1e-12):
        for ls in SMOOTHINGS:
            loss = FocalLoss(gamma=1.5, label_smoothing=ls)(logits_for_confidence(p_y), y)
            assert torch.isfinite(loss)
