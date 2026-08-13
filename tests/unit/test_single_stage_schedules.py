"""IC-11 — the collapsed curriculum's three schedules, pinned across their range.

CHANGES §17 replaces three stages with one, and the mechanism that makes that
possible is a *scheduling* property, not an architectural one: mixup and a
non-zero angular margin are mutually exclusive (the head cannot index a
per-sample margin by an interpolated label pair), and that incompatibility is
the only reason Stage 2 ever needed to be a separate stage. Switch mixup off at
epoch 110, warm one scalar margin in over 111–130, and the transition happens
inside one stage with one optimiser state and no EMA re-initialisation.

``test_mixup_and_a_margin_never_overlap`` is the load-bearing test in this file.
If it ever fails, the single-stage design is unsound and ``train_one_epoch``
will raise at runtime.
"""

from __future__ import annotations

import pytest

from spectralquadnet.optim.schedulers import (
    single_stage_label_smoothing,
    single_stage_lr,
    single_stage_margin,
    single_stage_uses_mixup,
)


@pytest.fixture
def scfg(cfg_default):
    return cfg_default


# ══════════════════════════════════════════════════════════════════════
#  The invariant the whole collapse rests on
# ══════════════════════════════════════════════════════════════════════


def test_mixup_and_a_margin_never_overlap(scfg) -> None:
    """The single-stage design is only valid because these two are disjoint.

    ``train_one_epoch`` raises ``ValueError`` on the combination, so an overlap
    here is not a subtle numerical issue — it is a run that dies at epoch 111.
    """
    for ep in range(1, int(scfg.single.epochs) + 1):
        if single_stage_uses_mixup(ep, scfg):
            assert single_stage_margin(ep, scfg) == 0.0, f"epoch {ep}: mixup with a margin"


def test_the_margin_window_opens_after_mixup_stops(scfg) -> None:
    assert int(scfg.single.margin_warmup_start) > int(scfg.single.mixup_epochs)


# ══════════════════════════════════════════════════════════════════════
#  Margin
# ══════════════════════════════════════════════════════════════════════


def test_the_margin_is_exactly_zero_before_the_window(scfg) -> None:
    """Not merely small: zero selects the head's plain cosine path, which is
    what makes it a NormFace classifier and mixup admissible."""
    for ep in (1, 50, int(scfg.single.margin_warmup_start) - 1):
        assert single_stage_margin(ep, scfg) == 0.0


def test_the_margin_reaches_its_target_and_stays(scfg) -> None:
    target = float(scfg.single.arcface_m)
    end = int(scfg.single.margin_warmup_end)
    assert single_stage_margin(end, scfg) == pytest.approx(target)
    assert single_stage_margin(end + 20, scfg) == pytest.approx(target)
    assert single_stage_margin(int(scfg.single.epochs), scfg) == pytest.approx(target)


def test_the_margin_ramp_is_monotone(scfg) -> None:
    start, end = int(scfg.single.margin_warmup_start), int(scfg.single.margin_warmup_end)
    values = [single_stage_margin(ep, scfg) for ep in range(start, end + 1)]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    assert values[0] == 0.0 and values[-1] == pytest.approx(float(scfg.single.arcface_m))


def test_the_ramp_is_cosine_so_it_starts_flat(scfg) -> None:
    """Linear would be a step in the objective's curvature; the margin's effect
    on the loss is steepest near zero."""
    start, end = int(scfg.single.margin_warmup_start), int(scfg.single.margin_warmup_end)
    mid = (start + end) // 2
    half_target = float(scfg.single.arcface_m) / 2
    assert single_stage_margin(mid, scfg) == pytest.approx(half_target, abs=0.03)
    first_step = single_stage_margin(start + 1, scfg) - single_stage_margin(start, scfg)
    middle_step = single_stage_margin(mid + 1, scfg) - single_stage_margin(mid, scfg)
    assert first_step < middle_step


# ══════════════════════════════════════════════════════════════════════
#  Learning rate
# ══════════════════════════════════════════════════════════════════════


def test_the_lr_warms_up_then_decays_once(scfg) -> None:
    lam = single_stage_lr(scfg)
    warmup = int(scfg.single.warmup_ep)
    # LambdaLR takes a 0-based index.
    warm = [lam(i) for i in range(warmup)]
    assert all(b > a for a, b in zip(warm, warm[1:], strict=False)), "warm-up rises"
    assert warm[0] > 0.0, "epoch 1 must not be a wasted epoch at lr=0"

    after = [lam(i) for i in range(warmup, int(scfg.single.epochs))]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:], strict=False)), "then decays"


def test_the_lr_has_no_restarts(scfg) -> None:
    """SGDR's restart at Stage-2 epoch 28 was followed by 21 stale epochs and
    then early stopping. One schedule, one thing to reason about."""
    lam = single_stage_lr(scfg)
    values = [lam(i) for i in range(int(scfg.single.warmup_ep), int(scfg.single.epochs))]
    rises = [b - a for a, b in zip(values, values[1:], strict=False) if b > a + 1e-9]
    assert not rises, f"the LR rose {len(rises)} times after warm-up"


def test_the_lr_lands_on_min_lr_at_the_end(scfg) -> None:
    lam = single_stage_lr(scfg)
    floor = float(scfg.single.min_lr) / float(scfg.single.max_lr)
    assert lam(int(scfg.single.epochs) - 1) == pytest.approx(floor, abs=1e-4)


def test_the_multiplier_never_leaves_the_unit_interval(scfg) -> None:
    lam = single_stage_lr(scfg)
    for i in range(int(scfg.single.epochs) + 20):
        assert 0.0 < lam(i) <= 1.0 + 1e-12


# ══════════════════════════════════════════════════════════════════════
#  Label smoothing and mixup
# ══════════════════════════════════════════════════════════════════════


def test_label_smoothing_decays_linearly_between_its_bounds(scfg) -> None:
    hi, lo = float(scfg.single.label_smooth_hi), float(scfg.single.label_smooth_lo)
    total = int(scfg.single.epochs)
    assert single_stage_label_smoothing(1, scfg) == pytest.approx(hi)
    assert single_stage_label_smoothing(total, scfg) == pytest.approx(lo)
    assert single_stage_label_smoothing((1 + total) // 2, scfg) == pytest.approx(
        (hi + lo) / 2, abs=1e-3
    )


def test_mixup_covers_the_epochs_it_says_it_does(scfg) -> None:
    cutoff = int(scfg.single.mixup_epochs)
    assert single_stage_uses_mixup(1, scfg)
    assert single_stage_uses_mixup(cutoff, scfg)
    assert not single_stage_uses_mixup(cutoff + 1, scfg)
    assert not single_stage_uses_mixup(int(scfg.single.epochs), scfg)


def test_every_epoch_of_the_budget_is_defined(scfg) -> None:
    """No index error, no NaN, anywhere in the range — including past the budget,
    which the optional Phase B tail reaches."""
    for ep in range(1, int(scfg.single.epochs) + 40):
        margin = single_stage_margin(min(ep, int(scfg.single.epochs)), scfg)
        smoothing = single_stage_label_smoothing(min(ep, int(scfg.single.epochs)), scfg)
        assert 0.0 <= margin <= float(scfg.single.arcface_m) + 1e-9
        assert (
            float(scfg.single.label_smooth_lo) - 1e-9
            <= smoothing
            <= float(scfg.single.label_smooth_hi) + 1e-9
        )
