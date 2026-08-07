"""GradNorm-balanced auxiliary weights (**T2-6** / OP-2).

``_compute_aux_loss`` weighted branches A and B at 2x and C/D at 1x, a constant
chosen to compensate for a measured gradient collapse in the spectral branches.
The number is not wrong; it is *unfalsifiable*. It has to be re-tuned by hand
whenever anything upstream changes, and between re-tunings it over- or
under-corrects silently.

The GradNorm rule computes the same compensation from the per-branch gradient
norms the loops already log:

    omega_b <- omega_b * (g_bar / g_b) ** alpha,   g_bar = mean_b g_b,  alpha = 0.5

T2-6's criterion has two halves. ``omega`` becomes a logged time series —
:func:`test_the_weights_reach_the_tracker` — and the branch norms converge
rather than staying separated by the hardcoded factor —
:func:`test_repeated_updates_drive_the_norms_together`.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.losses.auxiliary import (
    AUX_WEIGHT_BOUNDS,
    DEFAULT_BRANCH_WEIGHTS,
    GradNormAuxWeights,
    _compute_aux_loss,
)

pytestmark = pytest.mark.regression

BRANCHES = ("a", "b", "c", "d")


def norms(**kwargs: float) -> dict[str, float]:
    return {f"branch_{k}": v for k, v in kwargs.items()}


# ══════════════════════════════════════════════════════════════════════
#  The rule
# ══════════════════════════════════════════════════════════════════════


def test_it_starts_at_the_vector_it_replaces() -> None:
    """Epoch 1 is byte-identical to the constant, so nothing changes before evidence arrives."""
    assert GradNormAuxWeights().weights == DEFAULT_BRANCH_WEIGHTS


def test_a_weak_branch_is_weighted_up_and_a_strong_one_down() -> None:
    """The sign, on one update. ``omega_b`` moves inversely to ``g_b``."""
    gradnorm = GradNormAuxWeights(alpha=0.5, init=dict.fromkeys(("aux_a", "aux_b"), 1.0))

    updated = gradnorm.update(norms(a=0.25, b=4.0))

    assert updated["aux_a"] > 1.0
    assert updated["aux_b"] < 1.0


def test_the_update_matches_the_written_formula() -> None:
    """``omega * (g_bar / g_b) ** alpha``, to the float."""
    gradnorm = GradNormAuxWeights(alpha=0.5, init={"aux_a": 2.0, "aux_b": 1.0})

    updated = gradnorm.update(norms(a=1.0, b=3.0))

    mean = 2.0
    assert updated["aux_a"] == pytest.approx(2.0 * (mean / 1.0) ** 0.5)
    assert updated["aux_b"] == pytest.approx(1.0 * (mean / 3.0) ** 0.5)


def test_alpha_zero_is_the_pre_tier_2_behaviour_exactly() -> None:
    """The off switch has to be a true no-op, or "GradNorm off" is not a control."""
    gradnorm = GradNormAuxWeights(alpha=0.0)

    for _ in range(10):
        gradnorm.update(norms(a=0.01, b=100.0, c=1.0, d=1.0))

    assert gradnorm.weights == DEFAULT_BRANCH_WEIGHTS


def test_repeated_updates_drive_the_norms_together() -> None:
    """T2-6's criterion: A/B norms converge towards C/D's without the hardcoded 2x.

    Modelled as ``g_b ∝ omega_b * s_b``, i.e. a branch's gradient scales with
    the weight it is given — which is what makes the loop closable at all. The
    spread of the resulting norms must shrink.
    """
    sensitivity = {"a": 0.1, "b": 0.2, "c": 1.0, "d": 1.2}
    gradnorm = GradNormAuxWeights(alpha=0.5)

    def spread() -> float:
        g = [gradnorm.weights[f"aux_{b}"] * sensitivity[b] for b in BRANCHES]
        return max(g) / min(g)

    start = spread()
    for _ in range(30):
        gradnorm.update(
            norms(**{b: gradnorm.weights[f"aux_{b}"] * sensitivity[b] for b in BRANCHES})
        )

    assert spread() < start
    assert spread() == pytest.approx(1.0, abs=0.05), "the norms should end up balanced"


def test_the_weights_are_bounded() -> None:
    """A branch whose gradient vanishes for one epoch must not take the loop with it."""
    low, high = AUX_WEIGHT_BOUNDS
    gradnorm = GradNormAuxWeights(alpha=0.5)

    for _ in range(50):
        gradnorm.update(norms(a=1e-9, b=1.0, c=1.0, d=1e9))

    assert all(low <= v <= high for v in gradnorm.weights.values())


def test_a_branch_with_no_gradient_keeps_its_weight() -> None:
    """Absent or zero norms carry no information about the balance, so they change nothing."""
    gradnorm = GradNormAuxWeights(alpha=0.5)

    gradnorm.update(norms(a=1.0, b=2.0, c=0.0))

    assert gradnorm.weights["aux_c"] == DEFAULT_BRANCH_WEIGHTS["aux_c"]
    assert gradnorm.weights["aux_d"] == DEFAULT_BRANCH_WEIGHTS["aux_d"]


def test_a_single_observed_branch_is_not_enough_to_rebalance() -> None:
    """With one norm the mean *is* that norm, so the update would be a no-op with rounding."""
    gradnorm = GradNormAuxWeights(alpha=0.5)

    gradnorm.update(norms(a=5.0))

    assert gradnorm.weights == DEFAULT_BRANCH_WEIGHTS


def test_the_weights_reach_the_tracker() -> None:
    """T2-6's other half: ``omega`` becomes a logged time series."""
    gradnorm = GradNormAuxWeights(alpha=0.5)
    gradnorm.update(norms(a=0.5, b=0.5, c=2.0, d=2.0))

    scalars = gradnorm.scalars()

    assert set(scalars) == {f"aux_weight/branch_{b}" for b in BRANCHES}
    assert scalars["aux_weight/branch_a"] > scalars["aux_weight/branch_c"]


# ══════════════════════════════════════════════════════════════════════
#  The consumer
# ══════════════════════════════════════════════════════════════════════


def aux_output(scale: float = 1.0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(5)
    return {f"aux_{b}": torch.randn(4, 6, generator=gen) * scale for b in BRANCHES}


def test_compute_aux_loss_defaults_to_the_fixed_vector() -> None:
    """Passing no weights must reproduce the pre-Tier-2 number exactly."""
    out, y = aux_output(), torch.tensor([0, 1, 2, 3])
    criterion = torch.nn.CrossEntropyLoss()

    default = _compute_aux_loss(criterion, out, y, y, 1.0, use_mixup=False)
    explicit = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, weights=DEFAULT_BRANCH_WEIGHTS
    )

    assert torch.equal(default, explicit)


def test_compute_aux_loss_honours_supplied_weights() -> None:
    out, y = aux_output(), torch.tensor([0, 1, 2, 3])
    criterion = torch.nn.CrossEntropyLoss()

    _, components = _compute_aux_loss(
        criterion,
        out,
        y,
        y,
        1.0,
        use_mixup=False,
        return_components=True,
        weights={"aux_a": 4.0, "aux_b": 2.0, "aux_c": 1.0, "aux_d": 1.0},
    )

    unweighted = criterion(out["aux_a"], y)
    assert components["aux_a"] == pytest.approx(float(unweighted) * 4.0, rel=1e-6)


def test_a_partial_weight_dict_falls_back_per_branch() -> None:
    """A caller that names only one branch leaves the other three at their defaults."""
    out, y = aux_output(), torch.tensor([0, 1, 2, 3])
    criterion = torch.nn.CrossEntropyLoss()

    _, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True, weights={"aux_a": 3.0}
    )

    unweighted_d = criterion(out["aux_d"], y)
    assert components["aux_d"] == pytest.approx(float(unweighted_d) * 1.0, rel=1e-6)
