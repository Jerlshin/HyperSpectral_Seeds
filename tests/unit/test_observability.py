"""IC-2 and IC-6 — the two panels that described the controller, not the model.

**IC-2.** ``loss/branch_*`` carried :math:`\\omega_b \\mathcal L_b`, and
:math:`\\omega_b` spends most of training pinned at a clip bound. So Branch A's
curve crashing from 11 to ~1 at epoch 10 was *the weight hitting its 0.25 floor*,
not the branch learning: at epoch 10 overall training accuracy was 15.9%, and an
aux CE of 1.0 would imply that branch alone was at ~70% train accuracy. The two
are irreconcilable and the weighted-quantity reading resolves it (CHANGES §10.2).

**IC-6.** The clip fired on essentially every step — backbone pre-clip norms of
25–50 against a threshold of 1.0 — so the backbone was doing normalised-gradient
descent at a fixed step size and the LR schedule's *shape* was interacting with a
direction-only update (§8.1). Nothing measured that, because only the pre-clip
norm was logged and a pre-clip norm alone cannot say whether the clip bound.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralquadnet.losses.auxiliary import AuxComponents, _compute_aux_loss
from spectralquadnet.optim.param_groups import ClipReport, clip_grad_norm_by_group


def _aux_inputs() -> tuple[nn.Module, dict[str, torch.Tensor], torch.Tensor]:
    torch.manual_seed(0)
    out = {k: torch.randn(6, 5) for k in ("main", "aux_a", "aux_b", "aux_c", "aux_d")}
    return nn.CrossEntropyLoss(), out, torch.randint(0, 5, (6,))


# ══════════════════════════════════════════════════════════════════════
#  IC-2 — raw and weighted
# ══════════════════════════════════════════════════════════════════════


def test_components_expose_both_the_raw_and_the_weighted_term() -> None:
    criterion, out, y = _aux_inputs()
    _, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    assert isinstance(components, AuxComponents)
    assert set(components.raw) == {"aux_a", "aux_b", "aux_c", "aux_d"}
    assert set(components.weighted) == set(components.raw)


def test_weighted_equals_raw_times_the_branch_weight_to_1e_6() -> None:
    """IC-2's stated validation criterion, verbatim."""
    criterion, out, y = _aux_inputs()
    weights = {"aux_a": 4.0, "aux_b": 0.25, "aux_c": 1.0, "aux_d": 2.5}
    _, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True, weights=weights
    )
    for key, weight in weights.items():
        assert components.weighted[key].item() == pytest.approx(
            components.raw[key].item() * weight, abs=1e-6
        )


def test_the_raw_term_is_invariant_to_the_weight_and_the_weighted_term_is_not() -> None:
    """The whole point: a branch's *health* must be readable without the controller.

    Two runs with wildly different weight vectors see the same underlying
    branch losses. Only the weighted series moves — which is what made the
    audited panel unreadable.
    """
    criterion, out, y = _aux_inputs()
    _, floor = _compute_aux_loss(
        criterion,
        out,
        y,
        y,
        1.0,
        use_mixup=False,
        return_components=True,
        weights=dict.fromkeys(("aux_a", "aux_b", "aux_c", "aux_d"), 0.25),
    )
    _, ceiling = _compute_aux_loss(
        criterion,
        out,
        y,
        y,
        1.0,
        use_mixup=False,
        return_components=True,
        weights=dict.fromkeys(("aux_a", "aux_b", "aux_c", "aux_d"), 4.0),
    )
    for key in floor.raw:
        assert floor.raw[key].item() == pytest.approx(ceiling.raw[key].item(), abs=1e-9)
        assert floor.weighted[key].item() != pytest.approx(ceiling.weighted[key].item())


def test_the_total_is_unchanged_by_asking_for_components() -> None:
    """The existing bit-identity gate, restated against the new return type."""
    criterion, out, y = _aux_inputs()
    total_only = _compute_aux_loss(criterion, out, y, y, 1.0, use_mixup=False)
    total, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    assert total.item() == total_only.item()
    assert torch.allclose(sum(components.values()), total, atol=0, rtol=0)


def test_the_mapping_view_addresses_the_weighted_terms() -> None:
    """Back-compatibility: every caller written against a plain dict still works."""
    criterion, out, y = _aux_inputs()
    _, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    assert components["aux_a"] is components.weighted["aux_a"]
    assert sorted(components) == ["aux_a", "aux_b", "aux_c", "aux_d"]
    assert len(components) == 4


def test_scalars_emit_both_series_under_distinct_tags() -> None:
    criterion, out, y = _aux_inputs()
    _, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    tags = components.scalars()
    for branch in "abcd":
        assert f"loss/branch_{branch}_raw" in tags
        assert f"loss/branch_{branch}_weighted" in tags


def test_an_unknown_aux_head_is_discovered_and_defaults_to_weight_one() -> None:
    """`SpectralSeedNet` emits `aux_spatial`; the loss must not need a per-arch branch."""
    criterion, _, y = _aux_inputs()
    torch.manual_seed(1)
    out = {"main": torch.randn(6, 5), "aux_spatial": torch.randn(6, 5)}
    total, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    assert set(components.raw) == {"aux_spatial"}
    assert total.item() == pytest.approx(components.raw["aux_spatial"].item(), abs=1e-6)


# ══════════════════════════════════════════════════════════════════════
#  IC-6 — did the clip bind?
# ══════════════════════════════════════════════════════════════════════


class _Grouped(nn.Module):
    """A module whose parameter names hit all three clip groups."""

    def __init__(self) -> None:
        super().__init__()
        self.arcface_head = nn.Linear(4, 4)
        self.cross_interaction = nn.Linear(4, 4)
        self.branch_c = nn.Linear(4, 4)


def _with_gradients(scale: float) -> _Grouped:
    model = _Grouped()
    for p in model.parameters():
        p.grad = torch.full_like(p, scale)
    return model


def test_the_report_indexes_as_the_preclip_norms_it_replaced() -> None:
    report = clip_grad_norm_by_group(_with_gradients(1.0), max_norm=1.0)
    assert isinstance(report, ClipReport)
    assert set(report) == {"head", "fusion", "backbone"}
    assert all(float(v) > 0 for v in report.values())


def test_a_bound_clip_is_recorded_as_clipped_and_the_postclip_norm_is_the_threshold() -> None:
    report = clip_grad_norm_by_group(_with_gradients(10.0), max_norm=1.0)
    for group in ("head", "fusion", "backbone"):
        assert float(report.preclip[group]) > 1.0
        assert float(report.clipped[group]) == 1.0
        assert float(report.postclip[group]) == pytest.approx(1.0, abs=1e-5)


def test_an_unbound_clip_leaves_the_norm_alone_and_records_zero() -> None:
    report = clip_grad_norm_by_group(_with_gradients(1e-4), max_norm=5.0)
    for group in ("head", "fusion", "backbone"):
        assert float(report.clipped[group]) == 0.0
        assert float(report.postclip[group]) == pytest.approx(
            float(report.preclip[group]), abs=1e-9
        )


def test_clip_fraction_is_the_number_ic6_asks_to_watch() -> None:
    """≈1.0 at the audited threshold, 0.0 once the threshold clips outliers only."""
    audited = clip_grad_norm_by_group(_with_gradients(10.0), max_norm=1.0)
    retuned = clip_grad_norm_by_group(_with_gradients(1e-4), max_norm=5.0)
    assert float(audited.scalars()["grad_norm/clip_fraction"]) == pytest.approx(1.0)
    assert float(retuned.scalars()["grad_norm/clip_fraction"]) == pytest.approx(0.0)


def test_every_series_is_emitted_under_its_own_tag() -> None:
    tags = clip_grad_norm_by_group(_with_gradients(2.0), max_norm=1.0).scalars()
    for group in ("head", "fusion", "backbone"):
        assert f"grad_norm/preclip_{group}" in tags
        assert f"grad_norm/postclip_{group}" in tags
        assert f"grad_norm/clipped_{group}" in tags
    assert "grad_norm/clip_fraction" in tags


def test_the_report_stays_on_device_tensors() -> None:
    """It runs on the inner loop; a host sync here would be one per step."""
    report = clip_grad_norm_by_group(_with_gradients(1.0), max_norm=1.0)
    assert all(isinstance(v, torch.Tensor) for v in report.preclip.values())
    assert all(isinstance(v, torch.Tensor) for v in report.postclip.values())
    assert all(isinstance(v, torch.Tensor) for v in report.clipped.values())
