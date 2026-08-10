"""Per-branch diagnostics — must be wrappers over existing values, not new statistics.

These tests pin that property where it is checkable:

* :func:`branch_grad_norms` reproduces a hand-computed L2 norm over exactly the
  parameters whose names carry each prefix, and omits prefixes with no gradient
  (so a frozen head reads as absent, not as 0.0).
* :func:`hardest_classes_report` is a stable sort of the ``class_f1`` dict
  ``evaluate_per_class`` already returns.
* ``_compute_aux_loss(..., return_components=True)`` returns a total that is
  **bit-identical** to the default path's, and components that sum to it — the
  property that lets the golden Stage-1 loss gate keep passing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectralquadnet.engine.diagnostics import (
    BRANCH_PREFIXES,
    branch_grad_norms,
    epoch_tag,
    hardest_classes_report,
)
from spectralquadnet.losses.auxiliary import _compute_aux_loss


class _Toy(nn.Module):
    """Names chosen to hit three tracked prefixes plus one untracked module."""

    def __init__(self) -> None:
        super().__init__()
        self.branch_a = nn.Linear(4, 4, bias=False)
        self.branch_b = nn.Linear(4, 4, bias=False)
        self.arcface_head = nn.Linear(4, 4, bias=False)
        self.embed_net = nn.Linear(4, 4, bias=False)  # not in BRANCH_PREFIXES


# ══════════════════════════════════════════════════════════════════════
#  Per-branch gradient norms
# ══════════════════════════════════════════════════════════════════════


def test_branch_grad_norms_match_a_hand_computed_l2() -> None:
    model = _Toy()
    model.branch_a.weight.grad = torch.ones(4, 4)  # ‖·‖ = sqrt(16) = 4
    model.branch_b.weight.grad = torch.full((4, 4), 0.5)  # ‖·‖ = sqrt(16*0.25) = 2
    model.embed_net.weight.grad = torch.ones(4, 4)  # untracked prefix

    norms = branch_grad_norms(model)
    assert norms["branch_a"] == 4.0
    assert norms["branch_b"] == 2.0
    assert "embed_net" not in norms, "only BRANCH_PREFIXES are grouped"


def test_branch_grad_norms_omit_branches_without_gradients() -> None:
    """A frozen head has no `.grad`; it must be absent, not reported as 0.0.

    No stage freezes a head any more — HD-1 (T2-10) left one, and it trains
    from Stage 1's first epoch — but the distinction between "not training" and
    "training but flat" is the one this reporting exists to make, and a caller
    is free to freeze anything.
    """
    model = _Toy()
    model.branch_a.weight.grad = torch.ones(4, 4)
    norms = branch_grad_norms(model)
    assert set(norms) == {"branch_a"}
    assert branch_grad_norms(_Toy()) == {}, "no gradients anywhere → empty, not zeros"


def test_branch_prefixes_cover_the_plan_s_list() -> None:
    """The tracked prefix groups; the constant must not drift from them."""
    assert BRANCH_PREFIXES == (
        "branch_a.",
        "branch_b.",
        "branch_c.",
        "branch_d.",
        "cross_interaction.",
        "arcface_head.",
    )


def test_branch_grad_norms_sum_in_quadrature_to_the_group_total() -> None:
    """Two parameters under one prefix combine as a single L2 norm."""
    model = nn.Module()
    model.branch_a = nn.Sequential(nn.Linear(2, 2, bias=True))  # type: ignore[assignment]
    model.branch_a[0].weight.grad = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
    model.branch_a[0].bias.grad = torch.tensor([4.0, 0.0])
    assert branch_grad_norms(model)["branch_a"] == 5.0  # sqrt(3² + 4²)


# ══════════════════════════════════════════════════════════════════════
#  Epoch stamping
# ══════════════════════════════════════════════════════════════════════


def test_epoch_tag_names_the_epoch_and_the_budget() -> None:
    """The stamp every diagnostic line carries, so a log line dates itself.

    Diagnostics are emitted on a checkpoint improvement and on a stride, so
    consecutive lines are not consecutive epochs — without this, "macro F1 went
    down" cannot be placed against the schedule that moved.
    """
    assert epoch_tag(181, 600) == "[ep 181/600] "
    assert epoch_tag(181) == "[ep 181] ", "no budget known → epoch alone"


def test_epoch_tag_is_empty_outside_an_epoch_loop() -> None:
    """`train.py`'s Stage-1 recompute has no epoch; `[ep 0/600]` would be worse than none."""
    assert epoch_tag(0, 600) == ""
    assert epoch_tag(0) == ""


def test_class_difficulty_stamps_its_summary_with_the_epoch(monkeypatch) -> None:
    """The line the stages actually emit carries the tag ahead of the label.

    Pins the rendered string, not just the helper: the ``S1 class difficulty —
    macro F1=…`` line is the one a run is read back through, and it used to
    reach ``training.log`` with no way to tell epoch 181 from epoch 12.
    """
    from types import SimpleNamespace

    from spectralquadnet.engine import diagnostics

    monkeypatch.setattr(diagnostics, "evaluate_per_class", lambda *a, **k: {0: 0.9, 1: 0.2, 2: 0.4})
    cfg = SimpleNamespace(
        data=SimpleNamespace(num_classes=3),
        stage2=SimpleNamespace(cdws_max_weight=3.0, cdws_eps=0.05),
    )
    messages: list[str] = []
    tracker = SimpleNamespace(
        log_message=lambda text, level="info": messages.append(text),
        log_scalars=lambda tags, step: None,
        log_table=lambda tag, rows, step: None,
    )

    diagnostics.compute_class_difficulty(
        cfg, None, None, torch.device("cpu"), "S1", tracker, step=181, detail=False, total_steps=600
    )
    assert messages[0].startswith("[ep 181/600] S1 class difficulty — macro F1=")
    assert len(messages) == 1, "no ablation, no second line"


def test_branch_influence_is_its_own_line(monkeypatch) -> None:
    """The throttled ablation does not change the width of the line above it.

    The influence percentages used to be appended to the class-difficulty
    summary, which made the same diagnostic two different shapes on alternating
    epochs and pushed the hard-class count off the right edge of a narrow
    terminal. They are a second line now, carrying the same epoch stamp.
    """
    from types import SimpleNamespace

    from spectralquadnet.engine import diagnostics

    monkeypatch.setattr(diagnostics, "evaluate_per_class", lambda *a, **k: {0: 0.9, 1: 0.2})
    monkeypatch.setattr(
        diagnostics,
        "compute_branch_influence",
        lambda *a, **k: {"A": 31.2, "B": 24.9, "C": 22.0, "D": 21.9},
    )
    cfg = SimpleNamespace(
        data=SimpleNamespace(num_classes=2),
        stage2=SimpleNamespace(cdws_max_weight=3.0, cdws_eps=0.05),
    )
    messages: list[str] = []
    tracker = SimpleNamespace(
        log_message=lambda text, level="info": messages.append(text),
        log_scalars=lambda tags, step: None,
        log_table=lambda tag, rows, step: None,
    )

    diagnostics.compute_class_difficulty(
        cfg, None, None, torch.device("cpu"), "S1", tracker, step=181, detail=True, total_steps=600
    )
    assert len(messages) == 2
    assert "influence" not in messages[0]
    assert messages[1].startswith("[ep 181/600] S1 branch influence % (leave-one-out KL) —")
    assert "A:  31.2" in messages[1] and "D:  21.9" in messages[1]


# ══════════════════════════════════════════════════════════════════════
#  Hardest-classes report
# ══════════════════════════════════════════════════════════════════════


def test_hardest_classes_report_returns_the_bottom_k_in_order() -> None:
    class_f1 = {0: 0.9, 1: 0.1, 2: 0.5, 3: 0.3}
    rows = hardest_classes_report(class_f1, k=3)
    assert [r["class"] for r in rows] == [1, 3, 2]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["f1"] == 0.1


def test_hardest_classes_report_breaks_ties_on_class_id() -> None:
    """Stable across runs, so the table can be diffed between checkpoints."""
    rows = hardest_classes_report({5: 0.2, 2: 0.2, 9: 0.2}, k=3)
    assert [r["class"] for r in rows] == [2, 5, 9]


def test_hardest_classes_report_handles_k_larger_than_the_dict() -> None:
    assert len(hardest_classes_report({0: 0.4, 1: 0.6}, k=10)) == 2


# ══════════════════════════════════════════════════════════════════════
#  Per-branch auxiliary loss components
# ══════════════════════════════════════════════════════════════════════


def _aux_inputs() -> tuple[nn.Module, dict[str, torch.Tensor], torch.Tensor]:
    torch.manual_seed(0)
    out = {k: torch.randn(6, 5) for k in ("main", "aux_a", "aux_b", "aux_c", "aux_d")}
    return nn.CrossEntropyLoss(), out, torch.randint(0, 5, (6,))


def test_aux_components_are_bit_identical_to_the_summed_total() -> None:
    """The ``return_components`` flag must not perturb the number the training loop already used."""
    criterion, out, y = _aux_inputs()
    total_only = _compute_aux_loss(criterion, out, y, y, 1.0, use_mixup=False)
    total, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    assert total.item() == total_only.item()
    assert set(components) == {"aux_a", "aux_b", "aux_c", "aux_d"}
    assert torch.allclose(sum(components.values()), total, atol=0, rtol=0)


def test_aux_components_carry_the_2x_spectral_branch_weighting() -> None:
    """Branches A/B are weighted 2×, C/D 1× — the bias that keeps A/B alive."""
    criterion, out, y = _aux_inputs()
    out["aux_a"] = out["aux_c"].clone()  # identical logits, different weights
    _, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    assert torch.allclose(components["aux_a"], 2.0 * components["aux_c"])


def test_aux_components_skip_branches_absent_from_the_output() -> None:
    """Inference returns a bare tensor and Stage 3 may emit a partial dict."""
    criterion, out, y = _aux_inputs()
    del out["aux_d"]
    total, components = _compute_aux_loss(
        criterion, out, y, y, 1.0, use_mixup=False, return_components=True
    )
    assert set(components) == {"aux_a", "aux_b", "aux_c"}
    assert torch.allclose(sum(components.values()), total, atol=0, rtol=0)
