"""Scheduler bit-exactness — REFACTOR_PLAN.md §3.2.3.

The cheapest and strongest of the three §3.2 techniques: these schedules are pure
functions of primitives, so the relocated implementation can be called *side by
side* with the pre-refactor one over the whole epoch range each is ever asked
about, and compared with ``==`` rather than a tolerance.

This is what backs the plan's promise that "dynamic margin schedules … remain
100% identical" (§3.2.3): every LR multiplier and every ArcFace margin the three
stages will ever see is checked here, not sampled.

Coverage
────────
==========================  ====================  ====================================
Schedule                    Baseline location     Epoch range checked
==========================  ====================  ====================================
``arcface_margin``          lines 1797-1800       0-39 (warmup is 20) × 4 margin pairs
``sgdr_scheduler._l``       lines 1779-1794       0-399 × 4 (warmup, T_0, T_mult) sets
``phase_aware_lr``          lines 2218-2246       0-599 (``s1_epochs``) × 3 splits
==========================  ====================  ====================================

``phase_aware_lr`` is a closure inside ``run_stage1`` in the baseline, so it is
recovered by compiling that one nested ``def`` with the free variables it reads
(``CONFIG``, ``math``, ``p1_end``, ``p2_end``) bound in its namespace — the
genuine pre-refactor code, not a transcription of it.
"""

from __future__ import annotations

import ast
import math
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch

from spectralquadnet.optim.schedulers import arcface_margin, phase_aware_lr, sgdr_scheduler

pytestmark = pytest.mark.regression


# ══════════════════════════════════════════════════════════════════════
#  Recovering a nested baseline closure
# ══════════════════════════════════════════════════════════════════════


def nested_function(src: str, outer: str, inner: str, namespace: dict[str, Any]) -> Callable:
    """Compile the ``inner`` ``def`` nested inside ``outer`` against ``namespace``.

    ``namespace`` must supply every free variable the closure reads; anything
    missing surfaces as a ``NameError`` when the returned function is called,
    which is a loud enough failure for a test helper.
    """
    tree = ast.parse(src)
    outer_node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == outer), None
    )
    assert outer_node is not None, f"{outer}() not found in the baseline source"

    inner_node = next(
        (n for n in ast.walk(outer_node) if isinstance(n, ast.FunctionDef) and n.name == inner),
        None,
    )
    assert inner_node is not None, f"{inner}() not found inside {outer}()"

    module = ast.Module(body=[inner_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f"<baseline:{outer}.{inner}>", "exec"), namespace)  # noqa: S102
    return namespace[inner]


# ══════════════════════════════════════════════════════════════════════
#  ArcFace margin warmup  (Stage 2)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("m0", "m_target", "warmup_ep"),
    [
        (0.18, 0.35, 20),  # the shipped config
        (0.0, 0.5, 10),
        (0.3, 0.3, 5),  # degenerate: m0 == m_target
        (0.18, 0.35, 0),  # degenerate: division guarded by max(warmup_ep, 1)
    ],
)
def test_arcface_margin_matches_baseline(baseline, m0, m_target, warmup_ep):
    """Every margin Stage 2 can request is bit-identical to the baseline's."""
    for ep in range(0, 40):
        expected = baseline.arcface_margin(ep, m0, m_target, warmup_ep)
        actual = arcface_margin(ep, m0, m_target, warmup_ep)
        assert actual == expected, f"ep={ep}: {actual!r} != {expected!r}"


def test_arcface_margin_shipped_config_endpoints(cfg):
    """Sanity anchors on the composed config: starts at m0, pins at m after warmup."""
    m0, m, warm = cfg.stage2.arcface_m0, cfg.stage2.arcface_m, cfg.stage2.margin_warmup_ep
    assert arcface_margin(0, m0, m, warm) == m0
    assert arcface_margin(warm, m0, m, warm) == m
    assert arcface_margin(warm + 50, m0, m, warm) == m


# ══════════════════════════════════════════════════════════════════════
#  SGDR with warm restarts  (Stage 2)
# ══════════════════════════════════════════════════════════════════════


def _lambda_of(scheduler: torch.optim.lr_scheduler.LambdaLR) -> Callable[[int], float]:
    return scheduler.lr_lambdas[0]


def _dummy_optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)


@pytest.mark.parametrize(
    ("warmup_ep", "T_0", "T_mult", "eta_min_frac"),
    [
        (3, 25, 2, 1e-6 / 2.5e-4),  # the shipped Stage 2 config
        (5, 10, 2, 1e-3),  # the function's own defaults
        (0, 7, 1, 0.5),  # no warmup, non-growing cycles
        (10, 1, 3, 0.0),
    ],
)
def test_sgdr_lambda_matches_baseline(baseline, warmup_ep, T_0, T_mult, eta_min_frac):
    """The LR multiplier agrees for every epoch of a run 2.5× longer than Stage 2."""
    expected = _lambda_of(
        baseline.sgdr_scheduler(_dummy_optimizer(), warmup_ep, T_0, T_mult, eta_min_frac)
    )
    actual = _lambda_of(sgdr_scheduler(_dummy_optimizer(), warmup_ep, T_0, T_mult, eta_min_frac))
    for ep in range(0, 400):
        assert actual(ep) == expected(ep), f"ep={ep}: {actual(ep)!r} != {expected(ep)!r}"


def test_sgdr_stepped_lr_trajectory_matches_baseline(baseline, cfg):
    """End-to-end: stepping both schedulers yields identical ``param_groups`` LRs.

    Covers the wiring as well as the lambda — a scheduler attached with a
    different ``last_epoch`` or base LR would pass the lambda test and still
    train differently.
    """
    kw = dict(
        warmup_ep=cfg.stage2.warmup_ep,
        T_0=cfg.stage2.sgdr_T0,
        T_mult=cfg.stage2.sgdr_Tmult,
        eta_min_frac=cfg.stage2.min_lr / cfg.stage2.head_lr,
    )
    opt_old, opt_new = _dummy_optimizer(), _dummy_optimizer()
    sch_old = baseline.sgdr_scheduler(opt_old, **kw)
    sch_new = sgdr_scheduler(opt_new, **kw)

    # Both schedulers step without an intervening `optimizer.step()`; torch warns
    # about that ordering, which is irrelevant when no parameters are updated.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for ep in range(cfg.stage2.epochs):
            assert opt_new.param_groups[0]["lr"] == opt_old.param_groups[0]["lr"], f"ep={ep}"
            sch_old.step()
            sch_new.step()


# ══════════════════════════════════════════════════════════════════════
#  Stage 1's three-phase LR schedule
# ══════════════════════════════════════════════════════════════════════


def _baseline_phase_aware_lr(src: str, config: dict, p1_end: int, p2_end: int):
    return nested_function(
        src,
        "run_stage1",
        "phase_aware_lr",
        {"CONFIG": config, "math": math, "p1_end": p1_end, "p2_end": p2_end},
    )


def test_phase_aware_lr_matches_baseline(baseline, baseline_src, cfg):
    """All 600 Stage-1 epochs, across all three phases, are bit-identical.

    The boundaries are computed the way ``run_stage1`` computes them (lines
    2211-2213) so the three branches are exercised in their real proportions:
    warmup and Phase-1 decay, Phase-2 cosine, then the Phase-3 30-epoch restarts.
    """
    ep_total = cfg.stage1.epochs
    p1_end = int(ep_total * cfg.stage1.phase1_frac)
    p2_end = int(ep_total * (cfg.stage1.phase1_frac + cfg.stage1.phase2_frac))

    expected = _baseline_phase_aware_lr(baseline_src, baseline.CONFIG, p1_end, p2_end)
    actual = phase_aware_lr(cfg, p1_end, p2_end)

    for ep_idx in range(ep_total):
        assert actual(ep_idx) == expected(ep_idx), f"ep_idx={ep_idx}"


def test_s3_margin_schedule_matches_baseline(baseline, baseline_src, cfg):
    """Stage 3's ArcFace margin schedule is identical for all 120 epochs.

    ``_s3_margin`` stays a closure inside ``run_stage3_swa`` on both sides — only
    ``phase_aware_lr`` was lifted out — so both are recovered from source. It is
    included here because it is the second of the two "dynamic margin schedules"
    §3.2.3 promises are unchanged, and because a ``cfg.stage3.epochs`` typo would
    otherwise be invisible until a Stage 3 run diverged.
    """
    import inspect

    from spectralquadnet.engine.stages import stage3_sam_swa

    expected = nested_function(
        baseline_src, "run_stage3_swa", "_s3_margin", {"CONFIG": baseline.CONFIG, "math": math}
    )
    actual = nested_function(
        inspect.getsource(stage3_sam_swa),
        "run_stage3_swa",
        "_s3_margin",
        {"cfg": cfg, "math": math},
    )

    for ep in range(cfg.stage3.epochs + 1):
        assert actual(ep) == expected(ep), f"ep={ep}: {actual(ep)!r} != {expected(ep)!r}"


def test_phase_aware_lr_covers_all_three_phases(cfg):
    """Guards the test above: the epoch grid must actually reach every branch.

    Without this, a boundary bug that collapsed Phase 3 into Phase 2 would leave
    the comparison passing over a range that never enters the restart cycle.
    """
    ep_total = cfg.stage1.epochs
    p1_end = int(ep_total * cfg.stage1.phase1_frac)
    p2_end = int(ep_total * (cfg.stage1.phase1_frac + cfg.stage1.phase2_frac))
    assert 5 < p1_end < p2_end < ep_total

    lr = phase_aware_lr(cfg, p1_end, p2_end)
    assert lr(0) == pytest.approx(1 / 5.0)  # first warmup epoch
    assert lr(p1_end - 1) < 1.0  # Phase 1 has decayed
    assert lr(p2_end - 1) == pytest.approx(0.2, abs=1e-3)  # Phase 2 floor

    # Phase 3 restarts: the multiplier returns to its cycle peak every 30 epochs.
    peak = lr(p2_end)
    assert lr(p2_end + 30) == peak
    assert lr(p2_end + 29) < peak

    # …and the cycle stays inside [min_lr, mid_lr] as a fraction of max_lr.
    min_ratio = cfg.stage1.min_lr / cfg.stage1.max_lr
    mid_ratio = cfg.stage1.mid_lr / cfg.stage1.max_lr
    p3_values = [lr(e) for e in range(p2_end, ep_total)]
    assert min(p3_values) >= min_ratio
    assert max(p3_values) <= mid_ratio
