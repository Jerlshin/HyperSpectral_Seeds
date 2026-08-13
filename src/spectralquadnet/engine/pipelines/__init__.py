"""Curricula, behind one name (IC-11).

``cfg.pipeline`` selects which one runs. Four values, and the last three are
the same driver stopped at different points:

=================  ==============================================================
``pipeline``       What runs
=================  ==============================================================
``single``         CHANGES §17 — one stage, one objective, one schedule.
``stage1_only``    A8 arm 1 — the audited Stage 1 alone.
``stage1_stage2``  A8 arm 2 — Stages 1 and 2.
``three_stage``    A8 arm 3 — the audited curriculum in full.
=================  ==============================================================

A8 is the falsification test for 65% of the audited run's wall clock. Its three
arms have to be *the same code* stopped at different points, or a difference
between them could be a difference between three loops. That is why
``three_stage.py`` is parameterised by a last-stage index rather than copied.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from spectralquadnet.engine.pipelines import single, three_stage
from spectralquadnet.engine.pipelines.context import RunContext, build_run_context

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: ``cfg.pipeline`` → the function that runs it.
PIPELINES: dict[str, Callable[[RunContext], None]] = {
    "single": single.run,
    "stage1_only": three_stage.run,
    "stage1_stage2": three_stage.run,
    "three_stage": three_stage.run,
}

__all__ = ["PIPELINES", "RunContext", "build_run_context", "resolve_pipeline"]


def resolve_pipeline(name: str) -> Callable[[RunContext], None]:
    """The runner for ``cfg.pipeline``.

    Raises:
        ValueError: Unknown pipeline name.
    """
    runner = PIPELINES.get(str(name))
    if runner is None:
        raise ValueError(
            f"Unknown pipeline {name!r}. Expected one of: {', '.join(sorted(PIPELINES))}."
        )
    return runner
