"""Scoring, artifacts, figures and tables — everything downstream of a prediction.

Split from ``engine/`` because none of it needs a model, a device or the patch
cube: it reads prediction arrays and ``results/run.json``. That is what makes
every table and figure in the paper regenerable, offline, by someone who did not
run the experiments — which the audited project's seven Stage-1-only W&B panels
were not.

=================  =========================================================
Module             Responsibility
=================  =========================================================
``metrics``        Score an array; bootstrap and paired-bootstrap intervals
``artifacts``      The ``results/`` tree layout and ``run.json``, the contract
``figures``        matplotlib renderings (optional dependency)
``tables``         Markdown/CSV tables aggregated over many runs
=================  =========================================================
"""

from __future__ import annotations

from spectralquadnet.reporting.artifacts import RunArtifacts, load_manifest, publish
from spectralquadnet.reporting.metrics import (
    ClassificationResult,
    Interval,
    bootstrap_macro_f1,
    macro_f1,
    mean_and_range,
    paired_bootstrap_delta,
    score,
)

__all__ = [
    "ClassificationResult",
    "Interval",
    "RunArtifacts",
    "bootstrap_macro_f1",
    "load_manifest",
    "macro_f1",
    "mean_and_range",
    "paired_bootstrap_delta",
    "publish",
    "score",
]
