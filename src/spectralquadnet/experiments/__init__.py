"""The experimental programme of CHANGES §19–§21, as code.

The audited project documented 21 ablation levers and pulled zero of them, and
that — not the architecture — is the finding the whole revision turns on. This
package exists so that state cannot recur: every experiment is a declared value
in :mod:`~spectralquadnet.experiments.registry`, expanded into independently
executable runs by :mod:`~spectralquadnet.experiments.runner`, and aggregated
into tables and figures by :mod:`~spectralquadnet.experiments.aggregate`.

=================  ============================================================
Module             Responsibility
=================  ============================================================
``registry``       The grid: A1–A12 as data, each with a question and a
                   pre-registered decision rule
``runner``         Expanding an ablation into cells and executing them as
                   independent subprocesses
``protocol``       CHANGES §19's 2-fold × 3-seed leave-one-bundle-out sweep,
                   plus the stratified contrast arm
``aggregate``      Runs → per-arm mean ± range, paired bootstrap deltas,
                   tables and figures
``baselines``      LDA / LinearSVC on mean spectra under the identical protocol
``leakage``        Model-free measurement of the acquisition signal, from the
                   residual brightness the model never sees
``analysis``       A9 — is the persistent hard cluster genetic or a
                   segmentation failure?
``cli``            ``python -m spectralquadnet.experiments.cli``
=================  ============================================================

Run order (CHANGES §20): **A12 first** — until run-to-run variance is known, no
delta means anything — then A1, then A9, then A3/A8.
"""

from __future__ import annotations

from spectralquadnet.experiments.registry import ABLATIONS, Ablation, Arm, get, total_runs

__all__ = ["ABLATIONS", "Ablation", "Arm", "get", "total_runs"]
