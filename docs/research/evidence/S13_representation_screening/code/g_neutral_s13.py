#!/usr/bin/env python3
"""S13 gate G-neutral: the S13 code at its defaults must not move a single training number.

S11's gate (``evidence/S11_frozen_arms_execution/code/g_neutral.py``) unchanged —
the same miniature job through the code before (a worktree of ``413a11e``, the
commit every S11 cell and the S13 reference ran on) and after S13, compared step
loss by step loss, checkpoint tensor by tensor and held-out prediction by
prediction, in its three regimes (``shipped``, ``clip_binds``, ``x1_like`` = R1 in
miniature) — with two sensitivity controls added for the S13 arm keys:

``y2_mixstyle``  ``model.spatial_mixstyle=true`` under ``x1_like``
``y3_lean``      ``model.spectral_descriptor=snv_morph model.spatial_tail_strides=[2,2,2,1]
                 model.cbam_min_hw=3`` under ``x1_like``

Each must be *detected* as a change; if either compared equal, the gate would be
blind to the arms it is meant to fence off. S11's own controls (clip partition,
aux schedule) are kept. The new tree also writes the training-rows session κ in
every run (``evaluation.session_probe``, default on), so the comparison covers it.

Usage (repository root; ≈ 2 min)::

    git worktree add /tmp/s11_tree 413a11e
    python docs/research/evidence/S13_representation_screening/code/g_neutral_s13.py \\
        --old /tmp/s11_tree --new . \\
        --out docs/research/evidence/S13_representation_screening/g_neutral.json
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
S11_GATE = REPO / "docs/research/evidence/S11_frozen_arms_execution/code/g_neutral.py"

S13_SENSITIVITY: dict[str, tuple[str, list[str]]] = {
    "y2_mixstyle": ("x1_like", ["model.spatial_mixstyle=true"]),
    "y3_lean": (
        "x1_like",
        [
            "model.spectral_descriptor=snv_morph",
            "model.spatial_tail_strides=[2,2,2,1]",
            "model.cbam_min_hw=3",
        ],
    ),
}


def main() -> int:
    spec = importlib.util.spec_from_file_location("g_neutral", S11_GATE)
    assert spec and spec.loader
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SENSITIVITY.update(S13_SENSITIVITY)
    gate.NEW_ONLY.append("evaluation.session_probe=true")
    return int(gate.main())


if __name__ == "__main__":
    sys.exit(main())
