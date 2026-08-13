"""Runtime knobs that have to reach *every* stochastic site (IC-14 / N-2).

``Module.set_dropout`` used to walk ``nn.Dropout`` instances only, and
``nn.MultiheadAttention`` does not own one: its dropout is a plain ``float``
attribute consulted inside the fused attention kernel. Branch D has four such
sites and the fusion had two, so the Stage-2 and Stage-3 dropout schedules —
0.15 → 0.25 → 0.10 — silently did not apply to them. They ran the whole
audited run at the 0.15 they were constructed with (CHANGES §5.1).

Three separate places had independently noticed and worked around it: the
``_STOCHASTIC_MODULES`` tuple in ``engine/checkpoint.py`` (which forces them to
``eval()`` for the BatchNorm re-estimation pass), the ``silence_dropout``
fixture in ``tests/conftest.py``, and a comment in the model. One function
now owns the question of what "dropout" means.
"""

from __future__ import annotations

import torch.nn as nn


def set_dropout(module: nn.Module, p: float) -> int:
    """Set every dropout rate under ``module`` to ``p``.

    Covers both kinds of site:

    * ``nn.Dropout`` and its N-dimensional variants, via ``.p``;
    * ``nn.MultiheadAttention``, via its ``.dropout`` float.

    Args:
        module: Root of the subtree to modify, in place.
        p: The new rate.

    Returns:
        How many sites were changed — asserted by
        ``tests/unit/test_branch_drop.py`` so the count cannot silently fall to
        zero if torch reorganises the attention module again.
    """
    touched = 0
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            touched += 1
        elif isinstance(m, nn.MultiheadAttention):
            # A float attribute, not a submodule: `m.modules()` will never yield
            # a `Dropout` for it however deep the walk goes.
            m.dropout = p
            touched += 1
    return touched


def count_dropout_sites(module: nn.Module) -> dict[str, int]:
    """``{"dropout": n, "attention": m}`` — what :func:`set_dropout` would touch."""
    counts = {"dropout": 0, "attention": 0}
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            counts["dropout"] += 1
        elif isinstance(m, nn.MultiheadAttention):
            counts["attention"] += 1
    return counts
