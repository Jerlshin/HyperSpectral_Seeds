"""Device resolution.

The pre-refactor ``CONFIG["device"]`` (baseline line 141) held a live
``torch.device`` object built by
``torch.device("cuda" if torch.cuda.is_available() else "cpu")``. YAML cannot
carry one, so REFACTOR_PLAN.md §4.3 turns the key into a *strategy string* and
puts the resolution here. :func:`resolve_device` reproduces the original
expression exactly for the ``"auto"`` strategy.
"""

from __future__ import annotations

import torch


def resolve_device(strategy: str | torch.device = "auto") -> torch.device:
    """Turn a config ``device`` string into a concrete :class:`torch.device`.

    ``"auto"`` reproduces the baseline's
    ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``. Any other
    value (``"cuda"``, ``"cuda:1"``, ``"cpu"``, ``"mps"``) is passed straight to
    :class:`torch.device`, so an explicit choice is never silently overridden.
    """
    if isinstance(strategy, torch.device):
        return strategy
    if strategy == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(strategy)
