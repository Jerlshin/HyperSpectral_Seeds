"""Device resolution and accelerator-specific AMP settings.

``cfg.device`` is a *strategy string* (``"auto"``/``"cuda"``/``"cpu"``/``"mps"``)
rather than a live ``torch.device``, since YAML cannot carry the latter;
:func:`resolve_device` turns it into a concrete device.

``"auto"`` prefers **Metal (MPS) → CUDA → CPU**, so a default run on Apple
Silicon uses the GPU rather than falling through to CPU. An explicit
``device=cuda`` / ``device=cpu`` / ``device=mps`` is never overridden.

The autocast **dtype** is deliberately left at torch's per-device default
(fp16 on both Metal and CUDA) rather than promoted to bf16 on capable CUDA
hardware, to keep training numerics consistent across accelerators.

This module used to carry a ``no_grad_is_safe_for_dropout`` predicate, because
Metal's fused attention kernel rejects dropout under ``no_grad`` and
``update_bn_stats`` was the one caller that ran the model in ``train()`` mode
there. Tier 1 (T1-5) switches every stochastic module off for that pass
outright, which is the correct behaviour for BatchNorm re-estimation anyway, so
nothing is left that depends on the accelerator — the predicate is gone rather
than left inert. ``tests/unit/test_device.py`` still pins the upstream Metal
limitation that motivated it.
"""

from __future__ import annotations

import torch


def resolve_device(strategy: str | torch.device = "auto") -> torch.device:
    """Turn a config ``device`` string into a concrete :class:`torch.device`.

    ``"auto"`` picks the fastest locally available accelerator, preferring
    Apple's Metal backend, then CUDA, then CPU. Any other value (``"cuda"``,
    ``"cuda:1"``, ``"cpu"``, ``"mps"``) is passed straight to
    :class:`torch.device`, so an explicit choice is never silently overridden.
    """
    if isinstance(strategy, torch.device):
        return strategy
    if strategy == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(strategy)
