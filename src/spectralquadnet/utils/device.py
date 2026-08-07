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


def no_grad_is_safe_for_dropout(device: torch.device) -> bool:
    """Whether a ``train()``-mode forward under ``no_grad`` can use dropout.

    Metal routes attention through a fused inference kernel whenever grad is
    disabled, and that kernel raises ``NotImplementedError:
    scaled_dot_product_attention for MPS does not support dropout``. Grad-mode
    forwards take the math path, which handles dropout — so the only caller that
    runs the model in ``train()`` mode under ``no_grad``
    (:func:`~spectralquadnet.engine.checkpoint.update_bn_stats`) keeps grad
    enabled on Metal. That changes autograd bookkeeping only: forward values,
    and therefore the BatchNorm statistics being estimated, are identical.
    """
    return device.type != "mps"
