"""Device resolution and accelerator-specific AMP settings.

The pre-refactor ``CONFIG["device"]`` (baseline line 141) held a live
``torch.device`` object built by
``torch.device("cuda" if torch.cuda.is_available() else "cpu")``. YAML cannot
carry one, so REFACTOR_PLAN.md §4.3 turns the key into a *strategy string* and
puts the resolution here.

Declared deviation from the baseline — Apple Silicon support
────────────────────────────────────────────────────────────
``"auto"`` now prefers **Metal (MPS) → CUDA → CPU**, where the baseline only
knew ``cuda``-or-``cpu``. On an Apple Silicon host the baseline expression fell
through to ``cpu`` and left the GPU idle; this is a deliberate, requested
capability change, not a mechanical relocation, and it is the one place the
refactor changes which hardware a default run lands on. An explicit
``device=cuda`` / ``device=cpu`` / ``device=mps`` is still never overridden, so
reproducing the original CUDA lineage is a single override away.

Nothing about *numerics* changes with the device beyond the usual
accelerator-kernel differences; the regression gates in ``tests/regression/``
pin their tolerances on CPU and are unaffected.

The autocast **dtype** is deliberately left at torch's per-device default (fp16
on both Metal and CUDA). Selecting bf16 on capable CUDA hardware would be a free
stability win but would also change the numerics of the exact configuration the
three trained checkpoints came from, which §6 puts out of scope.
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
