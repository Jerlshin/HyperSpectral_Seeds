"""Validation-set evaluation: macro-F1, accuracy and the per-class breakdown.

:func:`_run_eval` keeps two details that look incidental and are not:

* ``autocast(..., enabled=False)`` forces fp32 evaluation even mid-AMP-training,
  so a stage's reported F1 never depends on the autocast state it was called
  from.
* Non-finite logits are ``nan_to_num``'d rather than raising — an unstable
  ArcFace epoch degrades the metric instead of killing a multi-hour run.

**Macro-F1 is the primary metric** for every checkpointing decision in all three
stages; accuracy is reported alongside but never gates a save.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast
from torch.utils.data import DataLoader


@torch.no_grad()
def _run_eval(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Run inference over the whole loader and collect predictions/targets.

    Returns:
        ``(preds, targets)`` as flat 1-D numpy arrays, concatenated across
        every batch in ``loader``.
    """
    model.eval()
    preds, targets = [], []
    with autocast(device_type=device.type, enabled=False):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[float, float]:
    """Macro-F1 and accuracy over ``loader``. Macro-F1 is the primary metric —
    it drives every checkpointing decision across all three stages; accuracy
    is reported alongside but never gates a save.
    """
    p, t = _run_eval(model, loader, device)
    return f1_score(t, p, average="macro", zero_division=0), accuracy_score(t, p)


def evaluate_per_class(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, num_classes: int
) -> dict[int, float]:
    """Per-class F1 over ``loader``, keyed by class index 0..num_classes-1."""
    p, t = _run_eval(model, loader, device)
    f1_arr = f1_score(t, p, average=None, zero_division=0, labels=list(range(num_classes)))
    return {i: float(v) for i, v in enumerate(f1_arr)}
