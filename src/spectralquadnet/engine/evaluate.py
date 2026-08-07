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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.amp import autocast
from torch.utils.data import DataLoader

from spectralquadnet.engine.batch import side_inputs, unpack_batch


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
        for batch in loader:
            x, y, mask, morph = unpack_batch(batch, device)
            logits = model(x, **side_inputs(mask, morph))
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


def evaluate_pr_and_confusion(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, num_classes: int
) -> tuple[dict[int, float], dict[int, float], torch.Tensor]:
    """Per-class **precision**, **recall** and the confusion matrix, in one pass.

    HD-3 / T2-8 needs all three: the signed rule
    ``M(c) = clip(m + m_delta (R_c - P_c), …)`` separates the two failure modes
    ``F1`` conflates, and the row-normalised confusion matrix aims the pairwise
    term at the classes ``c`` is actually mistaken for rather than at all 89
    others uniformly (§2.4.1, §3.5 HD-3).

    Returns:
        ``(precision, recall, confusion)`` — the first two keyed by class
        index, the third a raw ``(num_classes, num_classes)`` count matrix with
        rows indexed by the **true** class.
    """
    p, t = _run_eval(model, loader, device)
    labels = list(range(num_classes))
    prec = precision_score(t, p, average=None, zero_division=0, labels=labels)
    rec = recall_score(t, p, average=None, zero_division=0, labels=labels)
    cm = confusion_matrix(t, p, labels=labels)
    return (
        {i: float(v) for i, v in enumerate(prec)},
        {i: float(v) for i, v in enumerate(rec)},
        torch.from_numpy(np.asarray(cm)).float(),
    )


@torch.no_grad()
def collect_embeddings(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every sample's L2-normalised embedding and label, as two CPU tensors.

    The input to HD-2(iii)'s spherical *k*-means sub-centre seeding. Eval mode
    and fp32, like :func:`_run_eval`, so the embeddings the sub-centres are
    fitted to are the ones inference will produce.

    Returns:
        ``(embeddings, labels)`` of shapes ``(N, d)`` and ``(N,)``.
    """
    model.eval()
    embs, targets = [], []
    with autocast(device_type=device.type, enabled=False):
        for batch in loader:
            x, y, mask, morph = unpack_batch(batch, device)
            _, emb = model(x, return_embed=True, **side_inputs(mask, morph))
            embs.append(emb.detach().float().cpu())
            targets.append(y.cpu())
    return torch.cat(embs), torch.cat(targets)
