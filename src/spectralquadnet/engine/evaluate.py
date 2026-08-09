"""Validation-set evaluation: macro-F1, accuracy and the per-class breakdown.

:func:`_run_eval` keeps two details that look incidental and are not:

* ``autocast(..., enabled=False)`` forces fp32 evaluation even mid-AMP-training,
  so a stage's reported F1 never depends on the autocast state it was called
  from.
* Non-finite logits are ``nan_to_num``'d rather than raising — an unstable
  ArcFace epoch degrades the metric instead of killing a multi-hour run.

**Macro-F1 is the primary metric** for every checkpointing decision in all three
stages; accuracy is reported alongside but never gates a save.

Two host round-trips per batch, removed
───────────────────────────────────────
Stages 1 and 2 run this **twice per epoch** — once for the live model, once for
the EMA shadow — over the whole validation split, and it is a large share of
epoch wall time. It used to synchronise twice for every batch of that:

* ``if not torch.isfinite(logits).all()`` is a Python branch on a device
  tensor. The guard is now unconditional instead: ``nan_to_num`` on finite
  logits is the identity function, so applying it always produces exactly the
  values the branch produced, without asking the host what the answer was.
* ``preds.append(logits.argmax(1).cpu())`` copied every batch back one at a
  time. The per-batch results stay on the device and are concatenated in one
  transfer at the end.

Under DDP the split is sharded across ranks and re-joined by
:func:`~spectralquadnet.utils.distributed.gather_concat` before any metric is
computed, because a macro-F1 over half the classes is not half of the macro-F1.

What each pass is allowed to hold
─────────────────────────────────
Keeping the per-batch results on the device is what removed the transfers
above, and it means an evaluation's peak footprint is now *its own* to manage:
the accumulated predictions for the whole split, plus one batch's activations.
The loops below therefore drop each batch's tensors at the end of the iteration
that produced them, rather than letting the loop variable carry the last batch
alive through the concatenate/gather/transfer tail. Stages 1-3 run this twice
per epoch, immediately after a backward pass has already driven the caching
allocator to its high-water mark, so the batch that is not released here is the
one the next allocation has to grow the pool for — see
:func:`~spectralquadnet.utils.device.release_memory`, which the stage loops call
either side of these calls.
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
from spectralquadnet.utils.distributed import DistContext, gather_concat


@torch.inference_mode()
def _run_eval(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    dist: DistContext | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Run inference over the whole loader and collect predictions/targets.

    ``inference_mode`` rather than ``no_grad``: it additionally skips
    autograd's version-counter bookkeeping on every tensor produced, which is
    pure overhead for a pass whose results never require a gradient.

    Returns:
        ``(preds, targets)`` as flat 1-D numpy arrays, concatenated across
        every batch in ``loader`` — and, under DDP, across every rank.
    """
    model.eval()
    preds, targets = [], []
    with autocast(device_type=device.type, enabled=False):
        for batch in loader:
            x, y, mask, morph = unpack_batch(batch, device)
            logits = model(x, **side_inputs(mask, morph))
            # Unconditional, and therefore sync-free: on finite logits this is
            # the identity, so it yields what the `isfinite` branch yielded.
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            preds.append(logits.argmax(1))
            targets.append(y)
            # `logits` is the batch's largest tensor and the loop variable holds
            # the last one until the *next* iteration rebinds it — so without
            # this the whole activation chain behind one batch stays resident
            # across the `cat`, the gather and the transfer below. Two stages
            # call this twice an epoch; releasing the reference at the end of
            # the iteration that made it costs nothing and bounds the pass's
            # high-water mark to one batch.
            del x, mask, morph, logits

    dist = dist or DistContext()
    joined_preds = gather_concat(dist, torch.cat(preds))
    joined_targets = gather_concat(dist, torch.cat(targets))
    # The per-batch results are on the device and no longer needed once
    # concatenated; dropping the lists here rather than at return frees the
    # split's worth of them before the gather allocates its own buffers.
    del preds, targets
    # One device→host transfer for the whole split instead of one per batch.
    out_preds = joined_preds.cpu().numpy()
    out_targets = joined_targets.cpu().numpy()
    del joined_preds, joined_targets
    return out_preds, out_targets


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    dist: DistContext | None = None,
) -> tuple[float, float]:
    """Macro-F1 and accuracy over ``loader``. Macro-F1 is the primary metric —
    it drives every checkpointing decision across all three stages; accuracy
    is reported alongside but never gates a save.
    """
    p, t = _run_eval(model, loader, device, dist)
    return f1_score(t, p, average="macro", zero_division=0), accuracy_score(t, p)


def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    dist: DistContext | None = None,
) -> dict[int, float]:
    """Per-class F1 over ``loader``, keyed by class index 0..num_classes-1."""
    p, t = _run_eval(model, loader, device, dist)
    f1_arr = f1_score(t, p, average=None, zero_division=0, labels=list(range(num_classes)))
    return {i: float(v) for i, v in enumerate(f1_arr)}


def evaluate_pr_and_confusion(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    dist: DistContext | None = None,
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
    p, t = _run_eval(model, loader, device, dist)
    labels = list(range(num_classes))
    prec = precision_score(t, p, average=None, zero_division=0, labels=labels)
    rec = recall_score(t, p, average=None, zero_division=0, labels=labels)
    cm = confusion_matrix(t, p, labels=labels)
    return (
        {i: float(v) for i, v in enumerate(prec)},
        {i: float(v) for i, v in enumerate(rec)},
        torch.from_numpy(np.asarray(cm)).float(),
    )


# `no_grad`, not `inference_mode`, and deliberately so: tensors produced under
# inference mode are *inference tensors*, and the caller — HD-2(iii)'s spherical
# k-means — mutates what it derives from these (`centres[j] = x[worst]`) and
# copies the result into a live `nn.Parameter`. The extra bookkeeping
# `inference_mode` elides is worth having on `_run_eval`, which runs twice per
# epoch; this runs once per stage and does not need it.
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
            # Kept on the device and moved once below, rather than one blocking
            # copy per batch. `_spherical_kmeans` runs on the host, so the whole
            # tensor still lands there — just in a single transfer.
            embs.append(emb.detach().float())
            targets.append(y)
            del x, mask, morph, emb  # see `_run_eval` — one batch resident, not two
    # One device→host transfer for the whole split rather than one per batch;
    # `_spherical_kmeans` runs on the host, so everything lands there either way.
    out_embs = torch.cat(embs).cpu()
    out_targets = torch.cat(targets).cpu()
    del embs, targets
    return out_embs, out_targets
