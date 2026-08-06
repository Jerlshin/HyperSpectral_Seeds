"""Branch-influence ablation and per-class difficulty reporting.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`compute_branch_influence`       1333-1364
:func:`compute_class_difficulty`       2157-2183
=====================================  ==============

:func:`compute_branch_influence` relocates verbatim — it reads no ``CONFIG``.
:func:`compute_class_difficulty` carries the mechanical
``CONFIG["num_classes"]``/``["cdws_max_weight"]``/``["cdws_eps"]`` →
``cfg.data.num_classes`` / ``cfg.stage2.cdws_*`` rewrite and so takes ``cfg``
first.

:func:`compute_branch_influence` measures how much each branch matters by
zeroing it (``branch_mask``) and taking the KL divergence of the ablated
prediction from the full one, then normalising the four numbers to percentages.
Note the cost: ``max_batches × 5`` forward passes, which is why every caller
passes a small ``max_batches`` (3 from :func:`compute_class_difficulty`) and only
calls it on a checkpoint improvement.

The ``print`` here is the pre-refactor observability surface. Phase 4 (§4.2)
routes both return values and this line through an ``ExperimentTracker``; the
computations themselves stay exactly as they are — that is the whole point of
the plan's "diagnostics are additive" principle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spectralquadnet.engine.evaluate import evaluate_per_class
from spectralquadnet.losses.cdws import build_cdws_weights

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


@torch.no_grad()
def compute_branch_influence(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    max_batches: int = 5,
) -> dict[str, float]:
    model.eval()
    influences = torch.zeros(4, device=device)
    total = 0

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        logits_full = model(x)
        p_full = torch.softmax(logits_full, dim=1)

        for b in range(4):
            mask = torch.ones(4, device=device)
            mask[b] = 0.0
            logits_ab = model(x, branch_mask=mask)
            p_ab = torch.softmax(logits_ab, dim=1).clamp(min=1e-10)
            influences[b] += F.kl_div(p_ab.log(), p_full, reduction="batchmean")
        total += 1

    if total == 0:
        return {"A": 0, "B": 0, "C": 0, "D": 0}

    influences /= total
    total_inf = influences.sum().clamp(min=1e-8)
    influences = influences / total_inf * 100.0
    return {k: float(influences[i]) for i, k in enumerate("ABCD")}


def compute_class_difficulty(
    cfg: ExperimentConfig | Any,
    ema_shadow: nn.Module,
    val_ldr: DataLoader[Any],
    device: torch.device,
    label: str = "Stage",
) -> tuple[dict[int, float], dict[int, float]]:
    class_f1 = evaluate_per_class(ema_shadow, val_ldr, device, cfg.data.num_classes)
    cdws_wts = build_cdws_weights(
        class_f1, cfg.data.num_classes, cfg.stage2.cdws_max_weight, cfg.stage2.cdws_eps
    )
    macro = float(np.mean(list(class_f1.values())))
    n_hard = sum(1 for f in class_f1.values() if f < 0.50)

    branch_inf = compute_branch_influence(ema_shadow, val_ldr, device, max_batches=3)

    print(
        f"[INFO] {label} class difficulty — macro F1={macro:.3f}  "
        f"hard classes (<0.50 F1): {n_hard}/{cfg.data.num_classes}  |  "
        f"Branch influence % → "
        f"A:{branch_inf['A']:.1f}  B:{branch_inf['B']:.1f}  "
        f"C:{branch_inf['C']:.1f}  D:{branch_inf['D']:.1f}"
    )
    return class_f1, cdws_wts
