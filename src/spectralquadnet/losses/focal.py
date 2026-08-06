"""Focal loss with optional label smoothing.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:class:`FocalLoss`                     494-515
=====================================  ==============

Used by Stage 1 Phase 3 (``gamma=cfg.stage1.focal_gamma`` with the epoch-decayed
label smoothing), by Stage 2 (``gamma=cfg.stage2.focal_gamma``, no smoothing) and
by Stage 3 (``gamma=1.0``, hardcoded in the baseline at line 2529 and preserved).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss with optional label smoothing.
    Soft targets from LS, then focal modulation (1-pt)^γ preserves
    regularisation while sharpening focus on hard examples.
    """

    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        if self.ls > 0.0:
            with torch.no_grad():
                soft = torch.full_like(logits, self.ls / (C - 1))
                soft.scatter_(1, targets.view(-1, 1), 1.0 - self.ls)
            ce = -(soft * logp).sum(1)
        else:
            ce = F.nll_loss(logp, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()
