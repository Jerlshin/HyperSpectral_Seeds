"""Focal loss with optional label smoothing.

Used by Stage 1 Phase 3 (``gamma=cfg.stage1.focal_gamma`` with epoch-decayed
label smoothing), by Stage 2 (``gamma=cfg.stage2.focal_gamma``, no smoothing)
and by Stage 3 (``gamma=1.0``, hardcoded).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with optional label smoothing.

    Label smoothing softens the targets first; the focal modulation term
    ``(1 - p_t) ** gamma`` is then applied on top, so the model keeps the
    smoothing's regularisation while still being pushed to focus its
    gradient on hard (low-confidence) examples.
    """

    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean focal loss over the batch.

        Args:
            logits: Unnormalised class scores, shape ``(B, C)``.
            targets: Ground-truth class indices, shape ``(B,)``.

        Returns:
            Scalar mean focal loss over the batch.
        """
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
