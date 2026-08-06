"""Supervised-contrastive and prototype-contrastive auxiliary losses.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:class:`SupConLoss`                    518-537
:class:`ProtoNCELoss`                  540-559
=====================================  ==============

Both operate on the L2-normalised embedding ``SpectralQuadNet.forward`` returns
under ``out["emb"]`` when called with ``return_embed=True``. They are active in
Stage 1 Phase 3 (weights ``cfg.stage1.p3_{supcon,proto}_weight``), Stage 2
(``cfg.stage2.{supcon,proto}_weight``, ramped over the first 10 epochs) and
Stage 3 (0.02 / 0.01, hardcoded in the baseline at lines 2557-2558).

Note that passing a ``supcon`` module to ``train_one_epoch`` also disables AMP —
see the ``use_amp`` guard in ``engine/train_epoch.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss. Expects L2-normalised features."""

    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        sim = torch.mm(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        n_pos = pos_mask.float().sum(1)
        if not (n_pos > 0).any():
            return torch.zeros((), device=features.device, requires_grad=True)
        sim_m = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim_m - torch.logsumexp(sim_m, dim=1, keepdim=True)
        loss = -(pos_mask.float() * log_prob.masked_fill(self_mask, 0.0)).sum(1)
        valid = n_pos > 0
        return (loss[valid] / n_pos[valid]).mean()


class ProtoNCELoss(nn.Module):
    """Class-mean prototype contrastive CE."""

    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique()  # type: ignore[no-untyped-call]
        if len(classes) < 2:
            return (features * 0).sum()
        protos = F.normalize(torch.stack([features[labels == c].mean(0) for c in classes]), dim=1)
        sim = torch.mm(features, protos.T) / self.temperature
        c2l = {c.item(): i for i, c in enumerate(classes)}
        local = torch.tensor(
            [c2l[y.item()] for y in labels], dtype=torch.long, device=features.device
        )
        return F.cross_entropy(sim, local)
