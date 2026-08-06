"""Classification heads — sub-centre ArcFace and the per-branch auxiliary head.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=========================================  ==============
Symbol                                     Baseline lines
=========================================  ==============
:class:`AdaptiveSubcenterArcFaceHead`      617-678
:class:`AuxiliaryHead`                     1310-1326
=========================================  ==============

``AdaptiveSubcenterArcFaceHead`` owns a ``margins`` buffer of shape
``(num_classes,)`` — it is part of ``state_dict()`` and appears in the trained
checkpoints as ``arcface_head.margins``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSubcenterArcFaceHead(nn.Module):
    #: Declares the `register_buffer` below for the type checker. `nn.Module`'s
    #: `__getattr__` is typed `Tensor | Module`, so without this every use of
    #: `self.margins` reads as possibly-a-Module. A bare annotation creates no
    #: attribute and is erased by the AST no-op-move check (§3.2.1).
    margins: torch.Tensor

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        K: int = 2,
        s: float = 32.0,
        m_base: float = 0.35,
        m_delta: float = 0.10,
    ) -> None:
        super().__init__()
        self.K = K
        self.C = num_classes
        self.s = s
        self.m_base = m_base
        self.m_delta = m_delta
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))

    def update_margins_from_f1(self, class_f1: dict[int, float]) -> None:
        for c, f1 in class_f1.items():
            self.margins[c] = self.m_base + self.m_delta * (1.0 - min(float(f1), 1.0))
        print(
            f"[INFO] ArcFace margins  mean={self.margins.mean():.3f}  "
            f"min={self.margins.min():.3f}  max={self.margins.max():.3f}"
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        global_m: float | None = None,
    ) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        cosine = (
            F.linear(x_n, w_n).clamp(-1 + 1e-6, 1 - 1e-6).view(-1, self.C, self.K).max(dim=2).values
        )
        if labels is None or not self.training:
            return cosine * self.s
        m_per = (
            torch.full((x.shape[0],), global_m, device=x.device)
            if global_m is not None
            else self.margins[labels]
        )
        cosm = torch.cos(m_per)
        sinm = torch.sin(m_per)
        th = torch.cos(math.pi - m_per)
        mm = torch.sin(math.pi - m_per) * m_per
        sine = torch.sqrt(torch.clamp(1 - cosine**2, min=1e-6))
        tgt_c = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        tgt_s = sine.gather(1, labels.view(-1, 1)).squeeze(1)
        phi = tgt_c * cosm - tgt_s * sinm
        phi = torch.where(tgt_c > th, phi, tgt_c - mm)
        oh = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi.unsqueeze(1)) + ((1 - oh) * cosine)) * self.s

    def init_from_linear(self, linear_w: torch.Tensor) -> None:
        with torch.no_grad():
            wn = F.normalize(linear_w, dim=1)
            for k in range(self.K):
                noise = torch.randn_like(wn) * 0.01 * k
                self.weight[k :: self.K].copy_(wn + noise)
        print(f"[INFO] ArcFace (K={self.K}) bootstrapped from linear head.")


class AuxiliaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        # Conservative init: smaller std avoids saturating softmax early
        # `nn.Sequential.__getitem__` is typed `-> Module`, so `.weight`/`.bias` on
        # the indexed layer widen to `Tensor | Module`. Both are `nn.Linear` here.
        nn.init.trunc_normal_(self.net[0].weight, std=0.02)  # type: ignore[arg-type]
        nn.init.zeros_(self.net[0].bias)  # type: ignore[arg-type]
        nn.init.trunc_normal_(self.net[2].weight, std=0.02)  # type: ignore[arg-type]
        nn.init.zeros_(self.net[2].bias)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `nn.Module.__call__` is typed `-> Any` in torch's stubs.
        return self.net(x)  # type: ignore[no-any-return]
