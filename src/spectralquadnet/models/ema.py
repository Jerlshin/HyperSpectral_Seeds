"""Exponential moving average of model weights.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

======================  ==============
Symbol                  Baseline lines
======================  ==============
:class:`ModelEMA`       207-246
======================  ==============

``ModelEMA.state_dict()`` returns the *shadow model's* ``state_dict()``, which is
what ``save_ckpt`` persists under the bundle's ``"ema"`` key — so the shadow
shares every attribute-name constraint listed in REFACTOR_PLAN.md §3.1.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.max_decay = decay
        self._num_updates = 0
        # Deliberately `Any`: ModelEMA is model-agnostic, but callers reach through
        # the shadow to the wrapped model's own API (`use_arcface`, `arcface_head`,
        # `branch_drop_prob`), which a plain `nn.Module` annotation would reject.
        self.shadow: Any = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @property
    def current_decay(self) -> float:
        n = self._num_updates
        return min(self.max_decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._num_updates += 1
        d = self.current_decay
        lp = dict(model.named_parameters())
        for n, sp in self.shadow.named_parameters():
            if n in lp:
                sp.copy_(d * sp + (1.0 - d) * lp[n])
        lb = dict(model.named_buffers())
        for n, sb in self.shadow.named_buffers():
            if n in lb and sb.dtype.is_floating_point:
                sb.copy_(lb[n])

    def reinit_from(self, model: nn.Module) -> None:
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def state_dict(self) -> dict[str, Any]:
        return self.shadow.state_dict()  # type: ignore[no-any-return]

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.shadow.load_state_dict(sd)
