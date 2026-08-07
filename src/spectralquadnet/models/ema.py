"""Exponential moving average of model weights.

``ModelEMA.state_dict()`` returns the *shadow model's* ``state_dict()``, which is
what ``save_ckpt`` persists under the bundle's ``"ema"`` key — so the shadow's
top-level attribute names must match :class:`~spectralquadnet.models.spectral_quadnet.SpectralQuadNet`'s
checkpoint schema exactly.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains a shadow copy of a model whose weights are an exponential moving average.

    The shadow is typically what gets evaluated and checkpointed: EMA weights
    trade a small amount of representational freshness for a smoother, less
    noisy optimum, which usually generalises better than the raw trained
    weights alone.
    """

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
        """Warms up from a low decay towards ``max_decay`` over the first ~10 updates.

        Early in training the raw weights move quickly and are still useful
        signal, so a low initial decay lets the shadow track them closely;
        the ``(1+n)/(10+n)`` schedule saturates to ``max_decay`` as ``n`` grows.
        """
        n = self._num_updates
        return min(self.max_decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the shadow's parameters towards ``model``'s by ``current_decay``.

        Buffers (e.g. BatchNorm running stats) are copied outright rather
        than averaged, since an EMA of running statistics is not itself a
        meaningful running statistic.
        """
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
        """Hard-reset the shadow to ``model``'s current weights and restart the decay warm-up.

        Used at stage-transition boundaries, where the loss landscape shifts
        enough (e.g. new head, new loss) that continuing the old average
        would anchor the shadow to a stale optimum.
        """
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
