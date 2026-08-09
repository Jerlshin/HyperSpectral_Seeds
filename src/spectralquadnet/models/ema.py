"""Exponential moving average of model weights.

``ModelEMA.state_dict()`` returns the *shadow model's* ``state_dict()``, which is
what ``save_ckpt`` persists under the bundle's ``"ema"`` key — so the shadow's
top-level attribute names must match :class:`~spectralquadnet.models.spectral_quadnet.SpectralQuadNet`'s
checkpoint schema exactly.

Why the update is written with ``torch._foreach_*``
───────────────────────────────────────────────────
:meth:`ModelEMA.update` runs on **every optimiser step** of all three stages.
Written the obvious way — a Python loop calling ``sp.copy_(d * sp + (1-d) * lp)``
per parameter — it rebuilt ``dict(model.named_parameters())`` each time (≈200
string-keyed entries) and then issued four kernels per parameter: a scale, a
scale, an add and a copy. At this model's ~200 parameter tensors that is ~800
launches per step for an operation that moves 60 MB, which on any accelerator
fast enough to matter is entirely launch-bound. Measured on Metal at batch 32 it
cost 16 ms per step on its own.

The multi-tensor form issues **two** kernels for the whole model, and it is
written as ``mul_`` followed by ``add_`` of a *materialised* ``(1-d) * live``
rather than the shorter ``add_(live, alpha=1-d)`` on purpose: the ``alpha``
overload contracts to a fused multiply-add on some backends, which rounds once
where the original rounded twice. The long form keeps the arithmetic
bit-identical to the per-parameter loop it replaces, and the extra temporary is
one allocation from the caching allocator against the 400 the loop made.

The parameter pairing is resolved once and cached, keyed on the live model, so
the name lookup does not happen per step either.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

#: ``(shadow params, live params, shadow float buffers, live float buffers)`` —
#: name-matched and index-aligned, so the multi-tensor ops below can consume
#: them positionally.
_Tensors4 = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]

#: :data:`_Tensors4` keyed by the live model's ``id``, so a different model
#: rebuilds the pairing rather than silently updating from the wrong weights.
_Pairing = tuple[
    int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]


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
        #: ``(live_model_id, shadow_params, live_params, shadow_buffers, live_buffers)``
        #: — the name-matched pairing, resolved on first use. Keyed by the live
        #: model's identity so a different model (Stage 3 scores an SWA probe
        #: against the same wrapper) rebuilds it rather than silently updating
        #: from the wrong weights.
        self._pairing: (
            tuple[
                int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
            ]
            | None
        ) = None

    def _paired(self, model: nn.Module) -> _Tensors4:
        """Name-matched shadow/live parameter and float-buffer lists, memoised.

        Built in the shadow's ``named_parameters()`` order and filtered to the
        names the live model also has — the same ``if n in lp`` test the
        per-parameter loop applied, so a shadow entry with no live counterpart
        is still skipped rather than mis-paired positionally.
        """
        if self._pairing is not None and self._pairing[0] == id(model):
            return self._pairing[1], self._pairing[2], self._pairing[3], self._pairing[4]

        live_params = dict(model.named_parameters())
        shadow_p: list[torch.Tensor] = []
        model_p: list[torch.Tensor] = []
        for name, param in self.shadow.named_parameters():
            if name in live_params:
                shadow_p.append(param)
                model_p.append(live_params[name])

        if not shadow_p and live_params:
            # A name mismatch here is silent and catastrophic: the shadow simply
            # stops tracking, every stage keeps checkpointing it, and the run's
            # reported F1 comes from weights that never moved. The realistic
            # cause is being handed a wrapper — DDP prefixes every name with
            # ``module.`` and ``torch.compile`` with ``_orig_mod.`` — so the
            # message names the fix rather than the symptom.
            raise RuntimeError(
                "ModelEMA.update() matched none of the model's parameters against the shadow's. "
                "Pass the unwrapped module (see utils.device.unwrap_model): a DDP or compiled "
                f"wrapper renames every parameter. Live names begin "
                f"{sorted(live_params)[:2]}, shadow names begin "
                f"{[n for n, _ in self.shadow.named_parameters()][:2]}."
            )

        live_buffers = dict(model.named_buffers())
        shadow_b: list[torch.Tensor] = []
        model_b: list[torch.Tensor] = []
        for name, buffer in self.shadow.named_buffers():
            other = live_buffers.get(name)
            if other is not None and buffer.dtype.is_floating_point:
                shadow_b.append(buffer)
                model_b.append(other)

        self._pairing = (id(model), shadow_p, model_p, shadow_b, model_b)
        return shadow_p, model_p, shadow_b, model_b

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

        Arithmetically this is ``shadow = d * shadow + (1 - d) * live``, term
        for term and rounding for rounding — see the module docstring on why
        the two-kernel form is spelled out rather than folded into ``add_``'s
        ``alpha``.
        """
        self._num_updates += 1
        d = self.current_decay
        shadow_p, model_p, shadow_b, model_b = self._paired(model)

        if shadow_p:
            torch._foreach_mul_(shadow_p, d)
            torch._foreach_add_(shadow_p, torch._foreach_mul(model_p, 1.0 - d))
        if shadow_b:
            torch._foreach_copy_(shadow_b, model_b)

    def reinit_from(self, model: nn.Module) -> None:
        """Hard-reset the shadow to ``model``'s current weights and restart the decay warm-up.

        Used at stage-transition boundaries, where the loss landscape shifts
        enough (e.g. new head, new loss) that continuing the old average
        would anchor the shadow to a stale optimum.

        The source is **not** deep-copied first. It used to be, and that copy
        was a third full set of parameters resident on the accelerator for the
        length of the load — at a Stage 1 phase boundary, on top of the live
        model, the shadow, the optimiser's two AdamW moments and the outgoing
        phase's cached activations, which is where a run at the edge of its VRAM
        budget tips over (the ep-181 OOM). It bought nothing:
        ``Module.load_state_dict`` copies element-wise into the destination
        under ``no_grad``, so it neither aliases the source nor writes to it, and
        source and destination are distinct tensors regardless. Dropping it
        leaves the resulting weights bit-identical.
        """
        # `load_state_dict` copies into the existing tensors, so `_pairing`
        # stays valid — it holds the tensors themselves, not their values.
        self.shadow.load_state_dict(model.state_dict())
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def state_dict(self) -> dict[str, Any]:
        return self.shadow.state_dict()  # type: ignore[no-any-return]

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.shadow.load_state_dict(sd)
