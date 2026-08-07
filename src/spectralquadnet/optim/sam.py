"""Sharpness-Aware Minimisation optimiser wrapper.

Stage 3 only. The two-step contract is deliberate: :meth:`SAM.step` raises,
forcing callers through :meth:`~SAM.first_step` (ascend to the local worst
case) and :meth:`~SAM.second_step` (restore, then let the base optimiser
descend). The matching call sequence lives in
``engine/train_epoch.py::train_one_epoch_sam``.

:meth:`load_state_dict` re-points ``base_optimizer.param_groups`` at the
freshly-loaded groups; without it a resumed Stage 3 would update one set of
groups and step another.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


class SAM(torch.optim.Optimizer):
    """Wraps a base optimizer to seek minima that are flat, not just low.

    Rather than descending directly, each step first ascends by ``rho``
    along the gradient to the locally worst-case point in the weight
    neighbourhood (:meth:`first_step`), then computes the gradient *there*
    and lets the base optimizer descend from the original point using that
    worst-case gradient (:meth:`second_step`). Minimising this worst-case
    loss biases training towards wide, flat basins, which tend to
    generalise better than sharp minima with an equally low training loss.
    """

    def __init__(
        self,
        params: Iterable[Any],
        base_optimizer_cls: type[torch.optim.Optimizer],
        rho: float = 0.05,
        **kwargs: Any,
    ) -> None:
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascend each parameter by ``rho`` along its current gradient direction.

        Caches the pre-ascent weights in ``self.state[p]["old_p"]`` so
        :meth:`second_step` can restore them. Call ``loss.backward()``
        against the *un-ascended* weights before this, then recompute the
        loss and call ``loss.backward()`` again before :meth:`second_step`.
        """
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore pre-ascent weights, then descend using the worst-case gradient.

        Must be called after re-computing and backpropagating the loss at
        the ascended point set by :meth:`first_step`.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    # Deliberately breaks `Optimizer.step`'s contract so a caller cannot
    # accidentally take a single non-SAM step; the incompatible override is
    # the point, not an oversight.
    def step(self, closure: Callable[[], float] | None = None) -> None:  # type: ignore[override]
        raise NotImplementedError("Use first_step / second_step.")

    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        ns = [
            p.grad.norm(p=2).to(dev)
            for g in self.param_groups
            for p in g["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(ns), p=2).clamp(min=1e-6) if ns else torch.tensor(0.0)

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups
