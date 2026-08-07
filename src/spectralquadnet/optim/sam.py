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
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        # `adaptive` is SAM's own, not the base optimizer's — AdamW would
        # reject it — so it is deliberately absent from the `**kwargs`
        # forwarded below while still living in `defaults`, which is what puts
        # it on every param group.
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.adaptive = adaptive

    @staticmethod
    def _perturbs(group: dict[str, Any]) -> bool:
        """Whether this group takes part in the ascent at all (OP-5's alternative)."""
        return bool(group.get("perturb", True))

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascend each parameter by ``rho`` along its current gradient direction.

        Caches the pre-ascent weights in ``self.state[p]["old_p"]`` so
        :meth:`second_step` can restore them. Call ``loss.backward()``
        against the *un-ascended* weights before this, then recompute the
        loss and call ``loss.backward()`` again before :meth:`second_step`.

        Under ``adaptive`` the step is ``rho * theta^2 * g / ||theta * g||``
        — ASAM's element-wise normalisation, which is what makes the
        perturbation invariant to a per-parameter rescaling of the network.
        """
        norm = self._grad_norm()
        for group in self.param_groups:
            if not self._perturbs(group):
                continue
            scale = group["rho"] / (norm + 1e-12)
            adaptive = bool(group.get("adaptive", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                if adaptive:
                    e_w = e_w * p.data.pow(2)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore(self, zero_grad: bool = False) -> None:
        """Undo :meth:`first_step`'s ascent without taking a descent step.

        The abort path: if the loss at the ascended point comes out non-finite,
        the batch must be skipped *and the weights put back*, or training
        continues from ``theta + rho*g_hat`` and the next ``first_step``
        overwrites the cached ``old_p`` — leaving the perturbation permanently
        baked into the model.
        """
        for group in self.param_groups:
            for p in group["params"]:
                old_p = self.state[p].pop("old_p", None)
                if old_p is not None:
                    p.data = old_p
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
        """``||g||`` — or ASAM's ``||T_theta g||`` — over the perturbed groups only."""
        dev = self.param_groups[0]["params"][0].device
        ns = [
            ((p.data.abs() * p.grad) if g.get("adaptive", False) else p.grad).norm(p=2).to(dev)
            for g in self.param_groups
            if self._perturbs(g)
            for p in g["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(ns), p=2).clamp(min=1e-6) if ns else torch.tensor(0.0)

    @torch.no_grad()
    def perturbation_mass(self) -> dict[int, torch.Tensor]:
        """``||eps_group||^2`` per param-group index, without moving any weight.

        The measurement OP-5's validation criterion asks for: how the rho-budget
        is actually divided between the head and everything else. Callers pair
        it with each group's parameter count to get the share.
        """
        norm = self._grad_norm()
        out: dict[int, torch.Tensor] = {}
        for i, group in enumerate(self.param_groups):
            if not self._perturbs(group):
                out[i] = torch.zeros((), device=norm.device)
                continue
            scale = group["rho"] / (norm + 1e-12)
            adaptive = bool(group.get("adaptive", False))
            total = torch.zeros((), device=norm.device)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if adaptive:
                    e_w = e_w * p.data.pow(2)
                total = total + e_w.pow(2).sum().to(total)
            out[i] = total
        return out

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        super().load_state_dict(sd)
        self.base_optimizer.param_groups = self.param_groups
