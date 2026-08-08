"""Sharpness-Aware Minimisation optimiser wrapper.

Stage 3 only. The two-step contract is deliberate: :meth:`SAM.step` raises,
forcing callers through :meth:`~SAM.first_step` (ascend to the local worst
case) and :meth:`~SAM.second_step` (restore, then let the base optimiser
descend). The matching call sequence lives in
``engine/train_epoch.py::train_one_epoch_sam``.

:meth:`load_state_dict` re-points ``base_optimizer.param_groups`` at the
freshly-loaded groups; without it a resumed Stage 3 would update one set of
groups and step another.

Multi-tensor rewrite
────────────────────
Stage 3 runs this twice per batch on top of two full backward passes, and the
per-parameter form issued roughly ten kernels for each of the model's ~200
parameter tensors on every ``first_step`` alone — a clone, a scale, an optional
square-and-scale, an add — plus another ~200 in :meth:`_grad_norm`. Every one of
those operations is elementwise or a reduction over a single tensor, which is
exactly what ``torch._foreach_*`` exists to batch, so the same arithmetic now
issues a handful of launches for the whole model.

Two details keep it *the same* arithmetic rather than merely equivalent:

* the perturbation is still ``grad * scale`` then optionally ``* theta**2``
  then a plain ``add_`` — no ``alpha`` overload anywhere, because those contract
  to fused multiply-adds on some backends and round once where this rounds
  twice;
* the pre-ascent weights are copied into **reused** buffers rather than freshly
  cloned each step, and restored with a copy rather than a rebind. Same values,
  no per-step allocation, and the optimiser's own state stays keyed on the
  ``Parameter`` objects, which never change identity.
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
        #: Parameters perturbed by the last :meth:`first_step`, paired with the
        #: reusable buffers holding their pre-ascent values. Empty between
        #: steps; see the module docstring on why the buffers are reused.
        self._ascended: list[torch.Tensor] = []
        self._old_p: list[torch.Tensor] = []

    @staticmethod
    def _perturbs(group: dict[str, Any]) -> bool:
        """Whether this group takes part in the ascent at all (OP-5's alternative)."""
        return bool(group.get("perturb", True))

    def _stash(self, params: list[torch.Tensor]) -> None:
        """Record ``params``' current values into the reusable pre-ascent buffers.

        The buffers are allocated on the first step and kept; ``_foreach_copy_``
        refreshes them in one launch. They are rebuilt whenever the perturbed
        set changes shape — which happens exactly once, on the step where every
        parameter first has a gradient.
        """
        if len(self._old_p) != len(params) or any(
            b.shape != p.shape or b.dtype != p.dtype or b.device != p.device
            for b, p in zip(self._old_p, params, strict=True)
        ):
            self._old_p = [torch.empty_like(p) for p in params]
        torch._foreach_copy_(self._old_p, params)
        self._ascended = params

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascend each parameter by ``rho`` along its current gradient direction.

        Caches the pre-ascent weights so :meth:`second_step` can restore them.
        Call ``loss.backward()`` against the *un-ascended* weights before this,
        then recompute the loss and call ``loss.backward()`` again before
        :meth:`second_step`.

        Under ``adaptive`` the step is ``rho * theta^2 * g / ||theta * g||``
        — ASAM's element-wise normalisation, which is what makes the
        perturbation invariant to a per-parameter rescaling of the network.
        """
        norm = self._grad_norm()
        perturbed: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []

        for group in self.param_groups:
            if not self._perturbs(group):
                continue
            scale = group["rho"] / (norm + 1e-12)
            adaptive = bool(group.get("adaptive", False))
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            e_w = torch._foreach_mul(grads, scale.to(params[0]))
            if adaptive:
                torch._foreach_mul_(e_w, torch._foreach_pow([p.data for p in params], 2))
            perturbed.extend(params)
            updates.extend(e_w)

        if perturbed:
            self._stash(perturbed)
            torch._foreach_add_(perturbed, updates)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore(self, zero_grad: bool = False) -> None:
        """Undo :meth:`first_step`'s ascent without taking a descent step.

        The abort path: if the loss at the ascended point comes out non-finite,
        the batch must be skipped *and the weights put back*, or training
        continues from ``theta + rho*g_hat`` and the next ``first_step``
        overwrites the cached pre-ascent values — leaving the perturbation
        permanently baked into the model.
        """
        if self._ascended:
            torch._foreach_copy_(self._ascended, self._old_p[: len(self._ascended)])
            self._ascended = []
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore pre-ascent weights, then descend using the worst-case gradient.

        Must be called after re-computing and backpropagating the loss at
        the ascended point set by :meth:`first_step`.
        """
        if self._ascended:
            torch._foreach_copy_(self._ascended, self._old_p[: len(self._ascended)])
            self._ascended = []
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    # Deliberately breaks `Optimizer.step`'s contract so a caller cannot
    # accidentally take a single non-SAM step; the incompatible override is
    # the point, not an oversight.
    def step(self, closure: Callable[[], float] | None = None) -> None:  # type: ignore[override]
        raise NotImplementedError("Use first_step / second_step.")

    def _grad_norm(self) -> torch.Tensor:
        """``||g||`` — or ASAM's ``||T_theta g||`` — over the perturbed groups only.

        ``torch._foreach_norm(xs, 2)`` produces the same per-tensor 2-norms
        ``x.norm(p=2)`` does, bit for bit, in one launch instead of one per
        parameter; the outer ``torch.norm`` over the stack is unchanged, so the
        composed value is the composed value.
        """
        dev = self.param_groups[0]["params"][0].device
        ns: list[torch.Tensor] = []
        for g in self.param_groups:
            if not self._perturbs(g):
                continue
            params = [p for p in g["params"] if p.grad is not None]
            if not params:
                continue
            if g.get("adaptive", False):
                scaled = torch._foreach_mul(
                    [p.data.abs() for p in params], [p.grad for p in params]
                )
                ns.extend(torch._foreach_norm(scaled, 2))
            else:
                ns.extend(torch._foreach_norm([p.grad for p in params], 2))
        if not ns:
            return torch.tensor(0.0)
        total: torch.Tensor = torch.norm(torch.stack([n.to(dev) for n in ns]), p=2)
        return total.clamp(min=1e-6)

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
