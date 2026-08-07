"""Classification heads — sub-centre ArcFace and the per-branch auxiliary head.

``AdaptiveSubcenterArcFaceHead`` owns two buffers of the checkpoint schema:
``margins`` of shape ``(num_classes,)`` and, since Tier 2, ``confusion`` of
shape ``(num_classes, num_classes)``. Both appear in ``state_dict()`` as
``arcface_head.margins`` / ``arcface_head.confusion``.

Three Tier-2 items live here.

* **HD-1 (T2-10)** — this is now the *only* head. Stage 1 runs it at
  ``global_m = 0``, which is a plain cosine (NormFace) classifier, so the
  Stage-1 → Stage-2 transition changes the margin and nothing else.
* **HD-2 (T2-9)** — the class cosine is a temperature-annealed log-sum-exp over
  the class's sub-centres rather than a hard ``max_k``, the head reports the
  soft assignment so a load-balancing term can be added to the loss, and the
  sub-centres are seeded by spherical *k*-means on real embeddings instead of
  by jittering one vector.
* **HD-3 (T2-8)** — the per-class margin is driven by the *signed* difference
  ``R_c - P_c`` rather than by ``F1_c``, a pairwise confusion term targets the
  margin at the classes actually confused with ``y``, and ``theta + m`` is
  capped below ``pi/2``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

#: How far the cosine is held away from ±1 before ``sqrt(1 - c**2)`` is taken.
#:
#: ``d/dc sqrt(1-c^2) = -c / sqrt(1-c^2)`` blows up at the clamp: the original
#: ``1e-6`` admits a derivative of **707**, which the ``s = 48`` scaling then
#: multiplies, so one perfectly-aligned sample could dominate a global
#: ``clip_grad_norm_`` and dictate the update direction for every parameter in
#: the model (IMPROVEMENT_PLAN §2.4.4 N-4, §3.5 HD-4). At ``1e-3`` the bound is
#: **22.4**, a 32× reduction, and the angular resolution given up is
#: ``arccos(1 - 1e-3) = 2.6°`` against margins of 20–26° — immaterial.
#:
#: Read as a module global inside ``forward`` so a test can vary it.
COS_CLAMP_EPS: float = 1e-3

#: ``theta + m`` is capped here (HD-3, §3.5). Past ``pi/2`` the ArcFace
#: gradient magnitude ``s*sin(theta + m)`` starts *decreasing* with the angle,
#: so the samples a margin is meant to push hardest are pushed least (M-7).
#: The cap removes that regime entirely and subsumes the ``phi = cos - mm``
#: easy-margin guard it replaces, which only became active past ``pi - m``.
HALF_PI: float = math.pi / 2.0


def _spherical_kmeans(
    x: torch.Tensor, k: int, iters: int, generator: torch.Generator
) -> torch.Tensor:
    """``k`` unit-norm centres for the unit-norm rows of ``x``, by cosine *k*-means.

    Assignment is ``argmax_j cos(x_i, mu_j)`` and the update is the normalised
    mean of each cluster — the spherical variant, which is the right one for
    vectors an ArcFace head will compare by angle.

    Centres are seeded **k-means++**-style, by sampling each new centre with
    probability proportional to its squared cosine distance from the nearest
    centre chosen so far. Plain random-row seeding regularly lands two centres
    inside one mode and leaves a third mode uncovered — the local optimum
    Lloyd's algorithm cannot escape, and precisely the dead sub-centre this
    function exists to prevent.

    An empty cluster is re-seeded on the point currently *worst* served by its
    own centre, so ``k`` centres always come back distinct as long as ``x`` has
    ``k`` distinct rows. With fewer rows than ``k`` the surplus centres are
    seeded from the rows that exist plus small tangential noise, since a class
    with two samples cannot support three prototypes.

    Args:
        x: ``(n, d)`` unit-norm embeddings for one class.
        k: Number of sub-centres to produce.
        iters: Lloyd iterations.
        generator: CPU generator, so the seeding is reproducible.

    Returns:
        ``(k, d)`` unit-norm centres.
    """
    n = x.shape[0]
    if n <= k:
        pad = x[torch.randint(0, n, (k - n,), generator=generator)]
        centres = torch.cat([x, pad + 0.01 * torch.randn(pad.shape, generator=generator)], dim=0)
        return F.normalize(centres, dim=1)

    chosen = [int(torch.randint(0, n, (1,), generator=generator).item())]
    nearest = x @ x[chosen[0]]
    for _ in range(k - 1):
        weights = (1.0 - nearest).clamp_min(0.0).pow(2)
        if float(weights.sum()) <= 0.0:
            weights = torch.ones(n)
        pick = int(torch.multinomial(weights, 1, generator=generator).item())
        chosen.append(pick)
        nearest = torch.maximum(nearest, x @ x[pick])
    centres = x[chosen].clone()

    for _ in range(iters):
        sim = x @ centres.t()  # (n, k)
        assign = sim.argmax(dim=1)
        for j in range(k):
            members = x[assign == j]
            if members.numel() == 0:
                # Re-seed on the point its own centre serves worst.
                worst = sim.gather(1, assign.view(-1, 1)).squeeze(1).argmin()
                centres[j] = x[worst]
            else:
                centres[j] = members.sum(dim=0)
        centres = F.normalize(centres, dim=1)
    return centres


class AdaptiveSubcenterArcFaceHead(nn.Module):
    """Sub-centre ArcFace margin head with per-class, precision/recall-adaptive margins.

    Each class owns ``K`` sub-centre weight vectors. The class cosine pools
    them with a temperature-annealed log-sum-exp,

    .. math::
        \\cos\\theta_{i,c} = \\tau\\,\\log\\sum_k
        \\exp\\!\\big(\\hat e_i^\\top \\hat W_{c,k} / \\tau\\big),
        \\qquad \\tau: 0.20 \\to 0.02,

    which recovers the hard ``max_k`` exactly as ``tau -> 0`` but gives every
    sub-centre gradient while ``tau`` is large, so none can die before it has
    seen data (HD-2(i) / M-8). :meth:`balance_loss` is the companion
    load-balancing term.

    The additive angular margin is per-class, recalibrated by
    :meth:`update_margins_from_pr`, and capped so ``theta + m < pi/2``.

    Shape:
        Input embedding ``x``: ``(B, in_dim)``. Output logits: ``(B, num_classes)``.
    """

    #: Declares the `register_buffer`s below for the type checker. `nn.Module`'s
    #: `__getattr__` is typed `Tensor | Module`, so without this every use of
    #: `self.margins` reads as possibly-a-Module. A bare annotation creates no
    #: attribute at runtime.
    margins: torch.Tensor
    confusion: torch.Tensor

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        K: int = 2,
        s: float = 32.0,
        m_base: float = 0.35,
        m_delta: float = 0.20,
        m_min: float = 0.20,
        m_max: float = 0.50,
        tau: float = 0.0,
        pairwise_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.K = K
        self.C = num_classes
        self.s = s
        self.m_base = m_base
        self.m_delta = m_delta
        self.m_min = m_min
        self.m_max = m_max
        #: Sub-centre pooling temperature. A *schedule*, not state: the stage
        #: loops set it per epoch, so it is a plain float rather than a buffer
        #: and never enters the checkpoint.
        self.tau = float(tau)
        self.pairwise_delta = float(pairwise_delta)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))
        self.register_buffer("confusion", torch.zeros(num_classes, num_classes))

    # ── Calibration ───────────────────────────────────────────────────

    def set_tau(self, tau: float) -> None:
        """Set the sub-centre pooling temperature (``<= 0`` selects hard ``max_k``)."""
        self.tau = float(tau)

    def update_margins_from_pr(
        self, precision: Mapping[int, float], recall: Mapping[int, float]
    ) -> None:
        """Recalibrate each class's margin from the **signed** gap ``R_c - P_c`` (HD-3).

        .. math::
            M(c) = \\operatorname{clip}\\big(m_{base} + m_\\Delta (R_c - P_c),\\;
            m_{min},\\; m_{max}\\big)

        An additive angular margin is a *penalty on the class it is applied to*
        — it requires ``cos(theta_y + m)`` to beat every unmargined
        ``cos theta_c`` — so it shrinks that class's decision region. A class
        that **over-claims** (low precision, high recall) should therefore have
        its margin raised, and a class that **under-claims** (low recall) should
        have it *lowered*. ``F1`` conflates the two, and the rule this replaces
        raised the margin for every low-``F1`` class, which is the wrong
        direction for exactly the hard classes it targeted (§2.4.1 M-6).

        Args:
            precision: Per-class precision, keyed by class index.
            recall: Per-class recall, keyed by class index. Classes missing
                from either mapping keep their current margin.
        """
        for c in sorted(set(precision) & set(recall)):
            raw = self.m_base + self.m_delta * (float(recall[c]) - float(precision[c]))
            self.margins[c] = min(max(raw, self.m_min), self.m_max)
        print(
            f"[INFO] ArcFace margins  mean={self.margins.mean():.3f}  "
            f"min={self.margins.min():.3f}  max={self.margins.max():.3f}"
        )

    @torch.no_grad()
    def set_confusion(self, confusion: torch.Tensor) -> None:
        """Store the row-normalised confusion matrix driving the pairwise margin.

        ``Omega[y, c]`` becomes the fraction of class ``y``'s calibration
        samples predicted as ``c``; the diagonal is zeroed, since a class is
        never pushed away from itself. An all-zero ``Omega`` — the state a
        freshly constructed head is in — makes the pairwise term identically
        zero, so the head is a plain sub-centre ArcFace head until this is
        called.
        """
        cm = torch.as_tensor(confusion, dtype=self.confusion.dtype).to(self.confusion.device)
        cm = cm / cm.sum(dim=1, keepdim=True).clamp_min(1.0)
        cm = cm.clone()
        cm.fill_diagonal_(0.0)
        self.confusion.copy_(cm)

    @torch.no_grad()
    def init_subcentres_from_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        iters: int = 25,
        seed: int = 0,
    ) -> dict[str, float]:
        """Seed each class's ``K`` sub-centres by spherical *k*-means (HD-2(iii)).

        The scheme this replaces set every sub-centre to one vector plus
        ``0.01 * k`` noise, i.e. two random decoys 9–18° from a real prototype.
        Under hard ``max_k`` assignment a decoy that never wins never receives
        gradient, so it stays a decoy for the whole run — the mechanism behind
        the dead sub-centres 0-I counted (M-8). Real cluster centres start
        inside their own data, so every sub-centre wins something on step one.

        Args:
            embeddings: ``(N, in_dim)`` embeddings; normalised internally.
            labels: ``(N,)`` class indices aligned with ``embeddings``.
            iters: Lloyd iterations per class.
            seed: Seed for the CPU generator driving centre initialisation.

        Returns:
            ``{"classes_seeded": …, "min_separation": …}`` — how many classes
            had samples, and the smallest cosine gap achieved between two
            sub-centres of one class (1.0 would mean they coincide).
        """
        e = F.normalize(embeddings.detach().float(), dim=1)
        y = labels.detach().to(e.device)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        seeded = 0
        worst = -1.0
        for c in range(self.C):
            rows = (y == c).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            centres = _spherical_kmeans(e[rows].cpu(), self.K, iters, generator)
            self.weight[c * self.K : (c + 1) * self.K].copy_(centres.to(self.weight.device))
            seeded += 1
            if self.K > 1:
                gram = centres @ centres.t()
                off = gram - torch.eye(self.K) * 2.0
                worst = max(worst, float(off.max()))
        print(f"[INFO] ArcFace (K={self.K}) sub-centres seeded by spherical k-means.")
        return {"classes_seeded": float(seeded), "min_separation": worst}

    # ── Forward ───────────────────────────────────────────────────────

    def subcentre_cosines(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, K)`` clamped cosines between the embedding and every sub-centre."""
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        eps = COS_CLAMP_EPS
        return F.linear(x_n, w_n).clamp(-1 + eps, 1 - eps).view(-1, self.C, self.K)

    def pool_subcentres(self, sub: torch.Tensor) -> torch.Tensor:
        """Pool ``(B, C, K)`` sub-centre cosines to ``(B, C)``.

        ``tau * logsumexp(sub / tau)`` at ``tau > 0``, hard ``max_k`` otherwise.
        The log-sum-exp exceeds the max by up to ``tau * log K`` (0.22 at
        ``tau = 0.2``, ``K = 3``), which can leave the valid cosine range, so
        the result is re-clamped — the margin algebra below needs
        ``sqrt(1 - c^2)`` to be real.
        """
        pooled = (
            self.tau * torch.logsumexp(sub / self.tau, dim=2)
            if self.tau > 0.0
            else sub.max(dim=2).values
        )
        eps = COS_CLAMP_EPS
        return pooled.clamp(-1 + eps, 1 - eps)

    def assignment(self, sub: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """``(B, K)`` soft assignment of each sample to **its own class's** sub-centres.

        This is the ``pi`` of HD-2(ii). At ``tau <= 0`` the assignment is the
        hard one-hot, which carries no gradient — correct, because there is
        nothing left to balance once the pooling is a max.
        """
        own = sub.gather(1, labels.view(-1, 1, 1).expand(-1, 1, self.K)).squeeze(1)
        if self.tau > 0.0:
            return torch.softmax(own / self.tau, dim=1)
        return F.one_hot(own.argmax(dim=1), self.K).to(own.dtype)

    def balance_loss(self, assign: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """``sum_c KL(pi_c || uniform)`` over the classes present in the batch (HD-2(ii)).

        The standard mixture-of-experts load-balancing term, which is exactly
        the mechanism sub-centre ArcFace is missing: nothing else in the
        objective rewards using more than one sub-centre per class, so the
        first one to win keeps winning.

        Classes with no sample in the batch are skipped rather than counted as
        balanced — ``pi`` is undefined for them, and with a 16×8 balanced batch
        that is 74 of the 90 classes on every step.
        """
        onehot = F.one_hot(labels, self.C).to(assign.dtype)  # (B, C)
        counts = onehot.sum(dim=0)  # (C,)
        present = counts > 0
        if not bool(present.any()):
            return assign.new_zeros(())
        pi = (onehot.t() @ assign) / counts.clamp_min(1.0).unsqueeze(1)  # (C, K)
        pi = pi[present].clamp_min(1e-8)
        return (pi * (pi * self.K).log()).sum(dim=1).sum()

    def _margined_target(
        self, cosine: torch.Tensor, labels: torch.Tensor, global_m: float | None
    ) -> torch.Tensor:
        """``cos(theta_y + m)`` for the target column, with HD-3's ``pi/2`` cap."""
        with torch.no_grad():
            m_per = (
                torch.full((cosine.shape[0],), global_m, device=cosine.device)
                if global_m is not None
                else self.margins[labels]
            )
            # theta + m <= pi/2. Computed under no_grad: the cap is a scheduling
            # decision about *which* margin applies, not a term to differentiate
            # through, and `acos`'s derivative would otherwise re-introduce the
            # 1/sqrt(1-c^2) factor HD-4 exists to bound.
            theta_y = torch.acos(cosine.gather(1, labels.view(-1, 1)).squeeze(1))
            m_per = torch.minimum(m_per, (HALF_PI - theta_y).clamp_min(0.0))

        tgt_c = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        tgt_s = torch.sqrt(torch.clamp(1 - tgt_c**2, min=1e-6))
        return tgt_c * torch.cos(m_per) - tgt_s * torch.sin(m_per)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        global_m: float | None = None,
        return_assign: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Scaled cosine logits, with the additive angular margin at train time.

        Args:
            x: L2-normalisable embedding, shape ``(B, in_dim)``.
            labels: Ground-truth class indices, shape ``(B,)``. Required to
                apply the margin; ignored (and the margin skipped) if ``None``
                or if the module is not in training mode.
            global_m: If given, overrides the per-class ``margins`` buffer with
                a single scalar margin for every sample. ``0.0`` — what Stage 1
                passes under HD-1 — makes this a plain cosine classifier.
            return_assign: Also return the ``(B, K)`` soft sub-centre
                assignment for :meth:`balance_loss`.

        Returns:
            Logits scaled by ``self.s``, shape ``(B, num_classes)``; or
            ``(logits, assignment)`` when ``return_assign`` is set.
        """
        sub = self.subcentre_cosines(x)
        cosine = self.pool_subcentres(sub)

        if labels is None or not self.training:
            logits = cosine
        else:
            if global_m is None or global_m > 0.0:
                phi = self._margined_target(cosine, labels, global_m)
                logits = cosine.scatter(1, labels.view(-1, 1), phi.unsqueeze(1))
            else:
                # A zero global margin is exactly `cos(theta_y + 0)`; taking the
                # algebra above would be the same float at 10x the work.
                logits = cosine
            if self.pairwise_delta > 0.0:
                omega = self.confusion[labels].scatter(1, labels.view(-1, 1), 0.0)
                logits = logits - self.pairwise_delta * omega

        out = logits * self.s
        if return_assign:
            assign = (
                self.assignment(sub, labels)
                if labels is not None
                else sub.new_zeros((sub.shape[0], self.K))
            )
            return out, assign
        return out


class AuxiliaryHead(nn.Module):
    """Small MLP classifier attached to a single branch for deep supervision.

    Used only during Stage-1 training so each branch is individually
    discriminative before cross-modal fusion; not called during inference.
    """

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
