"""Branch B — a scale-invariant spectral index bank (BR-1 / T3-1).

What this branch used to be
───────────────────────────
Nine masked per-band statistics — mean, std, max, skew, kurtosis and four
percentiles — stacked into a ``(B, 9, C)`` tensor and pushed through three
1-D residual towers, 686 k parameters' worth. §2.2.5 proves that tensor has
rank :math:`\\le 2`. Under the per-pixel gain model :math:`x_{c,p} = a_p r_c`
every one of the nine is :math:`r_c` times a moment of the gain distribution,
so all nine rows are collinear with :math:`\\mathbf{r}` up to a per-statistic
scalar; Phase 0's 0-D measured :math:`\\sigma_3/\\sigma_1` on the real data and
confirmed it. 686 k parameters were reading a two-dimensional signal.

What it is now
──────────────
Three groups, none rank-deficient under that gain model, on ~95 k parameters:

**(i) A learned soft index bank** (:class:`SoftIndexBank`). Normalised
differences :math:`(u - v)/(u + v)` are the classical scale-invariant spectral
feature — exactly invariant to :math:`\\mathbf{r} \\mapsto a\\mathbf{r}` — and
rather than enumerate all :math:`\\binom{40}{2} = 780` band pairs the branch
learns 64 of them as *pairs of soft spectral regions*, each a simplex vector
over the band axis. 5,120 parameters, and each index is readable as "this
region against that one".

**(ii) Continuum-removed depths** (:class:`ContinuumDepths`). The band depth
:math:`1 - r_i/\\mathrm{hull}(r)_i` against the spectrum's own upper concave
envelope is the standard absorption-feature descriptor in reflectance
spectroscopy, and it is gain-free for the same reason: the envelope is
positively homogeneous, so the ratio is invariant.

**(iii) Morphology.** The eight persisted morphometrics of P-4 / T4-4 — physical
size and shape, which no spectral operator anywhere in the network can derive.

The concatenation is full rank by construction: (i) is invariant to the gain
that (ii)'s hull is computed against, and (iii) is not spectral at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Guards the normalised-difference denominator. Small enough that the
#: invariance of §3.3 BR-1 holds to ~1e-5 on unit-scale reflectance, large
#: enough that a spectrum whose two soft regions both integrate to zero
#: produces 0 rather than a NaN.
NDI_EPS: float = 1e-6


class SoftIndexBank(nn.Module):
    """BR-1(i) — ``n_indices`` learned normalised-difference indices.

    .. math::
        \\boldsymbol\\pi^{\\pm}_k = \\mathrm{softmax}(\\theta^{\\pm}_k) \\in \\Delta^{C-1},
        \\qquad
        z_k = \\frac{\\boldsymbol\\pi^{+\\top}_k \\mathbf r
                     - \\boldsymbol\\pi^{-\\top}_k \\mathbf r}
                    {|\\boldsymbol\\pi^{+\\top}_k \\mathbf r|
                     + |\\boldsymbol\\pi^{-\\top}_k \\mathbf r| + \\epsilon}

    The absolute values in the denominator are the plan's :math:`u + v` on the
    domain the plan assumes — reflectance is non-negative, so both soft averages
    are — and are what keeps the index finite and inside :math:`[-1, 1]` if a
    caller ever hands the branch a mean-centred spectrum instead.

    :math:`\\theta^{\\pm}` are raw ``nn.Parameter``\\ s rather than a ``Linear``,
    so ``SpectralQuadNet._init_weights`` (which walks ``Conv``/``Norm``/``Linear``
    only) leaves the small init that keeps the initial simplex vectors close to
    uniform — every band participating in every index at the start, none of them
    dead before they have seen a gradient.
    """

    def __init__(self, num_bands: int, n_indices: int = 64) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.n_indices = int(n_indices)
        self.theta_pos = nn.Parameter(torch.randn(self.n_indices, self.num_bands) * 0.02)
        self.theta_neg = nn.Parameter(torch.randn(self.n_indices, self.num_bands) * 0.02)

    def selectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The two ``(n_indices, C)`` simplex vectors — the interpretable half."""
        return torch.softmax(self.theta_pos, dim=-1), torch.softmax(self.theta_neg, dim=-1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """``(B, C) -> (B, n_indices)``."""
        pi_pos, pi_neg = self.selectors()
        u = r @ pi_pos.t()
        v = r @ pi_neg.t()
        return (u - v) / (u.abs() + v.abs() + NDI_EPS)


class ContinuumDepths(nn.Module):
    """BR-1(ii) — the ``n_depths`` deepest absorption features after hull removal.

    The upper concave envelope of the samples :math:`(\\lambda_i, r_i)` is, by
    Carathéodory's theorem in one dimension, the pointwise maximum over
    **chords**: for every pair of bands :math:`a \\le i \\le b`, the line through
    :math:`(\\lambda_a, r_a)` and :math:`(\\lambda_b, r_b)` evaluated at
    :math:`\\lambda_i`. Written that way the envelope is a maximum over
    :math:`O(C^3)` values, which is what this module used to compute.

    Why that had to change for the full cube
    ────────────────────────────────────────
    :math:`O(C^3)` is 64,000 chords at :math:`C = 40` and **16.8 million** at
    :math:`C = 256`. The old implementation held four dense
    :math:`(C, C, C)` buffers — 570 MB of them at 256 bands, resident for the
    whole run — and materialised a ``(B, chunk, C, C)`` activation per chunk,
    which is 268 MB per chunk at batch 128. That is not a full-cube operator with
    a large constant; it is a 40-band operator that does not fit.

    The exact :math:`O(C^2)` form
    ─────────────────────────────
    The chord is affine in the right endpoint's *slope*, with a **non-negative**
    coefficient for :math:`a \\le i`:

    .. math::
        \\mathrm{chord}(a, b, i)
        = r_a + (\\lambda_i - \\lambda_a)\\,
          \\underbrace{\\frac{r_b - r_a}{\\lambda_b - \\lambda_a}}_{s(a,\\,b)}

    so maximising over :math:`b \\ge i` is maximising :math:`s(a, b)` over
    :math:`b \\ge i` — a single **suffix maximum** along the band axis, obtained
    from one reversed :func:`torch.cummax`. The envelope is then a maximum over
    the left endpoint alone:

    .. math::
        \\mathrm{hull}(r)_i
        = \\max\\Big(r_i,\\; \\max_{a \\le i} \\mathrm{chord}\\big(a, b^*(a, i), i\\big)\\Big),
        \\qquad b^*(a, i) = \\arg\\max_{b \\ge i} s(a, b)

    Two tensors of :math:`O(B C^2)`, chunked over :math:`a` exactly as before, and
    no persistent buffer larger than the wavelength vector itself.

    The value is computed from the *selected* endpoints in the original
    :math:`(1 - t) r_a + t r_b` parameterisation rather than from the slope, so
    the arithmetic at each surviving chord is the one the previous
    implementation performed — the 40-band arms and the golden gates reproduce
    bit for bit, which ``tests/unit/test_masked_ops.py`` asserts against a
    literal transcription of the :math:`O(C^3)` enumeration.

    Ordering
    ────────
    The suffix maximum is a statement about :math:`\\lambda`, so the module works
    in a :math:`\\lambda`-ascending permutation of the band axis and permutes the
    input on the way in. On the shipped wavelength files that permutation is the
    identity; it exists so a CSV in selection order rather than spectral order is
    handled instead of silently producing a wrong hull.
    """

    lam: torch.Tensor
    order: torch.Tensor

    def __init__(
        self,
        wavelengths: torch.Tensor,
        n_depths: int = 16,
        chunk: int = 32,
        eps: float = 1e-5,
    ) -> None:
        """
        Args:
            wavelengths: ``(C,)`` band wavelengths, in any unit and any order.
            n_depths: How many of the deepest features to return, clamped to
                ``C``.
            chunk: Left endpoints processed per iteration. A pure memory knob:
                the chunks are combined with :func:`torch.maximum`, which selects
                one of its inputs exactly, so the chunk size **cannot** move a
                returned value.
            eps: Floor under the envelope in the depth ratio.
        """
        super().__init__()
        raw = wavelengths.detach().flatten().float()
        n_bands = int(raw.numel())
        self.n_bands = n_bands
        self.n_depths = min(int(n_depths), n_bands)
        self.chunk = max(1, int(chunk))
        self.eps = float(eps)

        # `stable=True` so an exactly repeated wavelength keeps band order, which
        # makes the operator a deterministic function of the wavelength vector.
        order = torch.argsort(raw, stable=True)
        # Non-persistent for the reason the chord buffers were: both are exact
        # functions of the wavelength vector the checkpoint already carries, and
        # a second copy is a second thing that can disagree with it.
        self.register_buffer("order", order, persistent=False)
        self.register_buffer("lam", raw[order], persistent=False)
        self.is_sorted = bool(torch.equal(order, torch.arange(n_bands)))

    def envelope(self, r: torch.Tensor) -> torch.Tensor:
        """The upper concave envelope of ``r``, evaluated at every band. ``(B, C)``.

        ``r`` and the return value are both in the caller's band order; the
        :math:`\\lambda`-ascending permutation is internal.
        """
        sorted_r = r if self.is_sorted else r[:, self.order]
        env = self._envelope_sorted(sorted_r)
        if self.is_sorted:
            return env
        out = torch.empty_like(env)
        out[:, self.order] = env
        return out

    def _envelope_sorted(self, r: torch.Tensor) -> torch.Tensor:
        """The envelope of a :math:`\\lambda`-ascending spectrum. ``(B, C) -> (B, C)``."""
        lam = self.lam.to(dtype=r.dtype, device=r.device)
        n_bands = int(lam.numel())

        # span[a, b] = λ_b − λ_a. Strictly positive exactly where b is a legal
        # right endpoint for left endpoint a, which is the `span > 0` predicate
        # the O(C³) form applied to the same quantity.
        span = lam[None, :] - lam[:, None]
        legal = span > 0
        neg_inf = r.new_full((), -torch.inf)
        i_idx = torch.arange(n_bands, device=r.device)[None, :]

        env = r.clone()
        for start in range(0, n_bands, self.chunk):
            stop = min(start + self.chunk, n_bands)
            width = stop - start
            r_a = r[:, start:stop, None]  # (B, A, 1)
            lam_a = lam[start:stop, None]  # (A, 1)

            # `where` on the *denominator*, not on the quotient: dividing by a
            # clamped non-positive span would overflow to inf on the branch
            # `where` discards, and a zero incoming gradient times an infinite
            # local one is NaN, not zero.
            denom = torch.where(legal[start:stop], span[start:stop], torch.ones_like(lam_a))
            slope = torch.where(legal[start:stop], (r[:, None, :] - r_a) / denom, neg_inf)

            # Suffix maximum over b: reversed cummax, then flipped back, so
            # `best[..., i]` is the largest slope reachable from some b >= i and
            # `b_star[..., i]` is the band that attains it.
            best, best_rev_idx = slope.flip(-1).cummax(dim=-1)
            best = best.flip(-1)
            b_star = (n_bands - 1 - best_rev_idx).flip(-1)  # (B, A, C_i)

            # Only a <= i contributes, and only where some legal b >= i exists.
            usable = torch.arange(start, stop, device=r.device)[:, None] <= i_idx
            usable = usable & torch.isfinite(best)

            lam_b = lam[b_star]  # (B, A, C_i)
            # The original interpolation weight, recomputed at the selected
            # endpoints from the same float32 expression the chord buffer held.
            # Forced to 0 where the chord is discarded, which makes `1 - t`
            # finite there and keeps the masked positions' gradient exactly zero.
            t = torch.where(
                usable,
                (lam[None, None, :] - lam_a)
                / torch.where(usable, lam_b - lam_a, lam_b.new_ones(())),
                lam_b.new_zeros(()),
            )
            r_b = torch.gather(r[:, None, :].expand(-1, width, -1), 2, b_star)
            chord = r_a * (1.0 - t) + r_b * t

            env = torch.maximum(env, chord.masked_fill(~usable, -torch.inf).amax(dim=1))
        return env

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """``(B, C) -> (B, n_depths)``, sorted deepest first."""
        env = self.envelope(r)
        depth = 1.0 - r / env.clamp(min=self.eps)
        return torch.topk(depth, self.n_depths, dim=1).values


class SpectralStatsBranch(nn.Module):
    """Branch B — index bank + continuum depths + morphology, projected to ``out_dim``.

    The class name is the schema's: ``SpectralQuadNet.branch_b`` is a top-level
    checkpoint key and this is the module behind it. What it computes is BR-1's
    replacement for the nine moments, not the moments.
    """

    def __init__(
        self,
        num_bands: int,
        wavelengths: torch.Tensor,
        out_dim: int = 256,
        n_indices: int = 64,
        n_depths: int = 16,
        n_morph: int = 8,
        hidden: int = 256,
        drop: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_morph = int(n_morph)
        self.index_bank = SoftIndexBank(num_bands, n_indices)
        self.continuum = ContinuumDepths(wavelengths, n_depths=n_depths)

        in_dim = self.index_bank.n_indices + self.continuum.n_depths + self.n_morph
        self.in_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, r: torch.Tensor, morph: torch.Tensor | None = None) -> torch.Tensor:
        """The raw ``(B, n_indices + n_depths + n_morph)`` descriptor, before the MLP.

        Exposed because it is what 0-D's rank probe has to be repeated on: the
        validation criterion for T3-1 is a property of *this* tensor, not of the
        branch embedding it is projected into.
        """
        z = self.index_bank(r)
        depths = self.continuum(r)
        if morph is None:
            morph = r.new_zeros(r.shape[0], self.n_morph)
        return torch.cat([z, depths, morph.to(dtype=r.dtype)], dim=1)

    def forward(self, r: torch.Tensor, morph: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, C)`` foreground mean spectrum (+ optional ``(B, 8)`` morphometrics) → ``(B, out_dim)``."""
        return self.forward_features(self.features(r, morph))

    def forward_features(self, feat: torch.Tensor) -> torch.Tensor:
        """The branch from the raw descriptor onwards — the twin of Branch A's."""
        return self.mlp(self.in_norm(feat))  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
