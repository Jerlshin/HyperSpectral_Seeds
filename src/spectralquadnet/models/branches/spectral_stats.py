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
    **chords**: for every pair of bands :math:`a, b` bracketing :math:`i`, the
    line through :math:`(\\lambda_a, r_a)` and :math:`(\\lambda_b, r_b)`
    evaluated at :math:`\\lambda_i`. The interpolation weights depend only on the
    wavelength grid, so they are precomputed once as buffers and the envelope is
    a masked maximum — exact, differentiable through the two active endpoints,
    and free of any iteration count to tune.

    The chord tensor is :math:`O(C^3)` in the band count. At :math:`C = 40` that
    is 64,000 entries and the forward chunks over the left endpoint to bound the
    activation to ``chunk * C * C`` per batch element. The buffers are
    **non-persistent**: they are a deterministic function of the wavelength
    vector already carried by ``wl_pe_cnn.pe``, and at cubic size there is no
    reason to put a second copy of that information into every checkpoint.
    """

    chord_w: torch.Tensor
    chord_valid: torch.Tensor

    def __init__(
        self,
        wavelengths: torch.Tensor,
        n_depths: int = 16,
        chunk: int = 8,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        lam = wavelengths.detach().flatten().float()
        n_bands = int(lam.numel())
        self.n_depths = min(int(n_depths), n_bands)
        self.chunk = max(1, int(chunk))
        self.eps = float(eps)

        la = lam[:, None, None]  # (C_a, 1, 1)
        lb = lam[None, :, None]  # (1, C_b, 1)
        li = lam[None, None, :]  # (1, 1, C_i)
        span = lb - la
        valid = (span > 0) & (li >= la) & (li <= lb)
        # t = 0 at the left endpoint, 1 at the right; the chord value is
        # (1 - t) * r_a + t * r_b, so only `t` needs storing.
        t = torch.where(valid, (li - la) / span.clamp(min=1e-12), torch.zeros_like(span + li))

        self.register_buffer("chord_w", t, persistent=False)
        self.register_buffer("chord_valid", valid, persistent=False)

    def envelope(self, r: torch.Tensor) -> torch.Tensor:
        """The upper concave envelope of ``r``, evaluated at every band. ``(B, C)``."""
        env = r.clone()
        n_bands = r.shape[1]
        for start in range(0, n_bands, self.chunk):
            stop = min(start + self.chunk, n_bands)
            t = self.chord_w[start:stop]  # (A, C_b, C_i)
            valid = self.chord_valid[start:stop]
            chords = (
                r[:, start:stop, None, None] * (1.0 - t)  # (B, A, C_b, C_i)
                + r[:, None, :, None] * t
            )
            chords = chords.masked_fill(~valid, -torch.inf)
            env = torch.maximum(env, chords.amax(dim=(1, 2)))
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
