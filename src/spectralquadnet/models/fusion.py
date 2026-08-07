"""Cross-modal fusion of the branch embeddings — FU-1(b)/2/4/5, T3-4.

The module this replaces was a Perceiver: four learned latent tokens
cross-attending to four modality tokens, twice, then averaged. §2.3 found five
separate defects in it and §3.4 prices the whole thing at 2.19 M parameters to
mix four 256-vectors. In order:

**M-1, latent collapse.** ``latents ~ randn(4, 256) * 0.02`` gives
:math:`\\|L_n\\| \\approx 0.32` against LayerNormed keys of norm 16 — a 50×
scale mismatch — so the first cross-attention returned nearly the same convex
combination for every latent, and 0-E measured
:math:`\\max_{n \\ne n'}\\cos(L_n, L_{n'})` in the trained checkpoints to confirm
it. Four latents, one function.

**M-2a, the destroyed confidence signal.** Each branch was independently
``LayerNorm``\\ ed *before* the gate saw it, and per-sample LayerNorm sets
:math:`\\|\\hat{\\mathbf b}\\| = \\sqrt d` identically for every branch and every
sample. A branch that produced almost nothing was rescaled to the same norm as
one that produced a confident answer, and the gate could not tell them apart.

**M-2b, an exclusive gate on a conjunctive problem.** ``Softmax`` over four
modalities forces :math:`\\sum_m g_m = 1`: attending to Branch C *costs*
attention on Branch B. Distinguishing two rice varieties needs spectral shape
**and** grain morphology, which is a conjunction the gate's own shape forbids.

**M-3, first order only.** Everything downstream of the latents was a weighted
*sum*. "This absorption depth together with that grain aspect ratio" is a
product, and no sum of the two vectors expresses it.

**N-10, the same block twice.** ``output_proj`` and ``EmbedNet`` were both
pre-LN residual MLPs on the same 256-vector, back to back.

FU-1(b) deletes the latents outright — with five modalities, latent
cross-attention compresses nothing — and replaces them with an explicit gated
low-rank bilinear pool:

.. math::
    \\boldsymbol\\nu = \\big(\\log\\|\\mathbf b_m\\|_2\\big)_m, \\qquad
    \\boldsymbol\\gamma = \\sigma\\Big(W_g\\big[\\hat{\\mathbf b}_1\\|\\cdots\\|
        \\hat{\\mathbf b}_M\\|\\boldsymbol\\nu\\big]\\Big) \\in (0,1)^M
.. math::
    \\mathbf f_1 = \\sum_m \\gamma_m \\hat{\\mathbf b}_m, \\qquad
    \\mathbf f_2 = \\sum_{m<m'} (U_m\\hat{\\mathbf b}_m) \\odot (U_{m'}\\hat{\\mathbf b}_{m'}),
    \\qquad \\mathbf f = W_o\\big[\\mathbf f_1 \\,\\|\\, V\\mathbf f_2\\big]

:math:`\\boldsymbol\\nu` is computed **before** normalisation, which is the
confidence scalar M-2a removed (FU-2); the gate is a sigmoid, so conjunctions
are expressible (FU-2); :math:`\\mathbf f_2` is the multiplicative term M-3
found missing; the per-modality norm is a ``BatchNorm1d``, a *dataset* statistic,
so a low-SNR sample is no longer amplified to unit scale (FU-2); and
``output_proj`` is gone, leaving :class:`EmbedNet` as the single post-fusion
residual block (FU-5).

0.50 M parameters against 2.19 M, with a strictly richer function class.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn


class MorphologyEmbed(nn.Module):
    """FU-4 — the fifth modality token: :math:`\\mathbf s_n \\in \\mathbb R^8 \\to \\mathbb R^{256}`.

    §2.2.10's M-13: the segmentation stage computes area, perimeter, the two
    axis lengths, eccentricity, solidity and the two ratios, *gates the patch on
    them*, and then throws them away. Grain size and shape are the features a
    human grader uses first, no spectral operator in the network can derive them
    from a resized 64×64 patch, and P-4 / T4-4 now persists them. 17 k
    parameters for the cheapest expected gain in the plan.
    """

    def __init__(self, n_morph: int = 8, hidden: int = 64, d: int = 256) -> None:
        super().__init__()
        self.n_morph = int(n_morph)
        self.net = nn.Sequential(
            nn.Linear(self.n_morph, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, morph: torch.Tensor) -> torch.Tensor:
        return self.net(morph)  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any


class CrossModalInteraction(nn.Module):
    """Gated low-rank bilinear fusion of ``num_modalities`` branch embeddings.

    The class name is the schema's — ``SpectralQuadNet.cross_interaction`` is a
    top-level checkpoint key. What it computes is FU-1(b), not a Perceiver.

    Args:
        num_modalities: 5 since FU-4 — branches A-D plus morphology.
        d: Branch embedding width.
        rank: :math:`r` in :math:`U_m \\in \\mathbb R^{r \\times d}`. The whole
            point of the low-rank factorisation: a full bilinear pool over five
            modalities would be :math:`10 d^2 = 655\\,\\mathrm{k}` per pair.
        drop: Dropout on the fused output.
    """

    def __init__(
        self,
        num_modalities: int = 5,
        d: int = 256,
        rank: int = 128,
        gate_hidden: int = 128,
        drop: float = 0.1,
    ):
        super().__init__()

        self.num_modalities = int(num_modalities)
        self.d = int(d)
        self.rank = int(rank)

        # FU-2: a *dataset* statistic, not a per-sample one. BatchNorm1d in
        # eval mode applies the running estimate, so a sample whose branch
        # produced almost nothing keeps a small vector instead of being
        # rescaled to unit norm.
        self.branch_norms = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(self.num_modalities)])

        # FU-2: sigmoid, and fed the pre-normalisation log-norms.
        self.modality_gate = nn.Sequential(
            nn.Linear(self.num_modalities * d + self.num_modalities, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_modalities),
            nn.Sigmoid(),
        )

        # FU-1(b): the second-order term.
        self.bilinear = nn.ModuleList(
            [nn.Linear(d, self.rank, bias=False) for _ in range(self.num_modalities)]
        )
        self.bilinear_out = nn.Linear(self.rank, d)
        self.output = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(drop)

    def gate_values(self, branches: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(gate, normalised_tokens)`` — the gate as a first-class diagnostic.

        Returned rather than hidden so ``sum_m g_m != 1`` and the gate's entropy
        are assertable properties (T3-4's validation criterion) instead of
        claims.
        """
        confidence = torch.stack(
            [torch.log(b.norm(dim=1) + 1e-6) for b in branches], dim=1
        )  # (B, M) — FU-2, before normalisation
        tokens = torch.stack(
            [norm(b) for norm, b in zip(self.branch_norms, branches, strict=True)],
            dim=1,
        )  # (B, M, d)
        gate = self.modality_gate(torch.cat([tokens.flatten(1), confidence], dim=1))
        return gate, tokens

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        gate, tokens = self.gate_values(branches)

        # First order — an independent gate per modality, so two can be on at once.
        first = (tokens * gate.unsqueeze(-1)).sum(dim=1)

        # Second order — every unordered pair, in the rank-r space.
        projected = [proj(tokens[:, m]) for m, proj in enumerate(self.bilinear)]
        second = torch.zeros_like(projected[0])
        for m, m2 in itertools.combinations(range(self.num_modalities), 2):
            second = second + projected[m] * projected[m2]

        fused = self.output(torch.cat([first, self.bilinear_out(second)], dim=1))
        return self.drop(fused)  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any


class EmbedNet(nn.Module):
    """Pre-norm MLP residual block that refines the fused token into the final embedding.

    Since FU-5 this is the *only* post-fusion block: ``CrossModalInteraction``'s
    ``output_proj`` was the same pre-LN residual MLP on the same vector and has
    been deleted (N-10).
    """

    def __init__(self, dim: int = 256, hidden: int = 512, drop: float = 0.1) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
        )

        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mlp(self.norm1(x)))
        return self.norm2(x)  # type: ignore[no-any-return]
