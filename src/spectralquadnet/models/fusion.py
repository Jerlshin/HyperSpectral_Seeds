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

    A5 / CHANGES §5.3 adds :attr:`mode`, because the bilinear term is 0.50 M
    parameters — 9.6% of the model — spent on second-order interactions across
    ten modality pairs, fitted from 6,036 training samples, in a fusion where
    three of the five modalities carried ≤6% influence each. That is a large
    hypothesis class in service of combining one strong signal and one weak one.
    The audit's verdict is "replace with concat + MLP"; the three modes here are
    what turns that verdict into a measurement.

    Args:
        num_modalities: 5 since FU-4 — branches A-D plus morphology. An A3 arm
            passes fewer.
        d: Branch embedding width.
        rank: :math:`r` in :math:`U_m \\in \\mathbb R^{r \\times d}`. The whole
            point of the low-rank factorisation: a full bilinear pool over five
            modalities would be :math:`10 d^2 = 655\\,\\mathrm{k}` per pair.
        gate_hidden: Hidden width of the gate MLP.
        drop: Dropout on the fused output.
        mode: One of
            :data:`~spectralquadnet.models.spectral_quadnet.FUSION_MODES`.
            ``bilinear_gate`` is the audited default; ``gate`` keeps the
            sigmoid gate and drops the second-order term; ``concat_mlp``
            discards both and concatenates the normalised tokens into a
            2-layer MLP.
    """

    def __init__(
        self,
        num_modalities: int = 5,
        d: int = 256,
        rank: int = 128,
        gate_hidden: int = 128,
        drop: float = 0.1,
        mode: str = "bilinear_gate",
    ):
        super().__init__()

        self.num_modalities = int(num_modalities)
        self.d = int(d)
        self.rank = int(rank)
        self.mode = str(mode)

        # FU-2: a *dataset* statistic, not a per-sample one. BatchNorm1d in
        # eval mode applies the running estimate, so a sample whose branch
        # produced almost nothing keeps a small vector instead of being
        # rescaled to unit norm.
        self.branch_norms = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(self.num_modalities)])

        # FU-2: sigmoid, and fed the pre-normalisation log-norms.
        # Absent in `concat_mlp`, which is the arm that asks whether a gate is
        # needed at all — constructing an unused one would put 0.17 M dead
        # parameters into that arm's reported budget.
        self.modality_gate = (
            nn.Sequential(
                nn.Linear(self.num_modalities * d + self.num_modalities, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.num_modalities),
                nn.Sigmoid(),
            )
            if self.mode in ("bilinear_gate", "gate")
            else None
        )

        # FU-1(b): the second-order term. Only in `bilinear_gate`.
        wants_bilinear = self.mode == "bilinear_gate"
        self.bilinear = (
            nn.ModuleList([nn.Linear(d, self.rank, bias=False) for _ in range(self.num_modalities)])
            if wants_bilinear
            else None
        )
        self.bilinear_out = nn.Linear(self.rank, d) if wants_bilinear else None

        if self.mode == "bilinear_gate":
            self.output: nn.Module = nn.Linear(2 * d, d)
        elif self.mode == "gate":
            self.output = nn.Linear(d, d)
        else:  # concat_mlp
            self.output = nn.Sequential(
                nn.Linear(self.num_modalities * d, d),
                nn.LayerNorm(d),
                nn.GELU(),
                nn.Linear(d, d),
            )
        self.drop = nn.Dropout(drop)

    def normalised_tokens(self, branches: list[torch.Tensor]) -> torch.Tensor:
        """``(B, M, d)`` — each modality through its own ``BatchNorm1d`` (FU-2)."""
        return torch.stack(
            [norm(b) for norm, b in zip(self.branch_norms, branches, strict=True)],
            dim=1,
        )

    def gate_values(self, branches: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(gate, normalised_tokens)`` — the gate as a first-class diagnostic.

        Returned rather than hidden so ``sum_m g_m != 1`` and the gate's entropy
        are assertable properties (T3-4's validation criterion) instead of
        claims.

        Raises:
            RuntimeError: ``mode="concat_mlp"``, which has no gate to report.
        """
        if self.modality_gate is None:
            raise RuntimeError(
                f"fusion_mode={self.mode!r} has no modality gate; "
                "gate diagnostics are only defined for 'bilinear_gate' and 'gate'."
            )
        confidence = torch.stack(
            [torch.log(b.norm(dim=1) + 1e-6) for b in branches], dim=1
        )  # (B, M) — FU-2, before normalisation
        tokens = self.normalised_tokens(branches)
        gate = self.modality_gate(torch.cat([tokens.flatten(1), confidence], dim=1))
        return gate, tokens

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        if self.mode == "concat_mlp":
            tokens = self.normalised_tokens(branches)
            return self.drop(self.output(tokens.flatten(1)))  # type: ignore[no-any-return]

        gate, tokens = self.gate_values(branches)

        # First order — an independent gate per modality, so two can be on at once.
        first = (tokens * gate.unsqueeze(-1)).sum(dim=1)

        if self.bilinear is None or self.bilinear_out is None:
            return self.drop(self.output(first))  # type: ignore[no-any-return]

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
