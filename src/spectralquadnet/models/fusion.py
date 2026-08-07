"""Cross-modal fusion of the four branch embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalInteraction(nn.Module):
    """
    Latent Cross-Modal Attention Fusion

    Modern multimodal fusion module inspired by:
    Perceiver IO, Flamingo, and Multimodal Transformers.

    Design goals:
    • preserve modality structure
    • enable cross-modal reasoning
    • scalable to more branches
    """

    def __init__(
        self,
        num_modalities: int = 4,
        d: int = 256,
        latent_tokens: int = 4,
        heads: int = 8,
        depth: int = 2,
        drop: float = 0.1,
    ):
        super().__init__()

        self.num_modalities = num_modalities
        self.d = d

        # normalize each modality independently
        self.branch_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_modalities)])

        # learned latent tokens (Perceiver-style fusion space)
        self.latents = nn.Parameter(torch.randn(latent_tokens, d) * 0.02)

        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            d, heads, dropout=drop, batch_first=True
                        ),
                        "self_attn": nn.MultiheadAttention(
                            d, heads, dropout=drop, batch_first=True
                        ),
                        "ff": nn.Sequential(
                            nn.LayerNorm(d),
                            nn.Linear(d, d * 4),
                            nn.GELU(),
                            nn.Dropout(drop),
                            nn.Linear(d * 4, d),
                        ),
                    }
                )
            )

        # modality gating (Mixture-of-Experts style)
        self.modality_gate = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, num_modalities),
            nn.Softmax(dim=-1),
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        B = branches[0].shape[0]

        # normalize branches
        # noqa B905: `branch_norms` and `branches` are always the same length by
        # construction (one LayerNorm per modality), so strict= would be inert here.
        tokens = torch.stack(
            [norm(b) for norm, b in zip(self.branch_norms, branches)],  # noqa: B905
            dim=1,
        )
        # shape: [B, M, D]

        # latent tokens
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # `nn.ModuleList` iterates as `Module`, which is not indexable to the type
        # checker; every element is the `nn.ModuleDict` built in `__init__`.
        for blk in self.blocks:
            # cross attention: latent queries modalities
            attn_out, _ = blk["cross_attn"](latents, tokens, tokens)  # type: ignore[index]
            latents = latents + attn_out

            # latent self-attention
            sa_out, _ = blk["self_attn"](latents, latents, latents)  # type: ignore[index]
            latents = latents + sa_out

            # feedforward
            latents = latents + blk["ff"](latents)  # type: ignore[index]

        # pooled fusion token
        fused = latents.mean(dim=1)

        # adaptive modality weighting
        gate = self.modality_gate(fused)

        weighted_modal = (tokens * gate.unsqueeze(-1)).sum(dim=1)

        fused = fused + weighted_modal

        return self.output_proj(fused)  # type: ignore[no-any-return]


class EmbedNet(nn.Module):
    """Pre-norm MLP residual block that refines the fused token into the final embedding."""

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
