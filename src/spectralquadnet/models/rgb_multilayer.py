"""Multi-layer foreground-token RGB branch (S36 architecture component).

S31 (F114/F115) found that, in a frozen DINOv2, blocks below the last carry most of the
within-acquisition RGB signal the class token leaves unused. This branch reads out
[class, foreground-weighted mean] after each of the last ``layers`` blocks (each through the
backbone's final norm, as in DINOv2's linear evaluation), projects every layer readout to a shared
width with its own LayerNorm-linear map, and averages the projections. The averaging keeps the
classifier size independent of ``layers``, and each layer's contribution is a learned
projection rather than a fixed concatenation.

Token handling is S32's (``rgb_branch.RGBBranch``): only the class token and the ``keep`` most
foreground patch tokens enter the transformer. Optional ``morph`` input carries the metric
morphometrics (F117) through a small separate map, so absolute size is not inferred from a
scale-normalised crop.
"""

from __future__ import annotations

import torch
from torch import nn

from spectralquadnet.models.rgb_branch import normalise, patch_foreground


class MultiLayerRGBBranch(nn.Module):
    def __init__(self, backbone: nn.Module, classes: int, keep: int = 128, layers: int = 4,
                 width: int = 768, morph_dim: int = 0, dropout: float = 0.0, train_blocks: int = 0) -> None:
        super().__init__()
        self.backbone = backbone
        self.keep = keep
        depth = len(backbone.blocks)  # type: ignore[arg-type]
        if not 1 <= layers <= depth:
            raise ValueError("layers must be within the backbone depth")
        self.layers = layers
        self.frozen_blocks = depth - train_blocks if 0 < train_blocks < depth else 0
        if self.frozen_blocks > depth - layers:
            raise ValueError("Every read-out layer must be trainable or all blocks frozen together")
        if self.frozen_blocks:
            for name, param in backbone.named_parameters():
                block = int(name.split(".")[1]) if name.startswith("blocks.") else -1
                if block < self.frozen_blocks and not name.startswith("norm."):
                    param.requires_grad_(False)
        dim = int(backbone.embed_dim)  # type: ignore[arg-type]
        self.project = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, width))
                                     for _ in range(layers))
        self.morph = nn.Sequential(nn.LayerNorm(morph_dim), nn.Linear(morph_dim, width)) if morph_dim else None
        self.embed_dim = width
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, classes)

    def embed(self, images: torch.Tensor, morph: torch.Tensor | None = None) -> torch.Tensor:
        weights = patch_foreground(images)
        depth = len(self.backbone.blocks)  # type: ignore[arg-type]
        first_read = depth - self.layers
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_blocks):
            tokens = self.backbone.prepare_tokens_with_masks(normalise(images))  # type: ignore[operator]
            order = torch.sort(weights, dim=1, descending=True, stable=True).indices[:, : self.keep]
            order = torch.sort(order, dim=1).values
            patches = torch.gather(tokens[:, 1:], 1, order[..., None].expand(-1, -1, tokens.shape[-1]))
            x = torch.cat([tokens[:, :1], patches], dim=1)
            for block in self.backbone.blocks[: self.frozen_blocks]:  # type: ignore[index]
                x = block(x)
        w = torch.gather(weights, 1, order)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        readouts = []
        for i, block in enumerate(self.backbone.blocks[self.frozen_blocks :], start=self.frozen_blocks):  # type: ignore[index]
            x = block(x)
            if i >= first_read:
                z = self.backbone.norm(x)  # type: ignore[operator]
                pooled = torch.cat([z[:, 0], torch.einsum("nt,ntd->nd", w.to(z.dtype), z[:, 1:])], dim=1)
                readouts.append(self.project[i - first_read](pooled))
        out = torch.stack(readouts).mean(0)
        if self.morph is not None:
            if morph is None:
                raise ValueError("This branch was built with morphometrics; pass them")
            out = out + self.morph(morph)
        return out

    def forward(self, images: torch.Tensor, morph: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(images, morph)
        return self.classifier(self.dropout(z)), z
