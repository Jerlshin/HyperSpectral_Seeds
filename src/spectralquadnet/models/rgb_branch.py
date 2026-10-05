"""Trainable RGB branch: a DINOv2 ViT fine-tuned on foreground tokens of the masked crop (S32).

The S20 crops zero every non-kernel pixel, and a kernel covers 45-133 of the 256 patch
positions (median 74). The branch therefore keeps the class token plus the ``keep`` patch
tokens with the largest foreground fraction (after the positional embedding, so geometry is
preserved) and runs the transformer on those only. The readout is the class token concatenated
with the foreground-weighted mean of the final patch tokens, the S31 frozen ``cls_fg`` readout.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HUB = Path.home() / ".cache/torch/hub"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov2(name: str, drop_path_rate: float = 0.0) -> nn.Module:
    """Public DINOv2 checkpoint from the local hub cache (no download)."""
    model: nn.Module = torch.hub.load(  # type: ignore[no-untyped-call]
        str(HUB / "facebookresearch_dinov2_main"), name, source="local", pretrained=False,
        drop_path_rate=drop_path_rate)
    state = torch.load(HUB / f"checkpoints/{name}_pretrain.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def patch_foreground(images: torch.Tensor, patch: int = 14) -> torch.Tensor:
    """Foreground fraction per patch (N x tokens) from N x 3 x H x W images in [0, 1]."""
    return F.avg_pool2d((images.amax(1, keepdim=True) > 0).float(), patch).flatten(1)


def normalise(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (images - mean) / std


class RGBBranch(nn.Module):
    """DINOv2 backbone on foreground tokens -> [class, fg-mean] embedding -> linear classifier."""

    def __init__(self, backbone: nn.Module, classes: int, keep: int = 128, dropout: float = 0.0,
                 train_blocks: int = 0) -> None:
        """``train_blocks`` > 0 freezes the embeddings and all but the top ``train_blocks`` blocks
        (partial fine-tuning); the frozen stem then runs without autograd."""
        super().__init__()
        self.backbone = backbone
        self.keep = keep
        depth = len(backbone.blocks)  # type: ignore[arg-type]
        self.frozen_blocks = depth - train_blocks if 0 < train_blocks < depth else 0
        if self.frozen_blocks:
            for name, param in backbone.named_parameters():
                block = int(name.split(".")[1]) if name.startswith("blocks.") else -1
                if block < self.frozen_blocks and not name.startswith("norm."):
                    param.requires_grad_(False)
        width = int(backbone.embed_dim)  # type: ignore[arg-type]
        self.embed_dim = 2 * width
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, classes)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: N x 3 x 224 x 224 in [0, 1] with background exactly zero."""
        weights = patch_foreground(images)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_blocks):
            tokens = self.backbone.prepare_tokens_with_masks(normalise(images))  # type: ignore[operator]
            # Stable order: most-foreground first, ties by raster position; then restore raster order.
            order = torch.sort(weights, dim=1, descending=True, stable=True).indices[:, : self.keep]
            order = torch.sort(order, dim=1).values
            patches = torch.gather(tokens[:, 1:], 1, order[..., None].expand(-1, -1, tokens.shape[-1]))
            x = torch.cat([tokens[:, :1], patches], dim=1)
            for block in self.backbone.blocks[: self.frozen_blocks]:  # type: ignore[index]
                x = block(x)
        w = torch.gather(weights, 1, order)
        for block in self.backbone.blocks[self.frozen_blocks :]:  # type: ignore[index]
            x = block(x)
        x = self.backbone.norm(x)  # type: ignore[operator]
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        return torch.cat([x[:, 0], torch.einsum("nt,ntd->nd", w.to(x.dtype), x[:, 1:])], dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(images)
        return self.classifier(self.dropout(z)), z


def layer_decay_groups(model: RGBBranch, lr: float, head_lr: float, decay: float,
                       weight_decay: float) -> list[dict[str, object]]:
    """AdamW groups: classifier at ``head_lr``; block i of L at lr * decay**(L - i); embeddings lowest."""
    depth = len(model.backbone.blocks)  # type: ignore[arg-type]
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            rate = head_lr
        elif name.startswith("backbone.blocks."):
            rate = lr * decay ** (depth - int(name.split(".")[2]))
        elif name.startswith("backbone.norm"):
            rate = lr
        else:  # patch/pos embeddings, class/mask tokens
            rate = lr * decay ** (depth + 1)
        wd = 0.0 if param.ndim == 1 or name.endswith(("pos_embed", "cls_token", "mask_token")) else weight_decay
        groups.setdefault((rate, wd), []).append(param)
    return [{"params": p, "lr": r, "weight_decay": wd, "base_lr": r} for (r, wd), p in groups.items()]
