"""Branch D — SpecFormer, a spatial-spectral factorised transformer.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=======================================  ==============
Symbol                                   Baseline lines
=======================================  ==============
:class:`MultiScaleSpectralTokenizer`     1030-1056
:class:`_PreLNBlock`                     1059-1075
:class:`SpecFormerBranch`                1078-1164
=======================================  ==============

``SpecFormerBranch`` accepts ``physical_wl`` and ``patch_size`` but uses neither
(the tokenizer is stride-based and carries no wavelength encoding). Both are
preserved verbatim — dropping dead parameters is an explicit non-goal
(REFACTOR_PLAN.md §6) and would change ``SpectralQuadNet.__init__``'s call site.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleSpectralTokenizer(nn.Module):
    def __init__(self, in_channels: int, d_model: int, stride: int = 4):
        super().__init__()
        # 3 parallel tokenizers with different receptive fields to capture
        # narrow absorption lines and broad spectral shapes
        out_c = d_model // 3
        rem = d_model - (out_c * 2)

        self.proj_small = nn.Conv1d(in_channels, out_c, kernel_size=3, stride=stride, padding=1)
        self.proj_medium = nn.Conv1d(in_channels, out_c, kernel_size=5, stride=stride, padding=2)
        self.proj_large = nn.Conv1d(in_channels, rem, kernel_size=7, stride=stride, padding=3)

        self.norm = nn.GroupNorm(1, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Channels, Length]
        t_s = self.proj_small(x)
        t_m = self.proj_medium(x)
        t_l = self.proj_large(x)

        # Ensure identical sequence lengths by truncating to the shortest
        min_len = min(t_s.size(2), t_m.size(2), t_l.size(2))
        t_s, t_m, t_l = t_s[..., :min_len], t_m[..., :min_len], t_l[..., :min_len]

        tokens = torch.cat([t_s, t_m, t_l], dim=1)
        return self.act(self.norm(tokens))  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any


class _PreLNBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, drop: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop), nn.Linear(d_ff, d), nn.Dropout(drop)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.ln1(x)
        h, _ = self.attn(lx, lx, lx, need_weights=False)
        x = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any


class SpecFormerBranch(nn.Module):
    """
    State-of-the-Art Spatial-Spectral Factorised Transformer.
    Processes Multi-Scale Spectral features first, then correlates Spatial grids.
    """

    def __init__(
        self,
        physical_wl: torch.Tensor,
        num_bands: int = 40,
        patch_size: int = 8,  # Kept for API compatibility, handled by MultiScale
        stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        out_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.tokenizer = MultiScaleSpectralTokenizer(in_channels=1, d_model=d_model, stride=stride)

        # Spectral cls token and positional embedding
        self.spec_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spec_cls, std=0.02)

        # Estimate number of tokens (approx num_bands // stride)
        n_tokens = (num_bands // stride) + 2
        self.spec_pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # Factorized Transformer Stages
        # 1. Spectral Attention (Local chemical composition)
        self.spectral_blocks = nn.ModuleList(
            [_PreLNBlock(d_model, n_heads, d_model * 2, dropout) for _ in range(n_layers // 2)]
        )

        # 2. Spatial Attention (Global seed morphology)
        self.spatial_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spatial_cls, std=0.02)

        self.spatial_blocks = nn.ModuleList(
            [_PreLNBlock(d_model, n_heads, d_model * 2, dropout) for _ in range(n_layers // 2)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, out_dim), nn.BatchNorm1d(out_dim), nn.GELU())

    def forward(self, grid_ms: torch.Tensor) -> torch.Tensor:
        # Expected input: [B, N, C] where N is number of spatial grids (e.g., 16)
        B, N, C = grid_ms.shape

        x_combo = grid_ms.unsqueeze(2).reshape(B * N, 1, C)

        # 2. Multi-Scale Tokenization
        tokens = self.tokenizer(x_combo).transpose(1, 2)  # [B*N, Seq, d_model]

        # 3. Spectral Stage (Within each grid cell)
        cls_tokens = self.spec_cls.expand(B * N, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Add Positional Embedding safely
        seq_len = tokens.size(1)
        if seq_len <= self.spec_pos_embed.size(1):
            tokens = tokens + self.spec_pos_embed[:, :seq_len, :]

        for blk in self.spectral_blocks:
            tokens = blk(tokens)

        # Extract the spectral CLS token to represent the entire grid cell's spectrum
        grid_features = tokens[:, 0, :]  # [B*N, d_model]

        # 4. Spatial Stage (Across grid cells)
        # Reshape back to separate batch and spatial dimensions
        spatial_tokens = grid_features.view(B, N, -1)  # [B, N, d_model]

        spatial_cls = self.spatial_cls.expand(B, -1, -1)
        spatial_tokens = torch.cat([spatial_cls, spatial_tokens], dim=1)

        for blk in self.spatial_blocks:
            spatial_tokens = blk(spatial_tokens)

        # Final classification token representing the whole Spatio-Spectral object
        global_feature = self.norm(spatial_tokens[:, 0, :])

        return self.proj(global_feature)  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
