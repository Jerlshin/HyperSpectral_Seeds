"""Branch D — a λ-aware spatial-spectral factorised transformer (BR-4 / T3-3).

The pre-Tier-3 branch accepted ``physical_wl`` and ``patch_size`` and used
neither. It tokenised by an index stride of 4 and added a **learned positional
embedding indexed by token position**, which makes the branch un-transferable in
the precise sense that matters for F-3: re-run band selection at a different
:math:`k` and "token 3" refers to a different region of the spectrum, so the
table has to be relearned from scratch. Worse, every 1-D convolution inside the
tokenizer was a finite difference on an irregular grid (C-5).

Three changes, all of which make :math:`\\lambda` a first-class axis:

**(i) λ-uniform tokenisation.** :math:`[\\lambda_{\\min}, \\lambda_{\\max}]` is cut
into ``n_tokens`` equal-width windows and the bands falling in each are averaged
(:class:`LambdaWindowPooling`). Token :math:`t` then always means the same
spectral region regardless of which bands the selector kept — and, as a
by-product, the tokenizer's convolutions run on a **uniform** axis, where index
distance and wavelength distance finally coincide.

**(ii) λ-derived token embeddings.** The learned ``spec_pos_embed`` table is
replaced by :func:`~spectralquadnet.models.blocks.positional.sinusoidal_wavelength_encoding`
evaluated at each window's centre wavelength :math:`\\bar\\lambda_t`. Zero
parameters, and shared with the band-axis encoding the CNN branches use. The
``spec_cls`` token keeps its own learned code, since it has no wavelength.

**(iii) Relative-λ attention bias.** Every spectral-stage attention logit gains
:math:`b_\\psi(\\bar\\lambda_t - \\bar\\lambda_u)`, a per-head scalar from a tiny
MLP over the Fourier features of the wavelength difference
(:class:`RelativeLambdaBias`, ~1.3 k parameters). A head can now specialise on
"bands 60 nm apart" — the natural unit for pairing an absorption feature with
its shoulder — instead of on "tokens 2 apart in an arbitrary index".

With the wavelength axis carried explicitly the branch needs less brute
capacity, so ``specf_dim`` drops 256 → 192 and the branch with it: 2.18 M →
≈ 1.24 M. ``model.specf_drop`` (dead since the reference implementation, N-1b)
is wired to the transformer dropout, and ``model.specf_patch`` now sets the
token count rather than a stride nothing consumed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectralquadnet.models.blocks.positional import sinusoidal_wavelength_encoding
from spectralquadnet.models.front_end import FourierFeatures


class LambdaWindowPooling(nn.Module):
    """BR-4(iii) — average the bands falling in each of ``n_tokens`` λ-uniform windows.

    The pooling matrix is a buffer built once from the wavelength vector: row
    ``t`` is uniform over the bands inside window ``t``. A window that no
    selected band falls into takes the single nearest band instead, so the token
    grid is always full and the branch cannot silently emit a zero token for a
    region the selector happened to skip.

    ``lam_range`` is the **domain the windows partition**, and it is
    deliberately not the observed ``(min, max)`` of ``wavelengths``. That
    distinction is the whole of F-3: a 20-band subset of a 40-band selection has
    a different observed maximum, so windows cut to the observed range would put
    token 3 at a different wavelength in the two — the same un-transferability
    the learned positional table had, arrived at by a different route. The
    branch passes the fixed ``(0, 1)`` that
    ``DataStore.load_wavelengths``'s min-max normalisation defines the axis on.

    Non-persistent: it is a deterministic function of the wavelength vector, and
    ``token_wl`` below is the part of it a reader of a checkpoint would want.
    """

    pool: torch.Tensor
    token_wl: torch.Tensor

    def __init__(
        self,
        wavelengths: torch.Tensor,
        n_tokens: int = 10,
        lam_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        lam = wavelengths.detach().flatten().float()
        n_bands = int(lam.numel())
        self.n_tokens = max(1, min(int(n_tokens), n_bands))

        lo, hi = lam_range if lam_range is not None else (float(lam.min()), float(lam.max()))
        edges = torch.linspace(lo, hi, self.n_tokens + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])

        pool = torch.zeros(self.n_tokens, n_bands)
        for t in range(self.n_tokens):
            left, right = edges[t], edges[t + 1]
            below = lam <= right if t == self.n_tokens - 1 else lam < right
            inside = (lam >= left) & below
            if not bool(inside.any()):
                inside = torch.zeros_like(inside)
                inside[int(torch.argmin((lam - centres[t]).abs()))] = True
            pool[t] = inside.float() / float(inside.sum())

        self.register_buffer("pool", pool, persistent=False)
        self.register_buffer("token_wl", centres, persistent=False)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """``(B, C) -> (B, n_tokens)``."""
        return torch.nn.functional.linear(spectra, self.pool)


class RelativeLambdaBias(nn.Module):
    """BR-4(ii) — a per-head additive attention bias :math:`b_\\psi(\\bar\\lambda_t - \\bar\\lambda_u)`.

    The bias depends only on the token grid, so it is computed once per forward
    and broadcast over the batch. It is handed to ``nn.MultiheadAttention`` as a
    float ``attn_mask``, which is exactly the additive-logit contract that
    argument implements — no re-implementation of attention is needed to get it.
    """

    offsets: torch.Tensor

    def __init__(self, token_wl: torch.Tensor, n_heads: int, n_freq: int = 16, hidden: int = 32):
        super().__init__()
        self.n_heads = int(n_heads)
        offsets = token_wl[:, None] - token_wl[None, :]
        self.register_buffer("offsets", offsets, persistent=False)

        span = float(offsets.abs().max().clamp(min=1e-6))
        self.features = FourierFeatures(n_freq=n_freq, span=span)
        self.mlp = nn.Sequential(
            nn.Linear(self.features.out_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_heads),
        )

    def forward(self, n_prefix: int = 1) -> torch.Tensor:
        """``(n_heads, L, L)`` with ``n_prefix`` leading CLS rows/columns left unbiased."""
        bias = self.mlp(self.features(self.offsets)).permute(2, 0, 1)
        if n_prefix <= 0:
            return bias  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
        return torch.nn.functional.pad(bias, (n_prefix, 0, n_prefix, 0), value=0.0)


class MultiScaleSpectralTokenizer(nn.Module):
    """Three parallel kernel widths (3/5/7) over the λ-uniform token axis.

    Narrow kernels resolve sharp absorption lines while wide kernels capture
    broad spectral shape; the three token streams are concatenated channel-wise
    into a single ``d_model``-wide sequence. Since BR-4(iii) the axis these run
    over is uniform in wavelength, so a kernel width is a *bandwidth* — which is
    what the three widths were always meant to mean.
    """

    def __init__(self, in_channels: int, d_model: int, stride: int = 1):
        super().__init__()
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


def expand_attn_bias(bias: torch.Tensor, n_batch: int) -> torch.Tensor:
    """``(H, L, L) -> (n_batch * H, L, L)``, the shape ``MultiheadAttention`` demands.

    The bias depends only on the token grid, so every batch element gets the
    same ``(H, L, L)`` block; ``expand`` is a view but ``reshape`` across the
    expanded stride has to copy, which is why this is worth doing once per
    branch forward rather than once per encoder block.
    """
    n_heads, seq, _ = bias.shape
    return bias.unsqueeze(0).expand(n_batch, -1, -1, -1).reshape(n_batch * n_heads, seq, seq)


class _PreLNBlock(nn.Module):
    """Standard pre-norm Transformer encoder block (MHSA + GELU feed-forward)."""

    def __init__(self, d: int, heads: int, d_ff: int, drop: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop), nn.Linear(d_ff, d), nn.Dropout(drop)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """``attn_bias`` is either the per-head ``(H, L, L)`` bias or the batched form.

        ``nn.MultiheadAttention`` takes a float ``attn_mask`` as an additive
        term on the logits, shaped ``(N * n_heads, L, S)`` — it does **not**
        broadcast a per-head bias over the batch, so the bias has to be
        materialised at that shape. Doing it here meant materialising it once
        per block over an identical input; :meth:`SpecFormerBranch.forward` now
        expands it once for the whole stack and passes the batched form
        straight through. A 3-D argument is taken as already batched, so the
        ``(H, L, L)`` form is still accepted for a caller holding one.
        """
        lx = self.ln1(x)
        mask = attn_bias
        if mask is not None and mask.shape[0] != x.shape[0] * self.attn.num_heads:
            mask = expand_attn_bias(mask, x.shape[0])
        h, _ = self.attn(lx, lx, lx, need_weights=False, attn_mask=mask)
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
        patch_size: int = 8,
        stride: int = 4,
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 4,
        out_dim: int = 256,
        dropout: float = 0.15,
        n_freq: int = 16,
    ) -> None:
        super().__init__()
        # BR-4(iii). `patch_size` finally means something: the token count is the
        # band count over the half-patch the old code used as an index stride, so
        # the shipped `specf_patch = 8` reproduces the 10 tokens the stride-4
        # tokenizer produced — on a λ-uniform grid instead of an index one.
        n_tokens = max(1, num_bands // max(1, patch_size // 2))
        # `(0, 1)` and not the observed range — see `LambdaWindowPooling`. This
        # is the line that makes the branch transferable across band counts,
        # which is T3-3's validation criterion and F-3's question. It also makes
        # the min-max normalisation a hard precondition rather than a
        # convention: raw nanometres would fall outside every window, each
        # window would take its nearest band, and the branch would quietly see
        # ten copies of band 0.
        lo, hi = float(physical_wl.min()), float(physical_wl.max())
        if lo < -1e-4 or hi > 1.0 + 1e-4:
            raise ValueError(
                f"physical_wl must be min-max normalised to [0, 1]; got [{lo:.4g}, {hi:.4g}]. "
                "`DataStore.load_wavelengths` does this normalisation; a raw-nanometre vector "
                "would put every band outside every λ-window."
            )
        self.windows = LambdaWindowPooling(physical_wl, n_tokens=n_tokens, lam_range=(0.0, 1.0))
        self.tokenizer = MultiScaleSpectralTokenizer(in_channels=1, d_model=d_model, stride=1)

        # Spectral cls token; the remaining token codes come from wavelength.
        self.spec_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.spec_cls, std=0.02)
        self.register_buffer(
            "spec_pos_embed",
            sinusoidal_wavelength_encoding(self.windows.token_wl, d_model).unsqueeze(0),
        )

        self.lambda_bias = RelativeLambdaBias(self.windows.token_wl, n_heads, n_freq=n_freq)

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

        # 1. λ-uniform windows — after this the axis is uniform in wavelength.
        pooled = self.windows(grid_ms.reshape(B * N, C)).unsqueeze(1)

        # 2. Multi-Scale Tokenization
        tokens = self.tokenizer(pooled).transpose(1, 2)  # [B*N, Seq, d_model]

        # 3. Spectral Stage (Within each grid cell)
        seq_len = tokens.size(1)
        # `register_buffer` widens to `Tensor | Module` for the type checker;
        # the same note as on `AdaptiveSubcenterArcFaceHead.margins`.
        tokens = tokens + self.spec_pos_embed[:, :seq_len, :]  # type: ignore[index]
        cls_tokens = self.spec_cls.expand(B * N, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Generated once per forward from the token grid, then expanded once
        # for the whole spectral stack rather than inside each block.
        bias = self.lambda_bias(n_prefix=1)[:, : seq_len + 1, : seq_len + 1]
        batched_bias = expand_attn_bias(bias.contiguous(), tokens.shape[0])
        for blk in self.spectral_blocks:
            tokens = blk(tokens, batched_bias)

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
