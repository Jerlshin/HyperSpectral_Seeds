"""Branch A — the continuum and derivative profile (BR-2 / T3-6, FE-1 / T3-5).

C-2's sharpest form is §2.2.2: **Branch A was a strict functional subset of
Branch D.** Both received ``extract_grid_spectra(x, 4)`` — the same
``(B, 16, C)`` tensor, byte for byte — and both reduced it to a 256-vector. Two
branches, 2.77 M parameters, one input. The "four disjoint views of the same
patch" claim was false for half of them.

BR-2 separates them without adding a parameter to the towers:

1. **A different input.** Branch A consumes the *shape* of each cell's spectrum:
   SNV (which removes the per-pixel gain and offset, so the branch cannot see
   brightness at all) followed by the first and second wavelength derivatives of
   FE-1(b). Branch D keeps the raw grid spectra. Derivatives and their
   antiderivative are related but not mutually reconstructible in a finite
   receptive field, and they emphasise opposite ends of the spectral frequency
   band.
2. **A finer grid.** :math:`4\\times4` over a 64×64 patch is a 256:1 spatial
   compression (§2.2.3); Branch A moves to :math:`8\\times8`, a 64:1 one. The
   branch processes cells independently, so the parameter count is unchanged and
   only the flattened batch grows. Branch D keeps :math:`4\\times4` — they are
   different branches with different inputs now, so there is no reason for them
   to share a grid either.
3. **Mass-weighted pooling** — done by the caller, which has :math:`\\omega_n`;
   see ``SpectralQuadNet.forward``.

The stem is FE-1(a)'s :class:`~spectralquadnet.models.front_end.LambdaConv1d`:
its kernel is generated from wavelength *offsets*, over the ``k`` nearest bands
in :math:`\\lambda`, so the first operation applied to a spectrum is no longer a
finite difference on an irregular grid (C-5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectralquadnet.models.blocks.conv_blocks import LargeKernelBlock1D
from spectralquadnet.models.front_end import LambdaConv1d, SpectralDerivatives, snv


class SpectralProfileBranch(nn.Module):
    """Processes the per-cell spectral *shape* through parallel multi-scale 1-D towers.

    Three ``LargeKernelBlock1D`` towers with kernel sizes 3/5/7 read the same
    stem output at different receptive fields, are concatenated, fused back
    down to ``tower_ch``, and attention-pooled across the band axis.

    This is the model's **largest activation consumer**, and BR-2's grid change
    is why. The branch processes cells independently, so ``grid_size_a = 8``
    turns a batch of 32 into ``32 × 64 = 2048`` sequences; each
    ``LargeKernelBlock1D`` then expands those to ``tower_ch × 4`` channels and
    the autograd graph retains that widened tensor *twice* — once for GELU's
    backward and once for ``pw2``'s. Six blocks over three towers is 2.9 GB of
    the 4.05 GB a batch-32 step holds, which is what puts ``stage1.batch = 128``
    at ≈ 16 GB and inside a hair of a 16 GB machine's whole address space.
    :attr:`grad_checkpoint` is the lever for it.
    """

    #: Recompute the three towers during the backward pass instead of keeping
    #: their activations alive — a batch-32 step goes from 4054 MB to 1901 MB,
    #: **2.13× less**, for 4.8% more time.
    #:
    #: Bit-exact, not approximately so: the towers are ``GroupNorm``/``GELU``/SE
    #: only, with no dropout, no BatchNorm and no other running state, so the
    #: recomputed forward is the identical function of identical inputs and the
    #: gradients that come out are the ones the stored activations would have
    #: produced. Nothing here reads the RNG, so the global stream is untouched
    #: too and the golden Stage-1 weight hashes hold either way.
    #:
    #: Set once from the runtime plan by
    #: :func:`~spectralquadnet.utils.device.apply_runtime_optimisations`.
    grad_checkpoint: bool = False

    def __init__(
        self,
        wavelengths: torch.Tensor,
        out_dim: int = 256,
        tower_ch: int = 96,
        wl_pe_module: nn.Module | None = None,
        n_freq: int = 16,
        kernel_neighbours: int = 5,
    ) -> None:
        super().__init__()
        self.wl_pe_module = wl_pe_module

        # FE-1(b): [SNV(r) || d/dλ || d²/dλ²], computed on the true λ metric.
        self.derivatives = SpectralDerivatives(wavelengths)

        # FE-1(a): a continuous-λ kernel over the 5 nearest bands in wavelength.
        self.stem = nn.Sequential(
            LambdaConv1d(
                wavelengths,
                in_channels=3,
                out_channels=tower_ch,
                k=kernel_neighbours,
                n_freq=n_freq,
            ),
            nn.GroupNorm(1, tower_ch),
            nn.GELU(),
        )

        # Three parallel receptive-field scales over the band axis
        self.tower_s = nn.Sequential(
            LargeKernelBlock1D(tower_ch, 3), LargeKernelBlock1D(tower_ch, 3)
        )
        self.tower_m = nn.Sequential(
            LargeKernelBlock1D(tower_ch, 5), LargeKernelBlock1D(tower_ch, 5)
        )
        self.tower_l = nn.Sequential(
            LargeKernelBlock1D(tower_ch, 7), LargeKernelBlock1D(tower_ch, 7)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(tower_ch * 3, tower_ch, 1, bias=False),
            nn.GroupNorm(1, tower_ch),
            nn.GELU(),
            LargeKernelBlock1D(tower_ch, 3),
        )

        self.attn_pool = nn.Sequential(
            nn.Conv1d(tower_ch, tower_ch // 4, 1),
            nn.GELU(),
            nn.Conv1d(tower_ch // 4, 1, 1),
        )

        self.proj = nn.Sequential(
            nn.Linear(tower_ch, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def shape_channels(self, ms: torch.Tensor) -> torch.Tensor:
        """``(B, C) -> (B, 3, C)`` — the branch's actual input, ``[SNV, ∂λ, ∂²λ]``.

        Exposed so ``tests/unit/test_branch_inputs.py`` can hash it against
        Branch D's and assert C-2 is closed, which is a claim about the *input*
        and cannot be made from the embedding.
        """
        channels: torch.Tensor = self.derivatives(snv(ms))
        return channels

    def forward(self, ms: torch.Tensor) -> torch.Tensor:
        """``(B, C)`` per-cell spectra → ``(B, out_dim)``."""
        return self.forward_channels(self.shape_channels(ms))

    def forward_channels(self, channels: torch.Tensor) -> torch.Tensor:
        """The branch from ``(B, 3, C)`` onwards, for a caller that already has the channels.

        ``SpectralQuadNet.forward`` builds every branch's input once, through
        :meth:`~spectralquadnet.models.spectral_quadnet.SpectralQuadNet.branch_inputs`,
        so the tensor the distinctness test hashes is the same object the branch
        consumes rather than a reconstruction of it.
        """
        x = self.stem(channels)

        if self.wl_pe_module is not None:
            x = self.wl_pe_module(x)

        towers = (self.tower_s, self.tower_m, self.tower_l)
        if self.grad_checkpoint and torch.is_grad_enabled():
            # `use_reentrant=False` is not optional: the reentrant implementation
            # needs an input that requires grad (`x` does, but only by accident of
            # the stem having parameters) and it breaks DDP's one-mark-per-variable
            # rule. Guarded on `is_grad_enabled` so an inference pass -- the EMA
            # shadow's, every validation epoch's -- takes the plain path instead of
            # paying for a recompute that will never happen.
            scales = [checkpoint(t, x, use_reentrant=False) for t in towers]
        else:
            scales = [t(x) for t in towers]

        x_fused = self.fusion(torch.cat(scales, dim=1))

        w = torch.softmax(self.attn_pool(x_fused), dim=2)
        return self.proj(torch.sum(x_fused * w, dim=2))  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
