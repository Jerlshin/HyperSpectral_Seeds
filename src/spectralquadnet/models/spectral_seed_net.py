"""``SpectralSeedNet`` — the two-pathway replacement (IC-10 / CHANGES §16.2).

Why two pathways and not four
─────────────────────────────
The audited evidence brackets exactly two information sources on this dataset,
and the numbers are the most useful pair in the whole project:

* **LDA on 40-band mean spectra scores 0.5916** accuracy
  (``dataset/band_selection_report.csv``, patch-level 5-fold, so also leaky).
  ~59 points are available from the global spectrum alone.
* **The full 5.19 M model reaches ~84.5%** under the same leaky protocol. ~25
  points are attributable to everything beyond the mean spectrum, and the
  fusion gate assigns 87% of that to Branch C.

So the architecture keeps a joint spectral–spatial operator and the global mean
spectrum, and drops the components that were competing for the same gradient:

===============================  =====================================================
Removed                          Why
===============================  =====================================================
Branch D (SpecFormer)            1.24 M parameters — 23.9% of the model — to run
                                 attention over **10 tokens**, at 3.1% fused
                                 influence. Highest parameter-per-unit-influence in
                                 the network (CHANGES §5.1).
Branch A as a *branch*           60% of the forward FLOPs for 5.6% influence, because
                                 it replicates a six-block tower stack over 64 grid
                                 cells. Its **physics is kept**: SNV and the exact
                                 Savitzky–Golay λ-derivatives survive as fixed,
                                 zero-parameter features into the spectral path.
Rank-128 bilinear fusion         0.50 M parameters for second-order interactions over
                                 ten modality pairs, fitted from 6,036 samples, when
                                 three of five modalities carried ≤6% each (§5.3).
Branch dropout                   Only two pathways remain, and the asymmetric policy
                                 biased the gate rather than regularising it (§5.2).
Three of four auxiliary heads    Four heads under a saturating controller made the
                                 auxiliary term ≈7.8× the main loss at epoch 20 (§7.1).
Sub-centres K=3 + KL balance     Seeded sub-centres were collinear at cos 0.987 —
                                 spherical k-means could not find three modes (§5.4).
Doubled morphometrics            They entered the model twice, in Branch B *and* as a
                                 fifth fusion token. Correctness, not tuning.
===============================  =====================================================

What is retained, and why
─────────────────────────
Branch C verbatim (the only evidence-supported discriminative component), the
index bank and continuum depths (1.8% of parameters, 0.005% of compute, and the
signal the entire NIR-chemometrics literature is built on — kept on theoretical
rather than evidential grounds, and flagged as such), ``MaskedSpectralECA`` (six
parameters), the exact-zero background invariant, mixup, EMA, D₄, and the
cosine/ArcFace head with its elaborations stripped.

≈2.8 M parameters and ≈1.4 GFLOP/sample: 54% of the parameters and 34% of the
compute of ``SpectralQuadNet``.

**``SpectralQuadNet`` is not deleted.** A3 and A8 need it as their control arm;
removing the thing an ablation is supposed to falsify would make the removals
above unfalsifiable, which is the defect this whole revision exists to correct.

Schema
──────
``SCHEMA_VERSION = 4`` and ``ARCH = "spectral_seed_net"``. Both are recorded in
every bundle, and :func:`~spectralquadnet.engine.checkpoint.load_ckpt` refuses a
cross-architecture load rather than partially matching tensors by name — a
``branch_c.*`` weight from the four-branch model *would* load into this one's
``spatial.*`` if anything renamed them, and the resulting number would be about
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralquadnet.models.blocks.attention import MaskedSpectralECA
from spectralquadnet.models.branches.spatial_cnn import SpatialCNNBranch
from spectralquadnet.models.branches.spectral_stats import ContinuumDepths, SoftIndexBank
from spectralquadnet.models.control import set_dropout as set_module_dropout
from spectralquadnet.models.front_end import SpectralDerivatives, snv
from spectralquadnet.models.fusion import EmbedNet
from spectralquadnet.models.heads import AdaptiveSubcenterArcFaceHead, AuxiliaryHead
from spectralquadnet.models.stats_ops import foreground_mask, masked_mean_spectrum

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: Width every pathway is projected to before the concatenation.
EMBED_DIM: int = 256


class SpectralPath(nn.Module):
    """Global chemometrics over the foreground mean spectrum.

    .. code-block:: text

        x̄ (B, C)
          ├─ SoftIndexBank(64)        learned normalised-difference indices
          ├─ ContinuumDepths(16)      hull-removed absorption depths
          ├─ snv(x̄)         (C)      ┐ Branch A's physics, as fixed features:
          ├─ D₁ snv(x̄)      (C)      │ zero parameters, exact on the irregular
          ├─ D₂ snv(x̄)      (C)      ┘ λ grid (Savitzky–Golay)
          └─ morph          (8)       physical size and shape
                → LayerNorm → MLP → (B, 256)

    The three SNV/derivative blocks are what makes removing Branch A a
    *relocation* rather than a loss. :class:`~spectralquadnet.models.front_end.SpectralDerivatives`
    holds two dense ``(C, C)`` operator buffers and no parameters, so the
    branch's entire physical content survives at a cost of one matmul against
    the 2.44 GFLOP the 64-cell tower replication used to spend.

    Morphometrics enter **here and nowhere else**. In ``SpectralQuadNet`` they
    were concatenated into Branch B *and* embedded as a fifth fusion token,
    which is double-counting rather than a design (CHANGES §5.1).
    """

    def __init__(
        self,
        num_bands: int,
        wavelengths: torch.Tensor,
        out_dim: int = EMBED_DIM,
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
        self.derivatives = SpectralDerivatives(wavelengths)

        # 3 * num_bands is [snv, D1, D2]; the derivative module stacks them.
        in_dim = self.index_bank.n_indices + self.continuum.n_depths + 3 * int(num_bands)
        in_dim += self.n_morph
        self.in_dim = in_dim

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

    def features(self, mean_spectrum: torch.Tensor, morph: torch.Tensor | None) -> torch.Tensor:
        """The raw ``(B, in_dim)`` descriptor, before the MLP.

        Exposed for the same reason ``SpectralStatsBranch.features`` is: claims
        about what this pathway *sees* — rank, distinctness from the spatial
        path — are properties of this tensor, not of the embedding.
        """
        r = mean_spectrum
        shape = self.derivatives(snv(r))  # (B, 3, C)
        parts = [
            self.index_bank(r),
            self.continuum(r),
            shape.flatten(1),
        ]
        if morph is None:
            morph = r.new_zeros(r.shape[0], self.n_morph)
        parts.append(morph.to(dtype=r.dtype))
        return torch.cat(parts, dim=1)

    def forward(self, mean_spectrum: torch.Tensor, morph: torch.Tensor | None = None) -> Any:
        return self.mlp(self.in_norm(self.features(mean_spectrum, morph)))


class SpectralSeedNet(nn.Module):
    """Two pathways, concatenated, one embedding block, one K=1 cosine head.

    Returns exactly the shapes ``SpectralQuadNet`` does, because every caller in
    ``engine/`` branches on them: a dict in ``.train()`` mode (``main`` plus the
    single ``aux_spatial`` head, ``emb`` when asked), the bare logits in
    ``.eval()``.

    Attribute names are the checkpoint schema. ``se``, ``spatial``,
    ``spectral``, ``fuse``, ``embed_net``, ``arcface_head`` and
    ``aux_head_spatial`` are the top-level keys of every schema-v4 bundle.
    """

    #: Checkpoint schema. Distinct from ``SpectralQuadNet``'s 3 — the two share
    #: no tensor names and no bundle is loadable across them.
    SCHEMA_VERSION: int = 4
    #: The ``model.arch`` value that selects this class.
    ARCH: str = "spectral_seed_net"

    def __init__(
        self,
        cfg: ExperimentConfig | Any,
        physical_wl: torch.Tensor,
        num_classes: int = 90,
        num_bands: int = 40,
        dropout: float = 0.15,
        wl_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        model_cfg = cfg.model
        self.n_morph = int(model_cfg.n_morphometrics)

        # ── Shared spectral attention — 6 parameters, free, kept ───────
        self.se = MaskedSpectralECA(num_bands)

        # ── Spatial path — Branch C verbatim ──────────────────────────
        self.spatial = SpatialCNNBranch(
            num_bands,
            EMBED_DIM,
            stem_channels=int(model_cfg.stem_channels),
            width_mult=float(model_cfg.spatial_width_mult),
        )

        # ── Spectral path ─────────────────────────────────────────────
        self.spectral = SpectralPath(
            num_bands=num_bands,
            wavelengths=physical_wl,
            out_dim=EMBED_DIM,
            n_indices=int(model_cfg.index_bank_size),
            n_depths=int(model_cfg.continuum_depths),
            n_morph=self.n_morph,
            hidden=int(model_cfg.spectral_hidden),
            drop=float(model_cfg.fusion_drop),
        )

        # ── Fusion: concat + MLP (CHANGES §16.2, replacing the pool) ──
        self.fuse = nn.Sequential(
            nn.Dropout(float(model_cfg.fusion_drop)),
            nn.Linear(2 * EMBED_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
        )

        self.embed_net = EmbedNet(EMBED_DIM, 2 * EMBED_DIM, dropout)

        # ── Head: K=1, single global margin ───────────────────────────
        # `per_class_margin` and `pairwise_penalty` are off by default (IC-9);
        # the buffers and the code stay so A7 can switch them on one at a time.
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            EMBED_DIM,
            num_classes,
            K=int(model_cfg.subcenter_K),
            s=float(cfg.single.arcface_s),
            m_base=float(cfg.single.arcface_m),
            m_delta=float(cfg.stage2.arcface_m_delta),
            m_min=float(cfg.stage2.arcface_m_min),
            m_max=float(cfg.stage2.arcface_m_max),
            tau=0.0,  # K=1 makes the pooling temperature meaningless: max of one.
            pairwise_delta=(
                float(cfg.stage2.pairwise_margin_delta) if bool(model_cfg.pairwise_penalty) else 0.0
            ),
        )

        # ── One auxiliary head, on the spatial path, fixed weight ─────
        self.aux_head_spatial = AuxiliaryHead(
            EMBED_DIM, int(model_cfg.aux_head_hidden), num_classes
        )
        self._init_weights()

    # ── Construction from a composed config ───────────────────────────

    @classmethod
    def from_config(cls, cfg: ExperimentConfig | Any, physical_wl: torch.Tensor) -> SpectralSeedNet:
        """Build from a composed experiment config — the single canonical path."""
        return cls(
            cfg=cfg,
            physical_wl=physical_wl,
            num_classes=cfg.data.num_classes,
            num_bands=cfg.data.num_bands,
            dropout=cfg.single.dropout,
            wl_embed_dim=cfg.model.wl_embed_dim,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Control API — the same surface the stages call ────────────────

    def set_dropout(self, p: float) -> None:
        """Every dropout rate, including any attention module's (IC-14)."""
        set_module_dropout(self, p)

    def set_subcentre_tau(self, tau: float) -> None:
        """Accepted and ignored at K=1: the pooled cosine is a max over one centre.

        Kept so the stage loops need no ``hasattr`` guard, and so switching
        ``model.subcenter_K`` back up for an ablation re-activates the schedule
        without a code change.
        """
        if self.arcface_head.K > 1:
            self.arcface_head.set_tau(tau)

    # ── Forward ───────────────────────────────────────────────────────

    def pathway_inputs(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        morph: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """What each pathway consumes, before any parameter touches it.

        The two-pathway analogue of ``SpectralQuadNet.branch_inputs``: the
        distinctness claim — that the spatial path sees texture × spectral
        position and the spectral path sees a per-band global average — is a
        claim about *these* tensors and cannot be made from the embeddings.
        """
        m = foreground_mask(x, mask)
        return {
            "spatial": x * m,
            "spectral": masked_mean_spectrum(x, m),
        }

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_embed: bool = False,
        arc_m: float | None = None,
        branch_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        morph: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Run both pathways, concatenate, embed and classify.

        Args:
            branch_mask: Accepted for signature compatibility with
                ``SpectralQuadNet`` and applied as ``(spatial, spectral)`` when
                given — the leave-one-pathway-out influence diagnostic uses it.
                There is no *stochastic* branch dropout here: with two pathways
                and the audited evidence that an asymmetric policy biases the
                gate, dropping one is an ablation, not a regulariser.
        """
        m = foreground_mask(x, mask)
        x = self.se(x, m)

        b_spatial = self.spatial(x, m)
        b_spectral = self.spectral(masked_mean_spectrum(x, m), morph)

        if branch_mask is not None:
            b_spatial = b_spatial * branch_mask[0]
            b_spectral = b_spectral * branch_mask[1]

        fused = self.fuse(torch.cat([b_spatial, b_spectral], dim=1))
        emb = self.embed_net(fused)
        emb_n = F.normalize(emb, dim=1)

        logits = self.arcface_head(emb_n, labels, global_m=arc_m)

        if self.training:
            out = {
                "main": logits,
                # Named for the pathway it supervises. `_compute_aux_loss`
                # discovers any `aux_*` key, so no per-architecture branch.
                "aux_spatial": self.aux_head_spatial(b_spatial),
            }
            if return_embed:
                out["emb"] = emb_n
            return out

        if return_embed:
            return logits, emb_n
        return logits  # type: ignore[no-any-return]
