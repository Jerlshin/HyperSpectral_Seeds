"""``SpectralSeedNet`` — the two-pathway 256-band network (IC-10 / CHANGES §16.2).

The primary architecture of this study. It consumes the **complete 256-band
VIS–NIR cube**: no band selection, no PCA, no reduction of any kind stands
between ``dataset/patches.npy`` and the first parameter. Every component below
is sized from ``data.num_bands`` rather than tuned for one band count, which is
what makes 256 the native case and the retained band-selection arms (k = 40,
k = 100, the band study's sweep) the *reduced* ones.

What "native at 256" means concretely
─────────────────────────────────────
Two components would otherwise make a full cube either unaffordable or
mis-shaped, and both are solved rather than parameterised around:

* the spatial path's 3-D stem derives its spectral strides from the band count
  (:func:`~spectralquadnet.models.branches.spatial_cnn.spectral_stride_schedule`),
  so 256 bands fold at depth 8 for 0.87 GMAC, instead of at depth 32 for
  2.89 GMAC and a ``Conv2d(2048 → 192)`` fold;
* the continuum hull is computed by an exact :math:`O(C^2)` suffix-maximum
  (:class:`~spectralquadnet.models.branches.spectral_stats.ContinuumDepths`)
  rather than by enumerating :math:`O(C^3)` chords, which at 256 bands would be
  16.8 M chords and 570 MB of resident buffers.

Nothing else in the network needed a band-count special case: the index bank,
the Savitzky–Golay derivative operators, ``MaskedSpectralECA``'s ECA kernel width
and the descriptor width are all functions of :math:`C` already.

Why two pathways and not four
─────────────────────────────
The audited evidence brackets exactly two information sources on this dataset,
and the numbers are the most useful pair in the whole project:

* **LDA on mean spectra scores 0.5916** accuracy at k = 40
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
index bank and continuum depths (the signal the entire NIR-chemometrics
literature is built on — kept on theoretical rather than evidential grounds, and
flagged as such), ``MaskedSpectralECA`` (ten parameters at 256 bands; its ECA
kernel width is ``f(log2 C)``), the exact-zero
background invariant, mixup, EMA, D₄, and the cosine/ArcFace head with its
elaborations stripped.

≈3.05 M parameters on the 256-band input, against ``SpectralQuadNet``'s ≈5.26 M
on the same input. The exact counts are produced by
:func:`~spectralquadnet.models.registry.parameter_breakdown` and written into
every run's results JSON, so they are a recorded artifact rather than a number
in a docstring.

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
from spectralquadnet.models.branches.spatial_cnn import (
    DEFAULT_FOLDED_DEPTH,
    DEFAULT_TAIL_STRIDES,
    SpatialCNNBranch,
)
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

#: The two pathways, in fusion (and ``branch_mask``) order. ``model.pathways``
#: names a non-empty subset of these (X2, S09 FW-16).
PATHWAYS: tuple[str, ...] = ("spatial", "spectral")

#: ``model.spectral_descriptor`` values (S13 Y3). ``full`` is the shipped
#: descriptor ``[index bank | continuum depths | SNV | D₁ | D₂ | morph]``;
#: ``snv_morph`` keeps only what S10 F49 found the MLP actually uses: ``[SNV | morph]``.
SPECTRAL_DESCRIPTORS: tuple[str, ...] = ("full", "snv_morph")

#: Parameter-name prefixes the per-module gradient norm is reported under
#: (S10 P0.5, F53). Every parameter belongs to exactly one entry, in this order;
#: ``tests/unit/test_s11_instrumentation.py`` pins the partition.
GRAD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gate", ("se.",)),
    ("stem", ("spatial.stem.",)),
    ("tail", ("spatial.stages.",)),
    ("proj", ("spatial.proj.",)),
    ("spectral", ("spectral.",)),
    ("fuse", ("fuse.",)),
    ("embed", ("embed_net.",)),
    ("head", ("arcface_head.",)),
    ("aux", ("aux_head_spatial.",)),
)

#: The model-declared clip partition (``clip_partition=model``, S10 P0.5):
#: ``fuse`` joins ``embed_net`` as ``fusion``. The last entry is the catch-all.
CLIP_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("head", ("arcface_head.",)),
    ("fusion", ("fuse.", "embed_net.")),
    ("backbone", ()),
)


def resolve_pathways(requested: Any) -> tuple[str, ...]:
    """Validate ``model.pathways`` and return it in canonical (fusion) order.

    Raises:
        ValueError: An unknown name, a duplicate, or no pathway at all — each
            would otherwise silently train a different network from the one the
            run directory claims.
    """
    names = [str(p).lower() for p in (requested if requested is not None else PATHWAYS)]
    unknown = sorted(set(names) - set(PATHWAYS))
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(
            f"model.pathways={list(names)!r}: expected a non-empty, duplicate-free subset of "
            f"{list(PATHWAYS)}"
        )
    return tuple(p for p in PATHWAYS if p in names)


class SpectralPath(nn.Module):
    """Global chemometrics over the foreground mean spectrum.

    .. code-block:: text

        x̄ (B, 256)
          ├─ SoftIndexBank(64)        learned normalised-difference indices
          ├─ ContinuumDepths(16)      hull-removed absorption depths
          ├─ snv(x̄)       (256)      ┐ Branch A's physics, as fixed features:
          ├─ D₁ snv(x̄)    (256)      │ zero parameters, exact on the irregular
          ├─ D₂ snv(x̄)    (256)      ┘ λ grid (Savitzky–Golay)
          └─ morph          (8)       physical size and shape
                → LayerNorm(856) → MLP(856 → 256 → 256)

    Descriptor width is ``n_indices + n_depths + 3C + n_morph``: **856** on the
    full cube, 208 on the retained 40-band arm. The full-resolution SNV and
    derivative vectors are the primary path's whole point — a continuum-removal
    or derivative feature computed on a selected subset is a different physical
    quantity from the same feature computed on the acquired spectrum, and the
    study's question is what the acquired spectrum supports.

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
        descriptor: str = "full",
    ) -> None:
        super().__init__()
        self.n_morph = int(n_morph)
        self.descriptor = str(descriptor)
        if self.descriptor not in SPECTRAL_DESCRIPTORS:
            raise ValueError(
                f"model.spectral_descriptor={descriptor!r}: expected one of {list(SPECTRAL_DESCRIPTORS)}"
            )
        if self.descriptor == "full":
            self.index_bank = SoftIndexBank(num_bands, n_indices)
            self.continuum = ContinuumDepths(wavelengths, n_depths=n_depths)
            self.derivatives = SpectralDerivatives(wavelengths)
            # 3 * num_bands is [snv, D1, D2]; the derivative module stacks them.
            in_dim = self.index_bank.n_indices + self.continuum.n_depths + 3 * int(num_bands)
        else:
            # S13 Y3: the index bank never left uniform, and the continuum and
            # derivative blocks reach the MLP at ≈ 1 % amplitude (S10 F49); none
            # of them is built, so none holds a parameter or a buffer.
            in_dim = int(num_bands)
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
        if self.descriptor == "full":
            shape = self.derivatives(snv(r))  # (B, 3, C)
            parts = [
                self.index_bank(r),
                self.continuum(r),
                shape.flatten(1),
            ]
        else:
            parts = [snv(r)]
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
        num_bands: int = 256,
        dropout: float = 0.15,
        wl_embed_dim: int = 16,
        input_side: int | None = None,
    ) -> None:
        super().__init__()
        model_cfg = cfg.model
        self.n_morph = int(model_cfg.n_morphometrics)
        self.num_bands = int(num_bands)
        #: Live pathways (X2). Read before construction so a bad value fails
        #: before any weight is drawn; every module is still built below, in the
        #: same order, so the live pathway's initial weights are the full model's.
        self.pathways = resolve_pathways(getattr(model_cfg, "pathways", PATHWAYS))

        # ── Shared spectral attention — 6 parameters, free, kept ───────
        self.se = MaskedSpectralECA(num_bands)

        # ── Spatial path — Branch C verbatim ──────────────────────────
        self.spatial = SpatialCNNBranch(
            num_bands,
            EMBED_DIM,
            stem_channels=int(model_cfg.stem_channels),
            width_mult=float(model_cfg.spatial_width_mult),
            stem_folded_depth=int(getattr(model_cfg, "stem_folded_depth", DEFAULT_FOLDED_DEPTH)),
            # S13 Y3 / Y2. Defaults are the shipped branch, bit for bit.
            tail_strides=list(getattr(model_cfg, "spatial_tail_strides", DEFAULT_TAIL_STRIDES)),
            cbam_min_hw=int(getattr(model_cfg, "cbam_min_hw", 0)),
            input_side=input_side,
            mixstyle=bool(getattr(model_cfg, "spatial_mixstyle", False)),
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
            descriptor=str(getattr(model_cfg, "spectral_descriptor", "full")),
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
        self._freeze_disabled_pathways()

    # ── Construction from a composed config ───────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: ExperimentConfig | Any,
        physical_wl: torch.Tensor,
        input_side: int | None = None,
    ) -> SpectralSeedNet:
        """Build from a composed experiment config — the single canonical path.

        ``input_side`` is the patch side (the pipeline passes the cube's); only
        ``model.cbam_min_hw > 0`` reads it (S13 Y3).
        """
        return cls(
            cfg=cfg,
            physical_wl=physical_wl,
            num_classes=cfg.data.num_classes,
            num_bands=cfg.data.num_bands,
            dropout=cfg.single.dropout,
            wl_embed_dim=cfg.model.wl_embed_dim,
            input_side=input_side,
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

    def _freeze_disabled_pathways(self) -> None:
        """Take a disabled pathway out of training entirely (X2).

        Its forward is skipped and its output replaced by zeros, so its
        parameters would receive no gradient anyway; freezing them makes that
        explicit to the optimiser (which skips ``requires_grad=False`` tensors),
        to the clip partition, and to DDP, which raises on a parameter that
        requires a gradient and never gets one. The spatial pathway takes its
        auxiliary head with it: that head reads the spatial output, so with the
        pathway off its term would be a constant-input classifier.
        """
        frozen: list[nn.Module] = []
        if "spatial" not in self.pathways:
            frozen += [self.spatial, self.aux_head_spatial]
        if "spectral" not in self.pathways:
            frozen.append(self.spectral)
        for module in frozen:
            module.requires_grad_(False)

    # ── Declarations the engine reads (S10 P0.1, P0.5) ────────────────

    def pathway_labels(self) -> tuple[str, ...]:
        """The maskable pathways in ``branch_mask`` order: ``("SPATIAL", "SPECTRAL")``.

        Read by the leave-one-pathway-out influence probe, so its log lines and
        ``influence/branch_*`` series name the two pathways rather than the four
        letters ``SpectralQuadNet`` uses (S09 §8, S10 B14). Always both: a
        pathway the X2 switch disabled reports an influence of 0.
        """
        return tuple(p.upper() for p in PATHWAYS)

    def grad_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """``(label, prefixes)`` for the per-module ``grad_norm/<label>`` series."""
        return GRAD_GROUPS

    def clip_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The partition ``clip_partition=model`` clips by: ``fuse`` joins ``fusion``."""
        return CLIP_GROUPS

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

        # X2: a disabled pathway is not run; its output is exactly zero in train
        # and eval alike, so the fusion sees the same input either way.
        b_spatial: torch.Tensor | None = self.spatial(x, m) if "spatial" in self.pathways else None
        b_spectral: torch.Tensor | None = (
            self.spectral(masked_mean_spectrum(x, m), morph)
            if "spectral" in self.pathways
            else None
        )
        if b_spatial is None:
            assert b_spectral is not None  # `resolve_pathways` refuses an empty set
            b_spatial = b_spectral.new_zeros(b_spectral.shape)
        if b_spectral is None:
            b_spectral = b_spatial.new_zeros(b_spatial.shape)

        if branch_mask is not None:
            b_spatial = b_spatial * branch_mask[0]
            b_spectral = b_spectral * branch_mask[1]

        fused = self.fuse(torch.cat([b_spatial, b_spectral], dim=1))
        emb = self.embed_net(fused)
        emb_n = F.normalize(emb, dim=1)

        logits = self.arcface_head(emb_n, labels, global_m=arc_m)

        if self.training:
            out = {"main": logits}
            if "spatial" in self.pathways:
                # Named for the pathway it supervises. `_compute_aux_loss`
                # discovers any `aux_*` key, so no per-architecture branch.
                # Absent under X2's `spectral_only`, which makes the aux term 0.
                out["aux_spatial"] = self.aux_head_spatial(b_spatial)
            if labels is not None and (arc_m is None or arc_m > 0.0):
                # The unpenalised logits, for `train/acc_plain` only (S10 P0.3):
                # under a margin `main` scores the target at cos(θ_y + m), so
                # its argmax is not the network's prediction. Detached and
                # computed without labels — no margin, no pairwise penalty — so
                # it adds nothing to the graph and moves no number.
                with torch.no_grad():
                    out["main_plain"] = self.arcface_head(emb_n.detach())
            if return_embed:
                out["emb"] = emb_n
            return out

        if return_embed:
            return logits, emb_n
        return logits  # type: ignore[no-any-return]
