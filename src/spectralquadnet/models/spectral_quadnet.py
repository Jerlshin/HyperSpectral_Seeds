"""The assembled four-branch network.

Composes the four modality branches (spectral profile, spectral index bank,
spectral-spatial CNN, λ-aware SpecFormer), shared spectral attention, the
morphology token, cross-modal fusion, per-branch auxiliary heads and **one**
classification head into a single ``nn.Module``.

HD-1 / T2-10 removed the second head. Stages 1-3 now share the sub-centre
cosine head and differ in the margin alone — ``cfg.stage1.arcface_m = 0``
makes Stage 1 a plain cosine (NormFace) classifier, which §2.3.8 shows is
nearly what the linear head already was, since ``EmbedNet``'s terminal
LayerNorm pins ``||e|| ~ 16``. The curriculum's remaining differences are the
margin, the sampler and the optimiser, and the six-way Stage-1 → Stage-2
discontinuity of §2.4.6 is gone.

Tier 3 (schema v3) changed what the branches *see*, which is the change §3.3's
controlling constraint is about: **each branch must see something the others
cannot reconstruct.**

===========  ==========================================  =========================
Branch       Input after Tier 3                          Unique information
===========  ==========================================  =========================
A            8×8 per-cell SNV spectra + ∂λ, ∂²λ          spectral *shape*, gain-free
B            NDI bank + continuum depths + morphometry   ratios and physical size
C            the (40, 64, 64) cube through a 3-D stem    spatial texture × spectral position
D            4×4 raw grid spectra, λ-uniform tokens      long-range band interactions
morphology   the 8 persisted morphometrics               size, as a fifth token
===========  ==========================================  =========================

Two side inputs arrive with it, both optional and both with exact fallbacks:
``mask`` (FE-2 / T3-7 — the persisted fill map, falling back to the ``> 1e-5``
band-sum threshold) and ``morph`` (P-4 / T4-4 — the eight morphometrics, falling
back to zeros). Neither array exists until ``scripts/prepare_dataset.py`` is
re-run, so the fallbacks are the operative path today and the model runs
unchanged without them.

Invariants:
    Attribute names are part of the checkpoint schema. ``self.se``,
    ``self.wl_pe_cnn``, ``self.branch_{a,b,c,d}``, ``self.morphology_embed``,
    ``self.cross_interaction``, ``self.aux_head_{a,b,c,d}``, ``self.embed_net``
    and ``self.arcface_head`` are the top-level keys of every schema-v3
    checkpoint; renaming any of them breaks ``load_state_dict(strict=True)``.
    Schema-v1 bundles carry a ``linear_head`` this model no longer has, and
    neither v1 nor v2 can be loaded into v3 at all —
    ``engine/checkpoint.py::remap_state_dict`` says so in as many words rather
    than silently dropping two thirds of the tensors.

    Construction order is significant. Each sub-module's ``_init_weights``
    draws from the same global torch RNG stream, so the order in which
    sub-modules are constructed in ``__init__`` determines the initial
    weights. Do not reorder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralquadnet.models.blocks.attention import MaskedSpectralECA
from spectralquadnet.models.blocks.positional import PhysicalWavelengthPE
from spectralquadnet.models.branches.spatial_cnn import SpatialCNNBranch
from spectralquadnet.models.branches.specformer import SpecFormerBranch
from spectralquadnet.models.branches.spectral_profile import SpectralProfileBranch
from spectralquadnet.models.branches.spectral_stats import SpectralStatsBranch
from spectralquadnet.models.control import set_dropout as set_module_dropout
from spectralquadnet.models.fusion import CrossModalInteraction, EmbedNet, MorphologyEmbed
from spectralquadnet.models.heads import AdaptiveSubcenterArcFaceHead, AuxiliaryHead
from spectralquadnet.models.stats_ops import (
    extract_grid_spectra_multi,
    foreground_mask,
    masked_mean_spectrum,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps hydra out of the model layer
    from spectralquadnet.config.schema import ExperimentConfig

#: Relative branch-drop rates for ``(A, B, C, D)``, scaled by
#: ``cfg.model.branch_drop_prob`` in :meth:`SpectralQuadNet.forward`.
#:
#: **BR-3 / T3-2 inverted this vector.** The reference implementation dropped
#: Branch C at 0.30 and Branch D at 0.20 and never dropped A or B — that is,
#: it regularised hardest against the one branch whose input no other branch can
#: reconstruct, and not at all against the two that were near-duplicates of each
#: other (§2.2.7, M-5). §3.3 BR-3's prescription is ``(0.15, 0.15, 0, 0.15)``:
#: drop the reconstructible branches, protect the irreplaceable one. Expressed
#: here as a profile scaled by the config so that ``branch_drop_prob = 0.20``
#: reproduces those rates and ``0.0`` means off, which is what N-1d wanted and
#: what lets Stage 3 actually disable branch masking.
BRANCH_DROP_PROFILE: tuple[float, float, float, float] = (0.75, 0.75, 0.0, 0.75)

#: The order branches are constructed, fused and auxiliary-supervised in.
#: ``cfg.model.enabled_branches`` selects a subset of it; the *order* is fixed
#: here so an ablation arm cannot silently reorder the RNG stream by listing
#: its branches differently.
BRANCH_ORDER: tuple[str, ...] = ("a", "b", "c", "d")

#: Accepted values of ``cfg.model.fusion_mode`` (A5 / CHANGES §5.3).
FUSION_MODES: tuple[str, ...] = ("bilinear_gate", "gate", "concat_mlp")


def _enabled_branches(model_cfg: Any) -> tuple[str, ...]:
    """``cfg.model.enabled_branches``, defaulting to all four."""
    listed = getattr(model_cfg, "enabled_branches", None)
    if not listed:
        return BRANCH_ORDER
    return tuple(str(b).strip().lower() for b in listed)


def _drop_profile(model_cfg: Any) -> tuple[float, ...]:
    """``cfg.model.branch_drop_profile``, defaulting to the audited asymmetric one.

    The default is the *historical* vector rather than the symmetric one, so a
    config written before A3 existed keeps reproducing its own run. The shipped
    configs set the key explicitly in both directions — the audited baseline to
    ``(0.75, 0.75, 0, 0.75)`` and everything else to symmetric — because
    "which branches get dropped" is a claim a reader should not have to infer
    from a default.
    """
    listed = getattr(model_cfg, "branch_drop_profile", None)
    if not listed:
        return BRANCH_DROP_PROFILE
    profile = tuple(float(p) for p in listed)
    if len(profile) != len(BRANCH_ORDER):
        raise ValueError(
            f"model.branch_drop_profile must have {len(BRANCH_ORDER)} entries "
            f"(one per branch in {BRANCH_ORDER}); got {profile!r}"
        )
    return profile


class SpectralQuadNet(nn.Module):
    """
    Four-branch hyperspectral classification model.

    Branches
    ────────
    A  SpectralProfileBranch  — per-cell SNV spectra and their λ-derivatives
    B  SpectralStatsBranch    — learned NDI bank + continuum depths + morphometry
    C  SpatialCNNBranch       — joint spectral-spatial 3-D stem over the cube
    D  SpecFormerBranch       — λ-uniform spectral patch transformer

    Deep Supervision (Stage 1 only)
    ────────────────────────────────
    Each branch has its own AuxiliaryHead so it is individually
    discriminative before cross-modal fusion.  During inference the
    aux heads are not called (forward returns a plain tensor).

    Head
    ────
    One ``AdaptiveSubcenterArcFaceHead`` for all three stages (HD-1).
    Stage 1 runs it at ``arc_m = 0`` (a cosine classifier), Stage 2 warms a
    global margin up and then hands over to the per-class vector, Stage 3
    anneals that same vector multiplicatively.
    """

    #: Checkpoint schema written by bundles of this architecture.
    SCHEMA_VERSION: int = 3
    #: The ``model.arch`` value that selects this class.
    ARCH: str = "spectral_quadnet"

    def __init__(
        self,
        cfg: ExperimentConfig | Any,
        physical_wl: torch.Tensor,
        num_classes: int = 90,
        num_bands: int = 256,
        dropout: float = 0.30,
        wl_embed_dim: int = 16,
    ) -> None:
        super().__init__()

        tower_ch = 96

        self.branch_drop_prob = cfg.model.branch_drop_prob
        self.grid_a = int(cfg.model.grid_size_a)
        self.grid_d = int(cfg.model.grid_size_d)
        self.n_morph = int(cfg.model.n_morphometrics)
        # A3. Ordered by BRANCH_ORDER, never by the config's listing — see that
        # constant. `enabled` is part of the checkpoint's meaning, so an arm's
        # bundle is not loadable into a differently-configured arm; `load_ckpt`
        # would fail strict and say so.
        self.enabled_branches = tuple(
            b for b in BRANCH_ORDER if b in set(_enabled_branches(cfg.model))
        )
        if not self.enabled_branches:
            raise ValueError(
                "model.enabled_branches must name at least one of "
                f"{BRANCH_ORDER}; got {list(_enabled_branches(cfg.model))!r}"
            )
        self.drop_profile = _drop_profile(cfg.model)

        # ── Shared spectral attention ─────────────────────────────────
        self.se = MaskedSpectralECA(num_bands)

        # ── Physical wavelength positional encoding (for 1-D CNN branches) ──
        self.wl_pe_cnn = PhysicalWavelengthPE(physical_wl, tower_ch)

        # ── The branches, in BRANCH_ORDER ─────────────────────────────
        # A disabled branch is not constructed at all, so an A3 arm is genuinely
        # the smaller model rather than the full one with a tensor multiplied by
        # zero — its parameter count, its FLOPs and its checkpoint all shrink.
        # `None` rather than a missing attribute: `nn.Module.__setattr__` would
        # reject the assignment, and every read site is already guarded.
        self.branch_a = (
            SpectralProfileBranch(
                wavelengths=physical_wl,
                out_dim=256,
                tower_ch=tower_ch,
                wl_pe_module=self.wl_pe_cnn,
                n_freq=wl_embed_dim,
            )
            if "a" in self.enabled_branches
            else None
        )
        self.branch_b = (
            SpectralStatsBranch(
                num_bands=num_bands,
                wavelengths=physical_wl,
                out_dim=256,
                n_indices=cfg.model.index_bank_size,
                n_depths=cfg.model.continuum_depths,
                n_morph=self.n_morph,
            )
            if "b" in self.enabled_branches
            else None
        )
        self.branch_c = (
            SpatialCNNBranch(
                num_bands,
                256,
                stem_channels=cfg.model.stem_channels,
                width_mult=float(getattr(cfg.model, "spatial_width_mult", 1.0)),
            )
            if "c" in self.enabled_branches
            else None
        )
        self.branch_d = (
            SpecFormerBranch(
                physical_wl=physical_wl,
                num_bands=num_bands,
                patch_size=cfg.model.specf_patch,
                d_model=cfg.model.specf_dim,
                n_heads=cfg.model.specf_heads,
                n_layers=cfg.model.specf_layers,
                out_dim=256,
                dropout=cfg.model.specf_drop,
                n_freq=wl_embed_dim,
            )
            if "d" in self.enabled_branches
            else None
        )

        # ── The fifth modality (FU-4) ─────────────────────────────────
        self.morphology_embed = MorphologyEmbed(n_morph=self.n_morph, d=256)

        # ── Cross-modal fusion ────────────────────────────────────────
        self.fusion_mode = str(getattr(cfg.model, "fusion_mode", "bilinear_gate"))
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"model.fusion_mode must be one of {FUSION_MODES}, got {self.fusion_mode!r}"
            )
        self.cross_interaction = CrossModalInteraction(
            num_modalities=len(self.enabled_branches) + 1,
            d=256,
            rank=cfg.model.fusion_rank,
            gate_hidden=cfg.model.fusion_gate_hidden,
            drop=cfg.model.fusion_drop,
            mode=self.fusion_mode,
        )

        # ── Auxiliary heads — one per enabled branch (deep supervision) ─
        aux_hidden = cfg.model.aux_head_hidden
        self.aux_head_a = AuxiliaryHead(256, aux_hidden, num_classes) if self.branch_a else None
        self.aux_head_b = AuxiliaryHead(256, aux_hidden, num_classes) if self.branch_b else None
        self.aux_head_c = AuxiliaryHead(256, aux_hidden, num_classes) if self.branch_c else None
        self.aux_head_d = AuxiliaryHead(256, aux_hidden, num_classes) if self.branch_d else None

        # ── Embedding network ─────────────────────────────────────────
        self.embed_net = EmbedNet(256, 512, dropout)

        # ── Classification head (one, for every stage — HD-1) ─────────
        # IC-9: the pairwise confusion penalty is now gated by an explicit flag
        # rather than by whether someone happened to leave its delta non-zero.
        # It was fitted on the selection split, never ablated, and aimed at the
        # hard classes that never moved (CHANGES §5.4); A7 arm 4 measures it.
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            256,
            num_classes,
            K=cfg.model.subcenter_K,
            s=cfg.stage2.arcface_s,
            m_base=cfg.stage2.arcface_m,
            m_delta=cfg.stage2.arcface_m_delta,
            m_min=cfg.stage2.arcface_m_min,
            m_max=cfg.stage2.arcface_m_max,
            tau=cfg.model.subcenter_tau_init,
            pairwise_delta=(
                cfg.stage2.pairwise_margin_delta
                if bool(getattr(cfg.model, "pairwise_penalty", True))
                else 0.0
            ),
        )
        self._init_weights()

    # ── What the fusion and the influence probe address ───────────────

    def pathway_labels(self) -> tuple[str, ...]:
        """Uppercased identifiers of the maskable pathways, in fusion order.

        ``compute_branch_influence`` builds its ablation vectors from this, so a
        two-branch A3 arm produces a two-entry influence table rather than four
        entries of which two are structurally zero.
        """
        return tuple(b.upper() for b in self.enabled_branches)

    # ── Construction from a composed config ───────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: ExperimentConfig | Any,
        physical_wl: torch.Tensor,
    ) -> SpectralQuadNet:
        """Build the network from a composed experiment config.

        Kept as the single canonical construction path so the Stage-1 dropout
        value, band count and class count cannot drift between ``train.py``, the
        regression tests and any future evaluation entrypoint.

        Args:
            cfg: Composed experiment config (``ExperimentConfig`` or an
                equivalent Hydra ``DictConfig``).
            physical_wl: Min-max normalised wavelength vector of shape
                ``(num_bands,)``.

        Returns:
            A freshly initialised ``SpectralQuadNet``.
        """
        return cls(
            cfg=cfg,
            physical_wl=physical_wl,
            num_classes=cfg.data.num_classes,
            num_bands=cfg.data.num_bands,
            dropout=cfg.stage1.dropout,
            wl_embed_dim=cfg.model.wl_embed_dim,
        )

    # ── Weight initialisation ─────────────────────────────────────────

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

    # ── Control API ──────────────────────────────────────────────────

    def set_dropout(self, p: float) -> None:
        """Set every dropout rate, **including** ``nn.MultiheadAttention``'s (IC-14).

        The walk used to cover ``nn.Dropout`` alone, which left Branch D's four
        attention sites pinned at their construction-time 0.15 for the whole run
        regardless of the stage schedule. See
        :func:`~spectralquadnet.models.control.set_dropout`.
        """
        set_module_dropout(self, p)

    def set_subcentre_tau(self, tau: float) -> None:
        """Set the head's sub-centre pooling temperature (HD-2(i), T2-9)."""
        self.arcface_head.set_tau(tau)

    # ── Forward ───────────────────────────────────────────────────────

    def _parametric_branch_inputs(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        morph: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Branch A's, B's and D's inputs — everything :meth:`forward` actually consumes.

        Branch C's input is deliberately absent. It is ``x * m``, a full
        ``(B, C, H, W)`` product, and :meth:`forward` never used the copy this
        built: it passes ``(x, m)`` straight to ``self.branch_c``, whose stem
        applies the mask itself at each of its three resolutions. Materialising
        it here allocated and wrote 80 MB per forward at batch 128 for a tensor
        that was immediately dropped — which, on a 20 GB Metal budget, is
        allocation the backward pass needed.

        :meth:`branch_inputs` still returns it, because the distinctness gate is
        a claim about what Branch C *sees* and has to be able to look.
        """
        n_batch, n_bands = x.shape[0], x.shape[1]
        # One masked cube shared by both grids; see `extract_grid_spectra_multi`.
        # Only the grids a live branch consumes are extracted — an A3 arm
        # without Branch A should not pay for an 8×8 grid nobody reads.
        grids: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        wanted: list[int] = []
        if self.branch_a is not None:
            wanted.append(self.grid_a)
        # De-duplicated: A and D need not share a grid, but nothing forbids a
        # config from setting both to 4, and pooling the same buffer twice to
        # get the same tensor twice is pure cost.
        if self.branch_d is not None and self.grid_d not in wanted:
            wanted.append(self.grid_d)
        if wanted:
            extracted = extract_grid_spectra_multi(x, tuple(wanted), mask=m)
            grids = dict(zip(wanted, extracted, strict=True))

        inputs: dict[str, torch.Tensor] = {}
        if self.branch_a is not None:
            grid_a, mass_a = grids[self.grid_a]
            inputs["a"] = self.branch_a.shape_channels(grid_a.reshape(-1, n_bands))
            inputs["a_mass"] = mass_a
        if self.branch_b is not None:
            inputs["b"] = self.branch_b.features(
                masked_mean_spectrum(x, m),
                morph if morph is not None else x.new_zeros(n_batch, self.n_morph),
            )
        if self.branch_d is not None:
            inputs["d"] = grids[self.grid_d][0]
        return inputs

    def branch_inputs(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        morph: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The tensor each branch actually consumes, before any parameter touches it.

        The §4.3 test ``test_branch_inputs_are_distinct`` hashes these pairwise;
        before BR-1…BR-4 two of them were byte-identical (§2.2.2). Exposed as a
        method rather than reconstructed in the test so the two cannot drift —
        :meth:`forward` computes the same tensors through
        :meth:`_parametric_branch_inputs`, and only ``"c"`` is added here, since
        the forward reaches Branch C's input through the branch itself.
        """
        m = foreground_mask(x, mask)
        inputs = self._parametric_branch_inputs(x, m, morph)
        if self.branch_c is not None:
            inputs["c"] = x * m
        return inputs

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
        """Run the four branches, fuse them and classify.

        Args:
            mask: Optional ``(B, H, W)`` fill map (FE-2 / T3-7). ``None`` falls
                back to the ``> 1e-5`` band-sum threshold, exactly.
            morph: Optional ``(B, 8)`` standardised morphometrics (P-4 / T4-4).
                ``None`` substitutes zeros, which is the mean of the
                train-standardised feature and therefore the neutral value.

        Returns:
            In ``.train()`` mode a dict of the main logits plus the four
            auxiliary heads' logits, plus ``"balance"`` when ``labels`` are
            given (HD-2's sub-centre load-balancing term) and ``"emb"`` when
            ``return_embed``; in ``.eval()`` mode the logits alone, or
            ``(logits, embedding)`` when ``return_embed``. Every caller in
            ``engine/`` branches on exactly this shape, so it is part of the
            contract, not an implementation detail.
        """
        m = foreground_mask(x, mask)

        # Apply shared spectral channel attention before branching. The mask is
        # the *input's*, so it is resolved once, above, and does not have to
        # survive whatever the attention does to the background (FE-2).
        x = self.se(x, m)

        inputs = self._parametric_branch_inputs(x, m, morph)
        n_batch = x.shape[0]

        # Each enabled branch's embedding, keyed by letter and built in
        # BRANCH_ORDER. With all four enabled this is the same four tensors, in
        # the same order, from the same expressions as before.
        raw: dict[str, torch.Tensor] = {}

        # --- BRANCH A (Spectral Profile) ---
        # Processes grid_a ** 2 regions independently, then pools the embeddings
        # by each cell's foreground mass (BR-2, M-12): a corner cell holding four
        # seed pixels no longer counts as much as one that is entirely seed.
        if self.branch_a is not None:
            ba_grid = self.branch_a.forward_channels(inputs["a"]).view(n_batch, -1, 256)
            weights = inputs["a_mass"].unsqueeze(-1)
            raw["a"] = (ba_grid * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-5)

        # --- BRANCH B (Spectral index bank) ---
        # Processes the global foreground mean spectrum and the morphometrics
        if self.branch_b is not None:
            raw["b"] = self.branch_b.forward_features(inputs["b"])

        # --- BRANCH C (Spectral-spatial CNN) ---
        # Processes full 64x64 cube with the spectral axis alive through 3 stages
        if self.branch_c is not None:
            raw["c"] = self.branch_c(x, m)

        # --- BRANCH D (SpecFormer) ---
        # Processes grid_d ** 2 regions independently, then correlates them
        if self.branch_d is not None:
            raw["d"] = self.branch_d(inputs["d"])

        # --- Branch Masking (Deep Supervision Dropout) ---
        # A3 / CHANGES §5.2: the per-branch rates come from
        # `model.branch_drop_profile` rather than a module constant, because the
        # audited asymmetric vector (C never dropped, the other three at 15%) is
        # what taught the fusion gate to route onto the always-present branch.
        # Branch C's 87% influence is confounded by it, and a symmetric profile
        # is what makes A3 a measurement of the branches instead of the policy.
        n_active = len(self.enabled_branches)
        anchor = next(iter(raw.values()))
        if branch_mask is not None:
            masked = {b: raw[b] * branch_mask[i] for i, b in enumerate(self.enabled_branches)}
        elif self.training and self.branch_drop_prob > 0.0:
            profile = [self.drop_profile[BRANCH_ORDER.index(b)] for b in self.enabled_branches]
            drop_probs = self.branch_drop_prob * torch.tensor(profile, device=anchor.device)
            keeps = (torch.rand(n_active, device=anchor.device) > drop_probs).float()
            safe_idx = torch.randint(0, n_active, (), device=anchor.device)
            safe_mask = F.one_hot(safe_idx, num_classes=n_active).float()
            keeps = torch.maximum(keeps, safe_mask)
            masked = {b: raw[b] * keeps[i] for i, b in enumerate(self.enabled_branches)}
        else:
            masked = dict(raw)

        # --- Fusion (the enabled branches plus morphology, since FU-4) ---
        be = self.morphology_embed(
            morph.to(dtype=x.dtype) if morph is not None else x.new_zeros(n_batch, self.n_morph)
        )
        joint_token = self.cross_interaction([masked[b] for b in self.enabled_branches] + [be])
        emb = self.embed_net(joint_token)

        emb_n = F.normalize(emb, dim=1)
        want_balance = self.training and labels is not None
        head_out = self.arcface_head(emb_n, labels, global_m=arc_m, return_assign=want_balance)
        logits, assign = head_out if want_balance else (head_out, None)

        if self.training:
            aux_heads = {
                "a": self.aux_head_a,
                "b": self.aux_head_b,
                "c": self.aux_head_c,
                "d": self.aux_head_d,
            }
            out = {"main": logits}
            for b in self.enabled_branches:
                head = aux_heads[b]
                if head is not None:
                    out[f"aux_{b}"] = head(raw[b])
            # HD-2(ii): the sub-centre load-balancing term. Produced here rather
            # than recomputed in the loops, because it needs the assignment the
            # head already formed on this forward.
            if assign is not None and labels is not None:
                out["balance"] = self.arcface_head.balance_loss(assign, labels)
            if return_embed:
                out["emb"] = emb_n
            return out

        if return_embed:
            return logits, emb_n
        return logits  # type: ignore[no-any-return]  # head is `nn.Module.__call__` -> Any
