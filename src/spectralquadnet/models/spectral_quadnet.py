"""The assembled four-branch network.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

===============================  ==============
Symbol                           Baseline lines
===============================  ==============
:class:`SpectralQuadNet`         1429-1631
===============================  ==============

Every method except ``__init__`` is byte-identical to the baseline — including
``forward``, the numerics-critical path — as verified by
``scripts/check_ast_no_op_move.py`` (REFACTOR_PLAN.md §3.2.1).

``__init__`` carries the migration's only two declared deviations, both required
and both enumerated in the AST checker's ``DECLARED_DEVIATIONS`` table:

1. **``_PHYSICAL_WL`` global → ``physical_wl`` parameter.** The baseline read a
   module-level global populated by ``_load_wavelengths_to_gpu``; that global is
   now :class:`~spectralquadnet.data.mmap_store.DataStore.wavelengths` and is
   injected explicitly (REFACTOR_PLAN.md §3.4).
2. **``cfg["key"]`` / ``cfg.get("key", default)`` → ``cfg.<group>.<key>``.** The
   flat ``CONFIG`` dict is now the composed, grouped Hydra config. Note that the
   ``.get(..., default)`` fallbacks disappear on purpose: the structured schema
   makes every key mandatory, so a missing key now fails at config-composition
   time instead of silently substituting a default that may differ from the YAML
   (REFACTOR_PLAN.md §6 — "values are not re-defaulted").

**Attribute names are sacred** (REFACTOR_PLAN.md §3.1). ``self.se``,
``self.wl_pe_cnn``, ``self.branch_{a,b,c,d}``, ``self.cross_interaction``,
``self.aux_head_{a,b,c,d}``, ``self.embed_net``, ``self.linear_head`` and
``self.arcface_head`` are the top-level keys of all three trained checkpoints;
renaming any of them breaks ``load_state_dict(strict=True)``.

**Construction order is sacred** (REFACTOR_PLAN.md §3.6). Each sub-module's
``_init_weights`` draws from the same global torch RNG stream, so the order in
which they are constructed below determines the initial weights. Do not reorder.
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
from spectralquadnet.models.fusion import CrossModalInteraction, EmbedNet
from spectralquadnet.models.heads import AdaptiveSubcenterArcFaceHead, AuxiliaryHead
from spectralquadnet.models.stats_ops import extract_grid_spectra, masked_spectral_stats

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps hydra out of the model layer
    from spectralquadnet.config.schema import ExperimentConfig


class SpectralQuadNet(nn.Module):
    """
    Four-branch hyperspectral classification model.

    Branches
    ────────
    A  SpectralProfileBranch  — raw signal + learnable derivatives
    B  SpectralStatsBranch    — masked band statistics (5 moments)
    C  SpatialCNNBranch       — 2-D spatial texture
    D  SpecFormerBranch       — spectral patch transformer

    Deep Supervision (Stage 1 only)
    ────────────────────────────────
    Each branch has its own AuxiliaryHead so it is individually
    discriminative before cross-modal fusion.  During inference the
    aux heads are not called (forward returns a plain tensor).

    Heads
    ─────
    Stage 1 : linear_head (CE / Focal, no margin)
    Stage 2+ : arcface_head (Sub-centre ArcFace)
    """

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

        # ── Shared spectral attention ─────────────────────────────────
        self.se = MaskedSpectralECA(num_bands)

        # ── Physical wavelength positional encoding (for 1-D CNN branches) ──
        self.wl_pe_cnn = PhysicalWavelengthPE(physical_wl, tower_ch)

        # ── Four spectral branches ────────────────────────────────────
        self.branch_a = SpectralProfileBranch(
            out_dim=256, tower_ch=tower_ch, wl_pe_module=self.wl_pe_cnn
        )
        self.branch_b = SpectralStatsBranch(
            num_bands=num_bands, out_dim=256, tower_ch=96, wl_pe_module=self.wl_pe_cnn
        )
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(
            physical_wl=physical_wl,
            num_bands=num_bands,
            patch_size=cfg.model.specf_patch,
            stride=cfg.model.specf_patch // 2,
            d_model=cfg.model.specf_dim,
            n_heads=cfg.model.specf_heads,
            n_layers=cfg.model.specf_layers,
            out_dim=256,
            dropout=0.10,
        )

        # ── Cross-modal fusion ────────────────────────────────────────
        self.cross_interaction = CrossModalInteraction(
            num_modalities=4, d=256, drop=cfg.model.fusion_drop
        )

        # ── Auxiliary heads — one per branch (deep supervision) ───────
        aux_hidden = cfg.model.aux_head_hidden
        self.aux_head_a = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_b = AuxiliaryHead(256, aux_hidden, num_classes)
        self.aux_head_c = AuxiliaryHead(256, aux_hidden, num_classes)  # ← branch C
        self.aux_head_d = AuxiliaryHead(256, aux_hidden, num_classes)

        # ── Embedding network ─────────────────────────────────────────
        self.embed_net = EmbedNet(256, 512, dropout)

        # ── Classification heads ──────────────────────────────────────
        self.linear_head = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout * 0.4), nn.Linear(256, num_classes)
        )
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            256,
            num_classes,
            K=cfg.model.subcenter_K,
            s=cfg.stage2.arcface_s,
            m_base=cfg.stage2.arcface_m,
            m_delta=cfg.stage2.arcface_m_delta,
        )
        self._use_arcface = False
        self._init_weights()

    # ── Construction from a composed config ───────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: ExperimentConfig | Any,
        physical_wl: torch.Tensor,
    ) -> SpectralQuadNet:
        """Build the network exactly as the baseline ``main()`` did (lines 2726-2732).

        Kept as the single canonical construction path so the Stage-1 dropout
        value, band count and class count cannot drift between ``train.py``, the
        regression tests and any future evaluation entrypoint.
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
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
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
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def use_arcface(self, flag: bool) -> None:
        self._use_arcface = flag

    def freeze_head(self, which: str) -> None:
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters():
            p.requires_grad_(False)

    def unfreeze_head(self, which: str) -> None:
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters():
            p.requires_grad_(True)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_embed: bool = False,
        arc_m: float | None = None,
        branch_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Run the four branches, fuse them and classify.

        Returns:
            In ``.train()`` mode a dict of the main logits plus the four
            auxiliary heads' logits (and ``"emb"`` when ``return_embed``); in
            ``.eval()`` mode the logits alone, or ``(logits, embedding)`` when
            ``return_embed``. Every caller in ``engine/`` branches on exactly
            this shape, so it is part of the contract, not an implementation
            detail.
        """
        # Apply shared spectral channel attention before branching
        x = self.se(x)

        # Global Stats (preserves statistical integrity for Branch B)
        ms, std, mx, skew, kurt, p10, p25, p75, p90 = masked_spectral_stats(x)

        # Regional Grid Spectra (for Branches A and D)
        grid_ms = extract_grid_spectra(x, grid_size=4)
        B, N, C = grid_ms.shape
        flat_grid_ms = grid_ms.reshape(B * N, C)

        # --- BRANCH A (Spectral Profile) ---
        # Processes 16 regions independently, then averages the embeddings
        ba_grid = self.branch_a(flat_grid_ms)
        ba_raw = ba_grid.view(B, N, -1).mean(dim=1)

        # --- BRANCH B (Spectral Stats) ---
        # Processes global seed
        bb_raw = self.branch_b(ms, std, mx, skew, kurt, p10, p25, p75, p90)

        # --- BRANCH C (Spatial CNN) ---
        # Processes full 64x64 cube
        bc_raw = self.branch_c(x)

        # --- BRANCH D (SpecFormer) ---
        # Processes 16 regions independently, then averages the embeddings
        bd_raw = self.branch_d(grid_ms)

        # --- Branch Masking (Deep Supervision Dropout) ---
        if branch_mask is not None:
            ba = ba_raw * branch_mask[0]
            bb = bb_raw * branch_mask[1]
            bc = bc_raw * branch_mask[2]
            bd = bd_raw * branch_mask[3]
        elif self.training:
            drop_probs = torch.tensor([0.0, 0.0, 0.30, 0.20], device=ba_raw.device)
            keeps = (torch.rand(4, device=ba_raw.device) > drop_probs).float()
            safe_idx = torch.randint(0, 4, (), device=ba_raw.device)
            safe_mask = F.one_hot(safe_idx, num_classes=4).float()
            keeps = torch.maximum(keeps, safe_mask)
            ba = ba_raw * keeps[0]
            bb = bb_raw * keeps[1]
            bc = bc_raw * keeps[2]
            bd = bd_raw * keeps[3]
        else:
            ba, bb, bc, bd = ba_raw, bb_raw, bc_raw, bd_raw

        # --- Fusion ---
        joint_token = self.cross_interaction([ba, bb, bc, bd])
        emb = self.embed_net(joint_token)

        if self._use_arcface:
            logits = self.arcface_head(F.normalize(emb, dim=1), labels, global_m=arc_m)
        else:
            logits = self.linear_head(emb)

        if self.training:
            out = {
                "main": logits,
                "aux_a": self.aux_head_a(ba_raw),
                "aux_b": self.aux_head_b(bb_raw),
                "aux_c": self.aux_head_c(bc_raw),
                "aux_d": self.aux_head_d(bd_raw),
            }
            if return_embed:
                out["emb"] = F.normalize(emb, dim=1)
            return out

        if return_embed:
            return logits, F.normalize(emb, dim=1)
        return logits  # type: ignore[no-any-return]  # head is `nn.Module.__call__` -> Any
