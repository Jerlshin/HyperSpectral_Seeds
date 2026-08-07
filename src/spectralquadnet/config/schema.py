"""Structured config schemas for SpectralQuadNet.

Every experiment knob is a typed dataclass field, registered with Hydra's
``ConfigStore`` so a malformed or missing key in ``configs/`` fails at
composition time rather than at some arbitrary point deep in training.
``scripts/check_config_roundtrip.py`` cross-checks every field against its
YAML value.

Design notes
────────────
* **YAML is the single source of truth for values.** Every value-carrying field
  defaults to ``omegaconf.MISSING`` so that a key missing from ``configs/`` fails
  loudly at composition time rather than silently falling back to a default that
  drifts from the YAML. :class:`TrackingConfig` is the exception, since its
  fields are genuinely optional (a run with no tracking backend configured is
  a valid, common case) and carries real defaults.
* **Field names follow a consistent prefix-per-group convention** (e.g.
  ``stage1.max_lr``, ``stage2.sgdr_T0``) rather than a flat namespace, so a
  field's owning group is always unambiguous from its path alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ══════════════════════════════════════════════════════════════════════
#  DATA  (CONFIG: patches_data … noise_std)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class DataConfig:
    """Dataset location, geometry and input-perturbation strengths.

    ``max_cutout_bands``/``noise_std`` group here rather than under
    :class:`ModelConfig` because they are read exclusively by
    ``RiceSeedDataset``'s augmentation primitives — they parameterise the
    input pipeline, not the model architecture.
    """

    patches_data: str = MISSING
    labels_path: str = MISSING
    wavelength_path: str = MISSING
    num_bands: int = MISSING
    num_classes: int = MISSING
    max_cutout_bands: int = MISSING
    noise_std: float = MISSING


# ══════════════════════════════════════════════════════════════════════
#  MODEL  (CONFIG: branch_drop_prob, subcenter_K, aux_head_hidden, wl_*, specf_*, fusion_*)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ModelConfig:
    """Architecture knobs consumed by ``SpectralQuadNet.__init__``.

    ``wl_embed_dim`` is accepted by the model constructor but currently
    unused in its body; kept in the schema for forward compatibility with
    the constructor's signature.
    """

    branch_drop_prob: float = MISSING
    subcenter_K: int = MISSING
    aux_head_hidden: int = MISSING
    wl_embed_dim: int = MISSING
    specf_patch: int = MISSING
    specf_dim: int = MISSING
    specf_heads: int = MISSING
    specf_layers: int = MISSING
    specf_drop: float = MISSING
    fusion_heads: int = MISSING
    fusion_drop: float = MISSING


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — 3-phase progressive augmentation  (CONFIG: s1_*, aux_loss_weight_*)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Stage1Config:
    """Stage 1: progressive augmentation curriculum with deep supervision.

    ``p3_*`` fields apply to Phase 3 only (hard-class oversampling + contrastive
    auxiliaries); ``aux_loss_weight_{init,final}`` drive the ``_aux_loss_weight``
    decay schedule used by the per-branch auxiliary heads.
    """

    epochs: int = MISSING
    phase1_frac: float = MISSING
    phase2_frac: float = MISSING
    batch: int = MISSING
    max_lr: float = MISSING
    mid_lr: float = MISSING
    min_lr: float = MISSING
    dropout: float = MISSING
    mixup: float = MISSING
    patience: int = MISSING
    accum: int = MISSING
    focal_gamma: float = MISSING
    label_smooth_hi: float = MISSING
    label_smooth_lo: float = MISSING
    ema_reinit_phases: bool = MISSING

    # ── Phase 3 · contrastive auxiliaries ─────────────────────────────
    p3_supcon_weight: float = MISSING
    p3_proto_weight: float = MISSING

    # ── Phase 3 · hard-class oversampling ─────────────────────────────
    p3_oversample: bool = MISSING
    p3_oversample_power: float = MISSING
    p3_oversample_max_w: float = MISSING
    p3_hard_f1_thresh: float = MISSING
    p3_oversample_eps: float = MISSING

    # ── Phase 3 · dropout boost ───────────────────────────────────────
    p3_dropout: float = MISSING

    # ── Auxiliary-head loss weighting (decays init → final) ───────────
    aux_loss_weight_init: float = MISSING
    aux_loss_weight_final: float = MISSING


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 — sub-centre ArcFace + SGDR  (CONFIG: s2_*, cdws_*, supcon_*, proto_*, bal_*)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Stage2Config:
    """Stage 2: sub-centre ArcFace with SGDR, SupCon/ProtoNCE and CDWS.

    ``cdws_*`` live under this group even though they are also read from
    Stage 1 (``compute_class_difficulty``) and Stage 3 (contrastive
    auxiliaries) code paths, since Stage 2 is where CDWS-weighted sampling
    is the primary batch-composition strategy.
    """

    epochs: int = MISSING
    batch: int = MISSING
    head_lr: float = MISSING
    back_lr: float = MISSING
    min_lr: float = MISSING
    warmup_ep: int = MISSING
    sgdr_T0: int = MISSING
    sgdr_Tmult: int = MISSING
    dropout: float = MISSING
    patience: int = MISSING

    # ── Sub-centre ArcFace (also read at model construction time) ─────
    arcface_s: float = MISSING
    arcface_m: float = MISSING
    arcface_m0: float = MISSING
    arcface_m_delta: float = MISSING
    margin_warmup_ep: int = MISSING
    focal_gamma: float = MISSING

    # ── Class-Difficulty-Weighted Sampling ────────────────────────────
    cdws_max_weight: float = MISSING
    cdws_eps: float = MISSING

    # ── Contrastive auxiliaries ───────────────────────────────────────
    supcon_weight: float = MISSING
    supcon_temp: float = MISSING
    proto_weight: float = MISSING
    proto_temp: float = MISSING

    # ── Class-balanced batch composition ──────────────────────────────
    bal_n_cls: int = MISSING
    bal_n_spc: int = MISSING


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 — SAM + greedy SWA  (CONFIG: s3_*)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Stage3Config:
    """Stage 3: Sharpness-Aware Minimisation with greedy SWA snapshotting."""

    epochs: int = MISSING
    swa_lr: float = MISSING
    cycle_len: int = MISSING
    sam_rho: float = MISSING
    greedy: bool = MISSING
    aux_loss_weight: float = MISSING


# ══════════════════════════════════════════════════════════════════════
#  TRACKING
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TrackingConfig:
    """Experiment-tracking backend selection.

    ``show_diagnostics`` and ``log_grad_norms`` gate the per-branch
    diagnostics: gradient norms *are* always computed and sent to the
    structured backends (W&B/TensorBoard) when tracking is enabled, but the
    console stays quiet about them unless ``show_diagnostics`` is set, to
    keep the default console output readable.
    """

    backend: str = "console"
    project: str | None = None
    entity: str | None = None
    log_dir: str | None = None
    watch_model: bool = False
    backends: list[str] = field(default_factory=list)  # used by backend == "multi"

    # ── Per-branch diagnostics ──────────────────────────────────────────
    #: Compute per-branch gradient norms each optimiser step and log the epoch
    #: mean. Costs one extra pass over ``named_parameters()`` per step.
    log_grad_norms: bool = True
    #: Echo scalar diagnostics to the console backend as well as to the
    #: structured backends. Off by default — see ``console_tracker`` docstring.
    show_diagnostics: bool = False


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT — composition root
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentConfig:
    """Top-level composed config.

    Root-level fields (run identity, shared training knobs, TTA, seed,
    device) are the settings that don't belong to any single stage or
    component and so have no group of their own.

    Two notable field shapes:

    * ``output_dir`` is not set directly; it is interpolated from the
      ``output_root``/``run_name`` pair in the composed YAML.
    * ``device`` is a resolution *strategy* string (``"auto"``/``"cuda"``/
      ``"cpu"``/``"mps"``) rather than a live ``torch.device`` object, since
      YAML cannot hold one — ``utils/device.py`` resolves it at runtime.
    """

    data: DataConfig = MISSING
    model: ModelConfig = MISSING
    stage1: Stage1Config = MISSING
    stage2: Stage2Config = MISSING
    stage3: Stage3Config = MISSING
    tracking: TrackingConfig = MISSING

    # ── Run identity & output location ────────────────────────────────
    run_name: str = MISSING
    output_root: str = MISSING
    output_dir: str = MISSING

    # ── Shared training knobs ─────────────────────────────────────────
    weight_decay: float = MISSING
    grad_clip: float = MISSING
    ema_decay: float = MISSING

    # ── Test-time augmentation ────────────────────────────────────────
    tta_spatial: int = MISSING
    tta_spectral: int = MISSING

    # ── Reproducibility & placement ───────────────────────────────────
    device: str = MISSING
    seed: int = MISSING


# ══════════════════════════════════════════════════════════════════════
#  ConfigStore registration
# ══════════════════════════════════════════════════════════════════════


def register_configs() -> None:
    """Register every schema with Hydra's ``ConfigStore``.

    Group schemas are stored as ``<group>/base_<group>`` so each YAML in that
    group can validate itself by listing ``- base_<group>`` in its ``defaults``.
    The top-level ``ExperimentConfig`` is stored as ``base_experiment``.
    """
    cs = ConfigStore.instance()
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="stage1", name="base_stage1", node=Stage1Config)
    cs.store(group="stage2", name="base_stage2", node=Stage2Config)
    cs.store(group="stage3", name="base_stage3", node=Stage3Config)
    cs.store(group="tracking", name="base_tracking", node=TrackingConfig)
    cs.store(name="base_experiment", node=ExperimentConfig)
