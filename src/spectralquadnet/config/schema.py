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
    #: FE-2 / T3-7. ``(N, S, S)`` fp16 fill map alpha, written by
    #: ``scripts/prepare_dataset.py`` under P-3. When set **and present**, the
    #: four masked modules take the mask as an input instead of re-deriving it
    #: from ``sum_c |x_c| > 1e-5``; an empty string keeps that threshold, which
    #: is the exact pre-Tier-3 behaviour and what lets the archived arrays
    #: reproduce.
    masks_path: str = MISSING
    #: FU-4 / P-4. ``(N, 8)`` morphometrics. Consumed by Branch B's descriptor
    #: and by the fifth fusion token. Empty substitutes zeros.
    morphology_path: str = MISSING
    num_bands: int = MISSING
    num_classes: int = MISSING
    max_cutout_bands: int = MISSING
    noise_std: float = MISSING

    # ── Split protocol (P-1, P-5 / T4-1, T4-5) ────────────────────────
    #: ``(N,)`` int64 scan id per patch, written by ``scripts/prepare_dataset.py``.
    #: Required by the grouped scheme; under ``stratified`` it is read when
    #: present purely to *measure* how many scans cross the train/eval boundary.
    groups_path: str = MISSING
    #: ``stratified`` — the pre-Tier-4 patch-level split, which puts every scan
    #: in all three splits (0-H measured 107/107); ``grouped`` — P-1's
    #: scan-disjoint split.
    split_scheme: str = MISSING
    #: Share of each class's data (groups, under ``grouped``) held out for
    #: ``val ∪ test``. 0.30 is the 70/15/15 the reference run used.
    split_eval_frac: float = MISSING
    #: Rotates which scans are held out. Sweeping ``0 … m-1`` is the
    #: leave-one-scan-out cross-validation §3.1 P-1 falls back to. Must stay 0
    #: under ``stratified``, which has no groups to rotate.
    split_fold: int = MISSING
    #: P-5 / T4-5. Share of the **training pool** carved off as ``calib``, the
    #: split the per-class margins, CDWS weights and Phase-3 oversampling
    #: weights are fitted on. 0.0 leaves them on ``val``, i.e. fitted on the
    #: split that also selects the checkpoint (C-9). The plan's 60/10/15/15
    #: is ``split_eval_frac=0.30`` with ``calib_frac ≈ 0.143``.
    calib_frac: float = MISSING
    #: IC-3 / CHANGES §19.1. What ``grouped`` does about a class captured in a
    #: single scan: ``error`` refuses and names them, ``patch_split`` accepts a
    #: patch-level split for those classes and records the leak in the report.
    #: The protocol section of the paper specifies ``error`` — "refuse to
    #: silently accept a leak".
    single_group_policy: str = MISSING
    #: IC-14. ``(N, 2, S, S)`` per-pixel ``(mean, sd)`` along λ, written by
    #: ``scripts/prepare_dataset.py``. **Never an input to any model**, and that
    #: is deliberate rather than an oversight: it is the per-pixel brightness
    #: the SNV divided out, which is also the strongest single carrier of
    #: acquisition-bundle identity (CHANGES §3.3). It is consumed by
    #: ``spectralquadnet.experiments.leakage``, which uses it to *measure* how
    #: much bundle identity the pipeline could exploit. Empty disables that
    #: probe; nothing else reads the key.
    gain_path: str = MISSING

    # ── Same-class CutMix geometry (OP-6 / T2-7) ──────────────────────
    #: Width, in bands, of the contiguous wavelength window swapped between
    #: two seeds of the same class. Label-preserving, so no soft target is
    #: needed and the ArcFace/mixup exclusion does not apply.
    cutmix_bands: int = MISSING
    #: Side length, in pixels, of the square region pasted from another seed
    #: of the same class.
    cutmix_spatial: int = MISSING


# ══════════════════════════════════════════════════════════════════════
#  MODEL  (CONFIG: branch_drop_prob, subcenter_K, aux_head_hidden, wl_*, specf_*, fusion_*)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ModelConfig:
    """Architecture knobs consumed by the model constructors.

    Every key here is read on the forward path and
    ``tests/unit/test_config_wiring.py`` proves it by perturbing each one and
    watching the model react — the §4.3 gate that would have caught all five
    dead paths of §2.7 automatically.

    Since CHANGES IC-10 this group serves **two** architectures, selected by
    :attr:`arch`. Keys belonging to a branch the selected architecture does not
    have are simply unread by it; the wiring test excuses them per-architecture
    rather than globally, so a key that is dead in *both* is still caught.
    """

    #: IC-10. Which network :func:`~spectralquadnet.models.registry.build_model`
    #: constructs.
    #:
    #: * ``spectral_quadnet`` — the audited four-branch v4 model (5.19 M
    #:   parameters). Retained unchanged because ablations A3 and A8 need it as
    #:   the control arm; removing it would make the removals unfalsifiable.
    #: * ``spectral_seed_net`` — CHANGES §16.2. Two pathways, concat + MLP,
    #:   K=1 head, one auxiliary head. ≈2.8 M parameters, ≈1.4 GFLOP.
    arch: str = MISSING

    branch_drop_prob: float = MISSING
    #: A3 / CHANGES §5.2. Per-branch drop *ratio* for ``(A, B, C, D)``, scaled
    #: by :attr:`branch_drop_prob`. The audited run used ``(0.75, 0.75, 0, 0.75)``
    #: — Branch C never dropped, the other three absent 15% of the time — which
    #: did not regularise the fusion gate so much as teach it that three of its
    #: four inputs were unreliable. Branch C's 87% influence is therefore
    #: confounded and cannot be read as "C is intrinsically best". The default
    #: is now **symmetric**, so A3 measures the branches rather than the policy.
    branch_drop_profile: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    #: A3. Which of the four branches are constructed at all. Dropping a branch
    #: here removes its parameters, its auxiliary head and its fusion modality,
    #: so a ``["b", "c"]`` arm is genuinely the smaller model rather than the
    #: full one with a tensor multiplied by zero.
    enabled_branches: list[str] = field(default_factory=lambda: ["a", "b", "c", "d"])
    #: A5 / CHANGES §5.3. How the modality tokens are combined.
    #: ``bilinear_gate`` is the audited rank-128 second-order pool over all ten
    #: pairs; ``gate`` keeps the sigmoid gate and drops the bilinear term;
    #: ``concat_mlp`` is the §16.2 replacement.
    fusion_mode: str = MISSING
    subcenter_K: int = MISSING
    #: Soft-to-hard sub-centre assignment temperature (HD-2(i) / T2-9). The
    #: class cosine is ``tau * logsumexp(cos_k / tau)``, annealed from
    #: ``subcenter_tau_init`` to ``subcenter_tau_final`` across a stage; at
    #: ``tau -> 0`` it recovers the hard ``max_k`` exactly.
    subcenter_tau_init: float = MISSING
    subcenter_tau_final: float = MISSING
    #: Weight on the mixture-of-experts load-balancing term
    #: ``sum_c KL(pi_c || uniform)`` that keeps sub-centres from dying (HD-2(ii)).
    subcenter_balance_weight: float = MISSING
    aux_head_hidden: int = MISSING
    #: Fourier-feature width of FE-1's continuous-λ kernel generator and of
    #: BR-4's relative-λ attention bias. Dead in the reference implementation
    #: (N-1c); §3.2 FE-1 nominates it as κ_φ's width rather than deleting it,
    #: and T3-5 wires it there.
    wl_embed_dim: int = MISSING
    #: BR-4(iii). Token count is ``num_bands // (specf_patch // 2)``, so the
    #: shipped 8 gives the 10 λ-uniform windows the old index stride of 4
    #: produced — on a wavelength-uniform axis instead of an index one.
    specf_patch: int = MISSING
    specf_dim: int = MISSING
    specf_heads: int = MISSING
    specf_layers: int = MISSING
    #: Wired to Branch D's transformer dropout by T3-3. Dead before that
    #: (N-1b): the branch was constructed with a hardcoded 0.10.
    specf_drop: float = MISSING
    fusion_drop: float = MISSING

    # ── Tier 3 · architectural redesign (IMPROVEMENT_PLAN §3.2-§3.4) ───
    #: BR-2 / T3-6. Branch A's grid. 4x4 over a 64x64 patch is a 256:1 spatial
    #: compression (§2.2.3); 8x8 makes it 64:1 at no parameter cost, since the
    #: branch processes cells independently.
    grid_size_a: int = MISSING
    #: Branch D's grid, kept at 4x4. A and D no longer share an input, so they
    #: no longer need to share a grid — which is half of what closes C-2.
    grid_size_d: int = MISSING
    #: BR-1(i). Number of learned normalised-difference indices. Each costs
    #: ``2 * num_bands`` parameters and is exactly invariant to the per-pixel
    #: and per-session gain that made the nine moments rank-2 (§2.2.5).
    index_bank_size: int = MISSING
    #: BR-1(ii). How many of the deepest continuum-removed absorption features
    #: Branch B reads.
    continuum_depths: int = MISSING
    #: FU-4 / P-4. Width of the persisted morphometric vector — the eight
    #: columns ``data/prep/segmentation.py::MORPHOMETRIC_NAMES`` writes.
    n_morphometrics: int = MISSING
    #: BR-3. Channel width the 3-D stem folds the spectral axis into before the
    #: 2-D ResBlock/CBAM tail. The spectral axis is folded, not deleted: each of
    #: the ``64 * depth`` channels entering the fold is a (spectral position x
    #: feature) pair.
    stem_channels: int = MISSING
    #: FU-1(b). Rank ``r`` of the bilinear projections ``U_m``. A full bilinear
    #: pool over five modalities would be ``10 * d ** 2``; this is ``5 * d * r``.
    fusion_rank: int = MISSING
    #: FU-1(b)/FU-2. Hidden width of the gate MLP, which reads the five
    #: normalised tokens *and* the five pre-normalisation log-norms.
    fusion_gate_hidden: int = MISSING

    # ── SpectralSeedNet (IC-10 / CHANGES §16.2) ───────────────────────
    #: A10. Multiplier on the spatial path's ResBlock widths. 1.0 is the
    #: §16.2 network; A10 sweeps {0.5, 0.75, 1.0, 1.5} to answer "is 5.19 M too
    #: big" with data instead of intuition.
    spatial_width_mult: float = MISSING
    #: Hidden width of the spectral path's MLP over
    #: ``[index bank ‖ continuum depths ‖ SNV(x̄) ‖ D₁ ‖ D₂ ‖ morph]``.
    spectral_hidden: int = MISSING
    #: IC-5 / CHANGES §7.1. Fixed weight on the single auxiliary head attached
    #: to the spatial path. Fixed, not scheduled and not GradNorm-controlled:
    #: four heads under a saturating controller made the auxiliary term ≈7.8×
    #: the main classification loss at epoch 20, so the fused head — the only
    #: path that produces an evaluation logit — was ≈11% of the gradient.
    aux_head_weight: float = MISSING

    # ── Head elaborations, off by default (IC-9, gated by A7) ─────────
    #: Whether ``update_margins_from_pr`` is called at all. The signed rule is
    #: better reasoned than the usual F1-driven one, but Stage 2's best epoch
    #: was 19 and the per-class vector took over at 21 — it was never active at
    #: the selected checkpoint (CHANGES §5.4). Kept in code, off by default,
    #: measured by A7.
    per_class_margin: bool = MISSING
    #: Whether the pairwise confusion penalty ``-δ·Ω[y, c]`` is fitted and
    #: applied. Fitted on the selection split in the audited run, never
    #: ablated, aimed at hard classes that never moved. A7 arm 4.
    pairwise_penalty: bool = MISSING


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

    # ── Unified head (HD-1 / T2-10) ───────────────────────────────────
    #: Stage 1's angular margin. HD-1 runs one head for all three stages;
    #: at ``m = 0`` it is a plain cosine (NormFace) classifier, which is what
    #: makes the Stage-1 → Stage-2 transition continuous and keeps mixup
    #: admissible in Phases 1-2.
    arcface_m: float = MISSING


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

    # ── Signed precision/recall margin rule (HD-3 / T2-8) ─────────────
    #: ``M(c) = clip(arcface_m + arcface_m_delta * (R_c - P_c), min, max)``.
    #: ``arcface_m_delta`` above is that rule's ``m_delta``; it kept its name
    #: and changed its value (0.10 → 0.20) when the F1-driven rule it used to
    #: parameterise was replaced. The sign is the whole point: an additive
    #: angular margin *shrinks* the margined class's region, so a low-recall
    #: class must have its margin **lowered**, which the old rule got
    #: backwards (M-6).
    arcface_m_min: float = MISSING
    arcface_m_max: float = MISSING
    #: ``delta`` in the pairwise term ``-1[c != y] * delta * Omega[y, c]``,
    #: which pushes a class away from the classes it is actually confused
    #: with instead of from all 89 others uniformly.
    pairwise_margin_delta: float = MISSING

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

    # ── Margin annealing (OP-4.2/4.3 / T2-1) ──────────────────────────
    #: Stage 3 keeps Stage 2's per-class margin *vector* and scales the whole
    #: of it by ``kappa: 1.0 -> margin_kappa_final``, updated only at cycle
    #: boundaries so every step between two SWA snapshots optimises one
    #: function. The scalar schedule it replaces discarded the calibration.
    margin_kappa_final: float = MISSING

    # ── SWA transient rejection (OP-4.5 / T2-3) ───────────────────────
    #: Cycles discarded from the SWA average before the first candidate is
    #: even considered, keeping Adam's second-moment warm-up transient out of
    #: the average. ``ceil((1/(1-beta2)) / steps_per_cycle)`` = 3 for the
    #: shipped loader.
    swa_warmup_cycles: int = MISSING

    # ── ASAM (OP-5 / T2-4) ────────────────────────────────────────────
    #: Element-wise normalisation of the SAM perturbation by ``|theta|``.
    #: SAM's rho-ball is not scale-invariant while the ArcFace head is, so a
    #: raw-space budget is spent flattening the classifier rather than the
    #: representation.
    sam_adaptive: bool = MISSING


# ══════════════════════════════════════════════════════════════════════
#  SINGLE STAGE — the collapsed curriculum (IC-11 / CHANGES §17)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SingleStageConfig:
    """One stage, one objective, one schedule.

    Stages 2 and 3 consumed 65% of the audited run's 18.7-hour wall clock and
    moved validation macro-F1 by +0.005 — 6.5 samples of a 1,294-sample split,
    against a ±0.020 sampling CI (CHANGES §9.3). The *only* thing Stage 2 did
    that Stage 1 did not was introduce a non-zero angular margin, and a margin
    is incompatible with mixup, which is why they were separated in the first
    place.

    This group achieves the same transition inside one stage: mixup runs to
    :attr:`mixup_epochs`, then a single global margin warms in over
    :attr:`margin_warmup_start`…:attr:`margin_warmup_end`. One optimiser state,
    one schedule, no EMA re-initialisation, ~45 min/run instead of 19 h — which
    is the real prize, because it converts a project that can afford one run
    into one that can afford forty.

    The three-stage modules are **not** deleted; ``pipeline=three_stage``
    still reaches them, because A8 is the experiment that decides whether this
    collapse was correct.
    """

    epochs: int = MISSING
    batch: int = MISSING
    accum: int = MISSING
    patience: int = MISSING

    # ── Optimiser ─────────────────────────────────────────────────────
    max_lr: float = MISSING
    min_lr: float = MISSING
    warmup_ep: int = MISSING
    dropout: float = MISSING

    # ── Objective ─────────────────────────────────────────────────────
    label_smooth_hi: float = MISSING
    label_smooth_lo: float = MISSING
    #: 0.0 is plain CE. CHANGES §7.4: focal γ addresses 1000:1 foreground/
    #: background imbalance; here the imbalance is 96:91, so γ>0 is
    #: down-weighting easy examples rather than correcting anything. Ablate
    #: (A11-adjacent) before re-adding.
    focal_gamma: float = MISSING
    aux_loss_weight: float = MISSING

    # ── Regularisation ────────────────────────────────────────────────
    #: Mixup α. The one demonstrably load-bearing regulariser in the audited
    #: run: switching it off moved training accuracy 42% → 96.6% in a single
    #: epoch while validation did not move at all (CHANGES §5.5).
    mixup: float = MISSING
    #: Last epoch mixup is active. Mixup and a non-zero margin are mutually
    #: exclusive by construction, so this is also when the margin may start.
    mixup_epochs: int = MISSING
    #: Single augmentation profile for the whole run. The three-phase
    #: curriculum's profiles differed by 2–4 percentage points of trigger
    #: probability; the only real transition it encoded was the mixup switch.
    aug_profile: str = MISSING

    # ── Head schedule ─────────────────────────────────────────────────
    #: Target global margin. Warmed 0 → this over the window below. At 0 the
    #: head is NormFace, which is what Stage 1 already ran.
    arcface_m: float = MISSING
    arcface_s: float = MISSING
    margin_warmup_start: int = MISSING
    margin_warmup_end: int = MISSING

    # ── Optional Phase B — only if A6 earns it (CHANGES §17) ──────────
    #: Epochs of ``CE + supcon_weight · SupCon`` appended after the main run,
    #: with a class-balanced sampler. 0 disables it, which is the default until
    #: A6 reports a gain exceeding run-to-run variance.
    supcon_epochs: int = MISSING
    supcon_weight: float = MISSING
    supcon_temp: float = MISSING
    #: Class-balanced batch composition for the SupCon phase: ``n_cls × n_spc``.
    bal_n_cls: int = MISSING
    bal_n_spc: int = MISSING


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION — what is selected on, and what is reported (CHANGES §19)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class EvaluationConfig:
    """Which split selects the checkpoint and which one is scored at the end.

    This group exists because the audited run's single largest statistical
    defect was that these were the same split. ``calib_frac=0.0`` put the
    per-class margins, the confusion matrix, the CDWS weights *and* the Phase-3
    oversampling weights on ``val``, then selected the checkpoint on ``val``,
    then reported the number from ``val`` (CHANGES §4.4). With ~472 epochs ×
    {live, EMA} that is a maximum over ~944 correlated draws, worth an expected
    +0.042 macro-F1 of pure selection bias — an order of magnitude more than
    everything Stages 2 and 3 produced.
    """

    #: Split the per-epoch checkpoint decision reads. ``calib`` is the protocol
    #: CHANGES §19.1 specifies; ``val`` reproduces the audited behaviour and is
    #: kept so the two can be compared rather than argued about.
    select_split: str = "calib"
    #: Split scored once, after every design decision is frozen. ``val_test``
    #: is §19.1's rule for the grouped protocol, where val and test are two
    #: halves of one held-out bundle and are therefore *not* independent of
    #: each other: they must be treated as one held-out set. ``test`` is the
    #: stratified arm's convention.
    report_split: str = "val_test"
    #: Score with and without the 12-view TTA, reporting both separately.
    tta: bool = True
    #: Bootstrap resamples behind the reported metric's confidence interval.
    #: 0 disables. Sampling noise on a ~1,300-patch split is ±0.020 at 95%, and
    #: a delta quoted without it is not interpretable (CHANGES §4.5).
    bootstrap_samples: int = 2000
    #: Write the 90×90 confusion matrix, the per-class table and the run's
    #: metric JSON under ``output_dir/results/``.
    save_artifacts: bool = True


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
#  RUNTIME — execution knobs that must not change a single number
# ══════════════════════════════════════════════════════════════════════


@dataclass
class RuntimeConfig:
    """How the run *executes*: worker counts, kernel selection, device topology.

    Every field here is a **throughput** knob, and the invariant that separates
    this group from every other one is that changing it must not change a
    reported metric. That is why it carries real defaults instead of
    :data:`omegaconf.MISSING` (the same exemption :class:`TrackingConfig` has):
    the values do not belong to the experiment, so a config that never mentions
    them is not under-specified.

    Two fields deliberately default to the *slower* setting, because the fast
    one trades arithmetic for time and this pipeline's contract is that it does
    not: :attr:`allow_tf32` drops matmul mantissas from 24 bits to 11, and
    :attr:`channels_last` re-selects convolution algorithms whose reduction
    order differs. Both are one flag away for anyone who wants them.

    :attr:`amp_dtype` is the one field here that admits to changing numerics
    rather than defaulting away from it, because the alternative was worse:
    Stage 1 trains under autocast either way, and the fp16 this group used to
    inherit from torch's per-device default is what put ``NaN`` in the loss.
    The knob exists so the choice is *recorded and overridable* instead of
    implicit — which is the same reason the two fields above exist.

    ``-1`` means "decide from the hardware" wherever a count is expected;
    :func:`~spectralquadnet.utils.device.resolve_runtime` is the one place that
    resolution happens.
    """

    # ── Data pipeline ─────────────────────────────────────────────────
    #: DataLoader worker processes. ``-1`` auto-selects: half the physical
    #: cores (capped at 8) on CUDA, where the host feed has to keep up with a
    #: T4/A100/H100; ``2`` on Metal and CPU, where the accelerator is the
    #: bottleneck and worker start-up under ``spawn`` is not free.
    num_workers: int = -1
    #: Page-locked staging buffers, so the H2D copy is a DMA that overlaps
    #: compute. ``-1`` auto-enables on CUDA only — there is no pinned-memory
    #: path on Metal, and asking for one raises.
    pin_memory: int = -1
    #: Keep workers alive between epochs. Under ``spawn`` (macOS default) a
    #: worker costs seconds to start, and Stage 1 restarts its loader 600
    #: times. ``-1`` follows ``num_workers > 0``.
    persistent_workers: int = -1
    #: Batches each worker runs ahead. 4 covers a stalled page-in from the
    #: 5.6 GB mmap without materially growing resident memory.
    prefetch_factor: int = 4
    #: Worker count for the evaluation loaders, which are built and discarded
    #: far more often than the training ones. ``-1`` follows ``num_workers``
    #: capped at 4.
    eval_num_workers: int = -1

    # ── Kernel and graph selection ────────────────────────────────────
    #: ``auto`` compiles on CUDA and leaves Metal in eager mode — measured on
    #: this model, ``torch.compile`` on the MPS backend is **2.2× slower** than
    #: eager (983 ms vs 437 ms per forward at batch 32), because the Metal
    #: inductor backend cannot yet fuse the 3-D stem. ``on``/``off`` force it.
    compile: str = "auto"
    #: Passed to ``torch.compile(backend=...)``.
    compile_backend: str = "inductor"
    #: Passed to ``torch.compile(mode=...)``.
    compile_mode: str = "default"
    #: NHWC/NDHWC tensors. cuDNN's Tensor Core convolution kernels want them,
    #: but they reduce in a different order, so this is opt-in.
    channels_last: bool = False
    #: TF32 matmuls on Ampere and later. **Off by default**: it is a precision
    #: change, and Turing (sm_75, the T4) has no TF32 path at all.
    allow_tf32: bool = False
    #: Autocast dtype for the AMP stages. ``auto`` resolves to **bf16** on any
    #: device that can run it and fp16 only where none can; ``bf16``/``fp16``
    #: force one; ``off`` trains in fp32. bf16 spends 13 mantissa bits to buy
    #: fp32's exponent range, which is the trade that matters here — fp16
    #: overflows at 65 504 and underflows at 6.1e-5, and Stage 1 hit both.
    #: Under bf16 the ``GradScaler`` is constructed disabled, since loss
    #: scaling only ever existed to keep fp16 gradients above that floor. See
    #: :func:`~spectralquadnet.utils.device.resolve_amp_dtype`.
    amp_dtype: str = "auto"
    #: cuDNN convolution autotuning. Already what ``set_seed`` leaves set; here
    #: so it is visible and overridable.
    cudnn_benchmark: bool = True
    #: ``auto`` uses AdamW's fused multi-tensor kernel on CUDA and its foreach
    #: path elsewhere. The fused kernel folds the whole step into one launch;
    #: it accumulates in the same precision but not in the same order, so it is
    #: the one default here that is not bit-exact against eager AdamW.
    fused_optimizer: str = "auto"
    #: Evaluate Branch C's three ``Conv3d`` stages as stacks of ``Conv2d``
    #: calls. Identical arithmetic in a different summation order (1.9e-07 on
    #: the logits) but a different *kernel*, and Metal's ``Conv3d`` backward is
    #: a bad one: **3.12× on the stem, 2.12× on the whole step** (2103 → 994 ms
    #: at batch 32), for 9% more activation memory. ``auto`` enables it on
    #: Metal only — cuDNN's 3-D kernels have no such defect, and
    #: ``torch.compile`` would rather be handed the fused operator.
    decompose_conv3d: str = "auto"
    #: Recompute Branch A's three towers in the backward pass instead of
    #: storing their activations: **2.13× less activation memory** (4054 →
    #: 1901 MB at batch 32) for 4.8% of step time, with bit-exact gradients —
    #: the towers hold no dropout, no BatchNorm and nothing that reads the RNG.
    #: ``auto`` enables it on Metal, where activations come from the same
    #: unified pool as the host's RAM and the failure mode is silent swapping
    #: rather than a clean OOM. Off by default on CUDA, where dedicated VRAM
    #: makes the recompute a cost rather than a rescue — turn it ``on`` there
    #: for a small card or a large ``stage1.batch``.
    #:
    #: Those figures are batch 32's, and they undersell it badly at the batch
    #: this config actually trains at. Measured on an M5 (16 GB, 12.7 GB working
    #: set) at ``stage1.batch = 128``: **49.6 ms/sample holding 9.4 GB with it
    #: on, against 780.1 ms/sample holding 14.6 GB with it off** — 15.7× slower,
    #: because without the recompute the step's working set clears the ceiling
    #: and the run pages. At this batch size the flag is not a memory
    #: optimisation, it is the thing that makes the batch runnable; treat
    #: turning it off on Metal as a decision to lower the batch too.
    checkpoint_branch_a: str = "auto"

    # ── Device topology ───────────────────────────────────────────────
    #: ``auto`` uses every visible CUDA device when the process was launched by
    #: ``torchrun``; ``ddp`` demands it and raises otherwise; ``off`` pins the
    #: run to one device. There is no DataParallel option — see
    #: :mod:`spectralquadnet.utils.distributed` for why single-process
    #: replication cannot hold the BatchNorm statistics invariant.
    multi_gpu: str = "auto"
    #: Convert every BatchNorm to SyncBatchNorm under DDP. This is what makes
    #: multi-GPU training *numerically* the same run as single-GPU: without it
    #: each replica normalises by its own shard and the fusion's five
    #: ``BatchNorm1d`` layers see a different mean than they would have.
    #: Turning it off is faster and no longer invariant.
    sync_batchnorm: bool = True
    #: NCCL/gloo rendezvous timeout, seconds.
    dist_timeout_s: int = 1800

    # ── Memory ────────────────────────────────────────────────────────
    #: Release the caching allocator's free blocks every N epochs. 0 disables
    #: the periodic sweep; the stage boundaries always sweep regardless, since
    #: that is where Stage 3's two extra model copies are freed.
    empty_cache_interval: int = 0

    # ── Console ───────────────────────────────────────────────────────
    #: Whether the per-epoch line is rendered. ``off`` suppresses it (banners,
    #: notices and diagnostic blocks still print); every other value renders it.
    #:
    #: There is no longer a *choice of rendering*: the console backend emits one
    #: appended line per epoch on every stdout, because the three environments
    #: this runs in — a macOS terminal, an SSH session piped to a file, a
    #: Kaggle/Colab cell — cannot all honour a redrawing bar, and the two that
    #: cannot produce an unreadable log rather than a degraded one. ``bar`` and
    #: ``rows`` are still accepted, and both mean "render the line", so an
    #: existing command line keeps composing.
    progress: str = "auto"
    #: Epoch stride for the expensive console diagnostics — the hardest-class
    #: block and the leave-one-branch-out influence percentages. A new best
    #: checkpoint renders them too, off-stride, since that is the epoch the
    #: saved weights describe. The *numbers* behind them (per-class F1, the CDWS
    #: weights) are still computed whenever a checkpoint needs them; only the
    #: rendering and the ablation are throttled.
    diagnostics_interval: int = 50


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
    single: SingleStageConfig = MISSING
    tracking: TrackingConfig = MISSING
    #: Execution knobs. Defaulted rather than MISSING so a config written
    #: before this group existed still composes — and still runs the same
    #: numbers, since nothing here is allowed to change one.
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    #: Selection/reporting protocol. Defaulted for the same reason
    #: :class:`RuntimeConfig` is: a config that never mentions it is not
    #: under-specified, it is asking for CHANGES §19's rules.
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ── Run identity & output location ────────────────────────────────
    run_name: str = MISSING
    output_root: str = MISSING
    output_dir: str = MISSING

    # ── Curriculum selection (IC-11) ──────────────────────────────────
    #: ``single`` runs :mod:`~spectralquadnet.engine.stages.single_stage`
    #: (CHANGES §17); ``three_stage`` runs the audited Stage 1 → 2 → 3
    #: pipeline, which A8 needs as its control arm. ``stage1_only`` and
    #: ``stage1_stage2`` are A8's other two arms — the same three-stage driver,
    #: stopped early, so the arms differ in exactly one thing.
    pipeline: str = MISSING

    # ── Shared training knobs ─────────────────────────────────────────
    weight_decay: float = MISSING
    grad_clip: float = MISSING
    ema_decay: float = MISSING
    #: GradNorm exponent for the per-branch auxiliary weights (OP-2 / T2-6):
    #: ``omega_b <- omega_b * (g_bar / g_b) ** aux_gradnorm_alpha``, applied
    #: once per epoch from the per-branch gradient norms the loops already
    #: log. At 0.0 the weights never move and the fixed ``A/B = 2x`` vector
    #: stands.
    aux_gradnorm_alpha: float = MISSING

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
    cs.store(group="single", name="base_single", node=SingleStageConfig)
    cs.store(group="tracking", name="base_tracking", node=TrackingConfig)
    cs.store(group="runtime", name="base_runtime", node=RuntimeConfig)
    cs.store(group="evaluation", name="base_evaluation", node=EvaluationConfig)
    cs.store(name="base_experiment", node=ExperimentConfig)
