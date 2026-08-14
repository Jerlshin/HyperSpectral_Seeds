#!/usr/bin/env python
"""AST-level "no-op move" diff against a pinned reference implementation.

Proves that each of these classes/functions is numerically equivalent to a
pinned historical reference: it extracts each class/function from the
reference implementation (via ``git show <baseline-sha>``, see
:mod:`_baseline`) and from its current location in ``src/spectralquadnet``,
normalises both ASTs, and compares ``ast.dump(node, annotate_fields=False)``.

Granularity is **per method**, not per class: for ``SpectralQuadNet``, what
matters is that ``forward`` — the numerics-critical path — comes through
untouched, and a whole-class comparison would hide that behind ``__init__``'s
unavoidable config rewiring.

Verdicts
────────
``IDENTICAL``  normalised ASTs match exactly. This is the expected result for
               the overwhelming majority of compared code.
``DECLARED``   the symbol appears in :data:`DECLARED_DEVIATIONS` (changed) or
               :data:`DECLARED_REMOVALS` (deleted outright) with a written
               reason, and its full diff is printed for review. Entries are of
               exactly three kinds, and each says which it is:

               * **relocation** — a config-access or dependency-injection
                   rewrite, numerics untouched. Every entry was of this kind
                   through Phase 5.
               * **correctness fix** — a deliberate change of behaviour,
                   carrying its ``IMPROVEMENT_PLAN.md`` item id (``T1-3``, …)
                   and a one-line statement of what now differs. Introduced by
                   Tier 1: the point of those items is that the reference
                   implementation was wrong, so "identical to the baseline" is
                   no longer the property to prove for them. Each is pinned
                   instead by a dedicated test under ``tests/unit/``.
               * **removal** — the symbol no longer exists. Introduced by
                   Tier 2, whose T2-10 deletes a whole classification head and
                   the API that selected it. A removal is the one change a
                   diff cannot show, so it needs the loudest declaration.
``NEW``        exists only in the current codebase (e.g. ``SpectralQuadNet.from_config``).
               Reported, never failed — new code is not a relocation.
``DRIFT``      changed without a declaration. **Fails the check.**

Normalisations applied (each provably numerics-inert)
─────────────────────────────────────────────────────
1. **Docstrings erased.** Documentation is free to differ between the
   reference implementation and the current one without registering as drift.
2. **Annotations erased** — parameter, return and ``AnnAssign`` annotations.
   Both the reference file and the current module carry ``from __future__
   import annotations``, so under PEP 563 annotations are unevaluated
   strings that cannot affect runtime behaviour, which is what lets type
   hints be modernised (``Optional[str]`` → ``str | None`` and ``Dict`` →
   ``dict``) without weakening the check. Parameter *names*, *order* and
   *default values* are still compared.
3. **Position attributes ignored** (``ast.dump`` default).

Usage
─────
    python scripts/check_ast_no_op_move.py            # summary + declared diffs
    python scripts/check_ast_no_op_move.py --verbose  # also list every IDENTICAL
    python scripts/check_ast_no_op_move.py --baseline-ref <sha>
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _baseline import BASELINE_REF, baseline_symbols  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "spectralquadnet"

# ══════════════════════════════════════════════════════════════════════
#  Relocation map — reference file → symbol → current module
# ══════════════════════════════════════════════════════════════════════

RELOCATIONS: dict[str, dict[str, str]] = {
    "hsi_training": {
        # ── models/ ────────────────────────────────────────────────────
        "ModelEMA": "models/ema.py",
        "AdaptiveSubcenterArcFaceHead": "models/heads.py",
        "AuxiliaryHead": "models/heads.py",
        "MaskedSpectralECA": "models/blocks/attention.py",
        "SEBlock1D": "models/blocks/attention.py",
        "CBAM": "models/blocks/attention.py",
        "ResBlock1D": "models/blocks/conv_blocks.py",
        "ResBlock2D": "models/blocks/conv_blocks.py",
        "LargeKernelBlock1D": "models/blocks/conv_blocks.py",
        "PhysicalWavelengthPE": "models/blocks/positional.py",
        "SpectralProfileBranch": "models/branches/spectral_profile.py",
        "SpectralStatsBranch": "models/branches/spectral_stats.py",
        "SpatialCNNBranch": "models/branches/spatial_cnn.py",
        "MultiScaleSpectralTokenizer": "models/branches/specformer.py",
        "_PreLNBlock": "models/branches/specformer.py",
        "SpecFormerBranch": "models/branches/specformer.py",
        "CrossModalInteraction": "models/fusion.py",
        "EmbedNet": "models/fusion.py",
        "extract_grid_spectra": "models/stats_ops.py",
        "masked_spectral_stats": "models/stats_ops.py",
        "SpectralQuadNet": "models/spectral_quadnet.py",
        # ── data/ ──────────────────────────────────────────────────────
        "RiceSeedDataset": "data/datasets.py",
        "ClassBalancedBatchSampler": "data/samplers.py",
        "HardClassOversampledSampler": "data/samplers.py",
        "build_splits": "data/loaders.py",
        "build_loaders": "data/loaders.py",
        "build_phase3_loader": "data/loaders.py",
        # ── losses/ ────────────────────────────────────────────────────
        "build_cdws_weights": "losses/cdws.py",
        "_mixup": "losses/mixup.py",
        "mixed_aug": "losses/mixup.py",
        "mixed_loss": "losses/mixup.py",
        "FocalLoss": "losses/focal.py",
        "SupConLoss": "losses/contrastive.py",
        "ProtoNCELoss": "losses/contrastive.py",
        "_aux_loss_weight": "losses/auxiliary.py",
        "_compute_aux_loss": "losses/auxiliary.py",
        # ── optim/ ─────────────────────────────────────────────────────
        "SAM": "optim/sam.py",
        "_wd_groups": "optim/param_groups.py",
        "build_optimizer_s1": "optim/param_groups.py",
        "build_optimizer_s2": "optim/param_groups.py",
        "build_optimizer_s3": "optim/param_groups.py",
        "sgdr_scheduler": "optim/schedulers.py",
        "arcface_margin": "optim/schedulers.py",
        # ── engine/ ────────────────────────────────────────────────────
        "train_one_epoch": "engine/train_epoch.py",
        "train_one_epoch_sam": "engine/train_epoch.py",
        "_run_eval": "engine/evaluate.py",
        "evaluate": "engine/evaluate.py",
        "evaluate_per_class": "engine/evaluate.py",
        "tta_predict": "engine/tta.py",
        "stage_ckpt_path": "engine/checkpoint.py",
        "stage_meta_path": "engine/checkpoint.py",
        "stage_exists": "engine/checkpoint.py",
        "latest_completed_stage": "engine/checkpoint.py",
        "save_ckpt": "engine/checkpoint.py",
        "_is_json_serialisable": "engine/checkpoint.py",
        "load_stage_meta": "engine/checkpoint.py",
        "load_ckpt": "engine/checkpoint.py",
        "update_bn_stats": "engine/checkpoint.py",
        "_pick_best_checkpoint": "engine/checkpoint.py",
        "compute_branch_influence": "engine/diagnostics.py",
        "compute_class_difficulty": "engine/diagnostics.py",
        # ── engine/stages/ ─────────────────────────────────────────────
        "run_stage1": "engine/stages/stage1_progressive.py",
        "run_stage2": "engine/stages/stage2_arcface.py",
        "run_stage3_swa": "engine/stages/stage3_sam_swa.py",
        "final_evaluation": "engine/stages/final_eval.py",
        # ── utils/ ─────────────────────────────────────────────────────
        # The golden capture in capture_golden.py needs `set_seed`.
        "set_seed": "utils/seed.py",
    },
    "data_setup_v3": {
        "download": "data/prep/download.py",
        "load_hsi": "data/prep/segmentation.py",
        "preprocess_raw": "data/prep/segmentation.py",
        "segment": "data/prep/segmentation.py",
        "pad_to_square": "data/prep/patch_extraction.py",
        "resize_patch": "data/prep/patch_extraction.py",
        "main": "data/prep/patch_extraction.py",
    },
    "band_selection": {
        "extract_mean_spectra": "data/prep/band_selection.py",
        "decorrelation_prefilter": "data/prep/band_selection.py",
        "fisher_discriminant_ratio": "data/prep/band_selection.py",
        "run_mrmr": "data/prep/band_selection.py",
        "run_spa": "data/prep/band_selection.py",
        "validate": "data/prep/band_selection.py",
        "find_elbow": "data/prep/band_selection.py",
        "save_outputs": "data/prep/band_selection.py",
        "main": "data/prep/band_selection.py",
    },
}

# Symbols renamed relative to the reference implementation (old name → new name).
RENAMES: dict[tuple[str, str], str] = {
    ("data_setup_v3", "main"): "build_patch_dataset",
    ("band_selection", "main"): "select_bands",
}

# ══════════════════════════════════════════════════════════════════════
#  Declared deviations — every sanctioned non-identity, with its reason
# ══════════════════════════════════════════════════════════════════════
#
#  Keys are "Symbol" for free functions and "Class.method" for methods. Anything
#  that differs and is NOT listed here fails the run.

#: Symbols the current codebase deliberately does not have. Same contract as
#: :data:`DECLARED_DEVIATIONS`: a plan item id and what replaced it.
DECLARED_REMOVALS: dict[str, str] = {
    "SpectralQuadNet.use_arcface": (
        "**T2-10 removal** (HD-1). There is one head for all three stages, so "
        "nothing selects between two. The `_use_arcface` flag it set is gone "
        "with it; `save_ckpt` still writes a constant `use_arcface: True` into "
        "the bundle so existing readers keep working. Pinned by "
        "`tests/unit/test_unified_head.py`."
    ),
    "SpectralQuadNet.freeze_head": (
        "**T2-10 removal** (HD-1). Stage 1 froze `arcface_head` and Stage 2 "
        "froze `linear_head`; with one head, freezing it would freeze the "
        "classifier outright. Both call sites are deleted."
    ),
    "SpectralQuadNet.unfreeze_head": "**T2-10 removal** (HD-1) — see `freeze_head`.",
    "AdaptiveSubcenterArcFaceHead.init_from_linear": (
        "**T2-10 + T2-9 removal** (HD-1, HD-2(iii)). There is no linear head to "
        "bootstrap from, and the scheme itself — one prototype plus `0.01 * k` "
        "noise, i.e. two random decoys 9-18° away — is the mechanism behind the "
        "dead sub-centres 0-I counted (M-8). Replaced by "
        "`init_subcentres_from_embeddings`, spherical k-means on the Stage-1 "
        "embeddings. Pinned by `tests/unit/test_subcentres.py`, which also "
        "measures that the scheme removed here leaves sub-centres dead."
    ),
    "AdaptiveSubcenterArcFaceHead.update_margins_from_f1": (
        "**T2-8 removal** (HD-3 / M-6). `F1` conflates the two failure modes "
        "that need opposite margin responses, and the rule raised the margin "
        "for every low-`F1` class — including the low-*recall* ones, for which "
        "a raise shrinks the region further. Replaced by "
        "`update_margins_from_pr`, driven by the signed `R_c - P_c`. Pinned by "
        "`tests/unit/test_margin_rule.py::test_margin_rule_sign`, which also "
        "measures the removed rule's opposite sign."
    ),
}

DECLARED_DEVIATIONS: dict[str, str] = {
    # ── Tier 2 (IMPROVEMENT_PLAN §4.2) ─────────────────────────────────
    "AdaptiveSubcenterArcFaceHead.__init__": (
        "**T2-8 + T2-9** (HD-3, HD-2). Four new constructor arguments — "
        "`m_min`/`m_max` (HD-3's clip range), `tau` (HD-2's sub-centre pooling "
        "temperature, a plain float so the schedule never enters the "
        "checkpoint) and `pairwise_delta` — and one new buffer, "
        "`confusion` of shape `(C, C)`, which is the schema-v2 addition "
        "`engine/checkpoint.py::remap_state_dict` back-fills with zeros. "
        "`m_delta`'s default moves 0.10 → 0.20 with the rule it parameterises. "
        "The weight tensor, its `xavier_uniform_` init and the `margins` buffer "
        "are unchanged. Pinned by `tests/unit/test_margin_rule.py` and "
        "`tests/unit/test_subcentres.py`."
    ),
    "RiceSeedDataset.<class-body>": (
        "**T2-7** (OP-6). `_PROFILES` gains `spec_cutmix`/`spat_cutmix` "
        "probabilities on all four active profiles; same-class CutMix is "
        "label-preserving, so it composes with the angular objective where "
        "mixup does not. `_INTENSITY_SCALE` and `_WARP_RANGE` are untouched, "
        "and the five original probabilities keep their values. Pinned by "
        "`tests/unit/test_cutmix.py`."
    ),
    "SAM.__init__": (
        "**T2-4** (OP-5). An `adaptive` flag selects ASAM, and lands in "
        "`defaults` (so it reaches every param group) rather than in the "
        "`**kwargs` forwarded to the base optimizer, which would reject it. "
        "Everything else — the group re-pointing, the `defaults.update` — is "
        "unchanged. Pinned by `tests/unit/test_asam.py`."
    ),
    "SAM.first_step": (
        "**T2-4** (OP-5). The step becomes `rho * theta^2 * g / ||theta * g||` "
        "under `adaptive`, which is scale-invariant and therefore defined on a "
        "head whose loss satisfies `L(cW) = L(W)`; and a group carrying "
        "`perturb: False` is skipped entirely, OP-5's stated alternative. At "
        "`adaptive=False` with no `perturb` key the arithmetic is the "
        "pre-Tier-2 one."
    ),
    "SAM._grad_norm": (
        "**T2-4** (OP-5). The norm is `||T_theta g||` under `adaptive` and "
        "skips non-perturbed groups, matching `first_step`'s denominator."
    ),
    "load_ckpt": (
        "**T2-10** (HD-1). The `use_arcface` flag it used to restore on both "
        "models is gone with the second head; instead the bundle's "
        "`schema_version` routes both state dicts through `remap_state_dict`, "
        "so schema-v1 checkpoints — every archived one — still load "
        "`strict=True`. The `torch.load` call and the return value are "
        "unchanged. Pinned by "
        "`tests/regression/test_state_dict_compatibility.py`."
    ),
    # ── Tier 1 correctness fixes (IMPROVEMENT_PLAN §4.2) ───────────────
    #  Behaviour changes, not relocations. Each names its plan item and the
    #  test that pins it.
    "AdaptiveSubcenterArcFaceHead.forward": (
        "**T1-7 correctness fix** (HD-4 / N-4). The cosine clamp widens from "
        "`1e-6` to the `COS_CLAMP_EPS = 1e-3` module constant, bounding "
        "`|d/dc sqrt(1-c^2)|` at 22.4 instead of 707 and keeping `1 - c^2` out "
        "of fp32's cancellation régime — near the old clamp that difference "
        "lost five significant figures, so the sine feeding the margin was "
        "numerically meaningless. Costs 2.6° of angular resolution against "
        "margins of 20–26°, and changes no prediction. Everything else in the "
        "head — the sub-centre max, the margin algebra, the `s` scaling — is "
        "byte-identical. Pinned by `tests/unit/test_arcface_head.py`."
        " **AMP correctness fix**: the body is wrapped in "
        "`autocast(enabled=False)` and the embedding is upcast on the way in, "
        "so the head's arithmetic is fp32 whatever precision the backbone ran "
        "in. Numerically inert for an fp32 caller — entering that context "
        "outside an autocast region is a no-op and `.float()` on an fp32 "
        "tensor returns it — which is what keeps the golden digests valid; "
        "under autocast it is the difference between a margin and a NaN, since "
        "bf16 resolves ~3.9e-3 near cos = 1 and the clamp it has to see "
        "through is 1e-3. `acos`'s argument is additionally clamped to "
        "[-1, 1] (it returns NaN, not a saturated value, an ulp outside), the "
        "pooling temperature is floored at `TAU_FLOOR` so `cos/tau` cannot "
        "overflow, and `balance_loss`'s `pi log(pi K)` becomes `torch.xlogy`, "
        "which is 0 rather than NaN at a dead sub-centre. Pinned by "
        "`tests/unit/test_amp_precision.py`."
    ),
    "FocalLoss.forward": (
        "**T1-3 correctness fix** (OP-1 / M-9). The focal modulator reads the "
        "unsmoothed `p_y = softmax(z)_y` instead of `exp(-ce_smoothed)`. The "
        "two are the same float at `label_smoothing = 0`, so Stage 2 and "
        "Stage 3 are bit-identical; with smoothing on, the old form was bounded "
        "below by `(1 - exp(-H(q)))^gamma` — 0.3955 at eps=0.10 — and could not "
        "down-weight a confident sample at all. The smoothed cross-entropy "
        "`ce` it multiplies is unchanged. Pinned by `tests/unit/test_focal.py`."
    ),
    "SpectralQuadNet.forward": (
        "**T1-6 correctness fix** (N-1d). The hardcoded branch-drop vector "
        "`[0.0, 0.0, 0.30, 0.20]` becomes `self.branch_drop_prob * "
        "BRANCH_DROP_PROFILE`, and the whole masking block is skipped when that "
        "probability is 0 — `model.branch_drop_prob` was stored and ignored, so "
        "Stage 3's `= 0.0` never disabled masking. At the shipped `0.20` the "
        "scaled vector is **bit-identical** to the literal and consumes the same "
        "RNG draws, which is what keeps the Stage-1 golden digests valid. The "
        "profile's shape (M-5) is deliberately unchanged. Everything upstream "
        "of the mask — `se`, the statistics, the four branches — and everything "
        "downstream — fusion, heads, the training-mode dict — is untouched. "
        "Pinned by `tests/unit/test_branch_drop.py`."
    ),
    "tta_predict": (
        "**T1-1 + T1-2 correctness fixes** (TT-1, TT-3). The spectral view moves "
        "into `spectral_view()`, which rescales about the **foreground** mean "
        "and re-masks, so the background stays exactly zero and the foreground "
        "fraction is preserved — the invariant every masked operator downstream "
        "infers by testing for zero. The forwards now run under "
        "`autocast(enabled=False)`, matching `engine/evaluate.py`, so the "
        "TTA/no-TTA comparison is not confounded with a precision change. The "
        "dihedral view set, the scale grid, the `s == 1.0` skip and the "
        "logit-space averaging are unchanged; an optional `mask` argument is "
        "added for FE-2. Measured effect on the archived checkpoints: "
        "-0.0046 / -0.0049 / +0.0030 macro-F1, every interval straddling zero. "
        "Pinned by `tests/unit/test_tta.py`."
    ),
    # ── models ─────────────────────────────────────────────────────────
    "SpectralQuadNet.__init__": (
        "§3.4 `_PHYSICAL_WL` global → injected `physical_wl` parameter; "
        "§5 `cfg[...]`/`cfg.get(...)` → grouped `cfg.model.*` / `cfg.stage2.*` "
        "attribute access (the `.get()` defaults drop out because the structured "
        "schema makes every key mandatory). No sub-module construction was "
        "reordered — §3.6 depends on that order."
    ),
    # ── data ───────────────────────────────────────────────────────────
    "RiceSeedDataset.__init__": (
        "§3.4 `_GPU_PATCHES`/`_GLOBAL_LABELS` globals → injected DataStore; "
        "`CONFIG['max_cutout_bands']`/`CONFIG['noise_std']`/`CONFIG['device']` "
        "cached on self from the injected data_cfg/device."
    ),
    "RiceSeedDataset._band_cutout": "`CONFIG['max_cutout_bands']` → `self.max_cutout_bands`.",
    "RiceSeedDataset._spectral_noise": "`CONFIG['noise_std']` → `self.noise_std`.",
    "RiceSeedDataset.__getitem__": "`CONFIG['device']` → `self.device` (2 call sites).",
    "build_splits": (
        "**T4-1 + T4-5 protocol change** (P-1, P-5). `CONFIG['labels_path']` → "
        "`cfg.data.labels_path`, and the function is now a thin wrapper over "
        "`build_split_bundle`, which selects between the original patch-level "
        "`train_test_split` recipe (`split_scheme=stratified`) and a "
        "scan-disjoint grouped split (`grouped`), and carves P-5's `calib` out "
        "of train. **At the shipped `stratified` / `calib_frac=0.0` the two "
        "`train_test_split` calls, their order, their `random_state=42` and "
        "the *unsorted* order of the arrays they return are unchanged**, so "
        "the 6,036/1,294/1,294 partition and the training stream it feeds are "
        "bit-identical — `tests/unit/test_splits.py::"
        "test_the_stratified_path_reproduces_the_reference_split` asserts that "
        "against a transcription of the baseline recipe. The returned tuple "
        "keeps its four-element shape; `train_idx` excludes `calib` when one "
        "is carved, which is P-5's requirement that calib never sees a "
        "gradient. Pinned by `tests/unit/test_splits.py`."
    ),
    "build_loaders": (
        "`CONFIG['bal_n_cls']`/`['bal_n_spc']` → `cfg.stage2.*`; the three "
        "RiceSeedDataset constructions now pass store/data_cfg/device through a "
        "shared `kw` dict instead of reading module globals."
    ),
    "build_phase3_loader": (
        "`CONFIG['s1_*']` → `cfg.stage1.*`; `_GLOBAL_LABELS` → `store.require_labels()`."
    ),
    # ── losses / optim ─────────────────────────────────────────────────
    "_aux_loss_weight": (
        "§5 `CONFIG['aux_loss_weight_{init,final}']` → `cfg.stage1.*`; the "
        "`max(...)` floor and the 0.7 decay factor are untouched."
    ),
    "_compute_aux_loss": (
        "§4.2 (Phase 4) per-branch loss diagnostic: the weighted term is bound to "
        "a local before being summed, so the same tensor can also be recorded in "
        "`components`, which is returned alongside the total under the new "
        "`return_components=False` flag. Default-off, so every pre-existing call "
        "site and return value is unchanged; the accumulation order, the "
        "`branch_weights` table and the arithmetic are untouched, making the sum "
        "bit-identical — `tests/regression/test_golden_forward_pass.py`'s Stage-1 "
        "loss gate covers exactly this path."
        " **T2-6** (OP-2): the `branch_weights` literal becomes the module "
        "constant `DEFAULT_BRANCH_WEIGHTS` plus an optional `weights` override, "
        "which `GradNormAuxWeights` supplies. Iteration order still comes from "
        "the default table, so the summation order — and the float — does not "
        "depend on the caller's dict, and `weights=None` is the old number "
        "exactly. Pinned by `tests/unit/test_gradnorm_aux.py`."
    ),
    "_wd_groups": "§5 `CONFIG['weight_decay']` → `cfg.weight_decay`.",
    "build_optimizer_s1": "`_wd_groups` gains its leading `cfg` argument.",
    "build_optimizer_s2": "`_wd_groups` gains its leading `cfg` argument (both call sites).",
    "build_optimizer_s3": "`_wd_groups` gains its leading `cfg` argument.",
    # ── engine ─────────────────────────────────────────────────────────
    "train_one_epoch": (
        "§5 `CONFIG['grad_clip']` → `cfg.grad_clip`; `_aux_loss_weight` gains its "
        "leading `cfg` argument. The loss algebra, the accumulation boundary, the "
        "non-finite skip and the EMA update site are untouched. §4.2 (Phase 4) "
        "adds an optional `tracker`: when it is None — how every regression gate "
        "calls this — no diagnostic is accumulated and the code path is the "
        "pre-refactor one. When present, per-branch aux losses and pre-clip "
        "per-branch gradient norms are summed as device tensors and resolved once "
        "at epoch end."
        " **T2-5 + T2-6 + T2-9 + T2-10**: the single global "
        "`clip_grad_norm_(model.parameters(), ...)` becomes "
        "`clip_grad_norm_by_group`, so a saturated head no longer divides the "
        "backbone's effective LR (OP-3 / M-10); an optional `aux_weights` "
        "closes OP-2's GradNorm loop once per epoch from the per-branch norms "
        'the loop already samples; `out["balance"]` is added at '
        "`cfg.model.subcenter_balance_weight` when present (HD-2(ii)); and the "
        "mixup guard tests the **margin** rather than which head is selected, "
        "because HD-1 left only one — Stage 1's `arc_m = 0` is a cosine "
        "classifier, which takes interpolated targets. Pinned by "
        "`tests/unit/test_grad_clip_groups.py`, "
        "`tests/unit/test_gradnorm_aux.py` and `tests/unit/test_cutmix.py`."
        " **AMP correctness fix**: an `amp_dtype` parameter is passed through "
        "to `autocast`, defaulting to `torch.bfloat16` instead of torch's "
        "per-device default of fp16 — the dtype under which this loop's "
        "non-finite-loss skip fired on every batch from Stage 1 epoch 50 on. "
        "Nothing else in the loop moves: the `use_amp` predicate, the scale / "
        "unscale_ / step / update sequence and the skip itself are unchanged, "
        "and under bf16 the scaler `run_stage1` supplies is constructed "
        "disabled, which torch documents as a pass-through on all four. Pinned "
        "by `tests/unit/test_amp_precision.py`."
    ),
    "train_one_epoch_sam": (
        "§5 `CONFIG['grad_clip']` → `cfg.grad_clip` (both clip call sites); §4.2 "
        "adds the same optional `tracker` and `current_ep` as `train_one_epoch`, "
        "inert when no tracker is passed. Gradient norms are sampled on the SAM "
        "ascent step, before its clip. "
        "**T1-4 + T1-10 correctness fixes** (OP-4.1, OP-7 / C-6, N-1e): both SAM "
        "steps now evaluate one `_objective()` — focal + SupCon + ProtoNCE + aux "
        "— where the descent step used to drop everything but focal, which is "
        "not a SAM step for any single objective; and the `proto` argument, "
        "previously accepted, weighted, logged and never added to a loss, is "
        "applied. Two supporting changes: `cos(ĝ_A, ĝ_D)` is logged as "
        "`sam/grad_cos` under the existing `log_grad_norms` gate, and a "
        "non-finite descent loss now calls `SAM.restore()` before skipping the "
        "batch instead of leaving the ascent perturbation permanently in the "
        "weights. An optional `ema` is updated per optimiser step, as "
        "`train_one_epoch` already did. Pinned by "
        "`tests/unit/test_stage3_sam.py`."
        " **T2-5 + T2-6 + T2-9**: both clips become per-group "
        "(`clip_grad_norm_by_group`), the optional `aux_weights` closes OP-2's "
        "loop as in `train_one_epoch`, and the sub-centre balance term joins "
        "`_objective` so both SAM steps still evaluate one function."
    ),
    "stage_ckpt_path": "§3.5 `CONFIG['output_dir']` → `cfg.output_dir`; filename template unchanged.",
    "stage_meta_path": "§3.5 `CONFIG['output_dir']` → `cfg.output_dir`; filename template unchanged.",
    "stage_exists": "`stage_{ckpt,meta}_path` gain their leading `cfg` argument.",
    "latest_completed_stage": "`stage_exists` gains its leading `cfg` argument; 3→2→1 order kept.",
    "save_ckpt": "`stage_meta_path` gains its leading `cfg` argument; bundle schema unchanged.",
    "update_bn_stats": (
        "**T1-5 correctness fix** (OP-4.6 / C-7e / N-12), superseding Phase 5's "
        "Metal workaround. Every stochastic module — `nn.Dropout` and the "
        "`nn.MultiheadAttention` internal dropout `set_dropout` cannot reach — is "
        "forced to `eval()` for the pass, while BatchNorm stays in `train()`. "
        "Inverted dropout inflates the variance by `p/(1-p)·E[a^2]`, so the "
        "train-mode pass recorded a per-channel σ larger than the eval-time one "
        "and attenuated both high-capacity branches' outputs non-uniformly. With "
        "nothing stochastic left, the earlier device-dependent grad context "
        "(`no_grad` off-Metal, `enable_grad` on Metal, because Metal's fused "
        "attention kernel rejects dropout under `no_grad`) is gone: the pass is "
        "plain `no_grad()` everywhere and the buffers are reproducible across "
        "accelerators. The BN reset/momentum handling and the single pass over "
        "the loader are unchanged; the caller now supplies a natural-prior "
        "loader. Pinned by `tests/unit/test_bn_stats.py`."
    ),
    "load_stage_meta": "`stage_meta_path` gains its leading `cfg` argument.",
    "_pick_best_checkpoint": "`load_stage_meta` gains its leading `cfg` argument.",
    "compute_class_difficulty": (
        "§5 `CONFIG['num_classes']` → `cfg.data.num_classes` and "
        "`CONFIG['cdws_{max_weight,eps}']` → `cfg.stage2.cdws_*`. §4.1/§4.2 "
        "(Phase 4): the `print` became `tracker.log_message`, and the branch "
        "influence dict `compute_branch_influence` already returned is now also "
        "routed to `tracker.log_scalars` plus a `hardest_classes_report` table. "
        "Every computation above those calls is untouched."
    ),
    # ── engine/stages ──────────────────────────────────────────────────
    "run_stage1": (
        "§5 `CONFIG[...]` → `cfg.stage1.*` (`CONFIG.get('s1_p3_dropout', 0.25)` → "
        "`cfg.stage1.p3_dropout`, whose YAML value is 0.25); collaborators gain "
        "their leading `cfg`/`store` arguments; the `phase_aware_lr` closure moved "
        "to `optim/schedulers.py` as a factory (§2 tree, §3.2.3) and is called "
        "here instead of defined here — `tests/unit/test_schedulers.py` proves all "
        "600 epochs of the schedule are unchanged. Phase boundaries, EMA re-init "
        "points, loss selection and the checkpoint condition are untouched."
        "§4.1 (Phase 4): the banner block became `tracker.banner`, the `[INFO]` notices `tracker.log_message`, and the per-epoch line a `log_row`/`log_scalars` pair. Observability-only — every logged value was already a local."
        " **T1-8** (§2.1.4): `save_ckpt` additionally records `best_source`, "
        "which of the live model and the EMA shadow won the `max` written as "
        "`val_f1`. Recording only — the selection rule and the saved value are "
        "unchanged; `final_evaluation` is what reads it."
        " **T2-10 + T2-9 + T2-6**: the three head-selection calls "
        "(`use_arcface(False)`, `unfreeze_head('linear')`, "
        "`freeze_head('arcface')`) and the trailing `unfreeze_head('arcface')` "
        "are gone — there is one head and it trains from epoch 1 — and the "
        "epoch now passes `arc_m=cfg.stage1.arcface_m` (0.0), which is what "
        "makes it a cosine classifier and keeps mixup admissible. Per epoch it "
        "also sets the sub-centre temperature from `subcentre_tau` and hands "
        "`train_one_epoch` a `GradNormAuxWeights`. Phase boundaries, EMA "
        "re-init points, loss selection and the checkpoint condition are still "
        "untouched. Pinned by `tests/unit/test_unified_head.py`."
        " **T4-5** (P-5): gains a `calib_ldr` parameter, and the two "
        "`compute_class_difficulty` calls — the Phase-2→3 oversampling "
        "measurement and the per-checkpoint CDWS fit — read it instead of "
        "`val_ldr`, which keeps driving `evaluate` and therefore selection. "
        "`calib_ldr=None` restores the pre-Tier-4 behaviour exactly. Pinned by "
        "`tests/unit/test_calib_split.py`."
        " **AMP correctness fix**: the bare `GradScaler(device=...)` becomes "
        "`make_grad_scaler(plan.amp_dtype, device)`, which pairs an *enabled* "
        "scaler with fp16 and a *disabled* one with bf16 — the new default — "
        "and the resolved dtype is passed to `train_one_epoch` and printed in "
        "the stage banner. This is the only stage that trains under autocast, "
        "so it is the only one that changes. Pinned by "
        "`tests/unit/test_amp_precision.py`."
    ),
    "run_stage2": (
        "§5 `CONFIG[...]` → `cfg.stage2.*` / `cfg.model.subcenter_K`; "
        "collaborators gain their leading `cfg` argument. The margin warmup "
        "switch, the 10-epoch contrastive ramp and the param_groups[0]/[2] LR "
        "readout are unchanged."
        "§4.1 (Phase 4): prints → tracker calls; the SGDR restart marker moved from a string suffix on the epoch line to its own row cell."
        " **T1-8** (§2.1.4): `best_source` is recorded alongside `val_f1`, exactly "
        "as in `run_stage1`."
        " **T2-8 + T2-10 + T2-9 + T2-6**: the head-selection and freeze calls "
        "are gone (HD-1), and the `class_f1` parameter with them — the margins "
        "are now calibrated in-stage by HD-3's signed rule from a fresh "
        "`evaluate_pr_and_confusion` pass, which also supplies the pairwise "
        "confusion matrix. Once at entry, not per epoch: 270 parameters fitted "
        "on the split that also selects the checkpoint is the §2.1.4/C-9 leak, "
        "and P-5 (T4-5) is the item that gives them their own split. Adds the "
        "per-epoch `subcentre_tau` and a `GradNormAuxWeights`. The margin "
        "warmup switch, the 10-epoch contrastive ramp and the "
        "param_groups[0]/[2] readout are unchanged. Pinned by "
        "`tests/unit/test_margin_rule.py`."
        " **T4-5** (P-5): that is now the split it fits on — a `calib_ldr` "
        "parameter routes the HD-3 calibration pass and the CDWS fit to the "
        "calibration split, leaving `val_ldr` free of fitted parameters. "
        "`calib_ldr=None` restores the pre-Tier-4 behaviour exactly. Pinned by "
        "`tests/unit/test_calib_split.py`."
    ),
    "run_stage3_swa": (
        "§5 `CONFIG[...]` → `cfg.stage3.*` / `cfg.stage2.dropout` / "
        "`cfg.weight_decay`; collaborators gain their leading `cfg` argument. The "
        "greedy 0.98 acceptance rule, the running SWA average, the hardcoded "
        "gamma/SupCon/ProtoNCE literals and `_s3_margin` are unchanged."
        "§4.1 (Phase 4): prints → tracker calls; the snapshot accept/reject marker moved from a string suffix on the epoch line to its own row cell."
        " **T1-9 + T1-5 correctness fixes** (OP-4.7, OP-4.6 / C-7f): the EMA "
        "shadow is re-initialised at stage entry, updated every step, scored "
        "every epoch as `val/f1_ema` (as Stages 1-2 do) and compared against the "
        "SWA average at the end — the better of the two is what the bundle's "
        "`ema` slot receives, with `best_source`/`swa_val_f1`/`ema_val_f1` in the "
        "sidecar. The stage previously overwrote the shadow with the SWA weights "
        "without either being evaluated. `update_bn_stats` is called on "
        "`build_natural_prior_loader(train_ldr)` rather than the CDWS-weighted "
        "loader, and the returned/saved F1 is `max(f1_swa, f1_ema)` rather than "
        "`f1_swa`. Pinned by `tests/unit/test_stage3_sam.py`."
        " **T2-1 + T2-2 + T2-3 + T2-4** (OP-4.2-4.5, OP-5) rewrite what remains. "
        "The `_s3_margin` scalar closure is gone: Stage 2's per-class vector is "
        "captured at entry and scaled by `stage3_margin_kappa`, which steps only "
        "at cycle boundaries, and `arc_m=None` is passed so the vector is what "
        "the head reads (C-7a, C-7b). Greedy acceptance evaluates the "
        "**candidate average** through a scratch `probe` model rather than "
        "testing `f1_live >= best_live_f1 * 0.98`, which was true by "
        "construction (C-7c). The first `swa_warmup_cycles` cycles are "
        "discarded before any candidate is considered (C-7d), and the accepted "
        "cycle index reaches the sidecar. SAM is constructed with "
        "`adaptive=cfg.stage3.sam_adaptive` (ASAM). Pinned by "
        "`tests/unit/test_stage3_swa.py`."
    ),
    "final_evaluation": (
        "§5 `CONFIG['tta_*']` → `cfg.tta_*`, `CONFIG['output_dir']` → "
        "`cfg.output_dir`. §4.1/§4.2 (Phase 4): prints → tracker calls, the three "
        "metrics per TTA mode also go to `log_scalars`, and the bottom-K classes "
        "of the same per-class F1 the classification report already tabulates are "
        "emitted as a `log_table`. The predictions and the three saved `.npy` "
        "files are produced by the same code as before. "
        "**T1-8 correctness fix** (§2.1.4): the evaluated weights are chosen by "
        "the bundle's `best_source` — the live model when it won the "
        "`max(F1_live, F1_ema)` the checkpoint was selected on — instead of "
        "always the EMA shadow. Bundles without the key, which is every bundle "
        "written before Tier 1, default to `ema` and so reproduce exactly. "
        "Pinned by `tests/unit/test_best_source.py`."
    ),
    # ── data/prep ──────────────────────────────────────────────────────
    "download": "module-level `ZIP_FILE`/`DATA_URL` globals → `PrepConfig` fields.",
    "load_hsi": (
        "`import spectral` + `spectral.settings.envi_support_nonlowercase_params` "
        "moved from module scope (baseline lines 19-20) into the function, so "
        "importing the package does not require the optional `spectral` dependency."
    ),
    "resize_patch": "module-level `PATCH_SIZE` global → explicit `patch_size` parameter.",
    "build_patch_dataset": (
        "was `main`; module-level `ZIP_FILE`/`NUM_BANDS`/`PATCH_SIZE`/`PATCHES_PATH`/"
        "`LABELS_PATH` globals → `PrepConfig` fields; `resize_patch` gains its "
        "`patch_size` argument. Pass structure and all magic constants unchanged. "
        "§4.4 (Phase 5): all three `with tempfile.TemporaryDirectory() as tmp: "
        "tmp = Path(tmp)` blocks bind the context manager to `tmp_dir` and keep "
        "`tmp = Path(tmp_dir)`, because rebinding one name from `str` to `Path` is "
        "unexpressible under `mypy --strict`. Every downstream use of `tmp` is "
        "byte-identical and still the same `Path`.\n"
        "        **T4-1/2/3/4 data-contract change** (P-1…P-4). The per-region "
        "body moves into `extract_patch`, which resizes the *mask* alongside "
        "the cube and divides by it (P-3, M-11) instead of masking then "
        "resizing, and applies the radiometric correction (P-2, C-1). The "
        "writer gains five outputs — `groups.npy` (the `scan_id` that makes a "
        "grouped split constructible at all), `masks.npy`, `gain.npy`, "
        "`morphology.npy` and `scan_table.csv` — and truncates rather than "
        "shipping the all-zero rows a cube that pass 1 counted and pass 2 "
        "failed on would leave, since a `-1` group would become its own split. "
        "The four-member resolution block, identical in both passes, is now "
        "`_resolve_cube_members`. Pass structure, the segmentation call and "
        "the label assignment are unchanged. Pinned by "
        "`tests/unit/test_patch_extraction.py` and "
        "`tests/unit/test_radiometry.py`."
    ),
    "extract_mean_spectra": "`CONFIG[...]` → `cfg.*`.",
    "decorrelation_prefilter": "`CONFIG['corr_threshold']` → `cfg.corr_threshold`.",
    "run_mrmr": "`CONFIG[...]` → `cfg.*`.",
    "run_spa": "`CONFIG[...]` → `cfg.*`.",
    "validate": "`CONFIG[...]` → `cfg.*`.",
    "find_elbow": "`CONFIG['elbow_pct']` → `cfg.elbow_pct`.",
    "save_outputs": "`CONFIG[...]` → `cfg.*`.",
    "select_bands": (
        "was `main`; `CONFIG[...]` → `cfg.*`; the module-level "
        "`warnings.filterwarnings` calls and `np.random.seed(CONFIG['seed'])` "
        "(baseline lines 58-60, 92) moved inside so importing the module no longer "
        "mutates global warning filters or the NumPy RNG (§3.6).\n"
        "        **T4-6** (M-14). The elbow is now *verified* rather than "
        "asserted: `verify_elbow` measures whether the curve extends past the "
        "chosen k, the verdict is written to `band_selection_elbow.json` and "
        "printed, and a non-demonstrable elbow says so. `deployed_curve_path` "
        "lets the deployed estimator's curve (F-3) decide the winner and the "
        "elbow instead of LDA/LinearSVC on mean spectra, and the report gains "
        "a `deployed` column. `find_elbow`'s arithmetic is untouched. Measured "
        "on the shipped `band_selection_report.csv`: **NOT demonstrable — the "
        "recorded curve stops at k = 40, the value it selected.** Pinned by "
        "`tests/unit/test_band_curve.py`."
    ),
    # ── Tier 3 (IMPROVEMENT_PLAN §4.2) ─────────────────────────────────
    #
    # Every entry below is a **redesign**, not a relocation and not a
    # correctness fix in the Tier-1 sense: the module still has the name the
    # checkpoint schema addresses it by, and computes something else. §3.3's
    # controlling constraint is that each branch must see something the others
    # cannot reconstruct, and three of the four did not.
    "SpectralStatsBranch.__init__": (
        "**T3-1 redesign** (BR-1). Branch B no longer reads the nine masked "
        "moments. §2.2.5 proves that tensor is rank <= 2 under the per-pixel "
        "gain model and 0-D measured it on the real data, so 686 k parameters "
        "were reading a two-dimensional signal. The three 1-D towers, the "
        "statistic-attention gate, the input projection, the attention pooling "
        "and the wavelength PE are all gone; in their place are a 64-entry "
        "learned normalised-difference index bank (5,120 params, exactly "
        "gain-invariant), 16 continuum-removed absorption depths, the 8 "
        "persisted morphometrics and a 2-layer MLP. 686,424 -> 94,896 "
        "parameters on a full-rank input. Pinned by "
        "`tests/unit/test_masked_ops.py`."
    ),
    "SpectralStatsBranch.forward": (
        "**T3-1 redesign** (BR-1) — see `SpectralStatsBranch.__init__`. The "
        "signature changes from nine `(B, C)` statistics to one `(B, C)` "
        "foreground mean spectrum plus optional `(B, 8)` morphometrics."
    ),
    "SpectralStatsBranch._init_weights": (
        "**T3-1 redesign** (BR-1). No `nn.Conv1d` remains in the branch, so "
        "the kaiming arm is gone; the index bank's `theta` are raw "
        "`nn.Parameter`s and keep the small init that starts every band "
        "participating in every index."
    ),
    "SpectralProfileBranch.__init__": (
        "**T3-5 + T3-6** (FE-1(a), BR-2). The `Conv1d(1, 96, 3)` stem becomes a "
        "`LambdaConv1d` whose kernel is *generated* from wavelength offsets over "
        "the 5 nearest bands in lambda (C-5: every 1-D convolution was a finite "
        "difference on an irregular grid), and the branch gains a "
        "`SpectralDerivatives` module so its input is `[SNV(r), d/dlambda, "
        "d2/dlambda2]` rather than the raw grid spectra Branch D also received "
        "(C-2, §2.2.2). The three towers, the fusion, the attention pooling and "
        "the projection are untouched. +10 k parameters, independent of the band "
        "count. Pinned by `tests/unit/test_front_end.py` and "
        "`tests/unit/test_branch_inputs.py`."
    ),
    "SpectralProfileBranch.forward": (
        "**T3-5 + T3-6** — see `SpectralProfileBranch.__init__`. The body is the "
        "same operations on the derivative channels; it now delegates to "
        "`forward_channels` so `SpectralQuadNet.branch_inputs` can build the "
        "input once and the distinctness test can hash the object the branch "
        "actually consumes."
    ),
    "SpatialCNNBranch.__init__": (
        "**T3-2 redesign** (BR-3). `band_reduce` — two 1x1 convolutions that "
        "collapsed the band axis before any spatial kernel ran — is replaced by "
        "a factorised 3-D stem (Conv3d 1->16->32->64 over (lambda, h, w), then a "
        "1x1 fold of the surviving 5 spectral positions into 192 channels). This "
        "is C-3: the network contained no joint spectral-spatial operator at "
        "all, so 'this absorption feature, in this part of the seed' was not in "
        "any module's hypothesis class. The ResBlock2D/CBAM tail is unchanged "
        "except for its input width. 1,694,158 -> 2,230,646 parameters, funded "
        "by BR-1. Pinned by `tests/unit/test_branch_c_stem.py`."
    ),
    "SpatialCNNBranch.forward": (
        "**T3-2 + T3-7** (BR-3, FE-2) — see `SpatialCNNBranch.__init__`. Takes "
        "the explicit mask and passes it to the stem, which re-zeros the padded "
        "region after every stage so the CNN can never learn the frame."
    ),
    "SpecFormerBranch.__init__": (
        "**T3-3 redesign** (BR-4). Three changes, all making lambda a "
        "first-class axis: (i) lambda-uniform tokenisation replaces the index "
        "stride, so token t always means the same spectral region; (ii) the "
        "learned `spec_pos_embed` table becomes a *buffer* derived from each "
        "window's centre wavelength, which is what makes the branch transferable "
        "across band counts (F-3, and T3-3's validation criterion); (iii) a "
        "relative-lambda attention bias `b_psi` is added to every spectral-stage "
        "logit. `specf_drop` (N-1b) and `specf_tokens` are wired, and `d_model` "
        "drops 256 -> 192. 2,180,866 -> 1,241,640 parameters. Pinned by "
        "`tests/unit/test_specformer_lambda.py`."
    ),
    "SpecFormerBranch.forward": (
        "**T3-3 redesign** (BR-4) — see `SpecFormerBranch.__init__`. The grid "
        "spectra are pooled into lambda-uniform windows before tokenisation, the "
        "positional code is added before the CLS token rather than after, and "
        "the spectral blocks receive the relative-lambda bias."
    ),
    "MultiScaleSpectralTokenizer.__init__": (
        "**T3-3** (BR-4(iii)). `stride` defaults to 1: the strided index "
        "tokenisation it implemented is replaced by lambda-uniform window "
        "pooling upstream, so the three kernel widths now run over a grid that "
        "is uniform in wavelength and a kernel width is a bandwidth. The "
        "three-tower structure and the channel split are unchanged."
    ),
    "_PreLNBlock.forward": (
        "**T3-3** (BR-4(ii)). One optional argument, `attn_bias`, forwarded to "
        "`nn.MultiheadAttention` as a float `attn_mask` — which is exactly the "
        "additive-logit contract that argument implements, so no attention "
        "arithmetic is re-implemented. `attn_bias=None` is the previous "
        "behaviour, and the spatial stage still passes nothing."
    ),
    "CrossModalInteraction.__init__": (
        "**T3-4 redesign** (FU-1(b), FU-2, FU-4, FU-5). The Perceiver is gone: "
        "with five modality tokens, latent cross-attention compresses nothing, "
        "and 0-E measured the four latents as collapsed onto one function "
        "(M-1). In its place: per-modality `BatchNorm1d` (a *dataset* "
        "statistic, so a low-SNR sample is not amplified to unit norm — M-2a), "
        "a **sigmoid** gate fed the pre-normalisation log-norms (M-2a, M-2b), "
        "low-rank bilinear projections over all ten modality pairs (M-3, which "
        "the first-order fusion could not express), and no `output_proj` "
        "(N-10). 2,190,916 -> 496,005 parameters. `fusion_heads` is deleted "
        "rather than wired (N-1a): nothing in the module can consume a head "
        "count. Pinned by `tests/unit/test_fusion.py`."
    ),
    "CrossModalInteraction.forward": (
        "**T3-4 redesign** (FU-1(b)) — see `CrossModalInteraction.__init__`."
    ),
    "MaskedSpectralECA.forward": (
        "**T3-7** (FE-2). Takes `mask` instead of re-deriving the foreground "
        "from `sum_c |x_c| > 1e-5`. `mask=None` reproduces that threshold "
        "bit-for-bit, so the archived arrays still evaluate identically; with "
        "the persisted fill map the two statistics become functions of the "
        "seed's pixels rather than of whether the background is still at zero, "
        "which is what makes them immune to a brightness transform (C-8, M-11). "
        "Pinned by `tests/unit/test_masked_ops.py`."
    ),
    "extract_grid_spectra": (
        "**T3-7 + T3-6** (FE-2, BR-2(2)). Takes the explicit mask (see "
        "`MaskedSpectralECA.forward`) and **returns the cell foreground mass "
        "alongside the spectra**. That mass was always computed as the "
        "normaliser and always discarded; returning it is what lets Branch A "
        "pool cells by coverage rather than count a corner cell holding four "
        "seed pixels as equal to one that is entirely seed (M-12, §2.2.9). The "
        "arithmetic of the mean itself is unchanged."
    ),
    "masked_spectral_stats": (
        "**T3-7** (FE-2). Takes the explicit mask; the nine statistics are "
        "otherwise computed exactly as before. The `== 0` comparison against a "
        "binary mask becomes `<= 0`, which is identical on a binary mask and "
        "correct on the soft fill map P-3 writes. Branch B no longer consumes "
        "this function (BR-1), but Phase 0's rank probe does, and its "
        "foreground-mean output is the new Branch B's input signal."
    ),
    "PhysicalWavelengthPE.__init__": (
        "**T3-3** (BR-4(i)). The encoding moves to the free function "
        "`sinusoidal_wavelength_encoding` so Branch D's token centres can use "
        "the *same* encoding at a different set of wavelengths — a band-indexed "
        "buffer cannot serve tokens that each cover a lambda-window. The "
        "arithmetic is identical and the `pe` buffer is byte-for-byte "
        "unchanged, which the golden init digests confirm."
    ),
    "SpectralQuadNet._init_weights": (
        "**T3-2** (BR-3). `nn.Conv3d` joins the kaiming arm, for the "
        "spectral-spatial stem. Without it the stem's three convolutions would "
        "keep torch's default init while every other convolution in the model "
        "got kaiming fan-out — a silent inconsistency in the one branch the "
        "tier moves capacity into."
    ),
    "_mixup": (
        "**T3-7 + T3-4** (FE-2, FU-4). One optional argument, `return_perm`. "
        "The fill map and the morphometrics have to be mixed with the *same* "
        "pairing the pixels were, and re-drawing `torch.randperm` would pair "
        "each patch's spectrum with another patch's mask. At "
        "`return_perm=False` — every call site outside `train_one_epoch` — the "
        "return value and the RNG draws are unchanged."
    ),
    "mixed_aug": "**T3-7 + T3-4** — see `_mixup`; the argument is forwarded.",
    "_run_eval": (
        "**T3-7 + T3-4** (FE-2, FU-4). Unpacks the batch through "
        "`engine/batch.py::unpack_batch`, which accepts both the 2-tuple the "
        "dataset yields without side arrays and the 4-tuple it yields with "
        "them, and forwards whichever are present to the model. With neither "
        "configured the call is identical to the previous `model(x)`."
    ),
    "compute_branch_influence": (
        "**T3-7 + T3-4** — see `_run_eval`. The local `mask` was also the name "
        "of the branch-ablation vector; it is renamed `ablation` so the two "
        "cannot be confused now that a real mask is in scope."
    ),
}


# ══════════════════════════════════════════════════════════════════════
#  Normalisation
# ══════════════════════════════════════════════════════════════════════


#: The four node types with a ``body`` that can open with a docstring.
_Documented = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module


class _Normalise(ast.NodeTransformer):
    """Erase docstrings and annotations; see the module docstring for why."""

    def _strip_docstring(self, node: _Documented) -> _Documented:
        self.generic_visit(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Return annotations are erased at *every* depth, not just on the
            # symbol being compared — nested closures such as
            # `masked_spectral_stats.gather_percentile` are annotatable too, and
            # under PEP 563 their annotation is as inert as any other.
            node.returns = None
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = _strip_docstring
    visit_AsyncFunctionDef = _strip_docstring
    visit_ClassDef = _strip_docstring
    visit_Module = _strip_docstring

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        self.generic_visit(node)
        if node.value is None:
            # A bare `x: int` declaration carries no runtime effect at all.
            return None
        return ast.Assign(targets=[node.target], value=node.value, type_comment=None)


def normalise(node: ast.AST) -> ast.AST:
    # `_Normalise` erases `returns` on every function it visits, including the
    # outermost one, so no special case is needed here.
    out: ast.AST = _Normalise().visit(copy.deepcopy(node))
    ast.fix_missing_locations(out)
    return out


def dump(node: ast.AST) -> str:
    return ast.dump(normalise(node), annotate_fields=False)


def render(node: ast.AST) -> list[str]:
    return ast.unparse(normalise(node)).splitlines()


def split_members(node: ast.AST) -> dict[str, ast.AST]:
    """Decompose a class into ``{member_name: node}``; free functions pass through.

    Non-function statements in a class body (``_PROFILES``, ``_INTENSITY_SCALE``, …)
    are grouped under the pseudo-member ``<class-body>`` so class-level constants
    are compared too.
    """
    if not isinstance(node, ast.ClassDef):
        return {"": node}

    members: dict[str, ast.AST] = {}
    leftovers: list[ast.stmt] = []
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members[stmt.name] = stmt
        else:
            leftovers.append(stmt)
    if leftovers:
        members["<class-body>"] = ast.Module(body=leftovers, type_ignores=[])

    # The class header itself (bases, decorators) is compared separately.
    members["<class-header>"] = ast.ClassDef(
        name=node.name,
        bases=node.bases,
        keywords=node.keywords,
        body=[ast.Pass()],
        decorator_list=node.decorator_list,
        type_params=[],
    )
    return members


# ══════════════════════════════════════════════════════════════════════
#  Comparison
# ══════════════════════════════════════════════════════════════════════


def new_symbols(rel_path: str) -> dict[str, ast.AST]:
    tree = ast.parse((SRC / rel_path).read_text())
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def run_check(baseline_ref: str, verbose: bool) -> int:
    results: list[tuple[str, str, str, list[str]]] = []  # verdict, qualname, note, diff
    new_cache: dict[str, dict[str, ast.AST]] = {}

    for baseline_file, mapping in RELOCATIONS.items():
        old_syms = baseline_symbols(baseline_file, baseline_ref)

        for old_name, rel_path in mapping.items():
            new_name = RENAMES.get((baseline_file, old_name), old_name)

            if rel_path not in new_cache:
                new_cache[rel_path] = new_symbols(rel_path)
            new_syms = new_cache[rel_path]

            if old_name not in old_syms:
                results.append(("DRIFT", old_name, f"missing from baseline {baseline_file}", []))
                continue
            if new_name not in new_syms:
                results.append(
                    ("DRIFT", new_name, f"not found in src/spectralquadnet/{rel_path}", [])
                )
                continue

            old_members = split_members(old_syms[old_name])
            new_members = split_members(new_syms[new_name])
            prefix = f"{new_name}." if isinstance(new_syms[new_name], ast.ClassDef) else ""

            for member in old_members:
                qual = f"{new_name}{'.' + member if member else ''}"
                # `<class-header>` compares bases/decorators; ignore the name itself,
                # which RENAMES already accounts for.
                if member not in new_members:
                    removal = DECLARED_REMOVALS.get(qual)
                    results.append(
                        (
                            ("DECLARED" if removal else "DRIFT"),
                            qual,
                            removal or f"member removed ({rel_path})",
                            [],
                        )
                    )
                    continue

                a, b = old_members[member], new_members[member]
                if member == "<class-header>":
                    # Both are the `ast.ClassDef` stubs `split_members` builds.
                    a = copy.deepcopy(a)
                    b = copy.deepcopy(b)
                    a.name = b.name = "_"  # type: ignore[attr-defined]

                if dump(a) == dump(b):
                    results.append(("IDENTICAL", qual, rel_path, []))
                    continue

                key = f"{new_name}.{member}" if member and prefix else new_name
                if member in ("<class-header>", "<class-body>"):
                    key = f"{new_name}.{member}"
                reason = DECLARED_DEVIATIONS.get(key)
                diff = list(
                    difflib.unified_diff(
                        render(a),
                        render(b),
                        fromfile=f"baseline@{baseline_ref[:7]}:{old_name}",
                        tofile=f"{rel_path}:{qual}",
                        lineterm="",
                    )
                )
                results.append(
                    (("DECLARED" if reason else "DRIFT"), qual, reason or rel_path, diff)
                )

            for member in new_members.keys() - old_members.keys():
                qual = f"{new_name}{'.' + member if member else ''}"
                results.append(("NEW", qual, rel_path, []))

    # ── Report ────────────────────────────────────────────────────────
    order = {"DRIFT": 0, "DECLARED": 1, "NEW": 2, "IDENTICAL": 3}
    counts = dict.fromkeys(order, 0)
    for verdict, *_ in results:
        counts[verdict] += 1

    print("═" * 78)
    print(f"  AST no-op-move check   (REFACTOR_PLAN.md §3.2.1)   baseline {baseline_ref[:7]}")
    print("═" * 78)

    for verdict, qual, note, diff in sorted(results, key=lambda r: (order[r[0]], r[1])):
        if verdict == "IDENTICAL" and not verbose:
            continue
        mark = {"IDENTICAL": "✓", "DECLARED": "~", "NEW": "+", "DRIFT": "✗"}[verdict]
        print(f"\n{mark} {verdict:<10} {qual}")
        print(f"    {note}")
        for line in diff:
            print(f"    │ {line}")

    print()
    print("─" * 78)
    print(
        f"  {counts['IDENTICAL']:>3} identical   "
        f"{counts['DECLARED']:>3} declared   "
        f"{counts['NEW']:>3} new   "
        f"{counts['DRIFT']:>3} drift"
    )
    print("─" * 78)

    if counts["DRIFT"]:
        print("\n✗ AST check FAILED — undeclared changes above.")
        return 1
    print("\n✓ AST check OK — every change is either identical or declared above.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ref", default=BASELINE_REF)
    ap.add_argument("--verbose", "-v", action="store_true", help="also list IDENTICAL symbols")
    args = ap.parse_args()
    return run_check(args.baseline_ref, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
