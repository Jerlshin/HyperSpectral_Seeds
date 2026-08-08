#!/usr/bin/env python3
"""Config round-trip check against a pinned reference implementation.

Asserts that every key in the reference implementation's flat ``CONFIG``
dict resolves to *exactly one* field in the composed Hydra config, with an
identical value — nothing silently dropped, renamed or re-defaulted relative
to that reference.

The rename table lives in :data:`RENAME` below and is the single source of truth;
``docs/config_rename_table.md`` is generated from it via ``--emit-markdown``.

Usage
-----
    python scripts/check_config_roundtrip.py                     # run the check
    python scripts/check_config_roundtrip.py --emit-markdown docs/config_rename_table.md

Exit code is 0 only when every key maps 1:1 and every value matches (modulo the
two intentional deviations recorded in :data:`INTENDED_VALUE_CHANGES`).
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Pinned commit holding the reference implementation this check compares against.
BASELINE_REF = "886560fe531c99197f20c2ebd06e0bc7ded8ac8f"
BASELINE_PATH = "HSI_modality_training/hsi_training.py"

EXPERIMENT = "experiment/output_v12_spa40"

# ══════════════════════════════════════════════════════════════════════
#  The rename table  (old CONFIG key → new dotted config path)
#
#  Rules applied, in order:
#    1. `s{1,2,3}_` prefix → `stage{1,2,3}.` group, remainder verbatim
#       (incl. original capitalisation: sgdr_T0, subcenter_K).
#    2. Everything else is assigned to the group that owns its concern in the
#       `configs/` layout; the field name is unchanged.
#    3. Keys with no owning group (Shared / TTA / seed / device) stay at the
#       experiment root.
# ══════════════════════════════════════════════════════════════════════

RENAME: dict[str, str] = {
    # ── Paths ─────────────────────────────────────────────────────────
    "patches_data": "data.patches_data",
    "labels_path": "data.labels_path",
    "wavelength_path": "data.wavelength_path",
    "output_dir": "output_dir",
    # ── Dataset ───────────────────────────────────────────────────────
    "num_bands": "data.num_bands",
    "num_classes": "data.num_classes",
    # ── Stage 1 — 3-phase progressive augmentation ────────────────────
    "s1_epochs": "stage1.epochs",
    "s1_phase1_frac": "stage1.phase1_frac",
    "s1_phase2_frac": "stage1.phase2_frac",
    "s1_batch": "stage1.batch",
    "s1_max_lr": "stage1.max_lr",
    "s1_mid_lr": "stage1.mid_lr",
    "s1_min_lr": "stage1.min_lr",
    "s1_dropout": "stage1.dropout",
    "s1_mixup": "stage1.mixup",
    "s1_patience": "stage1.patience",
    "s1_accum": "stage1.accum",
    "s1_focal_gamma": "stage1.focal_gamma",
    "s1_label_smooth_hi": "stage1.label_smooth_hi",
    "s1_label_smooth_lo": "stage1.label_smooth_lo",
    "s1_ema_reinit_phases": "stage1.ema_reinit_phases",
    # ── Stage 1 · Phase 3 contrastive losses ──────────────────────────
    "s1_p3_supcon_weight": "stage1.p3_supcon_weight",
    "s1_p3_proto_weight": "stage1.p3_proto_weight",
    # ── Stage 1 · Phase 3 hard-class oversampling ─────────────────────
    "s1_p3_oversample": "stage1.p3_oversample",
    "s1_p3_oversample_power": "stage1.p3_oversample_power",
    "s1_p3_oversample_max_w": "stage1.p3_oversample_max_w",
    "s1_p3_hard_f1_thresh": "stage1.p3_hard_f1_thresh",
    "s1_p3_oversample_eps": "stage1.p3_oversample_eps",
    # ── Stage 1 · Phase 3 dropout boost ───────────────────────────────
    "s1_p3_dropout": "stage1.p3_dropout",
    # ── Architecture ──────────────────────────────────────────────────
    "branch_drop_prob": "model.branch_drop_prob",
    "subcenter_K": "model.subcenter_K",
    "max_cutout_bands": "data.max_cutout_bands",  # read only by RiceSeedDataset aug
    "noise_std": "data.noise_std",  # read only by RiceSeedDataset aug
    # ── Auxiliary heads ───────────────────────────────────────────────
    "aux_head_hidden": "model.aux_head_hidden",
    "aux_loss_weight_init": "stage1.aux_loss_weight_init",
    "aux_loss_weight_final": "stage1.aux_loss_weight_final",
    # ── Stage 2 ───────────────────────────────────────────────────────
    "s2_epochs": "stage2.epochs",
    "s2_batch": "stage2.batch",
    "s2_head_lr": "stage2.head_lr",
    "s2_back_lr": "stage2.back_lr",
    "s2_min_lr": "stage2.min_lr",
    "s2_warmup_ep": "stage2.warmup_ep",
    "s2_sgdr_T0": "stage2.sgdr_T0",
    "s2_sgdr_Tmult": "stage2.sgdr_Tmult",
    "s2_dropout": "stage2.dropout",
    "s2_patience": "stage2.patience",
    "s2_arcface_s": "stage2.arcface_s",
    "s2_arcface_m": "stage2.arcface_m",
    "s2_arcface_m0": "stage2.arcface_m0",
    "s2_arcface_m_delta": "stage2.arcface_m_delta",
    "s2_margin_warmup_ep": "stage2.margin_warmup_ep",
    "s2_focal_gamma": "stage2.focal_gamma",
    "cdws_max_weight": "stage2.cdws_max_weight",
    "cdws_eps": "stage2.cdws_eps",
    "supcon_weight": "stage2.supcon_weight",
    "supcon_temp": "stage2.supcon_temp",
    "proto_weight": "stage2.proto_weight",
    "proto_temp": "stage2.proto_temp",
    "bal_n_cls": "stage2.bal_n_cls",
    "bal_n_spc": "stage2.bal_n_spc",
    # ── Stage 3 ───────────────────────────────────────────────────────
    "s3_epochs": "stage3.epochs",
    "s3_swa_lr": "stage3.swa_lr",
    "s3_cycle_len": "stage3.cycle_len",
    "s3_sam_rho": "stage3.sam_rho",
    "s3_greedy": "stage3.greedy",
    "s3_aux_loss_weight": "stage3.aux_loss_weight",
    # ── Shared ────────────────────────────────────────────────────────
    "weight_decay": "weight_decay",
    "grad_clip": "grad_clip",
    "ema_decay": "ema_decay",
    # ── TTA ───────────────────────────────────────────────────────────
    "tta_spatial": "tta_spatial",
    "tta_spectral": "tta_spectral",
    # ── Transformer branch (SpecFormer) + fusion ──────────────────────
    "wl_embed_dim": "model.wl_embed_dim",
    "specf_patch": "model.specf_patch",
    "specf_dim": "model.specf_dim",
    "specf_heads": "model.specf_heads",
    "specf_layers": "model.specf_layers",
    "specf_drop": "model.specf_drop",
    "fusion_drop": "model.fusion_drop",
    # ── Reproducibility & placement ───────────────────────────────────
    "device": "device",
    "seed": "seed",
}

#: Pre-refactor CONFIG keys with **no** home in the composed config, because the
#: thing they configured no longer exists. This is the other half of §2.7's
#: remedy for a dead key: wire it, or delete it. An entry here is a deletion,
#: and it needs a reason for the same purpose :data:`INTENDED_ADDITIONS` needs
#: one — so that "the key is gone" cannot quietly mean "the key was dropped".
INTENDED_REMOVALS: dict[str, str] = {
    "fusion_heads": (
        "T3-4 / FU-1(b) — N-1a's dead key, deleted rather than wired. It named "
        "the head count of `CrossModalInteraction`'s multi-head attention, and "
        "the gated low-rank bilinear fusion that replaced the Perceiver has no "
        "attention at all: with five modality tokens, latent cross-attention "
        "compresses nothing (§3.4 FU-1). Nothing in the model can consume a "
        "head count, so there is nothing to wire it to."
    ),
}

#: New config fields with no pre-refactor CONFIG ancestor. Each needs an explicit
#: justification here, otherwise the key diff below fails.
INTENDED_ADDITIONS: dict[str, str] = {
    "run_name": "§4.3 — run identity; feeds output_dir instead of hardcoding a path.",
    "output_root": "§4.3 — output_dir = ${output_root}/${run_name}.",
    # ── Tier 2 (IMPROVEMENT_PLAN §4.2) ────────────────────────────────
    "aux_gradnorm_alpha": (
        "T2-6 / OP-2 — GradNorm exponent for the per-branch auxiliary weights. "
        "At 0.0 the weights stay at the hardcoded A/B = 2x vector it replaces, "
        "so the pre-Tier-2 behaviour remains expressible."
    ),
    "data.cutmix_bands": "T2-7 / OP-6 — width of the same-class spectral CutMix window.",
    "data.cutmix_spatial": "T2-7 / OP-6 — side of the same-class spatial CutMix paste.",
    "model.subcenter_tau_init": "T2-9 / HD-2(i) — sub-centre pooling temperature at stage entry.",
    "model.subcenter_tau_final": (
        "T2-9 / HD-2(i) — temperature at stage end; at tau -> 0 the pooling is "
        "the hard max_k the head was defined on."
    ),
    "model.subcenter_balance_weight": (
        "T2-9 / HD-2(ii) — weight on sum_c KL(pi_c || uniform), the "
        "mixture-of-experts load-balancing term sub-centre ArcFace was missing."
    ),
    "stage1.arcface_m": (
        "T2-10 / HD-1 — Stage 1's margin under the unified head. 0.0 is not a "
        "placeholder: it makes the head a cosine (NormFace) classifier, which "
        "is what removes the Stage-1 -> Stage-2 discontinuity of §2.4.6."
    ),
    "stage2.arcface_m_min": "T2-8 / HD-3 — lower clip on the signed R-P margin rule.",
    "stage2.arcface_m_max": "T2-8 / HD-3 — upper clip on the signed R-P margin rule.",
    "stage2.pairwise_margin_delta": (
        "T2-8 / HD-3 — scale of the row-normalised confusion term that aims the "
        "margin at the classes each class is actually confused with."
    ),
    "stage3.margin_kappa_final": (
        "T2-1 / OP-4.2-4.3 — the multiplicative margin anneal's endpoint. Stage "
        "3 keeps Stage 2's per-class vector and scales all of it, stepping only "
        "at cycle boundaries."
    ),
    "stage3.swa_warmup_cycles": (
        "T2-3 / OP-4.5 — cycles discarded from the SWA average before the first "
        "candidate is considered, keeping Adam's 1/(1-beta2) second-moment "
        "transient out of it."
    ),
    "stage3.sam_adaptive": (
        "T2-4 / OP-5 — selects ASAM. SAM's rho-ball is not scale-invariant and "
        "the ArcFace head is, so a raw-space budget has no meaning there."
    ),
    # ── Tier 4 (IMPROVEMENT_PLAN §4.2) ────────────────────────────────
    "data.groups_path": (
        "T4-1 / P-1 — the per-patch scan id `scripts/prepare_dataset.py` now "
        "writes. Required by the grouped scheme; read under `stratified` too, "
        "where it is used only to measure how many scans cross the train/eval "
        "boundary (0-H measured 107 of 107)."
    ),
    "data.split_scheme": (
        "T4-1 / P-1 — `stratified` is the reference run's patch-level split, "
        "kept because the archived checkpoints were selected on it; `grouped` "
        "is the scan-disjoint protocol. The default stays `stratified` so this "
        "config keeps reproducing the run it describes; "
        "`configs/data/spa40_90class_pfix.yaml` is the P-fix protocol."
    ),
    "data.split_eval_frac": (
        "T4-1 / P-1 — share held out for val+test. 0.30 reproduces the "
        "reference 70/15/15 exactly on the stratified path."
    ),
    "data.split_fold": (
        "T4-1 / P-1 — rotates which scans are held out; sweeping it is the "
        "leave-one-scan-out cross-validation §3.1 falls back to when a class "
        "has too few scans for a three-way disjoint split. Must be 0 under "
        "`stratified`, which has no groups to rotate."
    ),
    "data.calib_frac": (
        "T4-5 / P-5 — share of the training pool carved off as `calib`, where "
        "the per-class margins, the CDWS weights and the Phase-3 oversampling "
        "weights are fitted. 0.0 leaves them on `val`, i.e. on the split that "
        "also selects the checkpoint (C-9), which is what the reference run "
        "did."
    ),
    # ── Tier 3 (IMPROVEMENT_PLAN §4.2) ────────────────────────────────
    "data.masks_path": (
        "T3-7 / FE-2 — the persisted fill map alpha `scripts/prepare_dataset.py` "
        "writes under P-3. Passing it makes the four masked modules functions of "
        "the seed's pixels rather than of `sum_c |x_c| > 1e-5`, and so immune to "
        "any global brightness transform. Empty falls back to that threshold, "
        "exactly, which is why the pre-Tier-3 arrays still reproduce."
    ),
    "data.morphology_path": (
        "T3-1/T3-4 / P-4 — the eight morphometrics, which Branch B's index bank "
        "and FU-4's fifth fusion token both consume. Empty substitutes zeros, "
        "the mean of the train-standardised feature."
    ),
    "model.grid_size_a": (
        "T3-6 / BR-2 — Branch A's grid, 4x4 -> 8x8. Costs no parameters (cells "
        "are processed independently) and takes the spatial compression from "
        "256:1 to 64:1."
    ),
    "model.grid_size_d": (
        "T3-6 / BR-2 — Branch D's grid, held at 4x4. A and D received a "
        "byte-identical tensor before Tier 3 (§2.2.2); separate keys are what "
        "make that impossible to reintroduce silently."
    ),
    "model.index_bank_size": (
        "T3-1 / BR-1(i) — number of learned normalised-difference indices, the "
        "gain-invariant replacement for the rank-2 moment tensor of §2.2.5."
    ),
    "model.continuum_depths": (
        "T3-1 / BR-1(ii) — how many of the deepest hull-removed absorption features Branch B reads."
    ),
    "model.n_morphometrics": (
        "T3-1/T3-4 / P-4 — width of the morphometric vector; the eight columns "
        "`data/prep/segmentation.py::MORPHOMETRIC_NAMES` writes."
    ),
    "model.stem_channels": (
        "T3-2 / BR-3 — width the 3-D spectral-spatial stem folds the spectral "
        "axis into before the 2-D tail. C-3: before this the network contained "
        "no joint spectral-spatial operator at all."
    ),
    "model.fusion_rank": (
        "T3-4 / FU-1(b) — rank of the bilinear projections U_m. The second-order "
        "term M-3 found missing, at 5*d*r rather than a full 10*d^2."
    ),
    "model.fusion_gate_hidden": (
        "T3-4 / FU-1(b)+FU-2 — hidden width of the sigmoid gate MLP, which reads "
        "the five normalised tokens and the five pre-normalisation log-norms."
    ),
}

#: Config subtrees that are net-new capabilities rather than relocated CONFIG keys.
EXCLUDED_SUBTREES: dict[str, str] = {
    "tracking": "§4.1 — experiment tracking is additive; the monolith used bare print().",
    "runtime": (
        "Execution knobs — DataLoader workers, pinned staging, torch.compile, fused "
        "AdamW, DDP topology, allocator sweeps, console rendering. Excluded rather "
        "than mapped because the reference implementation had no counterpart to map "
        "to: it fed the model one sample at a time on the training device with "
        "num_workers=0 and no notion of a second GPU. The group carries the "
        "invariant that nothing in it may change a reported number, which is what "
        "keeps it out of the experiment's identity — the two fields that *would* "
        "change one (allow_tf32, channels_last) default to off."
    ),
}

#: Keys whose *value* intentionally differs from the pre-refactor CONFIG.
INTENDED_VALUE_CHANGES: dict[str, str] = {
    "output_dir": (
        "§4.3 — hardcoded absolute machine-specific path replaced by "
        "${output_root}/${run_name}; points at the Phase 1 relocation target."
    ),
    "s2_arcface_m_delta": (
        "T2-8 / HD-3 — 0.10 → 0.20. The key kept its name and changed the rule "
        "it parameterises: it was the F1-driven rule's m_delta in "
        "`M(c) = m_base + m_delta (1 - F1_c)`, and is now the signed rule's in "
        "`M(c) = clip(m_base + m_delta (R_c - P_c), 0.20, 0.50)`. The plan "
        "specifies 0.20 for the latter (§3.5 HD-3). Reusing the key rather than "
        "adding a second one keeps a single margin-scale knob; the sign change "
        "is pinned by `tests/unit/test_margin_rule.py::test_margin_rule_sign`."
    ),
    "specf_dim": (
        "T3-3 / BR-4 — 256 → 192. Branch D's token embeddings are now derived "
        "from each λ-window's centre wavelength and its spectral attention "
        "carries a relative-λ bias, so the branch does not have to spend "
        "capacity rediscovering the wavelength axis from an arbitrary index "
        "table. The key kept its name and the branch lost 0.94 M parameters, "
        "which is what funds BR-3's 3-D stem in Branch C (§3.8). Pinned by "
        "`tests/unit/test_specformer_lambda.py`."
    ),
    "device": (
        "§4.3 — YAML cannot hold a torch.device object, so the config carries the "
        'resolution strategy ("auto") and utils/device.py performs the lookup. '
        "Phase 5 widens that lookup from the baseline's cuda-or-cpu to "
        "Metal → CUDA → CPU, so an Apple Silicon host uses its GPU instead of "
        "falling through to the CPU. An explicit device=cuda/cpu/mps still wins."
    ),
}


# ══════════════════════════════════════════════════════════════════════
#  Extraction
# ══════════════════════════════════════════════════════════════════════


def load_baseline_config(ref: str) -> dict[str, Any]:
    """Parse the ``CONFIG`` dict literal out of the reference implementation.

    Values are ``ast.literal_eval``'d where possible; non-literal entries (only
    ``device``, a ``torch.device(...)`` call) are kept as their source text.
    """
    if ref == "WORKTREE":
        source = (REPO_ROOT / BASELINE_PATH).read_text()
    else:
        source = subprocess.run(
            ["git", "show", f"{ref}:{BASELINE_PATH}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    tree = ast.parse(source)
    for node in ast.walk(tree):
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not any(getattr(t, "id", None) == "CONFIG" for t in targets):
            continue
        assert isinstance(node.value, ast.Dict)
        out: dict[str, Any] = {}
        for k, v in zip(node.value.keys, node.value.values, strict=True):
            assert isinstance(k, ast.Constant), f"non-literal CONFIG key: {k!r}"
            key = str(k.value)
            try:
                out[key] = ast.literal_eval(v)
            except ValueError:
                out[key] = f"<expr> {ast.unparse(v)}"
        return out
    raise RuntimeError(f"No CONFIG dict found in {BASELINE_PATH}@{ref}")


def compose_experiment() -> dict[str, Any]:
    """Compose the reference Hydra experiment exactly as ``train.py`` will."""
    from spectralquadnet.config.schema import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name=EXPERIMENT)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config to dotted leaf paths."""
    flat: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


# ══════════════════════════════════════════════════════════════════════
#  Check
# ══════════════════════════════════════════════════════════════════════


def run_check(ref: str) -> int:
    baseline = load_baseline_config(ref)
    composed = flatten(compose_experiment())

    errors: list[str] = []
    notes: list[str] = []

    # ── 1. Rename table covers the baseline exactly ───────────────────
    unmapped = sorted(set(baseline) - set(RENAME) - set(INTENDED_REMOVALS))
    stale = sorted(set(RENAME) - set(baseline))
    orphan_removals = sorted(set(INTENDED_REMOVALS) - set(baseline))
    if orphan_removals:
        errors.append(
            f"INTENDED_REMOVALS entries not present in the baseline CONFIG: {orphan_removals}"
        )
    if unmapped:
        errors.append(f"CONFIG keys with no entry in RENAME: {unmapped}")
    if stale:
        errors.append(f"RENAME entries not present in the baseline CONFIG: {stale}")

    # ── 2. Every mapped target exists, exactly once ───────────────────
    targets = list(RENAME.values())
    duplicated = sorted({t for t in targets if targets.count(t) > 1})
    if duplicated:
        errors.append(f"Two CONFIG keys map to the same config path: {duplicated}")

    missing_targets = sorted(t for t in targets if t not in composed)
    if missing_targets:
        errors.append(
            f"Config paths referenced by RENAME but absent from the composed config: {missing_targets}"
        )

    # ── 3. No unexplained new fields ──────────────────────────────────
    accounted = set(targets) | set(INTENDED_ADDITIONS)
    extras = sorted(
        path
        for path in composed
        if path not in accounted and path.split(".")[0] not in EXCLUDED_SUBTREES
    )
    if extras:
        errors.append(
            f"Composed config has fields with no CONFIG ancestor and no justification: {extras}"
        )

    # ── 4. Values are identical (bar documented deviations) ───────────
    for old_key, path in sorted(RENAME.items()):
        if path not in composed:
            continue
        old_value, new_value = baseline[old_key], composed[path]
        if old_key in INTENDED_VALUE_CHANGES:
            notes.append(f"  ~ {old_key}: {old_value!r} → {path} = {new_value!r}")
            continue
        if type(old_value) is not type(new_value) or old_value != new_value:
            errors.append(
                f"Value drift for {old_key} → {path}: "
                f"{old_value!r} ({type(old_value).__name__}) != {new_value!r} ({type(new_value).__name__})"
            )

    # ── Report ────────────────────────────────────────────────────────
    print(f"Baseline : {BASELINE_PATH}@{ref}  ({len(baseline)} CONFIG keys)")
    print(f"Composed : configs/{EXPERIMENT}.yaml  ({len(composed)} leaf fields)")
    print(f"Excluded : {', '.join(f'{k}.* ({v})' for k, v in EXCLUDED_SUBTREES.items())}")
    print(f"Added    : {', '.join(INTENDED_ADDITIONS)}")
    print(f"Removed  : {', '.join(INTENDED_REMOVALS) or '—'}")
    if notes:
        print("Intentional value changes (REFACTOR_PLAN.md §4.3):")
        print("\n".join(notes))

    if errors:
        print(f"\n✗ CONFIG round-trip FAILED — {len(errors)} problem(s):")
        for err in errors:
            print(f"  • {err}")
        return 1

    print(f"\n✓ CONFIG round-trip OK — all {len(baseline)} keys map 1:1 with identical values.")
    return 0


# ══════════════════════════════════════════════════════════════════════
#  Markdown emitter (keeps docs/config_rename_table.md in sync)
# ══════════════════════════════════════════════════════════════════════


def emit_markdown(out_path: Path, ref: str) -> None:
    baseline = load_baseline_config(ref)
    composed = flatten(compose_experiment())

    groups: dict[str, list[str]] = {}
    for old_key, path in RENAME.items():
        group = path.split(".")[0] if "." in path else "(experiment root)"
        groups.setdefault(group, []).append(old_key)

    lines = [
        "# CONFIG key-rename table",
        "",
        "**Generated** by `scripts/check_config_roundtrip.py --emit-markdown` — do not edit by hand.",
        "",
        f"Maps every key of the pre-refactor `CONFIG` dict (`{BASELINE_PATH}` @ `{ref[:7]}`, "
        f"{len(baseline)} keys) to its single home in `configs/`. Enforced by "
        "`scripts/check_config_roundtrip.py` (REFACTOR_PLAN.md §3.3).",
        "",
        "| Old `CONFIG` key | New config path | Value | File |",
        "|---|---|---|---|",
    ]
    file_of = {
        "data": "`configs/data/spa40_90class.yaml`",
        "model": "`configs/model/spectral_quadnet_v4.yaml`",
        "stage1": "`configs/stage1/progressive_3phase.yaml`",
        "stage2": "`configs/stage2/arcface_supcon.yaml`",
        "stage3": "`configs/stage3/sam_swa.yaml`",
        "(experiment root)": f"`configs/{EXPERIMENT}.yaml`",
    }
    for old_key in baseline:
        if old_key in INTENDED_REMOVALS:
            lines.append(f"| `{old_key}` | *(deleted)* 🗑️ | — | — |")
            continue
        path = RENAME[old_key]
        group = path.split(".")[0] if "." in path else "(experiment root)"
        value = composed.get(path, "—")
        marker = " ⚠️" if old_key in INTENDED_VALUE_CHANGES else ""
        lines.append(f"| `{old_key}` | `{path}`{marker} | `{value!r}` | {file_of[group]} |")

    lines += [
        "",
        "⚠️ = value intentionally differs from the pre-refactor constant:",
        "",
    ]
    for key, why in INTENDED_VALUE_CHANGES.items():
        lines.append(f"- **`{key}`** — {why} Pre-refactor value: `{baseline[key]!r}`.")
    lines += [
        "",
        "## Net-new fields (no `CONFIG` ancestor)",
        "",
    ]
    for key, why in INTENDED_ADDITIONS.items():
        lines.append(f"- **`{key}`** — {why}")
    for key, why in EXCLUDED_SUBTREES.items():
        lines.append(f"- **`{key}.*`** — {why}")
    lines += [
        "",
        "## 🗑️ Deleted `CONFIG` keys (nothing left to configure)",
        "",
    ]
    for key, why in INTENDED_REMOVALS.items():
        lines.append(f"- **`{key}`** — {why} Pre-refactor value: `{baseline[key]!r}`.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path} ({len(baseline)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref",
        default=BASELINE_REF,
        help="git ref holding the pre-refactor monolith, or WORKTREE for the on-disk file",
    )
    parser.add_argument(
        "--emit-markdown",
        type=Path,
        default=None,
        help="write the rename table to this markdown file instead of running the check",
    )
    args = parser.parse_args()

    if args.emit_markdown is not None:
        emit_markdown(args.emit_markdown, args.baseline_ref)
        return 0
    return run_check(args.baseline_ref)


if __name__ == "__main__":
    sys.exit(main())
