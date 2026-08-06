#!/usr/bin/env python3
"""Config round-trip check — REFACTOR_PLAN.md §3.3.

Migration-time verification (not a permanent test): asserts that every one of the
81 keys in the pre-refactor ``CONFIG`` dict resolves to *exactly one* field in the
composed Hydra config, with an identical value — nothing silently dropped,
renamed or re-defaulted by the mechanical migration.

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

#: Commit that recorded the pre-refactor monolith (see git log: "chore: record
#: pre-refactor baseline"). Everything §3.2/§3.3 compares against comes from here.
BASELINE_REF = "886560fe531c99197f20c2ebd06e0bc7ded8ac8f"
BASELINE_PATH = "HSI_modality_training/hsi_training.py"

EXPERIMENT = "experiment/output_v12_spa40"

# ══════════════════════════════════════════════════════════════════════
#  The rename table  (old CONFIG key → new dotted config path)
#
#  Rules applied, in order:
#    1. `s{1,2,3}_` prefix → `stage{1,2,3}.` group, remainder verbatim
#       (incl. original capitalisation: sgdr_T0, subcenter_K).
#    2. Everything else is assigned to the group that owns its concern, per
#       REFACTOR_PLAN.md §2's `configs/` layout; the field name is unchanged.
#    3. Keys with no group in §2's layout (Shared / TTA / seed / device) stay at
#       the experiment root.
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
    "fusion_heads": "model.fusion_heads",
    "fusion_drop": "model.fusion_drop",
    # ── Reproducibility & placement ───────────────────────────────────
    "device": "device",
    "seed": "seed",
}

#: New config fields with no pre-refactor CONFIG ancestor. Each needs an explicit
#: justification here, otherwise the key diff below fails.
INTENDED_ADDITIONS: dict[str, str] = {
    "run_name": "§4.3 — run identity; feeds output_dir instead of hardcoding a path.",
    "output_root": "§4.3 — output_dir = ${output_root}/${run_name}.",
}

#: Config subtrees that are net-new capabilities rather than relocated CONFIG keys.
EXCLUDED_SUBTREES: dict[str, str] = {
    "tracking": "§4.1 — experiment tracking is additive; the monolith used bare print().",
}

#: Keys whose *value* intentionally differs from the pre-refactor CONFIG.
INTENDED_VALUE_CHANGES: dict[str, str] = {
    "output_dir": (
        "§4.3 — hardcoded absolute machine-specific path replaced by "
        "${output_root}/${run_name}; points at the Phase 1 relocation target."
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
    """Parse the ``CONFIG`` dict literal out of the pre-refactor monolith.

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
    unmapped = sorted(set(baseline) - set(RENAME))
    stale = sorted(set(RENAME) - set(baseline))
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
