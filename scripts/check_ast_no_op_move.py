#!/usr/bin/env python
"""AST-level "no-op move" diff — REFACTOR_PLAN.md §3.2.1.

Proves that every symbol relocated in Phase 2 is a *mechanical* move: it extracts
each class/function from the pre-refactor ``hsi_training.py`` / ``band_selection.py``
/ ``data_setup_v3.py`` (via ``git show <baseline-sha>``) and from its new file,
normalises both ASTs, and compares ``ast.dump(node, annotate_fields=False)``.

Granularity is **per method**, not per class: relocating ``SpectralQuadNet`` is
only interesting if ``forward`` — the numerics-critical path — comes through
untouched, and a whole-class comparison would hide that behind ``__init__``'s
unavoidable config rewiring.

Verdicts
────────
``IDENTICAL``  normalised ASTs match exactly. This is the expected result for
               the overwhelming majority of relocated code.
``DECLARED``   the symbol appears in :data:`DECLARED_DEVIATIONS` with a written
               reason, and its full diff is printed for review. Every entry is a
               ``CONFIG``-access or global-injection rewrite that the migration
               plan explicitly sanctions (§5 Phase 2).
``NEW``        exists only post-refactor (e.g. ``SpectralQuadNet.from_config``).
               Reported, never failed — new code is not a relocation.
``DRIFT``      changed without a declaration. **Fails the check.**

Normalisations applied (each provably numerics-inert)
─────────────────────────────────────────────────────
1. **Docstrings erased.** Documentation was deliberately expanded during the move
   (every new module gains a baseline line-range table).
2. **Annotations erased** — parameter, return and ``AnnAssign`` annotations.
   Both the baseline and every new module carry ``from __future__ import
   annotations``, so under PEP 563 annotations are unevaluated strings that
   cannot affect runtime behaviour. This is what lets the migration modernise
   ``Optional[str]`` → ``str | None`` and ``Dict`` → ``dict`` without weakening
   the check. Parameter *names*, *order* and *default values* are still compared.
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
#  Relocation map — baseline file → symbol → new module (REFACTOR_PLAN.md §2.1)
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
        # ── utils/ ─────────────────────────────────────────────────────
        # §2.1 rows pulled forward from Phase 4: the Phase 2 gate's golden
        # capture cannot run without `set_seed`.
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

# Symbols renamed during the move (baseline name → new name).
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

DECLARED_DEVIATIONS: dict[str, str] = {
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
    "build_splits": "`CONFIG['labels_path']` → `cfg.data.labels_path`.",
    "build_loaders": (
        "`CONFIG['bal_n_cls']`/`['bal_n_spc']` → `cfg.stage2.*`; the three "
        "RiceSeedDataset constructions now pass store/data_cfg/device through a "
        "shared `kw` dict instead of reading module globals."
    ),
    "build_phase3_loader": (
        "`CONFIG['s1_*']` → `cfg.stage1.*`; `_GLOBAL_LABELS` → `store.require_labels()`."
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
        "`patch_size` argument. Pass structure and all magic constants unchanged."
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
        "mutates global warning filters or the NumPy RNG (§3.6)."
    ),
}


# ══════════════════════════════════════════════════════════════════════
#  Normalisation
# ══════════════════════════════════════════════════════════════════════


class _Normalise(ast.NodeTransformer):
    """Erase docstrings and annotations; see the module docstring for why."""

    def _strip_docstring(self, node):
        self.generic_visit(node)
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

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self.generic_visit(node)
        if node.value is None:
            # A bare `x: int` declaration carries no runtime effect at all.
            return None
        return ast.Assign(targets=[node.target], value=node.value, type_comment=None)


def normalise(node: ast.AST) -> ast.AST:
    out = _Normalise().visit(copy.deepcopy(node))
    if isinstance(out, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.returns = None
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
                    results.append(("DRIFT", qual, f"member removed ({rel_path})", []))
                    continue

                a, b = old_members[member], new_members[member]
                if member == "<class-header>":
                    a = copy.deepcopy(a)
                    b = copy.deepcopy(b)
                    a.name = b.name = "_"

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
    print("\n✓ AST check OK — every relocation is a declared no-op move.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ref", default=BASELINE_REF)
    ap.add_argument("--verbose", "-v", action="store_true", help="also list IDENTICAL symbols")
    args = ap.parse_args()
    return run_check(args.baseline_ref, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
