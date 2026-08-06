"""Load the pre-refactor monolith from git, without its import-time side effects.

Shared by ``check_ast_no_op_move.py`` (REFACTOR_PLAN.md §3.2.1) and
``capture_golden.py`` (§3.2.2). Not part of the installed package — this is
migration scaffolding that exists only while Phases 2-3 are in flight.

Why the module cannot simply be imported
────────────────────────────────────────
``HSI_modality_training/hsi_training.py`` executes eight module-level statements
at import (baseline SHA ``886560f``):

    line  23-25  warnings.filterwarnings(...)         — noise
    line  145    print(CONFIG)                        — noise
    line  147    Path(CONFIG["output_dir"]).mkdir(...) — recreates the directory
                                                        Phase 1 relocated
    line  148    torch.cuda.empty_cache()             — requires CUDA init
    line  150    torch.set_float32_matmul_precision   — mutates global torch state
    line  200    set_seed(CONFIG["seed"])             — consumes the global RNG

:func:`load_baseline_module` therefore parses the source and executes only its
*declarations* — imports, assignments, and every ``class``/``def`` — dropping
bare expression statements and the ``if __name__`` guard. The result is a module
object whose classes and functions are the genuine pre-refactor ones, obtained
without touching the filesystem or the RNG.

Dropping ``set_seed(CONFIG["seed"])`` is safe for the golden capture because
§3.2.2's procedure calls ``set_seed(42)`` explicitly immediately before model
construction — which is exactly the state the baseline's line-200 call left the
RNG in before ``main()`` ran.
"""

from __future__ import annotations

import ast
import subprocess
import types
from pathlib import Path

BASELINE_REF = "886560fe531c99197f20c2ebd06e0bc7ded8ac8f"
BASELINE_FILES = {
    "hsi_training": "HSI_modality_training/hsi_training.py",
    "band_selection": "band_selection.py",
    "data_setup_v3": "data_setup_v3.py",
}

# Statement types kept when stripping module-level side effects.
_DECLARATION_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def baseline_source(key_or_path: str, ref: str = BASELINE_REF) -> str:
    """Return a baseline file's text from git (never from the working tree)."""
    path = BASELINE_FILES.get(key_or_path, key_or_path)
    out = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=True,
    )
    return out.stdout


def strip_module_side_effects(tree: ast.Module) -> ast.Module:
    """Keep only declarations at module scope; drop calls and the ``__main__`` guard."""
    kept = [node for node in tree.body if isinstance(node, _DECLARATION_NODES)]
    return ast.Module(body=kept, type_ignores=[])


def load_baseline_module(
    key_or_path: str = "hsi_training", ref: str = BASELINE_REF
) -> types.ModuleType:
    """Execute a baseline file's declarations and return the resulting module."""
    src = baseline_source(key_or_path, ref)
    tree = strip_module_side_effects(ast.parse(src))
    ast.fix_missing_locations(tree)

    module = types.ModuleType(f"_baseline_{Path(key_or_path).stem}")
    module.__file__ = f"<{key_or_path}@{ref[:7]}>"
    exec(compile(tree, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module


def baseline_symbols(key_or_path: str = "hsi_training", ref: str = BASELINE_REF) -> dict:
    """Map every top-level ``class``/``def`` name to its AST node."""
    tree = ast.parse(baseline_source(key_or_path, ref))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
