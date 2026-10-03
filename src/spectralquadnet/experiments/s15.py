"""S15 — replicate the lean network (Y3) at seeds 1–2 and dissect it (Y5) (S14 D33).

S14 read the S13 single-seed screen: of the four route-A arms only Y3 (the lean
SpectralSeedNet) passed, and it scored above every X1 run in all three of its
cells. S14 froze the next round in
``docs/research/evidence/S14_screen_reading/preregistration_s14.json``
(``9e182670…``) — **10 GPU runs, nothing else**:

* **Y3 replication** — the S13 arm Y3 unchanged at seeds 1 and 2: grouped
  folds 0/1 and the stratified contrast (6 runs). With S13's seed-0 cells this is
  the parent design's 9-run Y3 (H21a/H21b); H21c–H21e are read on these fresh
  seeds only.
* **Y5 dissection** — Y3's changes split in two on the shipped network, grouped
  folds 0/1 at seed 0 (4 runs): ``desc_only`` (``model.spectral_descriptor=
  snv_morph``) and ``spatial_repair`` (``model.spatial_tail_strides=[2,2,2,1]
  model.cbam_min_hw=3``) (H22a/H22b).

As S11 and S13 did, the cells are built **from the frozen files** after every
hash is checked: R1 from the parent ``preregistration_s12.json``, each cell's
data, fold, **seed** and arm overrides from S14's ``cells.gpu``. Nothing S13 or
S11 already scored is in the plan (checked), so a session runs only what the
frozen file asks for.

The frozen file also requires the model, training, data and config code to be
the code the S13 cells ran (``aed5257``). Kaggle clones ``--depth 1``, so that
commit is not in the checkout; the guard is a **content digest** of every
training-relevant file (:data:`CODE_SCOPE`) pinned at ``aed5257``
(:data:`CODE_REFERENCE_DIGEST`) and recomputed on the machine that runs the
cells. This module and the scripts are outside the scope: they only build
commands.

Layout::

    <output_root>/s15/Y3/lean_{grouped,stratified}__f<fold>_s<1|2>/
    <output_root>/s15/Y5/{desc_only,spatial_repair}__f<fold>_s0/
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spectralquadnet.experiments import frozen, s13
from spectralquadnet.experiments.runner import RunSpec
from spectralquadnet.reporting.artifacts import load_manifest

CONFIG = s13.CONFIG
EXPERIMENT = "s15"
DEFAULT_OUTPUT_ROOT = s13.DEFAULT_OUTPUT_ROOT
PreregistrationError = s13.PreregistrationError

#: The chain of frozen files S15 rests on, with the SHA-256 each was frozen at.
PREREGISTRATIONS: dict[str, tuple[str, str]] = {
    **s13.PREREGISTRATIONS,
    "S14": (
        "docs/research/evidence/S14_screen_reading/preregistration_s14.json",
        "9e182670755e13a6ead29a761123841094a9b9fb6fae18392553893d65c1f3da",
    ),
}

#: What this module adds to the frozen strings, recorded in every cell's provenance.
ADDED_BY_S15: tuple[str, ...] = (
    "launcher: torchrun --standalone --nproc_per_node=N (runtime=kaggle_t4x2 requires it)",
    "run_name/output_dir: one directory per cell under <output_root>/s15/<arm>/",
    "seed: per cell, from preregistration_s14.json#cells.gpu[*].seed",
    "code identity: content digest of CODE_SCOPE equals the digest pinned at aed5257",
)

# ══════════════════════════════════════════════════════════════════════
#  Code identity (preregistration_s14.json → guards[1])
# ══════════════════════════════════════════════════════════════════════

#: The commit every S13 cell (and so every S15 comparison) ran on.
CODE_REFERENCE_COMMIT = "aed5257d71603edf4021e7763e325dd146c84170"
#: Everything that decides what a cell trains and how it is scored: the package
#: except the experiment runners, the Hydra configs, and the entry point.
#: Wider than the frozen guard's ``models/engine/data/config`` — stricter, never laxer.
CODE_SCOPE: tuple[tuple[str, str], ...] = (
    ("src/spectralquadnet", "*.py"),
    ("src/spectralquadnet", "*.yaml"),
    ("configs", "*.yaml"),
)
CODE_SCOPE_FILES: tuple[str, ...] = ("train.py",)
CODE_SCOPE_EXCLUDE: tuple[str, ...] = ("src/spectralquadnet/experiments/",)
#: ``code_digest`` of ``CODE_REFERENCE_COMMIT`` (``git archive``), recorded when S15 was implemented.
CODE_REFERENCE_DIGEST = "fade41e5764c4d3313dcc8e9ab9e270f3ccfda591ea81423bd2ead42a650462f"


def code_files(root: Path) -> list[str]:
    """Repository-relative paths of every file in the code-identity scope, sorted."""
    out: set[str] = set()
    for base, pattern in CODE_SCOPE:
        for path in (root / base).rglob(pattern):
            rel = path.relative_to(root).as_posix()
            if "__pycache__" in rel or any(rel.startswith(x) for x in CODE_SCOPE_EXCLUDE):
                continue
            out.add(rel)
    out.update(f for f in CODE_SCOPE_FILES if (root / f).exists())
    return sorted(out)


def code_digest(root: Path | None = None) -> tuple[str, int]:
    """``(sha256, n_files)`` over every scoped file's path and content."""
    root = root or frozen.repo_root()
    h = hashlib.sha256()
    files = code_files(root)
    for rel in files:
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashlib.sha256((root / rel).read_bytes()).digest())
    return h.hexdigest(), len(files)


def check_code_identity(root: Path | None = None) -> list[str]:
    """Empty when the training code is ``aed5257``'s, byte for byte."""
    digest, n = code_digest(root)
    if digest == CODE_REFERENCE_DIGEST:
        return []
    return [
        f"code identity: {n} scoped files digest to {digest[:12]}…, not aed5257's "
        f"{CODE_REFERENCE_DIGEST[:12]}… — model/training/data/config code changed since the S13 "
        "cells ran. preregistration_s14.json requires it unchanged (or G-neutral re-run on the "
        "default and the Y3 keys against aed5257) before any S15 cell runs."
    ]


# ══════════════════════════════════════════════════════════════════════
#  The plan
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class S15Cell(s13.S13Cell):
    """One GPU run of S15 — an :class:`~s13.S13Cell` that writes under ``s15/``."""

    def spec(self, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> RunSpec:
        root = Path(output_root) / EXPERIMENT / self.arm
        return RunSpec(
            experiment=f"{EXPERIMENT}/{self.arm}",
            arm=self.variant,
            fold=self.fold,
            seed=self.seed,
            config=CONFIG,
            overrides=self.overrides(),
            output_dir=str(root / f"{self.variant}__f{self.fold}_s{self.seed}"),
        )


@dataclass
class S15Plan:
    """Every S15 cell, plus the hashes the plan was built from."""

    cells: list[S15Cell] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)

    def select(self, arms: list[str] | None = None, cells: list[str] | None = None) -> list[S15Cell]:
        picked = [c for c in self.cells if not arms or c.arm in arms]
        if cells:
            picked = [c for c in picked if c.name in cells or c.name.split("/", 1)[1] in cells]
        return picked

    def cell(self, name: str) -> S15Cell:
        return next(c for c in self.cells if c.name == name)


def verify_preregistrations(root: Path | None = None) -> dict[str, str]:
    """``{study: sha256}`` for S12 (parent), S13 (amendment) and S14 (this round), after checking each."""
    root = root or frozen.repo_root()
    out: dict[str, str] = {}
    for study, (rel, expected) in PREREGISTRATIONS.items():
        path = root / rel
        if not path.exists():
            raise PreregistrationError(f"{study} pre-registration not found at {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise PreregistrationError(
                f"{rel} hashes to {digest}, not the frozen {expected}. The file was edited "
                "after freezing; restore it from git before running any arm."
            )
        out[study] = digest
    return out


def _load(study: str, root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((root / PREREGISTRATIONS[study][0]).read_text())
    return payload


def load_plan(root: Path | None = None) -> S15Plan:
    """Build every S15 cell from the frozen files.

    Raises:
        PreregistrationError: a hash moved; S14's regime is not the parent's; a
            cell's name disagrees with its fold/seed; or a cell repeats one S13
            already ran.
    """
    root = root or frozen.repo_root()
    hashes = verify_preregistrations(root)
    parent, design = _load("S12", root), _load("S14", root)
    regime = s13._split(parent["regime"]["overrides"])
    if s13._split(design["regime"]["overrides"]) != regime:
        raise PreregistrationError("preregistration_s14.json's regime is not the parent's R1")

    cells: list[S15Cell] = []
    for i, entry in enumerate(design["cells"]["gpu"]):
        arm, rest = str(entry["cell"]).split("/", 1)
        variant, suffix = rest.rsplit("__", 1)
        fold, seed = int(entry["fold"]), int(entry["seed"])
        if suffix != f"f{fold}_s{seed}":
            raise PreregistrationError(f"S14 cell {entry['cell']!r} does not match its fold/seed")
        cells.append(
            S15Cell(
                arm=arm,
                variant=variant,
                data=str(entry["data"]),
                fold=fold,
                seed=seed,
                regime=regime,
                arm_overrides=s13._split(str(entry["arm_overrides"])),
                source=(
                    f"S14 preregistration_s14.json#cells.gpu[{i}]; "
                    "regime: S12 preregistration_s12.json#regime.overrides"
                ),
            )
        )
    already = {c.name for c in s13.load_plan(root).cells}
    repeated = sorted(c.name for c in cells if c.name in already)
    if repeated:
        raise PreregistrationError(f"S15 would re-run cells S13 already ran: {repeated}")
    if len({c.name for c in cells}) != len(cells):
        raise PreregistrationError("S14 lists a cell twice")
    return S15Plan(cells=cells, hashes=hashes)


# ══════════════════════════════════════════════════════════════════════
#  Composition checks
# ══════════════════════════════════════════════════════════════════════

#: The frozen intent of each S15 arm, as values (preregistration_s14.json →
#: arms.*.change) — restated independently of the override strings.
ARM_INTENT: dict[tuple[str, str], dict[str, Any]] = {
    ("Y3", "lean_grouped"): s13._LEAN,
    ("Y3", "lean_stratified"): s13._LEAN,
    ("Y5", "desc_only"): {"model.spectral_descriptor": "snv_morph"},
    ("Y5", "spatial_repair"): {"model.spatial_tail_strides": [2, 2, 2, 1], "model.cbam_min_hw": 3},
}


def expected_values(cell: S15Cell) -> dict[str, Any]:
    """The resolved values ``cell`` must compose to: R1, the shipped keys, its arm."""
    values = s13.base_expected_values()
    if (cell.arm, cell.variant) not in ARM_INTENT:
        raise PreregistrationError(f"{cell.name}: no frozen intent for arm {cell.arm}/{cell.variant}")
    values.update(ARM_INTENT[(cell.arm, cell.variant)])
    values["data.split_scheme"] = cell.protocol
    values["data.split_fold"] = cell.fold
    values["seed"] = cell.seed
    return values


def check_cell(cell: S15Cell, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[str]:
    """Problems with ``cell``'s composition, as strings; empty when it is right."""
    return s13.check_cell(cell, output_root, expected=expected_values(cell))


def check_regime_is_r1(plan: S15Plan) -> list[str]:
    return s13.check_regime_is_r1(plan)  # type: ignore[arg-type]


def compose_cell(cell: S15Cell, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Any:
    return s13.compose_cell(cell, output_root)


def provenance(
    cell: S15Cell, spec: RunSpec, command: str, hashes: dict[str, str], code: tuple[str, int]
) -> dict[str, Any]:
    """The ``frozen_cell.json`` written into a cell's directory before it runs."""
    role = (
        "replication of S13 Y3 at a fresh seed (H21a/H21b with seeds 0–2; H21c–H21e fresh seeds only)"
        if cell.arm == "Y3"
        else "dissection of Y3, seed 0 (H22a desc_only / H22b spatial_repair; a screen, D28)"
    )
    return {
        "study": "S15",
        "design": "Y3 replication + dissection (S14 D33; preregistration_s14.json)",
        "role": role,
        "arm": cell.arm,
        "variant": cell.variant,
        "protocol": cell.protocol,
        "data": cell.data,
        "fold": cell.fold,
        "seed": cell.seed,
        "regime": list(cell.regime),
        "arm_overrides": list(cell.arm_overrides),
        "source": cell.source,
        "preregistration_sha256": hashes,
        "code_identity": {
            "reference_commit": CODE_REFERENCE_COMMIT,
            "reference_digest": CODE_REFERENCE_DIGEST,
            "digest": code[0],
            "n_files": code[1],
            "identical": code[0] == CODE_REFERENCE_DIGEST,
        },
        "added_by_s15": list(ADDED_BY_S15),
        "command": command,
        "output_dir": spec.output_dir,
    }


def summary_rows(plan: S15Plan, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    """One row per cell that has a ``run.json``: the quantities S15 is read on.

    A convenience for the operator; the reading itself (hypotheses on seeds 0–2 or
    fresh seeds, matched deltas, guards) belongs to the analysis study (S16).
    """
    rows: list[dict[str, Any]] = []
    for cell in plan.cells:
        manifest = load_manifest(Path(cell.spec(output_root).output_dir))
        if not manifest:
            rows.append({"cell": cell.name, "status": "not scored"})
            continue
        m = s13._metric
        tta = m(manifest, "results", "tta") or {}
        probe = m(manifest, "session_probe", "representations") or {}
        code = m(manifest, "run", "code") or {}
        rows.append(
            {
                "cell": cell.name,
                "status": "scored",
                "f1_tta": tta.get("macro_f1"),
                "f1_no_tta": m(manifest, "results", "no_tta", "macro_f1"),
                "same_recall": m(tta, "session", "same_session", "macro_recall"),
                "cross_recall": m(tta, "session", "cross_session", "macro_recall"),
                "attraction_cross": m(tta, "session", "attraction", "cross"),
                "kappa_embedding": (probe.get("embedding") or {}).get("kappa"),
                "parameters": m(manifest, "run", "parameters"),
                "epoch": m(manifest, "checkpoint", "epoch"),
                "commit": (code.get("commit") or "")[:7] if isinstance(code, dict) else None,
                "dirty": code.get("dirty") if isinstance(code, dict) else None,
            }
        )
    return rows
