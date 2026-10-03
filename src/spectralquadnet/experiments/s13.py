"""S13 — the route-A representation arms Y1–Y4, as a single-seed screen (D28).

The arms were frozen in S12's ``preregistration_s12.json`` (``88b377c5…``) as a
30-run design (seeds 0, 1, 2). Before any of them ran, the PI approved one
deviation — **one seed (0) instead of three** — recorded, with its rationale and
the exact cell list it implies, in the amendment
``docs/research/evidence/S13_representation_screening/preregistration_s13.json``.
The parent file is untouched; both hashes are checked here and nothing runs if
either has moved.

As in S11 (:mod:`spectralquadnet.experiments.frozen`), the cells are built **from
the files**: the R1 regime is read out of the parent's ``regime.overrides`` and
each cell's data, fold and arm overrides out of the amendment's ``cells.gpu``;
the seed out of ``deviation.seed``. This module adds only what S11 added — a
``torchrun`` launcher and one output directory per cell — plus the two CPU cells
that fuse Y1's single-pathway networks (:mod:`spectralquadnet.experiments.fusion`).

Layout::

    <output_root>/s13/<arm>/<variant>__f<fold>_s0/      one per GPU cell (10)
    <output_root>/s13/Y1/fused__f<fold>_s0/              Y1's fused cells (2, CPU)
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.experiments import frozen
from spectralquadnet.experiments.runner import RunSpec
from spectralquadnet.reporting.artifacts import RESULTS_DIR, RUN_MANIFEST, load_manifest

#: The experiment config every S13 cell composes (as S11's).
CONFIG = frozen.CONFIG
#: Overrides every cell carries besides the regime and its arm's own (as S11's).
BASE_OVERRIDES = frozen.BASE_OVERRIDES
#: Where S13 cells write, relative to the output root.
EXPERIMENT = "s13"
DEFAULT_OUTPUT_ROOT = frozen.DEFAULT_OUTPUT_ROOT

#: The parent pre-registration and the amendment, with the SHA-256 each was frozen at.
PREREGISTRATIONS: dict[str, tuple[str, str]] = {
    "S12": (
        "docs/research/evidence/S12_frozen_arms_reading/preregistration_s12.json",
        "88b377c5bc32f31a9ccb95356a88eb0be35e8cb51216b035c88f1761919ae7f4",
    ),
    "S13": (
        "docs/research/evidence/S13_representation_screening/preregistration_s13.json",
        "ef5982131df48c2ba8105c751fcf21f527a01ea2c0addf4ad59380f1a0989460",
    ),
}

#: What this module adds to the frozen strings, recorded in every cell's provenance.
ADDED_BY_S13: tuple[str, ...] = (
    "launcher: torchrun --standalone --nproc_per_node=N (runtime=kaggle_t4x2 requires it)",
    "run_name/output_dir: one directory per cell under <output_root>/s13/<arm>/",
)

PreregistrationError = frozen.PreregistrationError


def verify_preregistrations(root: Path | None = None) -> dict[str, str]:
    """``{study: sha256}`` for the parent and the amendment, after checking each hash."""
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


def _split(text: str) -> tuple[str, ...]:
    return tuple(shlex.split(text, posix=False))


@dataclass(frozen=True)
class S13Cell:
    """One GPU run of one S13 arm."""

    arm: str  # Y1 | Y2 | Y3 | Y4
    variant: str  # spectral_only | spatial_only | mixstyle | lean_grouped | lean_stratified | within_8020
    data: str  # the data group, e.g. ablation/u430k32_grouped
    fold: int
    seed: int
    regime: tuple[str, ...]  # R1, from the parent file
    arm_overrides: tuple[str, ...]
    source: str

    @property
    def name(self) -> str:
        return f"{self.arm}/{self.variant}__f{self.fold}_s{self.seed}"

    @property
    def protocol(self) -> str:
        return self.data.rsplit("_", 1)[-1]

    def overrides(self) -> tuple[str, ...]:
        """Every override the cell's ``train.py`` receives, before identity keys."""
        return (f"data={self.data}", *BASE_OVERRIDES, *self.regime, *self.arm_overrides)

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


@dataclass(frozen=True)
class FusedCell:
    """One CPU cell: Y1's two single-pathway networks of a fold, fused on calib."""

    arm: str
    variant: str
    fold: int
    seed: int
    spectral: str  # cell names of the inputs
    spatial: str

    @property
    def name(self) -> str:
        return f"{self.arm}/{self.variant}__f{self.fold}_s{self.seed}"

    def output_dir(self, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
        return Path(output_root) / EXPERIMENT / self.arm / f"{self.variant}__f{self.fold}_s{self.seed}"

    def is_complete(self, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> bool:
        return (self.output_dir(output_root) / RESULTS_DIR / RUN_MANIFEST).exists()


@dataclass
class S13Plan:
    """Every S13 cell, plus the hashes the plan was built from."""

    cells: list[S13Cell] = field(default_factory=list)
    fused: list[FusedCell] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    seed: int = 0

    def select(
        self, arms: list[str] | None = None, cells: list[str] | None = None
    ) -> list[S13Cell]:
        picked = [c for c in self.cells if not arms or c.arm in arms]
        if cells:
            picked = [c for c in picked if c.name in cells or c.name.split("/", 1)[1] in cells]
        return picked

    def cell(self, name: str) -> S13Cell:
        return next(c for c in self.cells if c.name == name)


def load_plan(root: Path | None = None) -> S13Plan:
    """Build every S13 cell from the parent pre-registration and the amendment.

    Raises:
        PreregistrationError: a hash moved, or the amendment names a cell this
            module cannot build.
    """
    root = root or frozen.repo_root()
    hashes = verify_preregistrations(root)
    parent, amendment = _load("S12", root), _load("S13", root)
    regime = _split(parent["regime"]["overrides"])
    seed = int(amendment["deviation"]["seed"])

    cells: list[S13Cell] = []
    for i, entry in enumerate(amendment["cells"]["gpu"]):
        arm, rest = str(entry["cell"]).split("/", 1)
        variant, suffix = rest.rsplit("__", 1)
        if suffix != f"f{int(entry['fold'])}_s{seed}":
            raise PreregistrationError(f"amendment cell {entry['cell']!r} does not match its fold/seed")
        cells.append(
            S13Cell(
                arm=arm,
                variant=variant,
                data=str(entry["data"]),
                fold=int(entry["fold"]),
                seed=seed,
                regime=regime,
                arm_overrides=_split(str(entry["arm_overrides"])),
                source=(
                    f"S13 preregistration_s13.json#cells.gpu[{i}]; "
                    "regime: S12 preregistration_s12.json#regime.overrides"
                ),
            )
        )

    names = {c.name for c in cells}
    fused: list[FusedCell] = []
    for entry in amendment["cells"]["cpu"]:
        arm, rest = str(entry["cell"]).split("/", 1)
        variant, suffix = rest.rsplit("__", 1)
        spectral, spatial = (str(x) for x in entry["inputs"])
        if not {spectral, spatial} <= names:
            raise PreregistrationError(f"fused cell {entry['cell']!r} names inputs that are not cells")
        fold = int(suffix.split("_")[0][1:])
        fused.append(FusedCell(arm, variant, fold, seed, spectral, spatial))
    return S13Plan(cells=cells, fused=fused, hashes=hashes, seed=seed)


# ══════════════════════════════════════════════════════════════════════
#  Composition checks
# ══════════════════════════════════════════════════════════════════════

#: Shipped values of every key an S13 arm may set — what a cell resolves to
#: unless its own arm sets the key.
_ARM_KEY_DEFAULTS: dict[str, Any] = {
    "model.pathways": ["spatial", "spectral"],
    "model.spatial_mixstyle": False,
    "model.spectral_descriptor": "full",
    "model.spatial_tail_strides": [2, 2, 2, 2],
    "model.cbam_min_hw": 0,
    "data.split_eval_frac": 0.30,
}


def _parse(overrides: tuple[str, ...]) -> dict[str, Any]:
    """``key=value`` overrides → ``{key: typed value}``, typed by Hydra's own parser."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    return {o.key_or_group: o.value() for o in OverridesParser.create().parse_overrides(list(overrides))}


#: The frozen intent of each arm, as values (S12 ``arms.*.change``) — restated
#: independently of the amendment's override strings, so a cell whose strings
#: drift from the arm it claims is caught (as S11's ``_X1``/``_X4`` do).
_LEAN = {
    "model.spectral_descriptor": "snv_morph",
    "model.spatial_tail_strides": [2, 2, 2, 1],
    "model.cbam_min_hw": 3,
}
ARM_INTENT: dict[tuple[str, str], dict[str, Any]] = {
    ("Y1", "spectral_only"): {"model.pathways": ["spectral"]},
    ("Y1", "spatial_only"): {"model.pathways": ["spatial"]},
    ("Y2", "mixstyle"): {"model.spatial_mixstyle": True},
    ("Y3", "lean_grouped"): _LEAN,
    ("Y3", "lean_stratified"): _LEAN,
    ("Y4", "within_8020"): {"data.split_eval_frac": 0.2},
}


def expected_values(cell: S13Cell) -> dict[str, Any]:
    """The resolved values ``cell`` must compose to."""
    values: dict[str, Any] = dict(frozen._INVARIANT)
    values.update(frozen._X1)  # R1 — `check_regime_is_r1` ties it to the parent file
    values.update(
        {
            "single.dropout": 0.15,
            "single.aug_profile": "medium",
            "stage1.aux_loss_weight_init": 0.65,
            "stage1.aux_loss_weight_final": 0.25,
            "data.morphology_path": "./dataset_u430k32/morphology.npy",
            "evaluation.save_logits": True,
            "evaluation.session_probe": True,
        }
    )
    values.update(_ARM_KEY_DEFAULTS)
    if (cell.arm, cell.variant) not in ARM_INTENT:
        raise PreregistrationError(f"{cell.name}: no frozen intent for arm {cell.arm}/{cell.variant}")
    values.update(ARM_INTENT[(cell.arm, cell.variant)])
    values["data.split_scheme"] = cell.protocol
    values["data.split_fold"] = cell.fold
    values["seed"] = cell.seed
    return values


def check_regime_is_r1(plan: S13Plan) -> list[str]:
    """The regime read from the parent file is X1's, key for key (S12 D24)."""
    if not plan.cells:
        return ["the plan has no cells"]
    parsed = _parse(plan.cells[0].regime)
    return [
        f"R1: {k} = {parsed.get(k)!r} in the parent file, X1 has {v!r}"
        for k, v in frozen._X1.items()
        if parsed.get(k) != v
    ]


def compose_cell(cell: S13Cell, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Any:
    """The config ``train.py`` would run for ``cell`` (identity keys included)."""
    argv = cell.spec(output_root).command()
    return load_experiment_config(CONFIG, overrides=list(argv[3:]))


def check_cell(cell: S13Cell, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[str]:
    """Problems with ``cell``'s composition, as strings; empty when it is right."""
    from omegaconf import OmegaConf

    problems: list[str] = []
    cfg = compose_cell(cell, output_root)
    for key, want in expected_values(cell).items():
        got = frozen._get(cfg, key)
        if OmegaConf.is_list(got):
            got = list(got)
        ok = abs(float(got) - float(want)) < 1e-12 if isinstance(want, float) else got == want
        if not ok:
            problems.append(f"{cell.name}: {key} = {got!r}, expected {want!r}")
    return problems


def provenance(
    cell: S13Cell, spec: RunSpec, command: str, hashes: dict[str, str]
) -> dict[str, Any]:
    """The ``frozen_cell.json`` written into a cell's directory before it runs."""
    return {
        "study": "S13",
        "design": "single-seed screen (D28; amendment preregistration_s13.json)",
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
        "added_by_s13": list(ADDED_BY_S13),
        "command": command,
        "output_dir": spec.output_dir,
    }


# ══════════════════════════════════════════════════════════════════════
#  Y1 fusion and the summary
# ══════════════════════════════════════════════════════════════════════


def fuse(
    plan: S13Plan,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> list[tuple[FusedCell, str]]:
    """Fuse every Y1 fold whose two networks are scored; ``[(cell, status)]``."""
    from spectralquadnet.data.loaders import build_split_bundle
    from spectralquadnet.experiments.fusion import fuse_cells
    from spectralquadnet.reporting.session import SessionMap
    from spectralquadnet.utils.provenance import code_revision

    done: list[tuple[FusedCell, str]] = []
    for fc in plan.fused:
        spectral, spatial = plan.cell(fc.spectral), plan.cell(fc.spatial)
        dirs = [Path(c.spec(output_root).output_dir) for c in (spectral, spatial)]
        if fc.is_complete(output_root) and not force:
            done.append((fc, "skipped (complete)"))
            continue
        if not all((d / RESULTS_DIR / RUN_MANIFEST).exists() for d in dirs):
            done.append((fc, "waiting (inputs not scored)"))
            continue
        cfg = compose_cell(spectral, output_root)
        smap, _reason = SessionMap.from_config(
            cfg.data, build_split_bundle(cfg), int(cfg.data.num_classes)
        )
        fuse_cells(
            dirs[0],
            dirs[1],
            fc.output_dir(output_root),
            num_classes=int(cfg.data.num_classes),
            sessions=smap,
            n_boot=int(cfg.evaluation.bootstrap_samples),
            seed=fc.seed,
            run={
                "study": "S13",
                "run_name": f"{EXPERIMENT}/{fc.name}",
                "split_scheme": str(cfg.data.split_scheme),
                "split_fold": fc.fold,
                "seed": fc.seed,
                "code": code_revision(),
                "preregistration_sha256": plan.hashes,
            },
        )
        done.append((fc, "fused"))
    return done


def _metric(manifest: dict[str, Any], *path: str) -> Any:
    node: Any = manifest
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def summary_rows(plan: S13Plan, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    """One row per GPU and fused cell that has a ``run.json``: the quantities S13 reads.

    A convenience for the operator; the reading itself (hypotheses, guards,
    matched deltas) belongs to the analysis study.
    """
    targets: list[tuple[str, Path]] = [
        (c.name, Path(c.spec(output_root).output_dir)) for c in plan.cells
    ] + [(f.name, f.output_dir(output_root)) for f in plan.fused]
    rows: list[dict[str, Any]] = []
    for name, out in targets:
        manifest = load_manifest(out)
        if not manifest:
            rows.append({"cell": name, "status": "not scored"})
            continue
        tta = _metric(manifest, "results", "tta") or {}
        probe = _metric(manifest, "session_probe", "representations") or {}
        code = _metric(manifest, "run", "code") or {}
        rows.append(
            {
                "cell": name,
                "status": "scored",
                "f1_tta": tta.get("macro_f1"),
                "f1_no_tta": _metric(manifest, "results", "no_tta", "macro_f1"),
                "same_recall": _metric(tta, "session", "same_session", "macro_recall"),
                "cross_recall": _metric(tta, "session", "cross_session", "macro_recall"),
                "attraction_cross": _metric(tta, "session", "attraction", "cross"),
                "kappa_embedding": (probe.get("embedding") or {}).get("kappa"),
                "fusion_weight": _metric(manifest, "fusion", "weight"),
                "epoch": _metric(manifest, "checkpoint", "epoch"),
                "commit": (code.get("commit") or "")[:7] if isinstance(code, dict) else None,
                "dirty": code.get("dirty") if isinstance(code, dict) else None,
            }
        )
    return rows
