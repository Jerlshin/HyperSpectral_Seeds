"""S11 — the frozen diagnostic arms X1, X2 and X4, expanded from their pre-registrations.

The arms were frozen before any of them ran, in two files whose SHA-256 is
recorded beside them and in the research log:

* ``docs/research/evidence/S09_post_sweep_forensics/preregistration_next.json``
  (``f896d0e5…``) — **X1** fit-first (nine exact commands) and **X2**
  attribution (three arms, described in prose);
* ``docs/research/evidence/S10_training_architecture_review/preregistration_s10.json``
  (``1c8ae693…``) — **X4** fit ceiling (an override string).

This module turns them into :class:`~spectralquadnet.experiments.runner.RunSpec`
cells **from the files themselves**: the X1 and X4 overrides are read out of the
JSON, not restated, and :func:`load_plan` refuses to build anything if either
file's hash has moved. What it adds is only what the frozen strings leave to
the operator, and every addition is listed in :data:`ADDED_BY_S11` so the
deviation is explicit:

1. **A launcher.** The frozen X1 commands read ``python train.py … runtime=kaggle_t4x2``,
   but that profile sets ``runtime.multi_gpu=ddp``, which refuses to start
   without ``torchrun``. Cells run under ``torchrun --nproc_per_node=N``.
2. **A distinct output directory per cell.** The frozen commands name none, so
   every X1 cell would default to ``outputs/seednet_full256_f{fold}_s{seed}`` —
   the *same* directory as the matching X2/X4 cell — and the pipeline's
   auto-resume would then re-score one arm's checkpoint as another's result.
   Cells write to ``<output_root>/s11/<arm>/<cell>__f<fold>_s<seed>/``.
3. **X2's pathway arms.** S09 froze them as "requires a ``model.pathways``
   switch"; S11 implements it as ``model.pathways=[spectral]`` /
   ``model.pathways=[spatial]`` (see ``SpectralSeedNet``).

Each cell's resolved config is checked against the frozen intent by
:func:`check_cell` (``scripts/run_frozen.py --check``): for X1, against the
config the frozen command itself composes, key for key.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spectralquadnet.config.compose import config_dir, load_experiment_config
from spectralquadnet.experiments.runner import RunSpec

#: The experiment config every frozen command composes.
CONFIG = "experiment/seednet_full256"
#: The input every frozen arm reads (the S08 sweep's pre-sliced k = 32 cube).
DATA_PREFIX = "ablation/u430k32"
#: Overrides every frozen command carries besides its arm's own.
BASE_OVERRIDES: tuple[str, ...] = ("runtime=kaggle_t4x2", "tracking=console_jsonl")
#: Where S11 cells write, relative to the output root.
EXPERIMENT = "s11"
#: Default output root — the S08 sweep's, so the reference and the arms sit side by side.
DEFAULT_OUTPUT_ROOT = "outputs/experiments_u430k32"

#: The two frozen files, repository-relative, with the SHA-256 each was frozen at.
PREREGISTRATIONS: dict[str, tuple[str, str]] = {
    "S09": (
        "docs/research/evidence/S09_post_sweep_forensics/preregistration_next.json",
        "f896d0e569e0c071f85fb1c7e2fbce23cd3266da1ef317204fa329af9bb542e7",
    ),
    "S10": (
        "docs/research/evidence/S10_training_architecture_review/preregistration_s10.json",
        "1c8ae6937796cd7867f58ae901c1630671869f949ef8067cfefd72b2d4922739",
    ),
}

#: X2's three arms. ``no_morph`` is the frozen file's own override; the two
#: pathway arms are S11's implementation of the switch it asked for.
X2_ARMS: dict[str, str] = {
    "no_morph": "data.morphology_path=''",
    "spectral_only": "model.pathways=[spectral]",
    "spatial_only": "model.pathways=[spatial]",
}

#: What S11 adds to the frozen strings, recorded in every cell's provenance.
ADDED_BY_S11: tuple[str, ...] = (
    "launcher: torchrun --standalone --nproc_per_node=N (runtime=kaggle_t4x2 requires it)",
    "run_name/output_dir: one directory per cell under <output_root>/s11/<arm>/",
    "X2 pathway arms: model.pathways=[spectral] / [spatial]",
)


class PreregistrationError(RuntimeError):
    """A frozen file is missing or its hash has moved."""


def repo_root() -> Path:
    """The repository root: the parent of ``configs/``."""
    return config_dir().parent


def verify_preregistrations(root: Path | None = None) -> dict[str, str]:
    """``{study: sha256}`` for both frozen files, after checking each hash.

    Raises:
        PreregistrationError: A file is missing, or its SHA-256 differs from the
            one it was frozen at — a changed file is no longer the pre-registration.
    """
    root = root or repo_root()
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
    rel, _ = PREREGISTRATIONS[study]
    payload: dict[str, Any] = json.loads((root / rel).read_text())
    return payload


@dataclass(frozen=True)
class FrozenCell:
    """One run of one frozen arm."""

    arm: str  # X1 | X2 | X4
    variant: str  # grouped | stratified | no_morph | spectral_only | spatial_only
    protocol: str  # grouped | stratified
    fold: int
    seed: int
    arm_overrides: tuple[str, ...]
    source: str  # which frozen file and key the overrides came from
    frozen_command: str | None = None  # X1 only: the literal frozen command

    @property
    def name(self) -> str:
        return f"{self.arm}/{self.variant}__f{self.fold}_s{self.seed}"

    def overrides(self) -> tuple[str, ...]:
        """Every override the cell's ``train.py`` receives, before identity keys."""
        return (f"data={DATA_PREFIX}_{self.protocol}", *BASE_OVERRIDES, *self.arm_overrides)

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
class FrozenPlan:
    """Every S11 cell, plus the hashes the plan was built from."""

    cells: list[FrozenCell] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)

    def select(
        self, arms: list[str] | None = None, cells: list[str] | None = None
    ) -> list[FrozenCell]:
        picked = [c for c in self.cells if not arms or c.arm in arms]
        if cells:
            picked = [c for c in picked if c.name in cells or c.name.split("/", 1)[1] in cells]
        return picked


def _split_overrides(text: str) -> tuple[str, ...]:
    return tuple(shlex.split(text, posix=False))


def _frozen_x1_overrides(command: str) -> tuple[list[str], dict[str, str]]:
    """Split one frozen X1 command into its arm overrides and its identity keys."""
    parts = shlex.split(command, posix=False)
    assert parts[:2] == ["python", "train.py"], command
    identity: dict[str, str] = {}
    rest: list[str] = []
    for part in parts[2:]:
        key = part.split("=", 1)[0]
        if part.startswith("--config-name="):
            identity["config"] = part.split("=", 1)[1]
        elif key in ("data", "data.split_fold", "seed", *(o.split("=")[0] for o in BASE_OVERRIDES)):
            identity[key] = part.split("=", 1)[1]
        else:
            rest.append(part)
    return rest, identity


def load_plan(root: Path | None = None) -> FrozenPlan:
    """Build every X1, X2 and X4 cell from the two frozen files.

    Raises:
        PreregistrationError: see :func:`verify_preregistrations`.
    """
    root = root or repo_root()
    hashes = verify_preregistrations(root)
    s09, s10 = _load("S09", root), _load("S10", root)
    cells: list[FrozenCell] = []

    # ── X1: the nine literal commands ───────────────────────────────────
    x1 = s09["arms"]["X1_fit_first"]
    x1_overrides = _split_overrides(x1["overrides"])
    for command in x1["commands"]:
        rest, ident = _frozen_x1_overrides(command)
        if tuple(rest) != x1_overrides:
            raise PreregistrationError(
                f"frozen X1 command does not carry the frozen X1 overrides: {command}"
            )
        protocol = ident["data"].rsplit("_", 1)[-1]
        cells.append(
            FrozenCell(
                arm="X1",
                variant=protocol,
                protocol=protocol,
                fold=int(ident.get("data.split_fold", 0)),
                seed=int(ident["seed"]),
                arm_overrides=x1_overrides,
                source="S09 preregistration_next.json#arms.X1_fit_first.commands",
                frozen_command=command,
            )
        )

    # ── X2: three arms × grouped folds 0, 1 × seeds 0, 1, shipped regime ─
    frozen_no_morph = s09["arms"]["X2_attribution"]["arms"]["no_morph"].split()[0]
    if frozen_no_morph != X2_ARMS["no_morph"]:
        raise PreregistrationError(f"X2 no_morph override is {frozen_no_morph!r} in the file")
    for variant, override in X2_ARMS.items():
        for fold in (0, 1):
            for seed in (0, 1):
                cells.append(
                    FrozenCell(
                        arm="X2",
                        variant=variant,
                        protocol="grouped",
                        fold=fold,
                        seed=seed,
                        arm_overrides=(override,),
                        source=(
                            "S09 preregistration_next.json#arms.X2_attribution.arms."
                            f"{variant} (runs: grouped folds 0,1 x seeds 0,1)"
                        ),
                    )
                )

    # ── X4: grouped f0 s0 + stratified f0 s0 ─────────────────────────────
    x4 = s10["arms"]["X4_fit_ceiling"]
    x4_overrides = _split_overrides(x4["overrides"])
    for protocol in ("grouped", "stratified"):
        cells.append(
            FrozenCell(
                arm="X4",
                variant=protocol,
                protocol=protocol,
                fold=0,
                seed=0,
                arm_overrides=x4_overrides,
                source="S10 preregistration_s10.json#arms.X4_fit_ceiling",
            )
        )
    return FrozenPlan(cells=cells, hashes=hashes)


# ══════════════════════════════════════════════════════════════════════
#  Composition checks
# ══════════════════════════════════════════════════════════════════════

#: What every cell must resolve to, by arm — the frozen intent as values.
_X1 = {
    "single.mixup_epochs": 30,
    "single.arcface_m": 0.0,
    "single.margin_warmup_start": 31,
    "single.margin_warmup_end": 31,
    "grad_clip": 50.0,
    "single.epochs": 200,
    "single.patience": 40,
}
_SHIPPED = {
    "single.mixup_epochs": 110,
    "single.arcface_m": 0.30,
    "single.margin_warmup_start": 111,
    "single.margin_warmup_end": 130,
    "grad_clip": 5.0,
    "single.epochs": 150,
    "single.patience": 25,
}
_X4 = {
    **_X1,
    "single.mixup_epochs": 0,
    "single.patience": 200,
    "single.label_smooth_hi": 0.0,
    "single.label_smooth_lo": 0.0,
    "stage1.aux_loss_weight_init": 0.0,
    "stage1.aux_loss_weight_final": 0.0,
    "single.dropout": 0.0,
    "single.aug_profile": "none",
}
#: Held at the S08 sweep's values in every arm (D21 guard 3, S11 D22).
_INVARIANT = {
    "single.aux_weight_schedule": "legacy",
    "clip_partition": "legacy",
    "single.max_lr": 5.0e-4,
    "single.mixup": 0.35,
    "weight_decay": 2.0e-4,
    "ema_decay": 0.999,
    "data.num_bands": 32,
    "runtime.multi_gpu": "ddp",
    "runtime.amp_dtype": "fp16",
    "evaluation.select_split": "calib",
    "evaluation.report_split": "val_test",
}


def _get(cfg: Any, dotted: str) -> Any:
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def expected_values(cell: FrozenCell) -> dict[str, Any]:
    """The resolved values ``cell`` must compose to."""
    values: dict[str, Any] = dict(_INVARIANT)
    if cell.arm == "X1":
        values.update(_X1)
    elif cell.arm == "X4":
        values.update(_X4)
    else:
        values.update(_SHIPPED)
        values["single.dropout"] = 0.15
        values["single.aug_profile"] = "medium"
        values["stage1.aux_loss_weight_init"] = 0.65
        values["stage1.aux_loss_weight_final"] = 0.25
    values["data.split_scheme"] = cell.protocol
    values["data.split_fold"] = cell.fold
    values["seed"] = cell.seed
    pathways = ["spatial", "spectral"]
    morph = "./dataset_u430k32/morphology.npy"
    if cell.variant == "spectral_only":
        pathways = ["spectral"]
    elif cell.variant == "spatial_only":
        pathways = ["spatial"]
    elif cell.variant == "no_morph":
        morph = ""
    values["model.pathways"] = pathways
    values["data.morphology_path"] = morph
    return values


def compose_cell(cell: FrozenCell, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Any:
    """The config ``train.py`` would run for ``cell`` (identity keys included)."""
    spec = cell.spec(output_root)
    argv = spec.command()
    overrides = [a for a in argv[3:]]  # drop interpreter, train.py, --config-name
    return load_experiment_config(CONFIG, overrides=overrides)


def check_cell(cell: FrozenCell, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[str]:
    """Problems with ``cell``'s composition, as strings; empty when it is right.

    Two checks: every value in :func:`expected_values`, and — for X1 — that the
    cell resolves to *exactly* the config the frozen command composes, apart
    from ``run_name``/``output_dir``/``hydra``, which S11 adds.
    """
    from omegaconf import OmegaConf

    problems: list[str] = []
    cfg = compose_cell(cell, output_root)
    for key, want in expected_values(cell).items():
        got = _get(cfg, key)
        if OmegaConf.is_list(got):
            got = list(got)
        ok = abs(float(got) - float(want)) < 1e-12 if isinstance(want, float) else got == want
        if not ok:
            problems.append(f"{cell.name}: {key} = {got!r}, expected {want!r}")
    if cell.frozen_command is not None:
        parts = shlex.split(cell.frozen_command, posix=False)[2:]
        name = next(p.split("=", 1)[1] for p in parts if p.startswith("--config-name="))
        frozen = load_experiment_config(
            name, overrides=[p for p in parts if not p.startswith("--")]
        )
        a = OmegaConf.to_container(cfg, resolve=True)
        b = OmegaConf.to_container(frozen, resolve=True)
        assert isinstance(a, dict) and isinstance(b, dict)
        for key in ("run_name", "output_dir", "hydra"):
            a.pop(key, None)
            b.pop(key, None)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            problems.append(f"{cell.name}: differs from its frozen command in {diff}")
    return problems


def provenance(
    cell: FrozenCell, spec: RunSpec, command: str, hashes: dict[str, str]
) -> dict[str, Any]:
    """The ``frozen_cell.json`` written into a cell's directory before it runs."""
    return {
        "study": "S11",
        "arm": cell.arm,
        "variant": cell.variant,
        "protocol": cell.protocol,
        "fold": cell.fold,
        "seed": cell.seed,
        "arm_overrides": list(cell.arm_overrides),
        "source": cell.source,
        "frozen_command": cell.frozen_command,
        "preregistration_sha256": hashes,
        "added_by_s11": list(ADDED_BY_S11),
        "command": command,
        "output_dir": spec.output_dir,
    }
