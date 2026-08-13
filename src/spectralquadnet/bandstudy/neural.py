"""The neural confirmation — the one claim the proxies cannot make.

What this stage is for
──────────────────────
Everything else in the band study is measured on foreground-masked mean
spectra, which throw away every spatial and textural cue. The project's own
bracket puts about 25 macro-F1 points of the deployed model in exactly what
that representation discards, and CHANGES F-3 is the standing prediction that
the two curves therefore disagree: a linear model on class-mean spectra
saturates long before a network that reads band × space structure does.

So the proxy sweep answers "which method, and roughly where does information
stop accumulating in the mean spectrum", and this stage answers the single
remaining question: **does the recommended budget hold for the network?** It is
a handful of runs at three or four budgets, not a second sweep — a full neural
budget × method grid is 240 GPU-days on this hardware and is not the experiment
anybody should run.

How it avoids a disk-space problem
──────────────────────────────────
Each arm points ``data.band_indices_path`` at a ``.npy`` of band indices and
``data.patches_data`` at the **full** 256-band cube; the dataset slices each
patch as it comes off the mmap. A k = 100 reduced cube is 14 GB, so
materialising one per (method, fold, budget) would be terabytes for an
experiment whose actual content is a list of integers. Nothing else about the
training run changes, which is what makes the arms comparable.

Every arm's other overrides are held identical on purpose, with two unavoidable
exceptions that are functions of k and are recorded on the spec:
``data.num_bands`` (it *is* the treatment) and the two augmentation widths that
are expressed in bands — a spectral CutMix window of 8 is meaningless at k = 5,
so both are clamped below k and the clamped value is written into the arm's
overrides where a reader can see it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectralquadnet.bandstudy.config import BandStudyConfig, StageResult
from spectralquadnet.bandstudy.pipeline import band_file, wavelength_file
from spectralquadnet.experiments.registry import DEFAULT_CONFIG
from spectralquadnet.experiments.runner import RunSpec, plan, run_all

_log = logging.getLogger("spectralquadnet.bandstudy.neural")

#: The experiment config every arm composes. The band study's question is about
#: the input representation, so it is asked of the repository's default
#: composition rather than of a bespoke one.
NEURAL_CONFIG = DEFAULT_CONFIG

#: The repository's shipped band counts, added as arms wherever the sweep
#: evaluated them. A confirmation that cannot compare against what the project
#: already ships answers a less useful question than one that can.
INCUMBENT_ARMS: tuple[tuple[str, int, str], ...] = (
    ("spa", 40, "incumbent: the shipped 40-band SPA subset"),
    ("mrmr", 100, "incumbent: the per-fold 100-band mRMR subset"),
)


@dataclass(frozen=True)
class NeuralArm:
    """One neural confirmation run, before it is expanded over seeds."""

    method: str
    budget: int
    fold: int
    why: str

    @property
    def name(self) -> str:
        return f"{self.method}_k{self.budget}"


def _clamped_aug(budget: int) -> dict[str, int]:
    """Augmentation widths that are expressed in bands, clamped below ``k``.

    ``cutmix_bands=8`` on a 5-band input swaps the entire spectrum, which is not
    a CutMix — it is a relabelling — and ``max_cutout_bands=3`` on 3 bands zeroes
    the input. Both are clamped to a fraction of the budget so the augmentation
    means the same thing at every budget, and the clamped values appear in the
    printed override list rather than being applied invisibly.
    """
    return {
        "data.cutmix_bands": max(1, min(8, budget // 4 or 1)),
        "data.max_cutout_bands": max(1, min(3, budget // 8 or 1)),
    }


def build_specs(
    cfg: BandStudyConfig,
    arms: list[NeuralArm],
    seeds: tuple[int, ...] = (0, 1, 2),
    output_root: str | Path | None = None,
) -> list[RunSpec]:
    """Expand arms × seeds into :class:`RunSpec`s the existing runner executes.

    Reuses :mod:`spectralquadnet.experiments.runner` rather than launching
    training here, which buys three properties the band study would otherwise
    have to reimplement: each cell is a subprocess and therefore genuinely
    independent, a completed cell is detected by its manifest and skipped on
    resume, and every cell prints the exact command that produced it so one
    number can be re-run by hand without the driver.

    Raises:
        FileNotFoundError: An arm's band index file does not exist. Named
            individually, because the usual cause is a budget the ``select``
            stage was not asked for.
    """
    root = Path(output_root) if output_root else cfg.neural_dir
    specs: list[RunSpec] = []
    missing: list[str] = []

    for arm in arms:
        bands = band_file(cfg, arm.fold, arm.method, arm.budget)
        wavelengths = wavelength_file(cfg, arm.fold, arm.method, arm.budget)
        if not bands.exists() or not wavelengths.exists():
            missing.append(f"{arm.method} k={arm.budget} fold={arm.fold} ({bands})")
            continue

        overrides = [
            # The cube the study itself read, not a hardcoded path: the arms
            # must slice the same array the band indices index into, and a run
            # pointed at a pre-reduced cube would apply 256-band indices to a
            # 40-band array.
            f"data.patches_data={cfg.patches_path}",
            f"data.band_indices_path={bands}",
            f"data.wavelength_path={wavelengths}",
            f"data.num_bands={arm.budget}",
            *[f"{k}={v}" for k, v in _clamped_aug(arm.budget).items()],
        ]
        for seed in seeds:
            specs.append(
                RunSpec(
                    experiment="band_study",
                    arm=arm.name,
                    fold=arm.fold,
                    seed=seed,
                    config=NEURAL_CONFIG,
                    overrides=tuple(overrides),
                    output_dir=str(root / f"{arm.name}__f{arm.fold}_s{seed}"),
                )
            )

    if missing:
        raise FileNotFoundError(
            "these neural arms have no band index file:\n  - "
            + "\n  - ".join(missing)
            + "\nRun `python -m spectralquadnet.bandstudy.cli select` with those budgets in "
            "--budgets first; the arms are named by (method, fold, k) and all three must match."
        )
    return specs


def arms_from_recommendation(
    cfg: BandStudyConfig, recommendation: dict[str, Any], include_references: bool = True
) -> list[NeuralArm]:
    """The arms to run: the recommendation, plus the fixed references.

    The recommendation contributes its conservative and aggressive budgets under
    the chosen method; the references contribute the full cube and the two
    shipped subsets. Both halves matter — without the references there is no
    baseline, and without the aggressive arm there is no test of whether the
    conservative one was over-cautious.
    """
    arms: list[NeuralArm] = []
    seen: set[tuple[str, int, int]] = set()

    def add(method: str, budget: int, fold: int, why: str) -> None:
        key = (method, budget, fold)
        if key not in seen:
            seen.add(key)
            arms.append(NeuralArm(method=method, budget=budget, fold=fold, why=why))

    method = str(recommendation.get("recommended_method", "uniform"))
    full = int(max(cfg.budgets))
    recommended = int(recommendation.get("recommended_budget", full))
    aggressive = int(recommendation.get("aggressive_budget", recommended))

    for fold in sorted(cfg.folds):
        add(method, recommended, fold, "the study's recommended budget")
        if aggressive != recommended:
            add(method, aggressive, fold, "the study's aggressive budget")
        if include_references:
            # The full band count is the reference every reduction is a
            # reduction *from*, and it is taken from the sweep's own grid rather
            # than assumed to be 256 — at k = full every method selects every
            # band, so which label carries it is a formality.
            add("uniform", full, fold, "reference: the full band set, unreduced")
            for ref_method, ref_budget, why in INCUMBENT_ARMS:
                if ref_budget in cfg.budgets and ref_budget != full:
                    add(ref_method, ref_budget, fold, why)
    return arms


def stage_neural(
    cfg: BandStudyConfig,
    seeds: tuple[int, ...] = (0, 1, 2),
    include_references: bool = True,
    execute: bool = False,
    stop_on_failure: bool = False,
) -> StageResult:
    """Emit — and optionally run — the neural confirmation grid.

    ``execute=False`` is the default and prints the plan instead of spending a
    GPU-day, which is the invocation a reviewer should see first. The plan is
    written to ``neural/plan.json`` and ``neural/commands.sh`` either way, so
    the runs can be distributed across machines by hand without this driver.
    """
    import time

    started = time.perf_counter()
    from spectralquadnet.bandstudy.analysis import load_recommendation
    from spectralquadnet.bandstudy.store import console, record_stage

    recommendation = load_recommendation(cfg)
    arms = arms_from_recommendation(cfg, recommendation, include_references)
    specs = build_specs(cfg, arms, seeds=seeds)
    todo, done = plan(specs, force=cfg.force)

    cfg.neural_dir.mkdir(parents=True, exist_ok=True)
    (cfg.neural_dir / "plan.json").write_text(
        json.dumps(
            {
                "config": NEURAL_CONFIG,
                "recommendation": {
                    "method": recommendation.get("recommended_method"),
                    "budget": recommendation.get("recommended_budget"),
                    "aggressive_budget": recommendation.get("aggressive_budget"),
                    "status": recommendation.get("status"),
                },
                "seeds": list(seeds),
                "arms": [
                    {
                        "name": a.name,
                        "method": a.method,
                        "budget": a.budget,
                        "fold": a.fold,
                        "why": a.why,
                    }
                    for a in arms
                ],
                "runs": [
                    {
                        "run_name": s.run_name,
                        "fold": s.fold,
                        "seed": s.seed,
                        "output_dir": s.output_dir,
                        "command": s.shell(),
                        "complete": s.is_complete,
                    }
                    for s in specs
                ],
            },
            indent=2,
        )
    )
    (cfg.neural_dir / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Neural confirmation of the band study's recommendation.\n"
        "# Each line is independent and re-runnable on its own; a finished run is\n"
        "# detected by its results manifest and skipped by the driver.\n"
        "set -euo pipefail\n\n" + "\n".join(s.shell() for s in specs) + "\n"
    )

    term = console()
    term.rule("[bold]neural confirmation")
    term.print(
        f"{len(arms)} arms × {len(seeds)} seeds = [bold]{len(specs)}[/bold] runs "
        f"([yellow]{len(done)}[/yellow] already complete, [bold]{len(todo)}[/bold] to run)"
    )
    for arm in arms:
        term.print(f"  [cyan]{arm.name:22s}[/cyan] fold {arm.fold}  — {arm.why}")
    term.print(f"\nplan → {cfg.neural_dir/'plan.json'}\ncommands → {cfg.neural_dir/'commands.sh'}")

    if not execute or cfg.dry_run:
        term.print(
            "\n[yellow]Nothing was run.[/yellow] Pass --execute to launch these, or run "
            "commands.sh line by line / across machines."
        )
        return StageResult(
            stage="neural",
            n_skipped=len(specs),
            seconds=time.perf_counter() - started,
            notes=[f"planned {len(specs)} runs; not executed"],
        )

    report = run_all(
        specs, "band_study", dry_run=False, force=cfg.force, stop_on_failure=stop_on_failure
    )
    report.write(cfg.neural_dir / "sweep.json")
    result = StageResult(
        stage="neural",
        n_done=len([o for o in report.outcomes if o.status == "completed"]),
        n_skipped=len([o for o in report.outcomes if o.status == "skipped"]),
        n_failed=len(report.failed),
        seconds=time.perf_counter() - started,
        notes=[
            f"  FAILED {f.spec.run_name} (rc={f.returncode}) — {f.log_path}" for f in report.failed
        ],
    )
    record_stage(cfg, result)
    return result


def collect_neural(cfg: BandStudyConfig, variant: str = "tta") -> Any:
    """Whatever neural runs have finished, as a frame of macro-F1 per arm.

    Returns an empty frame when nothing has run, so the report can say "not yet
    measured" rather than fabricate the comparison the whole stage exists to
    provide.
    """
    import pandas as pd

    from spectralquadnet.reporting.tables import collect_runs

    if not cfg.neural_dir.exists():
        return pd.DataFrame()
    run_dirs = sorted(p for p in cfg.neural_dir.iterdir() if (p / "results").is_dir())
    records = collect_runs(run_dirs, variant=variant)
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    # `arm__f<fold>_s<seed>` — recover the arm, and from it the method and k.
    names = frame["run_dir"].map(lambda p: Path(p).name.split("__")[0])
    frame["arm"] = names
    frame["method"] = names.map(lambda n: n.rsplit("_k", 1)[0])
    frame["budget"] = names.map(lambda n: int(n.rsplit("_k", 1)[1]) if "_k" in n else -1)
    return frame
