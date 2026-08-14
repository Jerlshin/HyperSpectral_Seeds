"""The primary protocol: leave-one-bundle-out, 2 folds × 3 seeds (CHANGES §19).

*"This section is the paper's methods section. Nothing else in the audit matters
if this is wrong."*

::

    for fold in {0, 1}:                # the COMPLETE leave-one-bundle-out CV
        data.split_scheme   = grouped  #   this dataset supports — there is no
        data.split_fold     = fold     #   third bundle, so no third fold
        data.calib_frac     = 0.15     # carved from TRAIN, by group
        single_group_policy = error    # refuse to silently accept a leak
        for seed in {0, 1, 2}:
            train -> select on calib -> score val ∪ test ONCE
    report mean ± range over 2 folds × 3 seeds   (6 runs)

Plus the contrast arm: the same thing under ``stratified``, reported *beside*
the grouped number rather than instead of it. The gap between them is the
project's headline result.

Three constraints this sweep cannot remove, which belong in the paper and not
in a footnote
─────────────────────────────────────────────────────────────────────────────
1. Training sees **one** acquisition bundle per class, so the training set
   contains **zero within-class acquisition variance**. The model cannot learn
   acquisition invariance because it never observes two acquisitions of one
   class. This is a data-collection ceiling, not a method limitation.
2. ``val`` and ``test`` are two halves of the **same** held-out bundle and are
   therefore not mutually independent. They are scored together, once, and
   selection happens on ``calib``.
3. Two folds is the maximum. Report both; never their maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectralquadnet.experiments.registry import CONTROL_CONFIG, DEFAULT_CONFIG
from spectralquadnet.experiments.runner import RunSpec

#: Folds the grouped protocol sweeps. ``(0, 1)`` is exhaustive on this dataset.
PROTOCOL_FOLDS: tuple[int, ...] = (0, 1)

#: Seeds each cell is repeated at. Three is the minimum that makes a mean ±
#: range meaningful; the audited run used one, and its own docs warned that a
#: single-seed delta below run-to-run variance is not evidence.
PROTOCOL_SEEDS: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class ProtocolArm:
    """One protocol being reported."""

    name: str
    data_config: str
    #: ``stratified`` has no groups to rotate, so it runs one fold at more
    #: seeds rather than two folds at fewer — the same number of runs, all of
    #: them meaningful.
    folds: tuple[int, ...]
    note: str


#: The two arms CHANGES §19 asks for: the honest protocol, and the leaky one
#: reported beside it as a contrast.
PROTOCOL_ARMS: tuple[ProtocolArm, ...] = (
    ProtocolArm(
        name="grouped",
        data_config="hsi256_grouped",
        folds=PROTOCOL_FOLDS,
        note="Leave-one-acquisition-bundle-out. The headline.",
    ),
    ProtocolArm(
        name="stratified",
        data_config="hsi256_stratified",
        folds=(0,),
        note="Patch-level. Reported as the contrast; the gap is the result.",
    ),
)


def build_specs(
    output_root: str | Path,
    config: str = DEFAULT_CONFIG,
    seeds: tuple[int, ...] = PROTOCOL_SEEDS,
    arms: tuple[ProtocolArm, ...] = PROTOCOL_ARMS,
    experiment: str = "protocol",
) -> list[RunSpec]:
    """Expand the protocol into its cells.

    ``stratified`` gets ``len(seeds)`` extra seeds to match the grouped arm's
    run count, so neither arm's mean is built from more runs than the other's —
    an asymmetry that would make the gap partly an artefact of sample size.
    """
    root = Path(output_root) / experiment
    specs: list[RunSpec] = []
    n_grouped_cells = len(PROTOCOL_FOLDS) * len(seeds)

    for arm in arms:
        arm_seeds = seeds
        if len(arm.folds) < len(PROTOCOL_FOLDS):
            extra = n_grouped_cells - len(arm.folds) * len(seeds)
            arm_seeds = tuple(list(seeds) + [max(seeds) + 1 + i for i in range(max(extra, 0))])
        for fold in arm.folds:
            for seed in arm_seeds:
                specs.append(
                    RunSpec(
                        experiment=experiment,
                        arm=arm.name,
                        fold=fold,
                        seed=seed,
                        config=config,
                        overrides=(f"data={arm.data_config}",),
                        output_dir=str(root / f"{arm.name}__f{fold}_s{seed}"),
                    )
                )
    return specs


def build_baseline_comparison_specs(
    output_root: str | Path,
    seeds: tuple[int, ...] = PROTOCOL_SEEDS,
    experiment: str = "protocol_baseline",
) -> list[RunSpec]:
    """The audited architecture under the *same* protocol as the replacement.

    CHANGES §21 Phase 3: *"Train SpectralSeedNet under §17 and §19. Compare to
    the current 5.19 M model under identical conditions."* Identical means the
    complete 256-band cube, the grouped split, the calibration split, the same
    folds and the same seeds — the only difference being the architecture and
    the curriculum. That is `quadnet_full256` with `pipeline=three_stage`, not
    the frozen audited replica, whose split and band count would vary too.
    """
    root = Path(output_root) / experiment
    return [
        RunSpec(
            experiment=experiment,
            arm="quadnet_three_stage",
            fold=fold,
            seed=seed,
            config=CONTROL_CONFIG,
            overrides=("pipeline=three_stage",),
            output_dir=str(root / f"quadnet_three_stage__f{fold}_s{seed}"),
        )
        for fold in PROTOCOL_FOLDS
        for seed in seeds
    ]


def constraints() -> list[str]:
    """The three limitations that must be stated in the paper, as text.

    Returned as data so the run banner, the README and the generated report
    cannot drift into three different phrasings of the same caveat.
    """
    return [
        "Training sees exactly ONE acquisition bundle per class, so the training set "
        "contains zero within-class acquisition variance. The model cannot learn "
        "acquisition invariance because it never observes two acquisitions of one class. "
        "This is a ceiling imposed by the data collection, not by the method.",
        "`val` and `test` are two halves of the SAME held-out bundle and are therefore not "
        "mutually independent. They are treated as one held-out set and scored once; "
        "checkpoint selection happens on `calib`, carved from train by group.",
        "Two folds is the maximum this dataset supports — there is no third bundle. "
        "Both are reported, as mean ± range. Never their maximum.",
    ]


def summary(specs: list[RunSpec]) -> dict[str, Any]:
    """A description of the planned sweep, for the CLI and the run log."""
    arms: dict[str, int] = {}
    for spec in specs:
        arms[spec.arm] = arms.get(spec.arm, 0) + 1
    return {
        "n_runs": len(specs),
        "runs_per_arm": arms,
        "folds": sorted({s.fold for s in specs}),
        "seeds": sorted({s.seed for s in specs}),
        "constraints": constraints(),
    }
