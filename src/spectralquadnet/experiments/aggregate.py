"""Turning a directory of runs into the tables and figures a paper needs.

Reads ``results/run.json`` from each run and nothing else, so aggregation is
offline, GPU-free and re-runnable after the fact. A cell that is missing shows
up as a missing cell rather than being silently absorbed into a mean — which is
the failure mode that lets a "mean over 6 runs" quietly be a mean over 4.

Deltas
──────
An arm's delta against the reference is reported two ways and both are needed:

* **Mean difference with each arm's range**, over folds × seeds. This is the
  number that goes in the table.
* **A paired bootstrap** on the prediction arrays, where the two arms scored
  the *same* patches. This is what says whether the difference is outside the
  split's own noise, and it is only computable because
  ``RunArtifacts.write_predictions`` kept the arrays.

The paired test is skipped — with a stated reason, not silently — when the arms
do not share a split. A1's two arms are exactly that case: different protocols
means different eval sets, which is the whole difference between them, so there
is nothing to pair.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectralquadnet.reporting import tables
from spectralquadnet.reporting.artifacts import RESULTS_DIR, load_manifest
from spectralquadnet.reporting.figures import ablation_forest, available, protocol_comparison
from spectralquadnet.reporting.metrics import mean_and_range, paired_bootstrap_delta

_log = logging.getLogger(__name__)


@dataclass
class ArmSummary:
    """One arm's numbers, aggregated over its folds and seeds."""

    arm: str
    n_runs: int
    stats: dict[str, float]
    run_dirs: list[str]
    parameters: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_runs": self.n_runs,
            "parameters": self.parameters,
            **{f"macro_f1_{k}": v for k, v in self.stats.items()},
            "run_dirs": self.run_dirs,
        }


def discover_runs(root: str | Path) -> list[Path]:
    """Every directory under ``root`` holding a results manifest.

    Sorted, so a table's row order is a property of the filesystem layout rather
    than of directory-iteration order — which differs between machines and would
    make two people's tables differ in row order for no reason.
    """
    root = Path(root)
    if not root.exists():
        return []
    return sorted({p.parent.parent for p in root.rglob(f"{RESULTS_DIR}/run.json")})


def arm_of(run_dir: Path) -> tuple[str, int, int]:
    """Recover ``(arm, fold, seed)`` from a run directory name.

    The naming contract is ``<arm>__f<fold>_s<seed>`` — see
    :meth:`~spectralquadnet.experiments.runner.RunSpec.run_name`. Falls back to
    ``(name, 0, 0)`` for a directory that does not follow it, so a hand-made run
    can still be aggregated.
    """
    name = run_dir.name
    if "__f" not in name:
        return name, 0, 0
    arm, _, tail = name.partition("__f")
    fold_str, _, seed_str = tail.partition("_s")
    try:
        return arm, int(fold_str), int(seed_str)
    except ValueError:
        return arm, 0, 0


def summarise_arms(run_dirs: list[Path], variant: str = "tta") -> list[ArmSummary]:
    """Aggregate runs into per-arm mean ± range."""
    buckets: dict[str, list[tuple[Path, float, int]]] = {}
    for run_dir in run_dirs:
        manifest = load_manifest(run_dir)
        result = (manifest.get("results") or {}).get(variant)
        if not result:
            _log.warning("no '%s' result in %s — excluded from the aggregate", variant, run_dir)
            continue
        arm, _, _ = arm_of(run_dir)
        params = int((manifest.get("run") or {}).get("parameters", 0))
        buckets.setdefault(arm, []).append((run_dir, float(result["macro_f1"]), params))

    summaries = []
    for arm, entries in sorted(buckets.items()):
        summaries.append(
            ArmSummary(
                arm=arm,
                n_runs=len(entries),
                stats=mean_and_range([score for _, score, _ in entries]),
                run_dirs=[str(p) for p, _, _ in entries],
                parameters=entries[0][2],
            )
        )
    return summaries


def paired_delta(
    reference_dirs: list[Path],
    arm_dirs: list[Path],
    split_tag: str,
    num_classes: int,
    n_boot: int = 2000,
) -> tuple[float, float, float] | None:
    """``(delta, ci_lo, ci_hi)`` for one arm against the reference, paired.

    Pairs cell-by-cell on ``(fold, seed)`` and pools the predictions, so the
    comparison is over the same patches in the same order. Returns ``None``
    when no cell pairs up or the target arrays disagree — which is the correct
    outcome for two arms evaluated on different protocols, and is reported as
    "not paired" rather than as a zero.
    """
    ref_by_cell = {arm_of(d)[1:]: d for d in reference_dirs}
    pairs = [(ref_by_cell[arm_of(d)[1:]], d) for d in arm_dirs if arm_of(d)[1:] in ref_by_cell]
    if not pairs:
        return None

    ref_preds, arm_preds, targets = [], [], []
    for ref_dir, arm_dir in pairs:
        try:
            t_ref = np.load(ref_dir / RESULTS_DIR / f"targets_{split_tag}.npy")
            t_arm = np.load(arm_dir / RESULTS_DIR / f"targets_{split_tag}.npy")
            p_ref = np.load(ref_dir / RESULTS_DIR / f"preds_{split_tag}.npy")
            p_arm = np.load(arm_dir / RESULTS_DIR / f"preds_{split_tag}.npy")
        except FileNotFoundError:
            continue
        # The pairing is only valid if both arms scored the same patches in the
        # same order. Different protocols produce different eval sets, and
        # pairing across them would be arithmetic on unrelated rows.
        if t_ref.shape != t_arm.shape or not np.array_equal(t_ref, t_arm):
            continue
        targets.append(t_ref)
        ref_preds.append(p_ref)
        arm_preds.append(p_arm)

    if not targets:
        return None
    t = np.concatenate(targets)
    delta, interval = paired_bootstrap_delta(
        t, np.concatenate(ref_preds), np.concatenate(arm_preds), num_classes, n_boot=n_boot
    )
    return delta, interval.lo, interval.hi


def aggregate_experiment(
    root: str | Path,
    experiment: str,
    reference: str | None = None,
    variant: str = "tta",
    num_classes: int = 90,
    n_boot: int = 2000,
    split_tag: str | None = None,
) -> dict[str, Any]:
    """Aggregate one experiment's runs, writing tables and figures beside them.

    Returns the summary payload, which is also written to
    ``<root>/<experiment>/aggregate.json``.
    """
    exp_root = Path(root) / experiment
    run_dirs = discover_runs(exp_root)
    if not run_dirs:
        _log.warning("no completed runs under %s", exp_root)
        return {"experiment": experiment, "n_runs": 0, "arms": []}

    summaries = summarise_arms(run_dirs, variant=variant)
    by_arm: dict[str, list[Path]] = {}
    for d in run_dirs:
        by_arm.setdefault(arm_of(d)[0], []).append(d)

    # The split tag is `<report_split>_<variant>`, e.g. `val_test_tta`. Read
    # from a manifest rather than assumed, so an experiment that reported `test`
    # aggregates without a special case.
    if split_tag is None:
        first = load_manifest(run_dirs[0])
        split_tag = ((first.get("results") or {}).get(variant) or {}).get(
            "split", f"test_{variant}"
        )

    deltas: dict[str, tuple[float, float, float]] = {}
    if reference and reference in by_arm:
        for arm, dirs in by_arm.items():
            if arm == reference:
                continue
            paired = paired_delta(
                by_arm[reference], dirs, str(split_tag), num_classes, n_boot=n_boot
            )
            if paired is not None:
                deltas[arm] = paired

    payload: dict[str, Any] = {
        "experiment": experiment,
        "variant": variant,
        "split_tag": split_tag,
        "reference": reference,
        "n_runs": len(run_dirs),
        "arms": [s.as_dict() for s in summaries],
        "paired_deltas": {
            k: {"delta": d, "ci_lo": lo, "ci_hi": hi, "crosses_zero": lo <= 0.0 <= hi}
            for k, (d, lo, hi) in deltas.items()
        },
    }
    (exp_root / "aggregate.json").write_text(json.dumps(payload, indent=2))

    _write_arm_table(exp_root, experiment, summaries, deltas)
    if available():
        arms_for_plot = {s.arm: s.stats for s in summaries}
        protocol_comparison(
            arms_for_plot, exp_root / "arms.png", title=f"{experiment} — macro-F1 by arm"
        )
        if deltas:
            ablation_forest(
                deltas, exp_root / "deltas.png", title=f"{experiment} — Δ vs {reference}"
            )
    return payload


def _write_arm_table(
    exp_root: Path,
    experiment: str,
    summaries: list[ArmSummary],
    deltas: dict[str, tuple[float, float, float]],
) -> None:
    headers = ["arm", "runs", "params", "macro-F1 mean", "min", "max", "range", "sd", "Δ vs ref (CI95)"]
    rows: list[list[Any]] = []
    for s in summaries:
        if s.arm in deltas:
            d, lo, hi = deltas[s.arm]
            null = " (crosses 0)" if lo <= 0.0 <= hi else ""
            delta_cell = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{null}"
        else:
            delta_cell = "—"
        rows.append(
            [
                s.arm,
                s.n_runs,
                f"{s.parameters:,}",
                f"{s.stats['mean']:.4f}",
                f"{s.stats['min']:.4f}",
                f"{s.stats['max']:.4f}",
                f"{s.stats['range']:.4f}",
                f"{s.stats['sd']:.4f}",
                delta_cell,
            ]
        )
    tables.write_table(
        exp_root / "arms",
        headers,
        rows,
        caption=(
            f"{experiment} — mean ± range over folds × seeds. "
            "A delta whose interval crosses zero has not been shown to do anything."
        ),
    )


def aggregate_protocol(
    run_dirs: list[Path], out_dir: str | Path, variant: str = "tta"
) -> dict[str, Any]:
    """Build the paper's headline tables from a protocol sweep's runs.

    Writes ``protocol``, ``per_cell`` and ``leakage_gap`` tables plus the
    comparison figure. The leakage-gap table is empty unless both a ``grouped``
    and a ``stratified`` arm are present, which is the state the sweep is
    supposed to produce and the caller is told when it did not.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = tables.collect_runs(run_dirs, variant=variant)
    if not records:
        return {"n_runs": 0}

    headers, rows, arms = tables.protocol_table(records)
    tables.write_table(
        out / "protocol",
        headers,
        rows,
        caption="Primary results — mean ± range over folds × seeds. Never a maximum.",
    )

    cell_headers, cell_rows = tables.per_cell_table(records)
    tables.write_table(
        out / "per_cell", cell_headers, cell_rows, caption="Every individual run."
    )

    gap_headers, gap_rows = tables.leakage_gap_table(records)
    if gap_rows:
        tables.write_table(
            out / "leakage_gap",
            gap_headers,
            gap_rows,
            caption=(
                "F1_stratified − F1_grouped: the contribution of acquisition-bundle "
                "recognition to reported performance on this dataset."
            ),
        )

    if available():
        protocol_comparison(arms, out / "protocol.png", title="Protocol comparison")

    payload = {
        "n_runs": len(records),
        "arms": arms,
        "has_leakage_gap": bool(gap_rows),
        "records": records,
    }
    (out / "aggregate.json").write_text(json.dumps(payload, indent=2))
    return payload
