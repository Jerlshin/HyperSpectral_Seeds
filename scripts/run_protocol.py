#!/usr/bin/env python3
"""Sweep the primary protocol and aggregate it (IC-12 / CHANGES §19).

::

    python scripts/run_protocol.py --dry-run       # print the plan, run nothing
    python scripts/run_protocol.py                 # 2 folds x 3 seeds + contrast
    python scripts/run_protocol.py --baseline      # also the classical baselines
    python scripts/run_protocol.py --aggregate-only

What it runs
────────────
``split_fold ∈ {0, 1} × seed ∈ {0, 1, 2}`` under the grouped protocol — the
complete leave-one-acquisition-bundle-out cross-validation this dataset supports
— plus a matched ``stratified`` contrast arm. Selection happens on ``calib``;
``val ∪ test`` is scored once per cell.

It then aggregates to mean ± range, emits the publication tables and the pooled
figures, and reports the ``F1_stratified − F1_grouped`` gap, which is the number
CHANGES §19.2 argues is the most valuable this project can produce: no
accessible published work on this dataset states a bundle-disjoint protocol, so
the gap quantifies how much of reported rice-seed HSI performance is acquisition
recognition rather than variety recognition.

Every cell is an independent ``train.py`` subprocess writing its own output
directory, so the sweep resumes after an interruption and any single number can
be reproduced on its own — ``--dry-run`` prints the exact command for each.

This is a thin wrapper. The logic lives in
:mod:`spectralquadnet.experiments.protocol` and
:mod:`spectralquadnet.experiments.aggregate`, so the CLI at
``python -m spectralquadnet.experiments.cli protocol`` and this script cannot
drift apart.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.experiments import baselines, protocol
from spectralquadnet.experiments.aggregate import aggregate_protocol, discover_runs
from spectralquadnet.experiments.registry import DEFAULT_CONFIG
from spectralquadnet.experiments.runner import DEFAULT_OUTPUT_ROOT, plan, run_all

_log = logging.getLogger("run_protocol")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(protocol.PROTOCOL_SEEDS))
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    parser.add_argument("--force", action="store_true", help="re-run completed cells")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip training; rebuild tables and figures from runs already on disk",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also fit LDA/LinearSVC on mean spectra under both protocols",
    )
    parser.add_argument(
        "--include-audited",
        action="store_true",
        help=(
            "also run the audited 5.19 M / three-stage model under the SAME grouped "
            "protocol — CHANGES §21 Phase 3's like-for-like comparison"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _run_baselines(output_root: str, seeds: list[int]) -> None:
    """The honest floor, under both protocols.

    Costs seconds and needs no GPU, and CHANGES §19.4 calls it *"the paper's
    most important baseline"*: LDA on 40-band mean spectra reaches 0.5916 under
    the leaky protocol, so ~59 of the model's points are available with no
    spatial information at all. Recomputed under `grouped`, it is the control
    every deep-learning number here is measured against.
    """
    for data_config in ("spa40_90class_pfix", "spa40_90class_stratified"):
        for fold in protocol.PROTOCOL_FOLDS if "pfix" in data_config else (0,):
            cfg = load_experiment_config(
                DEFAULT_CONFIG,
                overrides=[f"data={data_config}", f"data.split_fold={fold}", f"seed={seeds[0]}"],
            )
            out = Path(output_root) / "baselines" / f"{cfg.data.split_scheme}_f{fold}"
            _log.info("baselines → %s", out)
            results = baselines.run_baselines(cfg, output_dir=out)
            for name, result in results.items():
                ci = f" CI95={result.macro_f1_ci}" if result.macro_f1_ci else ""
                print(
                    f"  {cfg.data.split_scheme:11s} f{fold}  {name:24s} "
                    f"macro-F1={result.macro_f1:.4f}{ci}"
                )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    specs = protocol.build_specs(
        args.output_root, config=args.config, seeds=tuple(args.seeds), experiment="protocol"
    )
    if args.include_audited:
        specs.extend(
            protocol.build_baseline_comparison_specs(
                args.output_root, seeds=tuple(args.seeds), experiment="protocol"
            )
        )

    info = protocol.summary(specs)
    print(f"\nProtocol sweep — {info['n_runs']} runs  {info['runs_per_arm']}")
    print("\nConstraints that belong in the paper, not in a footnote:")
    for i, line in enumerate(info["constraints"], 1):
        print(f"  {i}. {line}\n")

    failed = 0
    if not args.aggregate_only:
        todo, done = plan(specs, force=args.force)
        print(f"{len(done)} already complete, {len(todo)} to run.\n")
        if args.dry_run:
            for spec in todo:
                print(spec.shell())
            return 0
        report = run_all(
            specs,
            "protocol",
            force=args.force,
            train_script=str(Path(__file__).resolve().parent.parent / "train.py"),
            stop_on_failure=args.stop_on_failure,
        )
        report.write(Path(args.output_root) / "protocol" / "sweep.json")
        failed = len(report.failed)
        for failure in report.failed:
            _log.error("FAILED %s — %s", failure.spec.run_name, failure.log_path)

    if args.baseline and not args.dry_run:
        _run_baselines(args.output_root, args.seeds)

    root = Path(args.output_root) / "protocol"
    run_dirs = discover_runs(root)
    if not run_dirs:
        _log.warning("no completed runs under %s — nothing to aggregate", root)
        return 1 if failed else 0

    payload = aggregate_protocol(run_dirs, root)
    print(f"\nAggregated {payload['n_runs']} runs → {root}")
    for name in ("protocol", "leakage_gap", "per_cell"):
        path = root / f"{name}.md"
        if path.exists():
            print(f"\n{path}:\n{path.read_text()}")
    if not payload.get("has_leakage_gap"):
        _log.warning(
            "No leakage-gap table — the sweep needs BOTH a grouped and a stratified arm "
            "to report the number CHANGES §19.2 calls the headline result."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
