"""One entrypoint for every experiment in CHANGES §19–§21.

::

    python -m spectralquadnet.experiments.cli list
    python -m spectralquadnet.experiments.cli protocol   [--dry-run]
    python -m spectralquadnet.experiments.cli ablate A1  [--dry-run] [--arms grouped]
    python -m spectralquadnet.experiments.cli baseline   [--data spa40_90class_pfix]
    python -m spectralquadnet.experiments.cli leakage
    python -m spectralquadnet.experiments.cli analyse    --run outputs/<run>
    python -m spectralquadnet.experiments.cli aggregate  [--experiment A1]
    python -m spectralquadnet.experiments.cli report

Every subcommand is independently runnable and idempotent: a completed cell is
skipped, so an interrupted grid resumes, and ``--dry-run`` prints the exact
per-cell command so any single number can be reproduced by hand without the
driver.

``--dry-run`` is the recommended first invocation of anything here. It costs
nothing and it prints the plan, which is the artifact a reviewer should see
before a GPU-day is spent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.experiments import aggregate, analysis, baselines, leakage, protocol, registry
from spectralquadnet.experiments.runner import (
    DEFAULT_OUTPUT_ROOT,
    SweepReport,
    expand,
    plan,
    run_all,
)
from spectralquadnet.reporting.artifacts import load_manifest

_log = logging.getLogger("spectralquadnet.experiments")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ══════════════════════════════════════════════════════════════════════
#  Subcommands
# ══════════════════════════════════════════════════════════════════════


def cmd_list(args: argparse.Namespace) -> int:
    """Print the grid, its costs and its ordering."""
    problems = registry.validate_all()
    if problems:
        for problem in problems:
            _log.error("registry: %s", problem)
        return 1

    print(f"\nAblation grid — {len(registry.ABLATIONS)} experiments, "
          f"{registry.total_runs()} training runs total\n")
    print(f"RUN FIRST: {registry.RUN_FIRST} — until run-to-run variance is known, no delta "
          "in this table means anything.\n")
    for key in sorted(registry.ABLATIONS):
        ablation = registry.ABLATIONS[key]
        deps = f"  (after {', '.join(ablation.depends_on)})" if ablation.depends_on else ""
        print(f"── {key}: {ablation.question}{deps}")
        print(f"   {ablation.n_runs} runs = {len(ablation.arms)} arms "
              f"× {len(ablation.folds)} folds × {len(ablation.seeds)} seeds")
        print(f"   rule: {ablation.decision_rule}")
        for arm in ablation.arms:
            marker = " *ref" if arm.name == ablation.reference else ""
            overrides = " ".join(arm.overrides) or "(config defaults)"
            print(f"     - {arm.name}{marker}: {overrides}")
            if arm.note:
                print(f"       {arm.note}")
        print()
    print(f"Analysis-only (no training run): {', '.join(registry.ANALYSIS_ONLY)}")
    print("  A9 — characterise the persistent hard classes. Highest value in the grid: it "
          "may change the research question rather than the model.\n")
    return 0


def cmd_protocol(args: argparse.Namespace) -> int:
    """Run the 2-fold × 3-seed protocol sweep and aggregate it."""
    specs = protocol.build_specs(args.output_root, config=args.config, experiment="protocol")
    info = protocol.summary(specs)
    print(f"\nProtocol sweep: {info['n_runs']} runs  {info['runs_per_arm']}")
    print("\nConstraints that must be stated in the paper:")
    for i, line in enumerate(info["constraints"], 1):
        print(f"  {i}. {line}\n")

    todo, done = plan(specs, force=args.force)
    print(f"{len(done)} already complete, {len(todo)} to run.\n")
    if args.dry_run:
        for spec in todo:
            print(spec.shell())
        return 0

    report = run_all(
        specs, "protocol", dry_run=False, force=args.force, stop_on_failure=args.stop_on_failure
    )
    _write_sweep(report, Path(args.output_root) / "protocol")
    _aggregate_protocol(args.output_root)
    return 1 if report.failed else 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Run one ablation's arms and aggregate them."""
    ablation = registry.get(args.key)
    problems = ablation.validate()
    if problems:
        for problem in problems:
            _log.error("%s", problem)
        return 1
    if ablation.depends_on:
        _log.info(
            "%s is interpreted against %s — run those first if you have not.",
            args.key,
            ", ".join(ablation.depends_on),
        )

    specs = expand(
        ablation,
        output_root=args.output_root,
        seeds=tuple(args.seeds) if args.seeds else None,
        folds=tuple(args.folds) if args.folds else None,
        arms=tuple(args.arms) if args.arms else None,
    )
    todo, done = plan(specs, force=args.force)
    print(f"\n{args.key}: {ablation.question}")
    print(f"rule: {ablation.decision_rule}")
    print(f"{len(specs)} cells — {len(done)} complete, {len(todo)} to run.\n")
    if args.dry_run:
        for spec in todo:
            print(spec.shell())
        return 0

    report = run_all(
        specs, args.key, dry_run=False, force=args.force, stop_on_failure=args.stop_on_failure
    )
    _write_sweep(report, Path(args.output_root) / args.key)
    payload = aggregate.aggregate_experiment(
        args.output_root, args.key, reference=ablation.reference, variant=args.variant
    )
    print(json.dumps(payload.get("paired_deltas", {}), indent=2))
    return 1 if report.failed else 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Fit LDA and LinearSVC on mean spectra under the configured protocol."""
    overrides = [f"data={args.data}"] if args.data else []
    overrides.extend(args.override or [])
    cfg = load_experiment_config(args.config, overrides=overrides)
    out = Path(args.output_root) / "baselines" / f"{cfg.data.split_scheme}_f{cfg.data.split_fold}"
    results = baselines.run_baselines(cfg, output_dir=out, n_boot=args.n_boot)
    for name, result in results.items():
        ci = f"  CI95={result.macro_f1_ci}" if result.macro_f1_ci else ""
        print(f"{name:24s} macro-F1={result.macro_f1:.4f}{ci}  acc={result.accuracy:.4f}")
    print(f"\n→ {out}")
    return 0


def cmd_leakage(args: argparse.Namespace) -> int:
    """Measure the acquisition signal from residual brightness alone."""
    cfg = load_experiment_config(args.config, overrides=args.override or [])
    out = Path(args.output_root) / "leakage"
    report = leakage.run_probe(cfg, output_dir=out, seed=int(cfg.seed))
    if report is None:
        print("gain.npy is not available — nothing measured. Run scripts/prepare_dataset.py.")
        return 0
    print()
    for line in report.lines():
        print(line)
    print(f"\n→ {out}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Run A9 against a finished run's results."""
    cfg = load_experiment_config(args.config, overrides=args.override or [])
    per_class: dict[int, float] = {}
    confusion = None
    if args.run:
        manifest = load_manifest(args.run)
        variant = (manifest.get("results") or {}).get(args.variant) or {}
        per_class = {int(r["class"]): float(r["f1"]) for r in variant.get("per_class", [])}
        cm_path = Path(args.run) / "results" / f"confusion_{variant.get('split','')}.npy"
        if cm_path.exists():
            import numpy as np

            confusion = np.load(cm_path)
        if not per_class:
            _log.warning("no per-class results in %s — falling back to the documented cluster",
                         args.run)

    report = analysis.run_analysis(
        cfg,
        per_class_f1=per_class or None,
        confusion=confusion,
        output_dir=args.run or (Path(args.output_root) / "a9"),
    )
    print(f"\nA9 — hard: {report.hard_classes}   easy: {report.easy_classes}\n")
    print(json.dumps(report.spectral_similarity, indent=2))
    print(json.dumps(report.patch_counts, indent=2))
    if report.morphometrics.get("available"):
        print(json.dumps(report.morphometrics["cohens_d"], indent=2))
    print(f"\nVERDICT\n{report.verdict}\n")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Rebuild tables and figures from runs already on disk. No GPU."""
    if args.experiment:
        ablation = registry.ABLATIONS.get(args.experiment)
        payload = aggregate.aggregate_experiment(
            args.output_root,
            args.experiment,
            reference=ablation.reference if ablation else None,
            variant=args.variant,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    _aggregate_protocol(args.output_root)
    for key in sorted(registry.ABLATIONS):
        if (Path(args.output_root) / key).exists():
            ablation = registry.ABLATIONS[key]
            aggregate.aggregate_experiment(
                args.output_root, key, reference=ablation.reference, variant=args.variant
            )
            print(f"aggregated {key}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Assemble every table into one Markdown report."""
    root = Path(args.output_root)
    out = root / "REPORT.md"
    sections: list[str] = [
        "# Results\n",
        "Generated by `python -m spectralquadnet.experiments.cli report`. "
        "Every number below is regenerable from the run directories alone.\n",
        "## Protocol constraints\n",
    ]
    sections.extend(f"{i}. {line}\n" for i, line in enumerate(protocol.constraints(), 1))

    for name in ("protocol", "leakage_gap", "per_cell"):
        path = root / "protocol" / f"{name}.md"
        if path.exists():
            sections.append(f"\n## {name.replace('_', ' ').title()}\n\n{path.read_text()}")

    for key in sorted(registry.ABLATIONS):
        path = root / key / "arms.md"
        if path.exists():
            ablation = registry.ABLATIONS[key]
            sections.append(
                f"\n## {key} — {ablation.question}\n\n"
                f"*Decision rule: {ablation.decision_rule}*\n\n{path.read_text()}"
            )

    a9 = root / "a9" / "results" / "a9_hard_classes.json"
    if a9.exists():
        payload = json.loads(a9.read_text())
        sections.append(f"\n## A9 — hard-class characterisation\n\n```\n{payload['verdict']}\n```\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections))
    print(f"→ {out}")
    return 0


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _write_sweep(report: SweepReport, out_dir: Path) -> None:
    path = report.write(out_dir / "sweep.json")
    _log.info(
        "%s: %d completed, %d failed → %s",
        report.experiment,
        len(report.completed),
        len(report.failed),
        path,
    )
    for failure in report.failed:
        _log.error("  FAILED %s (rc=%d) — %s", failure.spec.run_name, failure.returncode, failure.log_path)


def _aggregate_protocol(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root) / "protocol"
    run_dirs = aggregate.discover_runs(root)
    if not run_dirs:
        _log.warning("no completed protocol runs under %s", root)
        return {}
    payload = aggregate.aggregate_protocol(run_dirs, root)
    if not payload.get("has_leakage_gap"):
        _log.warning(
            "no leakage-gap table: the sweep needs BOTH a grouped and a stratified arm, and "
            "that gap is the project's headline result (CHANGES §19.2)."
        )
    return payload


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", default=registry.DEFAULT_CONFIG)
    parser.add_argument("--variant", default="tta", choices=["tta", "no_tta"])
    parser.add_argument("-v", "--verbose", action="store_true")


def _add_sweep(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    parser.add_argument("--force", action="store_true", help="re-run cells that already completed")
    parser.add_argument("--stop-on-failure", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectralquadnet.experiments.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="print the ablation grid and its costs")
    _add_common(p_list)
    p_list.set_defaults(func=cmd_list)

    p_proto = sub.add_parser("protocol", help="the 2-fold x 3-seed protocol sweep (CHANGES §19)")
    _add_common(p_proto)
    _add_sweep(p_proto)
    p_proto.set_defaults(func=cmd_protocol)

    p_abl = sub.add_parser("ablate", help="run one ablation (CHANGES §20)")
    p_abl.add_argument("key", help="e.g. A1, A3, A12")
    p_abl.add_argument("--arms", nargs="*", help="restrict to these arm names")
    p_abl.add_argument("--seeds", nargs="*", type=int)
    p_abl.add_argument("--folds", nargs="*", type=int)
    _add_common(p_abl)
    _add_sweep(p_abl)
    p_abl.set_defaults(func=cmd_ablate)

    p_base = sub.add_parser("baseline", help="LDA / LinearSVC on mean spectra")
    p_base.add_argument("--data", default=None, help="data config, e.g. spa40_90class_pfix")
    p_base.add_argument("--override", nargs="*", help="extra Hydra overrides")
    p_base.add_argument("--n-boot", type=int, default=2000)
    _add_common(p_base)
    p_base.set_defaults(func=cmd_baseline)

    p_leak = sub.add_parser("leakage", help="acquisition-signal probe on brightness alone")
    p_leak.add_argument("--override", nargs="*")
    _add_common(p_leak)
    p_leak.set_defaults(func=cmd_leakage)

    p_a9 = sub.add_parser("analyse", help="A9 — characterise the persistent hard classes")
    p_a9.add_argument("--run", default=None, help="a finished run directory to read results from")
    p_a9.add_argument("--override", nargs="*")
    _add_common(p_a9)
    p_a9.set_defaults(func=cmd_analyse)

    p_agg = sub.add_parser("aggregate", help="rebuild tables/figures from runs on disk")
    p_agg.add_argument("--experiment", default=None)
    _add_common(p_agg)
    p_agg.set_defaults(func=cmd_aggregate)

    p_rep = sub.add_parser("report", help="assemble every table into REPORT.md")
    _add_common(p_rep)
    p_rep.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
