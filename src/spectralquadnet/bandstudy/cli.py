"""One entrypoint for the spectral-dimensionality study.

::

    python -m spectralquadnet.bandstudy.cli list                 # the plan and its cost
    python -m spectralquadnet.bandstudy.cli all                  # the whole analysis
    python -m spectralquadnet.bandstudy.cli prepare              # features + splits
    python -m spectralquadnet.bandstudy.cli select               # band selection
    python -m spectralquadnet.bandstudy.cli proxy                # the budget sweep
    python -m spectralquadnet.bandstudy.cli analyse              # trends, flags, recommendation
    python -m spectralquadnet.bandstudy.cli confirm              # held-out, once, after analyse
    python -m spectralquadnet.bandstudy.cli neural [--execute]   # neural confirmation
    python -m spectralquadnet.bandstudy.cli report               # REPORT.md + figures
    python -m spectralquadnet.bandstudy.cli inspect              # what has run so far

Every subcommand is independently runnable and **resumable**: a cell that has
already produced its artifact is skipped, so an interrupted sweep picks up where
it stopped and a partial run still yields the tables for the cells that
finished. ``--dry-run`` prints the plan and costs nothing, and is the
recommended first invocation of anything here.

The one ordering that is not a convenience
──────────────────────────────────────────
``confirm`` runs *after* ``analyse``, and refuses otherwise. It is the only
stage that touches ``val ∪ test``, and it scores a fixed list that the
recommendation already fixed. Running it earlier would turn the held-out split
into a selection surface, which is the defect the whole study is built around
avoiding.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from spectralquadnet.bandstudy import methods as bsmethods
from spectralquadnet.bandstudy import neural as bsneural
from spectralquadnet.bandstudy import proxies as bsproxies
from spectralquadnet.bandstudy import report as bsreport
from spectralquadnet.bandstudy.config import (
    DEFAULT_BUDGETS,
    DEFAULT_METHODS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROXIES,
    QUICK_OVERRIDES,
    BandStudyConfig,
    StageResult,
    cost_estimate,
)
from spectralquadnet.bandstudy.pipeline import (
    stage_confirm,
    stage_prepare,
    stage_proxy,
    stage_select,
)
from spectralquadnet.bandstudy.store import (
    MANIFEST,
    console,
    record_stage,
    setup_logging,
    summary_table,
)

# ══════════════════════════════════════════════════════════════════════
#  Config assembly
# ══════════════════════════════════════════════════════════════════════


def config_from_args(args: argparse.Namespace) -> BandStudyConfig:
    """Build the study config from the parsed arguments.

    ``--quick`` is applied *before* the explicit flags, so an explicit
    ``--budgets`` still wins over the preset — otherwise "quick, but with my
    budgets" would silently be "quick".
    """
    overrides: dict[str, Any] = {}
    if getattr(args, "quick", False):
        overrides.update(QUICK_OVERRIDES)

    explicit = {
        "patches_path": args.patches,
        "labels_path": args.labels,
        "groups_path": args.groups,
        "wavelength_path": args.wavelengths,
        "output_root": args.output_root,
        "budgets": tuple(sorted(set(args.budgets))) if args.budgets else None,
        "methods": tuple(args.methods) if args.methods else None,
        "proxies": tuple(args.proxies) if args.proxies else None,
        "folds": tuple(args.folds) if args.folds else None,
        "replicates": args.replicates,
        "features": args.features,
        "seed": args.seed,
        "plateau_tol": args.plateau_tol,
        "stability_floor": args.stability_floor,
        "null_margin": args.null_margin,
        "n_boot": args.n_boot,
        "verbose": args.verbose,
        "force": args.force,
        "dry_run": getattr(args, "dry_run", False),
        "progress": not args.no_progress,
        "note": args.note or "",
    }
    overrides.update({k: v for k, v in explicit.items() if v is not None})

    unknown_methods = [m for m in overrides.get("methods", ()) if m not in bsmethods.METHODS]
    if unknown_methods:
        raise SystemExit(
            f"unknown method(s) {unknown_methods}. Available: {', '.join(sorted(bsmethods.METHODS))}"
        )
    unknown_proxies = [p for p in overrides.get("proxies", ()) if p not in bsproxies.PROXIES]
    if unknown_proxies:
        raise SystemExit(
            f"unknown proxy(s) {unknown_proxies}. Available: {', '.join(sorted(bsproxies.PROXIES))}"
        )
    return BandStudyConfig(**overrides)


# ══════════════════════════════════════════════════════════════════════
#  Subcommands
# ══════════════════════════════════════════════════════════════════════


def cmd_list(args: argparse.Namespace) -> int:
    """Print the plan, its cost and its methods. Runs nothing, reads nothing."""
    cfg = config_from_args(args)
    term = console()
    term.rule("[bold]band study — the plan")

    problems = cfg.validate(n_bands_available=max(cfg.budgets))
    term.print(
        f"budgets   {list(cfg.budgets)}\n"
        f"methods   {list(cfg.methods)}\n"
        f"proxies   {list(cfg.proxies)}\n"
        f"folds     {list(cfg.folds)}  ·  replicates {cfg.replicates} "
        f"({cfg.replicate_frac:.0%} subsamples) + 1 canonical\n"
        f"output    {cfg.root}   fingerprint {cfg.fingerprint()}\n"
    )

    term.print(
        summary_table(
            "Selection methods",
            ["method", "family", "supervised", "nested", "reference"],
            [
                [
                    name,
                    spec.family,
                    "yes" if spec.supervised else "no",
                    "yes" if spec.kind == "ranking" else "no",
                    spec.reference,
                ]
                for name, spec in sorted(bsmethods.METHODS.items())
                if name in cfg.methods
            ],
        )
    )
    term.print(
        summary_table(
            "Proxy estimators",
            ["proxy", "family", "cost in k"],
            [
                [n, s.family, s.cost_note]
                for n, s in sorted(bsproxies.PROXIES.items())
                if n in cfg.proxies
            ],
        )
    )

    cost = cost_estimate(cfg)
    term.print(
        summary_table(
            "Cost",
            ["quantity", "count"],
            [[k.replace("_", " "), f"{v:,}"] for k, v in cost.items()],
        )
    )
    term.print(
        "[dim]Selections are counted per (fold, replicate, method): the ranking methods produce "
        "one ordering and every budget is a prefix of it, so k does not multiply their cost. "
        "Proxy fits dominate the wall clock; identical band sets at one (fold, replicate, proxy) "
        "are fitted once.[/dim]\n"
    )
    term.print(
        "[dim]The neural stage is NOT included above — it is planned from the recommendation "
        "and printed by `neural` before anything is launched.[/dim]"
    )

    if problems:
        term.print("\n[red]Configuration problems:[/red]")
        for problem in problems:
            term.print(f"  [red]•[/red] {problem}")
        return 1
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    setup_logging(cfg, "prepare")
    _print_stage(stage_prepare(cfg))
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    setup_logging(cfg, "select")
    result = stage_select(cfg)
    _print_stage(result)
    return 1 if result.n_failed else 0


def cmd_proxy(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    setup_logging(cfg, "proxy")
    result = stage_proxy(cfg)
    _print_stage(result)
    return 1 if result.n_failed else 0


def cmd_analyse(args: argparse.Namespace) -> int:
    from spectralquadnet.bandstudy.analysis import run_analysis

    cfg = config_from_args(args)
    setup_logging(cfg, "analyse")
    payload = run_analysis(cfg)
    term = console()
    term.rule("[bold]analysis")

    recommendation = payload["recommendation"]
    term.print(
        summary_table(
            "Recommendation (decided on calib — the held-out split has not been read)",
            ["", "value"],
            [
                ["status", recommendation["status"]],
                ["recommended budget", recommendation["recommended_budget"]],
                ["aggressive budget", recommendation.get("aggressive_budget", "—")],
                ["recommended method", recommendation["recommended_method"]],
                ["rationale", recommendation["recommended_method_rationale"]],
                ["pooled σ", f"{recommendation['decision_inputs']['sigma_pooled']:.4f}"],
            ],
        )
    )
    flags = payload["flags"]
    if flags:
        term.print(
            summary_table(
                f"Automated checks — {len(flags)} raised",
                ["severity", "code", "message"],
                [
                    [
                        f["severity"],
                        f["code"],
                        f["message"][:110] + ("…" if len(f["message"]) > 110 else ""),
                    ]
                    for f in flags
                ],
            )
        )
    term.print(f"\n[dim]{recommendation['caveat']}[/dim]")
    term.print(f"\ntables → {cfg.analysis_dir}")
    record_stage(cfg, StageResult(stage="analyse", n_done=int(payload["n_curves"])))
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    setup_logging(cfg, "confirm")
    result = stage_confirm(cfg)
    _print_stage(result)
    return 1 if result.n_failed else 0


def cmd_neural(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    setup_logging(cfg, "neural")
    result = bsneural.stage_neural(
        cfg,
        seeds=tuple(args.seeds),
        include_references=not args.no_references,
        execute=args.execute,
        stop_on_failure=args.stop_on_failure,
    )
    _print_stage(result)
    return 1 if result.n_failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    setup_logging(cfg, "report")
    path = bsreport.build_report(cfg, render_figures=not args.no_figures)
    term = console()
    term.rule("[bold]report")
    term.print(f"→ {path}")
    term.print(f"→ {cfg.figures_dir}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Every stage that does not touch the held-out split or a GPU.

    ``confirm`` and ``neural`` are deliberately excluded and are named at the
    end: the first spends held-out evidence and the second spends GPU-days, and
    neither should happen because somebody typed ``all``.
    """
    from spectralquadnet.bandstudy.analysis import run_analysis

    cfg = config_from_args(args)
    setup_logging(cfg, "all")
    term = console()
    results: list[StageResult] = []

    results.append(stage_prepare(cfg))
    results.append(stage_select(cfg))
    results.append(stage_proxy(cfg))
    if not cfg.dry_run:
        payload = run_analysis(cfg)
        # Recorded, not just printed: `inspect` reads the manifest to say what
        # has run, and a stage missing from it reads as a stage that did not.
        analysed = StageResult(stage="analyse", n_done=payload["n_curves"])
        record_stage(cfg, analysed)
        results.append(analysed)

        bsreport.build_report(cfg, render_figures=not args.no_figures)
        reported = StageResult(stage="report", n_done=1)
        record_stage(cfg, reported)
        results.append(reported)

    term.rule("[bold]band study complete")
    term.print(
        summary_table(
            "Stages",
            ["stage", "done", "resumed", "failed", "seconds"],
            [[r.stage, r.n_done, r.n_skipped, r.n_failed, f"{r.seconds:.1f}"] for r in results],
        )
    )
    if not cfg.dry_run:
        term.print(f"\nreport → [bold]{cfg.report_path}[/bold]")
        term.print(
            "\nNext, in this order:\n"
            f"  1. read {cfg.report_path} §9\n"
            "  2. `python -m spectralquadnet.bandstudy.cli neural` (plan only, no GPU)\n"
            "  3. `python -m spectralquadnet.bandstudy.cli neural --execute` when you can spend "
            "the runs\n"
            "  4. `python -m spectralquadnet.bandstudy.cli confirm` to spend the held-out split, "
            "once\n"
        )
    return 1 if any(r.n_failed for r in results) else 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """What has run, what is left, and what the last recommendation was."""
    cfg = config_from_args(args)
    term = console()
    term.rule(f"[bold]{cfg.root}")

    manifest_path = cfg.root / MANIFEST
    if not manifest_path.exists():
        term.print(f"[yellow]No study at {cfg.root}[/yellow] — nothing has run yet.")
        return 0
    manifest = json.loads(manifest_path.read_text())

    same = manifest.get("fingerprint") == cfg.fingerprint()
    term.print(
        f"fingerprint on disk: {manifest.get('fingerprint')}  "
        + (
            "[green](matches this invocation)[/green]"
            if same
            else "[red](DIFFERENT from this invocation — resuming would mix two experiments)[/red]"
        )
    )
    stages = manifest.get("stages", {})
    if stages:
        term.print(
            summary_table(
                "Stages recorded",
                ["stage", "done", "resumed", "failed", "seconds", "finished"],
                [
                    [
                        name,
                        s.get("n_done", 0),
                        s.get("n_skipped", 0),
                        s.get("n_failed", 0),
                        s.get("seconds", 0),
                        s.get("finished", "—"),
                    ]
                    for name, s in stages.items()
                ],
            )
        )

    counts = []
    for label, path in (
        ("selections", cfg.selections_dir),
        ("band files", cfg.bands_dir),
        ("figures", cfg.figures_dir),
    ):
        n = len(list(path.rglob("*"))) if path.exists() else 0
        counts.append([label, n, str(path)])
    for label, path in (
        ("proxy records", cfg.proxy_dir / "records.jsonl"),
        ("confirm records", cfg.confirm_dir / "records.jsonl"),
    ):
        n = sum(1 for _ in path.open()) if path.exists() else 0
        counts.append([label, n, str(path)])
    term.print(summary_table("Artifacts", ["artifact", "count", "path"], counts))

    rec_path = cfg.analysis_dir / "recommendation.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        term.print(
            summary_table(
                "Last recommendation",
                ["", "value"],
                [
                    ["status", rec.get("status")],
                    ["budget", rec.get("recommended_budget")],
                    ["aggressive", rec.get("aggressive_budget")],
                    ["method", rec.get("recommended_method")],
                ],
            )
        )
    if cfg.report_path.exists():
        term.print(f"\nreport → {cfg.report_path}")
    return 0


# ══════════════════════════════════════════════════════════════════════
#  Wiring
# ══════════════════════════════════════════════════════════════════════


def _print_stage(result: StageResult) -> None:
    term = console()
    term.print(f"\n[bold]{result.line()}[/bold]")
    for note in result.notes[:10]:
        term.print(f"  [dim]{note}[/dim]")


def _add_common(parser: argparse.ArgumentParser) -> None:
    paths = parser.add_argument_group("inputs")
    paths.add_argument(
        "--patches", default=None, help="the FULL 256-band cube (default ./dataset/patches.npy)"
    )
    paths.add_argument("--labels", default=None)
    paths.add_argument("--groups", default=None)
    paths.add_argument("--wavelengths", default=None)
    paths.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    grid = parser.add_argument_group("the sweep")
    grid.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help=f"band counts to evaluate (default {list(DEFAULT_BUDGETS)})",
    )
    grid.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"selection methods (default {list(DEFAULT_METHODS)})",
    )
    grid.add_argument(
        "--proxies",
        nargs="+",
        default=None,
        help=f"proxy estimators (default {list(DEFAULT_PROXIES)})",
    )
    grid.add_argument("--folds", type=int, nargs="+", default=None)
    grid.add_argument("--replicates", type=int, default=None)
    grid.add_argument("--features", default=None, choices=["mean", "mean_sd"])
    grid.add_argument(
        "--quick", action="store_true", help="a tiny preset that exercises every stage in minutes"
    )

    rules = parser.add_argument_group("pre-registered decision rules")
    rules.add_argument("--plateau-tol", type=float, default=None)
    rules.add_argument("--stability-floor", type=float, default=None)
    rules.add_argument("--null-margin", type=float, default=None)
    rules.add_argument("--n-boot", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true", help="recompute cells that already have results"
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--note", default=None, help="free-text provenance for the manifest")
    parser.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectralquadnet.bandstudy.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, function, help_text in (
        ("list", cmd_list, "print the plan and its cost; runs nothing"),
        ("prepare", cmd_prepare, "cache mean spectra and build every fold's splits"),
        ("select", cmd_select, "run every band-selection method on training rows only"),
        ("proxy", cmd_proxy, "fit the proxy estimators at every budget, scored on calib"),
        ("analyse", cmd_analyse, "trends, plateaus, stability, flags and the recommendation"),
        ("report", cmd_report, "assemble REPORT.md and the figures"),
        ("inspect", cmd_inspect, "what has run so far in an output directory"),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
        if name in ("prepare", "select", "proxy"):
            p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
        if name in ("report",):
            p.add_argument("--no-figures", action="store_true")
        p.set_defaults(func=function)

    p_all = sub.add_parser(
        "all", help="prepare + select + proxy + analyse + report (no GPU, no held-out split)"
    )
    _add_common(p_all)
    p_all.add_argument("--dry-run", action="store_true")
    p_all.add_argument("--no-figures", action="store_true")
    p_all.set_defaults(func=cmd_all)

    p_confirm = sub.add_parser(
        "confirm",
        help="score the already-chosen configurations on val ∪ test, once (requires analyse)",
    )
    _add_common(p_confirm)
    p_confirm.add_argument("--dry-run", action="store_true")
    p_confirm.set_defaults(func=cmd_confirm)

    p_neural = sub.add_parser(
        "neural", help="plan (or run) the neural confirmation of the recommendation"
    )
    _add_common(p_neural)
    p_neural.add_argument("--dry-run", action="store_true")
    p_neural.add_argument(
        "--execute",
        action="store_true",
        help="actually launch the runs; without it the plan is only printed",
    )
    p_neural.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p_neural.add_argument(
        "--no-references",
        action="store_true",
        help="omit the full-cube and incumbent reference arms (not advised)",
    )
    p_neural.add_argument("--stop-on-failure", action="store_true")
    p_neural.set_defaults(func=cmd_neural)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        console().print(f"\n[red]{type(exc).__name__}[/red]: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
