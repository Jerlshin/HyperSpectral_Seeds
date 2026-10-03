#!/usr/bin/env python3
"""Run S13 — the route-A representation arms Y1–Y4 as a single-seed screen (seed 0).

::

    python scripts/run_s13.py --list                       # the 10 GPU + 2 fused cells, their sources
    python scripts/run_s13.py --check                      # compose every cell, assert its regime
    python scripts/run_s13.py --cfg-job                    # `train.py --cfg job` for every cell
    python scripts/run_s13.py --nproc-per-node 2 --dry-run # the exact commands

    # Kaggle T4 x2 — everything: 10 GPU cells, then Y1's two fused cells, then a summary
    python scripts/run_s13.py --nproc-per-node 2 --stream

    python scripts/run_s13.py --fuse-only                  # (re-)fuse Y1 from finished cells, CPU
    python scripts/run_s13.py --summary                    # what is scored so far

Every GPU cell is built **from the frozen files** after their SHA-256 is checked —
the parent ``preregistration_s12.json`` (``88b377c5…``: the R1 regime) and the
S13 amendment ``preregistration_s13.json`` (the one-seed cell list, D28) — and
nothing runs if either has moved. A cell writes to
``<output-root>/s13/<arm>/<variant>__f<fold>_s0/`` and, before it starts,
``frozen_cell.json`` there. A completed cell (``results/run.json``) is skipped on
rerun; an interrupted one resumes from ``last_stage1.pth``. After the GPU cells,
Y1's fused cells are written as soon as both of a fold's networks are scored.

The logic lives in :mod:`spectralquadnet.experiments.s13`; this is the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from spectralquadnet.experiments import s13
from spectralquadnet.experiments.runner import execute, plan

_log = logging.getLogger("run_s13")
TRAIN = Path(__file__).resolve().parent.parent / "train.py"
ARMS = ("Y1", "Y2", "Y3", "Y4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arms", nargs="+", choices=ARMS, help="default: all four")
    p.add_argument("--cells", nargs="+", help="restrict to cells, e.g. Y2/mixstyle__f0_s0")
    p.add_argument("--output-root", default=s13.DEFAULT_OUTPUT_ROOT)
    p.add_argument("--nproc-per-node", type=int, default=1, help="GPUs per cell (torchrun if > 1)")
    p.add_argument("--list", action="store_true", help="print the cells, run nothing")
    p.add_argument("--check", action="store_true", help="compose every cell and assert its regime")
    p.add_argument("--cfg-job", action="store_true", help="`train.py --cfg job` for every cell")
    p.add_argument("--dry-run", action="store_true", help="print the exact commands, run nothing")
    p.add_argument("--force", action="store_true", help="re-run completed cells")
    p.add_argument("--stream", action="store_true", help="echo each cell's output live")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--no-fuse", action="store_true", help="skip Y1's fusion after the GPU cells")
    p.add_argument("--fuse-only", action="store_true", help="only fuse Y1 (CPU), then summarise")
    p.add_argument("--summary", action="store_true", help="only print what is scored so far")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Hydra override for every cell (runtime knobs only, e.g. runtime.compile=off)",
    )
    return p.parse_args(argv)


def _command(spec, nproc: int, extra: list[str]) -> list[str]:  # type: ignore[no-untyped-def]
    return [*spec.command(train_script=str(TRAIN), nproc_per_node=nproc), *extra]


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary(the_plan: s13.S13Plan, output_root: str) -> None:
    rows = s13.summary_rows(the_plan, output_root)
    cols = ("f1_tta", "f1_no_tta", "same_recall", "cross_recall", "attraction_cross",
            "kappa_embedding", "fusion_weight", "epoch", "commit", "dirty")
    print("\nS13 cells (held-out = val ∪ test; TTA unless named; read in the analysis study)")
    print(f"  {'cell':30s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for row in rows:
        if row.get("status") != "scored":
            print(f"  {row['cell']:30s} {row['status']}")
            continue
        print(f"  {row['cell']:30s} " + " ".join(f"{_fmt(row.get(c)):>12s}" for c in cols))
    out = Path(output_root) / s13.EXPERIMENT / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=str) + "\n")
    print(f"→ {out}")


def run_fusion(the_plan: s13.S13Plan, output_root: str, force: bool) -> int:
    failures = 0
    try:
        for cell, status in s13.fuse(the_plan, output_root, force=force):
            print(f"  {cell.name:30s} {status}")
    except Exception as exc:  # report, keep the GPU results usable
        _log.error("fusion failed: %s", exc)
        failures += 1
    return failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        the_plan = s13.load_plan()
    except s13.PreregistrationError as exc:
        _log.error("%s", exc)
        return 2
    cells = the_plan.select(args.arms, args.cells)
    print(
        f"S13 single-seed screen (seed {the_plan.seed}) — {len(cells)} GPU cells, "
        f"{len(the_plan.fused)} fused | pre-registrations: "
        + ", ".join(f"{k} {v[:8]}…" for k, v in the_plan.hashes.items())
    )

    if args.summary:
        print_summary(the_plan, args.output_root)
        return 0

    if args.list:
        for c in cells:
            print(f"  {c.name:30s} data={c.data} {' '.join(c.arm_overrides)}")
            print(f"  {'':30s} ← {c.source}")
        for f in the_plan.fused:
            print(f"  {f.name:30s} CPU: fuse {f.spectral} + {f.spatial} (weight on calib)")
        print(f"  regime (R1, every cell): {' '.join(the_plan.cells[0].regime)}")
        return 0

    if args.check:
        problems = s13.check_regime_is_r1(the_plan)
        problems += [p for c in cells for p in s13.check_cell(c, args.output_root)]
        for line in problems:
            print("  ✗", line)
        bad = {p.split(":")[0] for p in problems}
        print(f"check: {len(cells) - len(bad & {c.name for c in cells})}/{len(cells)} cells OK")
        return 1 if problems else 0

    if args.cfg_job:
        failed = 0
        for c in cells:
            cmd = [sys.executable, *_command(c.spec(args.output_root), 1, args.override)[1:]]
            proc = subprocess.run([*cmd, "--cfg", "job"], capture_output=True, text=True)
            ok = proc.returncode == 0 and "single:" in proc.stdout
            failed += not ok
            print(f"  {'✓' if ok else '✗'} {c.name}")
            if not ok:
                sys.stderr.write(proc.stderr[-2000:])
        print(f"cfg-job: {len(cells) - failed}/{len(cells)} compose")
        return 1 if failed else 0

    if args.fuse_only:
        failures = run_fusion(the_plan, args.output_root, args.force)
        print_summary(the_plan, args.output_root)
        return 1 if failures else 0

    specs = {c.name: c.spec(args.output_root) for c in cells}
    todo, done = plan(list(specs.values()), force=args.force)
    print(f"{len(done)} already complete, {len(todo)} to run.\n")
    todo_dirs = {s.output_dir for s in todo}
    failures = 0
    for i, cell in enumerate(c for c in cells if specs[c.name].output_dir in todo_dirs):
        spec = specs[cell.name]
        command = _command(spec, args.nproc_per_node, args.override)
        shell = " ".join(shlex.quote(part) for part in command)
        if args.dry_run:
            print(shell)
            continue
        _log.info("[S13 %d/%d] %s", i + 1, len(todo), cell.name)
        out = Path(spec.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frozen_cell.json").write_text(
            json.dumps(s13.provenance(cell, spec, shell, the_plan.hashes), indent=1) + "\n"
        )
        outcome = execute(
            spec,
            train_script=str(TRAIN),
            extra_overrides=tuple(args.override),
            nproc_per_node=args.nproc_per_node,
            stream=args.stream,
        )
        if outcome.status == "failed":
            failures += 1
            _log.error("FAILED %s (rc=%d) — %s", cell.name, outcome.returncode, outcome.log_path)
            if args.stop_on_failure:
                break
    if args.dry_run:
        for f in the_plan.fused:
            print(f"# then, CPU: fuse {f.spectral} + {f.spatial} → s13/{f.name}")
        return 0
    if not args.no_fuse:
        print("\nY1 fusion (CPU, weight on calib):")
        failures += run_fusion(the_plan, args.output_root, args.force)
    print_summary(the_plan, args.output_root)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
