#!/usr/bin/env python3
"""Run the frozen S11 diagnostic arms — X1 fit-first, X2 attribution, X4 fit ceiling.

::

    python scripts/run_frozen.py --list                       # the 23 cells and their sources
    python scripts/run_frozen.py --check                      # compose every cell, assert the regime
    python scripts/run_frozen.py --cfg-job                    # `train.py --cfg job` for every cell
    python scripts/run_frozen.py --arms X1 X4 --nproc-per-node 2 --dry-run   # exact commands

    # Kaggle T4 x2 — X1 and X4 (11 runs), then X2 (12 runs)
    python scripts/run_frozen.py --arms X1 X4 --nproc-per-node 2 --stream
    python scripts/run_frozen.py --arms X2 --nproc-per-node 2 --stream

Every cell is built **from the frozen pre-registration files** (S09
``preregistration_next.json``, S10 ``preregistration_s10.json``) after their
SHA-256 is checked; nothing runs if either file has changed. A cell writes to
``<output-root>/s11/<arm>/<variant>__f<fold>_s<seed>/`` and, before it starts,
``frozen_cell.json`` there: the arm, the frozen source of its overrides, both
hashes and the exact command. A completed cell (``results/run.json``) is
skipped on rerun; an interrupted one resumes from ``last_stage1.pth``.

The logic lives in :mod:`spectralquadnet.experiments.frozen`; this is the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from spectralquadnet.experiments import frozen
from spectralquadnet.experiments.runner import execute, plan

_log = logging.getLogger("run_frozen")
TRAIN = Path(__file__).resolve().parent.parent / "train.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arms", nargs="+", choices=("X1", "X2", "X4"), help="default: all three")
    p.add_argument("--cells", nargs="+", help="restrict to cells, e.g. X1/grouped__f0_s0")
    p.add_argument("--output-root", default=frozen.DEFAULT_OUTPUT_ROOT)
    p.add_argument("--nproc-per-node", type=int, default=1, help="GPUs per cell (torchrun if > 1)")
    p.add_argument("--list", action="store_true", help="print the cells, run nothing")
    p.add_argument("--check", action="store_true", help="compose every cell and assert its regime")
    p.add_argument(
        "--cfg-job",
        action="store_true",
        help="run `train.py --cfg job` for every cell (S10 §9 step 5), run nothing",
    )
    p.add_argument("--dry-run", action="store_true", help="print the exact commands, run nothing")
    p.add_argument("--force", action="store_true", help="re-run completed cells")
    p.add_argument("--stream", action="store_true", help="echo each cell's output live")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "extra Hydra override for every cell (runtime knobs only, e.g. runtime.compile=off); "
            "recorded in frozen_cell.json"
        ),
    )
    return p.parse_args(argv)


def _command(spec, nproc: int, extra: list[str]) -> list[str]:  # type: ignore[no-untyped-def]
    return [*spec.command(train_script=str(TRAIN), nproc_per_node=nproc), *extra]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        frozen_plan = frozen.load_plan()
    except frozen.PreregistrationError as exc:
        _log.error("%s", exc)
        return 2
    cells = frozen_plan.select(args.arms, args.cells)
    if not cells:
        _log.error("no cell matches --arms %s --cells %s", args.arms, args.cells)
        return 2
    print(
        f"S11 frozen arms — {len(cells)} cells | pre-registrations: "
        + ", ".join(f"{k} {v[:8]}…" for k, v in frozen_plan.hashes.items())
    )

    if args.list:
        for c in cells:
            print(f"  {c.name:34s} {' '.join(c.arm_overrides)}")
            print(f"  {'':34s} ← {c.source}")
        return 0

    if args.check:
        problems = [p for c in cells for p in frozen.check_cell(c, args.output_root)]
        for line in problems:
            print("  ✗", line)
        print(
            f"check: {len(cells) - len({p.split(':')[0] for p in problems})}/{len(cells)} cells OK"
        )
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

    specs = {c.name: c.spec(args.output_root) for c in cells}
    todo, done = plan(list(specs.values()), force=args.force)
    print(f"{len(done)} already complete, {len(todo)} to run.\n")
    todo_dirs = {s.output_dir for s in todo}
    failures = 0
    for cell in cells:
        spec = specs[cell.name]
        if spec.output_dir not in todo_dirs:
            continue
        command = _command(spec, args.nproc_per_node, args.override)
        shell = " ".join(shlex.quote(part) for part in command)
        if args.dry_run:
            print(shell)
            continue
        out = Path(spec.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frozen_cell.json").write_text(
            json.dumps(frozen.provenance(cell, spec, shell, frozen_plan.hashes), indent=1) + "\n"
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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
