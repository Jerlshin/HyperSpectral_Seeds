#!/usr/bin/env python3
"""Run S15 — replicate the lean network (Y3) at seeds 1–2 and dissect it (Y5): 10 GPU runs.

::

    python scripts/run_s15.py --list                       # the 10 cells, their seeds and frozen sources
    python scripts/run_s15.py --check                      # hashes, code identity, R1, every cell's composition
    python scripts/run_s15.py --cfg-job                    # `train.py --cfg job` for every cell
    python scripts/run_s15.py --nproc-per-node 2 --dry-run # the exact commands

    # Kaggle T4 x2 — everything: Y3 at seeds 1-2 (6 runs) first, then the Y5 dissection (4)
    python scripts/run_s15.py --nproc-per-node 2 --stream

    python scripts/run_s15.py --summary                    # what is scored so far

Every cell is built **from the frozen files** after their SHA-256 is checked —
``preregistration_s12.json`` (``88b377c5…``: R1), ``preregistration_s13.json``
(``ef598213…``) and S14's ``preregistration_s14.json`` (``9e182670…``: the cells,
each with its own seed) — and nothing runs if any has moved, or if the
model/training/data/config code differs from ``aed5257``'s (the code the S13
cells ran; a content digest, so it works in Kaggle's shallow clone). Only the 10
frozen cells exist in the plan; none repeats an S13 cell. A cell writes to
``<output-root>/s15/<arm>/<variant>__f<fold>_s<seed>/`` and, before it starts,
``frozen_cell.json`` there. A completed cell (``results/run.json``) is skipped on
rerun; an interrupted one resumes from ``last_stage1.pth``.

The logic lives in :mod:`spectralquadnet.experiments.s15`; this is the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from spectralquadnet.experiments import s15
from spectralquadnet.experiments.runner import execute, plan

_log = logging.getLogger("run_s15")
TRAIN = Path(__file__).resolve().parent.parent / "train.py"
ARMS = ("Y3", "Y5")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arms", nargs="+", choices=ARMS, help="default: both (Y3 replication, Y5 dissection)")
    p.add_argument("--cells", nargs="+", help="restrict to cells, e.g. Y5/desc_only__f0_s0")
    p.add_argument("--output-root", default=s15.DEFAULT_OUTPUT_ROOT)
    p.add_argument("--nproc-per-node", type=int, default=1, help="GPUs per cell (torchrun if > 1)")
    p.add_argument("--list", action="store_true", help="print the cells, run nothing")
    p.add_argument("--check", action="store_true", help="hashes, code identity, R1 and every cell's composition")
    p.add_argument("--cfg-job", action="store_true", help="`train.py --cfg job` for every cell")
    p.add_argument("--dry-run", action="store_true", help="print the exact commands, run nothing")
    p.add_argument("--force", action="store_true", help="re-run completed cells")
    p.add_argument("--stream", action="store_true", help="echo each cell's output live")
    p.add_argument("--stop-on-failure", action="store_true")
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


def print_summary(the_plan: s15.S15Plan, output_root: str) -> None:
    rows = s15.summary_rows(the_plan, output_root)
    cols = ("f1_tta", "f1_no_tta", "same_recall", "cross_recall", "attraction_cross",
            "kappa_embedding", "parameters", "epoch", "commit", "dirty")
    print("\nS15 cells (held-out = val ∪ test; TTA unless named; read in the analysis study, S16)")
    print(f"  {'cell':30s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for row in rows:
        if row.get("status") != "scored":
            print(f"  {row['cell']:30s} {row['status']}")
            continue
        print(f"  {row['cell']:30s} " + " ".join(f"{_fmt(row.get(c)):>12s}" for c in cols))
    out = Path(output_root) / s15.EXPERIMENT / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=str) + "\n")
    print(f"→ {out}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        the_plan = s15.load_plan()
    except s15.PreregistrationError as exc:
        _log.error("%s", exc)
        return 2
    cells = the_plan.select(args.arms, args.cells)
    code = s15.code_digest()
    identity = s15.check_code_identity()
    print(
        f"S15 Y3 replication + dissection — {len(cells)} GPU cells | pre-registrations: "
        + ", ".join(f"{k} {v[:8]}…" for k, v in the_plan.hashes.items())
        + f" | code {code[0][:8]}… ({code[1]} files) "
        + ("= aed5257" if not identity else "≠ aed5257")
    )

    if args.summary:
        print_summary(the_plan, args.output_root)
        return 0

    if args.list:
        for c in cells:
            print(f"  {c.name:30s} data={c.data} {' '.join(c.arm_overrides)}")
            print(f"  {'':30s} ← {c.source}")
        print(f"  regime (R1, every cell): {' '.join(the_plan.cells[0].regime)}")
        return 0

    if args.check:
        problems = list(identity) + s15.check_regime_is_r1(the_plan)
        problems += [p for c in cells for p in s15.check_cell(c, args.output_root)]
        for line in problems:
            print("  ✗", line)
        bad = {p.split(":")[0] for p in problems}
        print(f"code identity: {'OK' if not identity else 'FAILED'}")
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

    if identity and not args.dry_run:
        for line in identity:
            _log.error("%s", line)
        return 2

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
        _log.info("[S15 %d/%d] %s", i + 1, len(todo), cell.name)
        out = Path(spec.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frozen_cell.json").write_text(
            json.dumps(s15.provenance(cell, spec, shell, the_plan.hashes, code), indent=1) + "\n"
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
        return 0
    print_summary(the_plan, args.output_root)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
