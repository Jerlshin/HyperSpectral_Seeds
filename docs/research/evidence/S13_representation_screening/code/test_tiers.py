#!/usr/bin/env python3
"""Compare two ``pytest --run-all -rfE`` logs — the S13 tree and ``413a11e`` — into ``test_tiers.json``.

Usage (after running ``pytest tests --run-all -rfE -o addopts="" -q > <log>`` in each tree)::

    python docs/research/evidence/S13_representation_screening/code/test_tiers.py \\
        --new new_all.txt --old old_all.txt \\
        --out docs/research/evidence/S13_representation_screening/test_tiers.json

A test that fails or errors on the S13 tree and not on ``413a11e`` is a regression;
the pre-existing set (FW-26 pinned reference, FW-11 golden drift, stale fixtures)
must be identical.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUMMARY = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|warnings?)")


def parse(path: Path) -> dict[str, object]:
    text = path.read_text()
    counts: dict[str, int] = {}
    for line in text.splitlines()[::-1]:
        if re.search(r"\d+ (passed|failed)", line) and (" in " in line):
            for n, kind in SUMMARY.findall(line):
                counts[kind.rstrip("s") if kind.startswith(("error", "warning")) else kind] = int(n)
            break
    failed = sorted({m.split(" - ")[0] for m in re.findall(r"^FAILED (\S+)", text, re.M)})
    errors = sorted({m.split(" - ")[0] for m in re.findall(r"^ERROR (\S+)", text, re.M)})
    return {"counts": counts, "failed": failed, "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    new, old = parse(args.new), parse(args.old)
    bad_new = set(new["failed"]) | set(new["errors"])  # type: ignore[arg-type]
    bad_old = set(old["failed"]) | set(old["errors"])  # type: ignore[arg-type]
    payload = {
        "command": "pytest tests --run-all -rfE -o addopts='' -q (macOS CPU, torch 2.13, ai_env)",
        "s13_tree": new,
        "base_413a11e": old,
        "regressions": sorted(bad_new - bad_old),
        "fixed": sorted(bad_old - bad_new),
        "failing_sets_identical": bad_new == bad_old,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k in ("regressions", "fixed", "failing_sets_identical")}))
    print("S13:", new["counts"], "| 413a11e:", old["counts"])


if __name__ == "__main__":
    main()
