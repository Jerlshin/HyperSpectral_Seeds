"""Publication tables, generated from a directory of runs.

Nothing here computes a metric — it reads ``results/run.json`` from each run and
arranges what it finds. That separation is deliberate: a table is a *view*, and
being able to regenerate every table in the paper from the results tree, without
a GPU, is what makes the numbers checkable by someone who did not run them.

Two output formats, both written for every table: Markdown (for the paper and
for a PR comment) and CSV (for a spreadsheet or a downstream script).

The reporting rules these tables enforce, from CHANGES §19.4
───────────────────────────────────────────────────────────
* **Mean ± range over seeds and folds, never a maximum.** A maximum over
  correlated runs is the same estimator that cost the audited project ~0.042
  macro-F1 of upward bias, applied at a coarser granularity.
* **The number of runs behind every cell is printed.** A mean of one is a
  single-seed result and the table says so rather than looking like six.
* **A delta is never shown without an interval.**
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from spectralquadnet.reporting.artifacts import load_manifest
from spectralquadnet.reporting.metrics import mean_and_range


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-flavoured Markdown table. Columns are left-aligned."""
    body = list(rows)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in body)
    return "\n".join(lines) + "\n"


def csv_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """The same table as CSV. Values are stringified, not quoted — keep them simple."""
    lines = [",".join(str(h) for h in headers)]
    lines.extend(",".join(str(c) for c in row) for row in rows)
    return "\n".join(lines) + "\n"


def write_table(
    path_stem: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]], caption: str = ""
) -> list[Path]:
    """Write ``<stem>.md`` and ``<stem>.csv``; return both paths."""
    body = list(rows)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    md = (f"**{caption}**\n\n" if caption else "") + markdown_table(headers, body)
    md_path = path_stem.with_suffix(".md")
    csv_path = path_stem.with_suffix(".csv")
    md_path.write_text(md)
    csv_path.write_text(csv_table(headers, body))
    return [md_path, csv_path]


# ══════════════════════════════════════════════════════════════════════
#  Collecting runs
# ══════════════════════════════════════════════════════════════════════


def collect_runs(run_dirs: Iterable[str | Path], variant: str = "tta") -> list[dict[str, Any]]:
    """Flatten each run's manifest into one record per run.

    Args:
        variant: ``"tta"`` or ``"no_tta"`` — which scored variant to read.
            Reported separately rather than combined, because TTA is an
            inference-time choice and folding it in hides which number is which.

    Runs with no manifest, or none for ``variant``, are skipped. The caller
    reports the count so a missing cell is visible rather than silently
    absorbed into a mean.
    """
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        manifest = load_manifest(run_dir)
        result = (manifest.get("results") or {}).get(variant)
        if not result:
            continue
        run = manifest.get("run") or {}
        ci = result.get("macro_f1_ci") or {}
        records.append(
            {
                "run_dir": str(run_dir),
                "arch": run.get("arch", "?"),
                "pipeline": run.get("pipeline", "?"),
                "split_scheme": run.get("split_scheme", "?"),
                "fold": run.get("split_fold", 0),
                "seed": run.get("seed", 0),
                "parameters": run.get("parameters", 0),
                "select_split": run.get("select_split", "?"),
                "report_split": result.get("split", "?"),
                "macro_f1": float(result.get("macro_f1", 0.0)),
                "balanced_accuracy": float(result.get("balanced_accuracy", 0.0)),
                "accuracy": float(result.get("accuracy", 0.0)),
                "ci_lo": float(ci.get("lo", 0.0)) if ci else None,
                "ci_hi": float(ci.get("hi", 0.0)) if ci else None,
            }
        )
    return records


def group_by(records: list[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Bucket records by a tuple of field names, preserving first-seen order."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(tuple(record.get(k) for k in keys), []).append(record)
    return grouped


# ══════════════════════════════════════════════════════════════════════
#  The paper's tables
# ══════════════════════════════════════════════════════════════════════


def protocol_table(records: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]], dict[str, dict[str, float]]]:
    """The headline table: one row per (arch, pipeline, protocol), mean ± range.

    Returns ``(headers, rows, arms)`` — ``arms`` keyed by a readable label and
    valued by :func:`~spectralquadnet.reporting.metrics.mean_and_range`, ready
    for :func:`~spectralquadnet.reporting.figures.protocol_comparison`.
    """
    headers = [
        "arch",
        "pipeline",
        "protocol",
        "params",
        "runs",
        "macro-F1 mean",
        "min",
        "max",
        "range",
        "sd",
    ]
    rows: list[list[Any]] = []
    arms: dict[str, dict[str, float]] = {}
    for (arch, pipeline, scheme), bucket in group_by(
        records, ("arch", "pipeline", "split_scheme")
    ).items():
        stats = mean_and_range([r["macro_f1"] for r in bucket])
        label = f"{arch}/{scheme}"
        arms[label] = stats
        rows.append(
            [
                arch,
                pipeline,
                scheme,
                f"{bucket[0]['parameters']:,}",
                stats["n"],
                f"{stats['mean']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['range']:.4f}",
                f"{stats['sd']:.4f}",
            ]
        )
    rows.sort(key=lambda r: (str(r[0]), str(r[2])))
    return headers, rows, arms


def per_cell_table(records: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    """Every individual run — the appendix table that makes the means auditable.

    A summary table with no per-cell breakdown behind it is a table a reviewer
    has to trust. This is the one they can check.
    """
    headers = ["arch", "pipeline", "protocol", "fold", "seed", "macro-F1", "CI95", "bal-acc", "run"]
    rows = [
        [
            r["arch"],
            r["pipeline"],
            r["split_scheme"],
            r["fold"],
            r["seed"],
            f"{r['macro_f1']:.4f}",
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if r["ci_lo"] is not None else "—",
            f"{r['balanced_accuracy']:.4f}",
            Path(r["run_dir"]).name,
        ]
        for r in sorted(
            records, key=lambda r: (r["arch"], r["split_scheme"], r["fold"], r["seed"])
        )
    ]
    return headers, rows


def leakage_gap_table(records: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    """``F1_stratified − F1_grouped`` per architecture — the project's headline result.

    CHANGES §19.2 and §23.2: this gap quantifies *how much of reported rice-seed
    HSI classification performance is acquisition-bundle recognition rather than
    variety recognition*. No published work on this dataset that the audit could
    access answers that, and it is arguably the most valuable number the project
    can produce.

    The gap is between two means over folds × seeds, so its uncertainty is
    reported as the two ranges rather than a single interval: the arms do not
    share a split (that is the entire difference between them), so no paired
    test applies.
    """
    headers = ["arch", "grouped mean", "grouped range", "stratified mean", "stratified range", "gap"]
    rows: list[list[Any]] = []
    for (arch,), bucket in group_by(records, ("arch",)).items():
        grouped = [r["macro_f1"] for r in bucket if r["split_scheme"] == "grouped"]
        strat = [r["macro_f1"] for r in bucket if r["split_scheme"] == "stratified"]
        if not grouped or not strat:
            continue
        g, s = mean_and_range(grouped), mean_and_range(strat)
        rows.append(
            [
                arch,
                f"{g['mean']:.4f}",
                f"{g['min']:.4f}–{g['max']:.4f} (n={g['n']})",
                f"{s['mean']:.4f}",
                f"{s['min']:.4f}–{s['max']:.4f} (n={s['n']})",
                f"{s['mean'] - g['mean']:+.4f}",
            ]
        )
    return headers, rows
