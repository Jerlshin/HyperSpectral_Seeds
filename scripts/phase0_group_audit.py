#!/usr/bin/env python3
"""Phase 0 · action 0-H — how many ``(session, scan)`` groups does the data have?

``IMPROVEMENT_PLAN.md`` §2.1.1 (C-1) argues that ``build_splits``' plain
``train_test_split`` puts seeds from the *same physical scan* into train and
test, so the reported number partly measures scan-level memorisation. Fixing
that needs a ``StratifiedGroupKFold`` over a ``scan_id``, which is only
feasible if there are enough groups per class. This script counts them.

How the group id is recovered without re-segmenting anything
────────────────────────────────────────────────────────────
``data/prep/patch_extraction.py`` writes patches **cube by cube**, in the order
``zipfile.ZipFile.infolist()`` yields the ``.hdr`` members, and every patch from
one cube gets that cube's single label. So the on-disk ``labels.npy`` is a
concatenation of constant-label blocks. This script:

1. rebuilds the cube table exactly as ``build_patch_dataset`` does (reading only
   the zip's central directory — no decompression, no segmentation);
2. collapses that table into **capture groups** — maximal runs of consecutive
   cubes sharing a label, which is precisely what a constant-label block in
   ``labels.npy`` corresponds to;
3. run-length-encodes ``labels.npy`` and aligns the two 1:1.

**Resolution limit, stated up front.** The archive holds two cubes per variety
per session (``BC15-01.hdr``, ``BC15-02.hdr``), and they are adjacent in write
order, so their patches form *one* run. Nothing in ``labels.npy`` records where
one cube ends and the next begins, so the finest group this reconstruction can
recover is the ``(session, variety)`` capture group, not the individual cube.
Splitting a group into its cubes would require re-running segmentation over the
archive. Both granularities are reported: the group count is exact, the cube
count comes from the zip index.

The alignment is verified — it fails loudly rather than reporting a guess — and
the recovered ``scan_id`` array is written out so ``P-1``/``T4-1`` can consume
it directly.

Usage::

    python scripts/phase0_group_audit.py
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from spectralquadnet.data.prep.config import PrepConfig


def build_cube_table(zip_path: Path) -> pd.DataFrame:
    """Rebuild ``build_patch_dataset``'s cube table from the zip's index alone.

    Mirrors ``data/prep/patch_extraction.py`` line for line: the same ``.hdr``
    filter, the same ``Data-VIS`` session extraction, the same
    ``stem.rsplit("-", 1)[0]`` variety rule and the same ``pd.factorize`` label
    assignment — so row order, and therefore patch order, is identical.
    """
    zf = zipfile.ZipFile(zip_path, "r")
    cubes = []
    for m in zf.infolist():
        fname = m.filename.lower()
        if not fname.endswith(".hdr") or fname.endswith("black.hdr"):
            continue
        parts = Path(m.filename).parts
        session = next((p for p in parts if p.startswith("Data-VIS")), None)
        if session is None:
            continue
        variety = Path(m.filename).stem.rsplit("-", 1)[0]
        cubes.append((session, variety, m.filename))
    df = pd.DataFrame(cubes, columns=["session", "variety", "member"])
    df["label"] = pd.factorize(df["variety"])[0]
    return df


def rle(a: npt.NDArray[Any]) -> list[tuple[int, int]]:
    """Run-length encode a 1-D integer array as ``[(value, length), ...]``."""
    if len(a) == 0:
        return []
    change = np.flatnonzero(np.diff(a)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(a)]])
    return [(int(a[s]), int(e - s)) for s, e in zip(starts, ends, strict=True)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/phase0", help="Directory for Phase-0 artifacts.")
    args = ap.parse_args()

    prep = PrepConfig()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    labels = np.load(prep.labels_path)
    df = build_cube_table(prep.zip_file)

    print("═" * 72)
    print("PHASE 0 · 0-H — scan/session group census")
    print("═" * 72)
    print(f"Patches on disk           : {len(labels):,}")
    print(f"Classes                   : {labels.max() + 1}")
    print(f"Cube (.hdr) rows in zip   : {len(df):,}")
    print(f"Distinct sessions in zip  : {df.session.nunique()}")
    print(f"Distinct varieties in zip : {df.variety.nunique()}")
    print(f"Distinct labels in zip    : {df.label.nunique()}")

    blocks = rle(labels)
    print(f"\nConstant-label blocks in labels.npy : {len(blocks)}")

    # ── Collapse cubes into capture groups ─────────────────────────────
    # A maximal run of consecutive same-label cubes is exactly one block in
    # labels.npy, because the writer emits them back to back.
    # `Any`-valued because a group row mixes ints, strings and lists; the
    # alternative (a dataclass) would only be re-flattened into the DataFrame below.
    groups: list[dict[str, Any]] = []
    for lbl, run_len in rle(df.label.to_numpy()):
        start = sum(int(g["n_cubes"]) for g in groups)
        rows = df.iloc[start : start + run_len]
        groups.append(
            {
                "scan_id": len(groups),
                "label": int(lbl),
                "n_cubes": run_len,
                "sessions": sorted(set(rows.session)),
                "varieties": sorted(set(rows.variety)),
                "members": list(rows.member),
            }
        )
    print(f"Capture groups from the zip index   : {len(groups)}")

    if len(groups) != len(blocks):
        raise SystemExit(
            f"ALIGNMENT FAILED — {len(groups)} capture groups vs {len(blocks)} label blocks. "
            "Refusing to report group counts from a guessed mapping."
        )

    scan_id = np.full(len(labels), -1, dtype=np.int64)
    patch_cursor = 0
    for g, (blk_label, blk_len) in zip(groups, blocks, strict=True):
        if blk_label != g["label"]:
            raise SystemExit(
                f"ALIGNMENT FAILED — group {g['scan_id']} has label {g['label']} "
                f"but block {g['scan_id']} has label {blk_label}."
            )
        scan_id[patch_cursor : patch_cursor + blk_len] = int(g["scan_id"])
        g["n_patches"] = blk_len
        patch_cursor += blk_len

    if patch_cursor != len(labels) or (scan_id < 0).any():
        raise SystemExit("ALIGNMENT FAILED — not every patch received a scan_id.")
    print(f"Patches assigned a scan_id          : {patch_cursor:,}/{len(labels):,}  ✓")

    multi = sum(1 for g in groups if int(g["n_cubes"]) > 1)
    print(
        f"Groups spanning >1 cube             : {multi} "
        f"(the -01/-02 pairs; not separable from labels.npy alone)"
    )
    mixed = [g for g in groups if len(g["sessions"]) > 1]
    print(f"Groups spanning >1 session          : {len(mixed)}")

    scans = pd.DataFrame(
        [
            {
                "scan_id": g["scan_id"],
                "session": g["sessions"][0],
                "variety": g["varieties"][0],
                "label": g["label"],
                "n_cubes": g["n_cubes"],
                "n_patches": g["n_patches"],
                "members": ";".join(g["members"]),
            }
            for g in groups
        ]
    )

    # ── The numbers 0-H asks for ───────────────────────────────────────
    n_scans = len(scans)
    per_class_scans = scans.groupby("label").size()
    per_class_cubes = scans.groupby("label")["n_cubes"].sum()
    varieties_per_scan = scans.groupby("scan_id")["variety"].nunique()
    scans_per_session = scans.groupby("session").size()
    sessions_per_class = scans.groupby("label")["session"].nunique()

    print("\n── Groups ─────────────────────────────────────────────────")
    print(f"Distinct (session, variety) groups   : {n_scans}")
    print(f"Distinct physical cubes (zip index)  : {int(scans.n_cubes.sum())}")
    print(f"Distinct sessions                    : {scans.session.nunique()}")
    print(
        f"Patches per group min/med/max        : "
        f"{scans.n_patches.min()}/{int(scans.n_patches.median())}/{scans.n_patches.max()}"
    )
    print(
        f"Groups per class  min/med/max        : "
        f"{per_class_scans.min()}/{int(per_class_scans.median())}/{per_class_scans.max()}"
    )
    print(
        f"Cubes per class   min/med/max        : "
        f"{per_class_cubes.min()}/{int(per_class_cubes.median())}/{per_class_cubes.max()}"
    )
    print(
        f"Sessions per class min/med/max       : "
        f"{sessions_per_class.min()}/{int(sessions_per_class.median())}/"
        f"{sessions_per_class.max()}"
    )
    print(
        f"Varieties per group (should be 1)    : "
        f"{varieties_per_scan.min()}–{varieties_per_scan.max()}"
    )

    print("\nGroups-per-class histogram:")
    for k, v in sorted(Counter(per_class_scans.tolist()).items()):
        print(f"  {k} group(s): {v} classes")
    print("Cubes-per-class histogram:")
    for k, v in sorted(Counter(per_class_cubes.tolist()).items()):
        print(f"  {k} cube(s): {v} classes")

    print("\nGroups per session:")
    for s, n in scans_per_session.items():
        print(f"  {s}: {n} groups, {int(scans[scans.session == s].n_patches.sum())} patches")

    # ── Feasibility verdict for a grouped split ────────────────────────
    min_scans = int(per_class_scans.min())
    min_cubes = int(per_class_cubes.min())
    n_single = int((per_class_scans == 1).sum())
    print("\n── Grouped-split feasibility (P-1 / T4-1) ─────────────────")
    if min_scans >= 3:
        verdict = (
            f"FEASIBLE at the (session, variety) level — every class has >= {min_scans} groups."
        )
    elif min_scans == 2:
        verdict = (
            "MARGINAL at the (session, variety) level — some class has only 2 groups, so a "
            "3-way group-disjoint split leaves it absent from one split."
        )
    else:
        verdict = (
            f"INFEASIBLE at the (session, variety) level — {n_single}/90 classes were captured "
            f"in exactly one group, so no class-preserving group-disjoint split exists. "
            f"Cube level: min {min_cubes} cubes/class"
            + (
                " — a cube-disjoint split IS constructible, but the two cubes of a pair share "
                "a session and an illumination setting, so it controls for seed identity and "
                "scan geometry, not for session-level radiometric drift."
                if min_cubes >= 2
                else " — not constructible either."
            )
        )
    print(verdict)

    # ── How much leakage does the *current* split have? ────────────────
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.data.loaders import build_splits

    cfg = load_experiment_config()
    _, tr, va, te = build_splits(cfg)
    tr_s, va_s, te_s = set(scan_id[tr]), set(scan_id[va]), set(scan_id[te])
    print("\n── Leakage in the CURRENT (ungrouped) split ───────────────")
    print(f"Groups in train                     : {len(tr_s)}")
    print(f"Groups in val                       : {len(va_s)}")
    print(f"Groups in test                      : {len(te_s)}")
    print(
        f"Groups shared train∩test            : {len(tr_s & te_s)}  "
        f"({len(tr_s & te_s) / max(len(te_s), 1):.1%} of test groups)"
    )
    print(f"Groups shared train∩val             : {len(tr_s & va_s)}")
    print(
        f"Test patches whose group is in train: "
        f"{int(np.isin(scan_id[te], list(tr_s)).sum()):,}/{len(te):,} "
        f"({np.isin(scan_id[te], list(tr_s)).mean():.1%})"
    )

    np.save(out_root / "scan_id.npy", scan_id)
    scans.to_csv(out_root / "scan_table.csv", index=False)
    summary = {
        "n_patches": int(len(labels)),
        "n_classes": int(labels.max() + 1),
        "n_groups": int(n_scans),
        "n_cubes": int(scans.n_cubes.sum()),
        "n_sessions": int(scans.session.nunique()),
        "group_granularity_note": (
            "A group is one (session, variety) capture; the -01/-02 cube pair inside a group "
            "is not separable from labels.npy alone."
        ),
        "patches_per_group": {
            "min": int(scans.n_patches.min()),
            "median": float(scans.n_patches.median()),
            "max": int(scans.n_patches.max()),
        },
        "groups_per_class": {
            "min": int(per_class_scans.min()),
            "median": float(per_class_scans.median()),
            "max": int(per_class_scans.max()),
            "histogram": {
                str(k): int(v) for k, v in sorted(Counter(per_class_scans.tolist()).items())
            },
        },
        "cubes_per_class": {
            "min": int(per_class_cubes.min()),
            "max": int(per_class_cubes.max()),
            "histogram": {
                str(k): int(v) for k, v in sorted(Counter(per_class_cubes.tolist()).items())
            },
        },
        "sessions_per_class": {
            "min": int(sessions_per_class.min()),
            "max": int(sessions_per_class.max()),
        },
        "varieties_per_group_max": int(varieties_per_scan.max()),
        "classes_with_one_group": n_single,
        "grouped_split_verdict": verdict,
        "current_split_leakage": {
            "groups_train": len(tr_s),
            "groups_val": len(va_s),
            "groups_test": len(te_s),
            "groups_shared_train_test": len(tr_s & te_s),
            "groups_shared_train_val": len(tr_s & va_s),
            "test_patches_with_group_seen_in_train": int(np.isin(scan_id[te], list(tr_s)).sum()),
            "test_patch_leak_fraction": float(np.isin(scan_id[te], list(tr_s)).mean()),
        },
    }
    (out_root / "group_audit.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nWrote {out_root / 'scan_id.npy'}, {out_root / 'scan_table.csv'}, "
        f"{out_root / 'group_audit.json'}"
    )


if __name__ == "__main__":
    main()
