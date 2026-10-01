#!/usr/bin/env python3
"""Write a finalist band set of ./dataset as a small, self-contained dataset to upload.

Thin CLI over :mod:`spectralquadnet.data.prep.preslice`. The default is the
evenly spaced 32-band finalist (``uniform430_k32``) stored as float16: 2.3 GB
instead of the 30.4 GB 215-band cube, with every side array, the session table,
the radiometry record, a provenance file (``band_axis.json``) and a checksummed
``MANIFEST.json``.

Usage
─────
    python scripts/build_presliced_dataset.py                       # -> ./dataset_u430k32
    python scripts/build_presliced_dataset.py --set uniform430_k48 --out ./dataset_u430k48
    python scripts/build_presliced_dataset.py --set all --out ./dataset_refl215_f16
    python scripts/build_presliced_dataset.py --kaggle-id <user>/rice-hsi-u430k32
    python scripts/build_presliced_dataset.py --verify ./dataset_u430k32

Train on it with ``data=ablation/u430k32_grouped`` (or ``_stratified``); the run
reports itself as a REDUCED arm — 32 of 215 acquired bands — because
``band_axis.json`` says so.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spectralquadnet.data.prep.preslice import (
    STORAGE_DTYPES,
    band_set_from_manifest,
    build_presliced,
    verify_presliced,
    write_kaggle_metadata,
)

DEFAULT_SET = "uniform430_k32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", type=Path, default=Path("./dataset"))
    parser.add_argument(
        "--set",
        default=DEFAULT_SET,
        help="a set in <finalists>/manifest.json, or 'all' for every band (precision-only)",
    )
    parser.add_argument("--finalists", type=Path, default=Path("./outputs/band_finalists"))
    parser.add_argument("--out", type=Path, default=None, help="default: ./dataset_u430k<k>")
    parser.add_argument("--dtype", choices=sorted(STORAGE_DTYPES), default="float16")
    parser.add_argument(
        "--with-gain", action="store_true", help="also copy gain.npy (leakage probe only)"
    )
    parser.add_argument(
        "--kaggle-id",
        default="",
        help="<user>/<slug>: also write dataset-metadata.json for `kaggle datasets create`",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        metavar="DIR",
        help="re-check an existing build against its MANIFEST.json and exit",
    )
    parser.add_argument(
        "--quick", action="store_true", help="with --verify: compare sizes, skip re-hashing"
    )
    return parser.parse_args(argv)


def _default_out(set_name: str) -> Path:
    if set_name == "all":
        return Path("./dataset_refl215_f16")
    return Path("./dataset_" + set_name.replace("uniform430_", "u430"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.verify is not None:
        problems = verify_presliced(args.verify, checksums=not args.quick)
        for line in problems:
            print(f"✗ {line}")
        if not problems:
            print(f"✓ {args.verify} matches its manifest")
        return 1 if problems else 0

    if args.set == "all":
        bands, source_sha = None, ""
    else:
        bands, source_sha = band_set_from_manifest(args.finalists, args.set)
    out = args.out or _default_out(args.set)

    def progress(done: int, total: int) -> None:
        print(f"\r  patches {done:>5d}/{total}", end="" if done < total else "\n", flush=True)

    manifest = build_presliced(
        args.source,
        out,
        bands,
        set_name=args.set,
        storage_dtype=args.dtype,
        include_gain=args.with_gain,
        expected_source_sha256=source_sha,
        progress=progress,
    )
    print(f"\n{out}  ({manifest['total_bytes'] / 1e9:.2f} GB, {manifest['bands']} bands)")
    for name, entry in manifest["files"].items():
        print(f"  {entry['bytes'] / 1e6:>10.1f} MB  {name}")

    if args.kaggle_id:
        meta = write_kaggle_metadata(
            out, args.kaggle_id, f"Rice seed HSI reflectance {args.set} ({args.dtype})"
        )
        print(f"\n{meta} written. Upload with:\n  kaggle datasets create -p {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
