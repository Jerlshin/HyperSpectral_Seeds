#!/usr/bin/env python3
"""Build the full 215-band reflectance cube as float16 straight from the Zenodo zip.

``scripts/build_presliced_dataset.py --set all`` writes the same directory, but
it reads the float32 ``./dataset/patches.npy`` (30.4 GB) first. This script is for a
machine that does not keep that cube: it runs the unchanged
:func:`~spectralquadnet.data.prep.patch_extraction.build_patch_dataset` (tile
reflectance, unresolved bands dropped: the defaults every study used) and streams
each extracted float32 patch into a float16 file instead. Only the 15.2 GB float16
cube ever reaches the disk. The float32 → float16 cast error is measured patch by
patch with the same accumulator ``build_presliced`` uses.

After the extraction it checks that the result is the same data the prior studies
trained on, and fails if it is not:

* ``labels``, ``groups``, ``masks``, ``morphology``, ``scan_table.csv``,
  ``wavelengths.csv`` and ``gain`` must equal the reference ``./dataset/`` byte for
  byte, and ``radiometry.json`` must equal it in everything except the archive path.
  This is the row alignment that the S21 complementary folds and every
  checkpoint depend on.
* With ``--k32-reference``, the ``uniform430_k32`` bands of the new cube must equal
  ``dataset_u430k32/patches.npy`` exactly. That cube was cast to float16 from the
  original float32 cube, so this is an end-to-end check of the reflectance values.

The output has the ``build_presliced`` layout (side arrays, ``band_axis.json`` for
all 215 of 215 bands, then ``MANIFEST.json`` written last) and passes
``build_presliced_dataset.py --verify``. The reference copies of the side files are
what get shipped. ``gain.npy`` and the white-tile QC files are compared and then
removed, as ``build_presliced`` leaves them out by default.

Usage
─────
    pip install -e ".[prep]"
    python scripts/build_refl215_f16.py --kaggle-id <user>/rice-hsi-refl215-f16
    python scripts/build_presliced_dataset.py --verify dataset_refl215_f16
    kaggle datasets create -p dataset_refl215_f16

On a finished build, ``--kaggle-id`` alone writes ``dataset-metadata.json`` and exits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spectralquadnet.data.prep import patch_extraction
from spectralquadnet.data.prep.config import PrepConfig
from spectralquadnet.data.prep.preslice import (
    BAND_AXIS_FILE,
    COPIED_ARRAYS,
    COPIED_METADATA,
    MANIFEST_FILE,
    _CastStats,
    _write_manifest,
    band_set_from_manifest,
    sha256_file,
    verify_presliced,
    write_kaggle_metadata,
)

#: Written by the extraction, compared with the reference, not shipped.
EXTRACTION_ONLY = ("gain.npy", "white_tiles.csv", "white_spectra.npz")
KAGGLE_TITLE = "Rice seed HSI reflectance all 215 bands (float16)"
#: Compared byte for byte with the reference ``./dataset/``.
BYTE_EQUAL = (*COPIED_ARRAYS, "scan_table.csv", "wavelengths.csv", "gain.npy")


class Float16Sink:
    """Stands in for the float32 ``.partial`` stream that ``build_patch_dataset`` writes.

    The extraction uses its output stream in three ways only: ``X[i] = patch``,
    ``X.shape`` and ``X.flush()`` (inside ``_finalise_stream``, which then truncates
    the file and renames it into place). This class supports exactly those three.
    Each row is cast on write and the error is measured.
    """

    def __init__(self, path: Path, shape: tuple[int, ...]) -> None:
        self.path = path
        self.shape = shape
        self.stats = _CastStats()
        self.rows_written = 0
        self._mm: npt.NDArray[Any] | None = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            path, mode="w+", dtype=np.float16, shape=shape
        )

    def __setitem__(self, index: int, value: npt.NDArray[Any]) -> None:
        assert self._mm is not None, "write after flush"
        block = np.asarray(value, dtype=np.float32)[None]
        self.stats.check_source(block, int(index))
        cast = block.astype(np.float16)
        self.stats.update(block, cast)
        self._mm[index] = cast[0]
        self.rows_written += 1

    def flush(self) -> None:
        # Called once by `_finalise_stream`. The mapping is released here, before
        # that function truncates and renames the file.
        if self._mm is not None:
            self._mm.flush()  # type: ignore[attr-defined]
            self._mm = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out", type=Path, default=Path("./dataset_refl215_f16"))
    p.add_argument("--archive", type=Path, default=Path("./dataset/rice_hsi.zip"))
    p.add_argument(
        "--reference", type=Path, default=Path("./dataset"), help="the ./dataset the studies used"
    )
    p.add_argument(
        "--k32-reference",
        type=Path,
        default=Path("./dataset_u430k32"),
        help="pre-sliced k32 cube to compare bit-exactly ('' skips)",
    )
    p.add_argument("--finalists", type=Path, default=Path("./outputs/band_finalists"))
    p.add_argument("--kaggle-id", default="", help="<user>/<slug>: write dataset-metadata.json")
    return p.parse_args(argv)


def check_against_reference(out: Path, reference: Path) -> list[str]:
    """Return a list of problems: rebuilt side files that differ from the reference."""
    problems = []
    for name in BYTE_EQUAL:
        if not (reference / name).exists():
            problems.append(f"reference {reference / name} missing")
        elif sha256_file(out / name) != sha256_file(reference / name):
            problems.append(f"{name}: rebuilt file differs from {reference / name}")
    built = json.loads((out / "radiometry.json").read_text())
    ref = json.loads((reference / "radiometry.json").read_text())
    built.pop("archive", None)
    ref.pop("archive", None)
    if built != ref:
        problems.append("radiometry.json: rebuilt record differs from the reference")
    return problems


def check_k32(out: Path, k32: Path, finalists: Path) -> dict[str, Any]:
    """Compare the new cube at the ``uniform430_k32`` bands with the shipped k32 cube."""
    axis = json.loads((k32 / BAND_AXIS_FILE).read_text())
    bands, source_sha = band_set_from_manifest(finalists, axis["set"])
    if bands.tolist() != axis["source_band_indices"]:
        raise SystemExit(f"{finalists} and {k32 / BAND_AXIS_FILE} disagree on the band set")
    if source_sha and source_sha != sha256_file(out / "wavelengths.csv"):
        raise SystemExit("the k32 set was cut from a different wavelength axis")
    full = np.load(out / "patches.npy", mmap_mode="r")
    small = np.load(k32 / "patches.npy", mmap_mode="r")
    if full.shape[0] != small.shape[0] or small.shape[1] != bands.size:
        raise SystemExit(f"shape mismatch: {full.shape} vs {small.shape}")
    differing_rows = 0
    max_abs = 0.0
    for lo in range(0, full.shape[0], 256):
        a = np.asarray(full[lo : lo + 256][:, bands])
        b = np.asarray(small[lo : lo + 256])
        if not np.array_equal(a.view(np.uint16), b.view(np.uint16)):
            diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
            differing_rows += int((diff.reshape(len(a), -1).max(1) > 0).sum())
            max_abs = max(max_abs, float(diff.max()))
    return {
        "reference": str(k32),
        "set": axis["set"],
        "bands": bands.size,
        "rows_compared": int(full.shape[0]),
        "rows_not_bit_identical": differing_rows,
        "max_abs_difference": max_abs,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.out
    if (out / MANIFEST_FILE).exists():
        if args.kaggle_id:  # a finished build: only add the upload metadata (outside the manifest)
            meta = write_kaggle_metadata(out, args.kaggle_id, KAGGLE_TITLE)
            print(f"{meta} written. Upload with:\n  kaggle datasets create -p {out}")
            return 0
        print(f"{out} already holds a finished build; verify it with build_presliced_dataset.py")
        return 1
    if not args.archive.exists():
        print(f"{args.archive} not found (python scripts/prepare_dataset.py --download-only)")
        return 1

    sinks: list[Float16Sink] = []

    def open_float16(path: Path, shape: tuple[int, ...], dtype: Any) -> Float16Sink:
        if np.dtype(dtype) != np.float32 or len(shape) != 4:
            raise RuntimeError(f"unexpected stream {path} {shape} {dtype}")
        # Same file name as the float32 stream: `_finalise_stream` renames `.partial` into place.
        sinks.append(Float16Sink(patch_extraction._partial(path), shape))
        return sinks[-1]

    # The only change to the extraction: where the patch stream is written, and as what.
    patch_extraction._open_stream = open_float16  # type: ignore[assignment]
    started = time.perf_counter()
    patch_extraction.build_patch_dataset(PrepConfig(root=out, archive=args.archive))
    elapsed = time.perf_counter() - started
    if len(sinks) != 1:
        raise SystemExit(f"expected one patch stream, saw {len(sinks)}")
    sink = sinks[0]

    cube = np.load(out / "patches.npy", mmap_mode="r")
    labels = np.load(out / "labels.npy")
    print(f"cube {cube.shape} {cube.dtype}; rows written {sink.rows_written}")
    if cube.dtype != np.float16 or not cube.shape[0] == labels.shape[0] == sink.rows_written:
        raise SystemExit("the extraction did not write one float16 row per label")

    problems = check_against_reference(out, args.reference)
    for line in problems:
        print(f"✗ {line}")
    if problems:
        return 1
    print(f"✓ side arrays, scan table, wavelengths and gain equal {args.reference}")

    # Ship the reference's own files (identical, except radiometry.json's archive path).
    for name in (*COPIED_ARRAYS, *COPIED_METADATA, "wavelengths.csv"):
        shutil.copy2(args.reference / name, out / name)
    for name in EXTRACTION_ONLY:
        (out / name).unlink()

    k32: dict[str, Any] | None = None
    if str(args.k32_reference) not in ("", "."):
        k32 = check_k32(out, args.k32_reference, args.finalists)
        print(f"k32 comparison: {k32}")
        if k32["rows_not_bit_identical"]:
            print("✗ the k32 bands of the new cube differ from the shipped k32 cube")
            return 1
        print("✓ the uniform430_k32 bands are bit-identical to dataset_u430k32/patches.npy")

    wl_lines = (out / "wavelengths.csv").read_text().strip().splitlines()[1:]
    n_bands = int(cube.shape[1])
    axis: dict[str, Any] = {
        "set": "all",
        "source_n_bands": n_bands,
        "source_band_indices": list(range(n_bands)),
        "source_wavelengths_sha256": sha256_file(out / "wavelengths.csv"),
        "wavelengths_nm": [round(float(line.split(",")[-1]), 6) for line in wl_lines],
        "storage_dtype": "float16",
        "cast_error": sink.stats.as_dict(),
        "rows": int(cube.shape[0]),
        "missing_metadata": [],
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "built_from": {
            "archive": args.archive.name,
            "archive_bytes": args.archive.stat().st_size,
            "script": "scripts/build_refl215_f16.py",
            "extraction_seconds": round(elapsed, 1),
            "side_files_equal_reference": str(args.reference),
            "k32_comparison": k32,
        },
        "note": (
            "Every band of the 215-band reflectance axis, stored as float16. band_geometry() "
            "reads this file and reports 215 of 215 bands: full spectrum, no band selection. "
            "Reduced only in precision; the cast error is above."
        ),
    }
    if len(axis["wavelengths_nm"]) != n_bands:
        raise SystemExit("wavelengths.csv does not describe the cube")
    (out / BAND_AXIS_FILE).write_text(json.dumps(axis, indent=2) + "\n")
    manifest = _write_manifest(out, axis, elapsed)
    if args.kaggle_id:
        write_kaggle_metadata(out, args.kaggle_id, KAGGLE_TITLE)

    problems = verify_presliced(out, checksums=True)
    for line in problems:
        print(f"✗ {line}")
    print(f"\n{out}  ({manifest['total_bytes'] / 1e9:.2f} GB, {manifest['bands']} bands)")
    for name, entry in manifest["files"].items():
        print(f"  {entry['bytes'] / 1e6:>10.1f} MB  {entry['sha256'][:16]}…  {name}")
    print(json.dumps(axis["cast_error"], indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
