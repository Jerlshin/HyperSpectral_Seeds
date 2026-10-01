"""Materialise a band subset of the reflectance cube as a self-contained dataset.

Why this exists
───────────────
``data.band_indices_path`` makes a k-band arm a config change, but it still
needs the full 215-band cube on the machine — 30.4 GB, most of which a 32-band
run never reads. On a remote accelerator (a Kaggle T4 x2 session) the cube has
to be *uploaded* first, and the upload is the cost. This module writes the
subset once, as a directory that is a complete ``./dataset/`` in its own right:

=======================  ===============================================
``patches.npy``          ``(N, k, 64, 64)``, float16 by default
``wavelengths.csv``      the k source rows, instrument band numbers kept
``labels.npy``           copied
``groups.npy``           copied
``masks.npy``            copied (already float16)
``morphology.npy``       copied
``scan_table.csv``       copied — the session breakdown reads it
``radiometry.json``      copied — how the reflectance was made
``gain.npy``             only with ``include_gain`` (leakage probe only)
``band_axis.json``       provenance: which source bands, and what storage cost
``MANIFEST.json``        every file's size and SHA-256; written **last**
=======================  ===============================================

The provenance is load-bearing, not decoration
──────────────────────────────────────────────
A 32-band cube read in full looks exactly like "the full acquired cube, no band
selection" to :func:`~spectralquadnet.data.mmap_store.band_geometry`: the cube,
the wavelength vector and ``data.num_bands`` all say 32. Without
``band_axis.json`` a reduced arm would print the primary-methodology banner and
record ``"band_selection": false`` in its results, which is the one claim the
project's methodology exists to keep honest. With it, the run knows it reads 32
of 215 acquired bands and says so.

Storage precision
─────────────────
float16 halves the upload (k=32: 2.26 GB instead of 4.52 GB). Reflectance here
lies in ``[0, ~1.5]``, far inside float16's range, and the background is exact
zero, which float16 represents exactly. For every value in float16's normal
range (|x| ≥ 6.1e-5) the rounding is a relative error of at most 2^-11 ≈ 4.9e-4
— below the 12-bit sensor's own quantisation step for any pixel darker than the
white tile. Smaller values lose relative precision and the very smallest
(< 3e-8) flush to zero. That is an argument, so the build also *measures* it and
writes it into ``band_axis.json``: the maximum absolute error, the maximum
relative error over normal-range values, the RMS error, how many non-zero values
fell below the normal range, and how many flushed to zero and the largest of
them. A value float16 cannot represent (|x| > 65504, or a non-finite input)
fails the build. ``storage_dtype=float32`` writes a bit-identical subset
instead.

On the 32-band finalist of the reflectance cube: max relative error 4.88e-4, and
16 of 2.9e8 non-zero values flushed to zero, the largest 4.9e-16 — no pixel's
foreground status (Σ|x| > 1e-5) changes.

The loader widens a float16 patch to float32 as it comes off the mapping, so
every augmentation and every model input is float32 exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_log = logging.getLogger(__name__)

#: The provenance sidecar ``band_geometry`` reads next to ``patches.npy``.
BAND_AXIS_FILE = "band_axis.json"
#: Written last; its presence means the build finished.
MANIFEST_FILE = "MANIFEST.json"

#: Side arrays copied verbatim — row-aligned with the cube, band-free.
COPIED_ARRAYS: tuple[str, ...] = ("labels.npy", "groups.npy", "masks.npy", "morphology.npy")
#: Metadata the reporting path reads — copied when present. Without
#: ``scan_table.csv`` a run reports no session breakdown; the build warns and
#: records the absence in ``MANIFEST.json`` rather than failing.
COPIED_METADATA: tuple[str, ...] = ("scan_table.csv", "radiometry.json")

STORAGE_DTYPES: dict[str, type[np.floating[Any]]] = {
    "float16": np.float16,
    "float32": np.float32,
}

#: Rows per read/write chunk: 128 x 215 x 64 x 64 x 4 B = 0.45 GB of source at
#: most (plus the cast and its error), which keeps the resident set bounded on a
#: laptop.
DEFAULT_CHUNK_ROWS = 128


class PresliceError(ValueError):
    """The requested subset cannot be written faithfully."""


def band_set_from_manifest(finalists_dir: Path, name: str) -> tuple[npt.NDArray[np.int64], str]:
    """The indices of finalist set ``name`` and the source-axis SHA-256 it was cut from.

    Raises:
        PresliceError: The manifest or the set does not exist.
    """
    manifest_path = Path(finalists_dir) / "manifest.json"
    if not manifest_path.exists():
        raise PresliceError(
            f"{manifest_path} not found — write it with scripts/write_finalist_bands.py"
        )
    manifest = json.loads(manifest_path.read_text())
    sets = manifest.get("sets", {})
    if name not in sets:
        raise PresliceError(f"band set {name!r} not in {manifest_path}; have {sorted(sets)}")
    bands = np.asarray(sets[name]["bands"], dtype=np.int64)
    return bands, str(manifest.get("source_sha256", ""))


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def build_presliced(
    source_root: Path,
    out_root: Path,
    band_indices: Sequence[int] | npt.NDArray[Any] | None,
    *,
    set_name: str,
    storage_dtype: str = "float16",
    include_gain: bool = False,
    expected_source_sha256: str = "",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Write ``source_root``'s cube restricted to ``band_indices`` into ``out_root``.

    Args:
        source_root: A ``./dataset/`` directory as ``prepare_dataset.py`` writes it.
        out_root: Destination. Must not already hold a finished build.
        band_indices: Indices into the source cube's band axis, or ``None`` for
            every band (a precision-only reduction).
        set_name: Recorded in the provenance, e.g. ``uniform430_k32``.
        storage_dtype: ``float16`` or ``float32``.
        include_gain: Also copy ``gain.npy`` — only the leakage probe reads it.
        expected_source_sha256: When given, the source ``wavelengths.csv`` must
            hash to it — the band set was cut from exactly that axis.
        progress: Called with ``(rows_done, rows_total)`` after each chunk.

    Returns:
        The ``MANIFEST.json`` payload.

    Raises:
        PresliceError: Unknown dtype, an index off the axis or duplicated, a
            source-axis mismatch, a value the storage dtype cannot hold, or a
            destination that already holds a finished build.
        FileNotFoundError: A required source file is missing.
    """
    source_root, out_root = Path(source_root), Path(out_root)
    if storage_dtype not in STORAGE_DTYPES:
        raise PresliceError(f"storage_dtype must be one of {sorted(STORAGE_DTYPES)}")
    if (out_root / MANIFEST_FILE).exists():
        raise PresliceError(
            f"{out_root} already holds a finished build ({MANIFEST_FILE}); delete it or pick "
            "another --out"
        )
    for name in ("patches.npy", "wavelengths.csv", *COPIED_ARRAYS):
        if not (source_root / name).exists():
            raise FileNotFoundError(source_root / name)
    missing_metadata = [n for n in COPIED_METADATA if not (source_root / n).exists()]
    for name in missing_metadata:
        _log.warning("%s has no %s — the pre-sliced cube will not carry it", source_root, name)

    wl_path = source_root / "wavelengths.csv"
    source_sha = sha256_file(wl_path)
    if expected_source_sha256 and expected_source_sha256 != source_sha:
        raise PresliceError(
            f"the band set was cut from a wavelength axis with SHA-256 "
            f"{expected_source_sha256[:12]}…, but {wl_path} hashes to {source_sha[:12]}…. "
            "Regenerate the set from this cube (scripts/write_finalist_bands.py)."
        )

    src = np.load(source_root / "patches.npy", mmap_mode="r")
    n_rows, n_source = int(src.shape[0]), int(src.shape[1])
    wl = pd.read_csv(wl_path, sep=None, engine="python")
    if len(wl) != n_source:
        raise PresliceError(f"{wl_path} has {len(wl)} rows but the cube stores {n_source} bands")

    idx = (
        np.arange(n_source, dtype=np.int64)
        if band_indices is None
        else np.asarray(band_indices, dtype=np.int64).reshape(-1)
    )
    if idx.size == 0 or int(idx.min()) < 0 or int(idx.max()) >= n_source:
        raise PresliceError(f"band indices must lie in 0..{n_source - 1}")
    if np.unique(idx).size != idx.size:
        raise PresliceError("band indices contain duplicates")
    contiguous_all = idx.size == n_source and bool(np.all(idx == np.arange(n_source)))

    out_root.mkdir(parents=True, exist_ok=True)
    dtype = STORAGE_DTYPES[storage_dtype]
    shape = (n_rows, int(idx.size), *src.shape[2:])
    dst = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        out_root / "patches.npy", mode="w+", dtype=dtype, shape=shape
    )

    stats = _CastStats()
    started = time.perf_counter()
    for lo in range(0, n_rows, chunk_rows):
        hi = min(n_rows, lo + chunk_rows)
        # A basic slice of the mapping is a view; the band index is the copy,
        # and it touches only the selected bands' pages.
        block = np.asarray(src[lo:hi] if contiguous_all else src[lo:hi][:, idx], dtype=np.float32)
        stats.check_source(block, lo)
        cast = block.astype(dtype)
        stats.update(block, cast)
        dst[lo:hi] = cast
        if progress is not None:
            progress(hi, n_rows)
    dst.flush()
    del dst
    elapsed = time.perf_counter() - started

    wl.iloc[idx].to_csv(out_root / "wavelengths.csv", index=False)
    for name in COPIED_ARRAYS:
        shutil.copy2(source_root / name, out_root / name)
    for name in COPIED_METADATA:
        if name not in missing_metadata:
            shutil.copy2(source_root / name, out_root / name)
    if include_gain:
        if not (source_root / "gain.npy").exists():
            raise FileNotFoundError(source_root / "gain.npy")
        shutil.copy2(source_root / "gain.npy", out_root / "gain.npy")

    wavelengths_nm = wl.iloc[idx, -1].to_numpy(dtype=float)
    axis = {
        "set": set_name,
        "source_n_bands": n_source,
        "source_band_indices": idx.tolist(),
        "source_wavelengths_sha256": source_sha,
        "wavelengths_nm": [round(float(w), 6) for w in wavelengths_nm],
        "storage_dtype": storage_dtype,
        "cast_error": stats.as_dict(),
        "rows": n_rows,
        "missing_metadata": missing_metadata,
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": (
            "A band subset of the acquired cube. band_geometry() reads this file and "
            "reports the run as a REDUCED arm; delete it and the run would claim the "
            "full cube."
        ),
    }
    (out_root / BAND_AXIS_FILE).write_text(json.dumps(axis, indent=2) + "\n")

    manifest = _write_manifest(out_root, axis, elapsed)
    _log.info(
        "wrote %s: %d x %d bands (%s), %.2f GB in %.0f s",
        out_root,
        n_rows,
        idx.size,
        storage_dtype,
        manifest["total_bytes"] / 1e9,
        elapsed,
    )
    return manifest


def verify_presliced(root: Path, *, checksums: bool = True) -> list[str]:
    """Re-check a build against its ``MANIFEST.json``. Returns the problems found.

    Sizes are always compared; ``checksums`` also re-hashes every file, which
    is what catches a truncated or corrupted upload.
    """
    root = Path(root)
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.exists():
        return [f"{manifest_path} missing — the build did not finish"]
    manifest = json.loads(manifest_path.read_text())
    problems: list[str] = []
    for name, entry in manifest["files"].items():
        path = root / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"{name}: {path.stat().st_size} bytes, manifest says {entry['bytes']}")
            continue
        if checksums and sha256_file(path) != entry["sha256"]:
            problems.append(f"{name}: SHA-256 mismatch")
    cube = np.load(root / "patches.npy", mmap_mode="r")
    axis = json.loads((root / BAND_AXIS_FILE).read_text())
    n_wl = len(pd.read_csv(root / "wavelengths.csv", sep=None, engine="python"))
    if not (cube.shape[1] == len(axis["source_band_indices"]) == n_wl):
        problems.append(
            f"band axis disagrees: cube {cube.shape[1]}, band_axis.json "
            f"{len(axis['source_band_indices'])}, wavelengths.csv {n_wl}"
        )
    labels = np.load(root / "labels.npy", mmap_mode="r")
    if cube.shape[0] != labels.shape[0]:
        problems.append(f"cube has {cube.shape[0]} rows, labels.npy {labels.shape[0]}")
    return problems


def write_kaggle_metadata(out_root: Path, dataset_id: str, title: str) -> Path:
    """``dataset-metadata.json`` for ``kaggle datasets create -p <out_root>``.

    The licence is left as ``other``: the source record's terms are the
    uploader's to state, not this script's to guess.
    """
    if "/" not in dataset_id:
        raise PresliceError("--kaggle-id must be <kaggle-username>/<dataset-slug>")
    payload = {"title": title, "id": dataset_id, "licenses": [{"name": "other"}]}
    path = Path(out_root) / "dataset-metadata.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ══════════════════════════════════════════════════════════════════════
#  Internals
# ══════════════════════════════════════════════════════════════════════


class _CastStats:
    """Running error of the storage cast, measured rather than argued."""

    #: Smallest normal float16; below it the relative-error bound no longer holds.
    NORMAL_MIN = float(np.finfo(np.float16).tiny)

    def __init__(self) -> None:
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.sq_sum = 0.0
        self.count = 0
        self.nonzero = 0
        self.subnormal = 0
        self.flushed = 0
        self.max_flushed = 0.0
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def check_source(self, block: npt.NDArray[np.float32], row0: int) -> None:
        if not np.isfinite(block).all():
            bad = row0 + int(np.argwhere(~np.isfinite(block))[0][0])
            raise PresliceError(f"non-finite value in the source cube at row {bad}")
        self.min_value = min(self.min_value, float(block.min()))
        self.max_value = max(self.max_value, float(block.max()))
        if max(abs(self.min_value), abs(self.max_value)) > float(np.finfo(np.float16).max):
            raise PresliceError(
                "the cube holds values outside float16's range; use --dtype float32"
            )

    def update(self, block: npt.NDArray[np.float32], cast: npt.NDArray[Any]) -> None:
        err = np.abs(cast.astype(np.float32) - block)
        magnitude = np.abs(block)
        self.max_abs = max(self.max_abs, float(err.max()))
        normal = magnitude >= self.NORMAL_MIN
        if normal.any():
            self.max_rel = max(self.max_rel, float((err[normal] / magnitude[normal]).max()))
        nonzero = magnitude > 0
        self.nonzero += int(nonzero.sum())
        self.subnormal += int((nonzero & ~normal).sum())
        flushed = nonzero & (cast == 0)
        if flushed.any():
            self.flushed += int(flushed.sum())
            self.max_flushed = max(self.max_flushed, float(magnitude[flushed].max()))
        self.sq_sum += float(np.square(err, dtype=np.float64).sum())
        self.count += int(block.size)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "max_abs": self.max_abs,
            "max_rel_normal_range": self.max_rel,
            "rms": float(np.sqrt(self.sq_sum / max(self.count, 1))),
            "nonzero_values": self.nonzero,
            "below_normal_range": self.subnormal,
            "flushed_to_zero": self.flushed,
            "max_flushed_magnitude": self.max_flushed,
            "source_min": self.min_value,
            "source_max": self.max_value,
        }


def _write_manifest(out_root: Path, axis: dict[str, Any], elapsed: float) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(out_root.iterdir()):
        if path.is_file() and path.name not in (MANIFEST_FILE, "dataset-metadata.json"):
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "set": axis["set"],
        "bands": len(axis["source_band_indices"]),
        "source_n_bands": axis["source_n_bands"],
        "storage_dtype": axis["storage_dtype"],
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "build_seconds": round(elapsed, 1),
        "files": files,
    }
    (out_root / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
