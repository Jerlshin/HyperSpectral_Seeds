"""Three-pass patch extraction: count → allocate → write.

The CLI entry point is ``scripts/prepare_dataset.py``.

The pass-1/pass-2 duplication (each walks the same zip archive and reruns
segmentation) is deliberate: the counting pass exists so pass 2 can allocate
the exact ``(N, num_bands, patch_size, patch_size)`` float32 array up front,
avoiding either a list-of-arrays intermediate or a resizable-array
implementation for a multi-GB output.

Tier 4 (IMPROVEMENT_PLAN §3.1) changes what this pipeline writes and, for one
item, the order in which it computes it:

* **P-1 / T4-1 — ``groups.npy``.** The cube identity ``<session>/<variety>-<n>``
  is available in the loop and was discarded. It is the group key a
  session/scan-disjoint split needs, and without it no split can be anything
  but patch-level. Written alongside ``scan_table.csv``, which names each id.
* **P-2 / T4-2 — radiometry.** The data are radiance, so a per-session
  illumination gain multiplies every spectrum in that session (§2.1.1).
  :mod:`~spectralquadnet.data.prep.radiometry` divides it out.
* **P-3 / T4-3 — the resize order.** ``INTER_AREA`` on an already-masked patch
  averages seed pixels with background zeros, so every boundary pixel came out
  attenuated by its fill fraction and the "exactly zero background" invariant
  held only in the interior (M-11). The mask is now resized **as well**, and it
  *is* the fill map: dividing by it restores the unattenuated spectrum, and
  thresholding it re-establishes an exact zero background.
* **P-4 / T4-4 — ``morphology.npy``.** Eight shape descriptors that
  ``segment`` already computed, gated on, and threw away (M-13).

Every one of these needs ``scripts/prepare_dataset.py`` to be re-run; none of
them changes an array that already exists on disk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from spectralquadnet.data.prep.config import SEED_ROWS, PrepConfig, TileConfig
from spectralquadnet.data.prep.download import download
from spectralquadnet.data.prep.radiometry import (
    RADIOMETRY_EPS,
    TILE_MODES,
    apply_radiometry,
    find_white_reference,
    resolve_radiometry,
    tile_reflectance,
    white_gain,
)
from spectralquadnet.data.prep.segmentation import (
    MORPHOMETRIC_NAMES,
    dark_correct,
    dark_frame,
    load_hsi,
    morphometrics,
    preprocess_raw,
    segment,
)
from spectralquadnet.data.prep.white_tile import (
    SOURCE_OWN,
    SOURCE_SESSION_GAIN,
    SOURCE_SESSION_SHAPE,
    SOURCE_UNRESOLVED,
    TileMeasurement,
    WhiteReferences,
    detect_white_tile,
    resolve_white_references,
)

#: P-3 / T4-3. A resized pixel is foreground when the resized mask says at
#: least half of it was covered. ``INTER_AREA`` makes that mask the exact fill
#: fraction alpha_p, so the threshold is a statement about coverage rather than
#: a tuned constant — and it replaces the ``> 1e-5`` band-sum test every masked
#: operator downstream currently re-derives (§2.6, FE-2).
FILL_THRESHOLD: float = 0.5


def pad_to_square(p: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Centre-pad an ``(H, W, C)`` patch with zeros to a square ``(S, S, C)``."""
    h, w, c = p.shape
    s = max(h, w)
    out = np.zeros((s, s, c), dtype=p.dtype)

    y = (s - h) // 2
    x = (s - w) // 2
    out[y : y + h, x : x + w] = p
    return out


def resize_patch(p: npt.NDArray[Any], patch_size: int) -> npt.NDArray[np.float32]:
    """Resize a square patch to ``(patch_size, patch_size)``, band by band.

    Bands are resized independently rather than as one multi-channel image
    because OpenCV's area interpolation is defined per 2-D plane.
    """
    out = np.zeros((patch_size, patch_size, p.shape[2]), dtype=np.float32)
    for i in range(p.shape[2]):
        out[:, :, i] = cv2.resize(
            p[:, :, i], (patch_size, patch_size), interpolation=cv2.INTER_AREA
        )
    return out


def extract_patch(
    cube: npt.NDArray[Any],
    labeled: npt.NDArray[Any],
    region: Any,
    patch_size: int,
    radiometry: str = "snv",
    eps: float = RADIOMETRY_EPS,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """One seed patch, resized in the P-3 order and normalised by P-2.

    The order is the whole point of T4-3, so it is written out step by step:

    1. crop the bounding box and build the region's binary mask;
    2. centre-pad **both** to a square;
    3. area-resize **both** to ``patch_size`` — the resized mask is the fill
       map :math:`\\alpha_p`, i.e. exactly the fraction of each output pixel
       that was seed;
    4. keep pixels with :math:`\\alpha_p >` :data:`FILL_THRESHOLD`, zero the
       rest, and divide the survivors by :math:`\\alpha_p`, which undoes the
       partial-coverage attenuation the old order baked in;
    5. apply the radiometric correction, computing the per-pixel gain it
       removes so brightness stays available as an explicit input.

    The old order masked first and resized second, so a boundary pixel that was
    30 % seed came out at 0.3x its true radiance and a pixel that was 0.001 %
    seed came out as a non-zero background — which is why the downstream
    ``> 1e-5`` foreground test was fragile (M-11, §2.6).

    Args:
        cube: ``(H, W, C)`` dark-corrected (and, under ``white``, already
            reflectance-converted) cube.
        labeled: ``(H, W)`` label image from :func:`segment`.
        region: The ``RegionProperties`` to extract.
        patch_size: Output edge, in pixels.
        radiometry: ``"snv"``, ``"white"`` or ``"none"`` — see
            :func:`~spectralquadnet.data.prep.radiometry.apply_radiometry`.
        eps: Denominator floor for the radiometric correction.

    Returns:
        ``(patch, alpha, gain)`` — ``(C, S, S)`` float32 patch, ``(S, S)``
        float32 fill map zeroed outside the kept foreground, and ``(2, S, S)``
        float32 per-pixel ``(mean, sd)`` along the band axis.
    """
    r0, c0, r1, c1 = region.bbox
    p = cube[r0:r1, c0:c1, :].astype(np.float32)
    mask = (labeled[r0:r1, c0:c1] == region.label).astype(np.float32)

    mask_sq = pad_to_square(mask[..., None])
    cube_sq = pad_to_square(p * mask[..., None])

    alpha = resize_patch(mask_sq, patch_size)[..., 0]
    resized = resize_patch(cube_sq, patch_size)

    keep = alpha > FILL_THRESHOLD
    resized[~keep] = 0.0
    resized[keep] /= alpha[keep][..., None]
    # The persisted map is zeroed outside `keep`, so `alpha > 0` and "this
    # pixel is foreground" are the same predicate. Keeping the sub-threshold
    # values would leave a mask that disagrees with the patch it describes on
    # exactly the pixels M-11 is about.
    alpha = np.where(keep, alpha, 0.0).astype(np.float32)

    normalised, gain = apply_radiometry(resized, keep, radiometry, eps)
    return (
        np.transpose(normalised, (2, 0, 1)).astype(np.float32),
        alpha,
        gain,
    )


def _resolve_cube_members(
    files: list[zipfile.ZipInfo], hdr_member: zipfile.ZipInfo
) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo, zipfile.ZipInfo] | None:
    """The ``(data, black_hdr, black_data)`` members a cube needs, or ``None``.

    Extracted from the two passes verbatim; both resolved the same four members
    with the same ``startswith``/``endswith`` rules, and a third copy would
    have arrived with T4-2's white-reference lookup.
    """
    stem = hdr_member.filename[:-4]
    data_member = next(
        (m for m in files if m.filename.startswith(stem) and not m.filename.endswith(".hdr")),
        None,
    )
    black_hdr = next((m for m in files if m.filename.lower().endswith("black.hdr")), None)
    if data_member is None or black_hdr is None:
        return None

    black_stem = black_hdr.filename[:-4]
    black_data = next(
        (m for m in files if m.filename.startswith(black_stem) and not m.filename.endswith(".hdr")),
        None,
    )
    if black_data is None:
        return None
    return data_member, black_hdr, black_data


def _materialise(zf: zipfile.ZipFile, members: list[zipfile.ZipInfo], tmp: Path) -> None:
    """Copy ``members`` out of the archive into ``tmp``, preserving their paths."""
    for m in members:
        out = tmp / m.filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _read_wavelengths(zf: zipfile.ZipFile) -> npt.NDArray[np.float64]:
    """The archive's ``wavelengths.csv``, as a float array in nm.

    Raises:
        RuntimeError: The archive has no ``wavelengths.csv``.
    """
    for m in zf.infolist():
        if m.filename.endswith("wavelengths.csv"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                with zf.open(m) as src, open(tmp / "wl.csv", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                wl_df = pd.read_csv(tmp / "wl.csv")
                return np.asarray(wl_df["Wavelength (nm)"].values, dtype=np.float64)
    raise RuntimeError("wavelengths.csv not found")


def _index_cubes(zf: zipfile.ZipFile) -> pd.DataFrame:
    """One row per seed cube: session, variety, member, label, scan key/id, session id."""
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
        cubes.append((session, variety, m))

    df = pd.DataFrame(cubes, columns=["session", "variety", "member"])
    df["label"] = pd.factorize(df["variety"])[0]

    # P-1 / T4-1. One `scan_id` per physical cube — the `<session>/<variety>-<n>`
    # key §3.1 names. Cube granularity, not `(session, variety)`: the -01/-02
    # pair of a variety is two separate acquisitions, and collapsing them would
    # throw away the only group boundary this dataset has for the 73 of 90
    # classes whose two cubes share a session (0-H).
    df["scan_key"] = df["session"] + "/" + df["member"].map(lambda m: Path(m.filename).stem)
    df["scan_id"] = np.arange(len(df), dtype=np.int64)
    df["session_id"] = pd.factorize(df["session"])[0]
    return df


def _load_scene(
    zf: zipfile.ZipFile, row: Any, tmp: Path
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
    """``(raw, dark)`` for one cube, **uncropped** and not dark-corrected, or ``None``."""
    files = [m for m in zf.infolist() if row.session in m.filename]
    resolved = _resolve_cube_members(files, row.member)
    if resolved is None:
        return None
    data_member, black_hdr, black_data = resolved
    _materialise(zf, [row.member, data_member, black_hdr, black_data], tmp)
    return load_hsi(tmp / row.member.filename), load_hsi(tmp / black_hdr.filename)


def _measure_tile(
    zf: zipfile.ZipFile, row: Any, wl: npt.NDArray[Any], tile_cfg: TileConfig, tmp: Path
) -> tuple[TileMeasurement, int]:
    """Measure one scan's white tile and count its seeds, in one read of the scene."""
    scene = _load_scene(zf, row, tmp)
    if scene is None:
        return (
            TileMeasurement(
                found=False,
                reason="cube members missing from the archive",
                n_bands=len(wl),
                spectrum=np.full(len(wl), np.nan),
                saturated_frac=np.full(len(wl), np.nan),
            ),
            0,
        )
    raw, dark = scene
    measurement = detect_white_tile(raw, dark_frame(dark), wl, tile_cfg)
    _, regs = segment(dark_correct(raw, dark)[:SEED_ROWS], wl)
    return measurement, len(regs)


def write_tile_qc(
    cfg: PrepConfig,
    df: pd.DataFrame,
    measurements: list[TileMeasurement],
    refs: WhiteReferences | None,
    wl: npt.NDArray[Any],
    path: Path | None = None,
) -> Path:
    """Write the per-scan tile QC table (and, with ``refs``, the spectra archive).

    Written **before** any patch is extracted, so a detection problem is visible
    — and fixable — before hours of extraction are spent on top of it.
    """
    rows = []
    for row, m in zip(df.itertuples(), measurements, strict=True):
        entry: dict[str, Any] = {
            "scan_id": int(row.scan_id),
            "scan_key": row.scan_key,
            "session": row.session,
            "session_id": int(row.session_id),
            **m.as_row(),
        }
        if refs is not None:
            src = refs.source[int(row.scan_id)]
            entry["n_bands_own"] = int((src == SOURCE_OWN).sum())
            entry["n_bands_session_filled"] = int(
                ((src == SOURCE_SESSION_SHAPE) | (src == SOURCE_SESSION_GAIN)).sum()
            )
            entry["n_bands_unresolved"] = int((src == SOURCE_UNRESOLVED).sum())
        rows.append(entry)
    out = path or cfg.white_tiles_path
    pd.DataFrame(rows).to_csv(out, index=False)
    if refs is not None:
        np.savez_compressed(
            cfg.white_spectra_path,
            scan_id=df["scan_id"].to_numpy(),
            session_id=df["session_id"].to_numpy(),
            wavelengths=np.asarray(wl, dtype=np.float64),
            measured=np.stack([m.spectrum for m in measurements]),
            saturated_frac=np.stack([m.saturated_frac for m in measurements]),
            resolved=refs.spectra,
            source=refs.source,
        )
    return out


def _open_stream(path: Path, shape: tuple[int, ...], dtype: Any) -> npt.NDArray[Any]:
    """A disk-backed output array, written in place and renamed into place at the end.

    The patch cube is ``N × 256 × 64 × 64`` float32 — 36 GB for this archive —
    so it cannot be allocated in RAM on a 16 GB machine; this writes it straight
    into a ``.partial`` file the size of the final array. The ``.partial``
    suffix means an interrupted run never leaves a file named ``patches.npy``.
    """
    stream: npt.NDArray[Any] = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        _partial(path), mode="w+", dtype=dtype, shape=shape
    )
    return stream


def _partial(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def _finalise_stream(stream: npt.NDArray[Any], path: Path, n_rows: int) -> None:
    """Truncate a streamed array to ``n_rows`` if needed, flush it, and move it into place.

    The trim is done **in place** (:func:`_truncate_npy_rows`) rather than by
    copying the kept rows into a new file: a copy needs a second 36 GB of free
    disk at the moment the first file is fullest.
    """
    partial = _partial(path)
    n_total = stream.shape[0]
    stream.flush()  # type: ignore[attr-defined]
    del stream
    if n_rows != n_total:
        _truncate_npy_rows(partial, n_rows)
    partial.replace(path)


def _truncate_npy_rows(path: Path, n_rows: int) -> None:
    """Shrink a C-order ``.npy`` file to its first ``n_rows`` rows without copying any data.

    The header is rewritten with the new shape at its original length — fewer
    rows never needs more digits, and the reader accepts any space padding
    before the header's closing newline — and the file is truncated after the
    last kept row.
    """
    fmt = np.lib.format
    with open(path, "r+b") as fh:
        version = fmt.read_magic(fh)  # type: ignore[no-untyped-call]
        read_header = fmt.read_array_header_1_0 if version == (1, 0) else fmt.read_array_header_2_0
        shape, fortran_order, dtype = read_header(fh)  # type: ignore[no-untyped-call]
        if fortran_order or not 0 <= n_rows <= shape[0]:
            raise ValueError(f"cannot truncate {path} (shape {shape}) to {n_rows} rows")
        offset = fh.tell()
        header_len = offset - (10 if version == (1, 0) else 12)
        new_shape = (n_rows, *shape[1:])
        text = repr(
            {
                "descr": fmt.dtype_to_descr(dtype),  # type: ignore[no-untyped-call]
                "fortran_order": False,
                "shape": new_shape,
            }
        )
        if len(text) + 1 > header_len:
            raise ValueError(f"new header for {path} does not fit in {header_len} bytes")
        fh.seek(offset - header_len)
        fh.write((text.ljust(header_len - 1) + "\n").encode("latin1"))
        fh.truncate(offset + int(np.prod(new_shape, dtype=np.int64)) * dtype.itemsize)


def probe_white_tiles(cfg: PrepConfig | None = None, limit: int = 10) -> pd.DataFrame:
    """Measure the white tile in up to ``limit`` scans, spread across sessions; extract nothing.

    The dry run for the ``tile`` modes: it reads real scenes, runs exactly the
    detector the extraction will run, and writes ``white_tiles_probe.csv`` so the
    thresholds in :class:`~spectralquadnet.data.prep.config.TileConfig` can be
    checked against real tiles before a full extraction is started.
    """
    cfg = cfg or PrepConfig()
    cfg.ensure_root()
    download(cfg)
    with zipfile.ZipFile(cfg.zip_file, "r") as zf:
        wl = _read_wavelengths(zf)
        df = _index_cubes(zf)
        # Round-robin over sessions, so a small probe sees every session's lamp.
        order = (
            df.assign(_rank=df.groupby("session_id").cumcount())
            .sort_values(["_rank", "session_id"])
            .head(max(1, int(limit)))
        )
        measurements = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for row in tqdm(order.itertuples(), total=len(order), desc="probing tiles"):
                m, _ = _measure_tile(zf, row, wl, cfg.tile, tmp)
                measurements.append(m)
                shutil.rmtree(tmp)
                tmp.mkdir()
    probe = order.drop(columns="_rank").reset_index(drop=True)
    out = write_tile_qc(cfg, probe, measurements, None, wl, cfg.root / "white_tiles_probe.csv")
    table = pd.read_csv(out)
    print(table.to_string(index=False))
    print(f"\nProbe → {out}  ({int(table['found'].sum())}/{len(table)} tiles found)")
    return table


def build_patch_dataset(cfg: PrepConfig | None = None) -> None:
    """Download, segment and extract fixed-size patches, saving the full data contract.

    Downloads the archive if needed, then makes two passes over every cube
    in it: pass 1 segments each cube to count the total number of seed
    patches (so the output arrays can be allocated exactly once), and pass 2
    re-segments and writes each patch through :func:`extract_patch`. Class
    labels are factorised from each cube's variety name.

    Under the ``tile`` modes pass 1 also finds and measures every scene's
    in-scene white tile, resolves one white spectrum per scan
    (:func:`~spectralquadnet.data.prep.white_tile.resolve_white_references`) and
    writes the QC table **before** pass 2 — refusing to start pass 2 if any
    scan has a band no measurement can supply. Pass 2 then segments on the
    dark-corrected **radiance**, exactly as the other modes do, and extracts
    from the reflectance cube. Segmenting on radiance is deliberate: the seed
    regions, and therefore the row order, labels, groups, masks and morphology,
    come out identical to an SNV extraction of the same archive, so the two
    datasets are row-aligned and can be compared kernel for kernel.

    Writes seven artifacts, all row-aligned on the patch index except the last:

    ==================== ====================== ====================================
    File                 Shape                  Item
    ==================== ====================== ====================================
    ``patches.npy``      ``(N, C, S, S)``       the patches themselves
    ``labels.npy``       ``(N,)``               class index
    ``groups.npy``       ``(N,)``               P-1 — cube-level ``scan_id``
    ``masks.npy``        ``(N, S, S)`` fp16     P-3 — the fill map alpha
    ``gain.npy``         ``(N, 2, S, S)``       P-2 — per-pixel (mean, sd)
    ``morphology.npy``   ``(N, 8)``             P-4 — shape descriptors
    ``scan_table.csv``   one row per cube       P-1 — what each ``scan_id`` is
    ==================== ====================== ====================================

    plus ``radiometry.json`` and ``wavelengths.csv`` (the cube's band axis)
    always, and ``white_tiles.csv`` / ``white_spectra.npz`` under the ``tile``
    modes. With :attr:`TileConfig.drop_unresolved_bands`, a band no tile in some
    scan's session could measure is dropped from **every** scan instead of
    refusing, so ``C`` is the kept count; ``radiometry.json`` lists the dropped
    bands and ``wavelengths.csv`` carries the kept ones' instrument indices.

    Args:
        cfg: Prep configuration; a default :class:`PrepConfig` is used if
            omitted.

    Raises:
        FileExistsError: ``patches.npy`` already exists in ``cfg.root`` and
            ``cfg.overwrite`` is off.
        RuntimeError: Under a ``tile`` mode, some scan has a band whose white
            level no measurement in its session could supply, and
            ``drop_unresolved_bands`` is off.
    """
    cfg = cfg or PrepConfig()
    cfg.ensure_root()
    if cfg.patches_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"{cfg.patches_path} already exists. A rebuild is hours of work and a changed "
            "radiometry is a new dataset: write it to a new --root (pointing --archive at the "
            "existing zip), or pass --overwrite to replace this one."
        )

    download(cfg)
    zf = zipfile.ZipFile(cfg.zip_file, "r")
    wl = _read_wavelengths(zf)

    # --------------------------------------------------------
    # P-2 / T4-2 — decide the radiometry, once, and say so
    # --------------------------------------------------------
    white_ref = find_white_reference(m.filename for m in zf.infolist())
    radiometry, why = resolve_radiometry(cfg.radiometry, white_ref)
    use_tile = radiometry in TILE_MODES
    print(f"Radiometry: {radiometry}  ({why})")

    df = _index_cubes(zf)
    print("Cubes:", len(df))
    print("Classes:", df.label.nunique())
    print("Scans:", df.scan_id.nunique(), " Sessions:", df.session_id.nunique())

    # ========================================================
    # PASS 1 — COUNT PATCHES (and, under `tile`, measure every tile)
    # ========================================================

    print("\nPass 1 — Counting patches" + (" and measuring white tiles..." if use_tile else "..."))
    total_patches = 0
    measurements: list[TileMeasurement] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for row in tqdm(df.itertuples(), total=len(df)):
            try:
                if use_tile:
                    m, n_seeds = _measure_tile(zf, row, wl, cfg.tile, tmp)
                    measurements.append(m)
                    total_patches += n_seeds
                else:
                    files = [m for m in zf.infolist() if row.session in m.filename]
                    resolved = _resolve_cube_members(files, row.member)
                    if resolved is None:
                        continue
                    data_member, black_hdr, black_data = resolved
                    _materialise(zf, [row.member, data_member, black_hdr, black_data], tmp)
                    cube = preprocess_raw(tmp / row.member.filename, tmp / black_hdr.filename)
                    _, regs = segment(cube, wl)
                    total_patches += len(regs)

                shutil.rmtree(tmp)
                tmp.mkdir()

            except Exception:
                print("FAIL:", row.member.filename)
                print(traceback.format_exc())
                if use_tile and len(measurements) < row.Index + 1:
                    measurements.append(
                        TileMeasurement(
                            found=False,
                            reason="scene failed to load",
                            n_bands=len(wl),
                            spectrum=np.full(len(wl), np.nan),
                            saturated_frac=np.full(len(wl), np.nan),
                        )
                    )

    print("Total patches:", total_patches)

    refs: WhiteReferences | None = None
    # Instrument bands written to the cube. Every band, unless a `tile` mode drops
    # the ones no tile could measure — from every scan, so the axis is common.
    keep = np.arange(len(wl))
    dropped = np.zeros(0, dtype=np.int64)
    if use_tile:
        refs = resolve_white_references(measurements, df["session_id"].tolist(), len(wl))
        qc = write_tile_qc(cfg, df, measurements, refs, wl)
        n_found = sum(m.found for m in measurements)
        print(f"White tiles: found in {n_found}/{len(df)} scans; values {refs.counts()}  → {qc}")
        if not refs.ok and cfg.tile.drop_unresolved_bands:
            dropped = np.unique(np.concatenate([np.asarray(b) for b in refs.unresolved.values()]))
            keep = np.setdiff1d(keep, dropped)
            if keep.size == 0:
                raise RuntimeError(f"every band is unresolved in some scan; see {qc}")
            print(
                f"Dropping {dropped.size} band(s) unresolved in {len(refs.unresolved)} scan(s), "
                f"{wl[dropped].min():.1f}-{wl[dropped].max():.1f} nm, from every scan: "
                f"{keep.size} of {len(wl)} bands kept  → {cfg.wavelengths_path}"
            )
        elif not refs.ok:
            bad = ", ".join(
                f"{df.iloc[i].scan_key} ({len(b)} bands)"
                for i, b in list(refs.unresolved.items())[:8]
            )
            raise RuntimeError(
                f"{len(refs.unresolved)} scan(s) have bands with no measurable white level — "
                f"{bad}{' …' if len(refs.unresolved) > 8 else ''}. No patch was written. "
                f"Inspect {qc} and retune PrepConfig.tile (see --probe-tiles), or drop "
                "those bands from every scan with --tile-drop-unresolved."
            )

    # ========================================================
    # PASS 2 — ALLOCATE EXACT MEMORY
    # ========================================================

    S = cfg.patch_size
    n_bands = keep.size if dropped.size else cfg.num_bands
    # Streamed to disk: at N × 256 × 64 × 64 float32 this is 36 GB.
    X = _open_stream(cfg.patches_path, (total_patches, n_bands, S, S), np.float32)
    y = np.zeros((total_patches,), dtype=np.int64)
    groups = np.full((total_patches,), -1, dtype=np.int64)  # P-1
    masks = np.zeros((total_patches, S, S), dtype=np.float16)  # P-3
    gains = np.zeros((total_patches, 2, S, S), dtype=np.float32)  # P-2
    morph = np.zeros((total_patches, len(MORPHOMETRIC_NAMES)), dtype=np.float32)  # P-4
    n_patches_per_scan = np.zeros((len(df),), dtype=np.int64)

    # ========================================================
    # PASS 3 — WRITE PATCHES
    # ========================================================

    print("\nPass 2 — Writing patches...")
    patch_index = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for row in tqdm(df.itertuples(), total=len(df)):
            try:
                if use_tile:
                    assert refs is not None
                    scene = _load_scene(zf, row, tmp)
                    if scene is None:
                        continue
                    raw, dark = scene
                    radiance = dark_correct(raw, dark)[:SEED_ROWS]
                    del raw
                    # Segment on radiance, extract from reflectance: see the docstring.
                    labeled, regs = segment(radiance, wl)
                    white = refs.spectra[row.scan_id]
                    if dropped.size:
                        radiance, white = radiance[..., keep], white[keep]
                    cube = tile_reflectance(radiance, white, cfg.tile.reference_reflectance)
                else:
                    files = [m for m in zf.infolist() if row.session in m.filename]
                    resolved = _resolve_cube_members(files, row.member)
                    if resolved is None:
                        continue
                    data_member, black_hdr, black_data = resolved

                    to_copy = [row.member, data_member, black_hdr, black_data]
                    white_members: list[zipfile.ZipInfo] = []
                    if radiometry == "white":
                        white_members = [
                            m
                            for m in files
                            if m.filename == white_ref or m.filename.startswith(str(white_ref)[:-4])
                        ]
                        to_copy += white_members
                    _materialise(zf, to_copy, tmp)

                    cube = preprocess_raw(tmp / row.member.filename, tmp / black_hdr.filename)

                    if radiometry == "white":
                        # `cube` is already `R - Dbar`, so only the division is left.
                        white = load_hsi(tmp / str(white_ref))
                        dark = load_hsi(tmp / black_hdr.filename)
                        cube = np.clip(cube / white_gain(white, dark), 0.0, None).astype(np.float32)

                    labeled, regs = segment(cube, wl)

                for r in regs:
                    patch, alpha, gain = extract_patch(cube, labeled, r, S, radiometry)

                    X[patch_index] = patch
                    masks[patch_index] = alpha.astype(np.float16)
                    gains[patch_index] = gain
                    morph[patch_index] = morphometrics(r)
                    y[patch_index] = row.label
                    groups[patch_index] = row.scan_id
                    patch_index += 1

                n_patches_per_scan[row.scan_id] = len(regs)

                shutil.rmtree(tmp)
                tmp.mkdir()

            except Exception:
                print("FAIL:", row.member.filename)
                print(traceback.format_exc())

    # A cube that pass 1 counted and pass 2 failed on would leave trailing
    # all-zero rows with `group == -1`. Those are not patches, and a -1 group
    # would silently become its own split under P-1, so they are dropped rather
    # than shipped.
    if patch_index != total_patches:
        print(f"\nWARNING: wrote {patch_index} of {total_patches} counted patches; truncating.")
        y = y[:patch_index]
        groups, masks = groups[:patch_index], masks[:patch_index]
        gains, morph = gains[:patch_index], morph[:patch_index]

    scan_table = pd.DataFrame(
        {
            "scan_id": df["scan_id"],
            "scan_key": df["scan_key"],
            "session": df["session"],
            "session_id": df["session_id"],
            "variety": df["variety"],
            "label": df["label"],
            "member": df["member"].map(lambda m: m.filename),
            "n_patches": n_patches_per_scan,
        }
    )

    x_shape = (patch_index, *X.shape[1:])
    _finalise_stream(X, cfg.patches_path, patch_index)
    np.save(cfg.labels_path, y)
    np.save(cfg.groups_path, groups)
    np.save(cfg.masks_path, masks)
    np.save(cfg.gain_path, gains)
    np.save(cfg.morphology_path, morph)
    scan_table.to_csv(cfg.scan_table_path, index=False)
    # The cube's own band axis, in the archive's format: `index` is the 1-based
    # instrument band, so a dropped band shows as a jump in it.
    wl_lines = ["index,Wavelength (nm)"]
    wl_lines += [f"{int(b) + 1},{float(wl[b]):.6f}" for b in keep]
    cfg.wavelengths_path.write_text("\n".join(wl_lines) + "\n")
    cfg.radiometry_log_path.write_text(
        json.dumps(
            {
                "requested": cfg.radiometry,
                "applied": radiometry,
                "reason": why,
                "tile": cfg.tile.as_dict() if use_tile else None,
                "white_reference_sources": refs.counts() if refs is not None else None,
                "tiles_found": (int(sum(m.found for m in measurements)) if use_tile else None),
                "bands": {
                    "n_instrument": int(len(wl)),
                    "n_kept": int(keep.size),
                    # 0-based instrument band indices, the convention of band-index files.
                    "dropped_instrument_index": [int(b) for b in dropped],
                    "dropped_nm": [round(float(wl[b]), 6) for b in dropped],
                    "dropped_reason": (
                        "no tile in the scan's session measured a white level "
                        "(saturated, or no tile found)"
                        if dropped.size
                        else None
                    ),
                },
                "n_scans": int(len(df)),
                "n_patches": int(patch_index),
                "archive": str(cfg.zip_file),
            },
            indent=2,
        )
    )

    print("\nSaved:")
    print("patches   :", x_shape)
    print("labels    :", y.shape)
    print("groups    :", groups.shape, f"({len(np.unique(groups))} scans)")
    print("masks     :", masks.shape)
    print("gain      :", gains.shape)
    print("morphology:", morph.shape, f"({', '.join(MORPHOMETRIC_NAMES)})")
    print("scan table:", cfg.scan_table_path)
    print("wavelength:", cfg.wavelengths_path, f"({keep.size} bands)")
    print("radiometry:", cfg.radiometry_log_path)
