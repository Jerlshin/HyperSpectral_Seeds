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

from spectralquadnet.data.prep.config import PrepConfig
from spectralquadnet.data.prep.download import download
from spectralquadnet.data.prep.radiometry import (
    RADIOMETRY_EPS,
    apply_radiometry,
    find_white_reference,
    resolve_radiometry,
    white_gain,
)
from spectralquadnet.data.prep.segmentation import (
    MORPHOMETRIC_NAMES,
    load_hsi,
    morphometrics,
    preprocess_raw,
    segment,
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


def build_patch_dataset(cfg: PrepConfig | None = None) -> None:
    """Download, segment and extract fixed-size patches, saving the full data contract.

    Downloads the archive if needed, then makes two passes over every cube
    in it: pass 1 segments each cube to count the total number of seed
    patches (so the output arrays can be allocated exactly once), and pass 2
    re-segments and writes each patch through :func:`extract_patch`. Class
    labels are factorised from each cube's variety name.

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

    Args:
        cfg: Prep configuration; a default :class:`PrepConfig` is used if
            omitted.
    """
    cfg = cfg or PrepConfig()
    cfg.ensure_root()

    download(cfg)
    zf = zipfile.ZipFile(cfg.zip_file, "r")

    # --------------------------------------------------------
    # Load wavelengths
    # --------------------------------------------------------
    wl = None
    for m in zf.infolist():
        if m.filename.endswith("wavelengths.csv"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                with zf.open(m) as src, open(tmp / "wl.csv", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                wl_df = pd.read_csv(tmp / "wl.csv")
                wl = wl_df["Wavelength (nm)"].values.astype(float)
            break

    if wl is None:
        raise RuntimeError("wavelengths.csv not found")

    # --------------------------------------------------------
    # P-2 / T4-2 — decide the radiometry, once, and say so
    # --------------------------------------------------------
    white_ref = find_white_reference(m.filename for m in zf.infolist())
    radiometry, why = resolve_radiometry(cfg.radiometry, white_ref)
    print(f"Radiometry: {radiometry}  ({why})")

    # --------------------------------------------------------
    # Index cubes
    # --------------------------------------------------------
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

    print("Cubes:", len(df))
    print("Classes:", df.label.nunique())
    print("Scans:", df.scan_id.nunique(), " Sessions:", df.session_id.nunique())

    # ========================================================
    # PASS 1 — COUNT PATCHES
    # ========================================================

    print("\nPass 1 — Counting patches...")
    total_patches = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for row in tqdm(df.itertuples(), total=len(df)):
            try:
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

    print("Total patches:", total_patches)

    # ========================================================
    # PASS 2 — ALLOCATE EXACT MEMORY
    # ========================================================

    S = cfg.patch_size
    X = np.zeros((total_patches, cfg.num_bands, S, S), dtype=np.float32)
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
        X, y = X[:patch_index], y[:patch_index]
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

    np.save(cfg.patches_path, X)
    np.save(cfg.labels_path, y)
    np.save(cfg.groups_path, groups)
    np.save(cfg.masks_path, masks)
    np.save(cfg.gain_path, gains)
    np.save(cfg.morphology_path, morph)
    scan_table.to_csv(cfg.scan_table_path, index=False)

    print("\nSaved:")
    print("patches   :", X.shape)
    print("labels    :", y.shape)
    print("groups    :", groups.shape, f"({len(np.unique(groups))} scans)")
    print("masks     :", masks.shape)
    print("gain      :", gains.shape)
    print("morphology:", morph.shape, f"({', '.join(MORPHOMETRIC_NAMES)})")
    print("scan table:", cfg.scan_table_path)
