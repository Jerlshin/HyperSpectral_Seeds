"""Three-pass patch extraction: count → allocate → write.

The CLI entry point is ``scripts/prepare_dataset.py``.

The pass-1/pass-2 duplication (each walks the same zip archive and reruns
segmentation) is deliberate: the counting pass exists so pass 2 can allocate
the exact ``(N, num_bands, patch_size, patch_size)`` float32 array up front,
avoiding either a list-of-arrays intermediate or a resizable-array
implementation for a multi-GB output.
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
from spectralquadnet.data.prep.segmentation import preprocess_raw, segment


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


def build_patch_dataset(cfg: PrepConfig | None = None) -> None:
    """Download, segment and extract fixed-size patches, saving ``patches.npy``/``labels.npy``.

    Downloads the archive if needed, then makes two passes over every cube
    in it: pass 1 segments each cube to count the total number of seed
    patches (so the output array can be allocated exactly once), and pass 2
    re-segments and writes each patch, centre-padded to square and resized
    to ``cfg.patch_size``. Class labels are factorised from each cube's
    variety name.

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

    print("Cubes:", len(df))
    print("Classes:", df.label.nunique())

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

                hdr_member = row.member
                stem = hdr_member.filename[:-4]

                data_member = next(
                    (
                        m
                        for m in files
                        if m.filename.startswith(stem) and not m.filename.endswith(".hdr")
                    ),
                    None,
                )
                black_hdr = next(
                    (m for m in files if m.filename.lower().endswith("black.hdr")), None
                )

                if data_member is None or black_hdr is None:
                    continue

                black_stem = black_hdr.filename[:-4]
                black_data = next(
                    (
                        m
                        for m in files
                        if m.filename.startswith(black_stem) and not m.filename.endswith(".hdr")
                    ),
                    None,
                )

                if black_data is None:
                    continue

                for m in [hdr_member, data_member, black_hdr, black_data]:
                    out = tmp / m.filename
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                cube = preprocess_raw(tmp / hdr_member.filename, tmp / black_hdr.filename)

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

    X = np.zeros((total_patches, cfg.num_bands, cfg.patch_size, cfg.patch_size), dtype=np.float32)
    y = np.zeros((total_patches,), dtype=np.int64)

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

                hdr_member = row.member
                stem = hdr_member.filename[:-4]

                data_member = next(
                    (
                        m
                        for m in files
                        if m.filename.startswith(stem) and not m.filename.endswith(".hdr")
                    ),
                    None,
                )
                black_hdr = next(
                    (m for m in files if m.filename.lower().endswith("black.hdr")), None
                )

                if data_member is None or black_hdr is None:
                    continue

                black_stem = black_hdr.filename[:-4]
                black_data = next(
                    (
                        m
                        for m in files
                        if m.filename.startswith(black_stem) and not m.filename.endswith(".hdr")
                    ),
                    None,
                )

                if black_data is None:
                    continue

                for m in [hdr_member, data_member, black_hdr, black_data]:
                    out = tmp / m.filename
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                cube = preprocess_raw(tmp / hdr_member.filename, tmp / black_hdr.filename)

                labeled, regs = segment(cube, wl)

                for r in regs:
                    r0, c0, r1, c1 = r.bbox
                    p = cube[r0:r1, c0:c1, :].copy()

                    mask = labeled[r0:r1, c0:c1] == r.label
                    p *= mask[..., None]

                    p = pad_to_square(p)
                    p = resize_patch(p, cfg.patch_size)
                    p = np.transpose(p, (2, 0, 1))

                    X[patch_index] = p
                    y[patch_index] = row.label
                    patch_index += 1

                shutil.rmtree(tmp)
                tmp.mkdir()

            except Exception:
                print("FAIL:", row.member.filename)
                print(traceback.format_exc())

    np.save(cfg.patches_path, X)
    np.save(cfg.labels_path, y)

    print("\nSaved:")
    print("patches:", X.shape)
    print("labels :", y.shape)
