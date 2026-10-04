#!/usr/bin/env python
"""Build validated RGB crops and full-band HSI assets without changing legacy data.

A scan is an atomic resumable shard. Manifest publication happens only when every
scan passes count, morphology, mask, k32 parity, unique-grid and geometric gates.
Run from the repository with PYTHONPATH=src; see S20 for exact invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from spectralquadnet.data.prep.multimodal import read_envi, sha256, spectral_summary, write_json
from spectralquadnet.data.prep.patch_extraction import extract_patch
from spectralquadnet.data.prep.radiometry import tile_reflectance
from spectralquadnet.data.prep.rgb import (
    decode_rgb,
    foreground_crop,
    grid_cells,
    pair_grids,
    rgb_descriptors,
    segment_rgb,
)
from spectralquadnet.data.prep.segmentation import dark_frame, morphometrics, segment


def run(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="spectralquadnet")
    cv2.setNumThreads(1)
    root, old, out = args.data, args.reference, args.output
    out.mkdir(parents=True, exist_ok=True)
    shards = out / "scans"
    shards.mkdir(exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)
    scans = pd.read_csv(root / "scan_table.csv")
    n = int(scans.n_patches.sum())
    labels, groups = np.load(root / "labels.npy"), np.load(root / "groups.npy")
    morphology = np.load(root / "morphology.npy")
    masks = np.load(root / "masks.npy", mmap_mode="r")
    legacy = np.load(old / "patches.npy", mmap_mode="r")
    band32 = np.array(json.loads((old / "band_axis.json").read_text())["source_band_indices"])
    white = np.load(root / "white_spectra.npz")
    keep = pd.read_csv(root / "wavelengths.csv").iloc[:, 0].to_numpy().astype(int) - 1
    wl = white["wavelengths"]
    source_files = [
        root / x
        for x in [
            "scan_table.csv",
            "labels.npy",
            "groups.npy",
            "morphology.npy",
            "masks.npy",
            "white_spectra.npz",
            "wavelengths.csv",
        ]
    ]
    source_files += [
        Path(__file__),
        Path("src/spectralquadnet/data/prep/rgb.py"),
        Path("src/spectralquadnet/data/prep/multimodal.py"),
        Path("src/spectralquadnet/data/prep/segmentation.py"),
        Path("src/spectralquadnet/data/prep/patch_extraction.py"),
        Path("src/spectralquadnet/data/prep/radiometry.py"),
    ]
    config = {
        "schema": 1,
        "crop_size": 224,
        "orientation": "encoded sensor; ignore EXIF",
        "source_hashes": {str(p): sha256(p) for p in source_files},
        "archive_sha256": args.archive_sha256,
        "full_dense": args.dense,
        "reference_manifest_sha256": sha256(old / "MANIFEST.json"),
    }
    config_path = out / "build_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Build configuration changed; choose a new output directory")
    write_json(config_path, config)
    if (out / "MANIFEST.json").exists():
        raise FileExistsError("Completed dataset exists; validate/reuse it instead of overwriting")
    if sha256(root / "rice_hsi.zip") != args.archive_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    start = time.monotonic()
    darks = {}
    with zipfile.ZipFile(root / "rice_hsi.zip") as z:
        names = z.namelist()
        for row in scans.itertuples():
            sid = int(row.scan_id)
            done = shards / f"{sid:03d}.json"
            if done.exists():
                continue
            t = time.monotonic()
            idx = np.flatnonzero(groups == sid)
            if len(idx) != row.n_patches or not np.all(labels[idx] == row.label):
                raise ValueError("Legacy scan/label identity mismatch")
            folder = str(Path(row.member).parent)
            if folder not in darks:
                darks[folder] = dark_frame(read_envi(z, folder + "/black.hdr"))
            raw = read_envi(z, row.member)
            cube = np.maximum(raw[:600].astype(np.float32) - darks[folder], 0)
            del raw
            hlabels, hr = segment(cube, wl)
            if len(hr) != len(idx):
                raise ValueError(f"Scan {sid}: HSI component count changed")
            if not np.allclose(
                np.array([morphometrics(r) for r in hr]), morphology[idx], rtol=1e-6, atol=1e-5
            ):
                raise ValueError(f"Scan {sid}: HSI identity/morphology mismatch")
            candidates = [
                p
                for p in names
                if p.rsplit(".", 1)[0] == row.member[:-4] and p.lower().endswith(".jpg")
            ]
            if len(candidates) != 1:
                raise ValueError("RGB sibling is not unique")
            payload = z.read(candidates[0])
            image = decode_rgb(payload)
            rgb_labels, rr, threshold = segment_rgb(image)
            hc = np.array([r.centroid for r in hr])
            rc = np.array([r.centroid for r in rr])
            pairs, qc = pair_grids(hc, rc)
            grid = grid_cells(hc)
            widx = np.flatnonzero(white["scan_id"] == sid)
            if len(widx) != 1:
                raise ValueError("Missing white reference identity")
            refl = tile_reflectance(cube[..., keep], white["resolved"][int(widx[0]), keep])
            del cube
            features, tokens, occupancies, crops, rgbm, rgbf, dense = [], [], [], [], [], [], []
            records = []
            maximum = 0.0
            different = 0
            for j, (r, rid) in enumerate(zip(hr, pairs, strict=True)):
                i = int(idx[j])
                rgbregion = rr[rid]
                patch, mask, _ = extract_patch(refl, hlabels, r, 64, "none")
                if not np.array_equal(mask.astype(np.float16), masks[i]):
                    raise ValueError(f"Kernel {i}: legacy mask mismatch")
                actual = patch[band32].astype(np.float16)
                delta = np.abs(actual.astype(np.float32) - legacy[i].astype(np.float32))
                maximum = max(maximum, float(delta.max()))
                different += int(np.count_nonzero(delta))
                if not np.array_equal(actual, legacy[i]):
                    raise ValueError(f"Kernel {i}: k32 reconstruction differs, max {delta.max()}")
                a, b, c = spectral_summary(patch, mask)
                features.append(a)
                tokens.append(b)
                occupancies.append(c)
                crops.append(foreground_crop(image, rgb_labels, rgbregion))
                rgbm.append(morphometrics(rgbregion))
                rgbf.append(rgb_descriptors(image, rgb_labels, rgbregion))
                if args.dense:
                    dense.append(patch.astype(np.float16))
                gr, gc = map(int, grid[j])
                records.append(
                    {
                        "index": i,
                        "kernel_id": f"{row.scan_key}/r{gr + 1:02d}c{gc + 1:02d}",
                        "scan_id": sid,
                        "session_id": int(row.session_id),
                        "label": int(row.label),
                        "grid_row": gr,
                        "grid_col": gc,
                        "hsi_region_label": int(r.label),
                        "rgb_region_label": int(rgbregion.label),
                        "hsi_y": float(r.centroid[0]),
                        "hsi_x": float(r.centroid[1]),
                        "rgb_y": float(rgbregion.centroid[0]),
                        "rgb_x": float(rgbregion.centroid[1]),
                        "hsi_bbox": list(map(int, r.bbox)),
                        "rgb_bbox": list(map(int, rgbregion.bbox)),
                        "rgb_area": float(rgbregion.area),
                        "residual_hsi_px": qc["residuals"][j],
                    }
                )
            values = dict(
                indices=idx,
                rgb=np.array(crops),
                rgb_morphology=np.array(rgbm),
                rgb_descriptors=np.array(rgbf),
                hsi_summary=np.array(features),
                hsi_regions=np.array(tokens),
                occupancy=np.array(occupancies),
            )
            if args.dense:
                values["patches"] = np.array(dense)
            np.savez_compressed(
                shards / f"{sid:03d}_masks.npz",
                rgb=rgb_labels.astype(np.uint8),
                hsi=hlabels.astype(np.uint16),
            )
            temp = shards / f"{sid:03d}.tmp.npz"
            np.savez(temp, **values)
            temp.replace(shards / f"{sid:03d}.npz")
            # Every scan has a review overlay; labels are grid IDs, not class names.
            view = cv2.resize(image, (1224, 816))
            hview = (
                (refl.mean(2) / max(float(np.percentile(refl.mean(2), 99)), 1e-6) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            hview = cv2.cvtColor(cv2.resize(hview, (448, 800)), cv2.COLOR_GRAY2RGB)
            for rec in records:
                text = f"{rec['grid_row'] + 1},{rec['grid_col'] + 1}"
                cv2.putText(
                    view,
                    text,
                    (int(rec["rgb_x"] / 4), int(rec["rgb_y"] / 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 40, 40),
                    1,
                )
                cv2.putText(
                    hview,
                    text,
                    (int(rec["hsi_x"] * 448 / 336), int(rec["hsi_y"] * 800 / 600)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 40, 40),
                    1,
                )
            canvas = np.full((850, 1672, 3), 245, np.uint8)
            canvas[24:840, :1224] = view
            canvas[24:824, 1224:] = hview
            cv2.putText(
                canvas,
                f"Scan {sid}; RGB sensor orientation / HSI; grid row,column",
                (12, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            Image.fromarray(canvas).save(out / "overlays" / f"{sid:03d}.jpg", quality=88)
            missing = sorted(
                set((y, x) for y in range(8) for x in range(6)) - set(map(tuple, grid))
            )
            write_json(
                done,
                {
                    "scan_id": sid,
                    "records": records,
                    "qc": qc,
                    "rgb_threshold": threshold,
                    "rgb_member": candidates[0],
                    "rgb_sha256": hashlib.sha256(payload).hexdigest(),
                    "shard_sha256": sha256(shards / f"{sid:03d}.npz"),
                    "native_masks_sha256": sha256(shards / f"{sid:03d}_masks.npz"),
                    "missing_hsi_cells": [list(map(int, c)) for c in missing],
                    "k32_max_abs": maximum,
                    "k32_different_values": different,
                    "seconds": time.monotonic() - t,
                },
            )
            print(
                f"scan {sid:03d}: {len(idx)} paired; RMSE {qc['rmse_hsi_px']:.3f}px; {time.monotonic() - t:.1f}s",
                flush=True,
            )
            if args.limit and sid + 1 >= args.limit:
                return
    # Assemble atomically published arrays; interrupted assembly is safe to repeat.
    first = np.load(shards / "000.npz")
    arrays = {
        key: np.lib.format.open_memmap(
            out / (key + ".npy"),
            mode="w+",
            dtype=first[key].dtype,
            shape=(n, *first[key].shape[1:]),
        )
        for key in first.files
        if key != "indices"
    }
    records, qc_rows, exclusions = [], [], []
    for row in scans.itertuples():
        sid = int(row.scan_id)
        meta = json.loads((shards / f"{sid:03d}.json").read_text())
        if sha256(shards / f"{sid:03d}.npz") != meta["shard_sha256"]:
            raise ValueError("Corrupt scan shard")
        with np.load(shards / f"{sid:03d}.npz") as data:
            for key, array in arrays.items():
                array[data["indices"]] = data[key]
        records.extend(meta["records"])
        qc_rows.append(
            {k: v for k, v in meta.items() if k not in ("records", "qc", "missing_hsi_cells")}
            | {k: v for k, v in meta["qc"].items() if k != "residuals"}
        )
        exclusions.extend(
            {
                "scan_id": sid,
                "grid_row": c[0],
                "grid_col": c[1],
                "reason": "not retained by historical HSI segmentation; RGB seed present",
            }
            for c in meta["missing_hsi_cells"]
        )
    for array in arrays.values():
        array.flush()
    pd.DataFrame(records).sort_values("index").to_csv(out / "kernel_manifest.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(out / "scan_qc.csv", index=False)
    pd.DataFrame(exclusions).to_csv(out / "exclusions.csv", index=False)
    for name in [
        "labels.npy",
        "groups.npy",
        "morphology.npy",
        "masks.npy",
        "scan_table.csv",
        "wavelengths.csv",
        "white_spectra.npz",
        "radiometry.json",
    ]:
        shutil.copy2(root / name, out / name)
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(out.iterdir())
        if p.is_file() and p.suffix in (".npy", ".csv", ".json", ".npz")
    }
    write_json(
        out / "MANIFEST.json",
        {
            "schema": 1,
            "complete": True,
            "kernels": n,
            "scans": len(scans),
            "files": files,
            "config": config,
            "python": platform.python_version(),
            "seconds_this_invocation": time.monotonic() - start,
            "calibration": "historical resolved per-scan white; pooled source flags retained; strict probes exclude any non-own wavelength",
            "dense_reconstruction": args.dense,
            "pairing": "unique 8x6 grid + homography orientation/residual QC; no pixelwise registration",
        },
    )
    print(f"Complete: {out}; {n} paired kernels", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("dataset"))
    p.add_argument("--reference", type=Path, default=Path("dataset_u430k32"))
    p.add_argument("--output", type=Path, default=Path("dataset_rgb_hsi_v1"))
    p.add_argument("--archive-sha256", required=True)
    p.add_argument(
        "--dense",
        action="store_true",
        help="Also materialize full215 float16 patches (~14 GiB); needs twice that during assembly",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this scan count for a preflight; resume with identical arguments except limit",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
