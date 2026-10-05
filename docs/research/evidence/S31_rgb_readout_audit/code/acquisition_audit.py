"""S31 label-free RGB acquisition audit: per-scan background colour, kernel brightness and edge
sharpness by session. Reads the native JPEGs from the verified archive and the saved native
masks; no variety label is used (session ids describe acquisition, they are not predictors).
Output: evidence/S31_rgb_readout_audit/acquisition_audit.csv (+ session summary on stdout).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from spectralquadnet.data.prep.rgb import decode_rgb

DATA = Path("dataset_rgb_hsi_v3")
OUT = Path("docs/research/evidence/S31_rgb_readout_audit/acquisition_audit.csv")


def main() -> None:
    scans = pd.read_csv(DATA / "scan_table.csv")
    rows = []
    with zipfile.ZipFile("dataset/rice_hsi.zip") as archive:
        for sid in scans.scan_id:
            meta = json.loads((DATA / f"scans/{sid:03d}.json").read_text())
            image = decode_rgb(archive.read(meta["rgb_member"]))  # encoded sensor orientation
            with np.load(DATA / f"scans/{sid:03d}_masks.npz") as m:
                fg = m["rgb"] > 0
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            # Plate background: pixels far from any kernel inside the central plate region.
            far = cv2.dilate(fg.astype(np.uint8), np.ones((61, 61), np.uint8)) == 0
            h, w = fg.shape
            plate = np.zeros_like(fg)
            ys, xs = np.nonzero(fg)
            plate[ys.min():ys.max(), xs.min():xs.max()] = True
            bg = far & plate
            # Edge sharpness: mean gradient magnitude on the kernel boundary band.
            edge = (cv2.dilate(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) - cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8))) > 0
            gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            core = cv2.erode(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            rows.append({"scan_id": sid, "session_id": int(scans.session_id[sid]),
                         **{f"bg_{c}": float(image[..., i][bg].mean()) for i, c in enumerate("rgb")},
                         "bg_gray_sd": float(gray[bg].std()),
                         **{f"kernel_{c}": float(image[..., i][core].mean()) for i, c in enumerate("rgb")},
                         "edge_gradient": float(np.hypot(gx, gy)[edge].mean()),
                         "core_laplacian_var": float(lap[core].var())})
            print(sid, end=" ", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print()
    print(df.drop(columns="scan_id").groupby("session_id").median().round(2).to_string())


if __name__ == "__main__":
    main()
