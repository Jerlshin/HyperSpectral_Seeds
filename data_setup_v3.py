import zipfile
import shutil
import subprocess
import traceback
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes

import spectral
spectral.settings.envi_support_nonlowercase_params = True


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("./dataset").resolve()
ROOT.mkdir(exist_ok=True)

DATA_URL = (
    "https://zenodo.org/records/3241923/files/"
    "RGB%20and%20VIS-NIR%20HSI%20Data%20for%2090%20Rice%20Seed%20Varieties.zip?download=1"
)

ZIP_FILE     = ROOT / "rice_hsi.zip"
PATCHES_PATH = ROOT / "patches.npy"
LABELS_PATH  = ROOT / "labels.npy"

PATCH_SIZE = 64
NUM_BANDS  = 256


# ============================================================
# DOWNLOAD
# ============================================================

def download():
    if ZIP_FILE.exists():
        print("Zip exists. Skipping download.")
        return

    print("Downloading dataset with curl...")

    cmd = [
        "curl",
        "-L",
        "--fail",
        "-C", "-",
        "--progress-bar",
        "-o", str(ZIP_FILE),
        DATA_URL
    ]

    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("Download failed.")

    print("Download complete.")


# ============================================================
# HSI LOADING
# ============================================================

def load_hsi(hdr):
    from spectral.io import envi
    img  = envi.open(str(hdr))
    cube = np.asarray(img.load())
    return cube.astype(np.float32)


def preprocess_raw(hdr, dark_hdr):
    cube = load_hsi(hdr)
    dark = load_hsi(dark_hdr)

    dark_mean = dark.mean(axis=0, keepdims=True)
    cube = np.clip(cube - dark_mean, 0.0, None)

    return cube[:600]


# ============================================================
# SEGMENTATION
# ============================================================

def segment(cube, wl):
    vis_mask = (wl > 450) & (wl < 700)
    vis_img  = cube[:, :, vis_mask].mean(axis=2)

    t = threshold_otsu(vis_img)
    binary = vis_img > (0.4 * t)

    binary = binary_fill_holes(binary)
    binary = clear_border(binary)
    binary = remove_small_objects(binary, 150)

    labeled = label(binary)
    regions = regionprops(labeled)

    regions = [
        r for r in regions
        if 300 < r.area < 800
        and r.eccentricity > 0.6
        and r.solidity > 0.85
    ]

    regions.sort(key=lambda r: (r.centroid[0], r.centroid[1]))
    return labeled, regions


# ============================================================
# PATCH UTILITIES
# ============================================================

def pad_to_square(p):
    h, w, c = p.shape
    s = max(h, w)
    out = np.zeros((s, s, c), dtype=p.dtype)

    y = (s - h) // 2
    x = (s - w) // 2
    out[y:y+h, x:x+w] = p
    return out


def resize_patch(p):
    out = np.zeros((PATCH_SIZE, PATCH_SIZE, p.shape[2]), dtype=np.float32)
    for i in range(p.shape[2]):
        out[:, :, i] = cv2.resize(
            p[:, :, i],
            (PATCH_SIZE, PATCH_SIZE),
            interpolation=cv2.INTER_AREA
        )
    return out


# ============================================================
# MAIN
# ============================================================

def main():

    download()
    zf = zipfile.ZipFile(ZIP_FILE, "r")

    # --------------------------------------------------------
    # Load wavelengths
    # --------------------------------------------------------
    wl = None
    for m in zf.infolist():
        if m.filename.endswith("wavelengths.csv"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                with zf.open(m) as src, open(tmp/"wl.csv","wb") as dst:
                    shutil.copyfileobj(src,dst)
                wl_df = pd.read_csv(tmp/"wl.csv")
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

    df = pd.DataFrame(cubes, columns=["session","variety","member"])
    df["label"] = pd.factorize(df["variety"])[0]

    print("Cubes:", len(df))
    print("Classes:", df.label.nunique())

    # ========================================================
    # PASS 1 — COUNT PATCHES
    # ========================================================

    print("\nPass 1 — Counting patches...")
    total_patches = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for row in tqdm(df.itertuples(), total=len(df)):

            try:
                files = [m for m in zf.infolist()
                         if row.session in m.filename]

                hdr_member = row.member
                stem = hdr_member.filename[:-4]

                data_member = next(
                    (m for m in files
                     if m.filename.startswith(stem)
                     and not m.filename.endswith(".hdr")),
                    None
                )
                black_hdr = next(
                    (m for m in files
                     if m.filename.lower().endswith("black.hdr")),
                    None
                )

                if data_member is None or black_hdr is None:
                    continue

                black_stem = black_hdr.filename[:-4]
                black_data = next(
                    (m for m in files
                     if m.filename.startswith(black_stem)
                     and not m.filename.endswith(".hdr")),
                    None
                )

                if black_data is None:
                    continue

                for m in [hdr_member, data_member, black_hdr, black_data]:
                    out = tmp / m.filename
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                cube = preprocess_raw(tmp / hdr_member.filename,
                                      tmp / black_hdr.filename)

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

    X = np.zeros((total_patches, NUM_BANDS, PATCH_SIZE, PATCH_SIZE),
                 dtype=np.float32)
    y = np.zeros((total_patches,), dtype=np.int64)

    # ========================================================
    # PASS 3 — WRITE PATCHES
    # ========================================================

    print("\nPass 2 — Writing patches...")
    patch_index = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for row in tqdm(df.itertuples(), total=len(df)):

            try:
                files = [m for m in zf.infolist()
                         if row.session in m.filename]

                hdr_member = row.member
                stem = hdr_member.filename[:-4]

                data_member = next(
                    (m for m in files
                     if m.filename.startswith(stem)
                     and not m.filename.endswith(".hdr")),
                    None
                )
                black_hdr = next(
                    (m for m in files
                     if m.filename.lower().endswith("black.hdr")),
                    None
                )

                if data_member is None or black_hdr is None:
                    continue

                black_stem = black_hdr.filename[:-4]
                black_data = next(
                    (m for m in files
                     if m.filename.startswith(black_stem)
                     and not m.filename.endswith(".hdr")),
                    None
                )

                if black_data is None:
                    continue

                for m in [hdr_member, data_member, black_hdr, black_data]:
                    out = tmp / m.filename
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                cube = preprocess_raw(tmp / hdr_member.filename,
                                      tmp / black_hdr.filename)

                labeled, regs = segment(cube, wl)

                for r in regs:
                    r0,c0,r1,c1 = r.bbox
                    p = cube[r0:r1, c0:c1, :].copy()

                    mask = (labeled[r0:r1, c0:c1] == r.label)
                    p *= mask[...,None]

                    p = pad_to_square(p)
                    p = resize_patch(p)
                    p = np.transpose(p, (2,0,1))

                    X[patch_index] = p
                    y[patch_index] = row.label
                    patch_index += 1

                shutil.rmtree(tmp)
                tmp.mkdir()

            except Exception:
                print("FAIL:", row.member.filename)
                print(traceback.format_exc())

    np.save(PATCHES_PATH, X)
    np.save(LABELS_PATH,  y)

    print("\nSaved:")
    print("patches:", X.shape)
    print("labels :", y.shape)


if __name__ == "__main__":
    main()