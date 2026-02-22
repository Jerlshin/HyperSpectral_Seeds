#!/usr/bin/env python3
"""
Prepare Rice HSI Dataset (Streaming, RAW-Safe)
=============================================

Builds:

    patches.npy : (N, 256, 64, 64)
    labels.npy  : (N,)

No full extraction. Disk safe.
"""

import os
import sys
import zipfile
import shutil
import logging
import subprocess
import traceback
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes


import warnings
import spectral

warnings.filterwarnings(
    "ignore",
    message="Unable to parse \"wavelength\" field"
)

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

ZIP_FILE = ROOT / "rice_hsi.zip"

PATCHES_PATH = ROOT / "patches.npy"
LABELS_PATH = ROOT / "labels.npy"

PATCH_SIZE = 64
NUM_BANDS = 256

LOG_FILE = ROOT / "prepare_dataset.log"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger("builder")


# ============================================================
# DOWNLOAD
# ============================================================

def download():

    if ZIP_FILE.exists():
        log.info("Zip exists. Skip download.")
        return

    log.info("Downloading dataset...")

    if shutil.which("wget"):
        subprocess.run(["wget", "-O", ZIP_FILE, DATA_URL], check=True)
        return

    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "-o", ZIP_FILE, DATA_URL], check=True)
        return

    import urllib.request
    urllib.request.urlretrieve(DATA_URL, ZIP_FILE)


# ============================================================
# ZIP
# ============================================================

def open_zip():

    zf = zipfile.ZipFile(ZIP_FILE, "r")

    sessions = {}

    for m in zf.infolist():

        if "Data-VIS" not in m.filename:
            continue

        parts = Path(m.filename).parts

        if len(parts) < 2:
            continue

        session = parts[1]

        sessions.setdefault(session, []).append(m)

    log.info("Found %d sessions", len(sessions))

    return zf, sessions


def extract_pair(zf, hdr_member, tmpdir):

    """
    Extracts:
        *.hdr
        *.raw
    """

    base = hdr_member.filename[:-4]

    raw_name = base + ".raw"

    hdr_path = tmpdir / hdr_member.filename
    raw_path = tmpdir / raw_name

    hdr_path.parent.mkdir(parents=True, exist_ok=True)

    # HDR
    with zf.open(hdr_member) as src, open(hdr_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    # RAW
    raw_member = None

    for m in zf.infolist():
        if m.filename == raw_name:
            raw_member = m
            break

    if raw_member is None:
        raise FileNotFoundError(raw_name)

    with zf.open(raw_member) as src, open(raw_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    return hdr_path, raw_path


# ============================================================
# HSI
# ============================================================

def load_hsi(hdr):

    from spectral.io import envi

    img = envi.open(str(hdr))
    cube = np.asarray(img.load())

    return cube.astype(np.float32)


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(hdr, dark_hdr):

    cube = load_hsi(hdr)
    dark = load_hsi(dark_hdr)

    dark = dark.mean(0)

    cube = np.clip(cube - dark, 0, None)

    return cube[:600]


# ============================================================
# SEGMENT
# ============================================================

def segment(cube, wl):

    vis = (wl > 450) & (wl < 700)

    if vis.sum() == 0:
        raise RuntimeError("No VIS bands selected. Check wavelengths.")

    img = cube[:, :, vis].mean(2)

    if not np.isfinite(img).all():
        raise RuntimeError("Image contains NaNs after VIS projection.")

    t = threshold_otsu(img)

    mask = img > 0.4 * t

    mask = binary_fill_holes(mask)
    mask = clear_border(mask)
    mask = remove_small_objects(mask, 150)

    lab = label(mask)

    regs = regionprops(lab)

    regs = [
        r for r in regs
        if 300 < r.area < 800
        and r.eccentricity > 0.6
        and r.solidity > 0.85
    ]

    regs.sort(key=lambda r: (r.centroid[0], r.centroid[1]))

    return lab, regs

# ============================================================
# PATCH OPS
# ============================================================

def pad(p):

    h, w, c = p.shape
    s = max(h, w)

    out = np.zeros((s, s, c), p.dtype)

    y = (s - h) // 2
    x = (s - w) // 2

    out[y:y+h, x:x+w] = p

    return out


def resize(p):

    out = np.zeros((PATCH_SIZE, PATCH_SIZE, p.shape[2]), np.float32)

    for i in range(p.shape[2]):
        out[:, :, i] = cv2.resize(p[:, :, i],
                                  (PATCH_SIZE, PATCH_SIZE))

    return out


def snv(p):

    return (p - p.mean()) / (p.std() + 1e-6)


# ============================================================
# MAIN
# ============================================================

def main():

    download()

    zf, sessions = open_zip()

    # Load wavelengths
    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        for m in zf.infolist():
            if m.filename.endswith("wavelengths.csv"):
                with zf.open(m) as src, open(tmp/"wl.csv","wb") as dst:
                    shutil.copyfileobj(src,dst)
                wl_df = pd.read_csv(tmp/"wl.csv")

                if "Wavelength (nm)" not in wl_df.columns:
                    raise RuntimeError("wavelengths.csv format unexpected")

                wl = wl_df["Wavelength (nm)"].values.astype(float)

                break


    # Build cube list
    cubes = []

    for sess, files in sessions.items():

        for m in files:

            if m.filename.endswith(".hdr") \
               and not m.filename.lower().endswith("black.hdr"):

                name = Path(m.filename).stem
                variety = name.rsplit("-",1)[0]

                cubes.append((sess,variety,m))


    df = pd.DataFrame(cubes,
                      columns=["session","variety","member"])

    df["label"] = pd.factorize(df["variety"])[0]

    log.info("Cubes: %d | Classes: %d",
             len(df), df.label.nunique())


    # -------------------------------------------------------
    # Detect seeds
    # -------------------------------------------------------

    rows = []

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        for i,row in df.iterrows():

            try:

                files = sessions[row.session]

                # extract cube + dark
                for m in files:

                    if m.filename == row.member.filename \
                       or m.filename.lower().endswith("black.hdr"):

                        extract_pair(zf, m, tmp)


                hdr = tmp / row.member.filename
                dark = next(tmp.rglob("black.hdr"))

                cube = preprocess(hdr, dark)

                lab, regs = segment(cube, wl)

                for r in regs:

                    rows.append({
                        "hdr": row.member.filename,
                        "session": row.session,
                        "label": row.label,
                        "bbox": r.bbox,
                        "rid": r.label
                    })

                log.info("[%d/%d] %d seeds",
                         i+1,len(df),len(regs))

                shutil.rmtree(tmp)
                tmp.mkdir()

            except Exception:

                log.error("FAIL %s",row.member.filename)
                log.error(traceback.format_exc())


    seed_df = pd.DataFrame(rows)

    log.info("Total seeds: %d",len(seed_df))


    # -------------------------------------------------------
    # Allocate
    # -------------------------------------------------------

    N = len(seed_df)

    X = np.lib.format.open_memmap(
        PATCHES_PATH,"w+",
        np.float16,
        (N,NUM_BANDS,PATCH_SIZE,PATCH_SIZE)
    )

    y = np.zeros(N,np.int64)


    # -------------------------------------------------------
    # Build patches
    # -------------------------------------------------------

    idx = 0

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        for i,row in seed_df.iterrows():

            try:

                files = sessions[row.session]

                for m in files:

                    if m.filename == row.hdr \
                       or m.filename.lower().endswith("black.hdr"):

                        extract_pair(zf,m,tmp)


                hdr = tmp / row.hdr
                dark = next(tmp.rglob("black.hdr"))

                cube = preprocess(hdr,dark)

                lab,_ = segment(cube,wl)

                r0,c0,r1,c1 = row.bbox

                p = cube[r0:r1,c0:c1]

                mask = lab[r0:r1,c0:c1]==row.rid

                p *= mask[...,None]

                p = pad(p)
                p = resize(p)
                p = snv(p)

                p = np.transpose(p,(2,0,1)).astype(np.float16)

                X[idx]=p
                y[idx]=row.label

                idx+=1

                shutil.rmtree(tmp)
                tmp.mkdir()

            except Exception:

                log.error("PATCH FAIL %s",row.hdr)
                log.error(traceback.format_exc())


    X.flush()
    np.save(LABELS_PATH,y)


    # -------------------------------------------------------
    # Verify
    # -------------------------------------------------------

    assert len(np.load(PATCHES_PATH,mmap_mode="r"))==len(y)

    log.info("Samples: %d",len(y))
    log.info("DONE.")



# ============================================================
# ENTRY
# ============================================================

if __name__=="__main__":

    try:
        main()
    except Exception:

        log.critical("FATAL")
        log.critical(traceback.format_exc())
        sys.exit(1)