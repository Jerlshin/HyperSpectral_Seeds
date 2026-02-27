import os
import re
import sys
import zipfile
import shutil
import logging
import subprocess
import traceback
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes

import warnings
import spectral

warnings.filterwarnings("ignore", message="Unable to parse")
spectral.settings.envi_support_nonlowercase_params = True

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

ROOT = Path("./dataset").resolve()
ROOT.mkdir(exist_ok=True)

DATA_URL = (
    "https://zenodo.org/records/3241923/files/"
    "RGB%20and%20VIS-NIR%20HSI%20Data%20for%2090%20Rice%20Seed%20Varieties.zip?download=1"
)

ZIP_FILE     = ROOT / "rice_hsi.zip"
PATCHES_PATH = ROOT / "patches.npy"
LABELS_PATH  = ROOT / "labels.npy"
LOG_FILE     = ROOT / "prepare_dataset.log"

PATCH_SIZE = 64
NUM_BANDS  = 256

# ════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("builder")


# ════════════════════════════════════════════════════════════
#  DOWNLOAD
# ════════════════════════════════════════════════════════════

def download() -> None:
    if ZIP_FILE.exists():
        log.info("Zip already exists — skipping download.")
        return

    log.info("Downloading dataset from Zenodo …")

    if shutil.which("wget"):
        subprocess.run(
            ["wget", "--progress=bar:force", "-O", str(ZIP_FILE), DATA_URL],
            check=True,
        )
        return

    if shutil.which("curl"):
        subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", str(ZIP_FILE), DATA_URL],
            check=True,
        )
        return

    import urllib.request
    log.info("No wget/curl found — using urllib (no progress bar).")
    urllib.request.urlretrieve(DATA_URL, ZIP_FILE)


# ════════════════════════════════════════════════════════════
#  ZIP INDEXING
# ════════════════════════════════════════════════════════════

def open_zip():
    """
    Open the zip and build a session → [ZipInfo members] index.

    FIX (Bug 5): Find the Data-VIS part regardless of nesting depth.
    The zip may be structured as:
        "TopFolder/Data-VIS-1/file.hdr"   (depth 2)
        "Data-VIS-1/file.hdr"             (depth 1)
    We search every path component for "Data-VIS".
    """
    zf = zipfile.ZipFile(ZIP_FILE, "r")
    sessions: dict[str, list] = {}

    for m in zf.infolist():
        parts = Path(m.filename).parts
        # Find whichever part starts with "Data-VIS"
        session = next(
            (p for p in parts if p.startswith("Data-VIS")),
            None,
        )
        if session is None:
            continue
        sessions.setdefault(session, []).append(m)

    log.info("Found %d Data-VIS sessions in zip.", len(sessions))
    return zf, sessions


def find_data_member(zf: zipfile.ZipFile, hdr_member) -> zipfile.ZipInfo:
    """
    FIX (Bug 4): Find the data file (.raw / .img / .bil / .bsq etc.)
    paired with an ENVI .hdr, rather than blindly appending ".raw".

    Strategy: same path stem, any extension except .hdr.
    """
    stem = hdr_member.filename[:-4]  # strip ".hdr"
    candidates = [
        m for m in zf.infolist()
        if m.filename.startswith(stem)
        and not m.filename.lower().endswith(".hdr")
        and not m.filename.lower().endswith(".jpg")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No data file found for HDR: {hdr_member.filename}"
        )
    # Prefer .raw, then .img, else take first
    for ext in (".raw", ".img", ".bil", ".bsq"):
        for c in candidates:
            if c.filename.lower().endswith(ext):
                return c
    return candidates[0]


def extract_pair(zf: zipfile.ZipFile, hdr_member, tmpdir: Path):
    """Extract an ENVI HDR + its data file to tmpdir."""
    hdr_path = tmpdir / hdr_member.filename
    hdr_path.parent.mkdir(parents=True, exist_ok=True)

    with zf.open(hdr_member) as src, open(hdr_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    data_member = find_data_member(zf, hdr_member)
    data_path   = tmpdir / data_member.filename
    data_path.parent.mkdir(parents=True, exist_ok=True)

    with zf.open(data_member) as src, open(data_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    return hdr_path, data_path


# ════════════════════════════════════════════════════════════
#  HSI I/O
# ════════════════════════════════════════════════════════════

def load_hsi(hdr: Path) -> np.ndarray:
    """Load an ENVI hyperspectral cube → float32 (H, W, B)."""
    from spectral.io import envi
    img  = envi.open(str(hdr))
    cube = np.asarray(img.load()).astype(np.float32)
    return cube


# ════════════════════════════════════════════════════════════
#  PREPROCESSING
# ════════════════════════════════════════════════════════════

def preprocess(hdr: Path, dark_hdr: Path) -> np.ndarray:
    """
    1. Load cube and dark reference.
    2. Subtract per-column dark mean (dark frame is a line scan).
    3. Clip to non-negative values.
    4. Crop top 600 rows (removes calibration panel at bottom).
    """
    cube = load_hsi(hdr)
    dark = load_hsi(dark_hdr)

    # Dark reference is a line scan → average over spatial rows
    dark_mean = dark.mean(axis=0, keepdims=True)  # (1, W, B)
    cube = np.clip(cube - dark_mean, 0.0, None)

    return cube[:600]


# ════════════════════════════════════════════════════════════
#  SEGMENTATION
# ════════════════════════════════════════════════════════════

def segment(cube: np.ndarray, wl: np.ndarray):
    """
    Segment individual rice seeds from a hyperspectral cube.

    Steps:
      1. Project to visible mean image (450–700 nm).
      2. Otsu threshold at 40% to separate seeds from background.
      3. Morphological cleanup (fill, close, clear borders, remove small).
      4. Filter regions by area, eccentricity, solidity
         (keeps rice-grain-shaped objects).
      5. Sort spatially (top→bottom, left→right).

    Returns labeled array + filtered region list.
    """
    vis_mask = (wl > 450) & (wl < 700)
    if vis_mask.sum() == 0:
        raise RuntimeError("No VIS bands in wavelength range 450–700 nm.")

    vis_img = cube[:, :, vis_mask].mean(axis=2)

    if not np.isfinite(vis_img).all():
        raise RuntimeError("NaN/Inf in VIS projection — check dark correction.")

    thresh = threshold_otsu(vis_img)
    binary = vis_img > (0.4 * thresh)

    # Morphological cleanup
    binary = binary_fill_holes(binary)
    binary = binary_closing(binary, disk(3))   # close thin gaps in seeds
    binary = clear_border(binary)              # remove seeds touching edge
    binary = remove_small_objects(binary, min_size=150)

    labeled = label(binary)
    regions = regionprops(labeled)

    # Shape-based filter: rice grains are elongated (eccentricity > 0.6),
    # compact (solidity > 0.85), and within a realistic pixel area range.
    regions = [
        r for r in regions
        if 300 < r.area < 800
        and r.eccentricity > 0.6
        and r.solidity > 0.85
    ]

    # Spatial sort: enables reproducible positional indexing
    regions.sort(key=lambda r: (r.centroid[0], r.centroid[1]))

    return labeled, regions


# ════════════════════════════════════════════════════════════
#  PATCH PROCESSING
# ════════════════════════════════════════════════════════════

def pad_to_square(patch: np.ndarray) -> np.ndarray:
    """Center-pad a (H, W, C) patch to (max(H,W), max(H,W), C)."""
    h, w, c = patch.shape
    s   = max(h, w)
    out = np.zeros((s, s, c), dtype=patch.dtype)
    y   = (s - h) // 2
    x   = (s - w) // 2
    out[y: y + h, x: x + w] = patch
    return out


def resize_patch(patch: np.ndarray, size: int = PATCH_SIZE) -> np.ndarray:
    """
    Resize each spectral band independently.

    FIX (Bug 2): Use cv2.INTER_AREA for downsampling (more accurate
    than INTER_LINEAR for aggregating pixel areas in HSI).
    """
    h, w = patch.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    out = np.zeros((size, size, patch.shape[2]), dtype=np.float32)
    for b in range(patch.shape[2]):
        out[:, :, b] = cv2.resize(patch[:, :, b], (size, size),
                                  interpolation=interp)
    return out


def snv_masked(patch: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """
    FIX (Bug 1): Standard Normal Variate computed on SEED PIXELS ONLY.

    Original code:  (patch - patch.mean()) / (patch.std() + 1e-6)
    Problem: mean/std computed over ALL pixels including background zeros.
    With a ~30% foreground ratio, background zeros drag the mean low,
    making the "normalized" background ≈ +mean/std ≈ large positive noise.
    The model cannot distinguish seed from background.

    Correct approach:
      - Compute global mean/std from seed pixels only (mask==True).
      - Normalize entire patch (seed and background) with those stats.
      - Background pixels that were 0 before become -mean/std, still
        distinguishable from real seed signal by the masked-mean branch
        in the model.

    Alternative (also valid, used here): after SNV, re-zero the background
    so the mask is preserved exactly.  This is what the model's
    masked_mean_spectrum() depends on.
    """
    # Expand mask for broadcasting: (H, W) → (H, W, 1)
    mask3d = seed_mask[:, :, np.newaxis]

    seed_pixels = patch[mask3d.squeeze(-1)]   # (N_seed_pixels, C)

    if seed_pixels.size == 0:
        return patch  # degenerate patch — caller will skip

    mu  = seed_pixels.mean()
    std = seed_pixels.std() + 1e-6

    normalized = (patch - mu) / std

    # Re-zero background so the model's mask detection still works
    normalized = normalized * mask3d

    return normalized.astype(np.float32)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main() -> None:

    download()
    zf, sessions = open_zip()

    # ── Load wavelengths ──────────────────────────────────────────
    # FIX (Bug 7): initialize wl = None; assert before use
    wl = None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for m in zf.infolist():
            if m.filename.endswith("wavelengths.csv"):
                wl_csv = tmp_path / "wl.csv"
                with zf.open(m) as src, open(wl_csv, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                wl_df = pd.read_csv(wl_csv)
                if "Wavelength (nm)" not in wl_df.columns:
                    raise RuntimeError(
                        f"Unexpected wavelengths.csv columns: {wl_df.columns.tolist()}"
                    )
                wl = wl_df["Wavelength (nm)"].values.astype(float)
                log.info(
                    "Wavelengths loaded: %d bands, %.1f–%.1f nm",
                    len(wl), wl.min(), wl.max()
                )
                break

    if wl is None:
        raise RuntimeError("wavelengths.csv not found in zip.")

    # ── Build cube index ──────────────────────────────────────────
    cubes = []
    for sess, files in sessions.items():
        for m in files:
            fname = m.filename.lower()
            if fname.endswith(".hdr") and not fname.endswith("black.hdr"):
                name    = Path(m.filename).stem
                variety = name.rsplit("-", 1)[0]
                cubes.append((sess, variety, m))

    df = pd.DataFrame(cubes, columns=["session", "variety", "member"])
    df["label"] = pd.factorize(df["variety"])[0]

    log.info(
        "Cube index: %d cubes | %d varieties",
        len(df), df["label"].nunique()
    )

    # ── Phase 1: Detect seeds, save metadata ─────────────────────
    # Only store (hdr_path, session, label, bbox, region_id) — no pixel data.
    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting"):
            try:
                files = sessions[row.session]

                # Extract this cube's HDR/data + the session's dark reference
                for m in files:
                    if (m.filename == row.member.filename or
                            m.filename.lower().endswith("black.hdr")):
                        extract_pair(zf, m, tmp_path)

                hdr_path  = tmp_path / row.member.filename
                dark_path = next(tmp_path.rglob("black.hdr"), None)

                if dark_path is None:
                    raise FileNotFoundError("black.hdr not found in session.")

                cube = preprocess(hdr_path, dark_path)
                labeled, regs = segment(cube, wl)

                for r in regs:
                    rows.append({
                        "hdr":     row.member.filename,
                        "session": row.session,
                        "label":   row.label,
                        "variety": row.variety,
                        "bbox":    r.bbox,
                        "rid":     r.label,
                    })

                log.info(
                    "[%d/%d] %s — %d seeds detected",
                    i + 1, len(df), row.variety, len(regs)
                )

                # Clear tmp for next cube (disk safety)
                shutil.rmtree(tmp_path)
                tmp_path.mkdir()

            except Exception:
                log.error("SEGMENT FAIL: %s", row.member.filename)
                log.error(traceback.format_exc())

    seed_df = pd.DataFrame(rows)
    log.info(
        "Total seed regions: %d across %d varieties",
        len(seed_df), seed_df["label"].nunique()
    )

    if len(seed_df) == 0:
        raise RuntimeError("No seeds detected. Check segmentation parameters.")

    # Sanity check: verify all 90 varieties have seeds
    missing = set(range(df["label"].nunique())) - set(seed_df["label"].unique())
    if missing:
        log.warning("Varieties with 0 seeds detected: %s", missing)

    # ── Phase 2: Extract patches ──────────────────────────────────
    # FIX (Bug 3 + Bug 6):
    #   - Group by HDR path → each cube is loaded ONCE per cube, not once per seed.
    #   - Collect successful patches into a list before stacking.
    #     This eliminates the "trailing zeros from failed patches" bug.

    patch_list:  list[np.ndarray] = []
    label_list:  list[int]        = []
    fail_count = 0

    # Group by cube so each cube is loaded only once
    grouped = seed_df.groupby("hdr")
    n_cubes = len(grouped)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for cube_hdr, group in tqdm(grouped, total=n_cubes, desc="Extracting patches"):
            session = group.iloc[0]["session"]

            try:
                # Locate ZipInfo for this HDR
                hdr_member = next(
                    m for m in sessions[session]
                    if m.filename == cube_hdr
                )

                # Extract cube + dark once for all seeds in this group
                files = sessions[session]
                for m in files:
                    if (m.filename == cube_hdr or
                            m.filename.lower().endswith("black.hdr")):
                        extract_pair(zf, m, tmp_path)

                hdr_path  = tmp_path / cube_hdr
                dark_path = next(tmp_path.rglob("black.hdr"), None)

                if dark_path is None:
                    raise FileNotFoundError("black.hdr not found for " + cube_hdr)

                cube    = preprocess(hdr_path, dark_path)
                labeled, _ = segment(cube, wl)  # re-segment to get consistent labels

                # Extract all seeds from this cube in one pass
                for row in group.itertuples():
                    try:
                        r0, c0, r1, c1 = row.bbox

                        patch     = cube[r0:r1, c0:c1, :].copy()           # (H, W, 256)
                        seed_mask = (labeled[r0:r1, c0:c1] == row.rid)     # (H, W)

                        # Apply seed mask to zero out background
                        patch = patch * seed_mask[:, :, np.newaxis]

                        # Sanity check: skip degenerate patches
                        if seed_mask.sum() == 0:
                            log.warning("Zero-area mask for rid=%d in %s", row.rid, cube_hdr)
                            fail_count += 1
                            continue

                        if not np.isfinite(patch).all():
                            log.warning("NaN/Inf in patch for rid=%d in %s", row.rid, cube_hdr)
                            fail_count += 1
                            continue

                        # Spatial processing
                        patch = pad_to_square(patch)      # (S, S, 256)
                        patch = resize_patch(patch)        # (64, 64, 256)

                        # FIX (Bug 1): SNV computed on seed pixels only
                        patch = snv_masked(patch, seed_mask_resized(seed_mask))

                        # Channel-first for PyTorch: (256, 64, 64)
                        patch = np.transpose(patch, (2, 0, 1)).astype(np.float16)

                        patch_list.append(patch)
                        label_list.append(int(row.label))

                    except Exception:
                        log.error("PATCH FAIL rid=%d in %s", row.rid, cube_hdr)
                        log.error(traceback.format_exc())
                        fail_count += 1

                # Clear tmp for next cube
                shutil.rmtree(tmp_path)
                tmp_path.mkdir()

            except Exception:
                log.error("CUBE FAIL: %s", cube_hdr)
                log.error(traceback.format_exc())
                fail_count += len(group)

    log.info(
        "Patch extraction complete: %d successful, %d failed.",
        len(patch_list), fail_count
    )

    if len(patch_list) == 0:
        raise RuntimeError("No patches extracted. Something went wrong.")

    # ── Save ─────────────────────────────────────────────────────
    # FIX (Bug 6): Stack only successful patches — no trailing zeros.
    N = len(patch_list)
    log.info("Saving %d patches to %s …", N, PATCHES_PATH)

    # Allocate memmap for final correct size
    X_mmap = np.lib.format.open_memmap(
        PATCHES_PATH, mode="w+", dtype=np.float16,
        shape=(N, NUM_BANDS, PATCH_SIZE, PATCH_SIZE)
    )
    for i, p in enumerate(tqdm(patch_list, desc="Writing memmap")):
        X_mmap[i] = p
    X_mmap.flush()

    y_arr = np.array(label_list, dtype=np.int64)
    np.save(LABELS_PATH, y_arr)

    # ── Verify ───────────────────────────────────────────────────
    X_verify = np.load(PATCHES_PATH, mmap_mode="r")
    y_verify = np.load(LABELS_PATH)

    assert X_verify.shape[0] == len(y_verify), "Shape mismatch!"
    assert X_verify.shape[1:] == (NUM_BANDS, PATCH_SIZE, PATCH_SIZE), \
        f"Unexpected patch shape: {X_verify.shape}"

    # Print label distribution
    unique, counts = np.unique(y_verify, return_counts=True)
    log.info(
        "Label distribution — min: %d, max: %d, mean: %.1f seeds/class",
        counts.min(), counts.max(), counts.mean()
    )
    log.info(
        "Patches shape : %s  dtype: %s",
        X_verify.shape, X_verify.dtype
    )
    log.info("Labels  shape : %s  dtype: %s", y_verify.shape, y_verify.dtype)
    log.info("Classes found : %d / 90", len(unique))
    log.info("DONE — dataset ready.")


def seed_mask_resized(original_mask: np.ndarray) -> np.ndarray:
    """
    Resize the binary seed mask to (PATCH_SIZE, PATCH_SIZE) so that
    snv_masked() can correctly identify seed pixels in the resized patch.

    This is applied after pad_to_square (which zero-pads), so the mask
    must also be square-padded before resizing.
    """
    h, w = original_mask.shape
    s = max(h, w)

    # Pad mask the same way pad_to_square pads the patch
    padded = np.zeros((s, s), dtype=np.uint8)
    y = (s - h) // 2
    x = (s - w) // 2
    padded[y: y + h, x: x + w] = original_mask.astype(np.uint8)

    resized = cv2.resize(
        padded, (PATCH_SIZE, PATCH_SIZE),
        interpolation=cv2.INTER_NEAREST  # binary mask: never interpolate
    )
    return resized.astype(bool)


# ════════════════════════════════════════════════════════════
#  ENTRY
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("FATAL ERROR")
        log.critical(traceback.format_exc())
        sys.exit(1)