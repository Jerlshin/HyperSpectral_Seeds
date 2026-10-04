"""Deterministic foreground RGB extraction and fail-closed grid correspondence.

Coordinates always refer to encoded sensor pixels, ignoring JPEG EXIF orientation.
These fixed thresholds describe the Zenodo 3241923 acquisition, not arbitrary photos.
No class/session labels enter segmentation or pairing. Historical HSI row order is
preserved by the caller; grid IDs are independent of connected-component ordering.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from sklearn.cluster import KMeans

Array = npt.NDArray[Any]


def decode_rgb(payload: bytes) -> Array:
    """Decode in raw sensor orientation (EXIF must never change kernel identity)."""
    image = cv2.imdecode(
        np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        raise ValueError("Invalid RGB payload")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def segment_rgb(image: Array) -> tuple[Array, list[Any], float]:
    """Segment at half resolution; return native-resolution masks and regions.

    The ROI excludes most hardware and the paper label. Shape/border checks reject
    remaining label/hardware components; exactly 48 valid seeds is a hard gate.
    Masks are refined with seeded native-resolution GrabCut (two iterations),
    using a spatially restricted low-threshold proposal and coarse-component overlap.
    """
    if image.shape != (3264, 4896, 3) or image.dtype != np.uint8:
        raise ValueError("Expected raw-orientation 3264x4896 uint8 RGB")
    small = cv2.resize(image, (2448, 1632), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    y0, y1, x0, x1 = int(0.07 * h), int(0.99 * h), int(0.33 * w), int(0.76 * w)
    threshold, _ = cv2.threshold(gray[y0:y1, x0:x1], 0, 255, cv2.THRESH_OTSU)
    binary = np.zeros_like(gray)
    binary[y0:y1, x0:x1] = gray[y0:y1, x0:x1] > threshold
    binary = binary_fill_holes(binary).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    native_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    native = np.zeros(image.shape[:2], np.int32)
    count = 0
    for i in range(1, n):
        x, y, width, height, area = stats[i]
        if not (700 < area < 6000 and x > x0 + 2 and x + width < x1 - 2):
            continue
        local = (labels[y : y + height, x : x + width] == i).astype(np.uint8)
        r = regionprops(local)[0]  # type: ignore[no-untyped-call]
        if r.eccentricity <= 0.6 or (r.axis_major_length <= 65 and r.solidity <= 0.7):
            continue
        # A wide local box recovers dark tips; overlap selects the same seed even
        # if adjacent seeds enter the box. The coarse stage only establishes identity.
        a, b = max(0, 2 * y - 160), max(0, 2 * x - 160)
        c, d = min(3264, 2 * (y + height) + 160), min(4896, 2 * (x + width) + 160)
        coarse = cv2.resize(local, (2 * width, 2 * height), interpolation=cv2.INTER_NEAREST) > 0
        gc = np.full((c - a, d - b), cv2.GC_PR_BGD, np.uint8)
        anchor = np.zeros_like(gc)
        anchor[2 * y - a : 2 * (y + height) - a, 2 * x - b : 2 * (x + width) - b][coarse] = 1
        support = cv2.dilate(anchor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 241))) > 0
        gc[support & (native_gray[a:c, b:d] > 0.7 * threshold)] = cv2.GC_PR_FGD
        gc[anchor > 0] = cv2.GC_FGD
        gc[:3] = gc[-3:] = cv2.GC_BGD
        gc[:, :3] = gc[:, -3:] = cv2.GC_BGD
        cv2.setRNGSeed(0)
        cv2.grabCut(
            image[a:c, b:d],
            gc,
            None,
            np.zeros((1, 65)),
            np.zeros((1, 65)),
            2,
            cv2.GC_INIT_WITH_MASK,
        )
        mask = binary_fill_holes((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        nn, ll, ss, _ = cv2.connectedComponentsWithStats(mask)
        if nn <= 1:
            raise ValueError("RGB component vanished at native resolution")
        coarse = cv2.resize(local, (2 * width, 2 * height), interpolation=cv2.INTER_NEAREST) > 0
        overlap = ll[2 * y - a : 2 * (y + height) - a, 2 * x - b : 2 * (x + width) - b][coarse]
        overlap_counts = np.bincount(overlap, minlength=nn)
        overlap_counts[0] = 0
        keep = int(overlap_counts.argmax())
        if keep == 0:
            raise ValueError("Native mask lost its coarse component")
        count += 1
        native[a:c, b:d][ll == keep] = count
    regions = list(regionprops(native))  # type: ignore[no-untyped-call]
    if len(regions) != 48:
        raise ValueError(f"Expected 48 RGB seeds, found {len(regions)}")
    return native, regions, float(threshold)


def grid_cells(centroids: Array, rows: int = 8, cols: int = 6) -> Array:
    """Assign row/column identities, permitting missing but never duplicate cells."""
    points = np.asarray(centroids, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < rows:
        raise ValueError("Invalid grid centroids")
    out = np.zeros((len(points), 2), dtype=np.int64)
    for axis, k in enumerate((rows, cols)):
        fit = KMeans(n_clusters=k, random_state=0, n_init=10).fit(points[:, axis : axis + 1])
        centers = fit.cluster_centers_[:, 0]
        order = np.argsort(centers)
        ranks = np.empty(k, dtype=np.int64)
        ranks[order] = np.arange(k)
        out[:, axis] = ranks[fit.labels_]
        spacing = float(np.median(np.diff(centers[order])))
        if spacing <= 0 or np.max(np.abs(points[:, axis] - centers[fit.labels_])) > 0.49 * spacing:
            raise ValueError("Grid centers are ambiguous or irregular")
    if len(np.unique(out, axis=0)) != len(out):
        raise ValueError("Duplicate grid identity")
    return out


def pair_grids(hsi_centroids: Array, rgb_centroids: Array) -> tuple[Array, dict[str, Any]]:
    """Pair by grid, then independently check centroid jitter using a homography.

    A homography is a QC approximation, not a claim of pixel registration. Compare
    all row/column reversals; reject an orientation that does not uniquely preserve
    the observed arrangement of individual kernels.
    """
    hc, rc = grid_cells(hsi_centroids), grid_cells(rgb_centroids)
    lookup = {tuple(cell): i for i, cell in enumerate(rc)}
    pairs = np.array([lookup[tuple(c)] for c in hc])
    spacing = float(
        np.median(np.diff(np.sort([np.mean(hsi_centroids[hc[:, 0] == i, 0]) for i in range(8)])))
    )
    scores = []
    residual = np.zeros(len(pairs))
    for flip_y, flip_x in [(False, False), (False, True), (True, False), (True, True)]:
        ids = np.array(
            [
                lookup[(7 - int(y) if flip_y else int(y), 5 - int(x) if flip_x else int(x))]
                for y, x in hc
            ]
        )
        source = np.asarray(rgb_centroids[ids, ::-1], dtype=np.float64)
        target = np.asarray(hsi_centroids[:, ::-1], dtype=np.float64)
        matrix, _ = cv2.findHomography(source, target, method=0)
        if matrix is None:
            raise ValueError("Degenerate grid registration")
        projected = cv2.perspectiveTransform(source[:, None], matrix)[:, 0]
        errors = np.linalg.norm(projected - target, axis=1)
        scores.append(float(np.sqrt(np.mean(errors**2))))
        if not flip_y and not flip_x:
            residual = errors
    if scores[0] != min(scores) or min(scores[1:]) < 1.5 * max(scores[0], 1e-6):
        raise ValueError(f"Uncertain physical orientation: RMSE {scores}")
    if residual.max() / spacing > 0.22:
        raise ValueError(
            f"Registration residual exceeds 0.22 row pitch: {residual.max() / spacing}"
        )
    return pairs, {
        "rmse_hsi_px": scores[0],
        "max_hsi_px": float(residual.max()),
        "max_fraction_row_pitch": float(residual.max() / spacing),
        "alternative_rmse_hsi_px": scores[1:],
        "residuals": residual.tolist(),
    }


def foreground_crop(image: Array, labeled: Array, region: Any, size: int = 224) -> Array:
    """Square-pad a native crop with 5% margin; preserve aspect, zero all background."""
    y0, x0, y1, x1 = region.bbox
    crop = image[y0:y1, x0:x1].copy()
    crop[labeled[y0:y1, x0:x1] != region.label] = 0
    h, w = crop.shape[:2]
    side = int(np.ceil(max(h, w) * 1.1))
    square = np.zeros((side, side, 3), dtype=np.uint8)
    a, b = (side - h) // 2, (side - w) // 2
    square[a : a + h, b : b + w] = crop
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def rgb_descriptors(image: Array, labeled: Array, region: Any) -> Array:
    """Native foreground color quantiles and grayscale local texture (no background)."""
    y0, x0, y1, x1 = region.bbox
    rgb = image[y0:y1, x0:x1]
    mask = (labeled[y0:y1, x0:x1] == region.label).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    values = lab[mask > 0]
    color = np.concatenate(
        [values.mean(0), values.std(0), np.quantile(values, [0.1, 0.5, 0.9], axis=0).ravel()]
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    core = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    if not core.any():
        raise ValueError("Empty RGB texture core")
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = np.concatenate(
        [
            np.quantile(gray[core], [0.1, 0.25, 0.5, 0.75, 0.9]),
            [
                gray[core].std(),
                np.mean(np.hypot(dx, dy)[core]),
                np.asarray(lap[core], dtype=np.float32).std(),
            ],
        ]
    )
    result: Array = np.concatenate([color, texture]).astype(np.float32)
    return result
