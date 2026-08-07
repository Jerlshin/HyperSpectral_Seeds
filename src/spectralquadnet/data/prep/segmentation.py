"""ENVI cube loading, dark-current correction and seed segmentation.

``spectral`` is imported lazily inside :func:`load_hsi` rather than at module
scope, both to defer its ``envi_support_nonlowercase_params = True``
import-time side effect and because it is an optional dependency
(``pip install -e ".[prep]"``) that training never needs.

:func:`morphometrics` (P-4 / T4-4) reads the eight shape descriptors off the
region ``segment`` already produced. They were computed, gated on — the
``300 < area < 800``, ``eccentricity > 0.6``, ``solidity > 0.85`` filter — and
then discarded (M-13), even though grain length and width are defining
cultivar descriptors and are stated in **physical pixel units**, the absolute
scale the 64x64 resize destroys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border

# scikit-image ships `py.typed` but leaves most of its public API unannotated, so
# every call below is an "untyped call in typed context" under `--strict`. The
# alternative — stub packages that do not exist for skimage — is not available;
# these four ignores are the narrowest possible scope.


def load_hsi(hdr: str | Path) -> npt.NDArray[np.float32]:
    """Load an ENVI hyperspectral cube from its ``.hdr`` file into a float32 array."""
    import spectral

    spectral.settings.envi_support_nonlowercase_params = True
    from spectral.io import envi

    img = envi.open(str(hdr))
    cube = np.asarray(img.load())
    return cube.astype(np.float32)


def preprocess_raw(hdr: str | Path, dark_hdr: str | Path) -> npt.NDArray[np.float32]:
    """Subtract the dark-current reference and crop to the first 600 rows.

    Args:
        hdr: Path to the raw scan's ``.hdr`` file.
        dark_hdr: Path to the dark-current reference scan's ``.hdr`` file.

    Returns:
        Dark-corrected cube, clipped to non-negative and cropped to rows
        ``[0, 600)`` (the sensor's valid-data band).
    """
    cube = load_hsi(hdr)
    dark = load_hsi(dark_hdr)

    dark_mean = dark.mean(axis=0, keepdims=True)
    cube = np.clip(cube - dark_mean, 0.0, None)

    return cube[:600]  # type: ignore[no-any-return]  # `np.clip` is typed `-> Any`


def segment(cube: npt.NDArray[Any], wl: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], list[Any]]:
    """Otsu-threshold the visible band mean and return the seed regions.

    Returns:
        ``(labeled, regions)`` — the label image and the filtered, centroid-sorted
        ``skimage`` region properties, top-left first.
    """
    vis_mask = (wl > 450) & (wl < 700)
    vis_img = cube[:, :, vis_mask].mean(axis=2)

    t = threshold_otsu(vis_img)  # type: ignore[no-untyped-call]
    binary = vis_img > (0.4 * t)

    binary = binary_fill_holes(binary)
    binary = clear_border(binary)  # type: ignore[no-untyped-call]
    binary = remove_small_objects(binary, 150)

    labeled = label(binary)  # type: ignore[no-untyped-call]
    regions = regionprops(labeled)  # type: ignore[no-untyped-call]

    regions = [
        r for r in regions if 300 < r.area < 800 and r.eccentricity > 0.6 and r.solidity > 0.85
    ]

    regions.sort(key=lambda r: (r.centroid[0], r.centroid[1]))
    return labeled, regions


# ══════════════════════════════════════════════════════════════════════
#  Morphometrics — P-4 / T4-4
# ══════════════════════════════════════════════════════════════════════

#: Column order of :func:`morphometrics`, and therefore of ``morphology.npy``.
#: Fixed by IMPROVEMENT_PLAN §3.1 P-4; the persisted array is meaningless
#: without it, so it is a module constant rather than a docstring.
MORPHOMETRIC_NAMES: tuple[str, ...] = (
    "area",
    "major_axis",
    "minor_axis",
    "axis_ratio",
    "eccentricity",
    "solidity",
    "equivalent_diameter",
    "perimeter_over_sqrt_area",
)

#: Guards the two ratios below against a degenerate region. ``segment`` already
#: rejects anything under 300 px, so neither denominator can be small in
#: practice; this only keeps a hand-built test region finite.
_SHAPE_EPS: float = 1e-8


def _prop(region: Any, *names: str) -> float:
    """First attribute of ``region`` that exists, by name.

    scikit-image renamed four ``regionprops`` attributes in 0.26
    (``major_axis_length`` → ``axis_major_length``, …) and deprecated the old
    spellings with a ``FutureWarning``. Both are accepted here so the
    extraction pipeline does not pin a scikit-image minor version.
    """
    for name in names:
        value = getattr(region, name, None)
        if value is not None:
            return float(value)
    raise AttributeError(f"region exposes none of {names}")


def morphometrics(region: Any) -> npt.NDArray[np.float32]:
    """The eight shape descriptors of §3.1 P-4, in :data:`MORPHOMETRIC_NAMES` order.

    All eight are already computed by ``regionprops`` inside :func:`segment`
    and three of them already gate the region filter; this only writes them
    down. They are in **physical pixel units** — deliberately, since that is
    the scale ``resize_patch`` removes and the one a cultivar descriptor lives
    on.

    The array is returned **unstandardised**. Standardisation is a function of
    the training split, which does not exist at extraction time; see
    :func:`spectralquadnet.data.morphometrics.standardise_morphometrics`,
    which fits on train alone.

    Args:
        region: One ``skimage.measure.RegionProperties``.

    Returns:
        ``(8,)`` float32.
    """
    area = _prop(region, "area")
    major = _prop(region, "axis_major_length", "major_axis_length")
    minor = _prop(region, "axis_minor_length", "minor_axis_length")
    return np.array(
        [
            area,
            major,
            minor,
            major / max(minor, _SHAPE_EPS),
            _prop(region, "eccentricity"),
            _prop(region, "solidity"),
            _prop(region, "equivalent_diameter_area", "equivalent_diameter"),
            _prop(region, "perimeter") / max(np.sqrt(area), _SHAPE_EPS),
        ],
        dtype=np.float32,
    )
