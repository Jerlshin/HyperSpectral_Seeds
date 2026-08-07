"""Radiometric normalisation — P-2 / T4-2.

The archive is **radiance**, not reflectance: ``preprocess_raw`` subtracts the
dark current and stops there. IMPROVEMENT_PLAN §2.1.1 (C-1) shows what that
leaves behind — a per-session illumination gain :math:`S_c` multiplying every
spectrum captured in that session, and a per-pixel geometric gain :math:`a_p`
from illumination angle and the resize fill factor. Both are acquisition
metadata wearing the costume of a spectral signature, and with an ungrouped
split both are perfectly predictive of the label.

Two remedies, in the plan's order of preference:

**(a) White-reference division** — the physically correct one:

.. math::
    \\varrho_{h,w,c} =
    \\frac{R_{h,w,c} - \\bar D_{w,c}}{\\bar W_{w,c} - \\bar D_{w,c} + \\epsilon}

It needs a white panel scanned under the same illumination.
:func:`find_white_reference` looks for one; **this archive has none** — the
only reference cube per session is ``black.hdr``, which
``preprocess_raw`` already consumes — so on this dataset the resolution is
always (b). The code path is implemented and tested regardless, because the
absence is a property of one archive, not of the method.

**(b) Per-pixel Standard Normal Variate** — the chemometric standard, applied
along :math:`\\lambda` *after* masking:

.. math::
    \\tilde x_{c,p} = \\frac{x_{c,p} - \\bar x_{\\cdot,p}}
                           {\\mathrm{sd}_c(x_{\\cdot,p}) + \\epsilon},
    \\qquad \\bar x_{\\cdot,p} = \\frac{1}{C_\\lambda}\\sum_c x_{c,p}

Under the gain model :math:`x_{c,p} = a_p r_c` this is **exactly**
:math:`(r_c - \\bar r)/\\mathrm{sd}(r)`: the per-pixel gain and the session gain
both cancel identically, not approximately (see
``tests/unit/test_radiometry.py::test_snv_removes_a_per_pixel_gain_exactly``).

The gain that SNV divides out is not thrown away. :func:`snv_normalise`
returns :math:`(\\bar x_{\\cdot,p}, \\mathrm{sd}_c)` as well, and
``patch_extraction`` persists them, so brightness remains available to the
network as an explicit input rather than as an unremovable component of every
spectrum.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

#: Denominator floor in both corrections. Radiance is O(1e3) here, so 1e-6 is
#: well below any real signal and exists only to keep a dead pixel finite.
RADIOMETRY_EPS: float = 1e-6

#: Accepted values of ``PrepConfig.radiometry``.
#:
#: ``auto`` picks ``white`` when the archive carries a white panel and ``snv``
#: when it does not, and says which it chose. ``none`` is the pre-Tier-4
#: behaviour, kept so the leak channel §2.1.1 describes stays reproducible.
RADIOMETRY_MODES: tuple[str, ...] = ("auto", "white", "snv", "none")

#: Filename fragments that would identify a white-reference scan. Checked
#: against the archive's ``.hdr`` members; ``black`` is excluded explicitly
#: because a dark reference is not a white one.
_WHITE_HINTS: tuple[str, ...] = ("white", "whiteref", "reference", "panel", "spectralon")


def find_white_reference(member_names: Iterable[str]) -> str | None:
    """Return the archive member holding a white panel, or ``None`` if there is none.

    Args:
        member_names: Every filename in the archive.

    Returns:
        The first ``.hdr`` member whose name suggests a white reference, or
        ``None``. On the Zenodo rice archive this is ``None``: the 190 ``.hdr``
        members are 180 seed cubes plus 9 ``black.hdr`` dark references and one
        ``chessboard.hdr`` geometric target.
    """
    for name in member_names:
        low = name.lower()
        if not low.endswith(".hdr") or "black" in low:
            continue
        if any(hint in low for hint in _WHITE_HINTS):
            return name
    return None


def resolve_radiometry(mode: str, white_reference: str | None) -> tuple[str, str]:
    """Resolve ``mode`` against what the archive actually provides.

    Returns:
        ``(effective_mode, reason)`` — the mode to apply and a one-line
        statement of why, meant to be printed into the extraction log so a
        dataset's radiometric provenance is recorded rather than inferred.

    Raises:
        ValueError: ``mode`` is not one of :data:`RADIOMETRY_MODES`, or
            ``white`` was requested and no panel exists.
    """
    if mode not in RADIOMETRY_MODES:
        raise ValueError(f"radiometry must be one of {RADIOMETRY_MODES}, got {mode!r}")

    if mode == "white":
        if white_reference is None:
            raise ValueError(
                "radiometry='white' requires a white panel in the archive and none was found. "
                "Use 'snv' (per-pixel Standard Normal Variate) or 'auto'."
            )
        return "white", f"white-reference division against {white_reference}"

    if mode == "auto":
        if white_reference is not None:
            return "white", f"auto → white-reference division against {white_reference}"
        return "snv", "auto → no white panel in the archive; per-pixel SNV along lambda"

    if mode == "snv":
        return "snv", "per-pixel SNV along lambda, after masking"

    return "none", "no radiometric normalisation — radiance domain (pre-Tier-4 behaviour)"


def white_gain(
    white: npt.NDArray[Any], dark: npt.NDArray[Any], eps: float = RADIOMETRY_EPS
) -> npt.NDArray[np.float32]:
    """The denominator :math:`\\bar W_{w,c} - \\bar D_{w,c} + \\epsilon`, shaped ``(1, W, C)``.

    Split out because ``preprocess_raw`` has already subtracted
    :math:`\\bar D`: a cube on that path needs only the division, and
    re-subtracting the dark reference would double-count it.
    """
    dbar = np.asarray(dark, dtype=np.float32).mean(axis=0, keepdims=True)
    wbar = np.asarray(white, dtype=np.float32).mean(axis=0, keepdims=True)
    # Annotated rather than returned inline: `ndarray.mean` is typed `-> Any`,
    # so the arithmetic below loses its dtype without the binding.
    gain: npt.NDArray[np.float32] = (wbar - dbar + np.float32(eps)).astype(np.float32)
    return gain


def white_reference_correct(
    cube: npt.NDArray[Any],
    white: npt.NDArray[Any],
    dark: npt.NDArray[Any],
    eps: float = RADIOMETRY_EPS,
) -> npt.NDArray[np.float32]:
    """Convert a **raw** radiance cube to reflectance by white/dark reference division.

    Implements §3.1 P-2(a). Both references are averaged over the scan
    direction (axis 0) exactly as ``preprocess_raw`` averages the dark cube, so
    :math:`\\bar D` and :math:`\\bar W` are per-``(column, band)`` — the right
    shape for a push-broom sensor, whose gain varies across the array and not
    along the scan.

    Args:
        cube: ``(H, W, C)`` raw radiance, **not** dark-corrected.
        white: ``(H_w, W, C)`` white panel scan.
        dark: ``(H_d, W, C)`` dark-current scan.
        eps: Denominator floor.

    Returns:
        ``(H, W, C)`` reflectance, clipped to non-negative.
    """
    dbar = np.asarray(dark, dtype=np.float32).mean(axis=0, keepdims=True)
    rho = (np.asarray(cube, dtype=np.float32) - dbar) / white_gain(white, dark, eps)
    clipped: npt.NDArray[np.float32] = np.clip(rho, 0.0, None).astype(np.float32)
    return clipped


def snv_normalise(
    patch: npt.NDArray[Any],
    mask: npt.NDArray[Any] | None = None,
    eps: float = RADIOMETRY_EPS,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Per-pixel SNV along the band axis, plus the gain it removes.

    Implements §3.1 P-2(b). The band axis is the **last** one, matching the
    ``(H, W, C)`` layout every function in ``data/prep`` works in.

    Background pixels — those ``mask`` excludes — are left at exactly ``0.0``
    in all three outputs. That is the invariant §2.6 shows the whole model
    depends on: `masked_spectral_stats`, the ECA gate and every TTA view infer
    the foreground by testing for zero, and SNV of a zero spectrum would
    otherwise be an arbitrary non-zero constant.

    Args:
        patch: ``(..., C)`` spectra, already masked (background zero).
        mask: ``(...)`` boolean foreground map. ``None`` treats every pixel as
            foreground, which is only correct for an unmasked cube.
        eps: Added to the standard deviation before dividing.

    Returns:
        ``(snv, mean, sd)`` — the normalised spectra with the same shape as
        ``patch``, and the two per-pixel gain maps with its leading shape.
    """
    x = np.asarray(patch, dtype=np.float32)
    mean = x.mean(axis=-1).astype(np.float32)
    sd = x.std(axis=-1).astype(np.float32)

    out = (x - mean[..., None]) / (sd[..., None] + np.float32(eps))

    if mask is not None:
        fg = np.asarray(mask).astype(bool)
        out = np.where(fg[..., None], out, np.float32(0.0))
        mean = np.where(fg, mean, np.float32(0.0))
        sd = np.where(fg, sd, np.float32(0.0))

    return out.astype(np.float32), mean.astype(np.float32), sd.astype(np.float32)


def apply_radiometry(
    patch: npt.NDArray[Any],
    mask: npt.NDArray[Any] | None,
    mode: str,
    eps: float = RADIOMETRY_EPS,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Dispatch the per-patch half of P-2 and return the patch with its gain map.

    ``white`` is a **cube-level** correction — it has to happen before
    segmentation, so by the time a patch exists it is already applied and this
    is a no-op that records the patch's own brightness. ``snv`` is the
    per-patch one.

    Args:
        patch: ``(H, W, C)`` masked patch.
        mask: ``(H, W)`` foreground map, or ``None``.
        mode: One of ``"white"``, ``"snv"``, ``"none"``.
        eps: Denominator floor.

    Returns:
        ``(patch, gain)`` where ``gain`` is ``(2, H, W)`` holding the per-pixel
        mean and standard deviation along the band axis. The gain is computed
        and persisted under every mode, so the brightness channel is available
        whether or not it was divided out.
    """
    normalised, mean, sd = snv_normalise(patch, mask, eps)
    gain = np.stack([mean, sd], axis=0).astype(np.float32)
    if mode == "snv":
        return normalised, gain
    return np.asarray(patch, dtype=np.float32), gain
