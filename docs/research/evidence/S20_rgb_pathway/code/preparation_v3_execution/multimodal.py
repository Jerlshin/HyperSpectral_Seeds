"""Archive readers, compact full-spectrum measurements and asset integrity helpers."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomic metadata publication; no completion marker survives a partial write."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_envi(z: zipfile.ZipFile, header: str) -> Array:
    """Read this archive's uint16 little-endian BIL cubes, validating every dimension.

    Only exact sibling .raw members are accepted: prefix matching can select JPEGs.
    No archive paths are extracted to disk. ZipFile verifies CRC on every read.
    """
    text = z.read(header).decode("ascii")

    def field(name: str) -> str:
        match = re.search(r"^" + re.escape(name) + r"\s*=\s*([^\r\n]+)", text, re.M | re.I)
        if match is None:
            raise ValueError(f"Missing ENVI field {name}")
        return match.group(1).strip()

    if (field("data type"), field("interleave").lower(), field("byte order")) != ("12", "bil", "0"):
        raise ValueError("Only uint16 little-endian BIL is supported")
    shape = (int(field("lines")), int(field("bands")), int(field("samples")))
    if any(n <= 0 for n in shape) or int(field("header offset")) != 0:
        raise ValueError("Invalid ENVI dimensions/offset")
    payload = z.read(str(Path(header).with_suffix(".raw")))
    if len(payload) != int(np.prod(shape)) * 2:
        raise ValueError("ENVI payload size mismatch")
    return np.frombuffer(payload, dtype="<u2").reshape(shape).transpose(0, 2, 1)


def spectral_summary(patch: Array, mask: Array) -> tuple[Array, Array, Array]:
    """Mean, SD, q10/q50/q90 plus 4x4 occupied-region means on exact resized patches.

    Region geometry is the historical aspect-preserving 64x64 patch, with empty
    regions explicit in occupancy. No fitted transform or label enters this step.
    """
    foreground = np.asarray(patch[:, mask > 0], dtype=np.float32)
    if foreground.shape[1] == 0 or not np.isfinite(foreground).all():
        raise ValueError("Empty or nonfinite foreground spectrum")
    summary = np.concatenate(
        [
            foreground.mean(1)[None],
            foreground.std(1)[None],
            np.quantile(foreground, [0.1, 0.5, 0.9], axis=1),
        ],
        axis=0,
    )
    tokens = np.zeros((16, patch.shape[0]), np.float32)
    occupancy = np.zeros(16, np.float32)
    for y in range(4):
        for x in range(4):
            m = mask[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16] > 0
            occupancy[y * 4 + x] = m.mean()
            if m.any():
                tokens[y * 4 + x] = patch[:, y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16][
                    :, m
                ].mean(1)
    return summary.astype(np.float32), tokens, occupancy
