#!/usr/bin/env python3
"""Write the finalist band sets — evenly spaced, >= 430 nm — for neural training.

Thin CLI over :mod:`spectralquadnet.bandstudy.finalists`. Reads only the
wavelength axis, so it needs neither the patch cube nor a GPU. Its output is
valid only for a cube with that same axis: the reflectance cube drops the bands
the white tile could not measure, so it gets its own sets::

    python scripts/write_finalist_bands.py \\
        --wavelengths dataset_reflectance/wavelengths.csv \\
        --out-dir outputs/band_finalists_reflectance

Usage
─────
    python scripts/write_finalist_bands.py                         # k = 16 24 32 48 64
    python scripts/write_finalist_bands.py --budgets 24 32 --out-dir outputs/bands_x

Then train a finalist by appending its overrides (printed below, and recorded in
``<out-dir>/manifest.json``), e.g.::

    python train.py data.band_indices_path=outputs/band_finalists/uniform430_k32.npy \\
        data.wavelength_path=outputs/band_finalists/uniform430_k32_wavelengths.csv \\
        data.num_bands=32 data.cutmix_bands=6 data.max_cutout_bands=2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from spectralquadnet.bandstudy.finalists import (
    FINALIST_BUDGETS,
    LOWER_BOUND_NM,
    read_wavelengths,
    write_finalists,
)
from spectralquadnet.data.datasets import _band_selection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wavelengths", type=Path, default=Path("./dataset/wavelengths.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/band_finalists"))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(FINALIST_BUDGETS))
    parser.add_argument("--lower-nm", type=float, default=LOWER_BOUND_NM)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = write_finalists(args.wavelengths, args.out_dir, tuple(args.budgets), args.lower_nm)
    for name, entry in manifest["sets"].items():
        # Read each file back through the training path's own validators, so a
        # file that would fail at startup fails here instead.
        _band_selection(
            SimpleNamespace(band_indices_path=entry["band_indices_path"], num_bands=entry["k"])
        )
        assert read_wavelengths(entry["wavelength_path"]).size == entry["k"]
        print(
            f"{name}: k={entry['k']}  {entry['min_nm']:.1f}-{entry['max_nm']:.1f} nm  "
            f"spacing {entry['mean_spacing_nm']:.1f} nm"
        )
        print("   " + " ".join(entry["overrides"]))
    print(f"manifest → {Path(args.out_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
