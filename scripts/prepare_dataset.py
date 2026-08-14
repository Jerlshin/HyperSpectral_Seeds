#!/usr/bin/env python3
"""Build ``dataset/patches.npy`` + ``dataset/labels.npy`` from the Zenodo archive.

Thin CLI wrapper around :mod:`spectralquadnet.data.prep`. All the work lives
in the package (``download`` → ``build_patch_dataset``); this file only
parses arguments and assembles a
:class:`~spectralquadnet.data.prep.config.PrepConfig`, matching the pattern
``train.py`` uses for training.

Requires the ``prep`` extra (ENVI reading, OpenCV, scikit-image)::

    pip install -e ".[prep]"

Usage
─────
    python scripts/prepare_dataset.py                     # full pipeline
    python scripts/prepare_dataset.py --root ./data       # elsewhere
    python scripts/prepare_dataset.py --download-only     # fetch the zip, stop

The output is the **256-band** patch cube, and it is what the primary pipeline
trains on directly — ``configs/data/hsi256_grouped.yaml`` points at
``dataset/patches.npy`` and ``dataset/wavelengths.csv``. There is no reduction
step between this script and ``python train.py``.

``scripts/select_bands.py`` is the entry point to the retained **band-selection
ablation pathway** and is optional: run it only to produce the reduced arrays
ablation A2 compares against (see ``docs/07_BAND_SELECTION_PATHWAY.md``).

Six arrays are written, all row-aligned on the patch index:

==================  ============================  ==================================
``patches.npy``     ``(N, 256, 64, 64)`` float32  the cube itself
``labels.npy``      ``(N,)`` int64                variety index, 0…89
``groups.npy``      ``(N,)`` int64                acquisition-bundle id — P-1
``masks.npy``       ``(N, 64, 64)`` float16       the fill map alpha — P-3
``morphology.npy``  ``(N, 8)`` float32            size/shape descriptors — P-4
``gain.npy``        ``(N, 2, 64, 64)`` float32    per-pixel (mean, sd) along λ — P-2
==================  ============================  ==================================

``gain.npy`` is never a model input; it is what the leakage probe measures
acquisition-bundle identity from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectralquadnet.data.prep.config import DATA_URL, PrepConfig
from spectralquadnet.data.prep.download import download
from spectralquadnet.data.prep.patch_extraction import build_patch_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = PrepConfig()
    parser.add_argument(
        "--root", type=Path, default=defaults.root, help="Dataset directory to populate."
    )
    parser.add_argument("--data-url", default=DATA_URL, help="Source archive URL.")
    parser.add_argument(
        "--patch-size", type=int, default=defaults.patch_size, help="Output patch edge, in pixels."
    )
    parser.add_argument(
        "--num-bands", type=int, default=defaults.num_bands, help="Bands per extracted patch."
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Fetch the archive and stop, skipping segmentation and patch extraction.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = PrepConfig(
        root=args.root,
        data_url=args.data_url,
        patch_size=args.patch_size,
        num_bands=args.num_bands,
    )
    if args.download_only:
        cfg.ensure_root()
        download(cfg)
        return
    # `build_patch_dataset` calls `download` itself.
    build_patch_dataset(cfg)


if __name__ == "__main__":
    main()
