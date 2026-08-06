#!/usr/bin/env python3
"""Build ``dataset/patches.npy`` + ``dataset/labels.npy`` from the Zenodo archive.

Thin CLI wrapper around :mod:`spectralquadnet.data.prep` — REFACTOR_PLAN.md §2.1,
replacing the root ``data_setup_v3.py`` script. All the work lives in the package
(``download`` → ``build_patch_dataset``); this file only parses arguments and
assembles a :class:`~spectralquadnet.data.prep.config.PrepConfig`, matching the
pattern ``train.py`` uses for training.

Requires the ``prep`` extra (ENVI reading, OpenCV, scikit-image)::

    pip install -e ".[prep]"

Usage
─────
    python scripts/prepare_dataset.py                     # full pipeline
    python scripts/prepare_dataset.py --root ./data       # elsewhere
    python scripts/prepare_dataset.py --download-only     # fetch the zip, stop

The output is the **256-band** patch cube. Run ``scripts/select_bands.py``
afterwards to produce the 40-band ``patches_spa_40b.npy`` the training config
actually reads.
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
    # `build_patch_dataset` calls `download` itself, exactly as the baseline's
    # `main()` did (data_setup_v3.py line 152).
    build_patch_dataset(cfg)


if __name__ == "__main__":
    main()
