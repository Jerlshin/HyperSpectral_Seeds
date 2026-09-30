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
    python scripts/prepare_dataset.py                     # ./dataset, exactly as trained on
    python scripts/prepare_dataset.py --probe-tiles 27    # measure tiles only, extract nothing
    python scripts/prepare_dataset.py --download-only     # fetch the zip, stop

The defaults build the dataset the configs read: **reflectance** from each
scan's in-scene Spectralon tile (``--radiometry tile``), with the bands no tile
could measure dropped from every scan (``--tile-drop-unresolved``; on this
archive the 41 bands 608.0-705.8 nm, clipped at 4095 DN in sessions 0-7). The
zip lands at ``<root>/rice_hsi.zip`` and can be deleted once the run is verified.
``--no-tile-drop-unresolved`` refuses on unresolved bands instead, and
``--radiometry snv`` builds the previous per-pixel-SNV radiance cube (a
different dataset: give it its own ``--root``).

The run writes ``white_tiles.csv`` and ``white_spectra.npz`` before extracting
any patch, and ``radiometry.json`` / ``wavelengths.csv`` at the end. An existing
``patches.npy`` is never replaced without ``--overwrite``.

The output is the **215-band** reflectance cube, and it is what the primary
pipeline trains on directly — ``configs/data/refl215_grouped.yaml`` points at
``dataset/patches.npy`` and ``dataset/wavelengths.csv``. There is no reduction
step between this script and ``python train.py``.

``scripts/select_bands.py`` is the entry point to the retained **band-selection
ablation pathway** and is optional: run it only to produce the reduced arrays
ablation A2 compares against (see ``docs/07_BAND_SELECTION_PATHWAY.md``).

Six arrays are written, all row-aligned on the patch index (``C`` = 215 here):

==================  ============================  ==================================
``patches.npy``     ``(N, C, 64, 64)`` float32    the cube itself
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

from spectralquadnet.data.prep.config import DATA_URL, PrepConfig, TileConfig
from spectralquadnet.data.prep.download import download
from spectralquadnet.data.prep.patch_extraction import build_patch_dataset, probe_white_tiles
from spectralquadnet.data.prep.radiometry import RADIOMETRY_MODES


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
    parser.add_argument(
        "--radiometry",
        choices=RADIOMETRY_MODES,
        default=defaults.radiometry,
        help="Radiometric domain. 'tile'/'tile_snv' divide each scene by its in-scene "
        "Spectralon tile (the default); 'snv' is the previous per-pixel SNV radiance cube; 'auto' never inspects a scene and resolves to 'snv' on this archive.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use this zip instead of <root>/rice_hsi.zip (downloaded there if absent).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing patches.npy in --root."
    )
    parser.add_argument(
        "--probe-tiles",
        type=int,
        default=0,
        metavar="N",
        help="Measure the white tile in N scans (spread across sessions), write "
        "white_tiles_probe.csv and stop. Extracts nothing.",
    )
    tile = TileConfig()
    parser.add_argument("--tile-saturation-dn", type=float, default=tile.saturation_dn)
    parser.add_argument("--tile-min-area", type=int, default=tile.min_area_px)
    parser.add_argument(
        "--tile-search-rows",
        type=int,
        nargs=2,
        default=None,
        metavar=("FIRST", "LAST"),
        help=f"Scan rows searched for the tile (default: {tile.search_rows[0]} to the end).",
    )
    parser.add_argument(
        "--tile-reflectance",
        type=float,
        default=tile.reference_reflectance,
        help="Certified reflectance of the panel.",
    )
    parser.add_argument(
        "--tile-drop-unresolved",
        action=argparse.BooleanOptionalAction,
        default=tile.drop_unresolved_bands,
        help="Drop, from every scan, the bands whose white level no tile in the session "
        "could measure; --no-tile-drop-unresolved refuses instead. The kept axis is "
        "written to wavelengths.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    defaults = TileConfig()
    tile = TileConfig(
        search_rows=(
            (int(args.tile_search_rows[0]), int(args.tile_search_rows[1]))
            if args.tile_search_rows
            else defaults.search_rows
        ),
        saturation_dn=args.tile_saturation_dn,
        min_area_px=args.tile_min_area,
        reference_reflectance=args.tile_reflectance,
        drop_unresolved_bands=args.tile_drop_unresolved,
    )
    cfg = PrepConfig(
        root=args.root,
        data_url=args.data_url,
        patch_size=args.patch_size,
        num_bands=args.num_bands,
        radiometry=args.radiometry,
        archive=args.archive,
        overwrite=args.overwrite,
        tile=tile,
    )
    if args.download_only:
        cfg.ensure_root()
        download(cfg)
        return
    if args.probe_tiles:
        probe_white_tiles(cfg, limit=args.probe_tiles)
        return
    # `build_patch_dataset` calls `download` itself.
    build_patch_dataset(cfg)


if __name__ == "__main__":
    main()
