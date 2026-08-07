#!/usr/bin/env python3
"""Reduce the 256-band cube to the SPA/mRMR-selected band subset.

Thin CLI wrapper around
:func:`spectralquadnet.data.prep.band_selection.select_bands` —
REFACTOR_PLAN.md §2.1, replacing the root ``band_selection.py`` script. The
pipeline itself (decorrelation pre-filter → mRMR → SPA → cross-validated elbow →
save) is unchanged; this file only parses arguments into a
:class:`~spectralquadnet.data.prep.config.BandSelectionConfig`.

Requires the ``prep`` extra::

    pip install -e ".[prep]"

Usage
─────
    python scripts/select_bands.py
    python scripts/select_bands.py --n-candidates 5 10 20 40
    python scripts/select_bands.py --patches-path ./dataset/patches.npy --seed 0

This is what produced ``dataset/patches_spa_40b.npy`` and
``dataset/wavelengths_spa_40b.csv``, the two files
``configs/data/spa40_90class.yaml`` points at.
"""

from __future__ import annotations

import argparse

from spectralquadnet.data.prep.band_selection import select_bands
from spectralquadnet.data.prep.config import BandSelectionConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = BandSelectionConfig()

    paths = parser.add_argument_group("paths")
    paths.add_argument("--patches-path", default=d.patches_path)
    paths.add_argument("--labels-path", default=d.labels_path)
    paths.add_argument("--wavelength-path", default=d.wavelength_path)
    paths.add_argument("--output-dir", default=d.output_dir)

    selection = parser.add_argument_group("selection")
    selection.add_argument(
        "--corr-threshold",
        type=float,
        default=d.corr_threshold,
        help="Drop a band if |r| exceeds this against any already-kept band.",
    )
    selection.add_argument("--n-select-max", type=int, default=d.n_select_max)
    selection.add_argument(
        "--n-candidates",
        type=int,
        nargs="+",
        default=d.n_candidates,
        help="Band counts to cross-validate when locating the accuracy elbow.",
    )
    selection.add_argument("--mi-neighbors", type=int, default=d.mi_neighbors)

    validation = parser.add_argument_group("validation")
    validation.add_argument("--cv-folds", type=int, default=d.cv_folds)
    validation.add_argument("--svc-C", dest="svc_C", type=float, default=d.svc_C)
    validation.add_argument(
        "--elbow-pct",
        type=float,
        default=d.elbow_pct,
        help="Fraction of peak accuracy that defines the elbow.",
    )

    parser.add_argument("--chunk-size", type=int, default=d.chunk_size)
    parser.add_argument("--seed", type=int, default=d.seed)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    select_bands(
        BandSelectionConfig(
            patches_path=args.patches_path,
            labels_path=args.labels_path,
            wavelength_path=args.wavelength_path,
            output_dir=args.output_dir,
            corr_threshold=args.corr_threshold,
            n_select_max=args.n_select_max,
            n_candidates=list(args.n_candidates),
            mi_neighbors=args.mi_neighbors,
            cv_folds=args.cv_folds,
            svc_C=args.svc_C,
            elbow_pct=args.elbow_pct,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
