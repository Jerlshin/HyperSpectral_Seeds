#!/usr/bin/env python3
"""Reduce the 256-band cube to an SPA/mRMR-selected band subset.

**Not part of the primary pipeline.** The study's primary path trains on the
complete 256-band cube ``scripts/prepare_dataset.py`` writes; nothing runs
between the two. This script is the build step for the retained **band-selection
ablation pathway** — ablation A2 and the arms in
``configs/data/ablation/`` — and running it changes nothing about a default
``python train.py``. See ``docs/07_BAND_SELECTION_PATHWAY.md``.

Prefer ``python -m spectralquadnet.bandstudy.cli`` for new work: it sweeps
twelve selection methods (including *evenly spaced* and *random* nulls) across
twenty budgets up to the full 256, and its neural confirmation arms slice the
full cube through ``data.band_indices_path`` instead of materialising one
reduced copy per cell. This script exists because the shipped 40-band arrays
were produced by it and the A2 control arm has to be reproducible.

Thin CLI wrapper around
:func:`spectralquadnet.data.prep.band_selection.select_bands` — this file
only parses arguments into a
:class:`~spectralquadnet.data.prep.config.BandSelectionConfig`; the pipeline
itself (decorrelation pre-filter → mRMR → SPA → cross-validated elbow → save)
lives in that module.

Requires the ``prep`` extra::

    pip install -e ".[prep]"

Usage
─────
    python scripts/select_bands.py                       # whole corpus (LEAKY)
    python scripts/select_bands.py --fold 0              # inside fold 0
    python scripts/select_bands.py --per-fold            # every protocol fold
    python scripts/select_bands.py --n-candidates 5 10 20 40
    python scripts/select_bands.py --patches-path ./dataset/patches.npy --seed 0

IC-4 / CHANGES §4.1 — where the selection sits relative to the split
────────────────────────────────────────────────────────────────────
Without ``--fold``, mRMR relevance is ``mutual_info_classif(X, y)`` over **every
patch**, including the ones that become test. The 40 selected bands are a
hyperparameter of the input representation, chosen with test labels in scope.
Feature selection outside the resampling loop is a known and quantified source
of optimism (Ambroise & McLachlan, PNAS 99(10):6562-6566, 2002 — who obtain
near-zero apparent error on *random labels* that way), and it is genuine label
leakage independent of the split protocol: it contaminates ``grouped`` too.

``--fold k`` restricts every step — the correlation pre-filter, the FDR
diagnostic, mRMR, SPA and the cross-validated curve — to fold ``k``'s training
patches, and writes to ``dataset/folds/patches_fold<k>_40b.npy``. The reduced
cube still covers every row, because the model has to score eval patches; what
changes is whose labels chose the bands.

``--per-fold`` runs both protocol folds, which is what ablation A2 compares
against the whole-corpus arm.

Produces the reduced patch array and wavelength CSV an **ablation** data config
points at: ``configs/data/ablation/spa40_*.yaml`` for the flat names, and A2's
``spa40_within_fold`` arm for the per-fold ones. No config on the primary path
reads either.
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

    protocol = parser.add_argument_group("protocol (IC-4)")
    protocol.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Restrict the selection to this fold's TRAINING patches. Omitted, the "
            "selection sees every patch's label including the eval ones — which is "
            "leakage, and is kept only as ablation A2's control arm."
        ),
    )
    protocol.add_argument(
        "--per-fold",
        action="store_true",
        help="Run --fold for every fold in --folds, writing one band set each.",
    )
    protocol.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    protocol.add_argument("--groups-path", default=d.groups_path)
    protocol.add_argument(
        "--split-scheme", default=d.split_scheme, choices=["grouped", "stratified"]
    )
    protocol.add_argument("--split-eval-frac", type=float, default=d.split_eval_frac)
    protocol.add_argument("--calib-frac", type=float, default=d.calib_frac)

    parser.add_argument("--chunk-size", type=int, default=d.chunk_size)
    parser.add_argument("--seed", type=int, default=d.seed)
    return parser.parse_args(argv)


def _config_for(args: argparse.Namespace, fold: int | None) -> BandSelectionConfig:
    return BandSelectionConfig(
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
        fold=fold,
        groups_path=args.groups_path,
        split_scheme=args.split_scheme,
        split_eval_frac=args.split_eval_frac,
        calib_frac=args.calib_frac,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    folds: list[int | None] = list(args.folds) if args.per_fold else [args.fold]
    for fold in folds:
        if len(folds) > 1:
            print(f"\n{'#' * 70}\n#  FOLD {fold}\n{'#' * 70}")
        select_bands(_config_for(args, fold))


if __name__ == "__main__":
    main()
