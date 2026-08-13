"""Feature extraction and the split boundaries every later stage is fenced by.

Two responsibilities, both of which exist to be done exactly once:

**The feature cache.** Reducing 8,624 × 256 × 64 × 64 float32 to per-patch
spectra is a 36 GB read. Doing it per method, per budget, per replicate would
dominate the study's runtime and produce identical numbers every time, so it
happens once and is memory-mapped thereafter. The cache key includes the
feature definition, so changing ``features`` writes a new file rather than
silently reusing the old one's contents under a new name.

**The split.** :func:`fold_splits` calls the *same* builder the training runs
use — :func:`spectralquadnet.data.loaders.grouped_split` — with the same
parameters, so the rows a selector may see are exactly the rows a training run
would put a gradient through. Approximating the partition here would leave the
band study and the runs it advises describing different data.

The one rule the rest of the package inherits
─────────────────────────────────────────────
:class:`FoldData` exposes ``train``, ``calib`` and ``heldout`` as separate
attributes and never a concatenation of them. Selection reads ``train``.
Decisions read ``calib``. ``heldout`` is reachable only through
:meth:`FoldData.reveal_heldout`, which logs the call. That is deliberate
friction: the single most consequential defect the project's audit found was a
selection that had quietly seen the split it was scored on, and an interface
where the wrong rows are one attribute access away invites its repeat.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from spectralquadnet.bandstudy.config import BandStudyConfig

_log = logging.getLogger("spectralquadnet.bandstudy.data")

#: Background threshold, identical to the one the model's ``foreground_mask``
#: and ``experiments.baselines`` use, so the study averages over the same
#: pixels the network sees.
FOREGROUND_EPS: float = 1e-5


# ══════════════════════════════════════════════════════════════════════
#  Feature extraction
# ══════════════════════════════════════════════════════════════════════


def _cache_key(cfg: BandStudyConfig, n_rows: int, n_bands: int) -> str:
    payload = {
        "patches": str(Path(cfg.patches_path).resolve()),
        "features": cfg.features,
        "rows": int(n_rows),
        "bands": int(n_bands),
        "eps": FOREGROUND_EPS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def extract_features(
    cfg: BandStudyConfig, progress: Any | None = None
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """``(spectra, features)`` for every patch, cached to disk.

    ``spectra`` is always the ``(N, C)`` foreground-masked mean spectrum: the
    band-selection methods operate on it regardless of what the proxy models
    are fed, because "which bands" is a question about bands and a selector
    handed a 512-column mean⊕sd matrix would be choosing among 512 things.

    ``features`` is what the proxies read — the same array under
    ``features="mean"``, or ``(N, 2C)`` mean ⊕ per-band spatial sd under
    ``features="mean_sd"``.

    The masked mean, rather than a plain spatial mean: the patches are exact
    zero outside the kernel, so including the padding would scale every
    spectrum by that patch's fill fraction and reintroduce a size-dependent
    gain the preprocessing removed.

    Raises:
        FileNotFoundError: The patch cube is not where the config says.
    """
    path = Path(cfg.patches_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. The band study reads the FULL 256-band cube written by "
            "`python scripts/prepare_dataset.py`; a pre-reduced cube can only answer questions "
            "about bands somebody already chose."
        )

    patches = np.load(path, mmap_mode="r")
    n, c = int(patches.shape[0]), int(patches.shape[1])
    key = _cache_key(cfg, n, c)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    mean_path = cfg.cache_dir / f"mean_spectra_{key}.npy"
    sd_path = cfg.cache_dir / f"sd_spectra_{key}.npy"
    meta_path = cfg.cache_dir / f"features_{key}.json"

    want_sd = cfg.features == "mean_sd"
    have = mean_path.exists() and (sd_path.exists() or not want_sd)
    if have and not cfg.force:
        _log.info("features: reusing cache %s", mean_path.name)
        mean = np.load(mean_path, mmap_mode="r")
        sd = np.load(sd_path, mmap_mode="r") if want_sd else None
        return np.asarray(mean, dtype=np.float32), _assemble(mean, sd)

    _log.info(
        "features: computing masked mean%s over %d patches × %d bands",
        " and sd" if want_sd else "",
        n,
        c,
    )
    mean = np.zeros((n, c), dtype=np.float32)
    sd = np.zeros((n, c), dtype=np.float32) if want_sd else None

    tracker = progress(range(0, n, cfg.chunk_size)) if progress else range(0, n, cfg.chunk_size)
    for start in tracker:
        stop = min(start + cfg.chunk_size, n)
        block = np.asarray(patches[start:stop], dtype=np.float32)
        flat = block.reshape(len(block), c, -1)
        mask = (np.abs(flat).sum(axis=1, keepdims=True) > FOREGROUND_EPS).astype(np.float32)
        valid = mask.sum(axis=2).clip(min=1.0)
        mu = (flat * mask).sum(axis=2) / valid
        mean[start:stop] = mu
        if sd is not None:
            # Masked second moment, so the sd describes the seed rather than
            # the seed plus a variable amount of exact-zero background.
            sq = ((flat**2) * mask).sum(axis=2) / valid
            sd[start:stop] = np.sqrt(np.clip(sq - mu**2, 0.0, None))

    np.save(mean_path, mean)
    if sd is not None:
        np.save(sd_path, sd)
    meta_path.write_text(
        json.dumps(
            {
                "patches_path": str(path.resolve()),
                "n_patches": n,
                "n_bands": c,
                "features": cfg.features,
                "foreground_eps": FOREGROUND_EPS,
                "key": key,
            },
            indent=2,
        )
    )
    _log.info("features: wrote %s", mean_path.name)
    return mean, _assemble(mean, sd)


def _assemble(mean: npt.NDArray[Any], sd: npt.NDArray[Any] | None) -> npt.NDArray[np.float32]:
    """Concatenate mean and sd into the proxy feature matrix."""
    if sd is None:
        return np.asarray(mean, dtype=np.float32)
    return np.concatenate(
        [np.asarray(mean, dtype=np.float32), np.asarray(sd, dtype=np.float32)], axis=1
    )


def feature_columns(bands: npt.NDArray[Any], n_bands_total: int, features: str) -> npt.NDArray[Any]:
    """Which columns of the proxy feature matrix a band subset selects.

    Under ``mean`` the answer is the band indices themselves. Under ``mean_sd``
    each band owns two columns — its mean at ``b`` and its sd at ``b + C`` —
    and taking only the first would evaluate a k-band subset on half the
    features the representation defines, which is a different experiment.
    """
    bands = np.asarray(bands, dtype=np.int64)
    if features == "mean":
        return bands
    return np.concatenate([bands, bands + int(n_bands_total)])


def load_wavelengths(cfg: BandStudyConfig, n_bands: int) -> npt.NDArray[np.float64]:
    """The physical wavelength of each stored band, in nm.

    Every band this study reports is reported in nanometres as well as by
    index, because an index is a fact about one array's layout and a wavelength
    is a fact about rice.

    Raises:
        ValueError: The CSV's row count disagrees with the cube's band count.
    """
    frame = pd.read_csv(cfg.wavelength_path, sep=None, engine="python")
    values = np.asarray(frame.iloc[:, -1].values, dtype=np.float64)
    if len(values) != n_bands:
        raise ValueError(
            f"{cfg.wavelength_path} has {len(values)} rows but the cube has {n_bands} bands. "
            "The wavelength CSV must describe the SAME cube the study reads."
        )
    return values


def load_morphology(cfg: BandStudyConfig, n_rows: int) -> npt.NDArray[np.float32] | None:
    """``(N, 8)`` morphometrics, or ``None`` when unavailable or disabled."""
    if not cfg.use_morphology:
        return None
    path = Path(cfg.morphology_path)
    if not path.exists():
        _log.warning("use_morphology is on but %s does not exist — continuing without it", path)
        return None
    morph = np.asarray(np.load(path), dtype=np.float32)
    if len(morph) != n_rows:
        raise ValueError(f"{path} has {len(morph)} rows but the cube has {n_rows}")
    return morph


# ══════════════════════════════════════════════════════════════════════
#  Splits
# ══════════════════════════════════════════════════════════════════════


@dataclass
class FoldData:
    """One fold's row indices, fenced by what each split is allowed to decide.

    Attributes:
        train: Rows a **selector** may see. Nothing else may.
        calib: Rows a **decision** may see — which k, which method. Held out
            from training, so a score on it is not the score of the data the
            selector fitted on; not the reported split, so reading it does not
            spend the held-out evidence.
        heldout: ``val ∪ test``. Reachable only via :meth:`reveal_heldout`.
    """

    fold: int
    train: npt.NDArray[np.int64]
    calib: npt.NDArray[np.int64]
    _heldout: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]
    groups: npt.NDArray[np.int64] | None
    report: dict[str, Any] = field(default_factory=dict)

    def reveal_heldout(self, reason: str) -> npt.NDArray[np.int64]:
        """``val ∪ test``, with the reason logged.

        There is exactly one legitimate reason — the ``confirm`` stage scoring
        an already-chosen configuration once — and the log line is what lets a
        reviewer check that no other stage called this.
        """
        _log.warning("HELD-OUT SPLIT REVEALED (fold %d): %s", self.fold, reason)
        return self._heldout

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "calib": int(self.calib.size),
            "heldout": int(self._heldout.size),
        }

    def summary(self) -> str:
        s = self.sizes
        return (
            f"fold {self.fold}: train {s['train']:,} (selection)  "
            f"calib {s['calib']:,} (decisions)  heldout {s['heldout']:,} (confirm only)"
        )


def fold_splits(cfg: BandStudyConfig, fold: int) -> FoldData:
    """Build one fold with the training pipeline's own split builder.

    Raises:
        FileNotFoundError: ``groups.npy`` is missing under the grouped scheme.
        ValueError: The split builder refuses — e.g. ``single_group_policy`` is
            ``error`` and a class was captured in one scan.
    """
    labels = np.asarray(np.load(cfg.labels_path)).astype(np.int64)

    if cfg.split_scheme == "stratified":
        from spectralquadnet.data.loaders import _stratified_split

        bundle = _stratified_split(labels, cfg.split_eval_frac, cfg.calib_frac, None)
        groups = None
    else:
        groups_path = Path(cfg.groups_path)
        if not groups_path.exists():
            raise FileNotFoundError(
                f"{groups_path} does not exist, so no acquisition-disjoint split can be built "
                "and the study would silently answer a question about trays. It is written by "
                "`python scripts/prepare_dataset.py`."
            )
        from spectralquadnet.data.loaders import grouped_split

        groups = np.asarray(np.load(groups_path)).astype(np.int64)
        bundle = grouped_split(
            labels,
            groups,
            eval_frac=cfg.split_eval_frac,
            calib_frac=cfg.calib_frac,
            fold=fold,
            single_group_policy=cfg.single_group_policy,
        )

    return FoldData(
        fold=fold,
        train=np.asarray(bundle.train, dtype=np.int64),
        calib=np.asarray(bundle.calib, dtype=np.int64),
        _heldout=np.sort(np.concatenate([bundle.val, bundle.test])).astype(np.int64),
        labels=labels,
        groups=groups,
        report=bundle.report.as_dict(),
    )


def replicate_rows(
    rows: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
    frac: float,
    seed: int,
) -> npt.NDArray[np.int64]:
    """A class-stratified subsample of ``rows`` — one replicate's view.

    Stratified rather than a plain bootstrap: with 90 classes and ~40 training
    patches each, an unstratified draw loses whole classes, and a selector that
    never saw class 57 is not a replicate of one that did — it is a selector
    for an 89-class problem.

    ``frac >= 1.0`` returns ``rows`` unchanged, which makes ``replicates=1`` a
    single deterministic selection on the full training split rather than a
    subsample of unknown provenance.
    """
    if frac >= 1.0:
        return np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    y = labels[rows]
    keep: list[npt.NDArray[Any]] = []
    for c in np.unique(y):
        pool = rows[y == c]
        take = max(2, int(round(len(pool) * frac)))
        take = min(take, len(pool))
        keep.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(keep)).astype(np.int64)
