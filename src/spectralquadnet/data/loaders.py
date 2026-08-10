"""Train/calib/val/test splits and DataLoader construction.

The ``random_state=42`` values in :func:`build_splits` are deliberately
hardcoded rather than sourced from ``cfg.seed``: trained checkpoints are
validated against the exact split those literals produce, so tying the split
to the run seed would risk silently re-partitioning the data whenever
``cfg.seed`` is overridden for an unrelated reason (e.g. varying model
initialisation across runs).

Two protocol items live here (IMPROVEMENT_PLAN §3.1).

**P-1 / T4-1 — grouped splits.** ``cfg.data.split_scheme = "grouped"`` reads
``groups.npy`` and holds out whole *scans*, enforcing

.. math::
    \\mathrm{scan}(i) \\in \\text{train}
    \\;\\Longrightarrow\\;
    \\mathrm{scan}(i) \\notin \\text{val} \\cup \\text{test}

which the patch-level ``stratified`` scheme does not: Phase 0's 0-H measured
**107 of 107 capture groups present in all three splits**, so there is no
val or test patch whose physical scan the model did not train on.

**P-5 / T4-5 — the calibration split.** ``cfg.data.calib_frac > 0`` carves an
inner ``calib`` split out of train. Per-class margins, CDWS weights and the
Phase-3 oversampling weights — 270+ fitted parameters — read *that*, so they
stop being fitted on the same split that selects the checkpoint (§2.1.4, C-9).

What is constructible here, and what is not
───────────────────────────────────────────
The plan's §4.2 criterion for T4-1 is "no ``scan_id`` appears in more than one
split". On **this** dataset that is unachievable for three splits at once, and
0-H is what says so: every variety was captured exactly twice, so a class has
at most two groups and can therefore appear in at most two of three splits.
Demanding full three-way group disjointness would leave 90-class macro-F1
undefined, because a class absent from test cannot be scored on it.

So the split builder enforces the contract the plan actually writes down —
train against ``val ∪ test`` — takes group disjointness wherever the group
counts allow it, and **reports every place it could not**:
:class:`SplitReport` names the classes whose val/test or calib boundary is
patch-level rather than group-level, and ``val_test_group_disjoint`` is
``False`` whenever that list is non-empty. Nothing is inferred and nothing is
silently mixed. Sweeping ``cfg.data.split_fold`` over ``0 … m-1`` is the
leave-one-scan-out cross-validation §3.1 names as the fallback for exactly
this case.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.morphometrics import standardise_morphometrics
from spectralquadnet.data.samplers import (
    ClassBalancedBatchSampler,
    DistributedBatchShardSampler,
    DistributedIndexShardSampler,
    HardClassOversampledSampler,
)
from spectralquadnet.utils.distributed import DistContext
from spectralquadnet.utils.warning_filters import silence_known_warnings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.data.mmap_store import DataStore
    from spectralquadnet.utils.device import RuntimePlan

#: Accepted values of ``cfg.data.split_scheme``.
SPLIT_SCHEMES: tuple[str, ...] = ("stratified", "grouped")

#: Random state for every split decision. A literal, for the reason in the
#: module docstring; the grouped path uses it to seed its own ``Generator`` so
#: the per-class group ordering is reproducible without touching the global
#: NumPy RNG (``train.py`` requires the split to consume no shared stream).
SPLIT_SEED: int = 42

#: Batch size every evaluation loader is built at. Was a literal repeated at
#: four call sites; evaluation never touches a gradient, so this is bounded by
#: activation memory alone and is not a hyperparameter.
EVAL_BATCH: int = 256

#: What :func:`grouped_split` does about a class with only one group. ``error``
#: refuses and names the classes — §3.1's "report the count explicitly … rather
#: than silently mixing". ``patch_split`` is the opt-in fallback: that class's
#: eval patches come from the same scan as its training patches, so its
#: contribution to the metric is leaky and the report says so.
SINGLE_GROUP_POLICIES: tuple[str, ...] = ("error", "patch_split")


# ══════════════════════════════════════════════════════════════════════
#  Split bundle and its report
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SplitReport:
    """What the split actually achieved, as opposed to what was requested.

    Every field exists because some claim about the protocol depends on it,
    and Phase 0 showed that claims about this split had been made from the
    split's *parameters* rather than from its result.
    """

    scheme: str
    fold: int
    n_patches: int
    n_classes: int
    n_groups: int
    sizes: dict[str, int]
    #: ``scan(train) ∩ scan(val ∪ test) = ∅`` — the §3.1 contract. False only
    #: under ``stratified`` or the ``patch_split`` fallback.
    train_eval_group_disjoint: bool
    #: Whether val and test additionally hold disjoint scans. Needs ≥ 2 eval
    #: groups per class, which this dataset does not have.
    val_test_group_disjoint: bool
    #: Whether calib holds scans train does not. Needs ≥ 2 train groups per class.
    calib_group_disjoint: bool
    groups_per_class_min: int = 0
    groups_per_class_max: int = 0
    #: Scans present in train **and** in val or test. 0 under a grouped split
    #: by construction; measured — not assumed — under ``stratified`` whenever
    #: ``groups.npy`` is available, which is 0-H's headline number computed at
    #: every startup instead of by an audit script.
    n_shared_groups_train_eval: int = 0
    classes_with_one_group: list[int] = field(default_factory=list)
    classes_sharing_groups_val_test: list[int] = field(default_factory=list)
    classes_sharing_groups_calib: list[int] = field(default_factory=list)
    #: Classes whose eval patches come from a scan that is also in train.
    #: Non-empty only under the ``patch_split`` fallback; these are the classes
    #: whose numbers remain leaky and must be reported as such.
    classes_leaking_into_eval: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for the run sidecar."""
        return {
            "scheme": self.scheme,
            "fold": self.fold,
            "n_patches": self.n_patches,
            "n_classes": self.n_classes,
            "n_groups": self.n_groups,
            "sizes": dict(self.sizes),
            "train_eval_group_disjoint": self.train_eval_group_disjoint,
            "val_test_group_disjoint": self.val_test_group_disjoint,
            "calib_group_disjoint": self.calib_group_disjoint,
            "groups_per_class_min": self.groups_per_class_min,
            "groups_per_class_max": self.groups_per_class_max,
            "n_shared_groups_train_eval": self.n_shared_groups_train_eval,
            "classes_with_one_group": list(self.classes_with_one_group),
            "classes_sharing_groups_val_test": list(self.classes_sharing_groups_val_test),
            "classes_sharing_groups_calib": list(self.classes_sharing_groups_calib),
            "classes_leaking_into_eval": list(self.classes_leaking_into_eval),
        }

    def summary(self) -> list[str]:
        """Human-readable lines for the startup banner."""
        total = max(self.n_patches, 1)
        sizes = "  ".join(f"{name}: {n:,} ({n / total:.0%})" for name, n in self.sizes.items() if n)
        lines = [f"Split: {self.scheme} (fold {self.fold})  {sizes}"]
        if self.scheme == "stratified":
            measured = (
                f"{self.n_shared_groups_train_eval} of {self.n_groups} scans are in train "
                "and in val/test"
                if self.n_groups
                else "groups.npy not available, so the leak is unmeasured"
            )
            lines.append(
                f"  ⚠ patch-level split — {measured} (C-1). "
                "Set data.split_scheme=grouped once groups.npy exists."
            )
            return lines
        lines.append(
            f"  {self.n_groups} scans, {self.groups_per_class_min}–"
            f"{self.groups_per_class_max} per class  |  "
            f"train↔val∪test group-disjoint: {self.train_eval_group_disjoint}  "
            f"val↔test: {self.val_test_group_disjoint}  "
            f"calib↔train: {self.calib_group_disjoint}"
        )
        if self.classes_sharing_groups_val_test:
            lines.append(
                f"  {len(self.classes_sharing_groups_val_test)} classes have one eval scan, "
                "so their val/test boundary is patch-level (val and test are both held out "
                "from train, so the headline contract still holds)."
            )
        if self.classes_leaking_into_eval:
            lines.append(
                f"  ⚠ {len(self.classes_leaking_into_eval)} classes have a single scan and "
                "were split at patch level: their eval patches share a scan with train."
            )
        return lines


@dataclass(frozen=True)
class Splits:
    """The four index arrays, the labels and groups they came from, and the report."""

    labels: npt.NDArray[Any]
    train: npt.NDArray[Any]
    calib: npt.NDArray[Any]
    val: npt.NDArray[Any]
    test: npt.NDArray[Any]
    groups: npt.NDArray[Any] | None
    report: SplitReport

    @property
    def has_calib(self) -> bool:
        """Whether a calibration split was carved (P-5 / T4-5)."""
        return self.calib.size > 0


# ══════════════════════════════════════════════════════════════════════
#  P-1 / T4-1 — the grouped split
# ══════════════════════════════════════════════════════════════════════


def _halve(idx: npt.NDArray[Any], rng: np.random.Generator) -> tuple[Any, Any]:
    """Shuffle and split in two, the larger half first. Used inside one class."""
    shuffled = rng.permutation(idx)
    cut = len(shuffled) - len(shuffled) // 2
    return shuffled[:cut], shuffled[cut:]


def grouped_split(
    labels: npt.NDArray[Any],
    groups: npt.NDArray[Any],
    *,
    eval_frac: float = 0.30,
    calib_frac: float = 0.0,
    fold: int = 0,
    seed: int = SPLIT_SEED,
    single_group_policy: str = "error",
) -> Splits:
    """Hold out whole scans per class, then carve val/test and calib.

    Works one class at a time, so every class is present in every split even
    when the classes have different group counts:

    1. order the class's groups deterministically and rotate by ``fold`` —
       sweeping ``fold`` over ``0 … m-1`` is leave-one-scan-out CV;
    2. take ``max(1, round(m * eval_frac))`` groups (never all of them) as the
       class's **eval** groups; the rest are its train pool;
    3. split the eval groups into val and test **by group** when there are two
       or more, and by patch when there is only one;
    4. carve ``calib_frac`` out of the train pool, by group when the pool has
       two or more and by patch otherwise.

    Step 2 is what guarantees the §3.1 contract regardless of what steps 3–4
    can manage: an eval group is never a train group, so no scan crosses the
    train ↔ ``val ∪ test`` boundary.

    Args:
        labels: ``(N,)`` class index per patch.
        groups: ``(N,)`` scan id per patch (``groups.npy``).
        eval_frac: Target share of a class's *groups* held out for evaluation.
        calib_frac: Share of the training pool reserved for calibration.
        fold: Rotates which groups are held out; ``0 … m-1`` enumerates the
            leave-one-scan-out folds.
        seed: Seeds the per-class group ordering and the patch-level fallbacks.
        single_group_policy: One of :data:`SINGLE_GROUP_POLICIES`.

    Raises:
        ValueError: ``labels`` and ``groups`` disagree in length, ``fold`` is
            negative, ``single_group_policy`` is unknown, or the policy is
            ``error`` and some class has a single group — in which case the
            message names those classes and their counts.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.shape != groups.shape:
        raise ValueError(f"labels {labels.shape} and groups {groups.shape} must be the same shape")
    if fold < 0:
        raise ValueError(f"split_fold must be >= 0, got {fold}")
    if single_group_policy not in SINGLE_GROUP_POLICIES:
        raise ValueError(
            f"single_group_policy must be one of {SINGLE_GROUP_POLICIES}, "
            f"got {single_group_policy!r}"
        )

    classes = np.unique(labels)
    counts = {int(c): int(np.unique(groups[labels == c]).size) for c in classes}
    singles = sorted(c for c, n in counts.items() if n < 2)

    if singles and single_group_policy == "error":
        shown = ", ".join(str(c) for c in singles[:10])
        raise ValueError(
            f"{len(singles)} of {len(classes)} classes were captured in a single scan "
            f"(classes {shown}{' …' if len(singles) > 10 else ''}), so no group-disjoint "
            "split exists for them. IMPROVEMENT_PLAN §3.1 P-1: report the count and fall "
            "back to leave-one-scan-out CV rather than silently mixing. Sweep "
            "`data.split_fold`, or pass single_group_policy='patch_split' to accept a "
            "patch-level split for those classes with the leak recorded in the report."
        )

    rng = np.random.default_rng(seed)
    train: list[Any] = []
    calib: list[Any] = []
    val: list[Any] = []
    test: list[Any] = []
    shared_vt: list[int] = []
    shared_cal: list[int] = []
    leaking: list[int] = []

    for c in classes:
        idx_c = np.flatnonzero(labels == c)
        groups_c = groups[idx_c]
        uniq = np.unique(groups_c)
        m = len(uniq)

        if m < 2:
            # `error` already raised; this is the declared fallback.
            leaking.append(int(c))
            shared_vt.append(int(c))
            train_idx, eval_idx = _halve(idx_c, rng)
            va_idx, te_idx = _halve(eval_idx, rng)
        else:
            order = np.roll(uniq[rng.permutation(m)], -(fold % m))
            n_eval = min(max(1, int(round(m * eval_frac))), m - 1)
            eval_groups, train_groups = order[:n_eval], order[n_eval:]

            eval_idx = idx_c[np.isin(groups_c, eval_groups)]
            train_idx = idx_c[np.isin(groups_c, train_groups)]

            if n_eval >= 2:
                half = n_eval - n_eval // 2
                va_idx = idx_c[np.isin(groups_c, eval_groups[:half])]
                te_idx = idx_c[np.isin(groups_c, eval_groups[half:])]
            else:
                shared_vt.append(int(c))
                va_idx, te_idx = _halve(eval_idx, rng)

        val.append(va_idx)
        test.append(te_idx)

        if calib_frac <= 0.0 or train_idx.size == 0:
            train.append(train_idx)
            continue

        train_groups_present = np.unique(groups[train_idx])
        if train_groups_present.size >= 2:
            n_cal = min(
                max(1, int(round(train_groups_present.size * calib_frac))),
                train_groups_present.size - 1,
            )
            cal_groups = train_groups_present[:n_cal]
            in_cal = np.isin(groups[train_idx], cal_groups)
            calib.append(train_idx[in_cal])
            train.append(train_idx[~in_cal])
        else:
            shared_cal.append(int(c))
            shuffled = rng.permutation(train_idx)
            n_cal = min(max(1, int(round(len(shuffled) * calib_frac))), len(shuffled) - 1)
            calib.append(shuffled[:n_cal])
            train.append(shuffled[n_cal:])

    parts = {
        "train": _concat(train),
        "calib": _concat(calib),
        "val": _concat(val),
        "test": _concat(test),
    }
    _assert_partition(parts, len(labels))

    report = SplitReport(
        scheme="grouped",
        fold=fold,
        n_patches=int(len(labels)),
        n_classes=int(len(classes)),
        n_groups=int(np.unique(groups).size),
        sizes={k: int(v.size) for k, v in parts.items()},
        train_eval_group_disjoint=not leaking,
        val_test_group_disjoint=not shared_vt,
        calib_group_disjoint=bool(calib_frac > 0) and not shared_cal,
        groups_per_class_min=min(counts.values()) if counts else 0,
        groups_per_class_max=max(counts.values()) if counts else 0,
        classes_with_one_group=singles,
        classes_sharing_groups_val_test=sorted(shared_vt),
        classes_sharing_groups_calib=sorted(shared_cal),
        classes_leaking_into_eval=sorted(leaking),
    )
    return Splits(
        labels=labels,
        train=parts["train"],
        calib=parts["calib"],
        val=parts["val"],
        test=parts["test"],
        groups=groups,
        report=report,
    )


def _concat(parts: list[Any]) -> npt.NDArray[Any]:
    """Concatenate per-class index arrays into one sorted array."""
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(parts)).astype(np.int64)


def _assert_partition(parts: dict[str, npt.NDArray[Any]], n: int) -> None:
    """Fail loudly if the four splits are not a partition of ``0 … n-1``.

    Cheap (one sort of N indices) against the cost of the failure it catches:
    a duplicated index is a patch that is trained on *and* tested on, which is
    the exact defect this tier exists to remove.
    """
    everything = np.concatenate([v for v in parts.values() if v.size])
    if everything.size != n or np.unique(everything).size != n:
        counts = {k: int(v.size) for k, v in parts.items()}
        raise AssertionError(
            f"splits are not a partition of {n} patches: sizes {counts}, "
            f"{everything.size} assigned, {np.unique(everything).size} distinct"
        )


# ══════════════════════════════════════════════════════════════════════
#  Entry points
# ══════════════════════════════════════════════════════════════════════


def _stratified_split(
    labels: npt.NDArray[Any],
    eval_frac: float,
    calib_frac: float,
    groups: npt.NDArray[Any] | None = None,
) -> Splits:
    """The pre-Tier-4 patch-level split, with an optional calib carve-out.

    Bit-identical to the original at ``eval_frac=0.30, calib_frac=0.0``:
    the same two ``train_test_split`` calls, in the same order, at the same
    ``random_state``, and the returned arrays are left in the **order
    ``train_test_split`` produced** rather than sorted. That order is
    load-bearing — ``DataLoader(shuffle=True)`` permutes positions, so
    re-ordering ``indices`` would re-order the training stream at a fixed seed
    and move the Stage-1 golden loss. It is kept because every archived
    checkpoint was trained and selected on this split, so reproducing their
    numbers requires reproducing it — not because it is a defensible protocol.
    It is not (C-1).
    """
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=eval_frac, stratify=labels, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)

    cal = np.empty(0, dtype=np.int64)
    if calib_frac > 0.0:
        tr, cal = train_test_split(tr, test_size=calib_frac, stratify=labels[tr], random_state=42)

    parts = {"train": tr, "calib": cal, "val": va, "test": te}
    _assert_partition(parts, len(labels))

    # When the scan ids exist, measure the leak rather than merely asserting it.
    n_groups = n_shared = 0
    if groups is not None:
        groups = np.asarray(groups)
        n_groups = int(np.unique(groups).size)
        train_scans = set(np.unique(groups[parts["train"]]).tolist())
        eval_scans = set(np.unique(groups[np.concatenate([parts["val"], parts["test"]])]).tolist())
        n_shared = len(train_scans & eval_scans)

    report = SplitReport(
        scheme="stratified",
        fold=0,
        n_patches=int(len(labels)),
        n_classes=int(np.unique(labels).size),
        n_groups=n_groups,
        sizes={k: int(v.size) for k, v in parts.items()},
        train_eval_group_disjoint=groups is not None and n_shared == 0,
        val_test_group_disjoint=False,
        calib_group_disjoint=False,
        n_shared_groups_train_eval=n_shared,
    )
    return Splits(
        labels=labels,
        train=parts["train"],
        calib=parts["calib"],
        val=parts["val"],
        test=parts["test"],
        groups=groups,
        report=report,
    )


def build_split_bundle(cfg: ExperimentConfig | Any) -> Splits:
    """Build the configured split and return all four index arrays with a report.

    ``cfg.data.split_scheme`` selects the protocol:

    * ``stratified`` — the pre-Tier-4 patch-level split (C-1 intact);
    * ``grouped`` — P-1's scan-disjoint split, which needs ``groups.npy``.

    ``cfg.data.calib_frac`` carves P-5's calibration split out of train under
    either scheme.

    Raises:
        ValueError: Unknown scheme, a non-zero ``split_fold`` under a scheme
            that has no groups to rotate, or ``groups.npy`` missing.
    """
    labels = np.load(cfg.data.labels_path)
    scheme = str(getattr(cfg.data, "split_scheme", "stratified"))
    eval_frac = float(getattr(cfg.data, "split_eval_frac", 0.30))
    calib_frac = float(getattr(cfg.data, "calib_frac", 0.0))
    fold = int(getattr(cfg.data, "split_fold", 0))

    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"data.split_scheme must be one of {SPLIT_SCHEMES}, got {scheme!r}")

    groups_path = Path(str(getattr(cfg.data, "groups_path", "")))
    groups = None
    if groups_path.name and groups_path.exists():
        groups = np.load(groups_path)
        if len(groups) != len(labels):
            raise ValueError(
                f"{groups_path} has {len(groups)} rows but labels.npy has {len(labels)} — "
                "they must be row-aligned; re-run scripts/prepare_dataset.py."
            )

    if scheme == "stratified":
        if fold != 0:
            raise ValueError(
                "data.split_fold only has meaning for a grouped split — there is nothing "
                "to rotate without groups. Set data.split_scheme=grouped."
            )
        # `groups` is optional here: it is not used to *build* the split, only
        # to measure how much of C-1 that split carries.
        return _stratified_split(labels, eval_frac, calib_frac, groups)

    if groups is None:
        raise ValueError(
            f"data.split_scheme=grouped needs the scan ids at data.groups_path "
            f"({groups_path or '<unset>'}), and that file does not exist. It is written by "
            "`python scripts/prepare_dataset.py` (P-1 / T4-1), which must be re-run."
        )
    return grouped_split(
        labels, groups, eval_frac=eval_frac, calib_frac=calib_frac, fold=fold, seed=SPLIT_SEED
    )


def build_splits(
    cfg: ExperimentConfig | Any,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """``(labels, train_idx, val_idx, test_idx)`` for the configured protocol.

    The four-tuple shape predates Tier 4 and is kept for the callers that do
    not need the calibration split. ``train_idx`` **excludes** calib, which is
    P-5's requirement: calib never contributes a gradient. Use
    :func:`build_split_bundle` to reach it.
    """
    bundle = build_split_bundle(cfg)
    return bundle.labels, bundle.train, bundle.val, bundle.test


# ══════════════════════════════════════════════════════════════════════
#  DataLoader construction
# ══════════════════════════════════════════════════════════════════════


def seed_worker(worker_id: int) -> None:
    """Seed a worker's ``random`` and NumPy streams from the one torch already set.

    ``DataLoader`` derives each worker's *torch* seed from its ``generator``, so
    the ``torch.rand``/``torch.randint`` guards in
    :meth:`~spectralquadnet.data.datasets.RiceSeedDataset.__getitem__` are
    reproducible out of the box. The two augmentations that do not use torch —
    :meth:`~spectralquadnet.data.datasets.RiceSeedDataset._spectral_warp`'s
    ``random.uniform`` — are not, and under ``spawn`` a worker's ``random``
    module starts from OS entropy. Deriving both from ``torch.initial_seed()``
    puts every stream in the worker back under ``cfg.seed``.

    The warning filters are re-installed here for the same ``spawn`` reason: a
    worker starts with a fresh ``warnings`` registry, so the ones ``train.py``
    installed in the parent do not reach it — and a worker's ``warnings.warn``
    goes to the *shared* stderr, landing in the middle of the parent's epoch
    line. See :mod:`spectralquadnet.utils.warning_filters`.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    silence_known_warnings()


def loader_options(plan: RuntimePlan | None = None, *, evaluation: bool = False) -> dict[str, Any]:
    """The performance keywords for a ``DataLoader``, or the pre-refactor defaults.

    ``plan=None`` yields ``num_workers=0, pin_memory=False`` — a synchronous,
    single-process feed, which is what every loader in this module used to be
    unconditionally. Passing a plan is what turns on worker processes, page-locked
    staging and prefetch; see :class:`~spectralquadnet.utils.device.RuntimePlan`.
    """
    if plan is None:
        return {"num_workers": 0, "pin_memory": False}
    return plan.eval_loader_kwargs if evaluation else plan.loader_kwargs


def _make_loader(
    dataset: Dataset[Any],
    *,
    plan: RuntimePlan | None,
    evaluation: bool,
    seed: int | None = None,
    **kwargs: Any,
) -> DataLoader[Any]:
    """One place that knows how a loader is configured on this hardware.

    Every ``DataLoader(...)`` in the package goes through here, so worker
    counts, pinning, prefetch depth and the per-worker RNG seeding cannot drift
    between the seven construction sites that used to hardcode
    ``num_workers=0``.
    """
    options = loader_options(plan, evaluation=evaluation)
    if options["num_workers"] > 0:
        kwargs["worker_init_fn"] = seed_worker
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            kwargs["generator"] = generator
    return DataLoader(dataset, **options, **kwargs)


def _shard_batch(batch_train: int, dist: DistContext) -> int:
    """The per-rank batch size whose ``world_size`` copies sum to ``batch_train``.

    This is what keeps a two-GPU run the same run: DDP averages the per-rank
    gradients, so the optimiser sees the gradient of the **global** batch, and
    the global batch has to be the single-GPU one or ``stage1.batch`` has
    silently doubled.

    Raises:
        ValueError: The batch does not divide by the world size. Unequal shards
            make the all-reduced mean a weighted average of differently sized
            means, which is not the global-batch gradient — so this refuses
            rather than approximating it.
    """
    if not dist.enabled:
        return batch_train
    if batch_train % dist.world_size:
        raise ValueError(
            f"batch size {batch_train} does not divide by world_size {dist.world_size}. "
            "DDP splits the batch so the global batch stays the configured one; an uneven "
            "split would change the effective batch. Use a world size that divides it, or "
            "set runtime.multi_gpu=off."
        )
    return batch_train // dist.world_size


def build_loaders(
    cfg: ExperimentConfig | Any,
    store: DataStore,
    device: torch.device | str,
    train_idx: npt.NDArray[Any],
    val_idx: npt.NDArray[Any],
    test_idx: npt.NDArray[Any],
    batch_train: int,
    balanced: bool = False,
    all_labels: npt.NDArray[Any] | None = None,
    train_aug: str = "none",
    class_weights: dict[int, float] | None = None,
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
    morph: npt.NDArray[Any] | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Build train/val/test loaders sharing one ``DataStore``/config/device.

    The train loader uses a :class:`~spectralquadnet.data.samplers.ClassBalancedBatchSampler`
    when ``balanced=True``, otherwise plain shuffled batching; val/test are
    always shuffled=False at :data:`EVAL_BATCH`.

    Args:
        cfg: Composed experiment config.
        store: Memory-mapped patch store shared by all three datasets.
        device: Device the batch is moved to. Recorded on the dataset; the move
            itself happens once per batch in ``engine/batch.py``.
        train_idx: Indices for the training split.
        val_idx: Indices for the validation split.
        test_idx: Indices for the test split.
        batch_train: **Global** batch size for the (non-balanced) train loader.
            Under DDP each rank draws ``batch_train // world_size``.
        balanced: Whether to use class-balanced batch sampling for training.
        all_labels: Full label array, required when ``balanced=True``.
        train_aug: Augmentation profile name for the training dataset.
        class_weights: Optional per-class weights for the balanced sampler.
        plan: Resolved runtime plan. ``None`` keeps the synchronous
            single-process feed, which is what the unit tests build.
        dist: DDP context. ``None`` is single-process.
        morph: Pre-fitted standardised morphometrics, to avoid refitting the
            train-split mean and scale once per call. ``None`` fits them here.

    Returns:
        ``(train_loader, val_loader, test_loader)``.
    """
    dist = dist or DistContext()
    # Annotated rather than inferred: the inferred value type is the union of the
    # three entries, which cannot be checked against `RiceSeedDataset`'s distinct
    # keyword types when unpacked with `**`.
    kw: dict[str, Any] = dict(
        store=store,
        data_cfg=cfg.data,
        device=device,
        morph=standardised_morphometrics(store, train_idx) if morph is None else morph,
    )

    ds = RiceSeedDataset(train_idx, aug_strength=train_aug, **kw)
    seed = int(getattr(cfg, "seed", 42))

    if balanced and all_labels is not None:
        samp: Any = ClassBalancedBatchSampler(
            all_labels[train_idx],
            cfg.stage2.bal_n_cls,
            cfg.stage2.bal_n_spc,
            class_weights=class_weights,
            # Every rank must compose the *same* batch before slicing it, or
            # the "global batch" the gradient average describes does not exist.
            seed=seed if dist.enabled else None,
        )
        if dist.enabled:
            samp = DistributedBatchShardSampler(samp, dist.rank, dist.world_size)
        tr_ldr = _make_loader(ds, plan=plan, evaluation=False, seed=seed, batch_sampler=samp)
    else:
        tr_ldr = _make_loader(
            ds,
            plan=plan,
            evaluation=False,
            seed=seed,
            batch_size=_shard_batch(batch_train, dist),
            shuffle=not dist.enabled,
            sampler=(
                torch.utils.data.DistributedSampler(
                    ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True, seed=seed
                )
                if dist.enabled
                else None
            ),
            drop_last=True,
        )

    va_ldr = build_eval_loader(RiceSeedDataset(val_idx, **kw), plan=plan, dist=dist)
    te_ldr = build_eval_loader(RiceSeedDataset(test_idx, **kw), plan=plan, dist=dist)
    return tr_ldr, va_ldr, te_ldr


def build_train_loader(
    dataset: Dataset[Any],
    batch: int,
    *,
    seed: int = SPLIT_SEED,
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
) -> DataLoader[Any]:
    """A shuffled, drop-last training loader over an **already-built** dataset.

    :func:`build_loaders` constructs its own datasets, so Stage 1's three
    per-phase loaders — which differ only in their augmentation profile — cannot
    go through it. They still need the same worker, pinning and DDP-sharding
    decisions, which is what this shares with it.

    Args:
        dataset: The dataset to iterate.
        batch: **Global** batch size; each rank draws ``batch // world_size``.
        seed: Seeds the loader's generator and the distributed sampler.
        plan: Resolved runtime plan; ``None`` keeps the synchronous feed.
        dist: DDP context; ``None`` is single-process.
    """
    dist = dist or DistContext()
    sampler: Sampler[int] | None = (
        torch.utils.data.DistributedSampler(
            dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=True, seed=seed
        )
        if dist.enabled
        else None
    )
    return _make_loader(
        dataset,
        plan=plan,
        evaluation=False,
        seed=seed,
        batch_size=_shard_batch(batch, dist),
        shuffle=not dist.enabled,
        sampler=sampler,
        drop_last=True,
    )


def build_eval_loader(
    dataset: Dataset[Any],
    *,
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
    batch_size: int = EVAL_BATCH,
) -> DataLoader[Any]:
    """An unshuffled loader over an evaluation split, sharded across ranks under DDP.

    ``drop_last=False`` and ``shuffle=False`` on both paths: every patch must be
    scored exactly once, and the order has to be the order the targets come back
    in. Under DDP the shards are re-joined by
    :func:`~spectralquadnet.utils.distributed.gather_concat`, which trims the
    padding ``DistributedSampler`` adds — so a val set whose size does not
    divide by the world size is still scored on its true size.
    """
    dist = dist or DistContext()
    sampler: Sampler[int] | None = (
        torch.utils.data.DistributedSampler(
            dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=False, drop_last=False
        )
        if dist.enabled
        else None
    )
    return _make_loader(
        dataset,
        plan=plan,
        evaluation=True,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
    )


def standardised_morphometrics(
    store: DataStore, train_idx: npt.NDArray[Any]
) -> npt.NDArray[Any] | None:
    """The ``(N, 8)`` morphometrics, standardised on ``train_idx`` **only** (P-4 / T4-4).

    ``None`` when the array was never written, which is the fallback path the
    model substitutes zeros for. When it *is* present, the mean and scale come
    from the training rows alone: fitting them over all 8,624 would let the val
    and test grain sizes set the origin of a feature the network then classifies
    with — a small leak, but the same kind as C-1's and pointless to accept when
    the fix is an index argument.

    Called once per :func:`build_loaders`, so all three splits share one fit and
    cannot disagree about where the origin is.
    """
    if store.morphology is None:
        return None
    standardised, _ = standardise_morphometrics(store.morphology, train_idx)
    return standardised


def build_calib_loader(
    cfg: ExperimentConfig | Any,
    store: DataStore,
    device: torch.device | str,
    calib_idx: npt.NDArray[Any],
    batch_size: int = EVAL_BATCH,
    train_idx: npt.NDArray[Any] | None = None,
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
) -> DataLoader[Any] | None:
    """An un-augmented, unshuffled loader over the calibration split (P-5 / T4-5).

    Returns ``None`` when no calibration split was carved, which is the signal
    every caller uses to fall back to ``val_ldr`` — i.e. to the pre-Tier-4
    behaviour, where the fitted parameters and the selection metric share a
    split.
    """
    if calib_idx.size == 0:
        return None
    # The morphometric origin is the *training* split's, never calib's — calib
    # is a held-out split like val and test, and standardising it on itself
    # would put a different origin under the margins than under the gradients.
    fit_on = train_idx if train_idx is not None else calib_idx
    return build_eval_loader(
        RiceSeedDataset(
            calib_idx,
            store=store,
            data_cfg=cfg.data,
            device=device,
            morph=standardised_morphometrics(store, fit_on),
        ),
        plan=plan,
        dist=dist,
        batch_size=batch_size,
    )


def build_natural_prior_loader(
    loader: DataLoader[Any],
    batch_size: int | None = None,
    plan: RuntimePlan | None = None,
) -> DataLoader[Any]:
    """Re-wrap ``loader``'s dataset in a plain shuffled loader, dropping any sampler.

    BatchNorm re-estimation after SWA averaging must see the **natural** class
    prior. Stage 3's training loader is a
    :class:`~spectralquadnet.data.samplers.ClassBalancedBatchSampler` carrying
    Stage-2 CDWS weights, which over-represents hard classes by up to
    ``cdws_max_weight``; estimating :math:`\\mu, \\sigma^2` under that prior and
    applying them at test under the natural one is a systematic shift of both
    branch outputs, since BN buffers are a global affine correction
    (IMPROVEMENT_PLAN §2.5.6(ii), §3.6 OP-4.6).

    Args:
        loader: The loader whose ``dataset`` to reuse. Its sampler, batch
            sampler and weights are all discarded.
        batch_size: Batch size for the new loader. Defaults to ``loader``'s own
            when it has one, else the size its batch sampler yields, else 128.

    Returns:
        A shuffled, unweighted, un-dropped-last loader over the same dataset.
    """
    if batch_size is None:
        bs = loader.batch_size
        if bs is None:
            sampler = loader.batch_sampler
            n_cls = getattr(sampler, "n_cls", None)
            n_spc = getattr(sampler, "n_spc", None)
            bs = n_cls * n_spc if n_cls and n_spc else 128
        batch_size = int(bs)
    return _make_loader(
        loader.dataset,
        plan=plan,
        # BN re-estimation is a forward-only pass over the training split, so it
        # is budgeted like an evaluation loader: workers, but never persistent
        # ones, since this loader is built once and dropped.
        evaluation=True,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


def build_phase3_loader(
    cfg: ExperimentConfig | Any,
    store: DataStore,
    train_ds: Dataset[Any],
    class_f1: dict[int, float],
    plan: RuntimePlan | None = None,
    dist: DistContext | None = None,
) -> DataLoader[Any]:
    """Build the Phase-3 DataLoader with hard-class oversampling.

    Uses :class:`~spectralquadnet.data.samplers.HardClassOversampledSampler`
    to give hard classes (low F1 from Phase 2) higher sampling probability.
    Falls back to a standard shuffled loader if oversampling is disabled in
    config, or if no per-class F1 was supplied.

    Args:
        cfg: Composed experiment config.
        store: Memory-mapped patch store, used to read training labels.
        train_ds: Phase-3 training dataset; must expose ``.indices`` (as
            :class:`~spectralquadnet.data.datasets.RiceSeedDataset` does).
        class_f1: Per-class F1 from the Phase 2 -> 3 boundary evaluation,
            driving the oversampling weights.

    Returns:
        DataLoader over ``train_ds``, oversampled or plain-shuffled.
    """
    dist = dist or DistContext()
    seed = int(getattr(cfg, "seed", 42))
    batch = _shard_batch(cfg.stage1.batch, dist)

    if not cfg.stage1.p3_oversample or not class_f1:
        return _make_loader(
            train_ds,
            plan=plan,
            evaluation=False,
            seed=seed,
            batch_size=batch,
            shuffle=not dist.enabled,
            sampler=(
                torch.utils.data.DistributedSampler(
                    train_ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True, seed=seed
                )
                if dist.enabled
                else None
            ),
            drop_last=True,
        )

    # `.indices`/`.split_labels` are `RiceSeedDataset`'s, not the generic
    # `Dataset` protocol's; the parameter type stays the duck-typed `Dataset` so
    # `DataLoader.dataset` can be handed straight in without a downcast. The
    # labels were rebuilt here with a Python loop over every training index;
    # the dataset already resolved them once at construction.
    train_labels = np.asarray(train_ds.split_labels)  # type: ignore[attr-defined]
    sampler: Any = HardClassOversampledSampler(
        labels=train_labels,
        class_f1=class_f1,
        num_samples=len(train_labels),
        oversample_power=cfg.stage1.p3_oversample_power,
        max_weight=cfg.stage1.p3_oversample_max_w,
        hard_f1_thresh=cfg.stage1.p3_hard_f1_thresh,
        eps=cfg.stage1.p3_oversample_eps,
    )
    if dist.enabled:
        sampler = DistributedIndexShardSampler(sampler, dist.rank, dist.world_size)
    return _make_loader(
        train_ds,
        plan=plan,
        evaluation=False,
        seed=seed,
        batch_size=batch,
        sampler=sampler,
        drop_last=True,
    )
