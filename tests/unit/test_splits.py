"""T4-1 / P-1 — the split protocol.

Carries §4.3's ``test_splits_are_group_disjoint``, the contract C-1 is the
violation of:

.. math::
    \\mathrm{scan}(i) \\in \\text{train}
    \\;\\Longrightarrow\\;
    \\mathrm{scan}(i) \\notin \\text{val} \\cup \\text{test}

Every defect test here has a companion that measures the defect on the *old*
behaviour, so none of them can pass vacuously:
:func:`test_the_stratified_split_shares_every_scan` shows the shipped protocol
putting all of its scans in all three splits, which is 0-H's 107-of-107
reproduced from the split code rather than from an audit script.

The fixtures build group structures by hand. Two of them matter:

* ``two_groups_per_class`` — this dataset's shape (0-H: every variety captured
  exactly twice). It is what makes the three-way *full* disjointness the §4.2
  criterion asks for impossible, and the §3.1 contract still achievable.
* ``many_groups_per_class`` — the shape the plan assumed, where val and test
  can hold disjoint scans too.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from spectralquadnet.data.loaders import (
    SPLIT_SCHEMES,
    Splits,
    build_split_bundle,
    build_splits,
    grouped_split,
)

N_CLASSES = 12
PER_GROUP = 20


def _make(groups_per_class: int, n_classes: int = N_CLASSES, per_group: int = PER_GROUP):
    """``(labels, groups)`` with ``groups_per_class`` scans of ``per_group`` patches each."""
    labels, groups = [], []
    gid = 0
    for c in range(n_classes):
        for _ in range(groups_per_class):
            labels.extend([c] * per_group)
            groups.extend([gid] * per_group)
            gid += 1
    return np.array(labels, dtype=np.int64), np.array(groups, dtype=np.int64)


@pytest.fixture
def two_groups_per_class():
    """This dataset's shape: two scans per class (0-H)."""
    return _make(2)


@pytest.fixture
def many_groups_per_class():
    """Four scans per class — enough for val and test to be group-disjoint too."""
    return _make(4)


def _scans(splits: Splits, name: str) -> set[int]:
    idx = getattr(splits, name)
    return set(np.unique(splits.groups[idx]).tolist()) if idx.size else set()


# ══════════════════════════════════════════════════════════════════════
#  §4.3 · test_splits_are_group_disjoint
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("groups_per_class", [2, 3, 4, 5])
@pytest.mark.parametrize("calib_frac", [0.0, 0.2])
def test_splits_are_group_disjoint(groups_per_class: int, calib_frac: float) -> None:
    """§4.3: ``scan(train) ∩ scan(val ∪ test) = ∅``, at every group count.

    This is the contract §3.1 writes down and the one the metric depends on:
    a scan in both train and test means the test number is partly a
    measurement of scan memorisation (C-1, §2.1.1).
    """
    labels, groups = _make(groups_per_class)
    s = grouped_split(labels, groups, calib_frac=calib_frac)

    train, val, test = _scans(s, "train"), _scans(s, "val"), _scans(s, "test")
    assert train & (val | test) == set()
    assert s.report.train_eval_group_disjoint

    # Calib is carved from train, so it must also stay out of the eval side.
    assert _scans(s, "calib") & (val | test) == set()


def test_the_stratified_split_shares_every_scan(two_groups_per_class) -> None:
    """The companion: the pre-Tier-4 protocol fails the test above.

    0-H measured 107 of 107 capture groups present in all three splits on the
    real data. Here the same thing is measured from the split code, so the
    defect is asserted rather than remembered.
    """
    from spectralquadnet.data.loaders import _stratified_split

    labels, groups = two_groups_per_class
    s = _stratified_split(labels, 0.30, 0.0, groups)
    train, val, test = _scans(s, "train"), _scans(s, "val"), _scans(s, "test")
    assert train & (val | test), "a patch-level split cannot be group-disjoint"
    assert s.report.n_shared_groups_train_eval == len(np.unique(groups))
    assert not s.report.train_eval_group_disjoint


# ══════════════════════════════════════════════════════════════════════
#  Grouped split — structure
# ══════════════════════════════════════════════════════════════════════


def test_every_class_is_present_in_every_split(two_groups_per_class) -> None:
    """Group disjointness must not cost a class its representation.

    A class absent from test cannot contribute to a 90-class macro-F1, so a
    split that drops classes does not measure the reported metric at all. This
    is why the §4.2 criterion — "no scan_id in more than one split" — is not
    the property enforced: with two scans per class it can only be met by
    leaving each class out of one split.
    """
    labels, groups = two_groups_per_class
    s = grouped_split(labels, groups, calib_frac=0.2)
    for name in ("train", "calib", "val", "test"):
        present = np.unique(labels[getattr(s, name)])
        assert (
            len(present) == N_CLASSES
        ), f"{name} is missing classes {set(range(N_CLASSES)) - set(present.tolist())}"


def test_the_splits_are_an_exact_partition(many_groups_per_class) -> None:
    """No patch is dropped and none appears twice — a duplicate is train-on-test."""
    labels, groups = many_groups_per_class
    s = grouped_split(labels, groups, calib_frac=0.25)
    everything = np.concatenate([s.train, s.calib, s.val, s.test])
    assert np.array_equal(np.sort(everything), np.arange(len(labels)))


def test_val_and_test_are_group_disjoint_when_the_groups_allow_it(
    many_groups_per_class,
) -> None:
    """With ≥ 2 eval scans per class, val and test hold different scans."""
    labels, groups = many_groups_per_class
    s = grouped_split(labels, groups, eval_frac=0.5)
    assert _scans(s, "val") & _scans(s, "test") == set()
    assert s.report.val_test_group_disjoint
    assert s.report.classes_sharing_groups_val_test == []


def test_val_and_test_share_a_scan_when_the_class_has_only_one_eval_group(
    two_groups_per_class,
) -> None:
    """And the report says so, per class, rather than claiming what it cannot.

    This is the honest half of T4-1 on this dataset: the headline contract
    (train ↔ val ∪ test) holds, the finer one (val ↔ test) does not, and the
    difference is recorded instead of glossed.
    """
    labels, groups = two_groups_per_class
    s = grouped_split(labels, groups)
    assert not s.report.val_test_group_disjoint
    assert s.report.classes_sharing_groups_val_test == list(range(N_CLASSES))
    # The contract that does hold, still holds.
    assert s.report.train_eval_group_disjoint


def test_calib_is_group_disjoint_when_the_train_pool_has_two_scans(
    many_groups_per_class,
) -> None:
    labels, groups = many_groups_per_class
    s = grouped_split(labels, groups, eval_frac=0.25, calib_frac=0.34)
    assert _scans(s, "calib") & _scans(s, "train") == set()
    assert s.report.calib_group_disjoint


# ══════════════════════════════════════════════════════════════════════
#  Leave-one-scan-out folds (§3.1's stated fallback)
# ══════════════════════════════════════════════════════════════════════


def test_folds_hold_out_different_scans(two_groups_per_class) -> None:
    """Sweeping ``split_fold`` is the leave-one-scan-out CV §3.1 falls back to."""
    labels, groups = two_groups_per_class
    fold0 = grouped_split(labels, groups, fold=0)
    fold1 = grouped_split(labels, groups, fold=1)
    assert _scans(fold0, "val") | _scans(fold0, "test") != _scans(fold1, "val") | _scans(
        fold1, "test"
    )


def test_the_folds_cover_every_scan(two_groups_per_class) -> None:
    """Across both folds, every scan is evaluated once — no scan is never tested."""
    labels, groups = two_groups_per_class
    evaluated: set[int] = set()
    for fold in (0, 1):
        s = grouped_split(labels, groups, fold=fold)
        evaluated |= _scans(s, "val") | _scans(s, "test")
    assert evaluated == set(np.unique(groups).tolist())


def test_the_fold_index_wraps(two_groups_per_class) -> None:
    """``fold`` is taken modulo the class's group count, so it never errors."""
    labels, groups = two_groups_per_class
    assert np.array_equal(
        grouped_split(labels, groups, fold=0).test, grouped_split(labels, groups, fold=2).test
    )


def test_the_split_is_deterministic(many_groups_per_class) -> None:
    labels, groups = many_groups_per_class
    a = grouped_split(labels, groups, calib_frac=0.25)
    b = grouped_split(labels, groups, calib_frac=0.25)
    for name in ("train", "calib", "val", "test"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_a_different_seed_gives_a_different_split(many_groups_per_class) -> None:
    labels, groups = many_groups_per_class
    a = grouped_split(labels, groups, seed=42)
    b = grouped_split(labels, groups, seed=7)
    assert not np.array_equal(a.test, b.test)


# ══════════════════════════════════════════════════════════════════════
#  Single-scan classes — §3.1's "report the count explicitly"
# ══════════════════════════════════════════════════════════════════════


def test_a_single_scan_class_is_refused_and_named() -> None:
    """§3.1: report the count explicitly rather than silently mixing.

    0-H found 73 of 90 classes captured in a single ``(session, variety)``
    group, which is why F-1 as written is not testable on this archive. The
    split refuses, names the classes and points at the fallback instead of
    producing a number that quietly contains the leak.
    """
    labels, groups = _make(2)
    groups[labels == 3] = 999  # collapse class 3 onto one scan

    with pytest.raises(ValueError) as excinfo:
        grouped_split(labels, groups)

    message = str(excinfo.value)
    assert "1 of 12 classes" in message
    assert "classes 3" in message
    assert "leave-one-scan-out" in message


def test_the_fallback_records_the_leak_it_accepts() -> None:
    """``patch_split`` is opt-in, and what it costs is in the report."""
    labels, groups = _make(2)
    groups[labels == 3] = 999

    s = grouped_split(labels, groups, single_group_policy="patch_split")
    assert s.report.classes_leaking_into_eval == [3]
    assert s.report.classes_with_one_group == [3]
    assert not s.report.train_eval_group_disjoint
    # …and the other eleven classes are still clean.
    clean = np.isin(labels, [c for c in range(N_CLASSES) if c != 3])
    train_scans = set(np.unique(groups[s.train][clean[s.train]]).tolist())
    eval_idx = np.concatenate([s.val, s.test])
    eval_scans = set(np.unique(groups[eval_idx][clean[eval_idx]]).tolist())
    assert train_scans & eval_scans == set()


def test_an_unknown_single_group_policy_is_rejected() -> None:
    labels, groups = _make(2)
    with pytest.raises(ValueError, match="single_group_policy"):
        grouped_split(labels, groups, single_group_policy="ignore")


def test_mismatched_labels_and_groups_are_rejected() -> None:
    labels, groups = _make(2)
    with pytest.raises(ValueError, match="same shape"):
        grouped_split(labels, groups[:-1])


# ══════════════════════════════════════════════════════════════════════
#  The stratified path is unchanged
# ══════════════════════════════════════════════════════════════════════


def test_the_stratified_path_reproduces_the_reference_split(cfg) -> None:
    """The shipped protocol is bit-identical to the recipe it replaced.

    Transcribed here rather than imported, because the point is that the two
    ``train_test_split`` calls, their order, their ``random_state`` and the
    order of the arrays they return all survived the Tier-4 rewrite — the
    archived checkpoints were selected on exactly this partition, and
    ``DataLoader(shuffle=True)`` permutes *positions*, so even re-ordering the
    index array would move the training stream.
    """
    labels = np.load(cfg.data.labels_path)
    indices = np.arange(len(labels))
    tr, tmp = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)

    got_labels, got_tr, got_va, got_te = build_splits(cfg)
    assert np.array_equal(got_labels, labels)
    assert np.array_equal(got_tr, tr)
    assert np.array_equal(got_va, va)
    assert np.array_equal(got_te, te)
    assert (len(tr), len(va), len(te)) == (6036, 1294, 1294)


def test_the_bundle_and_the_tuple_agree(cfg) -> None:
    bundle = build_split_bundle(cfg)
    labels, tr, va, te = build_splits(cfg)
    assert np.array_equal(bundle.train, tr)
    assert np.array_equal(bundle.val, va)
    assert np.array_equal(bundle.test, te)
    assert np.array_equal(bundle.labels, labels)
    assert not bundle.has_calib  # the shipped config carves none


# ══════════════════════════════════════════════════════════════════════
#  Config plumbing
# ══════════════════════════════════════════════════════════════════════


def test_every_new_data_key_is_read_by_the_split_builder(cfg) -> None:
    """The Tier-4 slice of §4.3's ``test_config_keys_are_wired``.

    N-1 is the defect of a config key that names something no code reads.
    Every key this tier adds is read by ``build_split_bundle``, and the way to
    prove it is to change each one and watch the outcome change — a key that
    is merely present would produce no reaction at all.
    """
    from omegaconf import OmegaConf

    for key in ("groups_path", "split_scheme", "split_eval_frac", "split_fold", "calib_frac"):
        assert key in cfg.data, f"{key} missing from the composed config"

    # split_eval_frac: a different fraction gives a different partition.
    wider = OmegaConf.merge(cfg, {"data": {"split_eval_frac": 0.5}})
    assert len(build_split_bundle(wider).test) != len(build_split_bundle(cfg).test)

    # calib_frac: a non-zero value carves a calib split out of train.
    with_calib = OmegaConf.merge(cfg, {"data": {"calib_frac": 0.1}})
    bundle = build_split_bundle(with_calib)
    assert bundle.has_calib
    assert len(bundle.train) + len(bundle.calib) == 6036

    # split_scheme: an unknown one is refused rather than defaulted.
    with pytest.raises(ValueError, match="split_scheme"):
        build_split_bundle(OmegaConf.merge(cfg, {"data": {"split_scheme": "kfold"}}))

    # split_fold: meaningless without groups, so it is refused rather than ignored.
    with pytest.raises(ValueError, match="split_fold"):
        build_split_bundle(OmegaConf.merge(cfg, {"data": {"split_fold": 1}}))

    # groups_path: the grouped scheme cannot proceed without it.
    with pytest.raises(ValueError, match="groups_path|does not exist"):
        build_split_bundle(
            OmegaConf.merge(cfg, {"data": {"split_scheme": "grouped", "groups_path": "./nope.npy"}})
        )


def test_the_pfix_config_selects_the_grouped_protocol() -> None:
    """``data=spa40_90class_pfix`` is the re-baseline config §4.4 sequences."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from spectralquadnet.config.schema import register_configs

    repo_root = Path(__file__).resolve().parents[2]
    register_configs()
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base="1.3"):
        pfix = compose(
            config_name="experiment/quadnet_audited",
            overrides=["data=spa40_90class_pfix"],
        )
    assert pfix.data.split_scheme == "grouped"
    assert pfix.data.calib_frac > 0.0
    assert pfix.data.split_scheme in SPLIT_SCHEMES


# ══════════════════════════════════════════════════════════════════════
#  The report
# ══════════════════════════════════════════════════════════════════════


def test_the_report_is_serialisable_and_states_what_it_measured(
    two_groups_per_class,
) -> None:
    """A run's log has to record what its number means (§4.5)."""
    import json

    labels, groups = two_groups_per_class
    report = grouped_split(labels, groups, calib_frac=0.2).report
    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["scheme"] == "grouped"
    assert payload["n_groups"] == 2 * N_CLASSES
    assert payload["groups_per_class_min"] == payload["groups_per_class_max"] == 2
    assert sum(payload["sizes"].values()) == len(labels)
    assert any("group-disjoint" in line for line in report.summary())
