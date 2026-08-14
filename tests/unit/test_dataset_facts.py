"""IC-13, IC-4 and IC-14 — documented facts, band-selection scope, dead paths.

**IC-13.** ``01`` §1.3 and ``02`` §2.8 both stated the dataset has 107 capture
scans, "measured 107/107". The executed run measured **180/180**. One of these
was wrong and it mattered: 107 is not divisible by 90 and contradicts the
"exactly two scans per class" statement made in the same paragraph. The Zenodo
record (3241923) images each of the 90 varieties as two bundles of 48 kernels,
so 90 × 2 = 180, and 8,624 / 180 = 47.9 patches per bundle.

**IC-4.** Band selection ran ``mutual_info_classif(X, y)`` over all 8,624
patches — every patch that would become test, with its label. That is genuine
label leakage, independent of the split protocol, and it contaminates
``grouped`` too (CHANGES §4.1).

**IC-14.** Four documented dead paths, one of which emitted live telemetry for
a loss that did not run.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
GROUPS = REPO_ROOT / "dataset" / "groups.npy"

#: 90 varieties × 2 class-pure acquisition bundles.
EXPECTED_BUNDLES = 180
EXPECTED_CLASSES = 90


# ══════════════════════════════════════════════════════════════════════
#  IC-13 — the documented bundle count
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.requires_dataset
def test_the_dataset_really_has_180_acquisition_bundles() -> None:
    """IC-13's stated validation criterion: ``len(np.unique(groups)) == 180``."""
    if not GROUPS.exists():
        pytest.skip(f"{GROUPS} missing — dataset/ is gitignored")
    groups = np.load(GROUPS)
    assert len(np.unique(groups)) == EXPECTED_BUNDLES


@pytest.mark.requires_dataset
def test_every_bundle_is_class_pure_and_every_class_has_two() -> None:
    """The structural fact the whole evaluation protocol rests on.

    Class-pure bundles are *why* a patch-level split leaks: a model that learns
    a tray's residual radiometric signature scores correctly on that tray's
    held-out patches without knowing anything about the variety.
    """
    labels_path = REPO_ROOT / "dataset" / "labels.npy"
    if not GROUPS.exists() or not labels_path.exists():
        pytest.skip("dataset/ is gitignored")
    groups, labels = np.load(GROUPS), np.load(labels_path)

    for bundle in np.unique(groups):
        assert len(np.unique(labels[groups == bundle])) == 1, f"bundle {bundle} is not class-pure"
    for label in np.unique(labels):
        assert len(np.unique(groups[labels == label])) == 2, f"class {label} has != 2 bundles"


@pytest.mark.parametrize("doc", ["01_ABSTRACT_AND_OVERVIEW.md", "02_DATASET_AND_PREPROCESSING.md"])
def test_the_docs_no_longer_claim_107_capture_scans(doc: str) -> None:
    """The docs and the run must agree about what the dataset is."""
    text = (DOCS / doc).read_text()
    offending = [
        line
        for line in text.splitlines()
        # A bare "107" elsewhere is fine; what is banned is 107 used as the
        # scan/bundle count, which is the claim that was wrong.
        if re.search(r"107\s*(capture scans|/\s*107)", line)
        and "was wrong" not in line
        and "stated" not in line
    ]
    assert not offending, f"{doc} still claims 107 capture scans:\n" + "\n".join(offending)


@pytest.mark.parametrize("doc", ["01_ABSTRACT_AND_OVERVIEW.md", "02_DATASET_AND_PREPROCESSING.md"])
def test_the_docs_state_the_corrected_figure(doc: str) -> None:
    assert "180" in (DOCS / doc).read_text(), f"{doc} does not state the 180-bundle figure"


# ══════════════════════════════════════════════════════════════════════
#  IC-4 — band selection sees training rows only
# ══════════════════════════════════════════════════════════════════════


def test_a_fold_restricts_the_selection_to_training_rows(tmp_path) -> None:
    """IC-4's validation criterion: no eval row index reaches the selector."""
    from spectralquadnet.data.loaders import grouped_split
    from spectralquadnet.data.prep.band_selection import resolve_train_indices
    from spectralquadnet.data.prep.config import BandSelectionConfig

    n_classes, per_bundle = 6, 8
    labels = np.repeat(np.arange(n_classes), 2 * per_bundle)
    groups = np.concatenate([np.repeat([2 * c, 2 * c + 1], per_bundle) for c in range(n_classes)])
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "groups.npy", groups)

    cfg = BandSelectionConfig(
        labels_path=str(tmp_path / "labels.npy"),
        groups_path=str(tmp_path / "groups.npy"),
        fold=0,
        split_scheme="grouped",
        calib_frac=0.15,
    )
    train_idx = resolve_train_indices(cfg)
    assert train_idx is not None

    reference = grouped_split(
        labels,
        groups,
        eval_frac=cfg.split_eval_frac,
        calib_frac=cfg.calib_frac,
        fold=0,
        seed=cfg.seed,
        single_group_policy="patch_split",
    )
    held_out = set(np.concatenate([reference.val, reference.test, reference.calib]).tolist())
    assert not (
        set(train_idx.tolist()) & held_out
    ), "an eval or calib row reached the band selector — that is the leak IC-4 closes"
    assert set(train_idx.tolist()) == set(reference.train.tolist())


def test_no_fold_reproduces_the_leaky_whole_corpus_selection() -> None:
    """Kept deliberately: it is ablation A2's control arm."""
    from spectralquadnet.data.prep.band_selection import resolve_train_indices
    from spectralquadnet.data.prep.config import BandSelectionConfig

    assert resolve_train_indices(BandSelectionConfig(fold=None)) is None


def test_a_fold_without_groups_refuses_rather_than_falling_back(tmp_path) -> None:
    """A silent fallback here would produce a leaky selection wearing IC-4's name."""
    from spectralquadnet.data.prep.band_selection import resolve_train_indices
    from spectralquadnet.data.prep.config import BandSelectionConfig

    np.save(tmp_path / "labels.npy", np.repeat(np.arange(4), 8))
    cfg = BandSelectionConfig(
        labels_path=str(tmp_path / "labels.npy"),
        groups_path=str(tmp_path / "absent.npy"),
        fold=0,
    )
    with pytest.raises(FileNotFoundError, match="prepare_dataset"):
        resolve_train_indices(cfg)


def test_a_per_fold_selection_writes_to_its_own_filename(tmp_path) -> None:
    """Two folds' band sets are not interchangeable; one filename would let the
    second silently overwrite the first, which is the failure IC-4 prevents."""
    from spectralquadnet.data.prep.band_selection import output_paths
    from spectralquadnet.data.prep.config import BandSelectionConfig

    flat = output_paths(BandSelectionConfig(output_dir=str(tmp_path)), "spa", 40)
    fold0 = output_paths(BandSelectionConfig(output_dir=str(tmp_path), fold=0), "spa", 40)
    fold1 = output_paths(BandSelectionConfig(output_dir=str(tmp_path), fold=1), "spa", 40)

    assert flat[0].name == "patches_spa_40b.npy", "the historical name is unchanged"
    assert fold0[0] != flat[0] and fold1[0] != fold0[0]
    assert "fold0" in fold0[0].name and "fold1" in fold1[0].name


def test_the_elbow_curve_runs_past_the_chosen_k() -> None:
    """M-14: a curve that stops at its own elbow satisfies the criterion vacuously."""
    from spectralquadnet.data.prep.config import BandSelectionConfig

    candidates = BandSelectionConfig().n_candidates
    assert max(candidates) >= 256, "the curve must reach the full band count"
    assert len([k for k in candidates if k > 40]) >= 5, "and have points past k=40"


# ══════════════════════════════════════════════════════════════════════
#  IC-14 — dead paths
# ══════════════════════════════════════════════════════════════════════


def test_the_specformer_no_longer_accepts_an_unread_stride() -> None:
    """An argument accepted, computed and discarded reads as a live knob."""
    from spectralquadnet.models.branches.specformer import SpecFormerBranch

    assert "stride" not in inspect.signature(SpecFormerBranch.__init__).parameters


def test_nothing_derives_the_token_count_from_the_band_count() -> None:
    """``specf_tokens`` is passed through, not divided into ``num_bands``.

    The derived form made a λ-window's *width* a function of ``k`` — 15 nm at
    k = 40, 2.4 nm on the full 256-band cube — so "token 3" denoted a different
    spectral region in the primary path and in every band-selection arm. That is
    exactly what λ-uniform tokenisation exists to prevent, so the derivation is
    pinned out of both the caller and the branch.
    """
    import torch

    from spectralquadnet.models import spectral_quadnet
    from spectralquadnet.models.branches.specformer import SpecFormerBranch

    assert "// 2" not in inspect.getsource(spectral_quadnet.SpectralQuadNet.__init__)

    # Behavioural, not a source grep: the same requested window count survives a
    # 6.4x change in band count, which is the property the derivation broke.
    for bands in (40, 256):
        wl = torch.linspace(0.0, 1.0, bands)
        branch = SpecFormerBranch(physical_wl=wl, num_bands=bands, n_tokens=16, d_model=24)
        assert branch.windows.n_tokens == 16, bands
        # …and the windows tile the same normalised λ domain either way, so
        # token t denotes the same spectral region in both.
        assert torch.allclose(
            branch.windows.token_wl,
            SpecFormerBranch(
                physical_wl=torch.linspace(0.0, 1.0, 40), num_bands=40, n_tokens=16, d_model=24
            ).windows.token_wl,
        )


def test_the_single_stage_applies_no_protonce() -> None:
    """ProtoNCE used in-batch class means as prototypes — an 8-sample estimate of
    a 256-d unit vector — to pull together the same positives SupCon was already
    pulling, at the same temperature, on the same embedding (CHANGES §7.2).
    """
    from spectralquadnet.engine.stages import single_stage

    source = inspect.getsource(single_stage)
    assert "proto=None" in source
    assert "proto_weight=0.0" in source


def test_no_stage_logs_a_weight_for_a_term_it_does_not_apply() -> None:
    """`sched/proto_weight` logged a constant for a loss that did not run.

    Dead code with live telemetry is worse than dead code: the panel asserts
    the term exists.
    """
    from spectralquadnet.engine.stages import single_stage

    assert "sched/proto_weight" not in inspect.getsource(single_stage)


def test_gain_is_never_a_model_input(cfg_default) -> None:
    """IC-14's resolution for ``gain.npy``.

    It is the residual per-pixel brightness the SNV divided out, which is also
    the strongest single carrier of acquisition-bundle identity. Feeding it to
    the classifier would hand the model the nuisance variable the grouped
    protocol exists to exclude — and it would raise the reported number, which
    is the direction nobody questions. It is wired to a *measurement* instead.
    """
    from spectralquadnet.experiments.leakage import assert_gain_is_not_a_model_input

    assert_gain_is_not_a_model_input(cfg_default)


def test_the_gain_key_is_consumed_by_the_leakage_probe(cfg_default) -> None:
    """IC-14: every persisted dataset artifact is either consumed or documented.

    ``gain_path`` is consumed — by the probe, not by the model — so it is
    neither dead nor an input.
    """
    from spectralquadnet.experiments import leakage

    assert hasattr(cfg_default.data, "gain_path")
    assert "gain_path" in inspect.getsource(leakage.run_probe)


@pytest.mark.parametrize(
    "artifact",
    [
        "patches_data",
        "labels_path",
        "wavelength_path",
        "groups_path",
        "masks_path",
        "morphology_path",
        "gain_path",
    ],
)
def test_every_persisted_artifact_has_a_config_key(cfg_default, artifact: str) -> None:
    """The inventory half of IC-14's criterion: nothing is written and unnamed."""
    assert hasattr(cfg_default.data, artifact)
