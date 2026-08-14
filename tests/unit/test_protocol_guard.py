"""IC-3 / IC-8 — the protocol is asserted, and the runtime invariant is restored.

CHANGES §4.2 is the audit's most consequential finding: the audited run used
``split_scheme=stratified``, so *"180 of 180 scans are in train and in val/test"*
— every class's two acquisition bundles present on both sides of the boundary.
The repository already shipped the fix and it was not used.

Two defects, one fix each:

* The ``SplitReport`` **measured** group-disjointness and nothing acted on it, so
  a protocol that silently degraded produced a number indistinguishable from one
  that did not. :func:`assert_protocol_holds` is CHANGES §21 Phase 0's assertion.
* The run's overrides set ``allow_tf32=True`` — which cuts matmul mantissas from
  24 bits to 11 — while disabling the two knobs that only change speed. The
  default composition restores the project's own stated invariant (§8.5).
"""

from __future__ import annotations

import numpy as np
import pytest

from spectralquadnet.config.compose import AUDITED_EXPERIMENT, load_experiment_config
from spectralquadnet.data.loaders import grouped_split
from spectralquadnet.engine.pipelines.context import assert_protocol_holds
from spectralquadnet.tracking.base import NullTracker


class _Recorder(NullTracker):
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def log_message(self, text: str, level: str = "info") -> None:
        self.messages.append((level, text))


def _two_bundles_per_class(n_classes: int = 6, per_bundle: int = 8):
    """The real dataset's shape in miniature: every class captured exactly twice."""
    labels = np.repeat(np.arange(n_classes), 2 * per_bundle)
    groups = np.concatenate([np.repeat([2 * c, 2 * c + 1], per_bundle) for c in range(n_classes)])
    return labels, groups


def _one_bundle_for_one_class(n_classes: int = 6, per_bundle: int = 8):
    """The pathological case `single_group_policy` exists for."""
    labels, groups = _two_bundles_per_class(n_classes, per_bundle)
    # Collapse class 0's two bundles into one.
    groups = groups.copy()
    groups[labels == 0] = 0
    return labels, groups


# ══════════════════════════════════════════════════════════════════════
#  IC-3 — the split refuses to degrade silently
# ══════════════════════════════════════════════════════════════════════


def test_a_grouped_split_of_two_bundle_classes_is_train_eval_disjoint() -> None:
    labels, groups = _two_bundles_per_class()
    splits = grouped_split(labels, groups, eval_frac=0.30, calib_frac=0.0)
    assert splits.report.train_eval_group_disjoint

    train_groups = set(np.unique(groups[splits.train]).tolist())
    eval_groups = set(np.unique(groups[np.concatenate([splits.val, splits.test])]).tolist())
    assert not (train_groups & eval_groups), "no bundle may cross the boundary"


def test_val_and_test_are_two_halves_of_one_bundle_and_the_report_says_so() -> None:
    """Not a bug — a data-collection ceiling that belongs in the paper.

    With two bundles per class, three-way group disjointness is mathematically
    impossible, so val and test are NOT independent of each other. That is why
    `evaluation.report_split=val_test` scores them together, once.
    """
    labels, groups = _two_bundles_per_class()
    splits = grouped_split(labels, groups, eval_frac=0.30, calib_frac=0.0)
    assert not splits.report.val_test_group_disjoint
    assert len(splits.report.classes_sharing_groups_val_test) == 6


def test_single_group_policy_error_refuses_and_names_the_classes() -> None:
    labels, groups = _one_bundle_for_one_class()
    with pytest.raises(ValueError, match="single scan"):
        grouped_split(labels, groups, single_group_policy="error")


def test_patch_split_accepts_the_leak_and_records_it() -> None:
    labels, groups = _one_bundle_for_one_class()
    splits = grouped_split(labels, groups, single_group_policy="patch_split")
    assert splits.report.classes_leaking_into_eval == [0]
    assert not splits.report.train_eval_group_disjoint


def test_the_guard_fails_a_run_whose_grouped_split_did_not_hold() -> None:
    """CHANGES §21 Phase 0's assertion: the report measured it, nothing acted."""
    cfg = load_experiment_config(overrides=["data.single_group_policy=patch_split"])
    labels, groups = _one_bundle_for_one_class()
    splits = grouped_split(labels, groups, single_group_policy="patch_split")

    with pytest.raises(ValueError, match="not the protocol that was asked for"):
        assert_protocol_holds(cfg, splits, _Recorder())


def test_the_guard_passes_a_genuinely_disjoint_grouped_split() -> None:
    cfg = load_experiment_config()
    labels, groups = _two_bundles_per_class()
    splits = grouped_split(labels, groups, eval_frac=0.30, calib_frac=0.15)
    assert_protocol_holds(cfg, splits, _Recorder())  # must not raise


def test_a_stratified_run_is_warned_that_its_number_is_a_mixture() -> None:
    cfg = load_experiment_config(overrides=["data=hsi256_stratified"])
    labels, groups = _two_bundles_per_class()
    splits = grouped_split(labels, groups, eval_frac=0.30, calib_frac=0.15)
    recorder = _Recorder()
    assert_protocol_holds(cfg, splits, recorder)

    warnings = [text for level, text in recorder.messages if level == "warn"]
    assert any("bundle recognition" in text for text in warnings)


# ══════════════════════════════════════════════════════════════════════
#  IC-3 — the default composition uses the protocol
# ══════════════════════════════════════════════════════════════════════


def test_the_default_experiment_composes_the_grouped_protocol(cfg_default) -> None:
    assert cfg_default.data.split_scheme == "grouped"
    assert float(cfg_default.data.calib_frac) == 0.15
    assert cfg_default.data.single_group_policy == "error"


def test_the_default_selects_on_calib_and_reports_val_test(cfg_default) -> None:
    """The audited run fitted 270+ parameters on `val`, selected on `val`, and
    reported `val` — a maximum over ~944 correlated draws (CHANGES §4.4)."""
    assert cfg_default.evaluation.select_split == "calib"
    assert cfg_default.evaluation.report_split == "val_test"


def test_the_stratified_contrast_arm_differs_in_exactly_one_thing() -> None:
    """A1's two arms must differ in the split and nothing else.

    Including the band count: an A1 whose arms also differed in the input
    representation would measure the two together and attribute the sum to the
    split, which is the shape of the defect A1 exists to quantify.
    """
    grouped = load_experiment_config(overrides=["data=hsi256_grouped"])
    strat = load_experiment_config(overrides=["data=hsi256_stratified"])
    assert grouped.data.split_scheme != strat.data.split_scheme
    for key in (
        "calib_frac",
        "num_bands",
        "num_classes",
        "cutmix_bands",
        "cutmix_spatial",
        "masks_path",
        "morphology_path",
        "patches_data",
    ):
        assert getattr(grouped.data, key) == getattr(strat.data, key), key


# ══════════════════════════════════════════════════════════════════════
#  IC-8 — the runtime invariant
# ══════════════════════════════════════════════════════════════════════


def test_the_default_restores_the_projects_own_runtime_invariant(cfg_default) -> None:
    """TF32 is the one knob here that changes a reported number, and it was on."""
    assert cfg_default.runtime.allow_tf32 is False, "TF32 cuts matmul mantissas 24 -> 11 bits"
    assert int(cfg_default.runtime.num_workers) == -1, "auto, not the run's serialised 0"
    assert str(cfg_default.runtime.compile) == "auto"
    assert str(cfg_default.runtime.amp_dtype) == "bf16"


def test_the_audited_replica_keeps_the_run_s_actual_overrides() -> None:
    """It is the control arm; its value is that it is unmodified."""
    cfg = load_experiment_config(AUDITED_EXPERIMENT)
    assert cfg.runtime.allow_tf32 is True
    assert int(cfg.runtime.num_workers) == 0
    assert str(cfg.runtime.compile) == "off"
    assert float(cfg.grad_clip) == 1.0
    assert float(cfg.aux_gradnorm_alpha) == 0.5


# ══════════════════════════════════════════════════════════════════════
#  IC-5 / IC-6 defaults
# ══════════════════════════════════════════════════════════════════════


def test_gradnorm_is_off_and_the_aux_weight_is_a_constant(cfg_default) -> None:
    assert float(cfg_default.aux_gradnorm_alpha) == 0.0
    assert float(cfg_default.model.aux_head_weight) == 0.2


def test_the_clip_threshold_was_raised_but_the_lr_was_not_co_tuned(cfg_default) -> None:
    """CHANGES §8.1: change it *alone*. Co-tuning both makes neither measurable."""
    audited = load_experiment_config(AUDITED_EXPERIMENT)
    assert float(cfg_default.grad_clip) == 5.0
    assert float(cfg_default.single.max_lr) == float(audited.stage1.max_lr)


# ══════════════════════════════════════════════════════════════════════
#  The primary methodology — one unambiguous path, asserted as config
# ══════════════════════════════════════════════════════════════════════


def test_the_default_experiment_reads_every_acquired_band(cfg_default) -> None:
    """No band selection, no reduction, no index-file subsetting on the primary path.

    This is the study's methodology stated as an assertion rather than as prose,
    so a config change that quietly reintroduced a reduction would fail here
    instead of being discovered in a results table months later.
    """
    assert cfg_default.data.num_bands == 256
    assert not str(cfg_default.data.band_indices_path)
    assert cfg_default.data.patches_data.endswith("/patches.npy")
    assert cfg_default.data.wavelength_path.endswith("/wavelengths.csv")


def test_no_reduced_band_config_can_be_reached_without_naming_it() -> None:
    """Every reduced arm lives under the `ablation/` group and must be asked for."""
    from spectralquadnet.config.compose import config_dir

    data_group = config_dir() / "data"
    primary = sorted(p.stem for p in data_group.glob("*.yaml"))
    reduced = sorted(p.stem for p in (data_group / "ablation").glob("*.yaml"))

    assert primary == ["hsi256_grouped", "hsi256_stratified"], primary
    assert reduced, "the band-selection pathway's arms must still ship"
    for name in primary:
        cfg = load_experiment_config(overrides=[f"data={name}"])
        assert cfg.data.num_bands == 256, name
        assert not str(cfg.data.band_indices_path), name


def test_every_shipped_experiment_declares_its_spectral_regime() -> None:
    """The three composition roots, and which input each is a claim about."""
    from spectralquadnet.config.compose import (
        AUDITED_EXPERIMENT,
        DEFAULT_EXPERIMENT,
        QUADNET_FULL256_EXPERIMENT,
    )

    expected = {
        DEFAULT_EXPERIMENT: (256, "spectral_seed_net", "grouped"),
        QUADNET_FULL256_EXPERIMENT: (256, "spectral_quadnet", "grouped"),
        # The frozen historical replica — the only shipped experiment that is not
        # 256 bands, and the only one whose numbers describe the audited run.
        AUDITED_EXPERIMENT: (40, "spectral_quadnet", "stratified"),
    }
    for name, (bands, arch, scheme) in expected.items():
        cfg = load_experiment_config(name)
        assert cfg.data.num_bands == bands, name
        assert cfg.model.arch == arch, name
        assert cfg.data.split_scheme == scheme, name


def test_the_four_branch_control_differs_from_the_default_in_the_architecture() -> None:
    """A3/A4/A5/A8 run against this, so everything else must be held identical."""
    from spectralquadnet.config.compose import QUADNET_FULL256_EXPERIMENT

    default = load_experiment_config()
    control = load_experiment_config(QUADNET_FULL256_EXPERIMENT)

    assert default.model.arch != control.model.arch
    for group, keys in {
        "data": ("num_bands", "patches_data", "split_scheme", "calib_frac", "cutmix_bands"),
        "evaluation": ("select_split", "report_split", "tta", "bootstrap_samples"),
        "single": ("epochs", "batch", "max_lr", "mixup", "mixup_epochs", "arcface_m"),
    }.items():
        for key in keys:
            assert getattr(getattr(default, group), key) == getattr(
                getattr(control, group), key
            ), f"{group}.{key}"
    assert default.pipeline == control.pipeline == "single"
    assert float(default.grad_clip) == float(control.grad_clip)
    # And the two confounds the audited model config carries are off here.
    assert list(control.model.branch_drop_profile) == [1.0, 1.0, 1.0, 1.0]
    assert control.model.per_class_margin is False
    assert control.model.pairwise_penalty is False
