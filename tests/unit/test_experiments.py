"""IC-12 — the ablation grid, the protocol sweep and their aggregation.

The finding this whole package exists to prevent recurring: CHANGES §5.5
documents 21 ablation levers and **zero** were pulled, so not one of the design
hypotheses behind four branches, three stages and eleven auxiliary mechanisms
was ever tested.

These tests assert the properties that make the grid trustworthy rather than
merely present: that every ablation has a pre-registered decision rule, that its
arms differ only in what they claim to, that a cell is independently executable
and idempotent, and that the aggregator reports a missing cell instead of
absorbing it into a mean.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spectralquadnet.experiments import protocol, registry
from spectralquadnet.experiments.aggregate import (
    arm_of,
    discover_runs,
    paired_delta,
    summarise_arms,
)
from spectralquadnet.experiments.runner import expand, plan
from spectralquadnet.reporting.artifacts import RunArtifacts
from spectralquadnet.reporting.metrics import mean_and_range, paired_bootstrap_delta, score
from spectralquadnet.reporting.tables import collect_runs, leakage_gap_table, protocol_table

# ══════════════════════════════════════════════════════════════════════
#  The registry
# ══════════════════════════════════════════════════════════════════════


def test_every_registered_ablation_is_well_formed() -> None:
    assert registry.validate_all() == []


def test_the_grid_covers_the_experiments_changes_asks_for() -> None:
    """A1–A12 minus A9, which is analysis rather than a training run."""
    expected = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A10", "A11", "A12"}
    assert set(registry.ABLATIONS) == expected
    assert registry.ANALYSIS_ONLY == ("A9",)


def test_variance_is_measured_first() -> None:
    """Until sigma is known, no delta in the grid means anything."""
    assert registry.RUN_FIRST == "A12"
    assert registry.ABLATIONS["A12"].depends_on == ()
    assert len(registry.ABLATIONS["A12"].seeds) >= 5, "sigma needs more than three draws"


@pytest.mark.parametrize("key", sorted(registry.ABLATIONS))
def test_every_ablation_states_a_question_and_a_decision_rule(key: str) -> None:
    """An ablation without a pre-registered rule is an invitation to read
    whichever number is convenient after the fact."""
    ablation = registry.ABLATIONS[key]
    assert ablation.question.endswith("?")
    assert len(ablation.decision_rule) > 40


def test_a1_is_the_leakage_gap_and_has_both_protocols() -> None:
    arms = {a.name for a in registry.ABLATIONS["A1"].arms}
    assert arms == {"grouped", "stratified"}


def test_a8_arms_are_the_same_driver_stopped_at_different_points() -> None:
    """Three separately-written loops would make a difference between them
    unattributable to the stages."""
    from spectralquadnet.engine.pipelines import PIPELINES
    from spectralquadnet.engine.pipelines.three_stage import PIPELINE_LAST_STAGE

    for name in ("stage1_only", "stage1_stage2", "three_stage"):
        assert PIPELINES[name] is PIPELINES["three_stage"]
    assert PIPELINE_LAST_STAGE == {"stage1_only": 1, "stage1_stage2": 2, "three_stage": 3}


def test_a3_uses_symmetric_dropout_so_it_measures_branches_not_the_policy(cfg) -> None:
    """Branch C's 87% influence is confounded by never being dropped."""
    a3 = registry.ABLATIONS["A3"]
    assert a3.config == registry.AUDITED_CONFIG
    assert all("enabled_branches" in " ".join(arm.overrides) for arm in a3.arms)


def test_the_reference_arm_exists_wherever_deltas_are_reported() -> None:
    for ablation in registry.ABLATIONS.values():
        if ablation.reference is not None:
            assert ablation.reference in {a.name for a in ablation.arms}


def test_get_names_the_alternatives_when_the_key_is_wrong() -> None:
    with pytest.raises(KeyError, match="Available"):
        registry.get("A99")


# ══════════════════════════════════════════════════════════════════════
#  Expansion and planning
# ══════════════════════════════════════════════════════════════════════


def test_an_ablation_expands_to_arms_times_folds_times_seeds(tmp_path) -> None:
    specs = expand(registry.ABLATIONS["A10"], output_root=tmp_path)
    assert len(specs) == 4 * 2 * 3
    assert len({s.output_dir for s in specs}) == len(specs), "every cell is its own directory"


def test_an_arm_that_pins_its_own_fold_is_not_duplicated(tmp_path) -> None:
    """A1's stratified arm has no folds to rotate; two folds of it is one cell twice."""
    specs = expand(registry.ABLATIONS["A1"], output_root=tmp_path)
    stratified = [s for s in specs if s.arm == "stratified"]
    assert len(stratified) == 3, "3 seeds, not 6"
    assert len([s for s in specs if s.arm == "grouped"]) == 6


def test_a_cell_prints_the_exact_command_that_reproduces_it(tmp_path) -> None:
    """The property that makes a single number checkable without the driver."""
    spec = expand(registry.ABLATIONS["A12"], output_root=tmp_path)[0]
    shell = spec.shell()
    assert "train.py" in shell
    assert "--config-name=" in shell
    assert "data.split_fold=" in shell and "seed=" in shell


def test_fold_placeholders_are_substituted_at_expansion(tmp_path) -> None:
    """A2's per-fold band arrays resolve without the config knowing about the sweep."""
    specs = expand(registry.ABLATIONS["A2"], output_root=tmp_path, arms=("bands_within_fold",))
    for spec in specs:
        assert "{fold}" not in " ".join(spec.overrides)
        assert f"fold{spec.fold}" in " ".join(spec.overrides)


def test_a_completed_cell_is_skipped_so_an_interrupted_grid_resumes(tmp_path) -> None:
    specs = expand(registry.ABLATIONS["A12"], output_root=tmp_path)
    todo, done = plan(specs)
    assert len(todo) == len(specs) and not done

    RunArtifacts.for_run(specs[0].output_dir).write_manifest({"run": {}})
    todo, done = plan(specs)
    assert len(done) == 1 and len(todo) == len(specs) - 1

    forced_todo, forced_done = plan(specs, force=True)
    assert len(forced_todo) == len(specs) and not forced_done


def test_the_protocol_sweep_is_two_folds_by_three_seeds_plus_a_contrast(tmp_path) -> None:
    specs = protocol.build_specs(tmp_path)
    grouped = [s for s in specs if s.arm == "grouped"]
    stratified = [s for s in specs if s.arm == "stratified"]
    assert sorted({s.fold for s in grouped}) == [0, 1]
    assert len(grouped) == 6
    assert len(stratified) == len(grouped), (
        "the two arms must be built from the same number of runs, or the gap is "
        "partly an artefact of sample size"
    )


def test_the_protocol_states_its_three_constraints() -> None:
    lines = protocol.constraints()
    assert len(lines) == 3
    assert any("zero within-class acquisition variance" in line for line in lines)
    assert any(
        "not\nmutually independent" in line or "not mutually independent" in line for line in lines
    )
    assert any("Never their maximum" in line for line in lines)


# ══════════════════════════════════════════════════════════════════════
#  Aggregation
# ══════════════════════════════════════════════════════════════════════


def _write_run(root, arm: str, fold: int, seed: int, macro_f1: float, scheme: str = "grouped"):
    """A minimal but structurally real run directory."""
    run_dir = root / f"{arm}__f{fold}_s{seed}"
    artifacts = RunArtifacts.for_run(run_dir)
    rng = np.random.default_rng(seed)
    targets = np.repeat(np.arange(10), 5)
    preds = targets.copy()
    n_wrong = int(round((1 - macro_f1) * len(targets)))
    if n_wrong:
        idx = rng.choice(len(targets), size=n_wrong, replace=False)
        preds[idx] = (preds[idx] + 1) % 10
    result = score(targets, preds, num_classes=10, split="val_test_tta", n_boot=0)
    artifacts.write_predictions("val_test_tta", preds, targets)
    artifacts.write_result(result)
    artifacts.write_manifest(
        {
            "run": {
                "arch": "spectral_seed_net",
                "pipeline": "single",
                "split_scheme": scheme,
                "split_fold": fold,
                "seed": seed,
                "parameters": 2_819_830,
                "select_split": "calib",
            },
            "results": {"tta": result.as_dict()},
        }
    )
    return run_dir


def test_arm_fold_and_seed_are_recoverable_from_the_directory_name() -> None:
    from pathlib import Path

    assert arm_of(Path("outputs/A1/grouped__f1_s2")) == ("grouped", 1, 2)
    assert arm_of(Path("outputs/handmade")) == ("handmade", 0, 0)


def test_runs_are_discovered_and_aggregated_to_mean_and_range(tmp_path) -> None:
    for i, f1 in enumerate([0.60, 0.64, 0.68]):
        _write_run(tmp_path, "grouped", fold=i % 2, seed=i, macro_f1=f1)

    run_dirs = discover_runs(tmp_path)
    assert len(run_dirs) == 3

    summaries = summarise_arms(run_dirs)
    assert len(summaries) == 1
    stats = summaries[0].stats
    assert stats["n"] == 3
    assert stats["min"] < stats["mean"] < stats["max"]
    assert stats["range"] == pytest.approx(stats["max"] - stats["min"])


def test_a_run_with_no_manifest_is_reported_missing_not_absorbed(tmp_path) -> None:
    _write_run(tmp_path, "grouped", 0, 0, 0.6)
    (tmp_path / "grouped__f0_s1").mkdir(parents=True)  # crashed before scoring
    assert len(discover_runs(tmp_path)) == 1


def test_mean_and_range_reports_the_run_count(tmp_path) -> None:
    """A mean of one must not look like a mean of six."""
    assert mean_and_range([0.5])["n"] == 1
    assert mean_and_range([0.5])["sd"] == 0.0
    assert mean_and_range([])["n"] == 0


def test_the_paired_delta_pairs_on_the_cell_and_needs_matching_targets(tmp_path) -> None:
    ref = [_write_run(tmp_path / "ref", "a", f, s, 0.60) for f in (0, 1) for s in (0, 1)]
    arm = [_write_run(tmp_path / "arm", "b", f, s, 0.70) for f in (0, 1) for s in (0, 1)]

    paired = paired_delta(ref, arm, "val_test_tta", num_classes=10, n_boot=64)
    assert paired is not None
    delta, lo, hi = paired
    assert delta > 0
    assert lo <= delta <= hi


def test_arms_evaluated_on_different_splits_are_not_paired(tmp_path) -> None:
    """A1's arms use different protocols, so there is nothing to pair."""
    ref = [_write_run(tmp_path / "ref", "grouped", 0, 0, 0.6)]
    other = tmp_path / "other" / "stratified__f0_s0"
    artifacts = RunArtifacts.for_run(other)
    targets = np.repeat(np.arange(10), 7)  # a different eval set entirely
    artifacts.write_predictions("val_test_tta", targets, targets)
    artifacts.write_manifest({"run": {}, "results": {}})

    assert paired_delta(ref, [other], "val_test_tta", num_classes=10, n_boot=16) is None


def test_a_paired_bootstrap_interval_brackets_the_observed_delta() -> None:
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 5, size=200)
    good = targets.copy()
    bad = targets.copy()
    bad[rng.choice(200, 60, replace=False)] = 0

    delta, interval = paired_bootstrap_delta(targets, bad, good, num_classes=5, n_boot=200)
    assert delta > 0
    assert interval.lo <= delta <= interval.hi
    assert interval.lo > 0, "a large, real improvement should not straddle zero"


# ══════════════════════════════════════════════════════════════════════
#  Tables
# ══════════════════════════════════════════════════════════════════════


def test_the_protocol_table_groups_by_architecture_and_scheme(tmp_path) -> None:
    for seed in range(3):
        _write_run(tmp_path, "grouped", 0, seed, 0.60 + 0.01 * seed, scheme="grouped")
        _write_run(tmp_path, "stratified", 0, 10 + seed, 0.85 + 0.01 * seed, scheme="stratified")

    records = collect_runs(discover_runs(tmp_path))
    assert len(records) == 6

    _headers, rows, arms = protocol_table(records)
    assert len(rows) == 2
    assert set(arms) == {"spectral_seed_net/grouped", "spectral_seed_net/stratified"}


def test_the_leakage_gap_table_is_the_projects_headline_result(tmp_path) -> None:
    for seed in range(3):
        _write_run(tmp_path, "grouped", 0, seed, 0.60, scheme="grouped")
        _write_run(tmp_path, "stratified", 0, 10 + seed, 0.85, scheme="stratified")

    headers, rows = leakage_gap_table(collect_runs(discover_runs(tmp_path)))
    assert len(rows) == 1
    # `_write_run` corrupts a rounded number of labels, so the realised F1s are
    # near 0.60/0.85 rather than exactly them; the gap is asserted as a signed
    # quantity of about the right size, which is what the table reports.
    assert headers[-1] == "gap"
    gap = float(rows[0][-1])
    assert 0.15 < gap < 0.35, "F1_stratified - F1_grouped, and the sign is the point"


def test_the_gap_table_is_empty_without_both_arms(tmp_path) -> None:
    """Better an absent table than one built from a single protocol."""
    _write_run(tmp_path, "grouped", 0, 0, 0.6, scheme="grouped")
    _headers, rows = leakage_gap_table(collect_runs(discover_runs(tmp_path)))
    assert rows == []


# ══════════════════════════════════════════════════════════════════════
#  Artifacts
# ══════════════════════════════════════════════════════════════════════


def test_the_results_tree_has_the_layout_aggregators_depend_on(tmp_path) -> None:
    run_dir = _write_run(tmp_path, "grouped", 0, 0, 0.7)
    results = run_dir / "results"
    for name in (
        "run.json",
        "metrics_val_test_tta.json",
        "confusion_val_test_tta.npy",
        "confusion_val_test_tta.csv",
        "per_class_val_test_tta.csv",
        "preds_val_test_tta.npy",
        "targets_val_test_tta.npy",
    ):
        assert (results / name).exists(), name


def test_the_manifest_merges_rather_than_overwrites(tmp_path) -> None:
    """A run scores several splits at different times; a crash between two of
    them must not strand the first."""
    artifacts = RunArtifacts.for_run(tmp_path / "run")
    artifacts.write_manifest({"run": {"seed": 1}, "results": {"no_tta": {"macro_f1": 0.5}}})
    artifacts.write_manifest({"results": {"tta": {"macro_f1": 0.6}}})

    payload = json.loads((tmp_path / "run" / "results" / "run.json").read_text())
    assert payload["run"]["seed"] == 1
    assert set(payload["results"]) == {"no_tta", "tta"}


def test_a_corrupt_manifest_does_not_strand_the_run(tmp_path) -> None:
    artifacts = RunArtifacts.for_run(tmp_path / "run")
    (tmp_path / "run" / "results" / "run.json").write_text("{ truncated")
    artifacts.write_manifest({"run": {"seed": 2}})
    payload = json.loads((tmp_path / "run" / "results" / "run.json").read_text())
    assert payload["run"]["seed"] == 2


def test_scoring_reports_everything_changes_19_4_requires() -> None:
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 5, size=100)
    preds = targets.copy()
    preds[:10] = (preds[:10] + 1) % 5

    result = score(targets, preds, num_classes=5, split="test", n_boot=100)
    assert 0.0 < result.macro_f1 <= 1.0
    assert result.balanced_accuracy > 0
    assert set(result.per_class_recall) == set(range(5))
    assert result.confusion.shape == (5, 5)
    assert result.macro_f1_ci is not None
    assert result.macro_f1_ci.lo <= result.macro_f1 <= result.macro_f1_ci.hi


def test_macro_f1_uses_a_fixed_label_set_so_resamples_are_comparable() -> None:
    """A bootstrap sample missing a rare class must not change the denominator."""
    from spectralquadnet.reporting.metrics import macro_f1

    targets = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    # Perfect on the two classes present, but averaged over ten denominators.
    assert macro_f1(targets, preds, num_classes=10) == pytest.approx(0.2)
