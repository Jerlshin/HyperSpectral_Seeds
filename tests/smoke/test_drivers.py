"""The experiment drivers run, plan correctly, and produce their artifacts.

These are the CLIs a reviewer touches first, so a broken ``--dry-run`` is a
broken front door. They are fast — the sweeps are planned, not executed — except
:func:`test_a_two_cell_ablation_runs_end_to_end`, which actually trains two tiny
cells so the subprocess contract (argv, exit codes, resumability, aggregation)
is exercised rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectralquadnet.experiments import registry
from spectralquadnet.experiments.cli import main as cli_main

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]


# ══════════════════════════════════════════════════════════════════════
#  The CLI's read-only subcommands
# ══════════════════════════════════════════════════════════════════════


def test_list_prints_the_grid_and_exits_clean(capsys) -> None:
    assert cli_main(["list"]) == 0
    out = capsys.readouterr().out
    assert "RUN FIRST: A12" in out
    for key in registry.ABLATIONS:
        assert key in out
    assert "A9" in out, "the analysis-only experiment must be listed too"


def test_list_fails_loudly_if_the_registry_is_malformed(monkeypatch, capsys) -> None:
    """A grid that cannot be trusted must not be runnable."""
    broken = registry.Ablation(
        key="BAD",
        question="?",
        decision_rule="x" * 50,
        arms=(registry.Arm("only"),),  # one arm is not an ablation
    )
    monkeypatch.setitem(registry.ABLATIONS, "BAD", broken)
    assert cli_main(["list"]) == 1


def test_a_protocol_dry_run_prints_one_command_per_cell(tmp_path, capsys) -> None:
    assert cli_main(["protocol", "--dry-run", "--output-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out

    assert "Constraints that must be stated in the paper" in out
    assert "zero within-class acquisition variance" in out

    commands = [line for line in out.splitlines() if "train.py" in line]
    assert len(commands) == 12, "2 folds x 3 seeds, plus a matched contrast arm"
    for command in commands:
        assert "--config-name=" in command
        assert "data.split_fold=" in command and "seed=" in command


def test_an_ablation_dry_run_prints_its_question_and_its_rule(tmp_path, capsys) -> None:
    assert cli_main(["ablate", "A1", "--dry-run", "--output-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert registry.ABLATIONS["A1"].question in out
    assert "rule:" in out
    assert len([line for line in out.splitlines() if "train.py" in line]) == 9


def test_a_dry_run_writes_nothing(tmp_path) -> None:
    cli_main(["ablate", "A12", "--dry-run", "--output-root", str(tmp_path)])
    assert not list(tmp_path.rglob("*.json"))


def test_arms_can_be_selected_individually(tmp_path, capsys) -> None:
    """Any single number must be reproducible without running the grid."""
    cli_main(
        ["ablate", "A3", "--arms", "bc", "--dry-run", "--output-root", str(tmp_path), "--seeds", "0"]
    )
    commands = [line for line in capsys.readouterr().out.splitlines() if "train.py" in line]
    assert len(commands) == 2, "one arm x 2 folds x 1 seed"
    assert all("enabled_branches=[b,c]" in c for c in commands)


def test_aggregate_on_an_empty_root_is_a_warning_not_a_crash(tmp_path) -> None:
    assert cli_main(["aggregate", "--output-root", str(tmp_path)]) == 0


# ══════════════════════════════════════════════════════════════════════
#  A real two-cell sweep
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def tiny_ablation(monkeypatch, synthetic_dataset, tmp_path):
    """A2-shaped ablation over the synthetic cube: two arms, one fold, one seed."""
    from _helpers import tiny_overrides as _overrides

    shared = tuple(
        o for o in _overrides(synthetic_dataset, tmp_path / "unused") if not o.startswith("output_dir")
        and not o.startswith("hydra.run.dir")
    )
    ablation = registry.Ablation(
        key="SMOKE",
        question="Does the driver work?",
        decision_rule="x" * 50,
        arms=(
            registry.Arm("wide", overrides=(*shared, "model.spatial_width_mult=0.25")),
            registry.Arm("narrow", overrides=(*shared, "model.spatial_width_mult=0.125")),
        ),
        seeds=(0,),
        folds=(0,),
        reference="wide",
    )
    monkeypatch.setitem(registry.ABLATIONS, "SMOKE", ablation)
    return ablation


def test_a_two_cell_ablation_runs_end_to_end(tiny_ablation, tmp_path) -> None:
    """The subprocess contract: argv, exit code, artifacts, aggregation."""
    root = tmp_path / "experiments"
    assert cli_main(["ablate", "SMOKE", "--output-root", str(root)]) == 0

    sweep = json.loads((root / "SMOKE" / "sweep.json").read_text())
    assert sweep["n_total"] == 2
    assert sweep["n_failed"] == 0, sweep["runs"]

    aggregate = json.loads((root / "SMOKE" / "aggregate.json").read_text())
    assert {a["arm"] for a in aggregate["arms"]} == {"wide", "narrow"}
    for arm in aggregate["arms"]:
        assert arm["n_runs"] == 1
        assert 0.0 <= arm["macro_f1_mean"] <= 1.0
        assert arm["parameters"] > 0

    # A delta with an interval, and the interval's relation to zero recorded.
    assert "narrow" in aggregate["paired_deltas"]
    delta = aggregate["paired_deltas"]["narrow"]
    assert delta["ci_lo"] <= delta["delta"] <= delta["ci_hi"]
    assert isinstance(delta["crosses_zero"], bool)

    table = (root / "SMOKE" / "arms.md").read_text()
    assert "macro-F1 mean" in table and "Δ vs ref" in table


def test_rerunning_the_sweep_skips_completed_cells(tiny_ablation, tmp_path) -> None:
    root = tmp_path / "experiments"
    assert cli_main(["ablate", "SMOKE", "--output-root", str(root)]) == 0
    assert cli_main(["ablate", "SMOKE", "--output-root", str(root)]) == 0

    sweep = json.loads((root / "SMOKE" / "sweep.json").read_text())
    assert all(run["status"] == "skipped" for run in sweep["runs"]), (
        "a 60-run grid must survive an interruption"
    )


def test_a_failing_cell_is_reported_and_the_command_exits_non_zero(
    monkeypatch, tmp_path
) -> None:
    """One bad cell must not be mistaken for a finished grid."""
    broken = registry.Ablation(
        key="BROKEN",
        question="Does a failure surface?",
        decision_rule="x" * 50,
        arms=(
            registry.Arm("a", overrides=("data.num_classes=notanumber",)),
            registry.Arm("b", overrides=("data.num_classes=notanumber",)),
        ),
        seeds=(0,),
        folds=(0,),
    )
    monkeypatch.setitem(registry.ABLATIONS, "BROKEN", broken)

    root = tmp_path / "experiments"
    assert cli_main(["ablate", "BROKEN", "--output-root", str(root)]) == 1

    sweep = json.loads((root / "BROKEN" / "sweep.json").read_text())
    assert sweep["n_failed"] == 2
    for run in sweep["runs"]:
        assert Path(run["log"]).exists(), "a failed cell's traceback is kept"


def test_the_report_assembles_whatever_is_on_disk(tiny_ablation, tmp_path, capsys) -> None:
    root = tmp_path / "experiments"
    cli_main(["ablate", "SMOKE", "--output-root", str(root)])
    assert cli_main(["report", "--output-root", str(root)]) == 0

    report = (root / "REPORT.md").read_text()
    assert "Protocol constraints" in report
    assert "regenerable from the run directories alone" in report
