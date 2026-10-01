"""S11 — the frozen arms X1, X2, X4 compose to exactly what was pre-registered.

S10 §9 step 5 makes this a gate: *"Dry-run every command … check mixup/margin/
aux/clip/epochs/patience in the printed config. Gate: all compose."* These tests
are that gate in the fast tier; ``scripts/run_frozen.py --cfg-job`` repeats it
through ``train.py --cfg job`` itself.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spectralquadnet.experiments import frozen


@pytest.fixture(scope="module")
def plan() -> frozen.FrozenPlan:
    return frozen.load_plan()


def test_the_plan_is_23_cells_in_the_frozen_layout(plan) -> None:
    by_arm = {arm: [c for c in plan.cells if c.arm == arm] for arm in ("X1", "X2", "X4")}
    assert len(by_arm["X1"]) == 9 and len(by_arm["X2"]) == 12 and len(by_arm["X4"]) == 2
    assert {(c.protocol, c.fold, c.seed) for c in by_arm["X1"]} == {
        *(("grouped", f, s) for f in (0, 1) for s in (0, 1, 2)),
        *(("stratified", 0, s) for s in (0, 1, 2)),
    }
    assert {(c.variant, c.fold, c.seed) for c in by_arm["X2"]} == {
        (v, f, s)
        for v in ("no_morph", "spectral_only", "spatial_only")
        for f in (0, 1)
        for s in (0, 1)
    }
    assert {(c.protocol, c.fold, c.seed) for c in by_arm["X4"]} == {
        ("grouped", 0, 0),
        ("stratified", 0, 0),
    }


def test_the_hashes_are_the_frozen_ones(plan) -> None:
    assert plan.hashes == {k: v for k, (_, v) in frozen.PREREGISTRATIONS.items()}


def test_every_cell_composes_to_its_frozen_regime(plan) -> None:
    problems = [p for c in plan.cells for p in frozen.check_cell(c)]
    assert not problems, problems


def test_the_check_catches_a_wrong_regime(plan) -> None:
    cell = next(c for c in plan.cells if c.arm == "X1")
    broken = frozen.FrozenCell(
        **{**cell.__dict__, "arm_overrides": (*cell.arm_overrides, "single.mixup_epochs=110")}
    )
    assert any("mixup_epochs" in p for p in frozen.check_cell(broken))


def test_every_cell_writes_to_its_own_directory(plan) -> None:
    dirs = [c.spec().output_dir for c in plan.cells]
    assert len(set(dirs)) == len(dirs)
    assert all("/s11/" in d for d in dirs)


def test_a_multi_gpu_cell_launches_under_torchrun(plan) -> None:
    command = plan.cells[0].spec().command(nproc_per_node=2)
    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc_per_node=2" in command


def test_a_tampered_preregistration_is_refused(tmp_path: Path) -> None:
    root = frozen.repo_root()
    for rel, _ in frozen.PREREGISTRATIONS.values():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, target)
    frozen.verify_preregistrations(tmp_path)  # untouched copies pass
    rel = frozen.PREREGISTRATIONS["S10"][0]
    payload = json.loads((tmp_path / rel).read_text())
    payload["hypotheses"]["H16"] = payload["hypotheses"]["H16"].replace("0.98", "0.90")
    (tmp_path / rel).write_text(json.dumps(payload, indent=1))
    with pytest.raises(frozen.PreregistrationError, match="edited after freezing"):
        frozen.load_plan(tmp_path)


def test_the_provenance_names_the_source_and_the_deviations(plan) -> None:
    cell = next(c for c in plan.cells if c.variant == "spectral_only")
    spec = cell.spec()
    record = frozen.provenance(cell, spec, "cmd", plan.hashes)
    assert record["arm"] == "X2" and record["arm_overrides"] == ["model.pathways=[spectral]"]
    assert record["preregistration_sha256"] == plan.hashes
    assert any("launcher" in line for line in record["added_by_s11"])
