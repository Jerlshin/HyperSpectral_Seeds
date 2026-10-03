"""S13 — the single-seed screen composes to exactly what the frozen files say (D28).

The parent design (``preregistration_s12.json``) is 30 runs; the amendment
(``preregistration_s13.json``) keeps every arm, control, protocol contrast and
grouped fold and runs each cell at seed 0. These tests pin that reduction and
the composition gate ``scripts/run_s13.py --check`` repeats.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spectralquadnet.experiments import frozen, s13


@pytest.fixture(scope="module")
def plan() -> s13.S13Plan:
    return s13.load_plan()


def test_ten_gpu_cells_two_fused_one_seed(plan) -> None:
    assert len(plan.cells) == 10 and len(plan.fused) == 2
    assert plan.seed == 0 and {c.seed for c in plan.cells} == {0}
    layout = {(c.arm, c.variant, c.protocol, c.fold) for c in plan.cells}
    assert layout == {
        *(("Y1", v, "grouped", f) for v in ("spectral_only", "spatial_only") for f in (0, 1)),
        *(("Y2", "mixstyle", "grouped", f) for f in (0, 1)),
        *(("Y3", "lean_grouped", "grouped", f) for f in (0, 1)),
        ("Y3", "lean_stratified", "stratified", 0),
        ("Y4", "within_8020", "stratified", 0),
    }
    assert {(f.fold, f.spectral, f.spatial) for f in plan.fused} == {
        (k, f"Y1/spectral_only__f{k}_s0", f"Y1/spatial_only__f{k}_s0") for k in (0, 1)
    }


def test_the_reduction_removes_only_seeds(plan) -> None:
    """Every (arm, variant, protocol, fold) of the 30-run parent design is still a cell."""
    parent = json.loads(
        (frozen.repo_root() / s13.PREREGISTRATIONS["S12"][0]).read_text()
    )["arms"]
    assert "grouped folds 0,1 × seeds 0,1,2 × {spectral, spatial}" in parent[
        "Y1_decoupled_pathways"
    ]["runs"]
    assert "grouped folds 0,1 × seeds 0,1,2 + stratified seeds 0,1,2" in parent[
        "Y3_lean_architecture"
    ]["runs"]
    # 12 + 6 + 9 + 3 = 30 parent runs → one per seed-free design point.
    assert len(plan.cells) * 3 == 30


def test_the_hashes_are_the_frozen_ones(plan) -> None:
    assert plan.hashes == {k: v for k, (_, v) in s13.PREREGISTRATIONS.items()}
    assert plan.hashes["S12"].startswith("88b377c5")


def test_the_regime_is_read_from_the_parent_and_is_r1(plan) -> None:
    assert s13.check_regime_is_r1(plan) == []
    assert all(c.regime == plan.cells[0].regime for c in plan.cells)


def test_every_cell_composes_to_its_frozen_regime(plan) -> None:
    problems = [p for c in plan.cells for p in s13.check_cell(c)]
    assert not problems, problems


def test_the_check_catches_a_wrong_arm_key(plan) -> None:
    cell = plan.cell("Y3/lean_grouped__f0_s0")
    broken = s13.S13Cell(**{**cell.__dict__, "arm_overrides": (*cell.arm_overrides, "model.cbam_min_hw=0")})
    assert any("cbam_min_hw" in p for p in s13.check_cell(broken))


def test_arm_keys_stay_at_their_defaults_outside_their_arm(plan) -> None:
    mix = s13.compose_cell(plan.cell("Y2/mixstyle__f0_s0"))
    assert mix.model.spatial_mixstyle is True and mix.model.spectral_descriptor == "full"
    y4 = s13.compose_cell(plan.cell("Y4/within_8020__f0_s0"))
    assert float(y4.data.split_eval_frac) == pytest.approx(0.2)
    assert y4.data.split_scheme == "stratified" and list(y4.model.pathways) == ["spatial", "spectral"]
    lean = s13.compose_cell(plan.cell("Y3/lean_stratified__f0_s0"))
    assert list(lean.model.spatial_tail_strides) == [2, 2, 2, 1] and lean.model.cbam_min_hw == 3


def test_every_cell_writes_to_its_own_directory(plan) -> None:
    dirs = [c.spec().output_dir for c in plan.cells] + [str(f.output_dir()) for f in plan.fused]
    assert len(set(dirs)) == len(dirs)
    assert all("/s13/" in d and d.endswith("_s0") for d in dirs)


def test_a_tampered_amendment_is_refused(tmp_path: Path) -> None:
    root = frozen.repo_root()
    for rel, _ in s13.PREREGISTRATIONS.values():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, target)
    s13.verify_preregistrations(tmp_path)
    rel = s13.PREREGISTRATIONS["S13"][0]
    payload = json.loads((tmp_path / rel).read_text())
    payload["deviation"]["seed"] = 1
    (tmp_path / rel).write_text(json.dumps(payload, indent=1))
    with pytest.raises(s13.PreregistrationError, match="edited after freezing"):
        s13.load_plan(tmp_path)


def test_the_provenance_names_the_design_and_both_hashes(plan) -> None:
    cell = plan.cell("Y1/spatial_only__f1_s0")
    record = s13.provenance(cell, cell.spec(), "cmd", plan.hashes)
    assert record["study"] == "S13" and "single-seed" in record["design"]
    assert record["arm_overrides"] == ["model.pathways=[spatial]"]
    assert record["preregistration_sha256"] == plan.hashes
    assert record["regime"] == list(plan.cells[0].regime)


def test_a_multi_gpu_cell_launches_under_torchrun(plan) -> None:
    command = plan.cells[0].spec().command(nproc_per_node=2)
    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
