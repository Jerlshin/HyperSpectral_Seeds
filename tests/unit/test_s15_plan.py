"""S15 — the Y3 replication and its dissection compose to exactly what S14 froze (D33).

``preregistration_s14.json`` lists 10 GPU cells, each with its own seed: Y3 at
seeds 1–2 (grouped folds 0/1 and the stratified contrast) and Y5, Y3's changes
split in two, at seed 0. These tests pin that plan, that nothing S13 already ran
is in it, the code-identity guard (model/training code = ``aed5257``'s), and that
the two dissection arms are an exact partition of Y3's change.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from spectralquadnet.experiments import frozen, s13, s15
from spectralquadnet.models.registry import build_model

LEAN = s13._LEAN


@pytest.fixture(scope="module")
def plan() -> s15.S15Plan:
    return s15.load_plan()


def test_ten_cells_each_with_its_frozen_seed(plan) -> None:
    layout = {(c.arm, c.variant, c.protocol, c.fold, c.seed) for c in plan.cells}
    assert len(plan.cells) == 10 and len(layout) == 10
    assert layout == {
        *(("Y3", "lean_grouped", "grouped", f, s) for f in (0, 1) for s in (1, 2)),
        *(("Y3", "lean_stratified", "stratified", 0, s) for s in (1, 2)),
        *(("Y5", v, "grouped", f, 0) for v in ("desc_only", "spatial_repair") for f in (0, 1)),
    }


def test_the_replication_runs_first(plan) -> None:
    """If a session is cut short, the confirmatory cells are the ones already done."""
    arms = [c.arm for c in plan.cells]
    assert arms == ["Y3"] * 6 + ["Y5"] * 4


def test_nothing_s13_already_ran_is_in_the_plan(plan) -> None:
    ran = {c.name for c in s13.load_plan().cells}
    assert not ran & {c.name for c in plan.cells}
    # Y3's seed-0 cells are S13's — read from there, never re-run here
    assert {"Y3/lean_grouped__f0_s0", "Y3/lean_grouped__f1_s0", "Y3/lean_stratified__f0_s0"} <= ran


def test_the_hashes_are_the_frozen_ones(plan) -> None:
    assert plan.hashes == {k: v for k, (_, v) in s15.PREREGISTRATIONS.items()}
    assert plan.hashes["S14"].startswith("9e182670")


def test_the_regime_is_r1_from_the_parent(plan) -> None:
    assert s15.check_regime_is_r1(plan) == []
    assert all(c.regime == plan.cells[0].regime for c in plan.cells)


def test_every_cell_composes_to_r1_plus_its_arm(plan) -> None:
    problems = [p for c in plan.cells for p in s15.check_cell(c)]
    assert not problems, problems


def test_the_check_catches_a_dissection_arm_with_the_other_half(plan) -> None:
    cell = plan.cell("Y5/desc_only__f0_s0")
    broken = s15.S15Cell(**{**cell.__dict__, "arm_overrides": (*cell.arm_overrides, "model.cbam_min_hw=3")})
    assert any("cbam_min_hw" in p for p in s15.check_cell(broken))


def test_the_dissection_arms_partition_y3(plan) -> None:
    desc = s15.ARM_INTENT[("Y5", "desc_only")]
    spat = s15.ARM_INTENT[("Y5", "spatial_repair")]
    assert not set(desc) & set(spat)
    assert {**desc, **spat} == LEAN == s15.ARM_INTENT[("Y3", "lean_grouped")]
    cfg = s15.compose_cell(plan.cell("Y5/desc_only__f1_s0"))
    assert cfg.model.spectral_descriptor == "snv_morph"
    assert list(cfg.model.spatial_tail_strides) == [2, 2, 2, 2] and cfg.model.cbam_min_hw == 0
    cfg = s15.compose_cell(plan.cell("Y5/spatial_repair__f1_s0"))
    assert cfg.model.spectral_descriptor == "full"
    assert list(cfg.model.spatial_tail_strides) == [2, 2, 2, 1] and cfg.model.cbam_min_hw == 3


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("Y3/lean_grouped__f0_s1", 2_725_700),        # S13 Y3's count (run.json)
        ("Y5/desc_only__f0_s0", 2_849_478 - 41_248),  # shipped − the spectral blocks
        ("Y5/spatial_repair__f0_s0", 2_849_478 - 82_530),  # shipped − the 2 × 2 CBAM gate
    ],
)
def test_each_arm_builds_trains_one_step_and_has_its_parameter_count(plan, name, params) -> None:
    cfg = s15.compose_cell(plan.cell(name))
    torch.manual_seed(0)
    model = build_model(cfg, torch.linspace(0.0, 1.0, 32), input_side=64).train()
    assert sum(p.numel() for p in model.parameters()) == params
    x = torch.rand(2, 32, 64, 64) + 0.1
    out = model(x, labels=torch.tensor([0, 1]), arc_m=0.0, morph=torch.zeros(2, 8))
    (out["main"].logsumexp(1).mean() + out["aux_spatial"].logsumexp(1).mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_every_cell_writes_to_its_own_s15_directory(plan) -> None:
    dirs = [c.spec().output_dir for c in plan.cells]
    assert len(set(dirs)) == len(dirs)
    assert all("/s15/" in d and d.endswith(f"_s{c.seed}") for d, c in zip(dirs, plan.cells, strict=True))
    assert plan.cells[0].spec().command()[-1] == "seed=1"


def test_the_code_is_the_code_the_s13_cells_ran() -> None:
    assert s15.check_code_identity() == []
    digest, n = s15.code_digest()
    assert digest == s15.CODE_REFERENCE_DIGEST and n > 100
    assert "train.py" in s15.code_files(frozen.repo_root())
    assert not any(f.startswith("src/spectralquadnet/experiments/") for f in s15.code_files(frozen.repo_root()))


def test_the_code_guard_sees_a_model_edit_and_ignores_a_runner_edit(tmp_path: Path) -> None:
    root = frozen.repo_root()
    for rel in s15.code_files(root):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, tmp_path / rel)
    assert s15.check_code_identity(tmp_path) == []
    runner = tmp_path / "src/spectralquadnet/experiments/s15.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("# a runner edit is outside the scope\n")
    assert s15.check_code_identity(tmp_path) == []
    target = tmp_path / "src/spectralquadnet/models/spectral_seed_net.py"
    target.write_text(target.read_text() + "\n# drift\n")
    assert s15.check_code_identity(tmp_path) and "aed5257" in s15.check_code_identity(tmp_path)[0]


def test_a_tampered_design_is_refused(tmp_path: Path) -> None:
    root = frozen.repo_root()
    for rel, _ in s15.PREREGISTRATIONS.values():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, tmp_path / rel)
    s15.verify_preregistrations(tmp_path)
    rel = s15.PREREGISTRATIONS["S14"][0]
    payload = json.loads((tmp_path / rel).read_text())
    payload["cells"]["gpu"][0]["seed"] = 0
    (tmp_path / rel).write_text(json.dumps(payload, indent=1))
    with pytest.raises(s15.PreregistrationError, match="edited after freezing"):
        s15.load_plan(tmp_path)


def test_the_provenance_names_the_role_hashes_and_code(plan) -> None:
    cell = plan.cell("Y5/spatial_repair__f1_s0")
    code = s15.code_digest()
    record = s15.provenance(cell, cell.spec(), "cmd", plan.hashes, code)
    assert record["study"] == "S15" and "dissection" in record["role"]
    assert record["seed"] == 0 and record["arm_overrides"] == ["model.spatial_tail_strides=[2,2,2,1]",
                                                               "model.cbam_min_hw=3"]
    assert record["preregistration_sha256"] == plan.hashes
    assert record["code_identity"]["identical"] is True
    rep = s15.provenance(plan.cells[0], plan.cells[0].spec(), "cmd", plan.hashes, code)
    assert "fresh seed" in rep["role"]


def test_a_multi_gpu_cell_launches_under_torchrun(plan) -> None:
    command = plan.cells[0].spec().command(nproc_per_node=2)
    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
