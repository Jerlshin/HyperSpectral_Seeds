"""Verify S22 preparation and pinned inputs without running training/analysis."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import verify_plan


def main() -> None:
    evidence = ROOT / "docs/research/evidence/S22_complementary_v5"
    path = ROOT / "configs/research/s22_complementary_v5.json"
    plan = verify_plan(path)
    assert sha256(path) == sha256(evidence / "preregistration.json")
    assert (path.with_suffix(".sha256").read_bytes()
            == (evidence / "preregistration.sha256").read_bytes())
    for name, digest in plan["analysis_inputs"].items():
        assert sha256(ROOT / name) == digest, name
    assert {(c["fold"], c["seed"]) for c in plan["cells"]} == {(f, s) for f in [0, 1] for s in [0, 1, 2]}
    profile = json.loads((evidence / "profile_final/profile.json").read_text())
    assert profile["held_out_rows_scored"] == 0 and profile["finite_gradients"]
    assert profile["parameters"] == 2725700 and profile["partition_scheme"] == "frozen_complementary"
    for doc in (ROOT / "docs/research/studies/S22_complementary_v5").glob("*.md"):
        for url in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            if "://" not in url and not url.startswith("#"):
                assert (doc.parent / url.split("#")[0]).exists(), url
    complete = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "outputs/s22_complementary_v5").glob("f*_s*/results/run.json"))
    write_json(evidence / "validation.json", {
        "frozen_plan_sha256": sha256(path),
        "training_input_hashes_verified": len(plan["input_hashes"]),
        "analysis_input_hashes_verified": len(plan["analysis_inputs"]),
        "six_cells_and_cpu_profile_verified": True,
        "local_completed_gpu_cells": complete,
        "scope": "Preparation integrity; does not validate GPU execution or claim a result.",
    })
    print((evidence / "validation.json").read_text())


if __name__ == "__main__":
    main()
