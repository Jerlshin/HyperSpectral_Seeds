"""Verify S21 evidence, prediction arithmetic, links and unchanged historical studies.

--assets additionally checks the large local arrays, native masks and frozen inputs.
This never fits a model or chooses a candidate.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import class_metrics, verify_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", action="store_true")
    args = parser.parse_args()
    evidence = ROOT / "docs/research/evidence/S21_complementary_rgb"
    shared = ROOT / "docs/research/evidence/S20_rgb_pathway"
    checks = {}
    initial = json.loads((shared / "starting_state.json").read_text())
    for name, digest in initial["historical_hashes"].items():
        assert sha256(ROOT / name) == digest, name
    checks["unchanged_historical_files"] = len(initial["historical_hashes"])
    plan_path = evidence / "preregistration.json"
    assert sha256(plan_path) == plan_path.with_suffix(".sha256").read_text().strip()
    plan = json.loads(plan_path.read_text())
    completion = json.loads((evidence / "COMPLETED.json").read_text())
    assert completion["plan_sha256"] == sha256(plan_path)
    for name, digest in completion["files"].items():
        if (evidence / name).exists():
            assert sha256(evidence / name) == digest, name
    checks["frozen_plan_and_copied_result_hashes"] = True
    predictions = pd.read_csv(evidence / "predictions.csv.gz")
    metrics = pd.read_csv(evidence / "metrics.csv")
    per_class = pd.read_csv(evidence / "per_class.csv")
    expected_rows = sum(len(s["val"]) + len(s["test"]) for s in plan["splits"].values())
    assert len(metrics) == 48 and len(predictions) == expected_rows * 24
    assert set(metrics.arm) == set(plan["arms"])
    for (fold, arm), block in predictions.groupby(["fold", "arm"]):
        rows = np.sort(np.r_[plan["splits"][str(fold)]["val"], plan["splits"][str(fold)]["test"]])
        assert np.array_equal(np.sort(block["index"]), rows)
        assert block["index"].is_unique
        stats = class_metrics(block.target.to_numpy(), block.prediction.to_numpy(), 90)
        saved = per_class[(per_class.fold == fold) & (per_class.arm == arm)].sort_values("label")
        for key, actual in stats.items():
            np.testing.assert_allclose(actual, saved[key], atol=1e-14)
        aggregate = metrics[(metrics.fold == fold) & (metrics.arm == arm)].iloc[0]
        np.testing.assert_allclose(stats["f1"].mean(), aggregate.f1, atol=1e-14)
    counts = np.bincount(np.concatenate([np.r_[s["val"], s["test"]] for s in plan["splits"].values()]), minlength=8624)
    assert (counts == 1).all()
    checks["all_8624_rows_held_once"] = True
    checks["prediction_arithmetic_verified"] = 48
    assert len(pd.read_csv(shared / "missing_component_review.csv")) == 16
    assert pd.read_csv(shared / "missing_component_review.csv").failed_gate.notna().all()
    audit = json.loads((shared / "asset_validation.json").read_text())
    assert audit["kernels"] == 8624 and audit["scans"] == 180
    assert audit["k32_different_values"] == 0 and audit["unique_crop_hashes"] == 8624
    checks["identity_and_exclusion_evidence"] = True
    if args.assets:
        verify_plan(plan_path)
        data = ROOT / plan["data"]
        for sid in range(180):
            meta = json.loads((data / "scans" / f"{sid:03d}.json").read_text())
            for suffix, key in [(".npz", "shard_sha256"), ("_masks.npz", "native_masks_sha256")]:
                assert sha256(data / "scans" / f"{sid:03d}{suffix}") == meta[key]
        checks["local_frozen_inputs_and_native_shards"] = True
    for source in (evidence / "code").glob("*.py"):
        ast.parse(source.read_text())
    docs = list((ROOT / "docs/research/studies/S21_complementary_rgb").glob("*.md"))
    docs += [ROOT / "docs/research" / n for n in ["MASTER_RESEARCH_PLAN.md", "RESEARCH_PROGRESS.md", "README.md"]]
    count = 0
    for doc in docs:
        for url in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            if "://" in url or url.startswith("#"):
                continue
            path = url.split("#")[0]
            if path:
                assert (doc.parent / path).exists(), (doc, path)
                count += 1
    checks["local_links"] = count
    figures = ROOT / "docs/research/figures/S21_complementary_rgb"
    for name in ["rgb_and_fusion.png", "spectral_controls.png"]:
        with Image.open(figures / name) as image:
            image.verify()
    checks["figure_files_valid"] = True
    checks["scope"] = "Evidence integrity and arithmetic; no independent-session validation or new GPU results."
    write_json(evidence / "validation.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
