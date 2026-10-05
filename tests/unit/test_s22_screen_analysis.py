"""The amended scorer accepts both directions at one seed and rejects wrong cells."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectralquadnet.data.prep.multimodal import sha256

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_s22_cuda_screening.py"


def seal(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    path.with_suffix(".sha256").write_text(sha256(path))


@pytest.mark.parametrize("wrong_seed", [False, True])
def test_two_fold_one_seed_analysis_is_fail_closed(tmp_path, monkeypatch, wrong_seed):
    monkeypatch.chdir(tmp_path)
    labels = np.repeat(np.arange(90), 4)
    data = tmp_path / "dataset_u430k32"
    data.mkdir()
    np.save(data / "labels.npy", labels)
    pd.DataFrame({"label": np.repeat(np.arange(90), 2),
                  "session_id": [8 if c < 17 and g else 0 for c in range(90) for g in range(2)]}).to_csv(data / "scan_table.csv", index=False)
    splits = {}
    for f in (0, 1):
        te = np.sort(np.r_[np.arange(90) * 4 + 2*f, np.arange(90) * 4 + 2*f+1])
        ca = np.arange(90) * 4 + 2*(1-f)
        splits[str(f)] = {"calib": ca.tolist(), "val": te[::2].tolist(), "test": te[1::2].tolist()}
    part = tmp_path / "partitions.json"
    part.write_text(json.dumps({"splits": splits, "temperatures": [1.0]}))
    plan = tmp_path / "plan.json"
    seal(plan, {"input_hashes": {}, "analysis_inputs": {}, "partition_plan": str(part),
                "cells": [{"fold": f, "seed": 0} for f in (0, 1)]})
    seal(tmp_path / "configs/research/s22_analysis_code05.json", {
        "input_hashes": {str(SCRIPT): sha256(SCRIPT)}, "training_plan_sha256": sha256(plan)})
    for fold in (0, 1):
        root = tmp_path / f"runs/f{fold}_s0"
        res = root / "results"
        res.mkdir(parents=True)
        (root / "provenance.json").write_text(json.dumps({
            "training_plan_sha256": sha256(plan), "partition_plan_sha256": sha256(part),
            "fold": fold, "seed": 1 if wrong_seed else 0, "action": "train"}))
        (res / "run.json").write_text("{}")
        ca = np.array(splits[str(fold)]["calib"])
        te = np.sort(np.r_[splits[str(fold)]["val"], splits[str(fold)]["test"]])
        baseline = np.where(labels[te] < 40, (labels[te]+1) % 90, labels[te])
        logits = np.zeros((len(te), 90))
        logits[np.arange(len(te)), baseline] = 2
        cal = np.zeros((len(ca), 90))
        cal[np.arange(len(ca)), labels[ca]] = 2
        np.savez(res / "logits_calib_tta.npz", rows=ca, targets=labels[ca], logits=cal)
        np.savez(res / "logits_val_test_tta.npz", rows=te, targets=labels[te], logits=logits)
        np.save(res / "rows_val_test_tta.npy", te)
        np.save(res / "preds_val_test_tta.npy", baseline)
        rgb = np.eye(90)[labels[te]]
        prob = tmp_path / f"outputs/s21_rgb_study/probabilities_f{fold}.npz"
        prob.parent.mkdir(parents=True, exist_ok=True)
        np.savez(prob, indices=te, dino_rgb=rgb)
    out = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--plan", str(plan), "--runs", str(tmp_path / "runs"), "--output", str(out)])
    if wrong_seed:
        with pytest.raises(ValueError, match="identity"):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        assert not out.exists()
    else:
        runpy.run_path(str(SCRIPT), run_name="__main__")
        result = json.loads((out / "hypothesis.json").read_text())
        assert result["H40_screen"] and result["development_gate"]
        assert "not evaluated" in result["H40_original_confirmatory"]
        assert len(pd.read_csv(out / "metrics.csv")) == 4
        assert len(pd.read_csv(out / "per_class.csv")) == 360
        with pytest.raises(FileExistsError):
            runpy.run_path(str(SCRIPT), run_name="__main__")
