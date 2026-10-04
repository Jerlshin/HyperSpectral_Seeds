"""Explicit training partitions are validated before fitted preprocessing."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spectralquadnet.data.prep.multimodal import sha256
from spectralquadnet.engine.pipelines.context import validate_split_override
from spectralquadnet.experiments.complementary_split import complementary_split
from spectralquadnet.experiments.frozen_rows import load_frozen_rows


def make_plan(tmp_path: Path):
    groups = np.repeat(np.arange(6), 7)
    labels = groups // 2
    for name, value in [("labels", labels), ("groups", groups)]:
        np.save(tmp_path / f"{name}.npy", value)
    plan = tmp_path / "plan.json"
    payload = {"data": str(tmp_path), "input_hashes": {
        str(tmp_path / f"{n}.npy"): sha256(tmp_path / f"{n}.npy") for n in ["labels", "groups"]},
        "splits": {"0": {k: v.tolist() for k, v in complementary_split(labels, groups, fold=0).items()}}}
    plan.write_text(json.dumps(payload))
    plan.with_suffix(".sha256").write_text(sha256(plan))
    return plan, labels, groups


def test_frozen_rows_fail_closed_on_manifest_and_identity_changes(tmp_path):
    plan, labels, groups = make_plan(tmp_path)
    bundle = load_frozen_rows(plan, tmp_path / "labels.npy", tmp_path / "groups.npy", 0)
    validate_split_override(bundle, labels, groups)
    with pytest.raises(ValueError, match="partition"):
        validate_split_override(replace(bundle, train=np.r_[bundle.train, bundle.train[:1]]), labels, groups)
    train, test = bundle.train.copy(), bundle.test.copy()
    train[0], test[0] = test[0], train[0]
    with pytest.raises(ValueError, match="leaks"):
        validate_split_override(replace(bundle, train=train, test=test), labels, groups)
    np.save(tmp_path / "labels.npy", labels + 1)
    with pytest.raises(ValueError, match="identity"):
        load_frozen_rows(plan, tmp_path / "labels.npy", tmp_path / "groups.npy", 0)
    plan.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        load_frozen_rows(plan, tmp_path / "labels.npy", tmp_path / "groups.npy", 0)
