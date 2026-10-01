"""The ``jsonl`` tracker: strict JSON, one record per call, appended across resumes."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import torch

from spectralquadnet.tracking import JsonlTracker, build_tracker
from spectralquadnet.tracking.jsonl_tracker import METRICS_FILE
from spectralquadnet.tracking.multi_tracker import MultiTracker


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_call_is_one_strict_json_record(tmp_path) -> None:
    path = tmp_path / METRICS_FILE
    trk = JsonlTracker(path)
    trk.log_hyperparams({"seed": 0, "data": {"num_bands": 32}})
    trk.progress_start("single", 150, "Single")
    trk.log_scalars(
        {
            "train/loss": torch.tensor(1.5),
            "val/f1": np.float32(0.25),
            "bad": math.nan,
            "worse": float("inf"),
            "steps": 12,
        },
        step=3,
    )
    trk.log_row("single", {"Loss": "1.5000", "ckpt": "✓"}, step=3)
    trk.log_table("session/val_test", [{"session": 0, "recall": np.float64(0.5)}], step=3)
    trk.log_message("careful", level="warn")
    trk.progress_stop("single")
    trk.close()
    trk.close()  # idempotent

    lines = path.read_text().splitlines()
    for line in lines:
        json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    records = _records(path)
    events = [r["event"] for r in records]
    assert events == [
        "run_start",
        "hyperparams",
        "stage_start",
        "scalars",
        "epoch",
        "table",
        "message",
        "stage_stop",
        "run_end",
    ]
    scalars = records[3]
    assert scalars["step"] == 3
    assert scalars["metrics"] == {
        "train/loss": 1.5,
        "val/f1": 0.25,
        "bad": None,
        "worse": None,
        "steps": 12,
    }
    assert records[4]["cells"]["ckpt"] == "✓"
    assert records[5]["rows"] == [{"session": 0, "recall": 0.5}]
    assert all("t" in r and "elapsed_s" in r for r in records)


def test_a_resumed_run_appends_to_the_same_record(tmp_path) -> None:
    path = tmp_path / METRICS_FILE
    for step in (1, 2):
        trk = JsonlTracker(path)
        trk.log_scalar("train/loss", 1.0 / step, step=step)
        trk.close()
    events = [r["event"] for r in _records(path)]
    assert events.count("run_start") == 2
    assert [r["step"] for r in _records(path) if r["event"] == "scalars"] == [1, 2]


def test_the_backend_is_reachable_from_config(tmp_path) -> None:
    cfg = SimpleNamespace(
        output_dir=str(tmp_path / "run"),
        tracking=SimpleNamespace(backend="multi", backends=["console", "jsonl"]),
        runtime=SimpleNamespace(progress="auto"),
    )
    trk = build_tracker(cfg)
    assert isinstance(trk, MultiTracker)
    trk.log_scalars({"x": 1.0}, step=1)
    trk.close()
    assert (tmp_path / "run" / METRICS_FILE).exists()


def test_the_shipped_tracking_configs_compose() -> None:
    from spectralquadnet.config.compose import load_experiment_config

    cfg = load_experiment_config(overrides=["tracking=console_jsonl"])
    assert cfg.tracking.backend == "multi"
    assert list(cfg.tracking.backends) == ["console", "jsonl"]
    assert load_experiment_config(overrides=["tracking=jsonl"]).tracking.backend == "jsonl"
