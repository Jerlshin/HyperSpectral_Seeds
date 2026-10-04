#!/usr/bin/env python
"""Prepared v5 rebaseline on frozen S21 rows; profile CPU or train an explicit GPU cell.

Requires an explicit profile or train action; never silently starts GPU training. The training plan must be
frozen after profiling. Existing S13/S15/S17 runners and historical defaults remain
unchanged. No saved v5 network may substitute for retraining on these acquisitions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.engine.pipelines import build_run_context, resolve_pipeline
from spectralquadnet.experiments.frozen_rows import load_frozen_rows
from spectralquadnet.experiments.rgb_probe import verify_plan
from spectralquadnet.tracking import build_tracker
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.device import resolve_device
from spectralquadnet.utils.distributed import init_distributed, shutdown


def prepare_output(output: Path, plan: Path, action: str, fold: int, seed: int, resume: bool) -> None:
    """Rank zero alone claims a new directory or validates an unfinished cell."""
    if output.exists():
        if not resume or action != "train" or (output / "results/run.json").exists():
            raise FileExistsError("Output exists; only an unfinished GPU cell may be resumed")
        prior = json.loads((output / "provenance.json").read_text())
        if (prior["training_plan_sha256"], prior["fold"], prior["seed"]) != (
            sha256(plan), fold, seed
        ):
            raise ValueError("Resume provenance differs from the frozen cell")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["profile", "train"])
    p.add_argument("--plan", type=Path, default=Path("configs/research/s22_complementary_v5.json"))
    p.add_argument("--fold", type=int, choices=[0, 1], default=0)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true", help="Resume an unfinished matching GPU cell")
    args = p.parse_args()
    spec = json.loads(args.plan.read_text()) if args.action == "profile" else verify_plan(args.plan)
    output = args.output or Path("outputs/s22_complementary_v5") / (
        "cpu_profile" if args.action == "profile" else f"f{args.fold}_s{args.seed}"
    )
    overrides = list(spec["v5_overrides"])
    overrides += [f"data.split_fold={args.fold}", f"seed={args.seed}", f"output_dir={output}",
                  f"run_name=s22_v5_f{args.fold}_s{args.seed}"]
    if args.action == "profile":
        overrides += ["runtime=default", "tracking=none", "device=cpu", "runtime.multi_gpu=off",
                      "runtime.num_workers=0", "runtime.eval_num_workers=0", "runtime.compile=off",
                      "runtime.prewarm_cache=off"]
        torch.set_num_threads(4)
    else:
        overrides += ["runtime=kaggle_t4x2", "tracking=console_jsonl", "device=cuda"]
        if not torch.cuda.is_available():
            raise RuntimeError("GPU training requested but CUDA is unavailable")
    cfg = load_experiment_config("experiment/seednet_full256", overrides=overrides)
    rows = load_frozen_rows(Path(spec["partition_plan"]), Path(cfg.data.labels_path),
                            Path(cfg.data.groups_path), args.fold)
    dist = init_distributed(cfg.runtime, fallback=resolve_device(cfg.device))
    tracker: ExperimentTracker = NullTracker()
    try:
        error: str | None = None
        if dist.is_main:
            try:
                prepare_output(output, args.plan, args.action, args.fold, args.seed, args.resume)
            except (OSError, ValueError, KeyError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = dist.broadcast_object(error)
        if error is not None:
            raise RuntimeError(error)
        if dist.is_main:
            write_json(output / "provenance.json", {"training_plan_sha256": sha256(args.plan),
                "partition_plan_sha256": sha256(Path(spec["partition_plan"])), "action": args.action,
                "fold": args.fold, "seed": args.seed, "gpu_results": False if args.action == "profile" else "pending"})
            OmegaConf.save(OmegaConf.structured(cfg), output / "resolved_config.yaml")
        tracker = build_tracker(cfg) if dist.is_main else NullTracker()
        ctx = build_run_context(cfg, tracker, dist, split_override=rows)
        if args.action == "train":
            resolve_pipeline(str(cfg.pipeline))(ctx)
        else:
            # One real training-only batch, forward/backward. No checkpoint selection,
            # test loader, held-out prediction, or performance result is produced.
            ids = rows.train[:2]
            x = torch.from_numpy(np.array(ctx.store.require_patches()[ids], dtype=np.float32))
            morph = torch.from_numpy(np.asarray(ctx.morph[ids])) if ctx.morph is not None else None
            labels = torch.from_numpy(rows.labels[ids].astype(np.int64))
            ctx.model.train()
            start = time.monotonic()
            scores = ctx.model(x, labels=labels, morph=morph)
            assert isinstance(scores, dict)
            loss = torch.nn.functional.cross_entropy(scores["main"], labels)
            loss.backward()  # type: ignore[no-untyped-call]
            finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in ctx.model.parameters())
            if not finite:
                raise ValueError("Nonfinite profile gradient")
            write_json(output / "profile.json", {"device": "cpu", "batch": 2,
                "forward_backward_seconds": time.monotonic() - start, "finite_gradients": finite,
                "parameters": sum(p.numel() for p in ctx.model.parameters()),
                "training_rows": ids.tolist(), "held_out_rows_scored": 0,
                "partition_scheme": rows.report.scheme,
                "note": "Engineering smoke profile, not a learning curve or GPU speed estimate."})
    finally:
        tracker.close()
        shutdown(dist)


if __name__ == "__main__":
    main()
