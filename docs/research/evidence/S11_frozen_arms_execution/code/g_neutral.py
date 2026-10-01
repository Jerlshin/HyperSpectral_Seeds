#!/usr/bin/env python3
"""S11 gate G-neutral: the S10 P0 changes must not move a single training number.

Runs the same miniature training job through two source trees — the code before
P0 (a ``git worktree`` of the parent commit) and the code after it — and compares
what each run *learned*, not what it logged:

* every optimiser step's training loss, recorded at the one place the loop reads
  it back (``train_epoch._resolve_step_scalars``);
* the per-epoch ``(loss, acc)`` pair the loop returns;
* the SHA-256 of every tensor in the selected checkpoint (live and EMA);
* the held-out predictions of the final evaluation (no-TTA and TTA).

Three regimes, all on CPU with ``runtime.num_workers=0`` (single RNG stream):

``shipped``   the S09 regime in miniature — mixup on epoch 1, the margin ramp on
              epochs 2–3, clip 5.0, label smoothing, the legacy aux schedule;
``clip_binds`` the same with ``grad_clip=0.05``, so the per-group clip binds on
              every step and the clip *partition* is exercised (P0.5's risk);
``x1_like``   X1's overrides in miniature (mixup 1 epoch, m = 0, clip 50).

The new tree runs with its telemetry fully on (clean-fit evaluation every epoch,
per-module gradient norms) — the point is that measuring does not perturb.

Usage (from the repository root)::

    python docs/research/evidence/S11_frozen_arms_execution/code/g_neutral.py \\
        --old <worktree of the parent commit> --new . \\
        --out docs/research/evidence/S11_frozen_arms_execution/g_neutral.json

Exit status 0 only if every compared quantity is identical (losses to 1e-6,
digests and predictions exactly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# ── The miniature dataset — the smoke tier's structure (two bundles/class) ──
N_CLASSES, BUNDLES, PER_BUNDLE, N_BANDS, SIDE = 8, 2, 12, 8, 16

REGIMES: dict[str, list[str]] = {
    "shipped": [],
    "clip_binds": ["grad_clip=0.05"],
    "x1_like": [
        "single.mixup_epochs=1",
        "single.arcface_m=0.0",
        "single.margin_warmup_start=2",
        "single.margin_warmup_end=2",
        "grad_clip=50.0",
    ],
}

#: What both trees run. Mirrors `tests/smoke/_helpers.py::tiny_overrides`, with
#: 3 epochs so the run crosses the mixup → margin boundary and the margin ramp.
BASE: dict[str, Any] = {
    "data.gain_path": "",
    "data.num_bands": N_BANDS,
    "data.num_classes": N_CLASSES,
    "data.cutmix_bands": 2,
    "data.max_cutout_bands": 1,
    "single.epochs": 3,
    "single.batch": 8,
    "single.warmup_ep": 1,
    "single.mixup_epochs": 1,
    "single.margin_warmup_start": 2,
    "single.margin_warmup_end": 3,
    "single.patience": 5,
    "model.stem_channels": 16,
    "model.stem_folded_depth": 1,
    "model.spatial_width_mult": 0.25,
    "model.spectral_hidden": 32,
    "model.index_bank_size": 8,
    "model.continuum_depths": 4,
    "model.aux_head_hidden": 16,
    "evaluation.bootstrap_samples": 0,
    "tta_spatial": 2,
    "tta_spectral": 1,
    "device": "cpu",
    "runtime.num_workers": 0,
    "runtime.compile": "off",
    "runtime.diagnostics_interval": 1,
    "runtime.checkpoint_every": 0,
    "tracking.backend": "jsonl",
}

#: Telemetry switched fully on in the NEW tree only (keys the old tree lacks).
NEW_ONLY: list[str] = ["tracking.log_grad_norms=true", "single.clean_fit_kernels=64"]

#: Series the new tree must have written, or the comparison proved nothing.
EXPECTED_SERIES: tuple[str, ...] = (
    "sched/aux_weight_applied",
    "sched/aux_weight_configured",
    "train/loss_main",
    "train/acc_dominant",
    "train/acc_plain",
    "grad_norm/stem",
    "grad_norm/fuse",
    "grad_norm/nonfinite_steps",
    "fit/clean_train_acc_live",
    "fit/clean_train_ce_ema",
)

#: Sensitivity controls: changes the harness MUST detect. If either compared
#: equal, the gate would be blind. The first is also the evidence behind S11
#: D22 — the model-declared clip partition is not neutral when a clip binds.
SENSITIVITY: dict[str, tuple[str, list[str]]] = {
    "clip_partition_model_when_clip_binds": ("clip_binds", ["clip_partition=model"]),
    "aux_weight_schedule_fixed": ("shipped", ["single.aux_weight_schedule=fixed"]),
}


def build_dataset(root: Path) -> dict[str, str]:
    """The smoke tier's synthetic cube (`tests/smoke/conftest.py`), seed 0."""
    rng = np.random.default_rng(0)
    n = N_CLASSES * BUNDLES * PER_BUNDLE
    patches = np.zeros((n, N_BANDS, SIDE, SIDE), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    groups = np.zeros(n, dtype=np.int64)
    masks = np.ones((n, SIDE, SIDE), dtype=np.float16)
    morph = rng.normal(size=(n, 8)).astype(np.float32)
    row = 0
    for c in range(N_CLASSES):
        sig = rng.normal(size=(N_BANDS, 1, 1)) * 0.5
        for b in range(BUNDLES):
            off = rng.normal(size=(N_BANDS, 1, 1)) * 0.1
            for _ in range(PER_BUNDLE):
                p = sig + off + rng.normal(size=(N_BANDS, SIDE, SIDE)) * 0.2
                p[:, :2, :] = 0.0
                p[:, -2:, :] = 0.0
                masks[row, :2, :] = 0.0
                masks[row, -2:, :] = 0.0
                patches[row], labels[row], groups[row] = p, c, c * BUNDLES + b
                row += 1
    root.mkdir(parents=True, exist_ok=True)
    for name, arr in (
        ("patches", patches),
        ("labels", labels),
        ("groups", groups),
        ("masks", masks),
        ("morphology", morph),
    ):
        np.save(root / f"{name}.npy", arr)
    wl = np.sort(rng.uniform(400, 1000, size=N_BANDS))
    (root / "wavelengths.csv").write_text(
        "Band,Wavelength (nm)\n" + "\n".join(f"{i},{v:.2f}" for i, v in enumerate(wl)) + "\n"
    )
    return {
        "data.patches_data": str(root / "patches.npy"),
        "data.labels_path": str(root / "labels.npy"),
        "data.groups_path": str(root / "groups.npy"),
        "data.masks_path": str(root / "masks.npy"),
        "data.morphology_path": str(root / "morphology.npy"),
        "data.wavelength_path": str(root / "wavelengths.csv"),
    }


#: Executed in a fresh interpreter per (tree, regime): Hydra, the DataStore and
#: torch's RNG are process-global, so two runs in one process are not two runs.
CHILD = r"""
import json, sys, hashlib
tree, out, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, tree + "/src")
sys.path.insert(0, tree)
import spectralquadnet.engine.train_epoch as te
steps = []
_orig = te._resolve_step_scalars
def _rec(loss, acc):
    v = _orig(loss, acc)
    steps.append(v[0])
    return v
te._resolve_step_scalars = _rec
epochs = []
_orig_epoch = te.train_one_epoch
import spectralquadnet.engine.stages.single_stage as ss
def _ep(*a, **k):
    r = _orig_epoch(*a, **k)
    epochs.append(list(r))
    return r
ss.train_one_epoch = _ep
import runpy
sys.argv = [tree + "/train.py", *argv]
try:
    runpy.run_path(tree + "/train.py", run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None):
        raise
json.dump({"steps": steps, "epochs": epochs}, open(out, "w"))
"""


def _digest(path: Path) -> dict[str, str]:
    import torch

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, str] = {}
    for slot in ("model", "ema"):
        for k, v in sorted(bundle[slot].items()):
            if hasattr(v, "detach"):
                out[f"{slot}.{k}"] = hashlib.sha256(
                    v.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
    out["__epoch__"] = str(bundle.get("epoch"))
    out["__best_source__"] = str(bundle.get("best_source"))
    return out


def _series(out_dir: Path) -> set[str]:
    """Every scalar key the run wrote to ``metrics.jsonl``."""
    keys: set[str] = set()
    path = out_dir / "metrics.jsonl"
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "scalars":
            keys.update(event.get("metrics", {}))
    return keys


def run(
    tree: Path,
    regime: str,
    data: dict[str, str],
    work: Path,
    new: bool,
    extra: list[str] | None = None,
    tag: str = "",
) -> dict[str, Any]:
    out_dir = work / f"{tree.name}_{regime}{tag}"
    overrides = [f"{k}={v}" for k, v in {**BASE, **data}.items()]
    overrides += REGIMES[regime]
    overrides += [f"output_dir={out_dir}", f"hydra.run.dir={out_dir / 'hydra'}"]
    if new:
        overrides += NEW_ONLY
    overrides += extra or []
    rec = work / f"{tree.name}_{regime}{tag}.json"
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(tree.resolve()), str(rec), *overrides],
        cwd=str(tree.resolve()),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not rec.exists():
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise RuntimeError(f"{tree.name}/{regime} failed (rc={proc.returncode})")
    payload = json.loads(rec.read_text())
    payload["digest"] = _digest(out_dir / "best_stage1.pth")
    res = out_dir / "results"
    payload["preds"] = {
        v: np.load(res / f"preds_val_test_{v}.npy").tolist() for v in ("no_tta", "tta")
    }
    payload["out_dir"] = str(out_dir)
    payload["series"] = sorted(_series(out_dir))
    payload["clean_fit_json"] = (out_dir / "clean_fit.json").exists()
    return payload


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    sa, sb = np.asarray(a["steps"]), np.asarray(b["steps"])
    same_len = sa.shape == sb.shape
    max_step = float(np.max(np.abs(sa - sb))) if same_len and sa.size else float("inf")
    ea, eb = np.asarray(a["epochs"]), np.asarray(b["epochs"])
    max_epoch = float(np.max(np.abs(ea - eb))) if ea.shape == eb.shape else float("inf")
    digest_diff = sorted(k for k in a["digest"] if a["digest"][k] != b["digest"].get(k))
    preds_equal = all(a["preds"][v] == b["preds"][v] for v in a["preds"])
    return {
        "n_steps": [int(sa.size), int(sb.size)],
        "max_abs_step_loss_diff": max_step,
        "max_abs_epoch_diff": max_epoch,
        "tensors_compared": len(a["digest"]),
        "tensors_differing": digest_diff,
        "heldout_preds_identical": preds_equal,
        "pass": same_len
        and max_step <= 1e-6
        and max_epoch <= 1e-6
        and not digest_diff
        and preds_equal,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", required=True, type=Path, help="source tree before P0")
    ap.add_argument("--new", required=True, type=Path, help="source tree after P0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument(
        "--plain-new",
        action="store_true",
        help="run the new tree without its extra telemetry (harness self-check old vs old)",
    )
    args = ap.parse_args()

    report: dict[str, Any] = {"regimes": {}}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        data = build_dataset(work / "data")
        olds: dict[str, dict[str, Any]] = {}
        for regime in args.regimes:
            old = run(args.old, regime, data, work, new=False)
            new = run(args.new, regime, data, work, new=not args.plain_new)
            olds[regime] = old
            result = compare(old, new)
            if not args.plain_new:
                missing = [k for k in EXPECTED_SERIES if k not in new["series"]]
                result["telemetry_missing"] = missing
                result["clean_fit_json_written"] = new["clean_fit_json"]
                result["pass"] = result["pass"] and not missing and new["clean_fit_json"]
            report["regimes"][regime] = result
            print(regime, json.dumps(result))
        if not args.plain_new:
            report["sensitivity"] = {}
            for name, (regime, extra) in SENSITIVITY.items():
                if regime not in olds:
                    continue
                changed = run(args.new, regime, data, work, new=True, extra=extra, tag=f"_{name}")
                diff = compare(olds[regime], changed)
                detected = not diff["pass"]
                report["sensitivity"][name] = {
                    "overrides": extra,
                    "regime": regime,
                    "detected": detected,
                    "max_abs_step_loss_diff": diff["max_abs_step_loss_diff"],
                    "tensors_differing": len(diff["tensors_differing"]),
                }
                print("sensitivity", name, json.dumps(report["sensitivity"][name]))
    report["pass"] = all(r["pass"] for r in report["regimes"].values()) and all(
        r["detected"] for r in report.get("sensitivity", {}).values()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("G-neutral:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
