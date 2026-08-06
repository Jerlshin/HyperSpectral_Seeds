#!/usr/bin/env python
"""Capture golden regression values from the pre-refactor model — §3.2.2.

Builds :class:`SpectralQuadNet` from the **baseline** ``hsi_training.py``
(``git show 886560f``, loaded side-effect-free by :mod:`_baseline`), runs one
fixed-seed forward pass and one fixed-seed training epoch, and writes the
artifacts that ``tests/regression/test_golden_forward_pass.py`` asserts against.

Artifacts written to ``tests/regression/golden/``
────────────────────────────────────────────────
``physical_wl_spa40.npy``       the min-max-normalised wavelength vector, so the
                                test never needs the gitignored ``dataset/``.
``forward_logits_seed42.npy``   ``(4, 90)`` eval-mode logits.
``init_state_sha256.json``      SHA-256 of every one of the 352 freshly
                                initialised state-dict tensors, plus a combined
                                digest. This is the sharpest available check on
                                §3.6's "identical weight initialization" claim —
                                it catches a construction-order change that a
                                4-sample forward pass might average away.
``stage1_epoch1_loss_seed42.json``  scalar loss and accuracy from one Stage-1
                                epoch over 32 synthetic samples, plus post-step
                                digests of the model and EMA weights.
``README.md``                   provenance: git SHA, versions, exact procedure.

Each procedure is defined **once** — :func:`forward_pass` and :func:`train_step` —
and applied to both the baseline and the refactored code, so the two runs cannot
diverge in setup. :func:`refactored_train_step` is imported by ``tests/conftest.py``
for the same reason.

Usage
─────
    python scripts/capture_golden.py            # capture + immediately verify
    python scripts/capture_golden.py --verify   # verify only, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _baseline import BASELINE_REF, load_baseline_module  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "tests" / "regression" / "golden"

SEED = 42
BATCH = 4
SPATIAL = 64
DEVICE = torch.device("cpu")  # CPU keeps the capture portable and deterministic

# §3.2.2's training step: "one train_one_epoch-equivalent step on 32 synthetic
# samples". Four batches of eight, so the accumulation boundary, the optimiser
# step and the EMA update all execute more than once.
TRAIN_SAMPLES = 32
TRAIN_BATCH = 8


# ══════════════════════════════════════════════════════════════════════
#  The shared procedures
# ══════════════════════════════════════════════════════════════════════


def forward_pass(
    build_model: Callable[[], torch.nn.Module],
    set_seed: Callable[[int], None],
    num_bands: int,
) -> tuple[np.ndarray[Any, Any], dict[str, torch.Tensor]]:
    """Seed → construct → eval → one forward pass. Identical on both sides.

    ``set_seed`` is called immediately before construction so the branches'
    ``_init_weights`` draws land in the same order the baseline's import-time
    seeding produced (REFACTOR_PLAN.md §3.6).
    """
    set_seed(SEED)
    model = build_model()
    model.to(DEVICE).eval()

    # A dedicated generator keeps the input independent of however much RNG the
    # model construction above consumed.
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(BATCH, num_bands, SPATIAL, SPATIAL, generator=gen).to(DEVICE)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, torch.Tensor), "eval-mode forward must return a plain tensor"
    return out.detach().cpu().numpy(), {k: v.detach().cpu() for k, v in model.state_dict().items()}


def synthetic_epoch(num_bands: int, num_classes: int) -> DataLoader[Any]:
    """A fixed 32-sample loader — same tensors on both sides, never shuffled."""
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(TRAIN_SAMPLES, num_bands, SPATIAL, SPATIAL, generator=gen)
    y = torch.randint(0, num_classes, (TRAIN_SAMPLES,), generator=gen)
    return DataLoader(TensorDataset(x, y), batch_size=TRAIN_BATCH, shuffle=False)


def train_step(
    build_model: Callable[[], torch.nn.Module],
    set_seed: Callable[[int], None],
    make_ema: Callable[[torch.nn.Module], Any],
    build_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
    run_epoch: Callable[..., tuple[float, float]],
    num_bands: int,
    num_classes: int,
    label_smoothing: float,
) -> dict[str, Any]:
    """One Stage-1 Phase-1 epoch: mixup + deep supervision + clip + step + EMA.

    Reproduces the configuration ``run_stage1`` uses for its first epoch —
    ``CrossEntropyLoss(label_smoothing=s1_label_smooth_hi)``, mixup on, no
    contrastive losses — and returns everything that could drift:

    * ``loss``/``acc`` — §3.2.2's scalar, compared for **exact** equality.
    * ``model_sha256``/``ema_sha256`` — digests of the weights *after* the step.
      These are what make the check bite: they cover the mixup draw, the
      auxiliary-loss weighting, gradient clipping, the AdamW parameter-group
      split and the EMA decay, none of which a single loss scalar pins down.

    ``scaler=None`` selects the non-AMP path, which is the only deterministic
    one on CPU. The seed is reset immediately before the epoch so the epoch's own
    randomness (mixup's ``np.random.beta`` and ``torch.randperm``, the branch
    dropout draws) starts from a known state regardless of how much RNG
    construction consumed — construction itself is already covered by
    ``init_state_sha256.json``.
    """
    set_seed(SEED)
    model = build_model()
    model.to(DEVICE)
    ema = make_ema(model)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loader = synthetic_epoch(num_bands, num_classes)

    set_seed(SEED)
    loss, acc = run_epoch(model, loader, optimizer, criterion, ema)

    return {
        "loss": float(loss),
        "acc": float(acc),
        "model_sha256": combined_digest(model.state_dict()),
        "ema_sha256": combined_digest(ema.state_dict()),
    }


def state_digests(sd: dict[str, torch.Tensor]) -> dict[str, str]:
    per_tensor = {
        name: hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()
        for name, t in sd.items()
    }
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_tensor.items())).encode()
    ).hexdigest()
    return {"__combined__": combined, **per_tensor}


def combined_digest(sd: dict[str, torch.Tensor]) -> str:
    return state_digests(sd)["__combined__"]


# ══════════════════════════════════════════════════════════════════════
#  Baseline side
# ══════════════════════════════════════════════════════════════════════


def _baseline_model_builder(mod: Any) -> Callable[[], torch.nn.Module]:
    cfg = mod.CONFIG

    def build() -> torch.nn.Module:
        # `_load_wavelengths_to_gpu` populates the module global that
        # SpectralQuadNet.__init__ reads. It consumes no RNG, so running it
        # after set_seed matches §3.6's mandated ordering exactly.
        mod._load_wavelengths_to_gpu(cfg["wavelength_path"], DEVICE)
        model: torch.nn.Module = mod.SpectralQuadNet(
            num_classes=cfg["num_classes"],
            num_bands=cfg["num_bands"],
            dropout=cfg["s1_dropout"],
            wl_embed_dim=cfg["wl_embed_dim"],
            cfg=cfg,
        )
        return model

    return build


def baseline_train_step(mod: Any) -> dict[str, Any]:
    cfg = mod.CONFIG
    return train_step(
        build_model=_baseline_model_builder(mod),
        set_seed=mod.set_seed,
        make_ema=lambda m: mod.ModelEMA(m, decay=cfg["ema_decay"]),
        build_optimizer=lambda m: mod.build_optimizer_s1(m, cfg["s1_max_lr"]),
        run_epoch=lambda m, ldr, opt, crit, ema: mod.train_one_epoch(
            m,
            ldr,
            opt,
            crit,
            None,
            ema,
            DEVICE,
            use_mixup=True,
            mixup_alpha=cfg["s1_mixup"],
            accum_steps=cfg["s1_accum"],
            current_ep=1,
            total_ep=cfg["s1_epochs"],
        ),
        num_bands=cfg["num_bands"],
        num_classes=cfg["num_classes"],
        label_smoothing=cfg["s1_label_smooth_hi"],
    )


def capture_baseline(
    ref: str,
) -> tuple[
    np.ndarray[Any, Any], dict[str, str], dict[str, Any], np.ndarray[Any, Any], dict[str, Any]
]:
    mod = load_baseline_module("hsi_training", ref)
    cfg = mod.CONFIG

    logits, sd = forward_pass(_baseline_model_builder(mod), mod.set_seed, cfg["num_bands"])
    loss = baseline_train_step(mod)

    wl = mod._PHYSICAL_WL.detach().cpu().numpy()
    meta = {
        "num_classes": cfg["num_classes"],
        "num_bands": cfg["num_bands"],
        "dropout": cfg["s1_dropout"],
        "wl_embed_dim": cfg["wl_embed_dim"],
        "wavelength_path": cfg["wavelength_path"],
    }
    return logits, state_digests(sd), loss, wl, meta


# ══════════════════════════════════════════════════════════════════════
#  Refactored side
# ══════════════════════════════════════════════════════════════════════


def refactored_train_step(cfg: Any, physical_wl: torch.Tensor) -> dict[str, Any]:
    """§3.2.2's training step on the post-refactor code.

    Imported by ``tests/conftest.py`` so the regression test and this capture
    script cannot drift apart in setup — the whole point of defining each
    procedure once.
    """
    from spectralquadnet.engine.train_epoch import train_one_epoch
    from spectralquadnet.models.ema import ModelEMA
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
    from spectralquadnet.optim.param_groups import build_optimizer_s1
    from spectralquadnet.utils.seed import set_seed

    return train_step(
        build_model=lambda: SpectralQuadNet.from_config(cfg, physical_wl),
        set_seed=set_seed,
        make_ema=lambda m: ModelEMA(m, decay=cfg.ema_decay),
        build_optimizer=lambda m: build_optimizer_s1(cfg, m, cfg.stage1.max_lr),
        run_epoch=lambda m, ldr, opt, crit, ema: train_one_epoch(
            cfg,
            m,
            ldr,
            opt,
            crit,
            None,
            ema,
            DEVICE,
            use_mixup=True,
            mixup_alpha=cfg.stage1.mixup,
            accum_steps=cfg.stage1.accum,
            current_ep=1,
            total_ep=cfg.stage1.epochs,
        ),
        num_bands=cfg.data.num_bands,
        num_classes=cfg.data.num_classes,
        label_smoothing=cfg.stage1.label_smooth_hi,
    )


def capture_refactored(
    wl: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
    from spectralquadnet.utils.seed import set_seed

    cfg = load_experiment_config()
    physical_wl = torch.from_numpy(wl).to(DEVICE)

    logits, sd = forward_pass(
        lambda: SpectralQuadNet.from_config(cfg, physical_wl), set_seed, cfg.data.num_bands
    )
    loss = refactored_train_step(cfg, physical_wl)

    meta = {
        "num_classes": cfg.data.num_classes,
        "num_bands": cfg.data.num_bands,
        "dropout": cfg.stage1.dropout,
        "wl_embed_dim": cfg.model.wl_embed_dim,
        "wavelength_path": cfg.data.wavelength_path,
    }
    return logits, state_digests(sd), loss, meta


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════


def write_readme(
    ref: str,
    meta: dict[str, Any],
    logits: np.ndarray[Any, Any],
    digests: dict[str, str],
    loss: dict[str, Any],
) -> None:
    (GOLDEN / "README.md").write_text(
        f"""# Golden regression values

Captured from the **pre-refactor** code by `scripts/capture_golden.py`
(REFACTOR_PLAN.md §3.2.2). Do not hand-edit — regenerate instead.

| | |
|---|---|
| Source git SHA | `{ref}` |
| Source file | `HSI_modality_training/hsi_training.py` |
| Captured (UTC) | {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} |
| torch | {torch.__version__} |
| numpy | {np.__version__} |
| Device | `cpu` |

## Forward-pass procedure

Defined once in `capture_golden.py::forward_pass` and run identically against the
baseline and the refactored model:

1. `set_seed(42)` — immediately before construction, so the per-branch
   `_init_weights` draws consume the global torch RNG in the same order the
   baseline's import-time seeding produced (§3.6).
2. Load the physical wavelengths (consumes no RNG) and build `SpectralQuadNet`
   with `num_classes={meta["num_classes"]}`, `num_bands={meta["num_bands"]}`,
   `dropout={meta["dropout"]}`, `wl_embed_dim={meta["wl_embed_dim"]}`.
3. `.to("cpu").eval()` — eval mode makes the forward deterministic (no branch
   dropout, no `torch.rand` draws) and returns a plain tensor rather than the
   training-mode dict of auxiliary logits.
4. Input: `torch.randn({BATCH}, {meta["num_bands"]}, {SPATIAL}, {SPATIAL})` from a
   dedicated `torch.Generator().manual_seed(42)`, so the input is independent of
   how much RNG construction consumed.
5. One `torch.no_grad()` forward pass.

## Training-step procedure

Defined once in `capture_golden.py::train_step` (§3.2.2's second artifact, the
Phase 3 gate). Same construction as above, then:

1. `ModelEMA(model, decay=0.999)` and `build_optimizer_s1(model, s1_max_lr)`.
2. `CrossEntropyLoss(label_smoothing=0.10)` and mixup on — Stage 1 Phase 1's
   exact configuration, so the mixup draw, the four auxiliary heads, gradient
   clipping, the AdamW weight-decay split and the EMA update all execute.
3. `set_seed(42)` again, immediately before the epoch, so the epoch's own
   randomness starts from a known state.
4. One `train_one_epoch` over {TRAIN_SAMPLES} synthetic samples in batches of
   {TRAIN_BATCH} (`scaler=None` — the non-AMP path is the deterministic one on CPU).

## Files

| File | Contents |
|---|---|
| `physical_wl_spa40.npy` | `float32 ({meta["num_bands"]},)` — min-max-normalised wavelengths from `{meta["wavelength_path"]}`. Committed so the test never needs the gitignored `dataset/`. |
| `forward_logits_seed42.npy` | `float32 {logits.shape}` — eval-mode logits. |
| `init_state_sha256.json` | SHA-256 per initialised state-dict tensor ({len(digests) - 1} entries) plus `__combined__`. Catches construction-order drift that a 4-sample forward could average away. |
| `stage1_epoch1_loss_seed42.json` | Scalar loss/accuracy plus combined SHA-256 of the model and EMA weights *after* the step. |

Combined init digest: `{digests["__combined__"]}`
Stage-1 epoch-1 loss: `{loss["loss"]!r}`

## Regenerating

    python scripts/capture_golden.py           # capture + verify
    python scripts/capture_golden.py --verify  # verify only

A regeneration is only legitimate when the *baseline* reference changes. If these
files need updating to make a test pass, the refactor has changed behaviour —
that is the failure the gate exists to catch.
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ref", default=BASELINE_REF)
    ap.add_argument("--verify", action="store_true", help="compare only; write nothing")
    args = ap.parse_args()

    GOLDEN.mkdir(parents=True, exist_ok=True)

    print("═" * 78)
    print(f"  Golden capture (§3.2.2)   baseline {args.baseline_ref[:7]}")
    print("═" * 78)

    print("\n[1/3] Running the PRE-refactor code (forward pass + 1 training epoch)...")
    base_logits, base_digests, base_loss, wl, base_meta = capture_baseline(args.baseline_ref)
    print(f"      logits {base_logits.shape} {base_logits.dtype}")
    print(
        f"      init digest {base_digests['__combined__'][:16]}…  ({len(base_digests) - 1} tensors)"
    )
    print(f"      epoch loss {base_loss['loss']!r}  acc {base_loss['acc']!r}")

    print("\n[2/3] Running the POST-refactor code...")
    new_logits, new_digests, new_loss, new_meta = capture_refactored(wl)
    print(f"      logits {new_logits.shape} {new_logits.dtype}")
    print(
        f"      init digest {new_digests['__combined__'][:16]}…  ({len(new_digests) - 1} tensors)"
    )
    print(f"      epoch loss {new_loss['loss']!r}  acc {new_loss['acc']!r}")

    print("\n[3/3] Comparing...")
    ok = True

    if base_meta != new_meta:
        ok = False
        print("  ✗ construction parameters differ (config round-trip regression):")
        for k in sorted(set(base_meta) | set(new_meta)):
            if base_meta.get(k) != new_meta.get(k):
                print(f"      {k}: baseline={base_meta.get(k)!r}  refactored={new_meta.get(k)!r}")
    else:
        print(
            "  ✓ construction parameters identical "
            f"(num_classes={base_meta['num_classes']}, num_bands={base_meta['num_bands']}, "
            f"dropout={base_meta['dropout']})"
        )

    mismatched = [k for k in base_digests if base_digests[k] != new_digests.get(k)]
    missing = set(base_digests) - set(new_digests)
    extra = set(new_digests) - set(base_digests)
    if mismatched or missing or extra:
        ok = False
        print(
            f"  ✗ init weights differ: {len(mismatched)} mismatched, "
            f"{len(missing)} missing, {len(extra)} unexpected"
        )
        for k in sorted(mismatched)[:10]:
            print(f"      {k}")
    else:
        print(f"  ✓ init weights bit-identical ({len(base_digests) - 1} tensors)")

    max_abs = float(np.max(np.abs(base_logits - new_logits)))
    if np.allclose(base_logits, new_logits, atol=1e-6):
        print(f"  ✓ logits match within atol=1e-6 (max |Δ| = {max_abs:.3e})")
    else:
        ok = False
        print(f"  ✗ logits differ: max |Δ| = {max_abs:.3e}")

    for key, label in [
        ("loss", "epoch loss"),
        ("acc", "epoch accuracy"),
        ("model_sha256", "post-step model weights"),
        ("ema_sha256", "post-step EMA weights"),
    ]:
        if base_loss[key] == new_loss[key]:
            print(f"  ✓ {label} exact match ({str(base_loss[key])[:24]})")
        else:
            ok = False
            print(f"  ✗ {label} differs: baseline={base_loss[key]!r} refactored={new_loss[key]!r}")

    if not ok:
        print("\n✗ Capture aborted — the refactored code does not reproduce the baseline.")
        return 1

    if args.verify:
        print("\n✓ Verify-only run passed; no files written.")
        return 0

    np.save(GOLDEN / "physical_wl_spa40.npy", wl.astype(np.float32))
    np.save(GOLDEN / "forward_logits_seed42.npy", base_logits.astype(np.float32))
    (GOLDEN / "init_state_sha256.json").write_text(json.dumps(base_digests, indent=2) + "\n")
    (GOLDEN / "stage1_epoch1_loss_seed42.json").write_text(json.dumps(base_loss, indent=2) + "\n")
    write_readme(args.baseline_ref, base_meta, base_logits, base_digests, base_loss)

    print(f"\n✓ Wrote 5 artifacts to {GOLDEN.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
