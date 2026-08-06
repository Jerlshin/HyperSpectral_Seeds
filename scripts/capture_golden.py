#!/usr/bin/env python
"""Capture golden forward-pass values from the pre-refactor model — §3.2.2.

Builds :class:`SpectralQuadNet` from the **baseline** ``hsi_training.py``
(``git show 886560f``, loaded side-effect-free by :mod:`_baseline`), runs one
fixed-seed forward pass, and writes the artifacts that
``tests/regression/test_golden_forward_pass.py`` asserts against.

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
``README.md``                   provenance: git SHA, versions, exact procedure.

The procedure is defined **once** in :func:`forward_pass` and applied to both the
baseline and the refactored model, so the two runs cannot diverge in setup.

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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _baseline import BASELINE_REF, load_baseline_module  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "tests" / "regression" / "golden"

SEED = 42
BATCH = 4
SPATIAL = 64
DEVICE = torch.device("cpu")  # CPU keeps the capture portable and deterministic


# ══════════════════════════════════════════════════════════════════════
#  The shared procedure
# ══════════════════════════════════════════════════════════════════════


def forward_pass(
    build_model: Callable[[], torch.nn.Module],
    set_seed: Callable[[int], None],
    num_bands: int,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
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


def state_digests(sd: dict[str, torch.Tensor]) -> dict[str, str]:
    per_tensor = {
        name: hashlib.sha256(np.ascontiguousarray(t.numpy()).tobytes()).hexdigest()
        for name, t in sd.items()
    }
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_tensor.items())).encode()
    ).hexdigest()
    return {"__combined__": combined, **per_tensor}


# ══════════════════════════════════════════════════════════════════════
#  Baseline side
# ══════════════════════════════════════════════════════════════════════


def capture_baseline(ref: str) -> tuple[np.ndarray, dict[str, str], np.ndarray, dict]:
    mod = load_baseline_module("hsi_training", ref)
    cfg = mod.CONFIG
    num_bands = cfg["num_bands"]

    def build() -> torch.nn.Module:
        # `_load_wavelengths_to_gpu` populates the module global that
        # SpectralQuadNet.__init__ reads. It consumes no RNG, so running it
        # after set_seed matches §3.6's mandated ordering exactly.
        mod._load_wavelengths_to_gpu(cfg["wavelength_path"], DEVICE)
        return mod.SpectralQuadNet(
            num_classes=cfg["num_classes"],
            num_bands=cfg["num_bands"],
            dropout=cfg["s1_dropout"],
            wl_embed_dim=cfg["wl_embed_dim"],
            cfg=cfg,
        )

    logits, sd = forward_pass(build, mod.set_seed, num_bands)
    wl = mod._PHYSICAL_WL.detach().cpu().numpy()
    meta = {
        "num_classes": cfg["num_classes"],
        "num_bands": cfg["num_bands"],
        "dropout": cfg["s1_dropout"],
        "wl_embed_dim": cfg["wl_embed_dim"],
        "wavelength_path": cfg["wavelength_path"],
    }
    return logits, state_digests(sd), wl, meta


# ══════════════════════════════════════════════════════════════════════
#  Refactored side
# ══════════════════════════════════════════════════════════════════════


def capture_refactored(wl: np.ndarray) -> tuple[np.ndarray, dict[str, str], dict]:
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
    from spectralquadnet.utils.seed import set_seed

    cfg = load_experiment_config()
    physical_wl = torch.from_numpy(wl).to(DEVICE)

    def build() -> torch.nn.Module:
        return SpectralQuadNet.from_config(cfg, physical_wl)

    logits, sd = forward_pass(build, set_seed, cfg.data.num_bands)
    meta = {
        "num_classes": cfg.data.num_classes,
        "num_bands": cfg.data.num_bands,
        "dropout": cfg.stage1.dropout,
        "wl_embed_dim": cfg.model.wl_embed_dim,
        "wavelength_path": cfg.data.wavelength_path,
    }
    return logits, state_digests(sd), meta


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════


def write_readme(ref: str, meta: dict, logits: np.ndarray, digests: dict[str, str]) -> None:
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

## Procedure

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

## Files

| File | Contents |
|---|---|
| `physical_wl_spa40.npy` | `float32 ({len(np.atleast_1d(meta["num_bands"])) * 0 + meta["num_bands"]},)` — min-max-normalised wavelengths from `{meta["wavelength_path"]}`. Committed so the test never needs the gitignored `dataset/`. |
| `forward_logits_seed42.npy` | `float32 {logits.shape}` — eval-mode logits. |
| `init_state_sha256.json` | SHA-256 per initialised state-dict tensor ({len(digests) - 1} entries) plus `__combined__`. Catches construction-order drift that a 4-sample forward could average away. |

Combined init digest: `{digests["__combined__"]}`

## Not yet captured

`stage1_epoch1_loss_seed42.json` — §3.2.2's second artifact. It needs
`train_one_epoch`, which Phase 3 relocates; the Phase 3 gate captures it.

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

    print("\n[1/3] Building the PRE-refactor model...")
    base_logits, base_digests, wl, base_meta = capture_baseline(args.baseline_ref)
    print(f"      logits {base_logits.shape} {base_logits.dtype}")
    print(
        f"      init digest {base_digests['__combined__'][:16]}…  ({len(base_digests) - 1} tensors)"
    )

    print("\n[2/3] Building the POST-refactor model...")
    new_logits, new_digests, new_meta = capture_refactored(wl)
    print(f"      logits {new_logits.shape} {new_logits.dtype}")
    print(
        f"      init digest {new_digests['__combined__'][:16]}…  ({len(new_digests) - 1} tensors)"
    )

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

    if not ok:
        print("\n✗ Capture aborted — the refactored model does not reproduce the baseline.")
        return 1

    if args.verify:
        print("\n✓ Verify-only run passed; no files written.")
        return 0

    np.save(GOLDEN / "physical_wl_spa40.npy", wl.astype(np.float32))
    np.save(GOLDEN / "forward_logits_seed42.npy", base_logits.astype(np.float32))
    (GOLDEN / "init_state_sha256.json").write_text(json.dumps(base_digests, indent=2) + "\n")
    write_readme(args.baseline_ref, base_meta, base_logits, base_digests)

    print(f"\n✓ Wrote 4 artifacts to {GOLDEN.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
