#!/usr/bin/env python
"""Capture golden regression values from a pinned reference implementation.

Builds :class:`SpectralQuadNet` from the reference ``hsi_training.py``
(``git show <ref>``, loaded side-effect-free by :mod:`_baseline`), runs one
fixed-seed forward pass and one fixed-seed training epoch, and writes the
artifacts that ``tests/regression/test_golden_forward_pass.py`` asserts against.

Three schema versions live side by side under ``tests/regression/golden/``.

``golden/``      **schema v1** — captured from the pinned reference
                 implementation. Frozen. These describe the two-head model
                 (``linear_head`` for Stage 1, ``arcface_head`` for Stages 2-3)
                 and are what Phases 1-5 and Tier 1 were held to. Still
                 re-verified against the pinned reference on every run.
``golden/v2/``   **schema v2** — the Tier-2 architecture, after HD-1 (T2-10)
                 collapsed the two heads into one and HD-3 (T2-8) added
                 ``arcface_head.confusion``. **Frozen at Tier-3 completion.**
                 No longer re-capturable: the code that produced it is gone.
                 Its ``init_state_sha256.json`` is still read, as the left-hand
                 side of the v2 → v3 structural delta below.
``golden/v3/``   **schema v3** — the Tier-3 architecture (T3-1 … T3-7). This is
                 the live drift gate.

No two of the three can be compared value-by-value, and in each case the reason
is mechanical rather than a regression:

* **v1 → v2.** Deleting ``linear_head`` removes its ``nn.Linear``'s two
  construction draws from the global RNG stream, and every ``kaiming_normal_`` /
  ``trunc_normal_`` in ``_init_weights`` runs *after* that point, so 146 of the
  350 initialised tensors are drawn from a shifted stream.
* **v2 → v3.** Tier 3 rebuilt three of the four branches and the fusion. 115
  tensors left, 70 arrived, and of the 236 whose *name* survived a large share
  changed shape — Branch D runs at ``d_model = 192`` where it ran at 256, and
  the fusion's per-modality norm is a ``BatchNorm1d`` where it was a
  ``LayerNorm``.

What **is** checkable, and what :func:`check_schema_delta` enforces, is that
every key that appeared or disappeared falls under a **declared prefix** in
:data:`V3_DELTA`, each carrying its ``IMPROVEMENT_PLAN.md`` item id. A key set
this large cannot be enumerated tensor by tensor without the declaration
becoming unreadable, and an unreadable declaration is not a check; a prefix
rule still fails on the one thing this gate exists to catch, which is a
structural change nobody wrote down.

Artifacts written per schema directory
─────────────────────────────────────
``physical_wl_spa40.npy``       the min-max-normalised wavelength vector, so the
                                test never needs the gitignored ``dataset/``.
``forward_logits_seed42.npy``   ``(4, 90)`` eval-mode logits.
``init_state_sha256.json``      SHA-256 of every one of the freshly initialised
                                state-dict tensors, plus a combined digest.
                                This is the sharpest available check on
                                identical weight initialization — it catches a
                                construction-order change that a 4-sample
                                forward pass might average away.
``stage1_epoch1_loss_seed42.json``  scalar loss and accuracy from one Stage-1
                                epoch over 32 synthetic samples, plus post-step
                                digests of the model and EMA weights.
``README.md``                   provenance: git SHA, versions, exact procedure.

Each procedure is defined **once** — :func:`forward_pass` and :func:`train_step` —
and applied to both the reference and the current code, so the two runs cannot
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
GOLDEN_V2 = GOLDEN / "v2"
GOLDEN_V3 = GOLDEN / "v3"

#: State-dict keys schema v2 drops relative to v1, and the one it adds. Frozen
#: history now — v2 is no longer re-captured — but kept because
#: ``tests/regression/test_golden_forward_pass.py`` still asserts it against the
#: two committed digest files, which is what keeps "HD-1 removed a head" from
#: quietly covering an unrelated change.
V2_REMOVED = ("linear_head.2.weight", "linear_head.2.bias")
V2_ADDED = ("arcface_head.confusion",)

#: Every state-dict key prefix that appears or disappears between v2 and v3,
#: with the plan item that moved it. ``check_schema_delta`` fails on any key
#: outside these prefixes.
V3_DELTA: dict[str, str] = {
    # ── removed ───────────────────────────────────────────────────────
    "branch_b.stat_attn": "T3-1 / BR-1 — the nine moments are gone, and with them the statistic-attention gate over them.",
    "branch_b.input_proj": "T3-1 / BR-1 — Branch B no longer has a band axis to project.",
    "branch_b.tower_s": "T3-1 / BR-1 — the three 1-D towers read a rank-2 tensor (§2.2.5); replaced by an MLP over a full-rank descriptor.",
    "branch_b.tower_m": "T3-1 / BR-1 — see `branch_b.tower_s`.",
    "branch_b.tower_l": "T3-1 / BR-1 — see `branch_b.tower_s`.",
    "branch_b.fusion": "T3-1 / BR-1 — nothing left to fuse; 686 k -> 95 k.",
    "branch_b.pool_attn": "T3-1 / BR-1 — no band axis to attention-pool over.",
    "branch_b.proj": "T3-1 / BR-1 — replaced by `branch_b.mlp`.",
    "branch_b.wl_pe_module": "T3-1 / BR-1 — the wavelength PE indexes bands, and Branch B no longer processes a per-band sequence.",
    "branch_c.band_reduce": "T3-2 / BR-3 — the two 1x1 convolutions that collapsed the spectral axis before any spatial kernel ran (C-3).",
    "cross_interaction.latents": "T3-4 / FU-1(b) — the four Perceiver latents, which 0-E measured as collapsed onto one function (M-1).",
    "cross_interaction.blocks": "T3-4 / FU-1(b) — the cross/self-attention stack. With five modality tokens, latent attention compresses nothing.",
    "cross_interaction.output_proj": "T3-4 / FU-5 — the same pre-LN residual MLP as `embed_net`, applied to the same vector (N-10).",
    # ── added ─────────────────────────────────────────────────────────
    "branch_a.derivatives": "T3-5 / FE-1(b) — the Savitzky-Golay operators on the irregular lambda grid (buffers, no parameters).",
    "branch_a.stem": "T3-5 / FE-1(a) — the Conv1d stem became a LambdaConv1d whose kernel is generated from wavelength offsets.",
    "branch_b.index_bank": "T3-1 / BR-1(i) — the two soft band selectors of the learned NDI bank.",
    "branch_b.in_norm": "T3-1 / BR-1 — LayerNorm over the [indices || depths || morphometrics] descriptor.",
    "branch_b.mlp": "T3-1 / BR-1 — the branch's whole parameter budget, on a full-rank input.",
    "branch_c.stem": "T3-2 / BR-3 — the factorised 3-D spectral-spatial stem. The only module in the network that sees the full cube.",
    "branch_d.lambda_bias": "T3-3 / BR-4(ii) — the relative-lambda attention bias b_psi.",
    "cross_interaction.branch_norms": "T3-4 / FU-2 — BatchNorm1d replaces LayerNorm, so the normaliser is a dataset statistic (M-2a); five of them, not four (FU-4).",
    "cross_interaction.bilinear": "T3-4 / FU-1(b) — the low-rank projections U_m of the second-order term (M-3).",
    "cross_interaction.bilinear_out": "T3-4 / FU-1(b) — V, projecting the bilinear pool back to d.",
    "cross_interaction.output": "T3-4 / FU-1(b) — W_o over [first order || second order].",
    "morphology_embed.net": "T3-4 / FU-4 — the fifth modality token, from the eight morphometrics M-13 computed and discarded.",
}

SEED = 42
BATCH = 4
SPATIAL = 64
DEVICE = torch.device("cpu")  # CPU keeps the capture portable and deterministic

# One train_one_epoch-equivalent step on 32 synthetic samples, four batches
# of eight, so the accumulation boundary, the optimiser step and the EMA
# update all execute more than once.
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
    ``_init_weights`` draws consume the global RNG in a known, reproducible
    order.
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

    * ``loss``/``acc`` — the epoch's scalars, compared for **exact** equality.
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
        # after set_seed keeps the weight-init RNG order deterministic.
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
    """The training step on the current codebase.

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
            # HD-1: Stage 1 runs the shared head at zero margin, which is what
            # makes mixup admissible — `train_one_epoch` raises otherwise.
            arc_m=cfg.stage1.arcface_m,
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
    readme = f"""# Golden regression values

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
    (GOLDEN / "README.md").write_text(readme)


def write_v3_readme(
    meta: dict[str, Any],
    logits: np.ndarray[Any, Any],
    digests: dict[str, str],
    loss: dict[str, Any],
) -> None:
    """Provenance for the schema-v3 artifacts — which is *not* the baseline."""
    total = len(digests) - 1
    delta = "\n".join(f"| `{prefix}.*` | {why} |" for prefix, why in V3_DELTA.items())
    readme = f"""# Golden regression values — schema v3

Captured from **the current codebase** by `scripts/capture_golden.py`, at the
completion of Tier 3 (IMPROVEMENT_PLAN §4.2). Do not hand-edit — regenerate
instead.

These are not baseline-equivalence values and must not be read as such. The
schema-v1 files in the parent directory are the baseline; `v2/` is the Tier-2
architecture, frozen. These exist because **Tier 3 (T3-1 … T3-7) rebuilt three
of the four branches and the fusion**, after which neither "reproduces the
pre-refactor implementation bit-for-bit" nor "reproduces Tier 2 bit-for-bit" is
a property that can hold. Their job is forward drift detection: from here on,
any unintended change to the model's numerics fails against *these*.

| | |
|---|---|
| Source | working tree at Tier-3 completion |
| Captured (UTC) | {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} |
| torch | {torch.__version__} |
| numpy | {np.__version__} |
| Device | `cpu` |
| Parameters | 5,194,578 (§3.8 budgets ≈ 5,323,000; v2 was 7,856,203) |

## What changed relative to v2

Every key that appeared or disappeared falls under one of these declared
prefixes, and `capture_golden.py::check_schema_delta` fails if one does not.
A tensor-by-tensor list of 185 keys would not be readable, and an unreadable
declaration is not a check.

| Prefix | Why |
|---|---|
{delta}

Of the {total} tensors here, the ones whose *name* survived v2 largely did not
survive it unchanged: Branch D runs at `d_model = 192` where it ran at 256, and
`cross_interaction.branch_norms` are `BatchNorm1d` where they were `LayerNorm`
(FU-2). Value comparison against v2 is meaningless for a second reason too —
rebuilding the branches shifts the global RNG stream every later
`_init_weights` draw reads from.

## Procedures

Identical to v1's — `capture_golden.py::forward_pass` and `::train_step`, both
defined once and applied to both sides.

## Files

| File | Contents |
|---|---|
| `forward_logits_seed42.npy` | `float32 {logits.shape}` — eval-mode logits. |
| `init_state_sha256.json` | SHA-256 per initialised state-dict tensor ({total} entries) plus `__combined__`. |
| `stage1_epoch1_loss_seed42.json` | Scalar loss/accuracy plus combined SHA-256 of the model and EMA weights *after* one Stage-1 epoch. |

`physical_wl_spa40.npy` is not duplicated here — the wavelength vector is a
property of the dataset, not of the model schema, so every version reads the
parent directory's copy.

Combined init digest: `{digests["__combined__"]}`
Stage-1 epoch-1 loss: `{loss["loss"]!r}`

## Regenerating

    python scripts/capture_golden.py           # capture + verify v1 and v3
    python scripts/capture_golden.py --verify  # verify only

A regeneration is legitimate when a **declared** architecture change lands —
one carrying an `IMPROVEMENT_PLAN.md` item id and a test that pins it. If these
files need updating to make a test pass and no such item exists, the change is
the drift the gate exists to catch.
"""
    (GOLDEN_V3 / "README.md").write_text(readme)


def undeclared(keys: set[str]) -> list[str]:
    """Keys not covered by any prefix in :data:`V3_DELTA`.

    A declaration matches a whole path segment, so ``cross_interaction.latents``
    is covered by its own entry and never by ``cross_interaction`` alone — a
    bare-prefix match would let a whole module's worth of new tensors hide
    behind one declared sibling.
    """
    return sorted(k for k in keys if not any(k == p or k.startswith(p + ".") for p in V3_DELTA))


def check_schema_delta(v2: dict[str, str], v3: dict[str, str]) -> bool:
    """The one value-free comparison v2 and v3 still admit.

    Every key that left or arrived must fall under a declared prefix. Digests
    are deliberately *not* compared: three branches and the fusion were rebuilt,
    which shifts the RNG stream every later ``_init_weights`` draw reads from,
    so a value comparison here would be noise. What this rules out is a
    structural change that nobody wrote down riding along with Tier 3.
    """
    removed = set(v2) - set(v3) - {"__combined__"}
    added = set(v3) - set(v2) - {"__combined__"}
    stray_removed, stray_added = undeclared(removed), undeclared(added)
    ok = not stray_removed and not stray_added
    if ok:
        print(
            f"  ✓ v2 → v3 delta is entirely declared "
            f"(−{len(removed)}, +{len(added)}, {len(V3_DELTA)} prefixes)"
        )
    else:
        print("  ✗ v2 → v3 delta contains undeclared keys:")
        if stray_removed:
            print(f"      removed, undeclared: {stray_removed[:10]}")
        if stray_added:
            print(f"      added, undeclared:   {stray_added[:10]}")
    return ok


def compare_committed(
    directory: Path, logits: np.ndarray[Any, Any], digests: dict[str, str], loss: dict[str, Any]
) -> bool:
    """Check a freshly captured triple against what is committed in ``directory``.

    Missing files are not a failure — that is a first capture. Present files
    that disagree are.
    """
    ok = True
    logits_path = directory / "forward_logits_seed42.npy"
    if logits_path.exists():
        committed = np.load(logits_path)
        max_abs = float(np.max(np.abs(committed - logits)))
        if np.allclose(committed, logits, atol=1e-6):
            print(f"  ✓ {directory.name}/logits match (max |Δ| = {max_abs:.3e})")
        else:
            ok = False
            print(f"  ✗ {directory.name}/logits drifted: max |Δ| = {max_abs:.3e}")

    digest_path = directory / "init_state_sha256.json"
    if digest_path.exists():
        committed_digests = json.loads(digest_path.read_text())
        drifted = [k for k, v in committed_digests.items() if digests.get(k) != v]
        if drifted:
            ok = False
            print(f"  ✗ {directory.name}/init digests drifted: {len(drifted)} tensors")
            for key in sorted(drifted)[:10]:
                print(f"      {key}")
        else:
            print(f"  ✓ {directory.name}/init digests match ({len(committed_digests) - 1} tensors)")

    loss_path = directory / "stage1_epoch1_loss_seed42.json"
    if loss_path.exists():
        committed_loss = json.loads(loss_path.read_text())
        drifted_keys = [
            key
            for key in ("loss", "acc", "model_sha256", "ema_sha256")
            if committed_loss[key] != loss[key]
        ]
        for key in drifted_keys:
            ok = False
            print(
                f"  ✗ {directory.name}/{key} drifted: "
                f"committed={committed_loss[key]!r} now={loss[key]!r}"
            )
        if not drifted_keys:
            print(f"  ✓ {directory.name}/stage-1 epoch scalars and digests match")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ref", default=BASELINE_REF)
    ap.add_argument("--verify", action="store_true", help="compare only; write nothing")
    args = ap.parse_args()

    GOLDEN.mkdir(parents=True, exist_ok=True)
    GOLDEN_V3.mkdir(parents=True, exist_ok=True)

    print("═" * 78)
    print(f"  Golden capture   baseline {args.baseline_ref[:7]}   schemas v1 + v3")
    print("═" * 78)

    print("\n[1/4] Running the PINNED REFERENCE (schema v1)...")
    base_logits, base_digests, base_loss, wl, base_meta = capture_baseline(args.baseline_ref)
    print(f"      logits {base_logits.shape} {base_logits.dtype}")
    print(
        f"      init digest {base_digests['__combined__'][:16]}…  ({len(base_digests) - 1} tensors)"
    )
    print(f"      epoch loss {base_loss['loss']!r}  acc {base_loss['acc']!r}")

    print("\n[2/4] Running the CURRENT code (schema v3)...")
    new_logits, new_digests, new_loss, new_meta = capture_refactored(wl)
    print(f"      logits {new_logits.shape} {new_logits.dtype}")
    print(
        f"      init digest {new_digests['__combined__'][:16]}…  ({len(new_digests) - 1} tensors)"
    )
    print(f"      epoch loss {new_loss['loss']!r}  acc {new_loss['acc']!r}")

    print("\n[3/4] Structural checks...")
    ok = True

    if base_meta != new_meta:
        ok = False
        print("  ✗ construction parameters differ (config round-trip regression):")
        for k in sorted(set(base_meta) | set(new_meta)):
            if base_meta.get(k) != new_meta.get(k):
                print(f"      {k}: baseline={base_meta.get(k)!r}  current={new_meta.get(k)!r}")
    else:
        print(
            "  ✓ construction parameters identical "
            f"(num_classes={base_meta['num_classes']}, num_bands={base_meta['num_bands']}, "
            f"dropout={base_meta['dropout']})"
        )

    v2_digest_path = GOLDEN_V2 / "init_state_sha256.json"
    if v2_digest_path.exists():
        ok &= check_schema_delta(json.loads(v2_digest_path.read_text()), new_digests)
    else:
        print(f"  ! {v2_digest_path} missing — v2 → v3 delta not checked")

    print("\n[4/4] Comparing against the committed artifacts...")
    ok &= compare_committed(GOLDEN, base_logits, base_digests, base_loss)
    ok &= compare_committed(GOLDEN_V3, new_logits, new_digests, new_loss)

    if not ok:
        print("\n✗ Capture aborted — see the failures above.")
        return 1

    if args.verify:
        print("\n✓ Verify-only run passed; no files written.")
        return 0

    np.save(GOLDEN / "physical_wl_spa40.npy", wl.astype(np.float32))
    np.save(GOLDEN / "forward_logits_seed42.npy", base_logits.astype(np.float32))
    (GOLDEN / "init_state_sha256.json").write_text(json.dumps(base_digests, indent=2) + "\n")
    (GOLDEN / "stage1_epoch1_loss_seed42.json").write_text(json.dumps(base_loss, indent=2) + "\n")
    write_readme(args.baseline_ref, base_meta, base_logits, base_digests, base_loss)

    np.save(GOLDEN_V3 / "forward_logits_seed42.npy", new_logits.astype(np.float32))
    (GOLDEN_V3 / "init_state_sha256.json").write_text(json.dumps(new_digests, indent=2) + "\n")
    (GOLDEN_V3 / "stage1_epoch1_loss_seed42.json").write_text(json.dumps(new_loss, indent=2) + "\n")
    write_v3_readme(new_meta, new_logits, new_digests, new_loss)

    print(
        f"\n✓ Wrote v1 → {GOLDEN.relative_to(REPO_ROOT)}/  and v3 → "
        f"{GOLDEN_V3.relative_to(REPO_ROOT)}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
