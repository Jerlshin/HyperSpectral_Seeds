# Golden regression values — schema v2

Captured from **the current codebase** by `scripts/capture_golden.py`, at the
completion of Tier 2 (IMPROVEMENT_PLAN §4.2). Do not hand-edit — regenerate
instead.

These are not baseline-equivalence values and must not be read as such. The
schema-v1 files in the parent directory are the baseline; these exist because
**T2-10 (HD-1) changed the architecture**, after which "reproduces the
pre-refactor implementation bit-for-bit" is not a property that can hold. Their
job is forward drift detection: from here on, any unintended change to the
model's numerics fails against *these*.

| | |
|---|---|
| Source | working tree at Tier-2 completion |
| Captured (UTC) | 2026-08-07 13:23:58 |
| torch | 2.13.0 |
| numpy | 2.1.3 |
| Device | `cpu` |

## What changed relative to v1

| | |
|---|---|
| Keys removed | `linear_head.2.weight`, `linear_head.2.bias` — HD-1 deleted the Stage-1 linear head |
| Keys added | `arcface_head.confusion` — HD-3's pairwise confusion buffer |
| Tensors whose value is unchanged | 204 of 351 |

The 147 that moved did **not** move because their computation
changed. Removing `linear_head` removes its `nn.Linear`'s two construction
draws from the global torch RNG stream, and `_init_weights` runs after that
point, so every subsequent `kaiming_normal_` / `trunc_normal_` reads from a
stream offset by two draws. The tensors that did not move are the ones
initialised by `ones_`/`zeros_`, which consume no randomness. This is checked,
not asserted: `capture_golden.py::check_schema_delta` fails if the key delta is
anything other than the line above.

## Procedures

Identical to v1's — `capture_golden.py::forward_pass` and `::train_step`, both
defined once and applied to both sides. The one call-site difference is that
the Stage-1 epoch now passes `arc_m=cfg.stage1.arcface_m` (`0.0`), because
under HD-1 there is no separate linear head to select and a zero margin is what
makes mixup admissible.

## Files

| File | Contents |
|---|---|
| `forward_logits_seed42.npy` | `float32 (4, 90)` — eval-mode logits through the unified cosine head. |
| `init_state_sha256.json` | SHA-256 per initialised state-dict tensor (351 entries) plus `__combined__`. |
| `stage1_epoch1_loss_seed42.json` | Scalar loss/accuracy plus combined SHA-256 of the model and EMA weights *after* one Stage-1 epoch. |

`physical_wl_spa40.npy` is not duplicated here — the wavelength vector is a
property of the dataset, not of the model schema, so both versions read the
parent directory's copy.

Combined init digest: `6a9b21c5b8fd57f6bd87b5ecef5f9e56ece584efe06c84772e65e468115c03be`
Stage-1 epoch-1 loss: `23.829429626464844`

## Regenerating

    python scripts/capture_golden.py           # capture + verify both schemas
    python scripts/capture_golden.py --verify  # verify only

A regeneration is legitimate when a **declared** architecture change lands —
one carrying an `IMPROVEMENT_PLAN.md` item id and a test that pins it. If these
files need updating to make a test pass and no such item exists, the change is
the drift the gate exists to catch.
