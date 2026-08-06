# Golden regression values

Captured from the **pre-refactor** code by `scripts/capture_golden.py`
(REFACTOR_PLAN.md §3.2.2). Do not hand-edit — regenerate instead.

| | |
|---|---|
| Source git SHA | `886560fe531c99197f20c2ebd06e0bc7ded8ac8f` |
| Source file | `HSI_modality_training/hsi_training.py` |
| Captured (UTC) | 2026-08-06 16:09:01 |
| torch | 2.13.0 |
| numpy | 2.1.3 |
| Device | `cpu` |

## Procedure

Defined once in `capture_golden.py::forward_pass` and run identically against the
baseline and the refactored model:

1. `set_seed(42)` — immediately before construction, so the per-branch
   `_init_weights` draws consume the global torch RNG in the same order the
   baseline's import-time seeding produced (§3.6).
2. Load the physical wavelengths (consumes no RNG) and build `SpectralQuadNet`
   with `num_classes=90`, `num_bands=40`,
   `dropout=0.15`, `wl_embed_dim=16`.
3. `.to("cpu").eval()` — eval mode makes the forward deterministic (no branch
   dropout, no `torch.rand` draws) and returns a plain tensor rather than the
   training-mode dict of auxiliary logits.
4. Input: `torch.randn(4, 40, 64, 64)` from a
   dedicated `torch.Generator().manual_seed(42)`, so the input is independent of
   how much RNG construction consumed.
5. One `torch.no_grad()` forward pass.

## Files

| File | Contents |
|---|---|
| `physical_wl_spa40.npy` | `float32 (40,)` — min-max-normalised wavelengths from `./dataset/wavelengths_spa_40b.csv`. Committed so the test never needs the gitignored `dataset/`. |
| `forward_logits_seed42.npy` | `float32 (4, 90)` — eval-mode logits. |
| `init_state_sha256.json` | SHA-256 per initialised state-dict tensor (352 entries) plus `__combined__`. Catches construction-order drift that a 4-sample forward could average away. |

Combined init digest: `174577e1f6be92042f5c8cdab4fdcb77071a973fc55c529fc5cb2ca64c8be983`

## Not yet captured

`stage1_epoch1_loss_seed42.json` — §3.2.2's second artifact. It needs
`train_one_epoch`, which Phase 3 relocates; the Phase 3 gate captures it.

## Regenerating

    python scripts/capture_golden.py           # capture + verify
    python scripts/capture_golden.py --verify  # verify only

A regeneration is only legitimate when the *baseline* reference changes. If these
files need updating to make a test pass, the refactor has changed behaviour —
that is the failure the gate exists to catch.
