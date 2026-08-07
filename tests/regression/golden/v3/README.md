# Golden regression values — schema v3

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
| Captured (UTC) | 2026-08-07 18:00:28 |
| torch | 2.13.0 |
| numpy | 2.1.3 |
| Device | `cpu` |
| Parameters | 5,194,578 (§3.8 budgets ≈ 5,323,000; v2 was 7,856,203) |

## What changed relative to v2

Every key that appeared or disappeared falls under one of these declared
prefixes, and `capture_golden.py::check_schema_delta` fails if one does not.
A tensor-by-tensor list of 185 keys would not be readable, and an unreadable
declaration is not a check.

| Prefix | Why |
|---|---|
| `branch_b.stat_attn.*` | T3-1 / BR-1 — the nine moments are gone, and with them the statistic-attention gate over them. |
| `branch_b.input_proj.*` | T3-1 / BR-1 — Branch B no longer has a band axis to project. |
| `branch_b.tower_s.*` | T3-1 / BR-1 — the three 1-D towers read a rank-2 tensor (§2.2.5); replaced by an MLP over a full-rank descriptor. |
| `branch_b.tower_m.*` | T3-1 / BR-1 — see `branch_b.tower_s`. |
| `branch_b.tower_l.*` | T3-1 / BR-1 — see `branch_b.tower_s`. |
| `branch_b.fusion.*` | T3-1 / BR-1 — nothing left to fuse; 686 k -> 95 k. |
| `branch_b.pool_attn.*` | T3-1 / BR-1 — no band axis to attention-pool over. |
| `branch_b.proj.*` | T3-1 / BR-1 — replaced by `branch_b.mlp`. |
| `branch_b.wl_pe_module.*` | T3-1 / BR-1 — the wavelength PE indexes bands, and Branch B no longer processes a per-band sequence. |
| `branch_c.band_reduce.*` | T3-2 / BR-3 — the two 1x1 convolutions that collapsed the spectral axis before any spatial kernel ran (C-3). |
| `cross_interaction.latents.*` | T3-4 / FU-1(b) — the four Perceiver latents, which 0-E measured as collapsed onto one function (M-1). |
| `cross_interaction.blocks.*` | T3-4 / FU-1(b) — the cross/self-attention stack. With five modality tokens, latent attention compresses nothing. |
| `cross_interaction.output_proj.*` | T3-4 / FU-5 — the same pre-LN residual MLP as `embed_net`, applied to the same vector (N-10). |
| `branch_a.derivatives.*` | T3-5 / FE-1(b) — the Savitzky-Golay operators on the irregular lambda grid (buffers, no parameters). |
| `branch_a.stem.*` | T3-5 / FE-1(a) — the Conv1d stem became a LambdaConv1d whose kernel is generated from wavelength offsets. |
| `branch_b.index_bank.*` | T3-1 / BR-1(i) — the two soft band selectors of the learned NDI bank. |
| `branch_b.in_norm.*` | T3-1 / BR-1 — LayerNorm over the [indices || depths || morphometrics] descriptor. |
| `branch_b.mlp.*` | T3-1 / BR-1 — the branch's whole parameter budget, on a full-rank input. |
| `branch_c.stem.*` | T3-2 / BR-3 — the factorised 3-D spectral-spatial stem. The only module in the network that sees the full cube. |
| `branch_d.lambda_bias.*` | T3-3 / BR-4(ii) — the relative-lambda attention bias b_psi. |
| `cross_interaction.branch_norms.*` | T3-4 / FU-2 — BatchNorm1d replaces LayerNorm, so the normaliser is a dataset statistic (M-2a); five of them, not four (FU-4). |
| `cross_interaction.bilinear.*` | T3-4 / FU-1(b) — the low-rank projections U_m of the second-order term (M-3). |
| `cross_interaction.bilinear_out.*` | T3-4 / FU-1(b) — V, projecting the bilinear pool back to d. |
| `cross_interaction.output.*` | T3-4 / FU-1(b) — W_o over [first order || second order]. |
| `morphology_embed.net.*` | T3-4 / FU-4 — the fifth modality token, from the eight morphometrics M-13 computed and discarded. |

Of the 306 tensors here, the ones whose *name* survived v2 largely did not
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
| `forward_logits_seed42.npy` | `float32 (4, 90)` — eval-mode logits. |
| `init_state_sha256.json` | SHA-256 per initialised state-dict tensor (306 entries) plus `__combined__`. |
| `stage1_epoch1_loss_seed42.json` | Scalar loss/accuracy plus combined SHA-256 of the model and EMA weights *after* one Stage-1 epoch. |

`physical_wl_spa40.npy` is not duplicated here — the wavelength vector is a
property of the dataset, not of the model schema, so every version reads the
parent directory's copy.

Combined init digest: `3bfb3567e3a3def362fac77cb0850316a2b0016e44b2dd36bd1b028cd2fe8a01`
Stage-1 epoch-1 loss: `23.080477237701416`

## Regenerating

    python scripts/capture_golden.py           # capture + verify v1 and v3
    python scripts/capture_golden.py --verify  # verify only

A regeneration is legitimate when a **declared** architecture change lands —
one carrying an `IMPROVEMENT_PLAN.md` item id and a test that pins it. If these
files need updating to make a test pass and no such item exists, the change is
the drift the gate exists to catch.
