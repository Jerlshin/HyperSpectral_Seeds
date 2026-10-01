# S08 · Neural confirmation on the reflectance cube

| | |
|---|---|
| **Status** | **protocol sweep run** (2026-10-01, u430k32 only) — analysed in [S09](../S09_post_sweep_forensics/README.md); budget arms (FW-03) not run |
| **Dates** | 2026-10-01 → |
| **Commits** | `8d3a7d2` (pre-sliced cube, Kaggle T4 × 2 profile, JSONL logging, mid-stage resume) |
| **Data** | refl-215 cube; pre-sliced `./dataset_u430k32` (uniform430 k = 32, float16, 2.33 GB) |
| **Code** | `train.py`, `scripts/run_protocol.py --data ablation/u430k32 --nproc-per-node 2`, `runtime=kaggle_t4x2`, `data=ablation/u430k32_{grouped,stratified}` |
| **Builds on** | F17, F21, F23 · **Could reverse** D04, D05, D06, D11 |
| **Backlog items** | [FW-02, FW-03, FW-04](../../FUTURE_WORK.md) |

## 1 · Question
What does SpectralSeedNet actually score under the grouped protocol on reflectance — overall, on
same-session and on cross-session varieties — how noisy is that, and how does it depend on the band
budget?

## 2 · Why we are doing this
Every result since the audit is a proxy result. The network the codebase is built around has never
been trained under the revised protocol.

## 3 · Hypotheses — to be frozen before any held-out row is read
Draft, to be moved into `HYPOTHESES.md` and frozen per WORKFLOW §3.2:
- **H9 (budget)** — uniform430 at some k ∈ {24, 32, 48, 64} is non-inferior (≥ −0.01 held-out
  macro-F1, beyond 2σ) to the full 215-band cube. Reverses D04 if supported.
- **H10 (session)** — the network's cross-session recall exceeds 0.05 on reflectance.
- **H11 (gap)** — `F1_stratified − F1_grouped` > 2σ (A1).
- Order: A12 (σ, 5 seeds) first; nothing else is interpretable without it.

## 4 · Planned method
Grouped, folds 0 and 1, 3 seeds; select on calib; score `val ∪ test` once, with and without TTA;
session report (D14) on every run. Arms: full 215 (`data=refl215_grouped`), uniform430 k ∈ {16, 24,
32, 48, 64} (`data.band_indices_path=outputs/band_finalists/uniform430_k{K}.npy …`, or the
pre-sliced cube for k = 32), and the stratified twin of the chosen arm for A1.

## 5 · Results
The protocol sweep ran on 2026-10-01: grouped folds 0, 1 × seeds 0–2 and stratified × seeds 0–5 on
`dataset_u430k32`, plus LDA/LinearSVC baselines (`outputs/experiments_u430k32/`). Headline: grouped
0.530 ± 0.009, stratified 0.712 ± 0.033 macro-F1 (TTA); cross-session recall 0.152. Everything the sweep
shows — and what it does not — is analysed in **[S09](../S09_post_sweep_forensics/README.md)** rather than
repeated here. H9 (budget) was not tested: only k = 32 ran. H10 and H11 were drafts, never frozen, so their
outcomes are recorded as observations (HYPOTHESES §4).

## 8 · Threats to validity (anticipated)
- fp16 on T4 vs bf16 elsewhere; the pre-sliced cube's float16 cast error (recorded in
  `band_axis.json`) — both are runtime/storage choices that must not move a metric; check on one arm.
- k = 32 was pre-sliced although the frozen rule gave k\* = 24 (D15, FW-09) — include k = 24.

## 10 · Run
See README §10 "Kaggle — GPU T4 x2" for the notebook cells. Locally:
```bash
python scripts/build_presliced_dataset.py
python scripts/run_protocol.py --dry-run --data ablation/u430k32                 # the plan, free
python scripts/run_protocol.py --nproc-per-node 2 --data ablation/u430k32 \
    --override runtime=kaggle_t4x2 --baseline                                     # 2 folds × 3 seeds, both protocols
```
