# S40 · SeedNet v5 on all 215 calibrated bands (seed 0, both corrected folds, Kaggle T4 x2)

| | |
|---|---|
| **Status** | **Prepared and frozen; not run.** No GPU result exists. The two CUDA cells run on Kaggle T4 x2 (README §10) |
| **Plan** | `configs/research/s40_full215_v5.json`, SHA-256 `744c65bccf17878c00e0bb5b872125e4cf74889c67d21f05656da870ab0e44da`, 172 hashed inputs |
| **Cells** | fold 0 seed 0, fold 1 seed 0. Exactly two; the runner refuses any other (checked) |
| **Matched reference** | S22 k32 v5 cells `outputs/s22_complementary_v5/f{0,1}_s0` (amendment05, `1dd50dae…`) |
| **Code** | `scripts/run_full215_v5.py` (driver), `scripts/build_refl215_f16.py` (cube), `configs/data/refl215_f16_grouped.yaml`; training is `scripts/run_complementary_v5.py`, unchanged; profile `scripts/profile_full215_v5.py`; tests `tests/unit/test_s40_full215.py`, `tests/smoke/test_s40_full215_driver.py` |
| **Evidence** | [`evidence/S40_full215_v5/`](../../evidence/S40_full215_v5/) |

## 1 · Question

SeedNet v5 has been trained only on `uniform430_k32`: 32 of the 215 white-tile reflectance bands,
stored as float16. That subset was chosen because it fit a 2.3 GB Kaggle upload. No experiment chose
it, and the band study found the label-free uniform null hard to beat (F11, F18). S40 asks what the
same network, regime, folds and seed do when they read **all 215 bands**. It is a development run:
one seed, a descriptive comparison, no gate (D59).

## 2 · What changes, and what does not

Only `data=` changes: `ablation/u430k32_grouped` → `refl215_f16_grouped`. Every other override, the
R1 regime, the frozen S21 rows, global batch 128, fp16 + GradScaler on two T4s (DDP, SyncBatchNorm),
200 epochs, patience 40, 12-view TTA and the result files are those of S22 amendment05.

**No model adaptation was needed.** Every width that depends on the band count is derived from
`data.num_bands`:

| | k32 (S22) | 215 (S40) |
|---|---|---|
| 3-D stem spectral strides / kernels | (2,2,1) / (7,5,5) | (8,2,2) / (15,5,5) |
| depth reaching the fold → fold input | 8 → 512 | 7 → 448 |
| bands never read by a stage-1 tap | 0 | 0 (one-hot check, every band) |
| spectral descriptor (`snv_morph`) | 32 + 8 | 215 + 8 |
| CutMix / cutout window (bands) | 6 / 2 | 43 / 16 (same 20 % / 7.5 %) |
| parameters | 2,725,700 | **2,761,782** (spatial −11,136, spectral +47,214, ECA +4) |

## 3 · The 215-band cube

`dataset/patches.npy` (float32, 30.4 GB) is no longer kept locally, and 19 GB was free.
`build_refl215_f16.py` reruns the unchanged extraction from `dataset/rice_hsi.zip` and streams
each patch to float16, so only the 15.2 GB cube reaches the disk (452 s). It then **checked identity
rather than assuming it:**

- `labels`, `groups`, `masks`, `morphology`, `scan_table.csv`, `wavelengths.csv` and `gain` are
  byte-identical to `./dataset/`. `radiometry.json` is equal apart from the archive path. This is
  the row alignment that the S21 folds, the morphometrics and the session table depend on.
  `labels`/`groups` hash to the S21 partition's recorded values;
- the 32 `uniform430_k32` bands of the new cube are **bit-identical** to
  `dataset_u430k32/patches.npy`, in all 8,624 rows. So S40 reads the very values the k32 runs
  read, plus the 183 bands they did not;
- float16 cast, measured: max relative error 4.88e-4 (normal range), RMS 4.7e-5. 163,761 of
  1.96e9 non-zero values flushed to zero, the largest 3.4e-15.

| File | Shape / content | Size |
|---|---|---:|
| `patches.npy` | (8624, 215, 64, 64) float16 | 15,189.3 MB |
| `masks.npy` | (8624, 64, 64) float16 | 70.6 MB |
| `labels.npy`, `groups.npy` | (8624,) int64 | 0.07 MB each |
| `morphology.npy` | (8624, 8) float32 | 0.3 MB |
| `wavelengths.csv`, `scan_table.csv`, `radiometry.json` | axis (383.2–1006.5 nm, 608–706 absent), sessions, radiometry | < 0.1 MB |
| `band_axis.json` | set `all`, 215 of 215, cast error, build record | 6 KB |
| `MANIFEST.json` | size + SHA-256 of every file (cube `4185aa93…`) | 1 KB |
| **total** | | **15.26 GB** |

`band_axis.json` records 215 of 215 source bands. The run therefore prints
`Spectral: 215 bands — the full acquired cube, no band selection` and records
`band_selection: false`, which is checked in the smoke run.

## 4 · Engineering validation (all CPU/MPS; no GPU result is claimed)

| Check | Result |
|---|---|
| Plan and inputs | `verify_plan`: 172 hashed inputs (all of `src/`, `configs/`, `train.py`, the runner, the driver, the builder, the S21 partition, the cube's identity files); all git-tracked |
| Authorized cells only | runner refuses f0/f1 × seeds 1, 2 ("not authorized"); seed 0 stops at "CUDA unavailable" on a Mac |
| Frozen rows | fold 0: train 3,681 · calib 630 · val 2,156 · test 2,157; fold 1: 3,683 · 630 · 2,154 · 2,157; disjoint; no held-out scan in train |
| DDP: each row scored once | S40 smoke on 2 gloo ranks with odd 33/33 halves: every val∪test and calib row exactly once. The production path did the same on T4 x2: S22 k32 cells scored 4,313 / 4,311 rows exactly once despite the odd 2,157-row test half ([audit](../../evidence/S40_full215_v5/s22_ddp_dedup_audit.json)) |
| Resume | smoke: SIGKILL of the whole launcher after epoch 1, then the same command → `[RESUME] … continuing at epoch 2`, then fold 1. A third run skips both cells. COMPLETE.json and per-cell checks are written |
| Unit tests | `test_s40_full215.py` (8): config ≡ primary axis, regime ≡ S22, stem coverage, parameter pin, train/eval shapes and finite gradients, both pathways live, plan cells/regime, driver cell states, float16 stream ↔ unchanged finaliser |

### Memory and runtime (measured on an Apple M5, extrapolated to a T4)

| Per rank (batch 64) | k32 | 215 | ratio |
|---|---:|---:|---:|
| training FLOPs per sample (fwd+bwd) | 3.08 G | 4.80 G | **1.56×** (95 % in the 3-D stem) |
| autograd-saved activations, fp32 (CPU) | 1.30 GB | 2.37 GB | 1.83× |
| held between forward and backward, fp16 autocast (MPS) | 1.06 GB | 1.73 GB | 1.64× |
| host CPU per augmented training sample | 0.27 ms | 1.59 ms | 5.9× |
| host CPU per evaluation sample | 0.13 ms | 1.09 ms | 8.4× |

- **GPU memory:** about 2–4 GB per T4, including the cuDNN workspace, out of 15 GB. The k32 run fit
  with a large margin, and batch, precision and checkpointing need no change.
- **Host:** the 15.3 GB cube fits the ~29–31 GB page cache. The driver's re-hash and the runtime's
  prewarm read it once; every later patch read is then served from RAM.
- **Wall clock:** k32 v5 took ≈ 23 min per cell (7.3 s/epoch, 177–189 epochs) and 47 min for both
  cells on T4 x2 (S22). The 215-band compute is 1.56× and the host feed 6× per sample, spread over
  2 workers per rank. The expected time is **≈ 35–60 min per cell, ≈ 1.2–2 h for the session**, with
  ≈ 3 h if the 4-vCPU feed binds. This is an extrapolation; the epoch-1 ETA line gives the real
  figure. Kaggle's limit is 12 h.
- MPS step times (13.7 s vs 2.5 s) mainly reflect Metal's slow Conv3d. They are recorded, not
  used for the estimate.

## 5 · Outputs (per cell, `outputs/s40_full215_v5/f{0,1}_s0/`)

The S22 set: `provenance.json` (plan SHA, fold, seed), `resolved_config.yaml`, `metrics.jsonl`,
`clean_fit.json`, `best_stage1.pth` + `stage1_meta.json` (selected on calib),
`last_stage1.pth/.json` (resume state), and `results/`: `run.json` (git commit, environment,
regime, parameter breakdown, band geometry, metrics + CIs, no-TTA and TTA), float16 logits + rows +
targets for val∪test **and calib**, predictions, confusion, per-class and per-session tables, and
the session probe. The driver adds `logs/f*_s0.log`, `sessions.jsonl` (host, GPUs, torch/CUDA,
commit, plan and cube hashes, every attempt), `checks/f*_s0.json` and `COMPLETE.json`.
Embeddings and train-row TTA can be extracted later on CPU from the checkpoints, as in S24/S27.

## 6 · Analysis (after the run)

This is a descriptive comparison with S22 f{0,1}_s0. It reports per-fold and mean macro-F1
(no-TTA, TTA), same/cross-session recall, from/to session 8, and a paired 2,000-resample class
bootstrap of the delta. No gate: one seed, and seed SD exists only at k32 (S39). Nothing here is a
band-selection or paper claim until it is replicated.
