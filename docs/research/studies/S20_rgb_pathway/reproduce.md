# S20 reproduction and artifact contract

Run commands from the repository root. Dependencies are the project's runtime plus
`prep` and `dev` extras; the feature cache additionally uses an existing local
DINOv2 checkout/checkpoint. Exact versions and source hashes are in the evidence.
No command below downloads a model or starts GPU training implicitly.

## Local artifact locations

- `dataset/rice_hsi.zip`: verified original archive, read-only.
- `dataset_u430k32/`: immutable historical reference.
- `dataset_rgb_hsi_v3/`: final paired assets, native masks, all-scan overlays,
  compact full-band summaries/regions, exclusions, build configuration and hashes.
- `outputs/s20_rgb_features_v3/`: four frozen 384-D RGB feature caches with provenance.
- `outputs/s20_rgb_study/`: immutable frozen-screen predictions, metrics and completion hashes.
- `docs/research/evidence/S20_rgb_pathway/`: tracked audit, predictive manifest,
  compact results, saved predictions and descriptive analysis code.

Large arrays, native masks and full probability matrices stay in ignored local
artifact folders. Compact predictions and scientific evidence are tracked under
`docs/research/`; the raw archive and checkpoint are not vendored. `v1` and `v2`
assets/caches are rejected preprocessing work, never model results. Their compact
provenance and QA examples are preserved in the study evidence.

## Build and validate assets

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/prepare_multimodal.py \
  --archive-sha256 92af3258ba72950e301ee7f4030f59cc88922e21f34d00a06d68debb6b046c14
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/research/evidence/S20_rgb_pathway/code/audit_assets.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/research/evidence/S20_rgb_pathway/code/review_missing.py
```

The builder refuses to overwrite a completed dataset. For interrupted runs, repeat
with identical source/configuration: completed shards are reused, their hashes are
checked during assembly, and a final manifest is published only after completion.
For a genuinely new preprocessing version use a new output directory. `--limit 1`
creates a preflight; after that disjoint `--scan-start/--scan-stop --shard-only`
workers can run, followed by a normal invocation to assemble. Do not edit source
while a version is building. `--dense` additionally writes the full 215×64×64 cube;
it was **not** used in S20 and needs about twice its 14-GiB size during assembly.

Inspect the pairing atlas and largest-residual crop pairs before predictive scoring.
Masks are reviewed estimates, not pixel-level ground truth. The strict spectral
axis excludes any wavelength with a non-own white reference; explicit historical
controls retain the old calibration policy.

## Extract frozen RGB features

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python scripts/cache_rgb_features.py \
  --model-repo /path/to/facebookresearch_dinov2_main \
  --checkpoint /path/to/dinov2_vits14_pretrain.pth
```

`--device cpu` is the default. CUDA extraction is prepared with `--device cuda` and
an independently named `--output`; it has not been executed here and cannot be
silently substituted into the completed screen. The exact checkpoint, all cached
Python source files, extraction script, data manifest, transforms, device, threads,
batch size and runtime version are recorded. Two independent workers with disjoint
`--views rgb gray` / `--views rgb32 silhouette` were used locally (two CPU threads
each). No views share a writable output file. Partial views must be restarted in a
new cache directory or their uncompleted `.partial.npy` explicitly removed.

## Freeze once, run once

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_rgb_study.py freeze
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python scripts/run_rgb_study.py run
```

These are historical reproduction instructions, **not instructions to rerun a
completed confirmation to obtain a preferred result**. The first command refuses
an existing plan. The second verifies the plan's separate SHA-256 and all frozen
inputs, then refuses an existing result directory. The JSON manifest is the runnable
experiment configuration: 30 arms, exact splits/axes, classifier, temperatures,
fusion choices, thresholds and bootstrap policy. To reproduce the same frozen run
for engineering verification, pass a new `--output` and label it a replay, not new
independent evidence. To ask a new scientific question, freeze a new study.

All predictor fitting uses training rows. Calibration chooses temperatures and one
predeclared fusion-weight grid; it shares training acquisitions and is not a transfer
estimate. No preprocessing/candidate choice is optimized on held-out predictions.
Historical v5 logits are aligned by exact legacy row and target identity.

## Analyze without retraining

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python docs/research/evidence/S20_rgb_pathway/code/analyze_results.py
python docs/research/evidence/S20_rgb_pathway/code/draw_figures.py
PYTHONPATH=src python docs/research/evidence/S20_rgb_pathway/code/validate_study.py --assets
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 pytest -q tests/unit/test_rgb_pathway.py
```

Analysis reads saved predictions only. `summary.json` remains byte-identical to the
runner output; `summary_with_accuracy_ci.json` adds a separately named descriptive
accuracy interval. F1 intervals resample fixed per-variety contributions, retaining
both selected legacy folds and original precision denominators (not necessarily both acquisition directions; see README.md). They are not population
intervals over new sessions. The v5 aggregate reuses three HSI seeds with the same
RGB probe per fold; it is not three RGB replications.
