# S22 amendment05: complete the screen on the existing two-T4 host

2026-10-05 · Before any S22 held-out score. Amendment02's **one seed × both folds**
allocation and H40-screen gates remain fixed. Restore the originally planned
`runtime=kaggle_t4x2`, two-rank DDP, fp16/GradScaler for both completed controls.

The local workerless MPS attempt completed six epochs but took about209–294s per
epoch. It was interrupted and preserved in `outputs/s22_partial_amendment03/f0_s0/`;
compact provenance, epoch marker and log are tracked in `amendment03_partial/`.
It produced no final evaluation and is not a completed screening cell. The initial
worker failure under02 also remains preserved. No partial-fit score is used to rank
models, choose the screening gate or tune the architecture.

Read-only discovery found the already configured Kaggle research account and the
prior private `jgfreak/hsi-training` notebook, attached to
`jerlshinjg/dataset-u430k32`. Its ten-file inventory matches the required assets.
The account reported30 available GPU hours. A new **private** kernel runs exactly
folds0/1 × seed0 sequentially; byte-level source/data checks must pass on the host
before training. Credentials are loaded locally for authenticated API calls and
are absent from the uploaded source bundle and research receipts.

The executable child is `configs/research/s22_screening_amendment05.json`, SHA256
`1dd50dae26f356ac963b4713ae5820a5476ede2129f8c554178a09abc7ea7f95`.
It hashes parent03 and retains the full02/original parent chain. The launcher source
is unchanged from02 and receives `--plan ...amendment05.json` explicitly. CUDA
availability and exactly two devices are required; no device fallback is allowed.
The separate `s22_analysis_code05.json` receipt pins the matching analyzer with
unchanged scoring/statistical rules. The previous sealed code/receipts remain intact.

## Current commands

```sh
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --plan configs/research/s22_screening_amendment05.json --fold 0 --seed 0
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --plan configs/research/s22_screening_amendment05.json --fold 1 --seed 0
# After returning both complete output folders:
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/analyze_s22_cuda_screening.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python docs/research/evidence/S22_complementary_v5/code/read_screen.py
```

Source bundle, kernel metadata and file hashes are in
`evidence/S22_complementary_v5/kaggle_dispatch_manifest.json`. The bootstrap saves
complete/partial output even on error. Runtime versions, GPU identities, source
origin and final per-cell provenance must be retained. This is portable source
identity, not a claim that the Kaggle environment equals the historical environment.

Original six-cell S22, amendments02/03 and both partial attempts remain historical.
No seed1/2 encoder is run by this amendment. S17/S18 stay unchanged.

Interruption bookkeeping correction: the stop decision observed five complete
epochs, as stated in the immutable05 plan's motivation. Delivery after asynchronous
GPU work allowed epoch6 to complete. `amendment03_partial/interruption_receipt.json`
records the actual marker and this correction; there was still no held-out evaluation.
