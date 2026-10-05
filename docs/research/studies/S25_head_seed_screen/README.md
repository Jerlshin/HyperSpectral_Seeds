# S25 · Selected-head initialization sensitivity

2026-10-05 · **Complete; H43 passes.** [Results](results.md): head gain is stable
at .010227–.010780 F1 versus equal-single, descriptive SD .000279; encoder seed
remains 0. The specification below was a prospective allocation extension after S23/S24, before any new head
fit or result. The original S23 single-head-seed study and stopping rule remain
unchanged. This separate frozen study adds **head seeds 1/2 on both fixed S22
seed-0 encoders** for the retained S23 candidate only. It does not replicate encoder
training, RGB pretraining, branch-removal candidates, or every intermediate baseline.

## Why this selective replication is justified

S23 gains .010780 F1 [.002308,.019329] over matched equal fusion, consistently
across both folds. Its margin over stronger TTA fusion is only .002717 and uncertain.
S24's smaller corrections do not pass the full learning/practical gate; both-feature
correction remains the provisional candidate. Head initialization could plausibly
change these small margins. S24 now supplies verified frozen feature caches, so
four additional head fits cost little and need no new encoder fit or GPU job.
This is the minimum useful initialization check before expensive full-system work.

## Frozen recipe and gate

Retain S23's exact model, training recipe, feature scales, probability temperatures
and partitions. Seed 0 outcomes are reused. Only head initialization/training RNG
changes to 1/2; head seed and encoder seed are separately recorded. Train gradients
and calib selection, including epoch 0; load held-out cache only after both new
fold-specific checkpoints have been selected. Replay the existing seed-0 head and
equal anchor on that cache and require exact S23 predictions before scoring new heads.

**H43-screen:** mean head gain versus equal-single ≥.01, paired variety interval
>0, positive fold-mean gains, positive two-fold gains for every head seed, and
nonnegative mean cross delta. Additionally, every head seed's two-fold mean F1
and cross recall must meet S22 equal-TTA. Report all cells and descriptive seed SD.
Intervals resample varieties after averaging these fixed head seeds; they do not
estimate encoder or independent-session variance. A pass is a provisional
architecture screen, not final paper confirmation.

Exactly four cached head fits and no sweep. No automatic new encoder training.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_head_seed_screen.py freeze
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_head_seed_screen.py run
```

Freeze pins all inherited plans/source/data/checkpoint/probe inputs, S23 outcomes
and seed-0 heads, S24 cache hashes, and the new runner. Guards refuse changed inputs,
inconsistent cached predictions or output overwrite. Full-system confirmation must
eventually vary HSI and head initialization together on both corrected folds, with
matched controls and relevant mechanism ablations. New acquisitions remain required.
