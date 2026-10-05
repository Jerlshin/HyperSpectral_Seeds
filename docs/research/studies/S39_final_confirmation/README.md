# S39 · Matched final confirmation of SeedNet-MX (replaces the S30 brief)

| | |
|---|---|
| **Status** | **frozen**. RGB seed cells queued locally; HSI seed cells prepared, **not launched** (need Kaggle authorization) |
| **Plan** | `configs/research/s39_final_confirmation.json`, SHA-256 `d1db0694929387b2b92ce91a0ff47c28e5b4910b58316d94e6ef23e2ccb5a770`. Frozen before S37 was scored |
| **HSI amendment** | `configs/research/s39_hsi_seeds_amendment06.json`, SHA-256 `ce3ded4bfa923c0d487d130f647c3266d57d73aed207448bc80937aacbb7112b`. It is S22 amendment05 with only the cell list changed |
| **Code** | `scripts/run_final_confirmation.py` (freeze · train-rgb · run [--partial]); `evidence/S39_final_confirmation/code/{seal_hsi_amendment,build_kaggle_bundle}.py`; test `tests/unit/test_final_confirmation.py` |
| **GPU bundle** | `outputs/s39_kaggle_push/` (built, not pushed); record in `evidence/S39_final_confirmation/kaggle_dispatch_manifest.json` |

## Why this replaces S30

S30 was designed to confirm the S29 system: frozen ViT-L, a learned head and v5. D55 retired that
system as a baseline, because the trained RGB branch beats it by +.071 (F119). Its contrasts
(head vs fixed, ViT-L vs ViT-S) concern retired components. S39 confirms the S36 architecture
against the systems the owner named as baselines, with seeds 0/1/2 on both corrected folds.

## Allocation

| Fits | Where | Cost |
|---|---|---|
| RGB branch seeds 1, 2 × folds 0, 1 (branch = S37 arm if H57 passed, else S32 ViT-B) | local MPS (queued) | ≈ 3.3 h (S32/S33 rate) |
| HSI v5 seeds 1, 2 × folds 0, 1 | private Kaggle 2×T4 | ≈ 95 min wall (S22 rate); **owner authorization required** |
| Seed-0 cells | reused (S22, S32/S37) | 0 |

## Frozen contrasts

Each contrast is a mean over 3 seeds × 2 folds. The paired variety CI is computed on seed-averaged
per-class scores. Gate: Δ ≥ threshold, CI > 0, positive on each fold, cross Δ ≥ 0, and
Δ > 2 × seed SD (G3).

| ID | Contrast | Threshold |
|---|---|---:|
| M1 | SeedNet-MX − equal fusion of v5 with the frozen ViT-L probe | .02 |
| M2 | SeedNet-MX − HSI v5 alone | .05 |
| M3 | trained RGB − frozen ViT-L probe (RGB alone; scored early with `--partial`) | .05 |
| M4 | SeedNet-MX − trained RGB alone (multimodal value) | .01 |

## To finish

```sh
# after the queued RGB cells (log outputs/s39_train.log): M3 is scored automatically (--partial)
python docs/research/evidence/S39_final_confirmation/code/build_kaggle_bundle.py   # already built
kaggle kernels push -p outputs/s39_kaggle_push            # ONLY with the owner's authorization
# download s39_cuda_outputs.tar.gz -> outputs/s22_complementary_v5/f{0,1}_s{1,2}/, then:
PYTHONPATH=src:scripts python scripts/run_final_confirmation.py run
```

Transfer claims additionally need away-from-session-8 gains. Broad session/lot claims need the
crossed acquisition pilot (FW-42), applied to the locked system.
