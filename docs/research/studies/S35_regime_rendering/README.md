# S35 · Class-conditional acquisition rendering (CCAR)

| | |
|---|---|
| **Status** | complete — **H55 and H56 fail**: class-conditional rendering changes nothing |
| **Dates** | 2026-10-05 |
| **Data** | S33 identity and σ = 0.5 rendered 4-view logits (S32 `ft_vitb`, S33 `vitb_acq`); S31 label-free acquisition regimes; S22 seed-0 v5 TTA |
| **Code** | `scripts/run_regime_rendering.py` (freeze · run); tests `tests/unit/test_regime_rendering.py` |
| **Plan** | `configs/research/s35_regime_rendering.json`, SHA-256 `37da5f94c73eadac22e79dc9f5ec19599a489440c43b0b14ed6ab3c6814ebd27`. Frozen before S33 was scored |
| **Registers** | F125 · D57 · H55–H56 |

## 1 · Question and mechanism

Session attraction (F113) is an asymmetric comparison. A sharp test kernel resembles every
variety trained on sharp images *because of the camera*, not the variety. S33's global rendering
softens every comparison. That removes the asymmetry, but it also discards the fine texture that
same-regime comparisons depend on (F115). CCAR scores each candidate variety on the test kernel as
it would look under *that variety's* training acquisition:

```text
logit_c(x) = z_c(render_σ=0.5(x))   if regime(scan(x)) = sharp and regime(c) = soft
           = z_c(x)                 otherwise
```

- `regime(scan)` is S31's label-free 2-means of plate-background colour and kernel sharpness. All
  33 session-8 scans and 8 others are `soft`; session ids are never read.
- `regime(c)` is the majority regime of c's outer-training rows.
- Only sharp test kernels scored against soft-trained varieties change. This is exactly the
  from-session-8 direction. Same-regime comparisons are identical to the unrendered branch by
  construction, so the F1 cost should be ≈ 0.

No network is trained. CCAR is a scoring rule over two forward passes.

## 2 · Results (held-out, both corrected folds; CPU, seconds)

The rule re-renders 22.8% (fold 0, 25 soft-regime varieties) and 12.8% (fold 1, 16) of the
held-out kernel × variety scores. Calib temperatures stayed at 0.5.

| Contrast | ΔF1 [CI] | Δcross [CI] | Δ from-s8 [cell CI] |
|---|---:|---:|---:|
| **H55** CCAR on the S33 branch (RGB) | +.0001 [−.0011, .0012] | −.0012 [−.0043, .0018] | −.0025 [−.0086, .0037] |
| **H56** same, after equal fusion | +.0001 [−.0008, .0010] | +.0018 [−.0012, .0055] | +.0037 [−.0037, .0110] |
| CCAR on the S32 branch (RGB) | −.0004 [−.0019, .0010] | +.0006 [−.0055, .0061] | +.0012 [−.0110, .0123] |
| CCAR on the S32 branch (system) | +.0009 [−.0007, .0026] | +.0043 [−.0019, .0104] | +.0086 [−.0037, .0208] |

## 3 · Finding and decision

- **F125**: rendering each candidate variety's RGB evidence into that variety's training
  acquisition regime leaves predictions and session attraction unchanged (±.003). The sharp/soft
  asymmetry is not the mechanism of session attraction for the trained branch. Together with
  F124, the scan-systematic cross-session error (F122) is not explained by the measured RGB
  optics. Lot/biological differences, or acquisition factors not measured here, remain. Only
  crossed acquisitions can separate them (FW-42). [E3]
- **[D57](../../DECISIONS.md)**: CCAR is not adopted. It was the candidate novel mechanism of this
  phase, and it is recorded as falsified (prior-art log:
  [S36 search log](../../evidence/S36_next_generation_architecture/search_log.md)).

## 4 · Reproduce

```sh
PYTHONPATH=src:scripts python scripts/run_regime_rendering.py freeze   # before S33 scoring
PYTHONPATH=src:scripts python scripts/run_regime_rendering.py run      # after S33 render
```
