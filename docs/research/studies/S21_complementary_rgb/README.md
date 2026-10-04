# S21 · RGB/HSI screen on complementary acquisition folds

2026-10-04 · **Complete: 24 CPU arms × two folds. No new GPU training.**
[S20](../S20_rgb_pathway/README.md) motivated this separately frozen follow-up;
[S22](../S22_complementary_v5/README.md) is the prepared neural continuation.

The repaired protocol confirms useful RGB appearance and simple fusion gains over
the HSI linear baseline. It also confirms that more aggregate F1 does not ensure
more transfer: full-band probes and feature concatenation lose cross-session recall,
and correct individual pairing gives no advantage over the scan-level shuffle.

## Design and provenance

The [frozen runnable plan](../../evidence/S21_complementary_rgb/preregistration.json)
has SHA-256 `e4d1e4a3d469f98c2c4c7e69b023a9835d56a12cda4159c108d4ea2504bf63ca`.
S20's 24 non-v5 arms, feature assets, classifier, calibration grid, fusion rules and
class-bootstrap estimand were fixed before S21 scoring. Historical v5 networks
were excluded because their training acquisitions differ from these partitions.

Independent keyed RNG streams choose class group order and patch carve. Exactly two
class-pure groups are required. Fold sizes are **4,313 and 4,311**; all **8,624 rows
are held out exactly once**, including both directions of all 17 cross-session
varieties. Group order is stable when calibration fraction or other class support
changes. Production training defaults and old splitters were not silently changed.
This is an S20-informed follow-up using the same acquisitions, not fresh independent
replication or an untouched generalization test.

## Quantitative results

| Selected arm | Macro-F1 | Same-session recall | Cross-session recall |
|---|---:|---:|---:|
| RGB hand descriptors | .351358 | .453699 | .015972 |
| DINO silhouette | .168546 | .183892 | .115127 |
| DINO grayscale | .400956 | .470844 | .138343 |
| DINO RGB at 32px | .323255 | .364629 | .156113 |
| DINO RGB | .450520 | .531532 | .142046 |
| HSI k32 quantiles + shape | .534850 | .672893 | .077941 |
| HSI strict full214 quantiles + shape | .552844 | .709531 | .046569 |
| Equal RGB + HSI k32 | .591820 | .727895 | .127858 |
| Equal RGB + HSI full214 | .603119 | .748319 | .112539 |
| Concatenated RGB/HSI features | .613476 | .766432 | .082149 |
| Within-scan shuffled RGB + HSI k32 | .605932 | .748428 | .128389 |

[All 24 arms, intervals and paired contrasts](results.md).
Calibration again selected equal weights in both folds. Points are mean fold macro
metrics, not metrics of pooled out-of-fold predictions.

![RGB evidence](../../figures/S21_complementary_rgb/rgb_and_fusion.png)

- **H36 passes:** RGB versus silhouette F1 +.281974, 95% paired class interval
  [.244211, .316018]. Appearance carries information beyond shape.
- **H37 passes:** equal RGB/HSI32 fusion versus HSI32 F1 +.056970
  [.039397, .075737]; cross recall +.049917 [.003676, .111698]. This supports
  complementarity with this weak HSI probe, not superiority to retrained v5.
- **H38 fails:** matched minus shuffled fusion F1 −.014112 [−.020092, −.007799].
  No individual-kernel coupling advantage. Scan/variety evidence is sufficient to
  explain these simple fusion gains; a shuffle using class-pure scans is diagnostic.
- **H39a/b fail:** nested64 improves F1 by .010254 but cross recall falls .010498
  [−.019077, −.003145]. Full214 mean improves F1 by about .075 but cross recall
  falls .044649 [−.078192, −.011642]. These probes do not justify spectral expansion.

RGB versus grayscale F1 +.049564 and versus 32px +.127265, but neither contrast
establishes positive cross-session gain. Concatenation versus equal fusion F1
+.021656, cross recall −.045709 [−.099390, −.000038]. Selecting solely by aggregate
F1 would choose a less transferable representation under this observed shift.

## Revised modelling direction

Keep k32 HSI, frozen RGB and equal probability fusion as independently inspectable
candidates. Rebaseline v5 on these exact partitions before choosing an upgraded
multimodal model. Do not add cross-attention, gating or broad fine-tuning merely
because S19 proposed them. RGB-only is weaker here; full spectra are retained as
research assets, not adopted as the new default. The next decisive evidence is
[S22's six GPU fits](../S22_complementary_v5/README.md), followed by independent
crossed acquisitions for any broad session/lot claim.

The 73 same-session and 17 cross-session classes measure different support regimes;
all bridge pairs still touch session 8. Class intervals are not session-population
intervals, no multiplicity correction is claimed, and historical test reuse remains.
Numerical warnings were audited as in S20: coefficients/probabilities are finite and
normalized, with identical train/calib explicit-sum checks; no cause is asserted.

## Reproduce without new selection

```sh
# Historical execution commands; existing plans/results refuse overwrite.
PYTHONPATH=src python scripts/run_rgb_complementary.py freeze
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python scripts/run_rgb_complementary.py run
# Safe: saved-result analyses and evidence validation, no retraining.
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python docs/research/evidence/S21_complementary_rgb/code/analyze_results.py
python docs/research/evidence/S21_complementary_rgb/code/draw_figures.py
PYTHONPATH=src python docs/research/evidence/S21_complementary_rgb/code/validate_study.py --assets
```

Large feature/probability caches remain under ignored `outputs/`; compact predictions,
per-variety metrics, hypotheses, completion/input hashes, numerical checks and figures
are tracked. Use [S20 reproduction](../S20_rgb_pathway/reproduce.md) for the shared
asset/extractor contract. Never relabel an engineering replay as independent evidence.
