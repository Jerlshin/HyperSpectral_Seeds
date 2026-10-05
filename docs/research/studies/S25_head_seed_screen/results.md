# S25 · Selected-head seed sensitivity on fixed HSI encoders

2026-10-05 · **Complete; H43-screen passes.** Four additional cheap head fits,
no new encoder fit. The original S23 seed-0 outcomes are reused unchanged.

Plan SHA256: `b45f08b1931ca41b1131974968ef8eba6c44a7bde08ce3381b40e684dc86c566`.
This is an explicit allocation extension for the selected candidate after its
positive matched gain and S24's rejected simplifications. Cached features make
the complete extra seed-1/2 run take **9.76 s**. The other intermediate candidates
are not expanded. HSI remains at encoder seed 0 on both corrected folds.

| Head seed | Fold 0 F1 | Fold 1 F1 | Mean F1 | Cross recall | F1 gain over equal-single | F1 gain over equal TTA |
|---|---:|---:|---:|---:|---:|---:|
| 0, reused S23 | .607767 | .600857 | .604312 | .233252 | .010780 | .002717 |
| 1 | .605278 | .602904 | .604091 | .235090 | .010559 | .002496 |
| 2 | .606330 | .601187 | .603758 | .236900 | .010227 | .002164 |

Mean head F1 **.604054**, same recall .714788, cross recall **.235081**.
Paired F1 gain versus equal-single **+.010522 [.001851,.018941]**; cross gain
+.031105 [−.001381,.068210]. The descriptive SD of the three head-seed F1
gains is **.000279**. H43 passes: positive fold-mean/individual-seed gains,
≥.01 mean effect, positive class interval and nonnegative mean cross delta.
Every head seed meets TTA's two-fold mean F1 and cross point estimates.

The practical advantage remains small: versus equal TTA, mean F1 gain
**+.002459 [−.006634,.011215]**, cross **+.025590 [−.006216,.060245]**.
The head is stable across these tested initializations, but superior practical
performance and transfer improvement remain unestablished. Direction recall
averaged over head seeds: toward session 8 .267947; away .202215.

All original seed-0 learned-head and equal-anchor predictions replay **exactly**
from S24's frozen cache, with zero disagreements on either fold. Every head seed
holds all 8,624 kernels out once. The new heads select on calibration before
loading the held-out cache. Saved rows/metrics, class contributions, curves,
selections, replay audit, paired intervals, direction analysis and hash manifests
are in `evidence/S25_head_seed_screen/screen_results/`; full head states remain
under `outputs/s25_head_seed_screen/`.

This is **head-only initialization sensitivity**. HSI encoder variance is still
unmeasured on corrected folds, and all acquisitions are reused. Variety intervals
retain these fixed head seeds; they do not estimate encoder or new-session variance.
Final-system confirmation must vary encoder and head initialization together.
The rejected S24 simplifications receive no automatic seed expansion.

Retain the anchored additive S23 head as the provisional learned system. Its small
TTA margin motivated [S26's fixed TTA-anchor check](../S26_tta_anchor/results.md),
which subsequently failed both calibration folds without new test scores. The next
proposed step is [S27](../S27_tta_trained_head/README.md): train against the intended
TTA anchor before expensive encoder replication. Eventual confirmation follows the shared-
encoder queue, with new crossed acquisitions required for broad transfer claims.
