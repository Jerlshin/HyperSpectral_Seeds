# S34 · Are the current multimodal baselines still competitive with a strong RGB branch?

| | |
|---|---|
| **Status** | complete — **H54 passes**: the strong-RGB fixed fusion supersedes the S29 learned system |
| **Dates** | 2026-10-05 |
| **Data** | S22 seed-0 v5 TTA; S32 calib-selected trained RGB branch; S31 frozen ViT-L component (`cls`, because H48 failed); S29 incumbent predictions |
| **Code** | `scripts/run_multimodal_reassessment.py` (freeze · run) |
| **Plan** | `configs/research/s34_multimodal_reassessment.json`, SHA-256 `cb8063a7848abf9b237175d1d8b6922c8d99f3763431c95600f7d26bc7455357`. Frozen before S32 was scored; components resolved from sealed outputs at run time |
| **Registers** | F119–F121, F123 · D55–D56 · H54 |

## 1 · Question and design

The owner's directive treats four systems as baselines: HSI v5, frozen ViT-L RGB, fixed RGB–HSI
fusion and the S29 learned fusion. With a trained RGB branch, which of them remain competitive?
What do the error structure and acquisition content of each branch imply for the next architecture?
- **Fixed fusions**, weights over (HSI, trained RGB, frozen RGB):
  - `equal_trained` (.5, .5, 0);
  - `equal_frozen` (.5, 0, .5);
  - `tri` (.5, .25, .25);
  - `rgb_ensemble` (0, .5, .5).
- **Candidate:** the better of `equal_trained` and `tri` on two-fold calib F1, written before scoring.
- **H54-screen:** candidate − S29 `head_l` (the incumbent learned system, .6272): F1 ≥ .01, CI > 0,
  both folds positive, cross Δ ≥ 0.
- **Kernel-pairing control:** `equal_trained_shuffled` permutes the trained-RGB probabilities among
  kernels of the same held-out scan. S20/S21 found matched pairing *worse* than shuffled for frozen
  features (H38, −.014).
- **Complementarity:** either-modality oracle, both-wrong fractions and error φ per pair.
- **Acquisition decodability:**
  - method: a train-only shrinkage LDA predicts the *session* from each representation and is
    scored on held-out bridge kernels;
  - why bridge kernels: their session differs from every training session of their own variety,
    so class identity cannot supply the answer;
  - representations: v5 embedding, trained RGB embedding, frozen ViT-L and ViT-S class tokens.

## 2 · Results (held-out, both corrected folds; CPU, minutes)

Resolved components: trained RGB = S32 `vitb`; frozen RGB = `cls` (S31 H48 failed). Calib chose
`equal_trained` (.8511) over `tri` (.8408).

| Arm | F1 | Same / cross | From / to session 8 | Attraction |
|---|---:|---:|---:|---:|
| HSI v5 TTA | .5591 | .6688 / .2086 | .178 / .240 | .549 |
| RGB trained | .6642 | .7672 / .2812 | .239 / .323 | .479 |
| RGB frozen (`cls`) | .4950 | .5746 / .1758 | .104 / .247 | .455 |
| S29 head ViT-L (incumbent) | .6272 | .7408 / .2480 | .180 / .316 | .580 |
| **`equal_trained` (candidate)** | **.6977** | **.8108 / .2953** | **.254 / .337** | .529 |
| `tri` | .6782 | .7951 / .2708 | .216 / .326 | .566 |
| `rgb_ensemble` | .6530 | .7629 / .2544 | .201 / .308 | .493 |
| `equal_trained_shuffled` (control) | .7202 | .8389 / .3023 | .233 / .372 | .525 |

**H54-screen: pass.** `equal_trained` − S29 `head_l`: **+.0706 [.0556, .0847]**, folds
+.0691 / +.0720, cross **+.0473 [.0008, .0938]**.

| Descriptive contrast | ΔF1 [CI] | Δcross [CI] |
|---|---:|---:|
| `tri` − `equal_trained` | −.0195 [−.0277, −.0111] | −.0245 [−.0517, .0023] |
| `rgb_ensemble` − RGB trained | −.0112 [−.0181, −.0049] | −.0268 [−.0490, −.0037] |
| matched − within-scan shuffled pairing | −.0225 [−.0284, −.0168] | −.0070 [−.0224, .0071] |
| `equal_trained` − HSI alone | +.1386 [.1154, .1612] | +.0867 [.0213, .1543] |
| `equal_trained` − S29 equal ViT-L | +.0773 [.0646, .0898] | +.0619 [.0221, .0999] |

**Complementarity:**

| Pair | Either-oracle (f0 / f1) | Both wrong: same / cross (f0 / f1) | Error φ (f0 / f1) |
|---|---:|---:|---:|
| HSI + RGB trained | .774 / .781 | .128 / .648 · .141 / .558 | .388 / .364 |
| HSI + RGB frozen | .708 / .716 | .192 / .724 · .204 / .630 | .324 / .314 |
| RGB trained + frozen | .720 / .731 | .182 / .703 · .182 / .645 | .486 / .474 |

**Acquisition decodability.** A train-only LDA predicts session; accuracy is on the 815 / 813
held-out bridge kernels, with chance well below .2:

| Representation | Test-session accuracy (f0 / f1) | Predicts own class's training session (f0 / f1) |
|---|---:|---:|
| HSI v5 embedding (256-D) | .589 / .604 | .114 / .145 |
| RGB trained embedding (2,048-D) | .495 / .544 | .021 / .100 |
| RGB frozen ViT-L class token | .420 / .556 | .037 / .070 |
| RGB frozen ViT-S class token | .450 / .586 | .025 / .057 |

## 3 · Findings and decisions

- **F119:** the strong-RGB fixed fusion supersedes every earlier system, including the S29 learned one.
- **F120:** adding frozen RGB to the trained branch hurts; training, not capacity, is the lever.
- **F121:** kernel-matched pairing is worse than within-scan shuffled pairing, so no kernel-level
  cross-modal synergy is visible.
- **F123:** the HSI representation carries at least as much acquisition signal as the RGB branches.
- **[D55](../../DECISIONS.md):** new baselines.
- **[D56](../../DECISIONS.md):** learned, stacked, complementary and kernel-interaction fusion are
  not identifiable on this design. Each variety has one training scan per fold, and both branches
  interpolate their training rows (F123).

## 4 · Threats

- One seed per branch.
- The shuffled-pairing control uses other kernels of the *same held-out scan*. It is a diagnostic
  of error correlation, not a deployable kernel-level rule; lot-level pooling is a separate
  inference regime (F122).
- Decodability depends on the probe and representation dimension (2,048-D RGB vs 256-D HSI).
  Compare directions, not small differences.

## 5 · Reproduce

```sh
PYTHONPATH=src:scripts python scripts/run_multimodal_reassessment.py freeze   # before S32 scoring
PYTHONPATH=src:scripts python scripts/run_multimodal_reassessment.py run      # after S31 and S32 are sealed
PYTHONPATH=src python docs/research/evidence/S27_tta_trained_head/code/archive_screen.py \
  outputs/s34_multimodal_reassessment configs/research/s34_multimodal_reassessment.json \
  docs/research/evidence/S34_multimodal_reassessment/screen_results
```
