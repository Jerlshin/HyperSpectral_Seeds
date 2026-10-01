# Glossary

The project's vocabulary. Several terms have a precise meaning here that differs from casual use;
when a result is ambiguous, it is usually because one of these was used loosely.

## Data units

| Term | Meaning |
|---|---|
| **Kernel / patch** | One segmented rice seed, cut to 64 × 64 pixels × all bands. 8,624 in total (of 8,640 imaged; 16 lost to the shape gate). |
| **Variety / class** | One of 90 rice varieties. 91–96 kernels each. |
| **Bundle** (also *scan*, *group*) | One tray of 48 kernels of a single variety, imaged in one pushbroom scan. Two per variety → 180 bundles. `groups.npy` holds the bundle ID of every patch. |
| **Session** | One imaging session (a date and run number, e.g. `Data-VIS-20170111-2`). 9 sessions. A session shares lamp, detector state and room conditions across all bundles imaged in it. |
| **Same-session variety** | Both bundles imaged in one session. 73 varieties. |
| **Cross-session variety** | Bundles imaged in two different sessions. 17 varieties: labels 4, 7, 12, 22, 38, 45, 59, 60, 61, 63, 64, 65, 66, 70, 71, 78, 79. |
| **Band axis** | The list of wavelengths a cube's band dimension addresses. **SNV-256**: 256 bands, 383.2–1006.5 nm, the former cube. **refl-215**: 215 bands, 383.2–605.6 and 708.2–1006.5 nm, the current `./dataset`. |

## Splits and protocols

| Term | Meaning |
|---|---|
| **stratified** | Patch-level split. Every bundle appears in train and eval. Leaky (F02); kept only as the A1 contrast arm. |
| **grouped** | Leave-one-bundle-out: per class, one bundle trains, the other is held out. Two folds (`split_fold` 0/1) swap them. The primary protocol (D01). |
| **calib** | 15 % carved from the training pool. Under grouped it is from the *same bundle* as train, so it is optimistic (F21). Used to decide, never to report. |
| **held-out** | `val ∪ test` — the other bundle. Scored once per question. |
| **Fold** | Which of a variety's two bundles is held out. There are only two. |
| **Leakage gap** | `F1_stratified − F1_grouped` for the same model — the Q1 headline (ablation A1). |

## Radiometry

| Term | Meaning |
|---|---|
| **SNV** | Standard Normal Variate per pixel across wavelength: subtract the pixel's mean, divide by its sd. Removes a scalar gain; keeps spectral *shape* — including the lamp's. |
| **Gain statistics (gstat)** | Per-kernel statistics of what SNV removed: log μ, log sd, −μ/sd. `gstat1` = −μ/sd only; `gstat3` = all three. Strongly session-informative (F26). |
| **White tile** | The Spectralon reference panel imaged in every scene below the seed tray. Dividing by it gives reflectance. |
| **Reflectance cube** | The current dataset: each scene divided by its own tile, 215 bands (D12). |

## Band selection

| Term | Meaning |
|---|---|
| **Budget (k)** | Number of bands kept. |
| **uniform** | k bands evenly spaced in index over the whole axis. A label-free null. |
| **uniform430** | k bands evenly spaced in *wavelength* from the first band ≥ 430 nm to the last; on refl-215 the dropped 608–706 nm gap is collapsed first. The shipped finalist rule (D11). |
| **random** | k bands drawn at random; the other null. |
| **glw / glw430** | Greedy forward selection maximising inner-CV LDA log-likelihood (S05), over all bands / bands ≥ 430 nm. |
| **mRMR, SPA, …** | The 12 band-study methods; see `docs/07` §2.4. |
| **Proxy** | A cheap stand-in model used to compare band sets: LDA, LinearSVC, ExtraTrees on mean spectra (S03); a small 2-D CNN on a 16 × 16 pooled cube (S05). Not the network. |
| **Plateau / knee** | Smallest k within 0.01 of the curve's own peak; *demonstrable* only if the curve extends beyond it. |
| **Null margin** | A method's advantage over `random` at the same k. A method that never clears 0.01 is "subsetting, not selecting". |
| **Session F-ratio** | Per band: between-session variance of class means over within-session variance, on training rows. ~1 under no session effect; > 2.2 is p ≈ 0.05. |
| **Session attraction** | Share of cross-session kernels predicted as a class whose training bundle came from the kernel's own session. Reported beside its chance rate. |

## Models

| Term | Meaning |
|---|---|
| **SpectralQuadNet** | The original four-branch model (A spectral profile, B index bank, C spatial 3-D CNN, D SpecFormer), 5.2 M parameters. Kept as the control arm. |
| **SpectralSeedNet** | The primary model since S02: spatial 3-D-stem CNN + spectral MLP over the masked mean spectrum, concatenated, K = 1 ArcFace head. 3.0 M parameters at 215 bands. |
| **TTA** | Test-time augmentation: 8 dihedral spatial views + 4 spectral-gain views, logits averaged. Always reported separately. |

## Evidence

| Term | Meaning |
|---|---|
| **E0–E4** | Evidence strength; see README §4. |
| **Pre-registration** | Hypotheses, arms and decision rules frozen to a hashed JSON before held-out data is touched (D09). |
| **Deviation** | What shipped differs from a frozen rule or plan; always recorded in `DECISIONS.md`. |
