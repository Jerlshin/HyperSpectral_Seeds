# S36 · Next-generation RGB–HSI architecture, derived from first principles and S31–S38 evidence

| | |
|---|---|
| **Status** | design study complete; RGB-branch readout decided by S37 (§4); validation = S39 (§7) |
| **Dates** | 2026-10-05 |
| **Inputs** | F114–F127; S19 prior-art review; targeted search log (`evidence/S36_next_generation_architecture/search_log.md`) |
| **Code** | `src/spectralquadnet/models/{rgb_branch,rgb_multilayer,acquisition_aware}.py`; tests `tests/unit/test_{rgb_branch,rgb_multilayer,acquisition_aware}.py`; diagnostic `evidence/S36_next_generation_architecture/code/error_decomposition.py` |
| **Registers** | F122 · D58 |

## 1 · Question

The owner asked for a genuinely novel, high-performing RGB–HSI model, not "two encoders +
concatenation". Every mechanism had to be tied to observed behaviour. What do the data support,
which candidate mechanisms did they falsify, and what must the next experiment establish?

## 2 · What the evidence says the system must do

| # | Observation (finding) | Requirement it imposes |
|---|---|---|
| 1 | Training an encoder on kernels is the only lever that moved transfer, in either modality (F118, F126, F127) | **Both branches are trained encoders.** Frozen foundation features, however read out, are not enough |
| 2 | Kernels fill 45–133 of 256 patches; foreground-token fine-tuning works; partial ViT-L ≈ ViT-B (F120) | **Object-centric RGB tokens** (class + 128 most-foreground patches); ViT-B is enough under local compute |
| 3 | Intermediate blocks carry within-acquisition RGB signal (F115); metric morphometrics are the one RGB cue that transfers (F117) | Readout candidate: multi-layer + metric morphometrics (S37 decides) |
| 4 | Matched kernel pairing is *worse* than within-scan shuffled pairing, with frozen and trained branches alike (F121) | **No kernel-level cross-modal interaction** (no cross-attention, no token fusion) |
| 5 | Both branches interpolate training rows; each variety has one training scan per fold (F123, D56) | **No learned fusion.** Calibrated, fixed evidence fusion; learned/stacked/complementary fusion is unidentifiable here |
| 6 | Colour statistics and hand spectra encode the session (F116, F126); HSI encodes more acquisition than trained RGB (F123) | No acquisition-conditioned gates. Inputs that transfer badly are not routed in separately |
| 7 | Measured RGB optics do not explain cross-session failure (F124, F125) | No optics augmentation or rendering |
| 8 | Same-session errors are kernel-random (scan accuracy → 1.00); cross-session errors are scan-systematic (≈ .41 / .59) (F122) | Report a **kernel/scan error decomposition**; offer **lot-level pooling** as an explicit, separately reported inference mode |

## 3 · Proposed system: SeedNet-MX (multimodal, evidence-level)

```text
RGB crop (masked, 224) ──► foreground-token DINOv2 ViT-B, fine-tuned ──► readout ──► p_RGB  (4-view TTA, calib T)
                                   [class + 128 fg patches]            (S37)          │
                                                                                    ├──► ½ p_RGB + ½ p_HSI ──► kernel decision
HSI patch (k32 reflectance) ──► SeedNet v5, trained ──────────────────────────► p_HSI  (R1 TTA, calib T)
                                                                                    └──► [lot mode] mean log-p over a lot's kernels
```

Implementation:
- `models/rgb_branch.py` (S32 branch) / `models/rgb_multilayer.py` (S37 readout);
- `models/acquisition_aware.py`: `AcquisitionAwareFusion` with CCAR switched **off** per D57, and `pool_lots`;
- HSI v5 is unchanged.

Development-screen performance (seed 0, both corrected folds; S32/S34):

| | F1 | Same / cross recall | From / to session 8 |
|---|---:|---:|---:|
| SeedNet-MX (S32 readout) | **.6977** | .8108 / .2953 | .254 / .337 |
| Incumbent S29 learned system | .6272 | .7408 / .2480 | .180 / .316 |
| HSI v5 alone | .5591 | .6688 / .2086 | .178 / .240 |
| Lot mode, all kernels of a scan (descriptive) | scan acc. 1.00 same-session | — | .41 / .59 scans |

## 4 · RGB readout (S37)

<!-- S37 outcome inserted when scored -->

## 5 · Novelty: what is claimed, and what is not

**Not claimed:**
- RGB + HSI late fusion on this dataset (FusedNet; S19 L06);
- foreground masking, multi-view TTA, layer-wise decay fine-tuning, or DINOv2 multi-layer readout,
  each known on its own.

**Claimed, with evidence and scope (development screens, reused acquisitions, 1 seed):**
1. **A trained object-centric RGB branch generalizes across sessions where frozen foundation
   features do not.** It moves away-from-session-8 recall from ≤ .104 to .239, beating HSI, which no
   readout, backbone size or fusion form achieved (F114, F118). To our knowledge (search log), no
   prior seed study separates trained from frozen foundation features under a session-grouped test.
2. **An identifiability argument for evidence-level fusion,** with falsification controls:
   - one training scan per variety ⇒ learned, stacked and complementary fusion cannot be trained
     honestly (D56);
   - the within-scan shuffle control shows kernel-level coupling has no signal (F121).
   The architecture is *deliberately* independent up to evidence, and the paper reports the controls
   that justify it.
3. **A kernel/scan error decomposition** separates kernel-random from scan-systematic error (F122).
   It shows that what remains after SeedNet-MX is acquisition- or lot-systematic, i.e. a data
   problem, not an architecture problem.

**Candidate novel mechanisms tested and falsified in this phase**, kept on record so they are not
re-proposed:
- class-conditional acquisition rendering (S35);
- measured-nuisance optics augmentation (S33);
- modality-role specialization (S38);
- readout-only frozen transfer (S31, cross clause).

## 6 · The mechanism the evidence points to next, and why it cannot be tested on these scans

The residual is scan-systematic and appears in both modalities. The modalities carry *different*
acquisition imprints (F123: RGB moved from-s8 while HSI did not; error φ ≈ .37). The natural next
mechanism is **cross-acquisition, cross-modal consistency learning**:
- train both encoders so that kernels of one variety imaged in *different* sessions/lots yield the
  same class evidence;
- use each modality's prediction on one acquisition as a consistency target for the other
  modality on another acquisition (co-training across acquisitions).

It needs ≥ 2 training acquisitions per variety. On the current folds each variety has exactly one,
so the objective has no signal (D56). This makes the crossed-session/lot acquisition (FW-42) a
prerequisite for the architecture's novel component, not only for its evaluation.

## 7 · Next experiment to validate the proposed architecture (S39, brief)

**Matched final confirmation of SeedNet-MX**, replacing S30 (whose contrasts concern retired
components, D55):
- **RGB:** the S37-selected readout at seeds 1, 2 × folds 0, 1. That is 4 local MPS fits, about
  3.3 h, or GPU.
- **HSI:** v5 at seeds 1, 2 × folds 0, 1. That is 4 GPU fits (≈ 95 min on 2×T4 per the S30
  brief) and needs Kaggle authorization. Each fit exports single-view embeddings and train-row TTA,
  so the S29 learned-system control can be refit per seed.
- **Contrasts** (frozen before any new fit is scored; means over seeds 0/1/2 × 2 folds; G3 Δ > 2 ×
  seed SD):
  - **M1** MX − S29 system (≥ .02);
  - **M2** MX − HSI v5 (≥ .05);
  - **M3** trained RGB − frozen ViT-L RGB (≥ .05);
  - **M4** MX − frozen-ViT-L equal fusion (≥ .02);
  - transfer reported per direction.
- **Then:** apply the locked system to the crossed acquisition pilot (FW-42) as the only test of
  the scan-systematic residual.

## 8 · Threats

- All S31–S38 numbers are seed-0 development screens on acquisitions reused since S20. The
  variety intervals do not include seed or new-session uncertainty.
- Partial ViT-L tuning only.
- The calib split is within-session and cannot select for transfer.
