# S20 implementation and data contract

2026-10-04. Dataset version `dataset_rgb_hsi_v3`; legacy assets remain untouched.

## Identity and masks

The archive's MD5 matches Zenodo `aeb8e4b9bfbf80550d368c13c766d761`; SHA-256 is
`92af3258ba72950e301ee7f4030f59cc88922e21f34d00a06d68debb6b046c14`. All 584 members
passed CRC validation. The [dataset record](https://zenodo.org/records/3241923)
identifies the 8×6 paired acquisition matrix and separate physical bundles.

The new ENVI reader validates dimensions, payload length, interleave, endian and
exact sibling name. It avoids the old prefix lookup's potential JPEG ambiguity.
JPEG EXIF tags vary; decoding always ignores them and preserves encoded sensor
coordinates. Ordinary auto-oriented decoding silently rotates some scans.

At half RGB resolution, scan-local Otsu thresholding inside the fixed plate ROI
produces candidate components. Fixed area/border/eccentricity checks remove paper
and hardware. Short components additionally need solidity; elongated grains with
awns remain eligible. Exactly 48 components are required. Native masks use two deterministic GrabCut iterations in an expanded local box.
The coarse component is certain foreground; a spatially restricted 0.70×Otsu
proposal initializes probable foreground, while surrounding pixels initialize
background. Outside the spatial support is certain background (v3). Component overlap selects the same seed rather than a neighboring
object. Merely lowering the threshold was rejected because it admitted plate
background; GrabCut with an unrestricted proposal also admitted background. All nonforeground crop
pixels are zero. Crops retain aspect ratio, have 5% margins on each side, and resize
to 224×224. Native scale and eight morphology values are separate explicit features.
No label text, plate position, scan ID or session ID is a predictor feature.

Grid row/column assignment uses deterministic one-dimensional clustering, unique
cell constraints and a nearest-center margin. The eight-by-six order is verified
with a centroid homography and all row/column reversal alternatives. A scan fails
if the identity orientation is not best by a ratio of at least 1.5 or a pair residual
exceeds .22 row pitch. This is object correspondence, not pixel registration.
The archived acquisition record supports bundle-disjoint physical identities;
image hashes cannot independently establish biological lot identity or rule out
unrecorded reimaging.

HSI segmentation is the original radiance algorithm, including historical
exclusions. Every retained component must match saved morphology, float16 masks,
and **every float16 k32 patch value exactly**. IDs are `scan_key/rXXcYY`; the
legacy array index is recorded separately. Missing cells do not shift any ID.
Native masks, per-scan atomic shards, overlays, source hashes and the final
completion manifest are retained. Partial datasets cannot enter the probe runner.

## Deliberate S19 plan revisions

1. Materialize full215 **compact measurements**, reconstructing each exact 64×64
   patch transiently and checking k32 parity. Retain mean, SD, q10/q50/q90 and 16
   occupied-region mean spectra. Avoid a redundant ~14 GiB float16 dense cube; the
   optional `--dense` builder is available if a later model actually needs it
   (assembly requires approximately twice the final disk footprint).
2. Use locally cached **DINOv2-S/14** for the first CPU transfer probe, not the
   proposed DINOv3/ConvNeXt pair. Its measured local throughput makes four controls
   feasible without new downloads or GPU claims. This is not a ranking of foundation
   models. DINOv3/ConvNeXt are deferred until the mechanism/result warrants them.
3. Keep historical 215/195/64 references, but add a **214-band own-white-only**
   reference. The wavelength 605.583333 nm is omitted from every scan in that arm:
   its three session-pooled white values cannot quietly enter a strict inductive
   comparison. Per-test-scan standards are allowed instrument calibration, not
   classifier fitting. Wavelength validity itself was audited across the archive;
   no labels selected that axis.
4. Preserve v5/S17 defaults and all frozen historical designs. The S15 digest guard
   intentionally refuses the expanded source tree; replay old runs from their pinned
   checkout. No historical guard is weakened to make new-source tests green.

## Rejected preprocessing v1

The first all-scan identity build passed counts and exact HSI parity, but the visual
review of worst residuals found truncated dark tips. Its masks used the detection
threshold for extraction. S20 rejected that version **before any classifier scoring**,
separated coarse detection from native mask refinement, and rebuilt crops/features.
The v1 manifests, source snapshot and QC figures remain under `rejected_v1` and
`code/preparation_execution`. Mask thresholds were developed label-free on observed
images, not on predictive results. Residual masks are reviewed, not assumed perfect;
a pixel-level segmentation ground-truth dataset is still absent.

## Features and evaluation

`hsi_summary.npy`: N×5×215 float32 in mean/SD/q10/q50/q90 order, measured on the
exact resized foreground before float16 storage. `hsi_regions.npy`: N×16×215
float32 region means; `occupancy.npy`: N×16 foreground fraction. Empty tokens are
zero with zero occupancy. No PCA, normalization or band selection is fitted here.
`rgb.npy`: N×224×224×3 uint8. `rgb_descriptors.npy`: 15 Lab color statistics plus
8 grayscale texture statistics; `rgb_morphology.npy`: original-pixel morphology.
The texture core is eroded so artificial black boundaries do not create texture.

Frozen DINOv2 uses its 384-D class token, eval mode, float32, ImageNet normalization,
no fine-tuning/TTA, and exactly the same crop geometry for RGB, grayscale, RGB
area-downsampled to32 then restored to224, and binary silhouette. The 32px control
is a resolution intervention, not a camera-matched HSI optical simulation.
The [model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
states Apache-2.0 and LVD-142M pretraining; rice-source overlap is not independently
auditable. Exact checkpoint and cached Python source hashes accompany every cache.

The separately hashed predictive manifest defines every arm, every split index,
source/input hashes, temperatures, fusion weights and decision thresholds. The
runner verifies them before fitting/scoring and refuses to overwrite results.
Shrinkage LDA and scaling fit train only; temperatures and the optional fusion
weight fit within-training-scan calib only. No refit on calib. Both grouped folds
are reported, not their maximum. There is no new GPU training or new acquisition.

Class-bootstrap intervals retain both selected folds of a variety; S20 does not guarantee both acquisition directions (see the post-freeze coverage correction in README.md). F1 intervals
resample fixed per-class F1 contributions, keeping original precision denominators;
they describe heterogeneity across these varieties, not new-session uncertainty.
The point estimate is mean fold macro-F1, not F1 from pooled out-of-fold predictions.
The deterministic CPU screen has no meaningful initialization-seed replication;
its results do not satisfy the neural replication claim of G3.

A second pre-score audit rejected v2 after one scan-150 crop grew into plate
background. Its manifest, code and worst-residual montage remain in `rejected_v2`.
Version 3 makes pixels outside the bounded native proposal certain background.
The same rule is applied to every scan; no per-class exceptions or predictive
results guided preprocessing. Scans 2, 27 and 150 were reviewed at full crop size
before the final rebuild. v2 feature extraction was interrupted and not scored.

## Engineering validation and historical replay

Thirteen targeted tests cover grid identity with omissions, ambiguous/flipped/moved
objects, EXIF handling, background removal, exact ENVI member/payload checks,
masked spectral statistics, frozen-input mutation, training-only fitting,
calibration/interval arithmetic, exact historical prediction alignment and the four
RGB interventions. Ruff passes for the new production code.

The broad unit run recorded 772 passes, 334 skips, two expected failures and eight
failures. Five reproduce on the starting HEAD in this environment: a Torch 2.14
autocast numerical expectation, two distributed tests blocked by sandbox socket
permissions, and two ANSI-color expectations. Three S15 tests reject the expanded
source digest, as the frozen historical replay guard should. At S20 execution, original training/model
files were unchanged; use their pinned checkout to replay S15. No guard or historical
expectation was relaxed. The broad run preceded the last three added targeted tests.

Final strict mypy passes for all six new production files, as does Ruff. Earlier
intermediate diagnostics are retained beside the final successful log. An OpenCV
stub incorrectly rejects a valid unused `None` rectangle; the documented type
annotation is the only suppression. These checks do not erase the broad-suite
limitations above.

Before freezing or any predictor fitting, the common temperature grid was expanded
to .25–256 (powers of two): shrinkage-LDA scores across very different feature
dimensions need not share neural-logit scale. This is a calibration-only choice
with a fixed grid and no held-out input; boundary selections are reported. Probes
fit/transform in float64 and export scaler/linear coefficients as numeric NPZ files.
The exported inference path is tested against the fitted sklearn pipeline.

## Acquisition metadata and review scope

All 180 RGB headers identify FUJIFILM X-M1, 35-mm focal-length setting. Sessions
0–7 use f/4; all 33 session-8 scans use f/1.6–2.2 and different exposure times.
The saved EXIF table is descriptive only and never enters a predictor. Session 8
is the endpoint of every cross-session bridge, so this is a concrete RGB acquisition
change, not a separately identifiable causal explanation of error.

The final review inspected all three pairing-atlas pages (26 scans), the 12 largest
residual pairs, and all 48 native crops in scans 2, 27 and 150. All 180 scans pass
automated gates. Maximum residual is 1.5851 HSI pixels (2.496% of grid-row pitch);
mean scan RMSE is .28188 pixels. The smallest wrong-orientation RMSE ratio is 8.64.
No retained identity was discarded to improve scores. All 16 omitted HSI cells
retain their documented historical exclusion reason.

The exact v3 preparation source is archived under `code/preparation_v3_execution`.
After execution a single OpenCV-stub type-ignore comment was added; the saved
`preparation_source_annotation.json` verifies identical Python ASTs. No executable
preprocessing behavior changed. Manifests retain the hashes of source actually run.

## Subsequent S21/S22 engineering integration

S21 adds an opt-in complementary splitter and frozen-row reader. S22 adds an opt-in
partition override to the existing training context, validated before fitting or
loader construction. This change was made after S20/S21 CPU scoring and is frozen
in S22; it is not a claim that current source matches historical S15. Historical
model mathematics/config defaults and all frozen guards remain intact. Final checks
and suite limitations are recorded in `../../evidence/S20_rgb_pathway/validation_summary.json`.

Both CPU screens emitted NumPy matrix-product warnings. The stored numeric audit
checks all 40 exported probes per study and every saved probability matrix for
finite values; probabilities normalize correctly. BLAS and explicit sums agree
exactly on training/calibration samples. The underlying warning cause remains
unknown; saved predictions were not changed or refit.
