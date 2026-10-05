# S38 · Should HSI contribute only spectral shape now that RGB is strong? (modality roles)

| | |
|---|---|
| **Status** | complete — **H59 fails**: the trained v5 encoder is the best HSI partner |
| **Dates** | 2026-10-05 |
| **Data** | S32 seed-0 trained ViT-B (4-view); S22 seed-0 v5 TTA; S21 HSI shrinkage-LDA probes (already calibrated); S24 v5 embeddings |
| **Code** | `scripts/run_modality_roles.py` (freeze · run), reusing S34's decodability probe and S33's direction contrast |
| **Plan** | `configs/research/s38_modality_roles.json`, SHA-256 `e5ea35716e45c7ef450eae65008b316946b6ca55db07006506e0bb538f57baf7` |
| **Registers** | F126–F127 · D57 · H59 |

## 1 · Question

Should each modality carry only the information whose transfer is supported? Three findings
motivated the test:
- the v5 HSI representation encodes acquisition more than trained RGB (F123);
- v5's spatial pathway is the session channel (S12), and spread statistics carry the session (F67);
- RGB now supplies spatial/texture strongly (F118).

The hypothesis: HSI should contribute only calibrated spectral *shape* (SNV mean spectrum).
Primary: equal(trained RGB, `hsi_snvmean214_own`) − equal(trained RGB, v5). H59 needs cross Δ ≥ .02
(CI > 0) with F1 Δ ≥ −.01.

## 2 · Results (held-out, both folds; CPU, minutes)

| HSI partner (each equal-fused with trained RGB) | HSI alone F1 / cross | Fused F1 | Fused cross | ΔF1 vs v5 partner [CI] | Δcross [CI] |
|---|---:|---:|---:|---:|---:|
| **v5 TTA (baseline)** | .5591 / .2086 | **.6977** | **.2953** | — | — |
| spectral shape, 214 bands (primary) | .5093 / .1495 | .6798 | .2585 | −.0179 [−.0391, .0006] | **−.0368 [−.0674, −.0061]** |
| spectral shape, 32 bands | .4088 / .1863 | .6781 | .2894 | −.0196 [−.0296, −.0090] | −.0059 [−.0376, .0241] |
| mean spectrum, 214 bands | .5085 / .1162 | .6696 | .2265 | −.0281 [−.0575, −.0045] | −.0687 [−.1078, −.0326] |
| quantiles, 214 bands | .5528 / .0466 | .6797 | .1853 | −.0180 [−.0512, .0106] | −.1099 [−.1704, −.0581] |
| quantiles, 32 bands | .5348 / .0779 | .6834 | .1928 | −.0143 [−.0291, .0001] | −.1025 [−.1505, −.0594] |
| tri: .5 RGB + .25 v5 + .25 shape-214 | — | **.7100** | .2911 | **+.0123 [.0052, .0196]** | −.0042 [−.0190, .0123] |

**Acquisition decodability**, measured as test-session accuracy on 815 / 813 held-out bridge
kernels with a train-only LDA:
- v5 embedding: .589 / .604;
- spectral shape: .712 / .825;
- mean spectrum: .621 / .766;
- quantiles: .863 / .850.

Every hand-crafted spectral representation carries *more* acquisition information than the trained
v5 embedding.

## 3 · Findings and decision

- **F126**: the trained v5 encoder is the best HSI partner for the trained RGB branch. Every
  hand-crafted spectral representation transfers worse in fusion, by up to −.11 cross, and encodes
  the test session more strongly. Role specialization is rejected. [E3]
- **F127** (synthesis, S31–S38): on these acquisitions, transfer gains came only from **training an
  encoder on kernels**, in either modality (RGB fine-tuning +.105 cross; v5 vs hand spectra in fusion
  +.04–.11). They never came from:
  - readouts of frozen features (S31);
  - measured-optics augmentation or rendering (S33, S35);
  - fusion forms or learned heads (S27–S29, S34);
  - assigning modality roles (S38).
- The three-way ensemble adds within-acquisition F1 only (+.012, cross n.s.). It is kept as a
  descriptive ensemble effect, not an architecture component.
- [D57](../../DECISIONS.md).

## 4 · Reproduce

```sh
PYTHONPATH=src:scripts python scripts/run_modality_roles.py freeze
PYTHONPATH=src:scripts python scripts/run_modality_roles.py run
```
