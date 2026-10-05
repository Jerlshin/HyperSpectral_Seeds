# S33 · Does a measured-nuisance RGB branch transfer away from session 8?

| | |
|---|---|
| **Status** | complete — **H52 and H53 fail**: the measured-nuisance treatment lowers away-from-session-8 recall |
| **Dates** | 2026-10-05 |
| **Data** | S20 crops; S21 corrected folds; S22 seed-0 v5 TTA; S32 ViT-B cells (matched control) |
| **Code** | `scripts/run_rgb_acquisition.py` (freeze · render · run); training through `scripts/run_rgb_finetune.py train --plan configs/research/s33_rgb_acquisition.json --out outputs/s33_rgb_acquisition --arm vitb_acq` |
| **Plan** | `configs/research/s33_rgb_acquisition.json`, SHA-256 `2873190a0c2ab9749455c8f91609ec70711ee6dacdcd7a84cbdf035aa185c32d`. Frozen after S32's ViT-B cells and before any S32/S33 held-out scoring |
| **Registers** | F124 · D57 · H52–H53 |

## 1 · Question and mechanism

F113: no model change has moved away-from-session-8 recall. S31 §3 measured what the RGB camera
does differently in session 8, label-free:
- crops carry about half the fine-texture energy, Gaussian-equivalent σ ≈ 0.5 px;
- red clipping falls from 14–21% to 4% of kernel pixels;
- there is no within-variety contrast in training to learn invariance from.

S19 §6 allows a nuisance augmentation only if it is *measured* and kept *within its measured
range*. S33 is that test, with the training-only blur and the test-time rendering as separable arms:
- **`vitb_acq`:** the S32 ViT-B recipe unchanged, plus Gaussian blur σ ~ U(0, 1.0) px on half the
  training crops. The foreground support is never changed. S32's ±10% exposure jitter already
  covers clipping.
- **Test-time rendering:** every calib/held-out crop is blurred at the measured σ = 0.5 before the
  four views, so all classes are compared in the softer regime. Blur is one-way.
- Arms: `ft_vitb_tta` (S32), `ft_vitb_render`, `acq_vitb_tta`, `acq_vitb_render`, each fused
  equally with v5 TTA. Primary contrast: `acq_vitb_render` − `ft_vitb_tta`, pre-declared,
  because calib cannot measure transfer.

## 2 · Results (held-out, both corrected folds, seed 0)

Training: two cells of 50.5 / 43.3 min on MPS. Calib F1 (id / 4-view) was .776 / .800 and
.790 / .804, matching S32 (.775 / .789, .784 / .817) within noise, so blur costs nothing *within* an acquisition.
Rendering pass: 4 models × calib + held-out, 15 min.

| Arm | RGB F1 | RGB from / to s8 | Fused F1 | Fused cross | Fused from / to s8 |
|---|---:|---:|---:|---:|---:|
| S32 `ft_vitb_tta` (control) | .6642 | .239 / .323 | .6977 | .2953 | .254 / .337 |
| `ft_vitb_render` (render only) | .6579 | .245 / .327 | .6974 | .3014 | .261 / .342 |
| `acq_vitb_tta` (augmentation only) | .6585 | .220 / .318 | .6973 | .2836 | .232 / .336 |
| **`acq_vitb_render` (primary)** | .6570 | .209 / .334 | .6963 | .2885 | .226 / .352 |

| Contrast | ΔF1 [CI] | Δ from-s8 recall [cell CI] | Δ to-s8 |
|---|---:|---:|---:|
| **H52** RGB: `acq_vitb_render` − `ft_vitb_tta` | −.0072 [−.0126, −.0021] | **−.0307 [−.0539, −.0062]** | +.011 |
| **H53** system | −.0014 [−.0059, .0034] | **−.0282 [−.0392, −.0172]** | +.015 |
| augmentation only (RGB) | −.0057 [−.0108, −.0007] | −.0196 [−.0404, .0024] | −.005 |
| rendering only (RGB) | −.0063 [−.0101, −.0025] | +.0061 [−.0110, .0220] | +.004 |

**Both gates fail, in the opposite direction to the mechanism.** Training with blur lowers
away-from-session-8 recall. Rendering at test time is neutral for transfer and costs .006 F1.

## 3 · Finding and decision

- **F124**: the session-8 optical difference measured in S31 (softer texture, less clipping) does
  not limit the trained branch's transfer. Simulating it in training removes useful fine texture
  (F115) without any compensating transfer gain. [E3, 1 seed]
- **[D57](../../DECISIONS.md)**: no measured-optics augmentation or rendering in the architecture.

## 4 · Threats

- One seed. The blur range was taken from a session-level texture median that mixes variety
  composition with optics (S31 §8).
- Exposure/clipping jitter was already ±10% in both arms and was not varied here.

## 5 · Reproduce

```sh
PYTHONPATH=src:scripts python scripts/run_rgb_acquisition.py freeze        # after S32's ViT-B cells, before scoring
for f in 0 1; do PYTHONPATH=src:scripts python scripts/run_rgb_finetune.py train --arm vitb_acq --fold $f \
  --plan configs/research/s33_rgb_acquisition.json --out outputs/s33_rgb_acquisition; done   # ~95 min MPS
PYTHONPATH=src:scripts python scripts/run_rgb_acquisition.py render         # ~15 min MPS
PYTHONPATH=src:scripts python scripts/run_rgb_acquisition.py run
```
