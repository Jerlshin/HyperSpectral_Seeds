# S02 · Architecture and protocol revision

| | |
|---|---|
| **Status** | implemented and verified by tests; **not yet validated by a training run** |
| **Dates** | 2026-08-13 → 2026-08-14 |
| **Commits** | `a4883c1` (revised architecture, evaluation, ablation framework), `2a459ef` (branches A and D dropped; full 256-band default), `e8dd0bc` (band study, S03) |
| **Data** | default switched from SPA-40 to the full cube; since S07 the 215-band reflectance cube |
| **Code** | `models/spectral_seed_net.py`, `engine/stages/single_stage.py`, `data/loaders.py` (`grouped_split`, `assert_protocol_holds`), `experiments/` (registry, protocol driver, leakage, baselines), `reporting/` |
| **Docs** | [`docs/01`–`06`](../../../01_ABSTRACT_AND_OVERVIEW.md) describe the result in full |
| **Findings** | F29 · **Decisions** D01–D07 |

## 1 · Question
Can the audit's implementation changes (IC-1 … IC-14) be made such that the protocol is enforced by
code rather than convention, and the simplified model is provably identical to the audited one
where the two overlap?

## 2 · Why we did this
S01: every claim was blocked by the protocol, and the model carried components with no evidence.

## 3 · Hypotheses
Engineering acceptance criteria rather than scientific hypotheses — each IC entry in CHANGES §24
names its validation (e.g. "the 180-of-180 warning does not appear"; "`aux_weight/*` is constant").

## 4 · Method — what changed

| Area | Before (audited) | After |
|---|---|---|
| Split | stratified, `calib_frac = 0` | **grouped**, 2 folds, `calib_frac = 0.15`; fails the run if grouped is requested and not realised |
| Reporting | val, max over ~944 selections | calib selects; `val ∪ test` scored once; mean ± range over 2 folds × 3 seeds; bootstrap CI |
| Input | 40 SPA bands (leaky) | **all acquired bands**; band selection opt-in (`data.band_indices_path`) |
| Model | SpectralQuadNet, 5.19 M, 4 branches + bilinear fusion | **SpectralSeedNet**, 3.0 M: spatial 3-D-stem CNN ‖ spectral MLP, concat, K = 1 ArcFace, one aux head |
| Curriculum | 3 stages, 69 hyperparameters, ~19 h | **1 stage**, 14 hyperparameters; mixup then a warmed global margin |
| Optimisation | GradNorm, clip 1.0, AMP off in contrastive phases | fixed aux 0.2, clip 5.0, bf16 throughout, TF32 off |
| Telemetry | stage-local W&B step | monotone global step; raw and weighted aux losses |
| Band-count handling | hardcoded stride halvings | stride schedule derived from `num_bands` (every band reaches a tap); exact O(C²) continuum hull |

Retained deliberately: SpectralQuadNet and the three-stage modules, as control arms for A3 and A8.

## 5 · Results
Verification only — no training run:

```
pytest --run-all            918 passed, 2 failed, 29 skipped
ruff / mypy --strict        clean (116 source files)
check_config_roundtrip.py   all 81 keys map 1:1
capture_golden.py --verify  v3/logits match (max |Δ| = 0.000e+00); 306 init digests match
```
The two failures are pre-existing Stage-1 golden-digest drift (environment, not code) — FW-11.

## 6 · Findings
- [F29](../../FINDINGS.md) — the 256-band-native mechanisms reproduce the audited 40-band control
  model bit-exactly (E4).

## 7 · Decisions this led to
D01–D07. **Two of them (D05, D06) were taken before the ablations that were meant to gate them
(A3, A8)** — see the *Deviation* notes in [DECISIONS](../../DECISIONS.md).

## 8 · Threats to validity
None of this shows the new model is *better*; it shows it is *correctly built* and that the
protocol cannot silently leak. Performance claims wait for S08.

## 9 · What would change these conclusions
A3 / A8 (FW-07). F17 already pulls against D04.

## 10 · Reproduce
```bash
pytest --run-all
python scripts/capture_golden.py --verify
python scripts/check_config_roundtrip.py
```

## 11 · Provenance
README §9 "Recorded result of this revision's validation"; `tests/`; `docs/config_migration_table.md`.
