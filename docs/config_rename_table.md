# CONFIG key-rename table

**Generated** by `scripts/check_config_roundtrip.py --emit-markdown` — do not edit by hand.

Maps every key of the pre-refactor `CONFIG` dict (`HSI_modality_training/hsi_training.py` @ `886560f`, 81 keys) to its single home in `configs/`. Enforced by `scripts/check_config_roundtrip.py` (REFACTOR_PLAN.md §3.3).

| Old `CONFIG` key | New config path | Value | File |
|---|---|---|---|
| `patches_data` | `data.patches_data` | `'./dataset/patches_spa_40b.npy'` | `configs/data/spa40_90class.yaml` |
| `labels_path` | `data.labels_path` | `'./dataset/labels.npy'` | `configs/data/spa40_90class.yaml` |
| `wavelength_path` | `data.wavelength_path` | `'./dataset/wavelengths_spa_40b.csv'` | `configs/data/spa40_90class.yaml` |
| `output_dir` | `output_dir` ⚠️ | `'outputs/output_v12_spa40'` | `configs/experiment/output_v12_spa40.yaml` |
| `num_bands` | `data.num_bands` | `40` | `configs/data/spa40_90class.yaml` |
| `num_classes` | `data.num_classes` | `90` | `configs/data/spa40_90class.yaml` |
| `s1_epochs` | `stage1.epochs` | `400` | `configs/stage1/progressive_3phase.yaml` |
| `s1_phase1_frac` | `stage1.phase1_frac` | `0.3` | `configs/stage1/progressive_3phase.yaml` |
| `s1_phase2_frac` | `stage1.phase2_frac` | `0.38` | `configs/stage1/progressive_3phase.yaml` |
| `s1_batch` | `stage1.batch` | `128` | `configs/stage1/progressive_3phase.yaml` |
| `s1_max_lr` | `stage1.max_lr` | `0.0005` | `configs/stage1/progressive_3phase.yaml` |
| `s1_mid_lr` | `stage1.mid_lr` | `0.00025` | `configs/stage1/progressive_3phase.yaml` |
| `s1_min_lr` | `stage1.min_lr` | `5e-06` | `configs/stage1/progressive_3phase.yaml` |
| `s1_dropout` | `stage1.dropout` | `0.15` | `configs/stage1/progressive_3phase.yaml` |
| `s1_mixup` | `stage1.mixup` | `0.35` | `configs/stage1/progressive_3phase.yaml` |
| `s1_patience` | `stage1.patience` | `50` | `configs/stage1/progressive_3phase.yaml` |
| `s1_accum` | `stage1.accum` | `1` | `configs/stage1/progressive_3phase.yaml` |
| `s1_focal_gamma` | `stage1.focal_gamma` | `1.5` | `configs/stage1/progressive_3phase.yaml` |
| `s1_label_smooth_hi` | `stage1.label_smooth_hi` | `0.1` | `configs/stage1/progressive_3phase.yaml` |
| `s1_label_smooth_lo` | `stage1.label_smooth_lo` | `0.04` | `configs/stage1/progressive_3phase.yaml` |
| `s1_ema_reinit_phases` | `stage1.ema_reinit_phases` | `True` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_supcon_weight` | `stage1.p3_supcon_weight` | `0.35` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_proto_weight` | `stage1.p3_proto_weight` | `0.15` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_oversample` | `stage1.p3_oversample` | `True` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_oversample_power` | `stage1.p3_oversample_power` | `0.65` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_oversample_max_w` | `stage1.p3_oversample_max_w` | `7.0` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_hard_f1_thresh` | `stage1.p3_hard_f1_thresh` | `0.8` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_oversample_eps` | `stage1.p3_oversample_eps` | `0.05` | `configs/stage1/progressive_3phase.yaml` |
| `s1_p3_dropout` | `stage1.p3_dropout` | `0.25` | `configs/stage1/progressive_3phase.yaml` |
| `branch_drop_prob` | `model.branch_drop_prob` | `0.2` | `configs/model/spectral_quadnet_v4.yaml` |
| `subcenter_K` | `model.subcenter_K` | `3` | `configs/model/spectral_quadnet_v4.yaml` |
| `max_cutout_bands` | `data.max_cutout_bands` | `3` | `configs/data/spa40_90class.yaml` |
| `noise_std` | `data.noise_std` | `0.02` | `configs/data/spa40_90class.yaml` |
| `aux_head_hidden` | `model.aux_head_hidden` | `128` | `configs/model/spectral_quadnet_v4.yaml` |
| `aux_loss_weight_init` | `stage1.aux_loss_weight_init` | `0.65` | `configs/stage1/progressive_3phase.yaml` |
| `aux_loss_weight_final` | `stage1.aux_loss_weight_final` | `0.25` | `configs/stage1/progressive_3phase.yaml` |
| `s2_epochs` | `stage2.epochs` | `150` | `configs/stage2/arcface_supcon.yaml` |
| `s2_batch` | `stage2.batch` | `128` | `configs/stage2/arcface_supcon.yaml` |
| `s2_head_lr` | `stage2.head_lr` | `0.00025` | `configs/stage2/arcface_supcon.yaml` |
| `s2_back_lr` | `stage2.back_lr` | `7e-05` | `configs/stage2/arcface_supcon.yaml` |
| `s2_min_lr` | `stage2.min_lr` | `1e-06` | `configs/stage2/arcface_supcon.yaml` |
| `s2_warmup_ep` | `stage2.warmup_ep` | `3` | `configs/stage2/arcface_supcon.yaml` |
| `s2_sgdr_T0` | `stage2.sgdr_T0` | `25` | `configs/stage2/arcface_supcon.yaml` |
| `s2_sgdr_Tmult` | `stage2.sgdr_Tmult` | `2` | `configs/stage2/arcface_supcon.yaml` |
| `s2_dropout` | `stage2.dropout` | `0.1` | `configs/stage2/arcface_supcon.yaml` |
| `s2_patience` | `stage2.patience` | `30` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_s` | `stage2.arcface_s` | `48.0` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_m` | `stage2.arcface_m` | `0.35` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_m0` | `stage2.arcface_m0` | `0.18` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_m_delta` | `stage2.arcface_m_delta` ⚠️ | `0.2` | `configs/stage2/arcface_supcon.yaml` |
| `s2_margin_warmup_ep` | `stage2.margin_warmup_ep` | `20` | `configs/stage2/arcface_supcon.yaml` |
| `s2_focal_gamma` | `stage2.focal_gamma` | `1.5` | `configs/stage2/arcface_supcon.yaml` |
| `cdws_max_weight` | `stage2.cdws_max_weight` | `3.0` | `configs/stage2/arcface_supcon.yaml` |
| `cdws_eps` | `stage2.cdws_eps` | `0.05` | `configs/stage2/arcface_supcon.yaml` |
| `supcon_weight` | `stage2.supcon_weight` | `0.4` | `configs/stage2/arcface_supcon.yaml` |
| `supcon_temp` | `stage2.supcon_temp` | `0.1` | `configs/stage2/arcface_supcon.yaml` |
| `proto_weight` | `stage2.proto_weight` | `0.18` | `configs/stage2/arcface_supcon.yaml` |
| `proto_temp` | `stage2.proto_temp` | `0.1` | `configs/stage2/arcface_supcon.yaml` |
| `bal_n_cls` | `stage2.bal_n_cls` | `16` | `configs/stage2/arcface_supcon.yaml` |
| `bal_n_spc` | `stage2.bal_n_spc` | `8` | `configs/stage2/arcface_supcon.yaml` |
| `s3_epochs` | `stage3.epochs` | `120` | `configs/stage3/sam_swa.yaml` |
| `s3_swa_lr` | `stage3.swa_lr` | `4e-05` | `configs/stage3/sam_swa.yaml` |
| `s3_cycle_len` | `stage3.cycle_len` | `8` | `configs/stage3/sam_swa.yaml` |
| `s3_sam_rho` | `stage3.sam_rho` | `0.015` | `configs/stage3/sam_swa.yaml` |
| `s3_greedy` | `stage3.greedy` | `True` | `configs/stage3/sam_swa.yaml` |
| `s3_aux_loss_weight` | `stage3.aux_loss_weight` | `0.1` | `configs/stage3/sam_swa.yaml` |
| `weight_decay` | `weight_decay` | `0.0002` | `configs/experiment/output_v12_spa40.yaml` |
| `grad_clip` | `grad_clip` | `1.0` | `configs/experiment/output_v12_spa40.yaml` |
| `ema_decay` | `ema_decay` | `0.999` | `configs/experiment/output_v12_spa40.yaml` |
| `tta_spatial` | `tta_spatial` | `8` | `configs/experiment/output_v12_spa40.yaml` |
| `tta_spectral` | `tta_spectral` | `4` | `configs/experiment/output_v12_spa40.yaml` |
| `wl_embed_dim` | `model.wl_embed_dim` | `16` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_patch` | `model.specf_patch` | `8` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_dim` | `model.specf_dim` ⚠️ | `192` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_heads` | `model.specf_heads` | `8` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_layers` | `model.specf_layers` | `4` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_drop` | `model.specf_drop` | `0.15` | `configs/model/spectral_quadnet_v4.yaml` |
| `fusion_heads` | *(deleted)* 🗑️ | — | — |
| `fusion_drop` | `model.fusion_drop` | `0.1` | `configs/model/spectral_quadnet_v4.yaml` |
| `device` | `device` ⚠️ | `'auto'` | `configs/experiment/output_v12_spa40.yaml` |
| `seed` | `seed` | `42` | `configs/experiment/output_v12_spa40.yaml` |

⚠️ = value intentionally differs from the pre-refactor constant:

- **`output_dir`** — §4.3 — hardcoded absolute machine-specific path replaced by ${output_root}/${run_name}; points at the Phase 1 relocation target. Pre-refactor value: `'/Users/jerlshin/FieldOfInterest/ResearchWork/HSI_RGB_seeds/Code/HSI_modality_training/output_v12_SPA40'`.
- **`s2_arcface_m_delta`** — T2-8 / HD-3 — 0.10 → 0.20. The key kept its name and changed the rule it parameterises: it was the F1-driven rule's m_delta in `M(c) = m_base + m_delta (1 - F1_c)`, and is now the signed rule's in `M(c) = clip(m_base + m_delta (R_c - P_c), 0.20, 0.50)`. The plan specifies 0.20 for the latter (§3.5 HD-3). Reusing the key rather than adding a second one keeps a single margin-scale knob; the sign change is pinned by `tests/unit/test_margin_rule.py::test_margin_rule_sign`. Pre-refactor value: `0.1`.
- **`specf_dim`** — T3-3 / BR-4 — 256 → 192. Branch D's token embeddings are now derived from each λ-window's centre wavelength and its spectral attention carries a relative-λ bias, so the branch does not have to spend capacity rediscovering the wavelength axis from an arbitrary index table. The key kept its name and the branch lost 0.94 M parameters, which is what funds BR-3's 3-D stem in Branch C (§3.8). Pinned by `tests/unit/test_specformer_lambda.py`. Pre-refactor value: `256`.
- **`device`** — §4.3 — YAML cannot hold a torch.device object, so the config carries the resolution strategy ("auto") and utils/device.py performs the lookup. Phase 5 widens that lookup from the baseline's cuda-or-cpu to Metal → CUDA → CPU, so an Apple Silicon host uses its GPU instead of falling through to the CPU. An explicit device=cuda/cpu/mps still wins. Pre-refactor value: `"<expr> torch.device('cuda' if torch.cuda.is_available() else 'cpu')"`.

## Net-new fields (no `CONFIG` ancestor)

- **`run_name`** — §4.3 — run identity; feeds output_dir instead of hardcoding a path.
- **`output_root`** — §4.3 — output_dir = ${output_root}/${run_name}.
- **`aux_gradnorm_alpha`** — T2-6 / OP-2 — GradNorm exponent for the per-branch auxiliary weights. At 0.0 the weights stay at the hardcoded A/B = 2x vector it replaces, so the pre-Tier-2 behaviour remains expressible.
- **`data.cutmix_bands`** — T2-7 / OP-6 — width of the same-class spectral CutMix window.
- **`data.cutmix_spatial`** — T2-7 / OP-6 — side of the same-class spatial CutMix paste.
- **`model.subcenter_tau_init`** — T2-9 / HD-2(i) — sub-centre pooling temperature at stage entry.
- **`model.subcenter_tau_final`** — T2-9 / HD-2(i) — temperature at stage end; at tau -> 0 the pooling is the hard max_k the head was defined on.
- **`model.subcenter_balance_weight`** — T2-9 / HD-2(ii) — weight on sum_c KL(pi_c || uniform), the mixture-of-experts load-balancing term sub-centre ArcFace was missing.
- **`stage1.arcface_m`** — T2-10 / HD-1 — Stage 1's margin under the unified head. 0.0 is not a placeholder: it makes the head a cosine (NormFace) classifier, which is what removes the Stage-1 -> Stage-2 discontinuity of §2.4.6.
- **`stage2.arcface_m_min`** — T2-8 / HD-3 — lower clip on the signed R-P margin rule.
- **`stage2.arcface_m_max`** — T2-8 / HD-3 — upper clip on the signed R-P margin rule.
- **`stage2.pairwise_margin_delta`** — T2-8 / HD-3 — scale of the row-normalised confusion term that aims the margin at the classes each class is actually confused with.
- **`stage3.margin_kappa_final`** — T2-1 / OP-4.2-4.3 — the multiplicative margin anneal's endpoint. Stage 3 keeps Stage 2's per-class vector and scales all of it, stepping only at cycle boundaries.
- **`stage3.swa_warmup_cycles`** — T2-3 / OP-4.5 — cycles discarded from the SWA average before the first candidate is considered, keeping Adam's 1/(1-beta2) second-moment transient out of it.
- **`stage3.sam_adaptive`** — T2-4 / OP-5 — selects ASAM. SAM's rho-ball is not scale-invariant and the ArcFace head is, so a raw-space budget has no meaning there.
- **`data.groups_path`** — T4-1 / P-1 — the per-patch scan id `scripts/prepare_dataset.py` now writes. Required by the grouped scheme; read under `stratified` too, where it is used only to measure how many scans cross the train/eval boundary (0-H measured 107 of 107).
- **`data.split_scheme`** — T4-1 / P-1 — `stratified` is the reference run's patch-level split, kept because the archived checkpoints were selected on it; `grouped` is the scan-disjoint protocol. The default stays `stratified` so this config keeps reproducing the run it describes; `configs/data/spa40_90class_pfix.yaml` is the P-fix protocol.
- **`data.split_eval_frac`** — T4-1 / P-1 — share held out for val+test. 0.30 reproduces the reference 70/15/15 exactly on the stratified path.
- **`data.split_fold`** — T4-1 / P-1 — rotates which scans are held out; sweeping it is the leave-one-scan-out cross-validation §3.1 falls back to when a class has too few scans for a three-way disjoint split. Must be 0 under `stratified`, which has no groups to rotate.
- **`data.calib_frac`** — T4-5 / P-5 — share of the training pool carved off as `calib`, where the per-class margins, the CDWS weights and the Phase-3 oversampling weights are fitted. 0.0 leaves them on `val`, i.e. on the split that also selects the checkpoint (C-9), which is what the reference run did.
- **`data.masks_path`** — T3-7 / FE-2 — the persisted fill map alpha `scripts/prepare_dataset.py` writes under P-3. Passing it makes the four masked modules functions of the seed's pixels rather than of `sum_c |x_c| > 1e-5`, and so immune to any global brightness transform. Empty falls back to that threshold, exactly, which is why the pre-Tier-3 arrays still reproduce.
- **`data.morphology_path`** — T3-1/T3-4 / P-4 — the eight morphometrics, which Branch B's index bank and FU-4's fifth fusion token both consume. Empty substitutes zeros, the mean of the train-standardised feature.
- **`model.grid_size_a`** — T3-6 / BR-2 — Branch A's grid, 4x4 -> 8x8. Costs no parameters (cells are processed independently) and takes the spatial compression from 256:1 to 64:1.
- **`model.grid_size_d`** — T3-6 / BR-2 — Branch D's grid, held at 4x4. A and D received a byte-identical tensor before Tier 3 (§2.2.2); separate keys are what make that impossible to reintroduce silently.
- **`model.index_bank_size`** — T3-1 / BR-1(i) — number of learned normalised-difference indices, the gain-invariant replacement for the rank-2 moment tensor of §2.2.5.
- **`model.continuum_depths`** — T3-1 / BR-1(ii) — how many of the deepest hull-removed absorption features Branch B reads.
- **`model.n_morphometrics`** — T3-1/T3-4 / P-4 — width of the morphometric vector; the eight columns `data/prep/segmentation.py::MORPHOMETRIC_NAMES` writes.
- **`model.stem_channels`** — T3-2 / BR-3 — width the 3-D spectral-spatial stem folds the spectral axis into before the 2-D tail. C-3: before this the network contained no joint spectral-spatial operator at all.
- **`model.fusion_rank`** — T3-4 / FU-1(b) — rank of the bilinear projections U_m. The second-order term M-3 found missing, at 5*d*r rather than a full 10*d^2.
- **`model.fusion_gate_hidden`** — T3-4 / FU-1(b)+FU-2 — hidden width of the sigmoid gate MLP, which reads the five normalised tokens and the five pre-normalisation log-norms.
- **`tracking.*`** — §4.1 — experiment tracking is additive; the monolith used bare print().
- **`runtime.*`** — Execution knobs — DataLoader workers, pinned staging, torch.compile, fused AdamW, DDP topology, allocator sweeps, console rendering. Excluded rather than mapped because the reference implementation had no counterpart to map to: it fed the model one sample at a time on the training device with num_workers=0 and no notion of a second GPU. The group carries the invariant that nothing in it may change a reported number, which is what keeps it out of the experiment's identity — the two fields that *would* change one (allow_tf32, channels_last) default to off.

## 🗑️ Deleted `CONFIG` keys (nothing left to configure)

- **`fusion_heads`** — T3-4 / FU-1(b) — N-1a's dead key, deleted rather than wired. It named the head count of `CrossModalInteraction`'s multi-head attention, and the gated low-rank bilinear fusion that replaced the Perceiver has no attention at all: with five modality tokens, latent cross-attention compresses nothing (§3.4 FU-1). Nothing in the model can consume a head count, so there is nothing to wire it to. Pre-refactor value: `4`.
