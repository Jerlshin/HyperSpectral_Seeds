# Configuration key reference

Every experiment knob is a typed dataclass field in `src/spectralquadnet/config/schema.py`,
registered with Hydra's `ConfigStore` so a malformed or missing key in `configs/` fails at
composition time. Value-carrying fields default to `omegaconf.MISSING`, so a key absent from
`configs/` fails loudly rather than silently falling back to a drifted default; `TrackingConfig`
and `RuntimeConfig` are the two exceptions, since both carry real defaults by design (§ below).

This table documents each group's keys, their shipped values, and what they mean, grouped the
way `configs/` is grouped. It is the ground-truth reference for `cfg.<group>.<key>` throughout
the rest of this suite.

---

## `data` — `configs/data/*.yaml`

Seven configs ship, in two clearly separated tiers.

**Primary — the 215-band white-tile reflectance cube in `./dataset/`, no band selection:**

| Config | Split | Role |
|---|---|---|
| **`refl215_grouped.yaml`** | `grouped` | **The default.** Leave-one-acquisition-bundle-out + a calibration split. |
| `refl215_stratified.yaml` | `stratified` | A1's *contrast* arm, identical in everything but `split_scheme`, so the gap between them measures the split and nothing else. |

Reflectance $\varrho = (R-\bar D)/W_\text{tile}$ per scan. 215, not 256 bands: the tile clips in
608.0–705.8 nm in every scan of sessions 0–7, and those 41 bands are dropped from every scan
(`02_DATASET_AND_PREPROCESSING.md`). A k-band finalist uses `outputs/band_finalists/`, cut from
this axis by `scripts/write_finalist_bands.py`.

**`configs/data/ablation/` — reduced-band arms, never on the primary path:**

| Config | Bands | Split | Role |
|---|---:|---|---|
| `spa40_grouped.yaml` | 40 | `grouped` | A2's reduced arm (SPA subset of the former 256-band SNV cube; its `patches_spa_40b.npy` must be rebuilt by `scripts/select_bands.py`). |
| `spa40_stratified.yaml` | 40 | `stratified` | Its leaky twin, if A1 is re-run at k = 40. |
| `spa40_audited.yaml` | 40 | `stratified` | **Frozen.** Reproduces the audited run's input and partition exactly; composed only by `experiment/quadnet_audited` and the golden capture. Do not tidy it. |
| `u430k32_grouped.yaml` | 32 | `grouped` | The evenly spaced 32-band finalist (`uniform430_k32`) read from the **pre-sliced** float16 cube in `./dataset_u430k32/` (`scripts/build_presliced_dataset.py`, 2.3 GB) — the Kaggle input. Its `band_axis.json` makes the run report "32 of 215 bands", not the full cube. `gain_path` empty (not uploaded). |
| `u430k32_stratified.yaml` | 32 | `stratified` | Its patch-level contrast twin, identical but for `split_scheme`. |

Values below are `refl215_grouped.yaml`'s; the last column gives the frozen replica's, which is
what the pre-refactor `CONFIG` keys map onto in `config_migration_table.md`.

| Key | Value | Meaning | `ablation/spa40_audited.yaml` |
|---|---|---|---|
| `patches_data` | `./dataset/patches.npy` | patch cube path — the direct product of `scripts/prepare_dataset.py` | `./dataset/patches_spa_40b.npy` |
| `labels_path` | `./dataset/labels.npy` | class index per patch | — |
| `wavelength_path` | `./dataset/wavelengths.csv` | the 215-band reflectance axis, 383.2–1006.5 nm without 608.0–705.8 nm | `./dataset/wavelengths_spa_40b.csv` |
| `band_indices_path` | `""` | **BS-1** — optional `.npy` of band indices sliced off the mmap as each patch is read. Empty on every primary config, and that emptiness *is* the no-band-selection methodology; the band study's neural arms set it | `""` |
| `masks_path` | `./dataset/masks.npy` | persisted fill map $\alpha$; empty uses the `sum_c\|x_c\|>10^{-5}` fallback (`02_DATASET_AND_PREPROCESSING.md` §2.3, `03_MODEL_ARCHITECTURE.md` §3.1) | `""` |
| `morphology_path` | `./dataset/morphology.npy` | persisted 8-column morphometrics; empty substitutes zeros | `""` |
| `num_bands` | `215` | **every band with a measured reflectance.** Checked against the cube and the wavelength CSV by `data/mmap_store.py::band_geometry` before the model is built | `40` |
| `num_classes` | `90` | rice-seed varieties | — |
| `groups_path` | `./dataset/groups.npy` | per-patch scan id; required by `grouped`, read under `stratified` only to measure train/eval scan overlap | — |
| `scan_table_path` | `./dataset/scan_table.csv` | one row per scan naming its acquisition **session**. Read only by the final evaluation's session breakdown (`spectralquadnet.reporting.session`): same- vs cross-session recall, attraction to the kernel's own session, session-prediction entropy. **Never a model input** | `""` (breakdown skipped) |
| `split_scheme` | `grouped` | `grouped` — scan-disjoint split (§2.8); `stratified` — patch-level, every scan in all three splits | `stratified` |
| `split_eval_frac` | `0.30` | share held out for val∪test | — |
| `split_fold` | `0` | which scan(s) are held out under `grouped`; must stay `0` under `stratified`. Sweeping `{0, 1}` is the complete leave-one-bundle-out CV this dataset supports | — |
| `calib_frac` | `0.15` | share of the training pool carved into `calib`, where per-class margins/CDWS/oversampling weights are fitted; `0.0` fits them on `val` instead | `0.0` |
| `max_cutout_bands` | `16` | max contiguous bands zeroed by the `cutout` augmentation — **7.5% of the band axis**, derived by `data/datasets.py::band_augmentation_widths` so the augmentation means the same thing at every band count | `3` |
| `noise_std` | `0.02` | base σ of the spectral-noise augmentation; a per-band amplitude, band-count independent | — |
| `cutmix_bands` | `43` | band-window width of same-class spectral CutMix — **20% of the band axis**, same derivation | `8` |
| `cutmix_spatial` | `24` | side length of same-class spatial CutMix; band-count independent | — |
| `single_group_policy` | `error` | **IC-3** — what `grouped` does about a class captured in a single bundle. `error` refuses and names them; `patch_split` accepts a patch-level split for those classes with the leak recorded in the report | `error` |
| `gain_path` | `./dataset/gain.npy` | **IC-14** — per-pixel `(mean, sd)` along λ. **Never a model input**: it is the residual brightness SNV divided out, and therefore the strongest single carrier of acquisition-bundle identity (CHANGES §3.3). Consumed by `spectralquadnet.experiments.leakage`, which *measures* the acquisition signal instead of feeding it to the classifier | `""` |

---

## `model` — `configs/model/{seed_net,quadnet_v4_audited}.yaml`

Two architectures ship, selected by `model.arch`. `seed_net` (the default) is CHANGES §16.2's
two-pathway replacement — **3,003,412** parameters on the 215-band primary input;
`quadnet_v4_audited` is the audited four-branch model — **5,242,710** at 215 bands and
5,194,578 at the audited 40 — retained unmodified because ablations A3/A4/A5/A8 are comparisons
*against* it. Every width in both is a function of `data.num_bands`; only three components move
with it at all (`03_MODEL_ARCHITECTURE.md` §3.7).

| Key | `seed_net` | `quadnet_v4_audited` | Meaning |
|---|---|---|---|
| `arch` | `spectral_seed_net` | `spectral_quadnet` | **IC-10** — which class `models/registry.py::build_model` constructs |
| `enabled_branches` | (unread) | `[a, b, c, d]` | **A3** — which branches are constructed at all; a disabled branch costs no parameters, no auxiliary head and no fusion modality |
| `branch_drop_prob` | `0.0` | `0.20` | overall branch-drop strength |
| `branch_drop_profile` | `[1, 1, 1, 1]` | `[0.75, 0.75, 0, 0.75]` | **A3 / §5.2** — per-branch drop ratio. The audited vector never dropped Branch C, which taught the fusion gate to route onto it; C's 87% influence is therefore *confounded*, and a symmetric profile is what makes A3 measure the branches rather than the policy |
| `fusion_mode` | `concat_mlp` | `bilinear_gate` | **A5 / §5.3** — `bilinear_gate` \| `gate` \| `concat_mlp` |
| `subcenter_K` | `1` | `3` | **IC-9** — sub-centres per class. The audited seeding logged a worst within-class sub-centre cosine of **0.987**: spherical *k*-means could not find three separated modes, so K=3 tripled the head's parameters to keep degenerate structure alive |
| `subcenter_balance_weight` | `0.0` | `0.01` | the KL load-balancing term; meaningless at K=1 |
| `per_class_margin` | `false` | `true` | **IC-9 / A7** — the signed $R-P$ margin rule. Off by default: Stage 2's best checkpoint was epoch 19 and the per-class vector took over at 21, so it was never active at the model that was reported |
| `pairwise_penalty` | `false` | `true` | **IC-9 / A7** — the row-normalised confusion penalty, fitted on the selection split in the audited run |
| `spatial_width_mult` | `1.0` | `1.0` | **A10** — scales the spatial path's ResBlock widths |
| `spectral_hidden` | `256` | (unread) | **IC-10** — the spectral MLP's hidden width |
| `aux_head_weight` | `0.2` | (unread) | **IC-5 / §7.1** — the fixed single-aux-head weight D07 describes. **Applied only under `single.aux_weight_schedule=fixed`**; under the default `legacy` the loop applies `stage1.aux_loss_weight_{init,final}`'s 0.65 → 0.25 decay, as every run up to S11 did (S10 F54) |
| `pathways` | `[spatial, spectral]` | (unread) | **X2 / S11** — live pathways. `[spectral]` / `[spatial]` zero the other pathway in train and eval, freeze it and skip its forward; `[spectral]` also drops the aux term, `[spatial]` the morphometrics (`03` §3.0) |
| `spectral_descriptor` | `full` | (unread) | **S13 Y3.** `full` = `[index bank ‖ continuum depths ‖ SNV ‖ D₁ ‖ D₂ ‖ morph]`; `snv_morph` = `[SNV ‖ morph]` — the blocks S10 F49 found inert are not built |
| `spatial_tail_strides` | `[2, 2, 2, 2]` | (unread) | **S13 Y3.** Strides of the four spatial-tail ResBlocks, each 1 or 2. The default ends at 1 × 1 (327,680 untrainable parameters, F48); `[2, 2, 2, 1]` ends at 2 × 2 |
| `cbam_min_hw` | `0` | (unread) | **S13 Y3.** A tail CBAM is kept only after a block whose output side ≥ this (else `nn.Identity`); `0` keeps all three, `3` drops the gate on the 2 × 2 map. Placement is computed for the patch side the pipeline passes to `build_model(…, input_side=…)` |
| `spatial_mixstyle` | `false` | (unread) | **S13 Y2.** Masked MixStyle after 3-D stem blocks 1 and 2 (p 0.5, Beta(0.1, 0.1), per-(channel, spectral-slice) foreground statistics, training only). Off builds nothing and draws no random number |

The keys below belong to branches `seed_net` does not have (`grid_size_a`, `grid_size_d`,
`specf_*`, `fusion_rank`, `fusion_gate_hidden`). They remain in the shared schema because
`quadnet_v4_audited` reads them and A3 is the experiment that decides whether their branches
stay removed. `tests/unit/test_branch_drop.py` and `tests/unit/test_config_wiring.py` enforce
that a key dead in **both** architectures is still a defect.

### The audited model's own keys — `configs/model/quadnet_v4_audited.yaml`

| Key | Value | Meaning |
|---|---|---|
| `branch_drop_prob` | `0.20` | base branch-dropout rate; realised per-branch rates are $(0.15,0.15,0.0,0.15)$ for A/B/C/D (§3.3) |
| `subcenter_K` | `3` | ArcFace sub-centres per class |
| `subcenter_tau_init` | `0.20` | sub-centre pooling temperature at stage entry |
| `subcenter_tau_final` | `0.02` | temperature at stage end; near $0$ the pooling is the hard $\max_k$ |
| `subcenter_balance_weight` | `0.01` | weight on $\sum_c \mathrm{KL}(\pi_c\|\mathrm{uniform})$, the sub-centre load-balancing term |
| `aux_head_hidden` | `128` | hidden width of each per-branch auxiliary head |
| `wl_embed_dim` | `16` | Fourier-feature width of Branch A's $\kappa_\phi$ kernel generator and Branch D's relative-λ bias |
| `grid_size_a` | `8` | Branch A's pooling grid ($8\times8$, 64 cells) |
| `grid_size_d` | `4` | Branch D's pooling grid ($4\times4$, 16 cells) |
| `index_bank_size` | `64` | learned normalised-difference indices in Branch B |
| `continuum_depths` | `16` | deepest continuum-removed absorption features Branch B reads |
| `n_morphometrics` | `8` | width of the persisted morphometric vector |
| `stem_channels` | `192` | channel width Branch C's 3-D stem folds the spectral axis into |
| `stem_folded_depth` | `8` | **256-band native** — the spectral *depth* the stem reduces the band axis to before folding. The three spectral strides and their kernel depths are derived from this and `data.num_bands`: `(8,2,2)` with kernels `(15,5,5)` at 256 bands, `(2,2,2)` with `(7,5,5)` at 40 — the audited schedule, unchanged (`03_MODEL_ARCHITECTURE.md` §3.0(a)) |
| `specf_tokens` | `10` / `32` | Branch D's λ-uniform window count, set **directly**. 10 on the audited replica, 32 in `experiment/quadnet_full256`. Deriving it from `num_bands` made a window's *width* a function of $k$, so token $t$ denoted a different spectral region in every arm |
| `specf_dim` | `192` | Branch D's transformer model width |
| `specf_heads` | `8` | Branch D's attention heads |
| `specf_layers` | `4` | Branch D's total pre-LN blocks (2 spectral-stage + 2 spatial-stage) |
| `specf_drop` | `0.15` | Branch D's transformer dropout |
| `fusion_rank` | `128` | rank $r$ of the bilinear projections $U_m$ in fusion |
| `fusion_gate_hidden` | `128` | hidden width of the fusion modality-gate MLP |
| `fusion_drop` | `0.10` | fusion output dropout |

## `single` — `configs/single/one_stage.yaml`

**IC-11 / CHANGES §17.** The collapsed curriculum: one stage, one objective, one schedule.
Stage hyperparameter count 69 → 14. `pipeline=single` (the default) reads this group;
`pipeline=three_stage` reads `stage1`/`stage2`/`stage3` below instead, which is A8's control arm.

| Key | Value | Meaning |
|---|---|---|
| `epochs` | `150` | replaces 400 / 150 / 120 across three stages |
| `batch` / `accum` | `128` / `1` | unchanged from Stage 1 |
| `patience` | `25` | early stop on **calib** macro-F1. Lower than Stage 1's 50 deliberately: ~472 epochs × {live, EMA} was ~944 correlated selection draws, worth an expected **+0.042** macro-F1 of upward bias (§4.5) |
| `max_lr` / `min_lr` / `warmup_ep` | `5e-4` / `5e-6` / `5` | linear warm-up then **one** cosine decay, no restarts. `max_lr` is unchanged from the audited run on purpose — `grad_clip` moves 1.0 → 5.0 and §8.1 forbids co-tuning both in one run |
| `dropout` | `0.15` | one rate throughout; removes an untested 0.15 / 0.25 / 0.10 schedule |
| `label_smooth_hi` / `_lo` | `0.10` / `0.04` | linear decay |
| `focal_gamma` | `0.0` | plain CE. Focal addresses 1000:1 foreground/background imbalance; here it is 96:91, so γ>0 was down-weighting easy examples, untested (§7.4) |
| `aux_loss_weight` | `0.2` | **not read by any code path** (S10 F54); kept so archived configs compose |
| `aux_weight_schedule` | `legacy` | **S10 P0.4 / D20.** The aux weight the loop applies: `legacy` = `max(stage1.aux_loss_weight_final, stage1.aux_loss_weight_init·(1 − 0.7·t/T))` = 0.65 → 0.25, what every run so far trained with; `fixed` = `model.aux_head_weight`. Logged per epoch as `sched/aux_weight_applied`. Stays `legacy` until X1/X2 report (D21) |
| `clean_fit_kernels` / `clean_fit_seed` | `1000` / `0` | **S10 P0.2 / D18.** Clean-fit probe: a fixed class-stratified training subset of this size, drawn by a private RNG from this seed alone (identical across run seeds); eval mode, no aug, margin 0, plain CE; live + EMA → `fit/*`, `clean_fit.json`. `0` switches it off |
| `mixup` / `mixup_epochs` | `0.35` / `110` | the one demonstrably load-bearing regulariser: switching it off moved training accuracy 42% → 96.6% in a single epoch while validation did not move (§5.5) |
| `aug_profile` | `medium` | one profile throughout; the three-phase curriculum's profiles differed by 2–4 pp of trigger probability |
| `arcface_m` / `arcface_s` | `0.30` / `32.0` | 48 is high for $d{=}256$ at 90 classes and was never tuned |
| `margin_warmup_start` / `_end` | `111` / `130` | **the whole collapse**: mixup and a non-zero margin are mutually exclusive by construction, which is the only reason Stage 2 needed its own stage. Switch mixup off, warm one scalar margin in over the window that follows |
| `supcon_epochs` | `0` | optional Phase B, off until A6 shows SupCon beats plain CE by more than run-to-run variance **with the sampler controlled** |
| `supcon_weight` / `supcon_temp` | `0.30` / `0.10` | Phase B's contrastive term |
| `bal_n_cls` / `bal_n_spc` | `16` / `8` | Phase B's class-balanced batch shape |

---

## `evaluation` — `configs/evaluation/*.yaml`

**CHANGES §19.** Which split selects the checkpoint and which is reported. This group exists
because the audited run's single largest statistical defect was that these were *the same split*:
`calib_frac=0.0` put the per-class margins, the confusion matrix, the CDWS weights and the
Phase-3 oversampling weights on `val`, then selected the checkpoint on `val`, then reported the
number from `val` (§4.4).

| Key | `held_out_once` (default) | `audited_replica` | Meaning |
|---|---|---|---|
| `select_split` | `calib` | `val` | which split the per-epoch checkpoint decision reads |
| `report_split` | `val_test` | `test` | scored **once**, after freezing. Under `grouped`, val and test are two halves of the *same* held-out bundle and are therefore not independent of each other, so they are treated as one set |
| `tta` | `true` | `true` | score with and without the 12-view TTA, reported separately |
| `bootstrap_samples` | `2000` | `2000` | percentile CI on macro-F1. Sampling noise on ~1,300 patches is ±0.020 at 95%, and the audited run's entire Stage-2 + Stage-3 gain was +0.005 |
| `save_artifacts` | `true` | `true` | write the confusion matrix, per-class table and figures under `output_dir/results/` and `figures/` |
| `save_logits` | `true` | `true` (dataclass default) | **D18 / S11.** Write the selected weights' float16 logits (+ targets, rows) for the reported split and for `calib`, ±TTA, as `results/logits_<split>_<variant>.npz` |
| `session_probe` | `true` | `true` (dataclass default) | **D26 / S13.** After the held-out scoring, the training-rows class-disjoint session κ of the selected weights' embedding and each live pathway → `results/session_probe.json`, `run.json → session_probe`, `session_probe/kappa_*`. Reads no held-out row; needs `data.scan_table_path`; single-stage pipeline |

---

## `stage1` — `configs/stage1/progressive_3phase.yaml`

| Key | Value | Meaning |
|---|---|---|
| `epochs` | `400` | Stage-1 budget; phase boundaries at $\lfloor400\cdot0.30\rfloor=120$ and $\lfloor400\cdot0.68\rfloor=272$ |
| `phase1_frac` | `0.30` | Phase 1 length as a fraction of `epochs` |
| `phase2_frac` | `0.38` | Phase 2 length as a fraction of `epochs` |
| `batch` | `128` | Stage-1 batch size |
| `max_lr` | `5.0e-4` | phase-aware LR schedule peak |
| `mid_lr` | `2.5e-4` | LR at the Phase 1→2 boundary |
| `min_lr` | `5.0e-6` | LR floor of the post-Phase-2 cosine restarts |
| `dropout` | `0.15` | base Stage-1 dropout |
| `mixup` | `0.35` | mixup Beta$(\alpha,\alpha)$ parameter, Phases 1–2 |
| `patience` | `50` | early-stopping patience, epochs without a macro-F1 improvement |
| `accum` | `1` | gradient-accumulation micro-batches |
| `focal_gamma` | `1.5` | Phase-3 focal-loss exponent |
| `label_smooth_hi` | `0.10` | label smoothing at epoch 1 |
| `label_smooth_lo` | `0.04` | label smoothing at the final epoch |
| `ema_reinit_phases` | `true` | hard-reset the EMA shadow at each phase boundary |
| `p3_supcon_weight` | `0.35` | Phase-3 SupCon loss weight |
| `p3_proto_weight` | `0.15` | Phase-3 ProtoNCE loss weight |
| `p3_oversample` | `true` | enable `HardClassOversampledSampler` in Phase 3 |
| `p3_oversample_power` | `0.65` | exponent $\gamma$ of the inverse-F1 oversampling weight |
| `p3_oversample_max_w` | `7.0` | cap $W_{\max}$ on the oversampling weight |
| `p3_hard_f1_thresh` | `0.80` | diagnostic-only threshold for the sampler's hard-class print |
| `p3_oversample_eps` | `0.05` | $\epsilon$ floor in the inverse-F1 weight |
| `p3_dropout` | `0.25` | dropout after the Phase 2→3 boundary |
| `aux_loss_weight_init` | `0.65` | auxiliary-loss weight at epoch 1 |
| `aux_loss_weight_final` | `0.25` | auxiliary-loss weight floor |
| `arcface_m` | `0.0` | Stage 1's angular margin; `0.0` makes `arcface_head` a plain cosine (NormFace) classifier |

## `stage2` — `configs/stage2/arcface_supcon.yaml`

| Key | Value | Meaning |
|---|---|---|
| `epochs` | `150` | Stage-2 budget |
| `batch` | `128` | realised as $16\times8$ class-balanced |
| `head_lr` | `2.5e-4` | `arcface_head` LR |
| `back_lr` | `7.0e-5` | backbone LR |
| `min_lr` | `1.0e-6` | SGDR floor fraction base |
| `warmup_ep` | `3` | SGDR linear warm-up length |
| `sgdr_T0` | `25` | first SGDR restart period |
| `sgdr_Tmult` | `2` | restart period growth factor |
| `dropout` | `0.10` | Stage-2 dropout |
| `patience` | `30` | early-stopping patience |
| `arcface_s` | `48.0` | ArcFace logit scale |
| `arcface_m` | `0.35` | margin warm-up target / signed-rule $m_{\text{base}}$ |
| `arcface_m0` | `0.18` | margin warm-up start |
| `arcface_m_delta` | `0.20` | signed-rule $m_\Delta$ |
| `margin_warmup_ep` | `20` | epochs of global-scalar margin warm-up before per-class hand-over |
| `focal_gamma` | `1.5` | Stage-2 focal-loss exponent |
| `arcface_m_min` | `0.20` | per-class margin clip floor |
| `arcface_m_max` | `0.50` | per-class margin clip ceiling |
| `pairwise_margin_delta` | `0.10` | scale of the confusion-matrix pairwise penalty |
| `cdws_max_weight` | `3.0` | CDWS weight cap |
| `cdws_eps` | `0.05` | CDWS $\epsilon$ floor |
| `supcon_weight` | `0.40` | SupCon weight at full ramp |
| `supcon_temp` | `0.10` | SupCon temperature |
| `proto_weight` | `0.18` | ProtoNCE weight at full ramp |
| `proto_temp` | `0.10` | ProtoNCE temperature |
| `bal_n_cls` | `16` | classes per balanced batch |
| `bal_n_spc` | `8` | samples per class per balanced batch |

## `stage3` — `configs/stage3/sam_swa.yaml`

| Key | Value | Meaning |
|---|---|---|
| `epochs` | `120` | Stage-3 budget, no early stopping |
| `swa_lr` | `4.0e-5` | cyclic-LR peak |
| `cycle_len` | `8` | epochs per SAM/SWA cycle |
| `sam_rho` | `0.015` | SAM/ASAM perturbation radius $\rho$ |
| `greedy` | `true` | require a candidate SWA blend to beat the running average before accepting it |
| `aux_loss_weight` | `0.10` | fixed auxiliary-loss weight, no schedule |
| `margin_kappa_final` | `0.85` | endpoint of the multiplicative margin anneal $\kappa$ |
| `swa_warmup_cycles` | `3` | cycles discarded before the first SWA candidate is considered |
| `sam_adaptive` | `true` | select ASAM (perturbation rescaled by $\lvert\theta\rvert$) over raw SAM |

## `tracking` — `configs/tracking/*.yaml`

| Key | Default | Meaning |
|---|---|---|
| `backend` | `"console"` | `none` / `console` / `wandb` / `tensorboard` / `jsonl` / `multi` |
| `project` | `None` | W&B project name |
| `entity` | `None` | W&B entity |
| `log_dir` | `None` | TensorBoard log directory |
| `watch_model` | `false` | enable `wandb.watch` gradient histograms |
| `backends` | `[]` | child backend list, used only when `backend == "multi"` |
| `log_grad_norms` | `true` | compute and log per-branch gradient norms each optimiser step |
| `show_diagnostics` | `false` | echo per-branch diagnostics to the console backend as well as structured backends |

`TrackingConfig` carries real defaults (rather than `MISSING`) because a run with no tracking
backend configured is a valid, common case.

`jsonl` appends every tracker call — scalars, per-epoch rows, tables (per-class F1, the session
breakdown), banners, notices — as one strict-JSON object per line to
`${output_dir}/metrics.jsonl` (`tracking/jsonl_tracker.py`). `tracking=console_jsonl` is
`multi` over `[console, jsonl]`, the headless-box default the Kaggle commands use.

## `runtime` — execution knobs (`cfg.runtime`, defaulted, not YAML-required)

Full mechanics and measurements for every field below are in `06_EXECUTION_AND_HARDWARE.md`
§6.2–6.4. `RuntimeConfig` carries real defaults rather than `MISSING`, since every field here is
a throughput knob that must never change a reported metric — a config that never mentions
`runtime` is not under-specified.

| Key | Default | Meaning |
|---|---|---|
| `num_workers` | `-1` | DataLoader workers; `-1` auto-resolves per accelerator |
| `pin_memory` | `-1` | page-locked staging; `-1` auto-enables on CUDA only |
| `persistent_workers` | `-1` | keep workers alive between epochs; `-1` follows `num_workers > 0` |
| `prefetch_factor` | `4` | batches each worker runs ahead |
| `eval_num_workers` | `-1` | evaluation-loader worker count; `-1` follows `num_workers`, capped at 4 |
| `prewarm_cache` | `"auto"` | read the cube and fill map once at startup so random reads hit the page cache; `auto` → CUDA and the files ≤ ½ RAM; under DDP local rank 0 only |
| `compile` | `"auto"` | `torch.compile`; `auto` → on for CUDA, off for Metal/CPU |
| `compile_backend` | `"inductor"` | passed to `torch.compile(backend=...)` |
| `compile_mode` | `"default"` | passed to `torch.compile(mode=...)` |
| `channels_last` | `false` | NHWC/NDHWC tensors; opt-in, changes convolution reduction order |
| `allow_tf32` | `false` | TF32 matmuls on Ampere+; opt-in, a precision change |
| `amp_dtype` | `"auto"` | AMP dtype; `auto` → bf16 wherever supported, else fp16 |
| `cudnn_benchmark` | `true` | cuDNN convolution autotuning |
| `fused_optimizer` | `"auto"` | AdamW's fused multi-tensor kernel; `auto` → CUDA only |
| `decompose_conv3d` | `"auto"` | Branch C's `Conv3d` stages as `Conv2d` stacks; `auto` → Metal only |
| `checkpoint_branch_a` | `"auto"` | recompute Branch A's towers in the backward; `auto` → Metal only |
| `multi_gpu` | `"auto"` | `auto`/`ddp`/`off` DDP activation |
| `sync_batchnorm` | `true` | convert BatchNorm → SyncBatchNorm under DDP |
| `dist_timeout_s` | `1800` | NCCL/gloo rendezvous timeout |
| `checkpoint_every` | `1` | epochs between full training-state writes (`last_stage1.pth` + `last_stage1.json`, background thread); an interrupted single-stage run resumes from it; `0` → only at the stage's end |
| `empty_cache_interval` | `0` | periodic allocator sweep, epochs; `0` disables |
| `progress` | `"auto"` | per-epoch console line; `off` suppresses it |
| `diagnostics_interval` | `50` | epoch stride for the hardest-class block and branch-influence ablation |

### Runtime profiles — `configs/runtime/*.yaml`

`seednet_full256` and `quadnet_full256` list `/runtime: default` **after** `_self_`, so a
profile selected as `runtime=<name>` is merged after their inline `runtime:` block and wins.
`default.yaml` is empty (the inline block stands).

| Profile | For | What it sets |
|---|---|---|
| `kaggle_t4x2` | Kaggle "GPU T4 x2", launched by `torchrun --nproc_per_node=2` | `amp_dtype: fp16` (Turing has fp16 Tensor Cores and no bf16 path — see the file for the trade-off), `multi_gpu: ddp`, `sync_batchnorm: true`, `num_workers: 4` (→ 2 per rank) / `eval_num_workers: 2`, `prewarm_cache: on`, `checkpoint_every: 1` |

## Root — `configs/experiment/*.yaml`

Three experiments ship — see the `experiment` section below for the full table.
**`seednet_full256`** is the default (`SpectralSeedNet`, the complete 215-band reflectance cube, one stage,
`grouped`, selection on `calib`); **`quadnet_full256`** is the four-branch control on the same
protocol and input; **`quadnet_audited`** is the audited run reproduced bit-for-bit, including
the three runtime overrides that make it incomparable to the shipped defaults. `pipeline`
selects the curriculum: `single` | `stage1_only` | `stage1_stage2` | `three_stage`, the last
three being A8's arms.

| Key | Meaning |
|---|---|
| `run_name` | run identity; feeds `output_dir` |
| `output_root` | base output directory |
| `output_dir` | `${output_root}/${run_name}`, where every checkpoint/sidecar/log is written |
| `weight_decay` | 2-D+ weight decay (§4.5) |
| `grad_clip` | per-group gradient-clip norm (§4.5): 5.0 shipped, 1.0 in the audited replica, 50.0 in X1/X4 |
| `clip_partition` | **S10 P0.5 / S11 D22.** `legacy` (default) = `CLIP_GROUPS` (on SeedNet `fuse` and the aux head clip with the backbone); `model` = the model's `clip_groups()` (`fuse` joins `fusion`). They differ only on steps where a clip binds — at 5.0, ≥ 82 % of SeedNet's steps — so the default is the partition every run so far used (`04` §4.5) |
| `ema_decay` | EMA decay ceiling $d_{\max}$ (§4.5) |
| `aux_gradnorm_alpha` | GradNorm exponent for the per-branch auxiliary weights; `0.0` freezes them at the fixed $A/B{=}2\times$ vector (§4.4) |
| `tta_spatial` | dihedral TTA views (§5.1) |
| `tta_spectral` | spectral-gain TTA views (§5.1) |
| `device` | resolution strategy string (`auto`/`cuda`/`cpu`/`mps`) — YAML cannot hold a live `torch.device`, so `utils/device.py` performs the lookup at runtime |
| `seed` | global RNG seed (`utils/seed.py::set_seed`) |

## `experiment` — the composition roots

Three ship, and which one a number came from is always recorded in `results/run.json`.

| Config | Composes | Role |
|---|---|---|
| **`seednet_full256.yaml`** | `data/refl215_grouped` · `model/seed_net` · `single/one_stage` · `evaluation/held_out_once` · `tracking/console` | **The default.** What bare `python train.py` runs, and the only configuration whose numbers are the study's headline. |
| `quadnet_full256.yaml` | `data/refl215_grouped` · `model/quadnet_v4_audited` (with symmetric branch dropout, head elaborations off, `specf_tokens: 32`) · `single/one_stage` · `evaluation/held_out_once` | The four-branch control arm **on the primary protocol and the primary input**, so an ablation arm differs from the default in the architecture alone. A3/A4/A5/A8 run here. |
| `quadnet_audited.yaml` | `data/ablation/spa40_audited` · `model/quadnet_v4_audited` · `stage{1,2,3}` · `evaluation/audited_replica` | The **frozen historical replica** of the audited run, and the subject of every golden regression digest. Not an ablation arm. |

The three-stage groups (`stage1`/`stage2`/`stage3`) compose in every experiment because
`pipeline=three_stage` reaches them for A8 and the head reads `stage2.arcface_m_*` for the
optional per-class margin rule A7 switches on. Every value in the tables above is the shipped
value of the composition named in that section's heading, unless noted.

---

## Maintenance

This reference is **hand-maintained** against `config/schema.py` and the shipped
`configs/*.yaml` files. `scripts/check_config_roundtrip.py` independently verifies that every
`configs/` key resolves to exactly one dataclass field, and remains the authority on config
completeness; its `--emit-markdown` output is a different, key-mapping-oriented artifact and is
written to **`config_migration_table.md`**, not here. Pointing `--emit-markdown` at this file
would silently destroy it.
