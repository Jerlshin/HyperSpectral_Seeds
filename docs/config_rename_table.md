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
| `s1_epochs` | `stage1.epochs` | `600` | `configs/stage1/progressive_3phase.yaml` |
| `s1_phase1_frac` | `stage1.phase1_frac` | `0.3` | `configs/stage1/progressive_3phase.yaml` |
| `s1_phase2_frac` | `stage1.phase2_frac` | `0.38` | `configs/stage1/progressive_3phase.yaml` |
| `s1_batch` | `stage1.batch` | `128` | `configs/stage1/progressive_3phase.yaml` |
| `s1_max_lr` | `stage1.max_lr` | `0.0005` | `configs/stage1/progressive_3phase.yaml` |
| `s1_mid_lr` | `stage1.mid_lr` | `0.00025` | `configs/stage1/progressive_3phase.yaml` |
| `s1_min_lr` | `stage1.min_lr` | `5e-06` | `configs/stage1/progressive_3phase.yaml` |
| `s1_dropout` | `stage1.dropout` | `0.15` | `configs/stage1/progressive_3phase.yaml` |
| `s1_mixup` | `stage1.mixup` | `0.35` | `configs/stage1/progressive_3phase.yaml` |
| `s1_patience` | `stage1.patience` | `160` | `configs/stage1/progressive_3phase.yaml` |
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
| `s2_patience` | `stage2.patience` | `80` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_s` | `stage2.arcface_s` | `48.0` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_m` | `stage2.arcface_m` | `0.35` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_m0` | `stage2.arcface_m0` | `0.18` | `configs/stage2/arcface_supcon.yaml` |
| `s2_arcface_m_delta` | `stage2.arcface_m_delta` | `0.1` | `configs/stage2/arcface_supcon.yaml` |
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
| `specf_dim` | `model.specf_dim` | `256` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_heads` | `model.specf_heads` | `8` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_layers` | `model.specf_layers` | `4` | `configs/model/spectral_quadnet_v4.yaml` |
| `specf_drop` | `model.specf_drop` | `0.15` | `configs/model/spectral_quadnet_v4.yaml` |
| `fusion_heads` | `model.fusion_heads` | `4` | `configs/model/spectral_quadnet_v4.yaml` |
| `fusion_drop` | `model.fusion_drop` | `0.1` | `configs/model/spectral_quadnet_v4.yaml` |
| `device` | `device` ⚠️ | `'auto'` | `configs/experiment/output_v12_spa40.yaml` |
| `seed` | `seed` | `42` | `configs/experiment/output_v12_spa40.yaml` |

⚠️ = value intentionally differs from the pre-refactor constant:

- **`output_dir`** — §4.3 — hardcoded absolute machine-specific path replaced by ${output_root}/${run_name}; points at the Phase 1 relocation target. Pre-refactor value: `'/Users/jerlshin/FieldOfInterest/ResearchWork/HSI_RGB_seeds/Code/HSI_modality_training/output_v12_SPA40'`.
- **`device`** — §4.3 — YAML cannot hold a torch.device object, so the config carries the resolution strategy ("auto") and utils/device.py performs the torch.device("cuda" if torch.cuda.is_available() else "cpu") lookup. Pre-refactor value: `"<expr> torch.device('cuda' if torch.cuda.is_available() else 'cpu')"`.

## Net-new fields (no `CONFIG` ancestor)

- **`run_name`** — §4.3 — run identity; feeds output_dir instead of hardcoding a path.
- **`output_root`** — §4.3 — output_dir = ${output_root}/${run_name}.
- **`tracking.*`** — §4.1 — experiment tracking is additive; the monolith used bare print().
