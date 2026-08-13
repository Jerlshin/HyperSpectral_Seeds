# 5 · Inference Engine, Diagnostics and Telemetry

---

## 5.1 Test-time augmentation

`engine/tta.py::tta_predict` averages logits over $n_{\text{spatial}} + n_{\text{spectral}}$
views of each batch. The shipped configuration is `tta_spatial = 8`, `tta_spectral = 4` — **12
views**, all of which contribute.

### Spatial views — the dihedral group $D_4$

$$
\mathcal{T}_{\text{spatial}} = \Big\{\,x \mapsto \mathrm{flip}_W^{\,f}\big(\mathrm{rot90}^{\,k}(x)\big)
\;:\; k \in \{0,1,2,3\},\; f \in \{0,1\}\,\Big\}
$$

enumerated in the fixed order $(k,f) = (0,\!F), (0,\!T), (1,\!F), (1,\!T), \dots, (3,\!T)$ and
truncated to the first $n_{\text{spatial}}$. At $n_{\text{spatial}} = 8$ this is the complete
symmetry group of the square — correct for this problem, since a segmented seed patch has no
canonical orientation (§2.3 sorts regions by centroid, not by pose).

### Spectral views — gain about the *foreground* mean, re-masked (T1-1)

$$
s_j = 0.95 + \frac{0.10\,j}{n_{\text{spectral}} - 1},\quad j = 0,\dots,3
\;\;\Longrightarrow\;\;
s \in \{0.95,\; 0.98\overline{3},\; 1.01\overline{6},\; 1.05\}
$$

$$
m = \texttt{foreground\_mask}(x),
\qquad
\mu_{b,c} = \frac{\sum_{h,w} x_{b,c,h,w}\,m_{b,h,w}}{\max(\sum_{h,w}m_{b,h,w},\,1)},
\qquad
x^{(s)} = \big(\mu + (x-\mu)\,s\big)\odot m
$$

i.e. each band's spatial contrast is scaled by $s$ about its **foreground-only** mean, and the
result is explicitly **re-masked** afterwards — the test-time analogue of the train-time
`multiplicative` augmentation primitive (§2.6). The $s=1.0$ view is skipped as a duplicate of the
identity spatial view; with $n_{\text{spectral}}=4$ no scale lands exactly on $1.0$, so all four
survive.

> **This is a correctness fix (T1-1), not a cosmetic change.** An earlier version of this
> transform took the mean over the *whole* patch, including the zero background, and did not
> re-mask afterward — which moved the background off exactly-zero (breaking every downstream
> mask-by-testing-for-zero operator: `MaskedSpectralECA`, `extract_grid_spectra_multi`,
> `masked_mean_spectrum`) and shifted the true foreground mean by a factor of the foreground area
> fraction, putting the view off the training manifold. The corrected transform reads the mean
> over foreground pixels only and re-masks the result, so all four spectral views stay
> consistent with what every masked module in the model assumes about its input. The measured
> effect of the fix on the recorded reference numbers is below; `tests/unit/test_tta.py` pins the
> corrected transform.

### Ensembling rule

$$
\bar{z} = \frac{1}{|V|}\sum_{v \in V} f_{\theta}\big(\mathcal{T}_v(x)\big),
\qquad \hat{y} = \arg\max_c \bar{z}_c
$$

**Logits are averaged, not softmax probabilities.** With ArcFace's $s = 48$ scaling, softmax
averaging would be dominated by whichever view happened to be most confident; logit averaging
keeps every view's contribution linear in its margin. TTA forward passes run under
`autocast(enabled=False)` — forced fp32, matching `engine/evaluate.py`'s unconditional fp32 eval,
so neither a reported metric nor the TTA-vs-no-TTA comparison is confounded by the caller's AMP
state. Masks are dihedral-transformed identically to the patch before either the spectral-view
mean or the model sees them; morphometrics pass through every view unchanged, since none of the
twelve transforms alter shape.

### Final evaluation protocol

`engine/stages/final_eval.py` runs the test loader twice — no-TTA then 12-view TTA — on the
**EMA shadow** of the checkpoint `_pick_best_checkpoint` selected, reports macro-F1, weighted
F1 and accuracy for each, prints a full `classification_report`, logs a bottom-10 hardest-class
table, and writes three arrays to `cfg.output_dir`:

| Artifact | Shape | Contents |
|---|---|---|
| `test_preds_noTTA.npy` | $(1294,)$ | $\arg\max$ over single-view logits |
| `test_preds_TTA.npy` | $(1294,)$ | $\arg\max$ over 12-view averaged logits |
| `test_targets.npy` | $(1294,)$ | ground truth for the test split |

Any reported metric is therefore recomputable from disk without re-running inference — which
is exactly what `test_recorded_test_predictions_match_their_reported_metrics` does.

### Recorded outcome (pre-Tier-3 reference checkpoint, re-scored through the corrected transform)

| Protocol | Macro F1 | Weighted F1 | Accuracy | $\Delta$ Macro F1 |
|---|---:|---:|---:|---:|
| Single view | 0.8770 | 0.8776 | 0.8779 | — |
| 12-view TTA (corrected spectral transform, T1-1) | **0.8889** | — | — | **+0.0119** |

> **Provenance.** The `outputs/output_v12_spa40/` run was produced under the pre-Tier-3
> architecture, and its checkpoints do not load into the current schema-v3 model (§3.8) — there is
> currently no way to regenerate a TTA number against the architecture this document otherwise
> describes. `outputs/` is git-ignored and the directory is not present in this tree (§5.4).
> The original console log for this run reported **0.8933** macro-F1 under TTA; that number was
> produced by the *pre-T1-1* spectral view (whole-patch mean, no re-mask — the bug described
> above), which additionally left every masked module reading a filled frame rather than a
> foreground-aware one. Re-scoring the identical checkpoint's recorded logits through the
> corrected transform gives **0.8889**, $+0.0119$ over the single-view result rather than
> $+0.0163$; the single-view row itself is unaffected by the fix. Paired bootstrap on the
> before/after TTA difference: $-0.0046$, 95% CI $[-0.0151,+0.0056]$, $p=0.40$ — within noise, but
> 0.8889 is what a fresh evaluation against this checkpoint under the current code actually
> produces, and is the number to cite.

Per-class outcome recorded against this checkpoint under TTA (as originally logged, precise
per-class breakdown not re-run against the corrected transform): 23 of 90 classes at
$F_1=1.000$, none below $0.50$, hardest five = 49 (0.519), 52 (0.533), 41 (0.538), 51 (0.629),
37 (0.640).

---

## 5.2 Diagnostic metrics

All three diagnostics are deliberately *wrappers over quantities the training loop already
computes*, not new statistics — `tests/unit/test_diagnostics.py` enforces that.

### Leave-one-branch-out influence (KL divergence)

`compute_branch_influence` ablates each branch in turn via the `branch_mask` argument
(§3.3) and measures how far the prediction moves:

$$
I_b \;=\; \frac{1}{|\mathcal{B}|}\sum_{\text{batch}}
D_{\mathrm{KL}}\Big(P_{\text{full}} \,\Big\|\, P_{\setminus b}\Big)
\;=\; \frac{1}{|\mathcal{B}|}\sum_{\text{batch}}\frac{1}{B}\sum_{i}\sum_{c}
P_{\text{full}}(c \mid x_i)\,\log\frac{P_{\text{full}}(c\mid x_i)}{P_{\setminus b}(c\mid x_i)}
$$

with $P_{\setminus b} = \mathrm{softmax}\big(f_\theta(x;\, \text{mask}_b = 0)\big)$ clamped
below at $10^{-10}$, and reported as percentages:

$$
\hat{I}_b = 100 \cdot \frac{I_b}{\max\big(\sum_{b'} I_{b'},\; 10^{-8}\big)},
\qquad \sum_b \hat{I}_b = 100
$$

An empty loader returns all zeros. **Cost is $5\,|\mathcal{B}|$ forward passes** (one full plus
four ablated per batch), which is why every caller passes a small `max_batches` — 3 from
`compute_class_difficulty` — and why it runs only on a checkpoint improvement. The four values
are logged as `influence/branch_{a,b,c,d}` and appear in the per-stage difficulty message.

Because the mask is applied to the *fused* path only, $\hat{I}_b$ measures each branch's
contribution through `CrossModalInteraction`, not its standalone discriminative power (which
the auxiliary heads measure separately).

### Per-class difficulty and the hardest-$K$ table

`compute_class_difficulty(cfg, ema_shadow, val_ldr, device, label, tracker, step)` is the
single entry point that produces every difficulty-derived quantity in the system:

1. $F_1^{(c)}$ for all 90 classes on the **EMA shadow** (`evaluate_per_class`).
2. CDWS weights $w_c$ from those $F_1$ (§4.2).
3. Macro $F_1 = \frac{1}{C}\sum_c F_1^{(c)}$ and the hard-class count
   $\big|\{c : F_1^{(c)} < 0.50\}\big|$.
4. Branch influence over 3 batches.
5. A bottom-$K$ table.

`hardest_classes_report(class_f1, k=10)` sorts by the pair $(F_1^{(c)},\, c)$ ascending, so ties
break on class id and the table is **stable across runs and diffable between checkpoints**:

$$
\mathcal{H}_K = \text{first } K \text{ of } \operatorname{sort}\big\{(F_1^{(c)}, c)\big\}_{c=0}^{89},
\qquad \text{rows } \{\texttt{rank}, \texttt{class}, \texttt{f1}\}
$$

The same $F_1$ dict feeds three consumers — the ArcFace per-class margins $M(y_i)$, the CDWS
sampler weights, and the Phase-3 oversampler — so difficulty is measured once per improvement
and reused, never recomputed inconsistently.

### Per-branch gradient norms

`branch_grad_norm_tensors` groups `named_parameters()` by six prefixes and returns one L2 norm
per group:

$$
g_{\text{prefix}} = \sqrt{\sum_{p\,:\,\text{name} \prec \text{prefix}} \big\|\nabla_p \mathcal{L}\big\|_2^2}
$$

$$
\text{prefixes} = \{\texttt{branch\_a}, \texttt{branch\_b}, \texttt{branch\_c}, \texttt{branch\_d},
\texttt{cross\_interaction}, \texttt{arcface\_head}\}
$$

This is the *same* quantity `clip_grad_norm_` returns, split by owner — the groups sum in
quadrature to the model total. Contract details that matter:

- **Sampled before the clip** (and after `scaler.unscale_`), so the numbers describe the true
  pre-clip gradient. Called after clipping, they would describe the clipped one.
- **A prefix with no gradient is omitted, not reported as $0.0$** — so a frozen head (Stage 1
  freezes `arcface_head`) stays distinguishable from one that trains but is flat.
- **Nothing crosses to the host per step.** Norms accumulate as device tensors; the whole epoch
  resolves in one stacked `.cpu()` transfer, avoiding a synchronisation per parameter per step.
  `branch_grad_norms` is the one-shot variant that does convert.

The diagnostic exists because gradient collapse in the spectral branches A/B is a real failure
mode of this architecture — the same one the auxiliary weighting (§4.4) counteracts. It is
gated on `tracking.log_grad_norms` (default `true`), and off entirely when no tracker is passed.

These same per-branch norms are now also a **consumer**, not just a logged quantity: whenever
`cfg.aux_gradnorm_alpha != 0`, the epoch-mean of this diagnostic drives the GradNorm auxiliary-
weight update (§4.4) once per epoch. A second, narrower diagnostic exists for Stage 3 only:
`sam/grad_cos` — the cosine similarity between the SAM ascent and descent gradients — sampled at
$\sim$32 evenly-spaced steps per epoch (`GRAD_COS_SAMPLES`) rather than every step, since it
requires a full flattened-gradient copy at 5.19 M parameters. It is computed via three dot
products over max-normalised inputs rather than `F.cosine_similarity`, because the latter's
split numerator/denominator reductions can disagree by up to $5\times10^{-3}$ over that many
elements — enough to report $1.005$ for a vector against itself.

### Per-branch auxiliary losses

`_compute_aux_loss(..., return_components=True)` returns the four *weighted* terms
$\omega_b\mathcal{L}_b$ alongside their sum. The summed total is bit-identical whether or not
components are requested — the flag must not perturb the number the training loop already
used, which is asserted directly.

### Accumulation cost model

Both epoch loops accumulate diagnostics as on-device tensors and resolve them **once per
epoch** in a single `torch.stack(...).cpu()`:

$$
\overline{\text{loss/branch}_b} = \frac{1}{n_{\text{aux}}}\sum_{\text{steps}} \omega_b\mathcal{L}_b,
\qquad
\overline{\text{grad\_norm/}\cdot} = \frac{1}{n_{\text{clip}}}\sum_{\text{opt steps}} g_{\cdot}
$$

so the per-step cost is a handful of adds. With `tracker=None` no diagnostic is accumulated at
all — the engine's default is `None` rather than an implicit `NullTracker`, precisely so the
no-tracking path carries zero overhead.

---

## 5.3 Telemetry

### The `ExperimentTracker` protocol

`tracking/base.py` defines a `runtime_checkable` `Protocol`, not a base class — backends
satisfy it structurally. It splits into two channels because the two audiences want different
things from the same run:

| Method | Channel | Purpose |
|---|---|---|
| `log_scalar(tag, value, step)` | machine | one metric, plottable over `step` |
| `log_scalars(tags, step)` | machine | a group sharing one `step` |
| `log_table(tag, rows, step)` | machine + human | tabular records (hardest classes, config) |
| `log_hyperparams(cfg)` | machine | the composed config, once per run |
| `watch(model)` | machine | optional gradient/parameter histograms |
| `close()` | lifecycle | must be idempotent |
| `banner(title, lines)` | human | stage header block |
| `log_message(text, level)` | human | one-line notice: `plain` / `info` / `warn` / `success` |
| `log_row(tag, cells, step)` | human | one pre-formatted row of a running table |

`log_row` takes **pre-formatted strings**, not numbers: the call site owns formatting
(`.4f` for a loss, `.1%` for an accuracy), and a backend seeing a `tag` for the first time
emits a header before the row.

### Backends

| `tracking.backend` | Class | Human channel | Machine channel |
|---|---|---|---|
| `none` | `NullTracker` | no-op | no-op |
| `console` *(default)* | `ConsoleTracker` | full `rich` rendering | quiet unless `show_diagnostics` |
| `wandb` | `WandbTracker` | **inert** | `wandb.log`, `wandb.Table`, `wandb.watch` |
| `tensorboard` | `TensorBoardTracker` | **inert** | `add_scalar`, Markdown `add_text` tables |
| `multi` | `MultiTracker` | fan-out | fan-out |

Both remote backends leave the human channel inert by design — a banner has no useful W&B
representation, and the numbers behind it already arrive as plottable series. To keep a
readable terminal *and* stream remotely:

```
python train.py tracking.backend=multi tracking.backends=[console,wandb]
```

`MultiTracker.close()` closes every child even if one raises (re-raising the first error), so a
failing backend cannot strand another's file handle or leave a W&B run unfinished. Nesting
`multi` inside `backends` raises, as does an empty list. Both optional SDKs are imported
**inside** their constructor, so `import spectralquadnet` never requires `wandb` or
`tensorboard`.

### Append-only console rendering

`ConsoleTracker` emits **one line per epoch**, written once and never redrawn:

```
[Stage 1 | Ep 181/400]  Time: 00:12:45  ETA: 00:41:12  dt: 42.1s  Loss: 15.2508  Tr: 61.2%  F1 live/ema: 0.771/0.685  Acc live/ema: 78.1%/72.0%  Best: 0.780  LR: 3.00e-04  LS: 0.084  auxW: 0.42  τ: 0.35  Ph: P2  m: 0.00  ckpt ✓
```

Deliberate design points:

1. **No live region, on any stdout.** There is no `rich.progress` bar, no `\r`, no `\033[2K`
   and no cursor motion. A redrawing bar is legible only where the console is *interactive*;
   an SSH session piped to a file gets thousands of overwritten half-lines, and a Kaggle/Colab
   cell reports a TTY it cannot drive, so `rich` appends every frame instead of repainting one.
   Rendering per environment meant three renderings of one run and three sets of bugs.
   `runtime.progress` therefore no longer selects a rendering — only `off` (suppress the epoch
   line) is distinguishable; `bar`/`rows` are accepted as legacy spellings and both render the
   line.
2. **The prefix and the clocks belong to the tracker, not the caller.** `progress_start(tag,
   total, description)` opens a *span* — the stage's label and epoch budget — and `log_row`
   builds `[Stage 1 | Ep 181/400]`, `Time` (elapsed since the tracker was built, continuous
   across stages) and `ETA` (extrapolated from the current span alone) from it. `progress_stop`
   reports where the stage actually ended, which for an early-stopped stage is the number the
   old bar threw away by completing itself to its total.
3. **Cells are `key: value`, markers are `key value`, and empty cells are absent.** `ckpt ✓`
   appears only on epochs that saved one — that is what makes `grep ckpt training.log` the list
   of improvements — and `stale: 3/50` (epochs without improvement, against `stage1.patience`)
   is its complement. Column widths are remembered per span and grow monotonically, so the
   columns line up down the log without any value being truncated into a frozen width.
4. **Blocks are plain text, not `rich` renderables.** Banners and the hardest-class table are
   drawn as aligned text whose width comes from the data; a `Panel`/`Table` is measured against a
   terminal width that is 80 whenever stdout is not a TTY, which is how one diagnostic came out
   box-drawn in a terminal, squeezed in a pipe and HTML-rendered in a notebook. Numeric columns
   share a decimal count, so the points line up. Glyphs (`✓ ★ κ τ ─`) degrade to ASCII on a
   stream whose encoding cannot carry them.
5. **Every line is mirrored into `training.log`** via the `spectralquadnet.console` logger,
   which `train.py` points at the file handler alone (`propagate=False`), so the file is the
   record of the run and the terminal still gets exactly one copy. Before this the file held
   only crashes: the tracker wrote to stdout, and the file handler only ever saw `logging`
   records.
6. **Scalars are quiet by default** — the epoch summary arrives via `log_row`; per-branch
   diagnostics arrive via `log_scalars` and are meant to be read as curves. Rendering both
   would print every epoch twice. `tracking.show_diagnostics=true` echoes them.

`watch` has no console equivalent of gradient histograms, so it prints the trainable parameter
count instead (`Params : 5.19M`).

Third-party warning noise is filtered in `utils/warning_filters.py` — entry by entry, by category
and message, each with a reason — and installed as an *import side effect* so it is in place
before `import torch` emits pynvml's deprecation. Warnings that survive the filters are routed
through `logging`, so they arrive as their own formatted line in both sinks instead of splicing
themselves into an epoch line from `stderr`.

### Scalar key catalogue

Every key any backend receives, by producer:

| Producer | Keys |
|---|---|
| `train_one_epoch` / `_sam` | `loss/branch_{a,b,c,d}`, `grad_norm/{branch_a,branch_b,branch_c,branch_d,cross_interaction,arcface_head}`, `grad_norm/preclip_{head,fusion,backbone}`, `train/{steps,skipped_batches,epoch_s}`; `sam/grad_cos` from the SAM loop only |
| Stage 1 | `train/{loss,acc}`, `val/{f1_live,acc_live,f1_ema,acc_ema,f1_best}`, `sched/{lr,label_smooth,aux_weight,subcentre_tau,phase}` |
| Stage 2 | `train/{loss,acc}`, `val/{…}`, `sched/{head_lr,back_lr,arcface_margin,supcon_weight,proto_weight,subcentre_tau}`, `margin/{mean,min,max}` |
| Stage 3 | `train/{loss,acc}`, `val/{f1_live,acc_live,f1_ema,acc_ema,f1_best}`, `sched/{lr,margin_kappa,arcface_margin,supcon_weight,proto_weight}`, `swa/{n_snapshots,n_rejected,f1_running}`, then once at stage end `swa/{f1,acc}` and `ema/{f1,acc}` |
| `compute_class_difficulty` | `diag/{macro_f1,hard_classes}`, `influence/branch_{a,b,c,d}` |
| `final_eval` | `test/{f1_macro,f1_weighted,acc}`, `test_tta/{f1_macro,f1_weighted,acc}` |
| Tables | `hardest_classes/{Phase2→3, S1, S2, test_tta}`, `config` |

Stage 3's `sched/{supcon_weight,proto_weight}` are the hardcoded `0.02`/`0.01` constants of §4.4,
logged so a curve exists for them rather than because they vary; `sched/arcface_margin` there is
the *mean* of the scaled per-class vector, not a scalar the stage optimises against.

`flatten_hyperparams` converts the nested composed config to dotted keys
(`stage1.max_lr → 0.0005`) for W&B/TensorBoard, stringifying lists since `hparams` accepts only
scalars.

---

## 5.4 The recorded reference run

The `outputs/output_v12_spa40/` run is a complete three-stage run — three checkpoints, three JSON
sidecars, three prediction arrays and a baseline console log.

> **It is not in this repository.** `outputs/` (and `*.pth`) are git-ignored, so the run directory
> was never tracked, and it is not present in the working tree either. The table below is the
> transcribed record of that run, not something a reader can re-derive from a checkout; it is kept
> because it is the last complete three-stage run on file. Everything else in these documents *is*
> derived from tracked sources.

| Stage | Epoch saved | val Macro F1 | val Acc | Extra metadata |
|---|---:|---:|---:|---|
| 1 | 488 / 600 | 0.8877 | 0.8872 | `class_f1`, `cdws_weights`, `phase3_class_f1` (90 entries each), `arcface_init_done=false` |
| 2 | 50 / 150 | 0.8867 | 0.8864 | `class_f1`, `cdws_weights`, `s2_val_f1` |
| 3 | 120 / 120 | 0.8745 | 0.8748 | `swa_n_snapshots=15`, `swa_n_rejected=0`, `note` |

Checkpoint selection ranks Stage 1 highest, so the reported test numbers come from
`best_stage1.pth`. Stage 3's own sidecar records the outcome: *"val_f1 did not beat Stage 2;
Stage 2 ckpt preferred for eval."* The `488 / 600` denominator is that run's `stage1.epochs`; the
shipped config now runs 400 (§4.1), which is why the recorded Stage-1 best sits past the current
budget.

> **Schema provenance.** This run predates both Tier 2's unified head (HD-1/T2-10) and Tier 3's
> branch redesign — its checkpoints carry a `linear_head` this model no longer has, and
> `load_ckpt` refuses to load any of them today (`SchemaTooOldError`, §3.8), since Tier 3 changed
> what three of the four branches *consume*, not merely their parameter counts. The metrics
> table above is preserved as the historical record of the pre-Tier-3 pipeline; it is not
> reproducible against the current model, and cannot be until a fresh three-stage run completes.

### Auto-resume semantics

`latest_completed_stage` probes $3 \to 2 \to 1$ and calls a stage complete only when **both**
`best_stage{n}.pth` and `stage{n}_meta.json` exist, so a crash between the two writes replays
the stage rather than skipping it. Pointing `output_dir` at a directory with all three present
runs final evaluation only:

```
python train.py output_dir=outputs/output_v12_spa40
```

### Checkpoint bundle schema

$$
\texttt{bundle} = \{\texttt{epoch}, \texttt{stage}, \texttt{model}, \texttt{ema},
\texttt{val\_f1}, \texttt{val\_acc}, \texttt{use\_arcface}, \texttt{schema\_version}\} \cup \texttt{metadata}
$$

`schema_version` is currently **3** (§3.8) — bundles missing the key (every checkpoint written
before Tier 2) are read as version 1. `use_arcface` is written as a constant `True` in every
fresh bundle, kept only for legacy readers (`scripts/phase0_*.py`, the resume banner); nothing
internal branches on it since HD-1 left exactly one head to select. `metadata` includes
`best_source ∈ {"live","ema","swa"}`, recording which weight set actually produced the recorded
`val_f1` — bundles predating this field (pre-Tier-1) default to `"ema"`, matching their actual
production behaviour.

The JSON sidecar is the bundle minus `model`/`ema`, filtered to JSON-serialisable values.
`load_stage_meta` re-integerises string keys, because JSON stringifies them and
`class_f1`/`cdws_weights` must come back as `{int: float}` for the samplers and CDWS to index.
`_pick_best_checkpoint` ranks by sidecar `val_f1`, falling back to `val_acc`, then to the
`.pth` bundle, then to $0.0$; if no checkpoint exists on disk at all it falls back to the last
path argument (Stage 3's).

Full execution-engine detail — `train.py`'s orchestration, runtime performance knobs, and
distributed (DDP) training — is in `06_EXECUTION_AND_HARDWARE.md`.

---

## 5.5 Ablation surface

The system exposes ablations through Hydra overrides — no code changes required. Nothing below
has been executed in this repository; the table documents the *lever*, not a result.

| Ablation | Override | Mechanism |
|---|---|---|
| Curriculum length | `stage1.epochs=…` `stage2.epochs=…` `stage3.epochs=…` | shorten or skip a stage |
| Phase split | `stage1.phase1_frac=…` `stage1.phase2_frac=…` | move the augmentation boundaries |
| Deep supervision off | `stage1.aux_loss_weight_init=0 stage1.aux_loss_weight_final=0` | zeroes $w_{\text{aux}}$ (the heads still run) |
| GradNorm aux weighting off | `aux_gradnorm_alpha=0` | root-level; freezes the per-branch weights at the fixed $A/B{=}2\times$ vector, all three stages |
| Contrastive off | `stage1.p3_supcon_weight=0 stage1.p3_proto_weight=0` / `stage2.{supcon,proto}_weight=0` | zero-weights the terms; the modules are still *passed* (the choice is gated on phase, not weight), so both losses are still computed and AMP stays disabled — the classification term's weight rises to 1.0 |
| Sub-centre count | `model.subcenter_K=1` | reduces ArcFace to the single-prototype form |
| Sub-centre balance off | `model.subcenter_balance_weight=0` | drops $\mathcal{L}_{\text{balance}}$; sub-centres can die again under hard assignment |
| Signed margin rule off | `stage2.arcface_m_delta=0` | collapses $M(c)$ to the constant $m_{\text{base}}$ for every class |
| Pairwise confusion margin off | `stage2.pairwise_margin_delta=0` | removes the $\Omega$-scaled non-target penalty; the target-column margin is unaffected |
| CDWS off | `stage2.cdws_max_weight=1.0` | flat class-selection probabilities |
| Hard-class oversampling off | `stage1.p3_oversample=false` | Phase 3 falls back to a plain shuffled loader |
| SAM strength | `stage3.sam_rho=…` | $\rho = 0$ degenerates to plain AdamW with an extra forward/backward |
| ASAM off | `stage3.sam_adaptive=false` | raw SAM — perturbation not rescaled by $\lvert\theta\rvert$ |
| Margin annealing off | `stage3.margin_kappa_final=1.0` | freezes Stage 2's calibrated margin vector for the whole of Stage 3 |
| SWA transient rejection off | `stage3.swa_warmup_cycles=0` | the first post-Adam-warmup cycle becomes a candidate immediately |
| Greedy SWA off | `stage3.greedy=false` | every cycle-end candidate is accepted unconditionally |
| Split protocol | `data=spa40_90class_pfix` | scan-disjoint (`grouped`) split + calibration split, instead of the leaky `stratified` reference protocol (§2.8) |
| TTA views | `tta_spatial=…` `tta_spectral=…` | 1–8 dihedral views, 0–$n$ spectral gains |
| Multi-GPU | `runtime.multi_gpu=ddp` (with `torchrun`) | see `06_EXECUTION_AND_HARDWARE.md` |
| Sweeps | `python train.py -m stage1.max_lr=1e-4,5e-4,1e-3` | Hydra multirun into `outputs/multirun/` |

Three caveats for anyone running these:

- **Branch ablation is not a config lever.** The `branch_mask` argument exists for the influence
  diagnostic, and the per-branch drop rates ($0.15,\,0.15,\,0,\,0.15$ at the shipped
  `branch_drop_prob = 0.20`) are a hardcoded profile in the forward pass (§3.3); removing a
  branch entirely requires a code change.
- **Same-class CutMix has no single on/off flag.** Its firing probability is baked into each
  augmentation profile, not exposed as a config field; `data.cutmix_bands`/`cutmix_spatial`
  control only the window/region size. Training with `train_aug="none"` skips it along with
  every other augmentation (§2.6).
- **Runs are not bit-reproducible** (§1.3): `cudnn.benchmark=True` and (on a single process)
  unseeded sampler RNGs. A single-seed ablation delta smaller than run-to-run variance is not
  evidence. What *is* pinned is weight initialisation, every schedule value, and one fixed-seed
  forward + backward step.
