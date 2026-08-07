# 4 · Curriculum, Objectives and Optimisation

Three stages run in sequence, each writing `best_stage{n}.pth` + `stage{n}_meta.json`. Each
stage module (`engine/stages/`) is **orchestration only**: every unit of work is a call into
`engine/`, `losses/` or `optim/`, which is what makes the schedules independently testable.

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Module | `stage1_progressive.py` | `stage2_arcface.py` | `stage3_sam_swa.py` |
| Epochs (max) | 600 | 150 | 120 (no early stop) |
| Head | `linear_head` | `arcface_head` | `arcface_head` |
| Optimiser | AdamW | AdamW (2 LRs) | SAM(AdamW) |
| LR schedule | phase-aware (3 phases) | warmup + SGDR | cyclic cosine, 8-epoch |
| Batch | 128 (shuffled / oversampled) | $16\times8 = 128$ balanced | $16\times8 = 128$ balanced |
| Patience (macro-F1) | 160 | 80 | — |
| Mixup | Phases 1–2 only | ✗ (raises if combined with ArcFace) | ✗ |
| AMP | Phases 1–2 | ✗ | ✗ (SAM is fp32) |
| Recorded best val F1 | 0.8877 (ep 488) | 0.8867 (ep 50) | 0.8745 (SWA) |

---

## 4.1 Stage 1 — three-phase progressive augmentation

### Phase boundaries

$$
p_1 = \lfloor E \cdot 0.30 \rfloor = 180,
\qquad
p_2 = \lfloor E \cdot (0.30 + 0.38) \rfloor = 408,
\qquad E = 600
$$

| Phase | Epochs | Augmentation | Loss | Mixup | Sampler |
|---|---|---|---|---|---|
| 1 · explore | 1 – 180 | `heavy` | CE + label smoothing | ✓ ($\alpha = 0.35$) | shuffled |
| 2 · consolidate | 181 – 408 | `medium` | CE + label smoothing | ✓ | shuffled |
| 3 · discriminate | 409 – 600 | `very_light` | Focal + SupCon + ProtoNCE | ✗ | hard-class oversampled |

At each boundary the EMA shadow is hard-reset (`ema_reinit_phases = true`) because the loss
landscape shifts enough that continuing the old average would anchor to a stale optimum.
Entering Phase 3, dropout is additionally raised $0.15 \to 0.25$ (`p3_dropout`) to counter
memorisation under lighter augmentation and repeated hard samples.

The Phase 2 → 3 transition is where the curriculum becomes *data-adaptive*: per-class $F_1$ is
measured on the EMA shadow (`compute_class_difficulty`) and drives the
`HardClassOversampledSampler` for the remainder of the stage (§2.7).

### Phase-aware learning rate

`phase_aware_lr` returns a **multiplier** on `stage1.max_lr` $= 5\times10^{-4}$; the callable
takes a 0-based index and converts internally to the 1-based epoch $e$.

$$
\eta(e)/\eta_{\max} =
\begin{cases}
\dfrac{e}{5}, & e \le 5 \quad\text{(linear warm-up)}\\[10pt]
0.6 + 0.2\Big(1 + \cos\big(\pi \tfrac{e-5}{p_1-5}\big)\Big), & 5 < e \le p_1 \quad (1.0 \to 0.6)\\[10pt]
0.2 + 0.2\Big(1 + \cos\big(\pi \tfrac{e-p_1}{p_2-p_1}\big)\Big), & p_1 < e \le p_2 \quad (0.6 \to 0.2)\\[10pt]
\rho_{\min} + (\rho_{\text{mid}} - \rho_{\min})\cdot\tfrac{1}{2}\Big(1+\cos\big(\pi\tfrac{(e - p_2 - 1) \bmod 30}{30}\big)\Big), & e > p_2
\end{cases}
$$

with $\rho_{\min} = \eta_{\min}/\eta_{\max} = 0.01$ and
$\rho_{\text{mid}} = \eta_{\text{mid}}/\eta_{\max} = 0.5$. In absolute terms: warm-up to
$5\times10^{-4}$, decay to $3\times10^{-4}$ by epoch 180, to $1\times10^{-4}$ by epoch 408,
then 30-epoch cosine restarts oscillating between $5\times10^{-6}$ and $2.5\times10^{-4}$.
The periodic Phase-3 peak exists to escape the overfit basin that light augmentation alone
cannot prevent late in training. `tests/unit/test_schedulers.py` checks all 600 values.

### Label-smoothing decay

Linear across the whole stage, independent of phase:

$$
\varepsilon(e) = \varepsilon_{\text{hi}}(1 - t) + \varepsilon_{\text{lo}}\,t,
\qquad t = \frac{e-1}{E-1},
\qquad \varepsilon_{\text{hi}} = 0.10 \to \varepsilon_{\text{lo}} = 0.04
$$

### Checkpointing

Every epoch scores both the live model and the EMA shadow; on
$\max(F_1^{\text{live}}, F_1^{\text{ema}}) > F_1^{\text{best}}$ the stage recomputes class
difficulty and saves a bundle carrying `class_f1`, `cdws_weights`, `arcface_init_done=False`
and `phase3_class_f1`. Patience is 160 epochs without improvement; the recorded run's best
landed at epoch 488.

---

## 4.2 Stage 2 — sub-centre ArcFace and metric learning

### Entry transitions

1. `set_dropout(0.10)`, `use_arcface(True)`, freeze `linear_head`, unfreeze `arcface_head`.
2. EMA hard-reset from the live model, shadow also switched to ArcFace.
3. If Stage 1's sidecar has `arcface_init_done = False`, `train.py` bootstraps
   $\mathbf{W}_{c,k}$ from the trained linear head (§3.5) — on both the model and the shadow.
4. Per-class margins initialised from Stage 1's `class_f1` via `update_margins_from_f1`.
5. Batch composition switches to `ClassBalancedBatchSampler` with Stage 1's CDWS weights.

### Angular-margin warm-up $m(t)$

A cosine ramp from $m_0 = 0.18$ to $m = 0.35$ over the first `margin_warmup_ep` $= 20$ epochs:

$$
m(t) = m_0 + (m - m_0)\cdot\tfrac{1}{2}\Big(1 - \cos\big(\pi \tfrac{t}{t_{\text{warm}}}\big)\Big),
\qquad t = e - 1 < 20
$$

with $m(t) = m$ for $t \ge t_{\text{warm}}$. **The handover matters:** while warming up, the
scalar $m(t)$ is passed as `global_m` and overrides the margins buffer for every sample; once
$e - 1 \ge 20$ the call site passes `arc_m=None`, and the head switches to its **per-class
adaptive margins** $M(y_i) \in [0.35, 0.45]$. So the schedule is: one global margin, ramped;
then per-class margins, recalibrated from validation $F_1$ at each checkpoint improvement.

### SGDR

`sgdr_scheduler` returns a multiplier with linear warm-up then cosine restarts whose lengths
grow geometrically. With $T_0 = 25$, $T_{\text{mult}} = 2$, $t_{\text{warm}} = 3$ and
$\eta_{\min}^{\text{frac}} = \eta_{\min}/\eta_{\text{head}} = 4\times10^{-3}$:

$$
\eta(e)/\eta_0 =
\begin{cases}
\max\!\big(e / t_{\text{warm}},\; 10^{-6}\big), & e < t_{\text{warm}}\\[6pt]
\eta_{\min}^{\text{frac}} + \tfrac{1}{2}\big(1 - \eta_{\min}^{\text{frac}}\big)\big(1 + \cos(\pi\, r)\big), & e \ge t_{\text{warm}}
\end{cases}
$$

where $r = (t - T_{\text{elapsed}})/T_{\text{cur}}$ is the position within the current cycle,
found by walking $T_{\text{cur}} \leftarrow T_0, 2T_0, 4T_0, \dots$ Restarts therefore land at
epochs $\mathbf{28}$ and $\mathbf{78}$, which the stage prints as `↻R1` / `↻R2`.

Both parameter groups share the multiplier but not the base LR:
head $2.5\times10^{-4}$, backbone $7\times10^{-5}$ (§4.5).

### Contrastive ramp

SupCon and ProtoNCE weights are linearly ramped in over the first 10 epochs so the margin
loss establishes itself first:

$$
\lambda_{\text{sc}}(e) = 0.40 \cdot \min(1,\, e/10),
\qquad
\lambda_{\text{pt}}(e) = 0.18 \cdot \min(1,\, e/10)
$$

### Class-Difficulty-Weighted Sampling (CDWS)

`losses/cdws.py::build_cdws_weights` converts per-class validation $F_1$ into inverse-difficulty
sampling weights, mean-normalised so no class dominates:

$$
\tilde{w}_c = \min\!\Big(\frac{1}{F_1^{(c)} + \epsilon},\; W_{\max}\Big),
\qquad
w_c = \frac{\tilde{w}_c}{\frac{1}{C}\sum_{c'} \tilde{w}_{c'}}
$$

with $\epsilon = 0.05$ (`cdws_eps`), $W_{\max} = 3.0$ (`cdws_max_weight`), and $F_1^{(c)} = 0$
for any class missing from the dict. A class at $F_1 = 1.0$ gets $\tilde{w} \approx 0.95$; a
class at $F_1 = 0.3$ gets $\approx 2.86$; anything below $F_1 \approx 0.283$ saturates at
$W_{\max}$. Compared with the Phase-3 oversampler (§2.7) this is *gentler* — no exponent, a
$3\times$ rather than $7\times$ cap — because it composes with class-balanced batches rather
than replacing them. The weights are recomputed at every checkpoint improvement and persisted
in each sidecar under `cdws_weights`, so Stage 3 inherits Stage 2's measurement.

---

## 4.3 Stage 3 — SAM + greedy SWA fine-tuning

### Sharpness-Aware Minimisation

SAM seeks minima that are *flat*, not merely low, by minimising the worst-case loss in a
$\rho$-ball around the weights:

$$
\min_{\theta}\; \max_{\|\varepsilon\|_2 \le \rho}\; \mathcal{L}(\theta + \varepsilon)
$$

Each batch takes two steps ($\rho = 0.015$):

**Ascent** (`first_step`) — cache $\theta$, move to the locally worst-case point along the
gradient direction, using the *global* gradient norm across all parameter groups:

$$
\hat{\varepsilon} = \rho\,\frac{\mathbf{g}}{\|\mathbf{g}\|_2 + 10^{-12}},
\qquad
\|\mathbf{g}\|_2 = \max\Big(\big\|\,\|\nabla_p\mathcal{L}\|_2\,\big\|_2,\; 10^{-6}\Big),
\qquad \theta \leftarrow \theta + \hat{\varepsilon}
$$

**Descent** (`second_step`) — restore $\theta$ from the cache and let AdamW step with the
gradient computed *at the perturbed point*:

$$
\theta \leftarrow \theta_{\text{old}}, \qquad \theta \leftarrow \mathrm{AdamW}\big(\theta,\; \nabla\mathcal{L}(\theta + \hat{\varepsilon})\big)
$$

`SAM.step()` deliberately raises `NotImplementedError`, so a caller cannot accidentally take a
single non-SAM step. Gradients are clipped to `grad_clip = 1.0` before *both* steps. The
ascent loss/accuracy are what the epoch reports; the descent loss is computed only for its
gradient.

### Cyclic LR and margin

$$
\eta(e) = \eta_{\text{swa}}\Big(0.3 + 0.35\big(1 + \cos\big(\pi\,\tfrac{(e-1) \bmod 8}{8}\big)\big)\Big)
\in \big[0.33\,\eta_{\text{swa}},\; \eta_{\text{swa}}\big],
\qquad \eta_{\text{swa}} = 4\times10^{-5}
$$

$$
m_{\text{S3}}(e) = 0.25 + 0.05\cos\!\Big(\frac{\pi e}{120}\Big) \;:\; 0.30 \to 0.20
$$

The Stage-3 margin is always passed explicitly as `global_m`, so the per-class adaptive
margins are bypassed for the whole stage — a deliberate late-training relaxation
($m$ *decreases*, unlike Stage 2's ramp).

### Greedy SWA snapshotting

At the end of every 8-epoch cycle (15 cycles over 120 epochs), the live weights are averaged
into a running SWA state only if they are within 2 % of the best validation $F_1$ seen so far
(the running best is updated *before* the test, so it includes the current epoch):

$$
\text{accept} \iff F_1^{\text{live}}(e) \;\ge\; 0.98 \cdot \max_{e' \le e} F_1^{\text{live}}(e')
$$

$$
\theta^{\text{SWA}} \leftarrow (1 - \beta)\,\theta^{\text{SWA}} + \beta\,\theta^{(n)},
\qquad \beta = \frac{1}{n}, \quad n = \text{accepted count}
$$

which is exactly the running mean of accepted snapshots. Non-floating-point entries (integer
buffers such as `num_batches_tracked`) are copied rather than averaged. If nothing is
accepted, the final live model is used.

**BatchNorm re-estimation is mandatory.** `update_bn_stats` resets every BN layer's running
statistics and momentum to `None` (cumulative-average mode) and runs one pass over the
training loader in `train()` mode — without it the averaged weights carry BN buffers that
correspond to no model that ever existed. On Metal this pass keeps grad *enabled*, because
MPS routes attention through a fused inference kernel with no dropout support when grad is
off; forward values, and therefore the estimated statistics, are identical either way.

Finally the SWA weights are copied into the EMA shadow and the stage **always saves**, adding
a `note` when it failed to beat Stage 2 — the selection decision belongs to
`_pick_best_checkpoint`, not to the stage. Recorded run: **15 accepted, 0 rejected**.

---

## 4.4 Mathematical objectives

### Focal loss (`losses/focal.py`)

With $\log p = \log\mathrm{softmax}(z)$ and smoothing $\varepsilon$, the smoothed target is
$q_c = 1-\varepsilon$ for $c = y$ and $\varepsilon/(C-1)$ otherwise:

$$
\ell_i = -\sum_c q_{i,c}\log p_{i,c}
\quad(\text{or } -\log p_{i,y_i} \text{ when } \varepsilon = 0),
\qquad
\mathcal{L}_{\text{focal}} = \frac{1}{B}\sum_i \big(1 - e^{-\ell_i}\big)^{\gamma}\,\ell_i
$$

The modulation uses $p_t = e^{-\ell}$, so smoothing is applied *first* and the focal term
sharpens on top of it. Used with $\gamma = 1.5$ (Stage 1 Phase 3 and Stage 2) and
$\gamma = 1.0$ (Stage 3).

### ArcFace objective

There is no separate "ArcFace loss" module: the head produces margin-penalised logits
(§3.5) and those are fed to the Focal criterion. For clarity, with $s = 48$ and per-sample
margin $m_i$:

$$
\mathcal{L}_{\text{arc}} = -\frac{1}{B}\sum_i \big(1-p_{i,y_i}\big)^{\gamma}
\log \frac{e^{\,s\cos(\theta_{i,y_i} + m_i)}}
          {e^{\,s\cos(\theta_{i,y_i} + m_i)} + \sum_{c \ne y_i} e^{\,s\cos\theta_{i,c}}}
$$

with $\cos\theta_{i,c} = \max_k \hat{\mathbf{e}}_i^{\top}\hat{\mathbf{W}}_{c,k}$ (sub-centre max)
and $m_i = m(t)$ during warm-up, $M(y_i)$ afterwards.

### Supervised contrastive (`SupConLoss`, $\tau = 0.10$)

Over $\ell_2$-normalised embeddings, with $P(i) = \{j \ne i : y_j = y_i\}$:

$$
\mathcal{L}_{\text{supcon}} = \frac{1}{|A|}\sum_{i \in A}
\frac{-1}{|P(i)|}\sum_{p\in P(i)}
\log\frac{\exp(\hat{\mathbf{e}}_i^{\top}\hat{\mathbf{e}}_p/\tau)}
         {\sum_{a \ne i}\exp(\hat{\mathbf{e}}_i^{\top}\hat{\mathbf{e}}_a/\tau)}
$$

where $A = \{i : |P(i)| > 0\}$ — anchors with no in-batch positive are excluded from the mean
rather than contributing zero. If *no* anchor has a positive, the loss is a grad-carrying zero.
Treating every same-class member as a positive is what makes the $16\times8$ balanced batch
(8 positives per anchor) the right batch shape for this stage.

### Prototype contrastive (`ProtoNCELoss`, $\tau = 0.10$)

Batch-local class prototypes, then plain cross-entropy against them:

$$
\boldsymbol{\mu}_c = \frac{\bar{\mathbf{e}}_c}{\|\bar{\mathbf{e}}_c\|_2},\quad
\bar{\mathbf{e}}_c = \frac{1}{|\{i: y_i = c\}|}\sum_{i: y_i = c}\hat{\mathbf{e}}_i
$$

$$
\mathcal{L}_{\text{proto}} = -\frac{1}{B}\sum_i
\log\frac{\exp(\hat{\mathbf{e}}_i^{\top}\boldsymbol{\mu}_{y_i}/\tau)}
         {\sum_{c \in \mathcal{C}_{\text{batch}}}\exp(\hat{\mathbf{e}}_i^{\top}\boldsymbol{\mu}_{c}/\tau)}
$$

Only classes present in the batch form prototypes; with fewer than 2 present the loss returns
zero. It is $O(B|\mathcal{C}_{\text{batch}}|)$ rather than SupCon's $O(B^2)$, at the cost of
discarding intra-class structure below the mean.

### Auxiliary deep-supervision loss (`losses/auxiliary.py`)

$$
\mathcal{L}_{\text{aux}} = \sum_{b \in \{A,B,C,D\}} \omega_b\,\mathcal{L}_{\text{crit}}\big(z^{\text{aux}}_b,\, y\big),
\qquad
\omega_A = \omega_B = 2.0,\quad \omega_C = \omega_D = 1.0
$$

The $2\times$ bias on the spectral branches is load-bearing: without it the spatial branches
dominate and A/B produce near-zero gradients (`tests/unit/test_diagnostics.py` pins the
weighting). A missing `aux_*` key contributes nothing. Under mixup each term uses the
interpolated loss below.

Its weight decays linearly with training progress, but never to zero — branches must keep
receiving gradient:

$$
w_{\text{aux}}(e) = \max\!\Big(w_{\text{final}},\; w_{\text{init}}\big(1 - 0.7\,\tfrac{e}{E}\big)\Big),
\qquad w_{\text{init}} = 0.65,\; w_{\text{final}} = 0.25
$$

which decays $0.65 \to 0.25$ and reaches the floor at $e \approx 528$ of 600. Stage 3 ignores
this schedule entirely and uses a fixed $w_{\text{aux}} = 0.10$.

### Mixup (`losses/mixup.py`)

$$
\lambda \sim \mathrm{Beta}(\alpha, \alpha),\quad \alpha = 0.35,
\qquad
\tilde{x} = \lambda x + (1-\lambda)x_{\pi},
\qquad
\mathcal{L} = \lambda\,\mathcal{L}_{\text{crit}}(z, y) + (1-\lambda)\,\mathcal{L}_{\text{crit}}(z, y_{\pi})
$$

with $\pi$ a random permutation of the batch. Mixup is **mutually exclusive with ArcFace** —
soft-interpolated targets are incompatible with an angular margin indexed by a single label,
and `train_one_epoch` raises `ValueError` on the combination. Only batch-level mixup is
implemented; there is no CutMix operator in this codebase (the analogous spatial/spectral
occlusion is `_band_cutout`, applied per sample in the dataset, §2.6).

### Compound totals

**Stage 1, Phases 1–2** (CE path, mixup on):

$$
\mathcal{L}_{\text{total}} = \underbrace{\lambda\mathcal{L}_{\text{CE}}(z, y) + (1{-}\lambda)\mathcal{L}_{\text{CE}}(z, y_\pi)}_{\text{main}}
\;+\; w_{\text{aux}}(e)\sum_b \omega_b\Big[\lambda\mathcal{L}_{\text{CE}}(z_b, y) + (1{-}\lambda)\mathcal{L}_{\text{CE}}(z_b, y_\pi)\Big]
$$

**Stage 1, Phase 3 and Stage 2** (contrastive path, mixup off) — note the classification term
is *down-weighted* by the contrastive weights, so the three terms form a convex combination:

$$
\boxed{\;
\mathcal{L}_{\text{total}} =
\big(1 - \lambda_{\text{sc}} - \lambda_{\text{pt}}\big)\,\mathcal{L}_{\text{cls}}
\;+\; \lambda_{\text{sc}}\,\mathcal{L}_{\text{supcon}}
\;+\; \lambda_{\text{pt}}\,\mathcal{L}_{\text{proto}}
\;+\; w_{\text{aux}}(e)\,\mathcal{L}_{\text{aux}}
\;}
$$

| | $\mathcal{L}_{\text{cls}}$ | $\lambda_{\text{sc}}$ | $\lambda_{\text{pt}}$ | effective $\mathcal{L}_{\text{cls}}$ weight |
|---|---|---:|---:|---:|
| Stage 1 Phase 3 | Focal($\gamma{=}1.5$, $\varepsilon(e)$) on linear logits | 0.35 | 0.15 | 0.50 |
| Stage 2 | Focal($\gamma{=}1.5$) on ArcFace logits | $0.40\min(1,e/10)$ | $0.18\min(1,e/10)$ | $\ge 0.42$ |

**Stage 3** (SAM ascent step):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}(\gamma{=}1) + 0.02\,\mathcal{L}_{\text{supcon}} + 0.10\,\mathcal{L}_{\text{aux}}
$$

The SAM **descent** step recomputes only $\mathcal{L}_{\text{focal}}(\gamma{=}1)$ at the
perturbed point — the contrastive and auxiliary terms do not enter the worst-case gradient.
A `proto` module and `proto_weight = 0.01` are passed into `train_one_epoch_sam`, but that
loop applies no ProtoNCE term, so **ProtoNCE is inactive in Stage 3**.

---

## 4.5 Optimisation rules

### Weight-decay parameter groups

`optim/param_groups.py::_wd_groups` splits parameters by a single rule:

$$
p \in \text{no-decay} \iff \operatorname{ndim}(p) = 1 \;\;\vee\;\; \text{name ends with } \texttt{.bias}
$$

This places **every normalisation affine parameter** (`GroupNorm`, `LayerNorm`, `BatchNorm1d/2d`
weights and biases are all 1-D) and **every bias** in a group with `weight_decay = 0.0`; all
2-D+ weights get `weight_decay = 2\times10^{-4}`. The physical wavelength encoding is excluded
by construction rather than by rule — `PhysicalWavelengthPE.pe` is a registered *buffer*, so it
receives neither gradient nor decay. Frozen parameters (`requires_grad = False`) are skipped
entirely, which is how head freezing removes a head from the optimiser.

| Builder | Groups | Used by |
|---|---|---|
| `build_optimizer_s1` | `[wd, no_wd]` at `stage1.max_lr` | Stage 1 |
| `build_optimizer_s2` | `[head-wd, head-no-wd, backbone-wd, backbone-no-wd]` | Stage 2 |
| `_wd_groups(...)` wrapped in `SAM` | `[wd, no_wd]` at `stage3.swa_lr` | Stage 3 |

Stage 2 splits on the `arcface_head` name prefix, giving the head
$2.5\times10^{-4}$ and the backbone $7\times10^{-5}$. **The group order is load-bearing**:
`stage2_arcface.py` reads `param_groups[0]` and `param_groups[2]` back as the head and
backbone learning rates for logging.

### Gradient handling

| Rule | Detail |
|---|---|
| Clipping | `clip_grad_norm_(model.parameters(), 1.0)` — a single global norm over the whole model, applied on the accumulation boundary (Stages 1–2) and before both SAM steps (Stage 3) |
| Accumulation | `stage1.accum = 1`; the optimiser steps, and the EMA updates, only on the boundary — so the EMA advances once per *optimiser* step, not per batch |
| Non-finite loss | zero the gradients and skip the batch, never raise — an unstable ArcFace epoch degrades a metric rather than killing a multi-hour run |
| Gradient norms | sampled after `scaler.unscale_` and **before** the clip, so the diagnostic reports true pre-clip norms (§5.2) |

### Mixed precision

$$
\texttt{use\_amp} \;=\; (\texttt{supcon is None}) \;\wedge\; (\texttt{scaler is not None})
$$

Stage 1 Phases 1–2 therefore train under `autocast` with a `GradScaler` **bound to the active
device** (`GradScaler(device=device.type)` — a bare `GradScaler()` binds to CUDA and silently
no-ops on Metal). Stage 1 Phase 3 and all of Stage 2 pass a SupCon module and so run fp32;
Stage 3 passes no scaler and is fp32 by construction. Evaluation is always fp32.

### EMA

$$
d_n = \min\Big(d_{\max},\; \frac{1+n}{10+n}\Big),
\qquad
\theta^{\text{EMA}} \leftarrow d_n\,\theta^{\text{EMA}} + (1 - d_n)\,\theta
$$

with $d_{\max} = 0.999$ and $n$ the update count. The warm-up lets the shadow track fast early
movement instead of lagging it. **Floating-point buffers are copied outright, not averaged**
(integer buffers such as `num_batches_tracked` are left untouched) — an EMA of BatchNorm
running statistics is not itself a running statistic. `reinit_from` hard-resets the shadow and
restarts the warm-up; it is called at both Stage-1 phase boundaries and on entry to Stage 2.

### Device selection

`cfg.device = "auto"` resolves **Metal (MPS) → CUDA → CPU**; an explicit `cuda`/`cpu`/`mps` is
never overridden. Autocast dtype is left at torch's per-device default (fp16 on both Metal and
CUDA) rather than promoted to bf16, to keep numerics consistent across accelerators.
