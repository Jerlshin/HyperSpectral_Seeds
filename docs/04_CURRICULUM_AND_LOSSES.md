# 4 · Curriculum, Objectives and Optimisation

> **The default curriculum is now one stage, not three** (CHANGES.md IC-11 / §17;
> `configs/single/one_stage.yaml`, `engine/stages/single_stage.py`). This document describes the
> audited three-stage curriculum, which is retained *unmodified* because ablation **A8** is the
> falsification test for it and needs it as a control arm.
>
> The measured case, from the audited run:
>
> | Stage | Wall clock | Δ val macro-F1 | Δ in eval samples | Hours per 0.001 F1 |
> |---|---:|---:|---:|---:|
> | 1 | 6.6 h | (baseline 0.842) | — | — |
> | 2 | 2.7 h | **+0.002** | **2.6 / 1294** | 1,325 |
> | 3 | ~9.5 h | **+0.003** | **3.9 / 1294** | 3,167 |
>
> 65% of the wall clock bought 6.5 samples of a 1,294-sample split, against a ±26-sample 95% CI
> and an estimated **+0.042** of running-maximum selection bias. Worse, Stage 2's best epoch is
> **19** — two epochs *before* its headline mechanism, the per-class signed-margin vector, takes
> over at epoch 21.
>
> The one thing Stage 2 did that Stage 1 did not is introduce a non-zero angular margin, and the
> only reason that needed its own stage is that a margin is incompatible with mixup. The
> collapsed curriculum switches mixup off at epoch 110 and warms one scalar margin in over
> 111–130, achieving the same transition inside one stage with one optimiser state and no EMA
> re-initialisation. Run `pipeline=three_stage` to reach everything below.

Three stages run in sequence, each writing `best_stage{n}.pth` + `stage{n}_meta.json`. Each
stage module (`engine/stages/`) is **orchestration only**: every unit of work is a call into
`engine/`, `losses/` or `optim/`, which is what makes the schedules independently testable.

**There is one classification head, not two.** All three stages train and evaluate through the
same `arcface_head` (§3.5), differing only in the margin passed at each call.
`stage1.arcface_m = 0.0` makes Stage 1 a plain cosine (NormFace) classifier, which is nearly
what a two-head design's linear head would be, since `EmbedNet`'s terminal LayerNorm pins
$\|\mathbf{e}\|\approx16$ regardless. Because only the margin regime changes at the Stage 1→2
boundary — not the head, loss, sampler, optimiser or augmentation all at once —
`tests/unit/test_unified_head.py` confirms the last Stage-1 forward and the first Stage-2 forward
agree on every non-target logit to $10^{-5}$.

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Module | `stage1_progressive.py` | `stage2_arcface.py` | `stage3_sam_swa.py` |
| Epochs (max) | 400 | 150 | 120 (no early stop) |
| Head | `arcface_head` @ $m{=}0$ | `arcface_head` @ warmed/per-class $m$ | `arcface_head` @ annealed per-class $m$ |
| Optimiser | AdamW, per-group clip | AdamW (2 LRs), per-group clip | SAM/ASAM(AdamW), per-group clip |
| LR schedule | phase-aware (3 phases) | warmup + SGDR | cyclic cosine, 8-epoch |
| Batch | 128 (shuffled / oversampled) | $16\times8=128$ balanced | $16\times8=128$ balanced |
| Patience (macro-F1) | 50 | 30 | — |
| Mixup | Phases 1–2 only | ✗ | ✗ |
| Same-class CutMix | all profiles except `none` | `very_light` profile | `light` profile |
| AMP | Phases 1–2 | ✗ (SupCon forces fp32) | ✗ (SAM is fp32 by construction) |

---

## 4.1 Stage 1 — three-phase progressive augmentation

### Phase boundaries

$$
p_1 = \lfloor E\cdot0.30\rfloor = 120, \qquad p_2 = \lfloor E\cdot(0.30+0.38)\rfloor = 272, \qquad E=400
$$

| Phase | Epochs | Augmentation | Loss | Mixup | Sampler |
|---|---|---|---|---|---|
| 1 · explore | 1–120 | `heavy` | CE + label smoothing | ✓ ($\alpha{=}0.35$) | shuffled |
| 2 · consolidate | 121–272 | `medium` | CE + label smoothing | ✓ | shuffled |
| 3 · discriminate | 273–400 | `very_light` | Focal + SupCon + ProtoNCE | ✗ | hard-class oversampled |

Both boundaries are `int(...)` truncations of a float product, not rounded — because
`0.30 + 0.38` is $0.6799\overline{9}$ in binary rather than exactly $0.68$, the truncated
boundary is not guaranteed to equal the naive product rounded to the nearest epoch. The schedule
is always computed from the truncated values, never from the fractions.

At each boundary the EMA shadow is hard-reset (`ema_reinit_phases = true`) and, entering Phase 3,
dropout is raised $0.15\to0.25$ (`p3_dropout`). The sub-centre pooling temperature $\tau$
(§3.5) anneals continuously across the **whole stage**, independent of phase, via
`subcentre_tau` — `subcenter_tau_init = 0.20 \to subcenter_tau_final = 0.02` — and is pushed to
both the live model and the EMA shadow every epoch.

**Leaving a phase also frees its loader.** `_release_stale_phase_loaders` deletes every strictly
earlier phase's entry from `loaders_by_phase` at each boundary and sweeps the allocator, because
`persistent_workers=True` keeps a loader's worker processes alive for exactly as long as
something holds the loader — otherwise Phase 3 would run with the Phase-1 and Phase-2 pools still
resident: four idle `spawn`-ed interpreters, each with torch imported and its own mapping of the
patch cube. The dict is mutated in place deliberately, since it is the only reference to those
loaders and copying it first would free nothing. Phase 3 keeps its own entry even after
`build_phase3_loader` replaces it, because the oversampled loader is built over that entry's
dataset — and it costs nothing, the entry never being iterated and so never spawning a worker.

The Phase 2 → 3 transition is where the curriculum becomes data-adaptive: per-class $F_1$ is
measured on the EMA shadow (`compute_class_difficulty`, using the calibration split when carved,
else `val`) and drives the `HardClassOversampledSampler` for the remainder of the stage (§2.7).

### The unified head at Stage 1

Every call into `train_one_epoch` passes `arc_m = cfg.stage1.arcface_m = 0.0`. The head's
`forward` takes an explicit fast path at `global_m == 0.0` that skips the margin algebra
entirely (§3.5) — bit-identical to running the full computation at a vanishing margin, verified
directly. Because $\arg m=0$, Stage 1 is compatible with mixup (`train_one_epoch` raises
`ValueError` only when `use_mixup and (arc_m is None or arc_m > 0.0)`), which is what keeps
mixup available through Phases 1–2.

### Phase-aware learning rate

`phase_aware_lr` returns a **multiplier** on `stage1.max_lr` $=5\times10^{-4}$, built as a
standalone factory (not an inline closure) specifically so its whole trajectory can be
pinned bit-exact against a reference implementation:

$$
\eta(e)/\eta_{\max} =
\begin{cases}
\dfrac{e}{5}, & e\le5 \quad\text{(linear warm-up)}\\[8pt]
0.6+0.2\Big(1+\cos\big(\pi\tfrac{e-5}{p_1-5}\big)\Big), & 5<e\le p_1 \quad(1.0\to0.6)\\[8pt]
0.2+0.2\Big(1+\cos\big(\pi\tfrac{e-p_1}{p_2-p_1}\big)\Big), & p_1<e\le p_2 \quad(0.6\to0.2)\\[8pt]
\rho_{\min}+(\rho_{\text{mid}}-\rho_{\min})\cdot\tfrac12\Big(1+\cos\big(\pi\tfrac{(e-p_2-1)\bmod30}{30}\big)\Big), & e>p_2
\end{cases}
$$

with $\rho_{\min}=\eta_{\min}/\eta_{\max}=0.01$, $\rho_{\text{mid}}=\eta_{\text{mid}}/\eta_{\max}=0.5$.
In absolute terms: warm-up to $5\times10^{-4}$, decay to $3\times10^{-4}$ by epoch 120, to
$1\times10^{-4}$ by epoch 272, then 30-epoch cosine restarts oscillating between $5\times10^{-6}$
and $2.5\times10^{-4}$. `tests/unit/test_schedulers.py` checks `stage1.epochs = 400` (the
shipped config) against a baseline reference, across three phase-length configurations.

### Label-smoothing decay

Linear across the **whole stage**, independent of phase:

$$
\varepsilon(e) = \varepsilon_{\text{hi}}(1-t) + \varepsilon_{\text{lo}}\,t,
\qquad t=\frac{e-1}{E-1}, \qquad \varepsilon_{\text{hi}}=0.10\to\varepsilon_{\text{lo}}=0.04
$$

### Checkpointing

Every epoch scores both the live model and the EMA shadow; on
$\max(F_1^{\text{live}},F_1^{\text{ema}})>F_1^{\text{best}}$ the stage records `best_source ∈
{"live","ema"}`, recomputes class difficulty, and saves a bundle carrying `class_f1`,
`cdws_weights` and `phase3_class_f1`. `arcface_init_done=False` is also written into the
metadata, kept for readers that check it, though with one head there is nothing left to
bootstrap. Patience is 50 epochs without improvement, and the decision is broadcast under DDP
so every rank stops on the same epoch.

---

## 4.2 Stage 2 — sub-centre ArcFace and metric learning

### Entry transitions

There is no linear→ArcFace bootstrap — the head is the same object Stage 1 trained. Entry does:

1. `model.set_dropout(cfg.stage2.dropout)` ($0.10$).
2. EMA hard-reset (`ema.reinit_from(model)`).
3. **Margin calibration** (below) — computed once from precision/recall on the fit split and
   written into `arcface_head.margins`/`.confusion` on both the live model and the EMA shadow.
4. Batch composition switches to `ClassBalancedBatchSampler` with Stage 1's CDWS weights.

### Angular-margin warm-up, then per-class hand-over

A cosine ramp from $m_0=0.18$ to $m_{\text{base}}=0.35$ over the first `margin_warmup_ep` $=20$
epochs, passed as a **global scalar** override:

$$
m(t) = m_0+(m_{\text{base}}-m_0)\cdot\tfrac12\Big(1-\cos\big(\pi\tfrac{t}{t_{\text{warm}}}\big)\Big), \qquad t=e-1<20
$$

**The hand-over matters**: while $t<20$, $m(t)$ is passed as `global_m` and overrides the
`margins` buffer for every sample; once $t\ge20$ the call site passes `arc_m=None`, and the head
switches to its per-class adaptive vector $M(c)$. So the schedule is: one global margin, ramped;
then per-class margins, calibrated once at stage entry and annealed multiplicatively for the rest
of Stage 3 (§4.3) — not recalibrated epoch to epoch.

### The signed precision/recall margin rule

$$
\boxed{\, M(c) = \operatorname{clip}\big(m_{\text{base}} + m_\Delta\,(R_c - P_c),\; m_{\min},\; m_{\max}\big) \,}
$$

$m_{\text{base}}=$ `arcface_m` $=0.35$, $m_\Delta=$ `arcface_m_delta` $=0.20$, $m_{\min}=0.20$,
$m_{\max}=0.50$. $P_c,R_c$ come from `evaluate_pr_and_confusion` on the fit loader, evaluated
once at Stage-2 entry, not re-fitted per epoch. The sign is deliberately opposite that of a
naive F1-driven rule $M(c)=m_{\text{base}}+m_\Delta(1-F_1^{(c)})$: an additive angular margin
*shrinks* the margined class's decision region, so an over-claiming class ($R_c>P_c$) needs a
**larger** margin and an under-claiming one ($R_c<P_c$) needs a **smaller** one — an F1-driven
rule moves every low-$F_1$ class the same direction regardless of which kind of error it is
making, which is backwards for exactly half of them. Two classes tied on $F_1$ can land on
opposite sides of $m_{\text{base}}$ under this rule; `tests/unit/test_margin_rule.py` pins the
sign property directly.

**Pairwise confusion term.** Alongside the margin, `set_confusion` stores a row-normalised,
zero-diagonal confusion matrix $\Omega$ from the same calibration pass. Every non-target logit
for sample $i$ (true class $y_i$) is additionally penalised by
`pairwise_margin_delta` $=0.10$ scaled by how often $y_i$ is actually confused with that column:
$\mathrm{logit}_{i,c}\mathrel{-}=s\,\delta_{\text{pw}}\,\Omega_{y_i,c}$ — aiming the margin at
the classes a class is actually confused with, not uniformly at all 89 others (§3.5).

### SGDR

`sgdr_scheduler` — linear warm-up then cosine restarts of geometrically growing length:

$$
\eta(e)/\eta_0 =
\begin{cases}
\max(e/t_{\text{warm}},\,10^{-6}), & e<t_{\text{warm}}\\[6pt]
\eta_{\min}^{\text{frac}} + \tfrac12(1-\eta_{\min}^{\text{frac}})(1+\cos(\pi r)), & e\ge t_{\text{warm}}
\end{cases}
$$

$r=(t-T_{\text{elapsed}})/T_{\text{cur}}$, $T_{\text{cur}}$ walking $T_0,2T_0,4T_0,\dots$ Shipped:
$T_0=25$, $T_{\text{mult}}=2$, $t_{\text{warm}}=3$,
$\eta_{\min}^{\text{frac}}=\eta_{\min}/\eta_{\text{head}}=4\times10^{-3}$ — restarts land at
epochs **28** and **78** (printed `↻R1`/`↻R2`). Two LRs share the multiplier but not the base:
head $2.5\times10^{-4}$ (`param_groups[0]`), backbone $7\times10^{-5}$ (`param_groups[2]`) — the
group order (head-wd, head-no-wd, backbone-wd, backbone-no-wd, split on the `arcface_head` name
prefix) is load-bearing since the stage reads those indices back for logging.

### Contrastive ramp

$$
\lambda_{\text{sc}}(e) = 0.40\cdot\min(1,e/10), \qquad \lambda_{\text{pt}}(e) = 0.18\cdot\min(1,e/10)
$$

so the margin loss establishes itself over the first 10 epochs before SupCon/ProtoNCE reach full
weight (temperatures $0.10$ each).

### Class-Difficulty-Weighted Sampling (CDWS)

$$
w_c = \frac{\tilde w_c}{\frac1C\sum_{c'}\tilde w_{c'}}, \qquad \tilde w_c = \min\!\Big(\frac{1}{F_1^{(c)}+\epsilon},\,W_{\max}\Big)
$$

$\epsilon=$ `cdws_eps` $=0.05$, $W_{\max}=$ `cdws_max_weight` $=3.0$, missing class $\Rightarrow
F_1{=}0$. Compared with the Phase-3 oversampler (§2.7) this is gentler — no exponent, a $3\times$
rather than $7\times$ cap — because it composes with class-balanced batches rather than
replacing them. Recomputed at every checkpoint improvement and persisted in the sidecar, so
Stage 3 inherits Stage 2's measurement.

### Checkpointing

Same $\max(F_1^{\text{live}},F_1^{\text{ema}})$ scheme as Stage 1; margin summary
(`margin/{mean,min,max}`) logged every epoch from one stacked device→host transfer.

---

## 4.3 Stage 3 — SAM/ASAM + greedy SWA fine-tuning

### Sharpness-Aware Minimisation, and its adaptive variant

SAM seeks flat minima by minimising the worst-case loss in a $\rho$-ball:

$$
\min_\theta\;\max_{\|\varepsilon\|_2\le\rho}\;\mathcal{L}(\theta+\varepsilon)
$$

Each batch takes two steps, evaluating the **identical compound objective** at both the ascent
and descent points (Focal + SupCon + auxiliary + balance — §4.4's Stage-3 row):

**Ascent** (`first_step`) — cache $\theta$, move along the (optionally element-wise-rescaled)
gradient direction:

$$
\hat\varepsilon = \rho\,\frac{\mathbf{g}}{\|\mathbf{g}\|_2+10^{-12}} \;\;\text{(SAM)}
\qquad\text{or}\qquad
\hat\varepsilon = \rho\,\theta^2\odot\frac{\mathbf{g}}{\|\theta\odot\mathbf{g}\|_2}\;\;\text{(ASAM)}
$$

$$
\theta \leftarrow \theta+\hat\varepsilon
$$

**Descent** (`second_step`) — restore $\theta$ from the cache, then let AdamW step with the
gradient computed **at the ascended point**:

$$
\theta \leftarrow \theta_{\text{old}}, \qquad \theta\leftarrow\mathrm{AdamW}(\theta,\,\nabla\mathcal{L}(\theta+\hat\varepsilon))
$$

`sam_adaptive = true` (shipped) selects ASAM: the perturbation is rescaled elementwise by
$|\theta|$, so it is invariant to a per-parameter rescaling of the weights — measured to put
$\ge10\times$ less of the perturbation budget on the ArcFace head than raw SAM would, on the
model's real weights. `SAM.step()` raises `NotImplementedError` so a caller cannot accidentally
take a single non-SAM step; if the descent-point loss is non-finite, `restore()` puts the
weights back without stepping rather than baking in a permanent perturbation. Gradients are
clipped **per parameter group** (§4.5) before both steps.

### Margin — a per-class vector, frozen within a cycle

Stage 2's calibrated margin vector is captured once at Stage-3 entry
(`base_margins = arcface_head.margins.clone()`) and scaled multiplicatively, **stepping only at
cycle boundaries**:

$$
\kappa(e) = 1.0+(\kappa_{\text{final}}-1.0)\cdot\frac{\lfloor(e-1)/L\rfloor}{N_{\text{cycles}}-1},
\qquad
\text{margins}(e) = \kappa(e)\cdot\text{base\_margins}
$$

$L=$ `cycle_len` $=8$, $N_{\text{cycles}}=\lceil120/8\rceil=15$, $\kappa_{\text{final}}=$
`margin_kappa_final` $=0.85$ — **constant within a cycle**, so every step between two SWA
snapshots optimises one fixed objective, which is what SWA's averaging assumes. `arc_m=None` is
passed throughout Stage 3: the per-class vector, not a scalar override, is what's optimised
against.

### Cyclic LR

$$
\eta(e) = \eta_{\text{swa}}\Big(0.3+0.7\cdot\tfrac12\big(1+\cos\big(\pi\tfrac{(e-1)\bmod8}{8}\big)\big)\Big) \in [0.3\,\eta_{\text{swa}},\,\eta_{\text{swa}}], \qquad \eta_{\text{swa}}=4\times10^{-5}
$$

### Greedy SWA snapshotting — a real accept/reject test

At every cycle boundary ($e\bmod8=0$), a **candidate** running-average blend is scored
*before* it becomes the accepted average:

$$
\theta^{\text{cand}} = (1-\beta)\,\theta^{\text{SWA}} + \beta\,\theta^{(n)}, \qquad \beta=\frac1n
$$

$$
\text{accept} \iff \neg\,\texttt{greedy} \;\lor\; F_1^{\text{cand}} > F_1^{\text{SWA}}_{\text{running}}
$$

evaluated by loading the candidate into a scratch probe model and running it against `val`.
`swa_warmup_cycles = 3` cycles are discarded **before any candidate is even considered** — no
snapshot, no rejection counted — keeping Adam's second-moment warm-up transient
($1/(1-\beta_2)\approx1000$ steps, $\approx3$ cycles at the shipped batch/loader size) out of the
average. `greedy=false` disables the filter and accepts every candidate unconditionally (the
"average everything" control mode).

### BatchNorm re-estimation

Mandatory after Stage 3's training loop, and run against a **shuffled, unweighted** rebuild of
the training loader — not the CDWS-weighted one training used — since BN statistics estimated
under a re-weighted class prior would be biased against the natural test-time prior. Every
`BatchNorm{1,2}d`'s running stats are reset and `momentum` set to `None` (cumulative-average
mode); every stochastic module (`nn.Dropout*` **and** `nn.MultiheadAttention`, the layer
`set_dropout` cannot reach) is forced to `.eval()`; the whole pass then runs under plain
`torch.no_grad()`, identically on every accelerator — dropout under `eval()` is also the
numerically correct setting for the statistics being re-estimated, since inverted dropout would
inflate their variance.

### EMA and SWA — two averaging schemes, scored and the winner kept

The EMA shadow keeps updating throughout Stage 3 exactly as in Stages 1–2. At the end, **both**
the SWA average and the EMA shadow are evaluated on `val`, and whichever wins is what actually
gets saved as the shadow:

$$
\text{best\_source} = \begin{cases}\text{"swa"}, & F_1^{\text{SWA}}\ge F_1^{\text{EMA}}\\ \text{"ema"}, & \text{otherwise}\end{cases}
$$

if `"swa"` wins, the SWA weights are copied into the EMA shadow before saving, so the bundle's
`ema` slot always holds whichever averaging scheme actually produced the recorded `val_f1`.
Stage 3 **always saves**, whether or not it beats Stage 2 — a `note` field records the comparison
verbatim; `_pick_best_checkpoint`, not the stage, decides which checkpoint final evaluation uses.

Stage-3-specific loss weights are **hardcoded constants**, not config fields:
$\gamma_{\text{focal}}=1.0$, SupCon weight $0.02$, ProtoNCE weight $0.01$ — but see §4.4: the
Stage-3 training loop does not actually apply a ProtoNCE term despite the constant being defined,
so ProtoNCE is inactive in Stage 3 in practice. `aux_loss_weight` ($0.10$, `cfg.stage3`) is fixed,
with no decay schedule. `branch_drop_prob` is set to $0$ at Stage-3 entry (branch masking off);
`torch.compile` graphs are dropped at the Stage 2→3 boundary (`reset_compilation()`), since SAM's
double backward and the per-cycle margin vector would otherwise trigger continual recompilation.

---

## 4.4 Mathematical objectives

### Focal loss (`losses/focal.py`)

With $\log p=\log\mathrm{softmax}(z)$ and label smoothing $\varepsilon$, the cross-entropy term
uses the **smoothed** target, but the focal modulator reads the **unsmoothed** $p_{y}$ so it can
still reach $0$ (with smoothing on, $\exp(-\text{smoothed CE})$ is bounded below by
$\exp(-H(q))>0$, which a smoothed modulator could never fully close):

$$
\ell_i = -\sum_c q_{i,c}\log p_{i,c} \quad (q\text{ smoothed}),
\qquad
\mathcal{L}_{\text{focal}} = \frac1B\sum_i\big(1-p_{i,y_i}\big)^\gamma\,\ell_i
$$

$\gamma=1.5$ (Stage 1 Phase 3, Stage 2), $\gamma=1.0$ (Stage 3).

### ArcFace-via-head

No separate "ArcFace loss" module: the head produces margin- and confusion-penalised logits
(§3.5), fed to Focal. With $s=48$, $M_i=$ the per-sample margin in force (§4.1–4.3), and the
pairwise term folded in:

$$
\mathcal{L}_{\text{arc}} = -\frac1B\sum_i(1-p_{i,y_i})^\gamma
\log\frac{e^{s\cos(\theta_{i,y_i}+M_i)}}
{e^{s\cos(\theta_{i,y_i}+M_i)}+\sum_{c\ne y_i}e^{s(\cos\theta_{i,c}-\delta_{\text{pw}}\Omega_{y_i,c})}}
$$

### Sub-centre load-balancing

$$
\mathcal{L}_{\text{balance}} = \sum_{c\in\text{batch}}\mathrm{KL}(\boldsymbol\pi_c\,\|\,\mathrm{Uniform}_K)
$$

weighted by `subcenter_balance_weight = 0.01`, added whenever the head returns a `"balance"` key
(training mode with labels). A uniform assignment costs $0$; full collapse costs
$|\mathcal{C}|\log K$ (§3.5).

### Supervised contrastive (`SupConLoss`, $\tau=0.10$)

$$
\mathcal{L}_{\text{supcon}} = \frac{1}{|A|}\sum_{i\in A}\frac{-1}{|P(i)|}\sum_{p\in P(i)}
\log\frac{\exp(\hat{\mathbf{e}}_i^\top\hat{\mathbf{e}}_p/\tau)}{\sum_{a\ne i}\exp(\hat{\mathbf{e}}_i^\top\hat{\mathbf{e}}_a/\tau)}
$$

$A=\{i:|P(i)|>0\}$ — anchors with no in-batch positive are excluded from the mean, not
zero-contributed. The $16\times8$ balanced batch (8 positives/anchor) is the batch shape this
loss is designed around.

### Prototype contrastive (`ProtoNCELoss`, $\tau=0.10$)

$$
\boldsymbol\mu_c = \frac{\bar{\mathbf{e}}_c}{\|\bar{\mathbf{e}}_c\|_2}, \qquad
\mathcal{L}_{\text{proto}} = -\frac1B\sum_i\log\frac{\exp(\hat{\mathbf{e}}_i^\top\boldsymbol\mu_{y_i}/\tau)}{\sum_{c\in\mathcal{C}_{\text{batch}}}\exp(\hat{\mathbf{e}}_i^\top\boldsymbol\mu_c/\tau)}
$$

Zero with fewer than 2 classes present. $O(B|\mathcal{C}_{\text{batch}}|)$ against SupCon's
$O(B^2)$. **Passed into Stage 3's training loop as a module with `proto_weight=0.01`, but that
loop applies no ProtoNCE term** — ProtoNCE is inactive in Stage 3 despite the weight being
defined.

### Auxiliary deep-supervision loss and its GradNorm reweighting

$$
\mathcal{L}_{\text{aux}} = \sum_{b\in\{A,B,C,D\}}\omega_b\,\mathcal{L}_{\text{crit}}(z^{\text{aux}}_b,y)
$$

$\omega$ defaults to the fixed vector $\omega_A=\omega_B=2.0,\;\omega_C=\omega_D=1.0$
(load-bearing — without the $2\times$ bias on the spectral branches, A/B produce near-zero
gradients), but is **updated once per epoch** by a GradNorm rule reading the epoch-mean
per-branch gradient norms the loop already computes:

$$
\omega_b^{(t+1)} = \operatorname{clip}\Big(\omega_b^{(t)}\cdot\big(\bar g/g_b\big)^\alpha,\; 0.25,\; 4.0\Big),
\qquad \bar g = \frac14\sum_b g_b
$$

$\alpha=$ `aux_gradnorm_alpha` $=0.5$ (a **root-level** config field, read identically by all
three stages, not Stage-1-only). At $\alpha=0$ the update is a no-op and the fixed vector stands
exactly. A branch absent or with non-positive norm keeps its current weight.

Its **base** weight (before GradNorm scaling) decays linearly with Stage-1 progress, floored
above zero:

$$
w_{\text{aux}}(e) = \max\big(w_{\text{final}},\,w_{\text{init}}(1-0.7\,e/E)\big), \qquad w_{\text{init}}=0.65,\; w_{\text{final}}=0.25
$$

reaching the floor at $e\approx352$ of 400. Because `progress` is $e/E$, this weight is a
function of the **stage budget** and not only of the epoch, so `stage1.epochs` is an input to
epoch 1's loss value, not merely to the stage's total duration. Stage 2 uses no such decay (aux
loss enters at a fixed relative weight within the classification/contrastive convex
combination); Stage 3 uses a fixed $w_{\text{aux}}=0.10$, no schedule.

### Mixup (`losses/mixup.py`)

$$
\lambda\sim\mathrm{Beta}(\alpha,\alpha),\;\alpha=0.35, \qquad
\tilde x=\lambda x+(1-\lambda)x_\pi, \qquad
\mathcal{L}=\lambda\mathcal{L}_{\text{crit}}(z,y)+(1-\lambda)\mathcal{L}_{\text{crit}}(z,y_\pi)
$$

Side inputs (mask, morphometrics) are mixed identically, with the same $\pi,\lambda$. Stage 1
Phases 1–2 only. **Mutually exclusive with a non-zero ArcFace margin** — `train_one_epoch` raises
`ValueError` if `use_mixup and (arc_m is None or arc_m > 0.0)`; since Phase 1–2 run at
`arc_m=0.0`, the guard passes.

### Same-class CutMix — label-preserving, not a loss term

Two data-level operators, applied inside `RiceSeedDataset.__getitem__` before the batch is ever
formed — **not** a loss-function change. A same-class partner from the same split is drawn
(uniform over the class's pool minus the anchor, one draw, no rejection sampling; classes with
fewer than 2 members in the split are skipped):

$$
\text{spectral: } x_{[t:t+w]} \leftarrow x^{\text{partner}}_{[t:t+w]}, \quad w=\texttt{cutmix\_bands}=8
$$
$$
\text{spatial: } x_{[:,\,r:r+s,\,c:c+s]} \leftarrow x^{\text{partner}}_{[:,\,r:r+s,\,c:c+s]}, \quad s=\texttt{cutmix\_spatial}=24
$$

with window/region start drawn uniformly at random each call. Because the partner is
**same-class**, the label is untouched — no soft target is produced, so unlike mixup it composes
freely with a non-zero ArcFace margin (the mixup/ArcFace exclusion guard does not fire for
CutMix). Probability of firing is a per-augmentation-profile pair
(`spec_cutmix`, `spat_cutmix`), present in every profile except `none`:

| Profile | `spec_cutmix` | `spat_cutmix` |
|---|---:|---:|
| `heavy` | 0.10 | 0.10 |
| `medium` | 0.08 | 0.08 |
| `very_light` | 0.06 | 0.06 |
| `light` | 0.06 | 0.06 |
| `none` | — | — |

Guards are appended **after** the five original augmentation draws (band drop, cutout, noise,
warp, multiplicative) and short-circuit on their probability before drawing from the RNG, so a
profile with CutMix off reproduces the exact RNG stream a CutMix-free profile would. This gives
Stage 2/3 — which otherwise run with no mixing regulariser at $\sim$67 samples/class — some
intra-class variation without ever touching a label.

### Compound totals

**Stage 1, Phases 1–2** (mixup on, `arc_m=0`):

$$
\mathcal{L}_{\text{total}} = \big[\lambda\mathcal{L}_{\text{CE}}(z,y)+(1{-}\lambda)\mathcal{L}_{\text{CE}}(z,y_\pi)\big]
+ w_{\text{aux}}(e)\sum_b\omega_b\big[\lambda\mathcal{L}_{\text{CE}}(z_b,y)+(1{-}\lambda)\mathcal{L}_{\text{CE}}(z_b,y_\pi)\big]
$$

**Stage 1 Phase 3 and Stage 2** (contrastive path, mixup off, non-zero margin from Phase-3
onward / Stage 2):

$$
\boxed{\;
\mathcal{L}_{\text{total}} = \big(1-\lambda_{\text{sc}}-\lambda_{\text{pt}}\big)\mathcal{L}_{\text{cls}}
+ \lambda_{\text{sc}}\mathcal{L}_{\text{supcon}} + \lambda_{\text{pt}}\mathcal{L}_{\text{proto}}
+ w_{\text{aux}}(e)\,\mathcal{L}_{\text{aux}} + w_{\text{bal}}\,\mathcal{L}_{\text{balance}}
\;}
$$

| | $\mathcal{L}_{\text{cls}}$ | $\lambda_{\text{sc}}$ | $\lambda_{\text{pt}}$ |
|---|---|---:|---:|
| Stage 1 Phase 3 | Focal($\gamma{=}1.5$) on $m{=}0$ head logits | 0.35 | 0.15 |
| Stage 2 | Focal($\gamma{=}1.5$) on warmed/per-class-margin logits | $0.40\min(1,e/10)$ | $0.18\min(1,e/10)$ |

**Stage 3** (SAM ascent and descent steps both evaluate this identical objective; ProtoNCE weight
is defined but the term is not actually applied, per above):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}(\gamma{=}1) + 0.02\,\mathcal{L}_{\text{supcon}} + 0.10\,\mathcal{L}_{\text{aux}} + w_{\text{bal}}\,\mathcal{L}_{\text{balance}}
$$

---

## 4.5 Optimisation rules

### Weight-decay parameter groups

`optim/param_groups.py::_wd_groups` splits by a single rule:

$$
p\in\text{no-decay} \iff \operatorname{ndim}(p)=1 \;\lor\; \text{name ends } \texttt{.bias}
$$

placing every normalisation affine parameter and every bias at `weight_decay = 0.0`; all 2-D+
weights get $2\times10^{-4}$. Buffers (e.g. `PhysicalWavelengthPE.pe`) receive neither gradient
nor decay by construction. `build_optimizer_s1`/`s3`: 2 groups, single LR. `build_optimizer_s2`:
4 groups, split on the `arcface_head` name prefix into head/backbone $\times$ wd/no-wd — **group
order is load-bearing**, `stage2_arcface.py` reads `param_groups[0]`/`[2]` back for logging.

### Per-group gradient clipping

Clipping is applied independently within each of three named groups — not as a single global
norm over the whole model — so the ArcFace head's $s{=}48$-amplified gradient cannot dominate
the total norm and divide the effective learning rate of everything else:

$$
\texttt{CLIP\_GROUPS} = \big\{\text{head}: \{\texttt{arcface\_head.}\},\;\; \text{fusion}: \{\texttt{cross\_interaction.},\,\texttt{embed\_net.}\},\;\; \text{backbone}: \text{everything else}\big\}
$$

Each group is clipped independently to `grad_clip = 1.0` via `nn.utils.clip_grad_norm_`, applied
at the accumulation boundary (Stages 1–2) or before both SAM steps (Stage 3); pre-clip per-group
norms are what the diagnostic channel reports (§5.2).

### Mixed precision

$$
\texttt{use\_amp} = (\texttt{supcon is None}) \land (\texttt{scaler is not None})
$$

AMP is silently disabled whenever a SupCon module is passed — Stage 1 Phase 3 and all of Stage 2
always pass one, so both run in full fp32; only Stage 1 Phases 1–2 (no SupCon) train under
`autocast` with a `GradScaler` **explicitly bound to the active device**
(`GradScaler(device=device.type)` — a bare `GradScaler()` binds to CUDA and silently becomes a
no-op pass-through on any other accelerator, disabling loss scaling without an error). Stage 3
passes no scaler at all and is fp32 by construction (SAM's two-pass ascent/descent contract is
fundamentally incompatible with per-step loss rescaling). `engine/evaluate.py` and `engine/tta.py`
both force `autocast(enabled=False)` unconditionally, so a reported metric never depends on the
caller's AMP state.

### EMA

$$
d_n = \min\Big(d_{\max},\,\frac{1+n}{10+n}\Big), \qquad \theta^{\text{EMA}}\leftarrow d_n\theta^{\text{EMA}}+(1-d_n)\theta
$$

$d_{\max}=$ `ema_decay` $=0.999$, $n$ the update count (advanced once per *optimiser* step, not
per micro-batch — `accum=1` shipped, so these coincide). Floating-point buffers are copied
outright, not averaged; the update is implemented with multi-tensor `torch._foreach_*` ops (2
kernel launches for the whole model rather than one per tensor), spelled out as an explicit
`mul_` then `add_` of a materialised `(1-d)\theta` rather than a fused `add_(alpha=)`, to keep
bit-identical rounding to a naive per-parameter loop. `reinit_from` hard-resets the shadow and
restarts the decay warm-up; called at both Stage-1 phase boundaries and unconditionally on entry
to Stage 2 and Stage 3. **`ModelEMA.update` requires the unwrapped module** — handing it a
DDP- or `torch.compile`-wrapped model raises `RuntimeError` (its parameter matching is by name),
rather than silently failing to track.

### Device selection

`cfg.device = "auto"` resolves **Metal (MPS) → CUDA → CPU**; an explicit `cuda`/`cpu`/`mps` is
never overridden. Execution-performance knobs (worker counts, `torch.compile`, TF32,
DDP topology) live entirely under `cfg.runtime` and are covered in
`06_EXECUTION_AND_HARDWARE.md` — none of them are permitted to change a reported metric.
