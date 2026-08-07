# 3 · Model Architecture

Notation throughout: $B$ = batch, $C = 40$ bands, $H = W = 64$, $N_g = 16$ grid cells
($4\times4$), $D = 256$ branch/embedding width, $K = 90$ classes. Grid-flattened tensors are
written with leading dimension $BN_g = 16B$. All shapes are verified against a live traced
forward pass of the shipped configuration.

---

## 3.1 Forward contract

`SpectralQuadNet.forward(x, labels=None, return_embed=False, arc_m=None, branch_mask=None)`
returns a *different type per mode*, and every caller in `engine/` branches on exactly this
shape — it is contract, not implementation detail:

| Mode | `return_embed` | Return |
|---|---|---|
| `.train()` | `False` | `{"main", "aux_a", "aux_b", "aux_c", "aux_d"}`, each $(B, 90)$ |
| `.train()` | `True` | the above plus `"emb"` $= \hat{\mathbf{e}} \in \mathbb{R}^{B\times256}$ |
| `.eval()` | `False` | logits $(B, 90)$ |
| `.eval()` | `True` | `(logits, emb)` |

`tests/regression/test_golden_forward_pass.py::test_training_mode_returns_auxiliary_logits`
pins the five training-mode keys and their shapes.

### Control API

| Method | Effect |
|---|---|
| `use_arcface(flag)` | routes the head: `linear_head` (Stage 1) vs `arcface_head` (Stage 2+); persisted in the checkpoint as `use_arcface` |
| `set_dropout(p)` | sets `p` on **every** `nn.Dropout` module uniformly, overriding construction-time values |
| `freeze_head(which)` / `unfreeze_head(which)` | toggles `requires_grad` on one head, which also removes/adds it from the optimiser's parameter groups |

`set_dropout` does **not** affect `nn.MultiheadAttention`'s internal dropout, which is a float
attribute rather than a module — attention dropout stays at its construction value
(`fusion_drop = 0.10` in fusion, `0.10` in SpecFormer) for the whole run.

---

## 3.2 Shared front-end

### Masked spectral ECA (`MaskedSpectralECA`)

Applied to the raw cube before any branch. It is a *residual excitation* gate: weights lie in
$(1, 2)$, so a band can be amplified but never deleted — the failure mode a standard SE block
has on hyperspectral input.

With the foreground mask $m_{b,h,w} = \mathbb{1}\big[\sum_c |x_{b,c,h,w}| > 10^{-5}\big]$ and
$n_b = \max(\sum_{h,w} m_{b,h,w},\, 10^{-5})$:

$$
\mu_{b,c} = \frac{1}{n_b}\sum_{h,w} x_{b,c,h,w}\, m_{b,h,w},
\qquad
\nu_{b,c} = \max_{(h,w)\,:\,m=1} x_{b,c,h,w}
$$

$$
\mathbf{y}_b = \operatorname{stack}(\mu_b, \nu_b) \in \mathbb{R}^{2\times C},
\qquad
g_{b,c} = \sigma\big(\operatorname{Conv1d}_{2\to1,\,k}(\mathbf{y}_b)\big)_c,
\qquad
x' = x + x \odot g
$$

The kernel width follows the ECA adaptive rule
$t = \big\lfloor\,|\log_2 C / 2 + 1|\,\big\rfloor$, $k = t$ if odd else $t+1$; at $C = 40$ this
gives $k = 3$, i.e. **6 parameters for the entire block**. The $\mathrm{Conv1d}$ over the band
axis is what makes it *spectral*: adjacent bands gate each other.

### Regional grid spectra (`extract_grid_spectra`, stateless)

Background-excluding adaptive pooling into a $4\times4$ grid — each cell's mean is taken over
its *valid* pixels only:

$$
G_{b,(i,j),c} \;=\; \frac{\operatorname{AvgPool}_{4\times4}\!\big(x'\odot m\big)_{b,c,i,j}}
                          {\max\!\big(\operatorname{AvgPool}_{4\times4}(m)_{b,i,j},\; 10^{-5}\big)}
\;\in\; \mathbb{R}^{B \times 16 \times 40}
$$

Cells that are entirely background clamp to $0$ rather than dividing by zero.

### Masked spectral statistics (`masked_spectral_stats`, stateless)

Nine per-band statistics over foreground pixels only, each $(B, 40)$, computed in `float32`
regardless of autocast state. With $\mathrm{flat} \in \mathbb{R}^{B\times C\times HW}$,
$n_b = \max(\sum_p m_{b,p}, 1)$ and $\delta_{b,c,p} = (\mathrm{flat}_{b,c,p} - \mu_{b,c})\,m_{b,p}$:

| # | Statistic | Definition |
|---|---|---|
| 1 | mean | $\mu_{b,c} = \frac{1}{n_b}\sum_p \mathrm{flat}_{b,c,p}\, m_{b,p}$ |
| 2 | std | $\sigma_{b,c} = \sqrt{\frac{1}{n_b}\sum_p \delta^2 + 10^{-5}}$ |
| 3 | max | $\max_{p:\,m=1}\mathrm{flat}_{b,c,p}$, with the all-background case forced to $0$ |
| 4 | skewness | $\operatorname{clamp}\!\Big(\frac{n_b^{-1}\sum_p \delta^3}{\sigma^3 + 10^{-4}},\; -10,\; 10\Big)$ |
| 5 | kurtosis | $\operatorname{clamp}\!\Big(\frac{n_b^{-1}\sum_p \delta^4}{\sigma^4 + 10^{-4}},\; 0,\; 20\Big)$ — raw, not excess |
| 6–9 | $p_{10}, p_{25}, p_{75}, p_{90}$ | order statistic at index $\min\!\big(\lfloor n_b\, q \rfloor,\; HW-1\big)$ of the foreground-only sorted values (background filled with $+\infty$ so it sorts to the tail) |

Every output passes through `nan_to_num(·, 0)`. Because $\delta$ is masked, background pixels
contribute exactly zero to all moment sums — the property the extraction pipeline's
"zero outside the component" invariant (§2.3) exists to guarantee.

### Physical wavelength positional encoding (`PhysicalWavelengthPE`)

Shared by branches A and B (one module instance, referenced from three attribute paths).
With $d = 96$, $h = d/2 = 48$:

$$
\omega_j = \exp\!\Big(-j\,\frac{\ln 10^4}{h-1}\Big),\quad j = 0,\dots,47
\qquad
E_{\mathrm{wl}} \in \mathbb{R}^{40\times96},\quad
E_{\mathrm{wl}}[i,:] = \big[\sin(\tilde\lambda_i \boldsymbol{\omega}) \,\|\, \cos(\tilde\lambda_i \boldsymbol{\omega})\big]
$$

and the application is additive over the channel axis of a 1-D feature map:
$F \leftarrow F + E_{\mathrm{wl}}^{\top}$, broadcasting $(1, 96, 40)$ over $(\cdot, 96, 40)$.
$E_{\mathrm{wl}}$ is a registered buffer: **it is part of `state_dict()`** and appears in every
checkpoint.

### Reusable blocks

| Block | Structure |
|---|---|
| `SEBlock1D(C, r{=}8)` | $x \odot \sigma\big(W_2\,\mathrm{GELU}(W_1\,\mathrm{GAP}(x))\big)$, hidden $=\max(C/8, 8)$ |
| `ResBlock1D(C_i, C_o, k, \text{dil})` | $\mathrm{GELU}\big(\mathrm{SE}(\mathrm{GN}(\mathrm{Conv}_k(\mathrm{GELU}(\mathrm{GN}(\mathrm{Conv}_k(x)))))) + \mathrm{skip}(x)\big)$, `GroupNorm(1, ·)` |
| `LargeKernelBlock1D(d, k)` | ConvNeXt-style: depthwise $\mathrm{Conv}_k \to \mathrm{GN}(1,d) \to 1{\times}1\,(d{\to}4d) \to \mathrm{GELU} \to 1{\times}1\,(4d{\to}d) \to \mathrm{SE} \to + x$ |
| `ResBlock2D(C_i, C_o, s)` | bottleneck $1{\times}1 \to 3{\times}3_{(s)} \to 1{\times}1$, `GroupNorm(min(8,·))`, GELU, projected skip; $\text{mid} = \max(C_o/2, C_i)$ |
| `CBAM(c, r{=}8)` | channel then spatial gating (§3.3, Branch C) |

---

## 3.3 The four branches

### Branch A — Spectral Profile (`SpectralProfileBranch`)

**Input:** the 16 regional mean spectra, flattened to $(16B, 40)$ — the branch processes each
grid cell *independently* and the 16 embeddings are averaged afterwards, so it is a
permutation-invariant regional descriptor.

Pipeline: a single-channel stem lifts the raw spectrum to 96 channels, $E_{\mathrm{wl}}$ is
added, three parallel `LargeKernelBlock1D` towers with kernel widths $\{3, 5, 7\}$ read the
same stem output at different receptive fields (two blocks each), their outputs are
concatenated and fused back to 96 channels, and the band axis is collapsed by **learned
attention pooling**:

$$
\mathbf{w} = \operatorname{softmax}_{\lambda}\!\big(\mathrm{Conv}_{96\to1}(\mathrm{GELU}(\mathrm{Conv}_{96\to24}(F)))\big),
\qquad
\mathbf{z} = \sum_{\lambda} F_{:,:,\lambda}\, \mathbf{w}_{:,\lambda}
$$

$\mathbf{z} \in \mathbb{R}^{16B \times 96}$ is projected to $D$ by
$\mathrm{Linear} \to \mathrm{LayerNorm} \to \mathrm{GELU} \to \mathrm{Dropout}(0.15)$, then
reduced over the grid:
$\mathbf{b}_A = \frac{1}{16}\sum_{n=1}^{16} \mathbf{z}^{(n)} \in \mathbb{R}^{B\times256}$.

Depthwise large kernels are used instead of dilated convolutions specifically because
dilation leaves blind spots across wide absorption valleys.

### Branch B — Spectral Statistics (`SpectralStatsBranch`)

**Input:** the nine masked statistics, stacked to $(B, 9, 40)$ — a *global* seed descriptor
(no grid). A statistic-level gate re-weights the nine channels before anything else:

$$
S \leftarrow S \odot \sigma\big(W_2\,\mathrm{GELU}(W_1\,\mathrm{GAP}_\lambda(S))\big),
\qquad W_1: 9\to16,\; W_2: 16\to9
$$

so the network learns *which moments matter* rather than being forced to use all nine
equally. A $1\times1$ projection lifts $9 \to 96$ channels, $E_{\mathrm{wl}}$ is added, three
`ResBlock1D` towers with kernels $\{1, 3, 5\}$ run in parallel (two blocks each), the
concatenation is fused by two more `ResBlock1D(k{=}5)`, and the same softmax attention pooling
and projection as Branch A produce $\mathbf{b}_B \in \mathbb{R}^{B\times256}$.

The branch is band-count agnostic: `in_channels` is fixed at $9$ — the moment count — not at
`num_bands`, which is accepted for signature symmetry and unused.

### Branch C — Spatial CNN (`SpatialCNNBranch`)

**Input:** the full gated cube $(B, 40, 64, 64)$ — the only branch that sees spatial texture.

*Band reduction* is a depthwise $1\times1$ (`groups=num_bands`, one scalar per band) followed
by a dense $1\times1$ to 64 channels, so per-band scaling is learned separately from
cross-band mixing.

*Backbone*: four `ResBlock2D` stages, each stride 2, with `CBAM` after the first three:

$$
64{\times}64 \;\to\; 128{\times}32^2 \;\to\; 192{\times}16^2 \;\to\; 256{\times}8^2 \;\to\; 256{\times}4^2
$$

CBAM is sequential channel-then-spatial gating:

$$
x \leftarrow x \odot \sigma\big(\mathrm{MLP}(\operatorname{avgpool}_{hw} x) + \mathrm{MLP}(\operatorname{maxpool}_{hw} x)\big)
$$
$$
x \leftarrow x \odot \sigma\big(\mathrm{Conv}_{7\times7}\big([\operatorname{mean}_c x \,\|\, \operatorname{max}_c x]\big)\big)
$$

with a shared two-layer MLP $c \to \max(c/8, 8) \to c$ for the channel stage.

*Pooling* uses signed power normalisation, which is stable under the heavy-tailed activations
a ResNet stack produces, followed by $\ell_2$ normalisation of the concatenation:

$$
\operatorname{pn}(u) = \operatorname{sign}(u)\sqrt{\max(|u|, 10^{-8})},
\qquad
\mathbf{b}_C = \mathrm{proj}\Big(\tfrac{\mathbf{v}}{\|\mathbf{v}\|_2}\Big),\quad
\mathbf{v} = \big[\operatorname{pn}(\operatorname{mean}_{hw} h) \,\|\, \operatorname{pn}(\max_{hw} h)\big] \in \mathbb{R}^{512}
$$

with $\mathrm{proj} = \mathrm{Linear}(512{\to}256) \to \mathrm{BatchNorm1d} \to \mathrm{GELU}$.

### Branch D — SpecFormer (`SpecFormerBranch`)

**Input:** the same $(B, 16, 40)$ grid spectra as Branch A, but factorised into two attention
stages: *spectral within a cell*, then *spatial across cells*.

**Multi-scale tokenisation.** Each cell's spectrum is tokenised by three parallel strided
1-D convolutions with kernels $\{3,5,7\}$ (narrow kernels resolve sharp absorption lines,
wide kernels carry broad shape). With $d_{\text{model}} = 256$ the channel split is
$\lfloor 256/3 \rfloor = 85, 85, 86$; stride is `specf_patch // 2` $= 4$, giving

$$
L = \left\lfloor \frac{40 + 2p - k}{4} \right\rfloor + 1 = 10 \quad \text{for all three kernels}
$$

(the streams are additionally truncated to the shortest length, a no-op here). The
concatenation passes `GroupNorm(1, 256)` and GELU.

**Stage D1 — spectral.** A learned `spec_cls` token is prepended, giving 11 tokens; a learned
positional embedding of capacity $\lfloor 40/4 \rfloor + 2 = 12$ is added over the first 11
positions. Two pre-LN encoder blocks run, and **token 0 becomes the cell's spectral summary**.

**Stage D2 — spatial.** The 16 cell summaries are re-assembled to $(B, 16, 256)$, a learned
`spatial_cls` is prepended (17 tokens), two more pre-LN blocks run, and token 0 — now a
spatio-spectral object descriptor — is `LayerNorm`ed and projected
$\mathrm{Linear}(256{\to}256) \to \mathrm{BatchNorm1d} \to \mathrm{GELU}$.

Each pre-LN block is standard:

$$
x \leftarrow x + \mathrm{Drop}\big(\mathrm{MHSA}(\mathrm{LN}(x))\big),
\qquad
x \leftarrow x + \mathrm{Drop}\big(\mathrm{FF}(\mathrm{LN}(x))\big)
$$

with 8 heads, $d_{\text{ff}} = 2d = 512$, dropout $0.10$. The layer budget splits evenly:
`specf_layers = 4` $\Rightarrow$ 2 spectral + 2 spatial blocks.

`physical_wl` and `patch_size` are accepted by this branch and **unused** — the tokenizer is
stride-based and carries no wavelength encoding.

### Branch masking (training-time regularisation)

Before fusion, branch embeddings can be zeroed. With an explicit `branch_mask` (used by the
influence diagnostic, §5.2) the mask is applied verbatim. Otherwise, in training mode:

$$
p = (0,\; 0,\; 0.30,\; 0.20),
\qquad
k_b = \mathbb{1}[u_b > p_b],\quad u_b \sim \mathcal{U}(0,1)
$$
$$
k \leftarrow \max\big(k,\; \operatorname{onehot}(s)\big),\quad s \sim \mathcal{U}\{0,1,2,3\}
\qquad
\mathbf{b}_x \leftarrow \mathbf{b}_x \cdot k_x
$$

The "safe index" guarantees at least one branch always survives. Only the *fused* path is
masked — the auxiliary heads always read the **unmasked** embeddings $\mathbf{b}^{\text{raw}}$,
so deep supervision keeps flowing to a dropped branch. In `.eval()` no masking occurs and no
RNG is consumed, which is what makes eval-mode forwards bit-deterministic.

> `model.branch_drop_prob` (config `model.branch_drop_prob = 0.20`) is stored on the module
> and set to $0$ by Stage 3, but the forward pass uses the hardcoded vector above; the
> attribute does not currently influence the drop probabilities.

---

## 3.4 Fusion

### `CrossModalInteraction` — Perceiver-style latent cross-attention

The four branch embeddings are LayerNormed **independently** (preserving modality structure)
and stacked into $T \in \mathbb{R}^{B\times4\times256}$. A learned latent array
$L \in \mathbb{R}^{4\times256}$ (initialised $\mathcal{N}(0, 0.02^2)$) is broadcast over the
batch, and $\text{depth} = 2$ blocks of

$$
L \leftarrow L + \mathrm{MHA}(Q{=}L,\; K{=}T,\; V{=}T) \qquad\text{(cross: latents query modalities)}
$$
$$
L \leftarrow L + \mathrm{MHA}(Q{=}K{=}V{=}L) \qquad\text{(latent self-attention)}
$$
$$
L \leftarrow L + \mathrm{FF}\big(\mathrm{LN}(L)\big),\quad \mathrm{FF}: 256 \to 1024 \to 256
$$

run with 8 heads and dropout $0.10$. The latents are pooled and combined with a
**Mixture-of-Experts style modality gate**:

$$
\mathbf{f} = \frac{1}{4}\sum_{n=1}^{4} L_n,
\qquad
\mathbf{g} = \operatorname{softmax}\big(W_2\,\mathrm{GELU}(W_1 \mathbf{f})\big) \in \Delta^3,
\qquad
\mathbf{f} \leftarrow \mathbf{f} + \sum_{m=1}^{4} g_m T_m
$$

so the fused token carries both the cross-attended latent summary and a *directly gated*
convex combination of the modalities — the residual path that keeps a single strong branch
usable even if cross-attention collapses. Output is
$\mathrm{Dropout}(\mathrm{GELU}(W_o \mathrm{LN}(\mathbf{f})))$.

Attention scales with the number of modalities, not their product, so adding a fifth branch
costs one extra key/value token.

> Config `model.fusion_heads = 4` is declared in the schema and YAML but **not passed** at the
> construction site; the module's default of 8 heads is what runs. `model.fusion_drop = 0.10`
> *is* wired.

### `EmbedNet` — pre-norm residual refinement

$$
\mathbf{e} = \mathrm{LN}_2\Big(\mathbf{u} + \mathrm{Drop}\big(\mathrm{MLP}(\mathrm{LN}_1(\mathbf{u}))\big)\Big),
\qquad \mathrm{MLP}: 256 \to 512 \to 256
$$

$\mathbf{e}$ is the embedding consumed by the classification heads, by SupCon/ProtoNCE (as
$\hat{\mathbf{e}} = \mathbf{e}/\|\mathbf{e}\|_2$) and by the t-SNE figure.

---

## 3.5 Classification heads

### Linear head (Stage 1)

$\mathrm{GELU} \to \mathrm{Dropout}(0.4\,p) \to \mathrm{Linear}(256 \to 90)$, where
$p$ = `stage1.dropout` $= 0.15$, so the head's own dropout is $0.06$ at construction.

### Adaptive sub-centre ArcFace head (Stage 2+)

Weight $\mathbf{W} \in \mathbb{R}^{(90 \cdot 3)\times256}$ (`xavier_uniform_`), i.e. $K = 3$
sub-centres per class; a `margins` buffer of shape $(90,)$ initialised to $m_{\text{base}}$
travels in the checkpoint.

**Cosine logit** — both inputs are $\ell_2$-normalised and the class score is the *max* over
its sub-centres:

$$
\cos\theta_{i,c} = \max_{k\in[3]} \operatorname{clamp}\!\big(\hat{\mathbf{e}}_i^{\top}\hat{\mathbf{W}}_{c,k},\, -1{+}10^{-6},\, 1{-}10^{-6}\big)
$$

**Inference / no labels:** $\;\text{logits} = s\,\cos\theta$, with $s = 48$.

**Training with labels:** the per-sample margin is $m_i = $ `global_m` if supplied (margin
warm-up) else $M(y_i) = \mathbf{margins}[y_i]$. Writing $c_i = \cos\theta_{i,y_i}$ and
$s_i = \sqrt{\max(1 - c_i^2, 10^{-6})}$:

$$
\phi_i \;=\; c_i\cos m_i - s_i \sin m_i \;=\; \cos(\theta_{i,y_i} + m_i)
$$

$$
\phi_i \;\leftarrow\;
\begin{cases}
\phi_i, & c_i > \cos(\pi - m_i)\\[2pt]
c_i - m_i \sin(\pi - m_i), & \text{otherwise}
\end{cases}
$$

$$
\text{logits}_{i,c} \;=\; s\big(\mathbb{1}[c = y_i]\,\phi_i + \mathbb{1}[c \ne y_i]\cos\theta_{i,c}\big)
$$

The second line is the standard "easy-margin" guard: past $\theta + m > \pi$ the cosine is no
longer monotone, so the penalty is replaced by a linear one, preserving a usable gradient.

**Adaptive per-class margin.** `update_margins_from_f1(class_f1)` sets

$$
M(c) = m_{\text{base}} + m_\Delta\big(1 - \min(F_1^{(c)}, 1)\big) \in [0.35,\, 0.45]
$$

Classes absent from the dict keep their current margin.

**Bootstrap from the linear head.** `init_from_linear(W_{\text{lin}})` copies the Stage-1 head's
row-normalised weights into all $K$ sub-centres with increasing jitter, so sub-centres start
near a known-good boundary yet are not identical:

$$
\mathbf{W}_{c,k} \;\leftarrow\; \frac{\mathbf{w}^{\text{lin}}_c}{\|\mathbf{w}^{\text{lin}}_c\|_2} + 0.01\,k\,\boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0, I),\quad k = 0,1,2
$$

Sub-centre $k=0$ is therefore an exact copy of the normalised linear head.

### Auxiliary heads (deep supervision)

Four identical `AuxiliaryHead`s, $\mathrm{Linear}(256{\to}128) \to \mathrm{GELU} \to
\mathrm{Linear}(128{\to}90)$, initialised `trunc_normal_(std=0.02)` with zero bias
(conservative, to avoid saturating the softmax early). They are called **only in training
mode** and read the *unmasked* branch embeddings. Their loss weighting is in §4.3.

---

## 3.6 Tensor shape matrix

Input contract: $x \in \mathbb{R}^{B \times 40 \times 64 \times 64}$, `float32`.

### Shared front-end

| Stage | Module | Input | Output |
|---|---|---|---|
| Spectral gate | `se` (`MaskedSpectralECA`) | $(B, 40, 64, 64)$ | $(B, 40, 64, 64)$ |
| ↳ gate conv | `se.conv` (`Conv1d(2→1, k=3)`) | $(B, 2, 40)$ | $(B, 1, 40)$ |
| Grid spectra | `extract_grid_spectra(·, 4)` | $(B, 40, 64, 64)$ | $(B, 16, 40)$ → flat $(16B, 40)$ |
| Masked stats | `masked_spectral_stats` | $(B, 40, 64, 64)$ | $9 \times (B, 40)$ |

### Branch A — Spectral Profile ($BN_g = 16B$)

| Step | Module | Input | Output |
|---|---|---|---|
| A.0 | unsqueeze | $(16B, 40)$ | $(16B, 1, 40)$ |
| A.1 | `stem` (`Conv1d(1→96,k=3)`+GN+GELU) | $(16B, 1, 40)$ | $(16B, 96, 40)$ |
| A.2 | `wl_pe_module` ($+E_{\mathrm{wl}}^{\top}$) | $(16B, 96, 40)$ | $(16B, 96, 40)$ |
| A.3 | `tower_s` / `tower_m` / `tower_l` ($k = 3/5/7$) | $(16B, 96, 40)$ | $3 \times (16B, 96, 40)$ |
| A.4 | concat | $3 \times (16B, 96, 40)$ | $(16B, 288, 40)$ |
| A.5 | `fusion` ($1{\times}1$ + `LargeKernelBlock1D`) | $(16B, 288, 40)$ | $(16B, 96, 40)$ |
| A.6 | `attn_pool` (+ softmax over $\lambda$) | $(16B, 96, 40)$ | $(16B, 1, 40)$ |
| A.7 | weighted sum over $\lambda$ | $(16B, 96, 40)$ | $(16B, 96)$ |
| A.8 | `proj` (Linear+LN+GELU+Drop) | $(16B, 96)$ | $(16B, 256)$ |
| A.9 | reshape + mean over 16 cells | $(B, 16, 256)$ | $\mathbf{b}_A\;(B, 256)$ |

### Branch B — Spectral Statistics

| Step | Module | Input | Output |
|---|---|---|---|
| B.0 | stack 9 statistics | $9\times(B, 40)$ | $(B, 9, 40)$ |
| B.1 | `stat_attn` (gate) | $(B, 9, 40)$ | $(B, 9, 1)$ → applied |
| B.2 | `input_proj` ($1{\times}1$, $9{\to}96$) | $(B, 9, 40)$ | $(B, 96, 40)$ |
| B.3 | `wl_pe_module` | $(B, 96, 40)$ | $(B, 96, 40)$ |
| B.4 | `tower_s` / `tower_m` / `tower_l` ($k = 1/3/5$) | $(B, 96, 40)$ | $3 \times (B, 96, 40)$ |
| B.5 | concat | — | $(B, 288, 40)$ |
| B.6 | `fusion` (2 × `ResBlock1D`, $k{=}5$) | $(B, 288, 40)$ | $(B, 96, 40)$ |
| B.7 | `pool_attn` + weighted sum | $(B, 96, 40)$ | $(B, 96)$ |
| B.8 | `proj` | $(B, 96)$ | $\mathbf{b}_B\;(B, 256)$ |

### Branch C — Spatial CNN

| Step | Module | Input | Output |
|---|---|---|---|
| C.1 | `band_reduce.0` (depthwise $1{\times}1$, groups $=40$) | $(B, 40, 64, 64)$ | $(B, 40, 64, 64)$ |
| C.2 | `band_reduce.1` ($1{\times}1$, $40{\to}64$) + GN(8) + GELU | $(B, 40, 64, 64)$ | $(B, 64, 64, 64)$ |
| C.3 | `stages.0` `ResBlock2D(64→128, s2)` | $(B, 64, 64, 64)$ | $(B, 128, 32, 32)$ |
| C.4 | `stages.1` `CBAM(128)` | $(B, 128, 32, 32)$ | $(B, 128, 32, 32)$ |
| C.5 | `stages.2` `ResBlock2D(128→192, s2)` | $(B, 128, 32, 32)$ | $(B, 192, 16, 16)$ |
| C.6 | `stages.3` `CBAM(192)` | $(B, 192, 16, 16)$ | $(B, 192, 16, 16)$ |
| C.7 | `stages.4` `ResBlock2D(192→256, s2)` | $(B, 192, 16, 16)$ | $(B, 256, 8, 8)$ |
| C.8 | `stages.5` `CBAM(256)` | $(B, 256, 8, 8)$ | $(B, 256, 8, 8)$ |
| C.9 | `stages.6` `ResBlock2D(256→256, s2)` | $(B, 256, 8, 8)$ | $(B, 256, 4, 4)$ |
| C.10 | concat $\operatorname{pn}$(mean), $\operatorname{pn}$(max), then $\ell_2$-normalise | $(B, 256, 4, 4)$ | $(B, 512)$ |
| C.11 | `proj` (Linear+BN1d+GELU) | $(B, 512)$ | $\mathbf{b}_C\;(B, 256)$ |

### Branch D — SpecFormer ($BN_g = 16B$)

| Step | Module | Input | Output |
|---|---|---|---|
| D.0 | reshape | $(B, 16, 40)$ | $(16B, 1, 40)$ |
| D.1 | `tokenizer.proj_small` ($k{=}3$, $s{=}4$) | $(16B, 1, 40)$ | $(16B, 85, 10)$ |
| D.2 | `tokenizer.proj_medium` ($k{=}5$, $s{=}4$) | $(16B, 1, 40)$ | $(16B, 85, 10)$ |
| D.3 | `tokenizer.proj_large` ($k{=}7$, $s{=}4$) | $(16B, 1, 40)$ | $(16B, 86, 10)$ |
| D.4 | concat + GN(1,256) + GELU | — | $(16B, 256, 10)$ |
| D.5 | transpose | $(16B, 256, 10)$ | $(16B, 10, 256)$ |
| D.6 | prepend `spec_cls` + `spec_pos_embed[:11]` | $(16B, 10, 256)$ | $(16B, 11, 256)$ |
| D.7 | `spectral_blocks.{0,1}` (pre-LN, 8 heads) | $(16B, 11, 256)$ | $(16B, 11, 256)$ |
| D.8 | take token 0, reshape | $(16B, 11, 256)$ | $(B, 16, 256)$ |
| D.9 | prepend `spatial_cls` | $(B, 16, 256)$ | $(B, 17, 256)$ |
| D.10 | `spatial_blocks.{0,1}` | $(B, 17, 256)$ | $(B, 17, 256)$ |
| D.11 | token 0 → `norm` (LN) | $(B, 17, 256)$ | $(B, 256)$ |
| D.12 | `proj` (Linear+BN1d+GELU) | $(B, 256)$ | $\mathbf{b}_D\;(B, 256)$ |

### Fusion, embedding and heads

| Step | Module | Input | Output |
|---|---|---|---|
| F.0 | branch masking (train only) | $4\times(B, 256)$ | $4\times(B, 256)$ |
| F.1 | `cross_interaction.branch_norms.{0..3}` + stack | $4\times(B, 256)$ | $T\;(B, 4, 256)$ |
| F.2 | latents broadcast | $(4, 256)$ | $L\;(B, 4, 256)$ |
| F.3 | $2 \times$ [cross-attn ← $T$; self-attn; FF] | $(B, 4, 256)$ | $(B, 4, 256)$ |
| F.4 | mean over latents | $(B, 4, 256)$ | $(B, 256)$ |
| F.5 | `modality_gate` ($256{\to}64{\to}4$, softmax) | $(B, 256)$ | $(B, 4)$ |
| F.6 | gated modality sum + residual | $(B,4,256), (B,4)$ | $(B, 256)$ |
| F.7 | `output_proj` (LN+Linear+GELU+Drop) | $(B, 256)$ | $(B, 256)$ |
| E.1 | `embed_net` (`EmbedNet`) | $(B, 256)$ | $\mathbf{e}\;(B, 256)$ |
| H.1a | `linear_head` (Stage 1) | $(B, 256)$ | $(B, 90)$ |
| H.1b | `arcface_head` (Stage 2+): $\hat{\mathbf{e}}\hat{\mathbf{W}}^{\top}$ | $(B, 256) \times (270, 256)$ | $(B, 270)$ |
| H.2b | reshape + max over $K$ | $(B, 270)$ | $(B, 90, 3) \to (B, 90)$ |
| X.1–4 | `aux_head_{a,b,c,d}` (train only) | $4\times(B, 256)$ | $4\times(B, 90)$ |
| X.5 | `emb` (when `return_embed`) | $(B, 256)$ | $\hat{\mathbf{e}}\;(B, 256)$ |

---

## 3.7 Parameter budget

Measured on the shipped configuration (`configs/model/spectral_quadnet_v4.yaml`):

| Component | Parameters | Share |
|---|---:|---:|
| `se` (MaskedSpectralECA) | 6 | 0.00 % |
| `wl_pe_cnn` (buffer only) | 0 | 0.00 % |
| `branch_a` — Spectral Profile | 592,753 | 7.5 % |
| `branch_b` — Spectral Stats | 686,424 | 8.7 % |
| `branch_c` — Spatial CNN | 1,694,158 | 21.5 % |
| `branch_d` — SpecFormer | 2,180,866 | 27.7 % |
| `cross_interaction` — fusion | 2,190,916 | 27.8 % |
| `aux_head_{a,b,c,d}` — $4 \times 44{,}506$ | 178,024 | 2.3 % |
| `embed_net` | 263,936 | 3.3 % |
| `linear_head` | 23,130 | 0.3 % |
| `arcface_head` ($270 \times 256$) | 69,120 | 0.9 % |
| **Total (all trainable)** | **7,879,333** | 100 % |

---

## 3.8 Architectural invariants

1. **Attribute names are checkpoint schema.** The 14 top-level names — `se`, `wl_pe_cnn`,
   `branch_{a,b,c,d}`, `cross_interaction`, `aux_head_{a,b,c,d}`, `embed_net`, `linear_head`,
   `arcface_head` — are the keys of every trained checkpoint. Renaming any breaks
   `load_state_dict(strict=True)`; `tests/regression/test_state_dict_compatibility.py` pins them.
2. **Construction order is initialisation.** Every `_init_weights` draws from the same global
   torch RNG stream, in `__init__` order, so reordering sub-module construction changes the
   initial weights. `train.py` documents the required call order at its `set_seed` site:
   *config → `set_seed` → `DataStore` → `SpectralQuadNet` → `ModelEMA`*, with nothing in
   between consuming the global RNG. `test_weight_init_is_bit_identical` hashes all 352
   state-dict tensors against a golden capture.
3. **Weight initialisation policy.** Conv1d/Conv2d: `kaiming_normal_(mode='fan_out',
   nonlinearity='relu')`; norm layers: $\gamma = 1$, $\beta = 0$; Linear:
   `trunc_normal_(std=0.02)` with zero bias; ArcFace weight: `xavier_uniform_`; both CLS
   tokens: `trunc_normal_(std=0.02)`; fusion latents and SpecFormer positional embedding:
   $\mathcal{N}(0, 0.02^2)$.
4. **Buffers travel.** `wl_pe_cnn.pe`, `branch_{a,b}.wl_pe_module.pe` $(40, 96)$ and
   `arcface_head.margins` $(90,)$ are `register_buffer`s and therefore part of `state_dict()`.
5. **`ModelEMA.state_dict()` is the shadow's**, so the shadow must carry the identical key
   structure — asserted for all three stages.

### Config keys not wired to behaviour

Kept in the schema and YAML (the config round-trip gate requires all 81 pre-refactor keys to
have a home), but not read on the path they name:

| Key | Value | Status |
|---|---|---|
| `model.fusion_heads` | 4 | not passed to `CrossModalInteraction`; the module's default of **8 heads** runs |
| `model.specf_drop` | 0.15 | not passed to `SpecFormerBranch`; the call site hardcodes dropout **0.10** |
| `model.wl_embed_dim` | 16 | accepted by `SpectralQuadNet.__init__`, unused in its body |
| `model.branch_drop_prob` | 0.20 | stored on the module; forward uses the hardcoded $(0, 0, 0.30, 0.20)$ |
| `SpectralStatsBranch(num_bands=…)` | 40 | accepted for interface symmetry; the branch is band-count agnostic |
| `SpecFormerBranch(physical_wl, patch_size)` | — | accepted, unused (`specf_patch` *is* used, as $2\times$ the stride) |
