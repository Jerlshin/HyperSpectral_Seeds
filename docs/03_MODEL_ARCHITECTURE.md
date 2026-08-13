# 3 · Model Architecture

Notation throughout: $B$ = batch, $C = 40$ bands, $H = W = 64$. The two grid branches use
**different** grids since Tier 3: Branch A pools onto an $8\times8$ grid
($N_a = 64$ cells, flattened batch $BN_a = 64B$), Branch D onto a $4\times4$ grid
($N_d = 16$ cells, flattened batch $BN_d = 16B$) — before Tier 3 they shared one $4\times4$
grid and received a byte-identical input; separate `grid_size_a`/`grid_size_d` config keys now
make that impossible to reintroduce silently. $D = 256$ is the fusion/embedding width,
$K_{\text{cls}} = 90$ classes, $K_{\text{sub}} = 3$ ArcFace sub-centres per class. All shapes
and parameter counts below are measured from a live `SpectralQuadNet.from_config` build of the
shipped configuration (`configs/model/spectral_quadnet_v4.yaml`), not estimated.

This is the **Tier-3 architectural redesign** (the `T3-*` items referenced throughout; the
planning documents they were tracked in are no longer part of the repository, so this suite plus
`tests/` is the record). It replaced a 4-branch, 7.88 M-parameter model in which two
branches received a byte-identical input, the statistics branch read a provably rank-2 tensor,
no module combined spectral and spatial extent, and fusion was Perceiver-style latent
cross-attention. The current model has **5,194,578 trainable parameters** across four branches
plus a fifth, non-spectral morphology token, fused by a gated low-rank bilinear pool, and
classified by **one** adaptive sub-centre ArcFace head shared across all three curriculum
stages (`04_CURRICULUM_AND_LOSSES.md`).

---

## 3.1 Forward contract

```python
SpectralQuadNet.forward(
    x, labels=None, return_embed=False, arc_m=None,
    branch_mask=None, mask=None, morph=None,
)
```

`x`$\in\mathbb{R}^{B\times40\times64\times64}$ is required; `mask`$\in\mathbb{R}^{B\times64\times64}$
(the persisted fill map $\alpha$) and `morph`$\in\mathbb{R}^{B\times8}$ (the persisted
morphometrics) are both **optional with exact fallbacks** — `mask=None` reproduces the
pre-Tier-3 `\sum_c|x_c|>10^{-5}` threshold bit-for-bit, and `morph=None` substitutes a zero
vector. Neither array exists until `scripts/prepare_dataset.py` is re-run, so the fallbacks are
the operative path for the archived reference run.

There is **no `linear_head`.** HD-1 (T2-10) removed the second head entirely: Stage 1, Stage 2
and Stage 3 all run the same `arcface_head`, differing only in the margin passed at each call
(`stage1.arcface_m = 0.0` makes it a plain cosine/NormFace classifier — §4.1). The forward
return type is a strict function of `self.training` and `return_embed`:

| Mode | `return_embed` | Return |
|---|---|---|
| `.train()` | `False` | `{"main", "aux_a", "aux_b", "aux_c", "aux_d"}`, each $(B, 90)$, plus `"balance"` (scalar) when labels are supplied |
| `.train()` | `True` | the above plus `"emb"` $= \hat{\mathbf{e}} \in \mathbb{R}^{B\times256}$ |
| `.eval()` | `False` | logits $(B, 90)$ |
| `.eval()` | `True` | `(logits, emb)` |

`tests/unit/test_unified_head.py` pins both the absence of `linear_head`/`use_arcface`/
`freeze_head`/`init_from_linear`/`update_margins_from_f1` (all removed with the second head) and
the presence of exactly one head, `arcface_head`.

### Control API

| Method | Effect |
|---|---|
| `set_dropout(p)` | sets `p` on every `nn.Dropout` module uniformly, overriding construction-time values |
| `set_subcentre_tau(tau)` | forwards to `arcface_head.set_tau(tau)` — the soft-to-hard sub-centre pooling temperature (§3.5) |

`set_dropout` does **not** reach `nn.MultiheadAttention`'s internal dropout float, used inside
Branch D's transformer blocks — it walks `nn.Dropout` instances only. This is a known gap
(`engine/checkpoint.py`'s `_STOCHASTIC_MODULES` list routes around it for BatchNorm
re-estimation, §4.3); Branch D's attention dropout stays at its construction value
(`specf_drop = 0.15`) for the whole run.

---

## 3.2 Shared front-end

### Masked spectral ECA (`MaskedSpectralECA`)

Applied to the raw cube **once**, before any branch — a residual excitation gate whose output
lies in $[x, 2x]$ per band, so a band can be amplified but never deleted:

$$
\mu_{b,c} = \frac{\sum_{h,w} x_{b,c,h,w}\, m_{b,h,w}}{\max(\sum_{h,w} m_{b,h,w},\, 10^{-5})},
\qquad
\nu_{b,c} = \max_{(h,w)\,:\,m=1} x_{b,c,h,w}
$$

$$
\mathbf{y}_b = \operatorname{stack}(\mu_b, \nu_b) \in \mathbb{R}^{2\times C},
\qquad
g_{b,c} = \sigma\big(\operatorname{Conv1d}_{2\to1,\,k=3}(\mathbf{y}_b)\big)_c,
\qquad
x' = x + x \odot g
$$

**6 parameters for the entire block** (a $\mathrm{Conv1d}(2\to1, k=3)$, bias-free). Every branch
sees the same post-gate cube $x'$; the gate is not per-branch.

### Foreground mask (`foreground_mask`, `stats_ops.py`)

The single definition of "background" used everywhere in the model:

$$
m_{b,h,w} =
\begin{cases}
\alpha_{b,h,w} & \text{persisted fill map supplied} \\
\mathbb{1}\!\left[\sum_c |x_{b,c,h,w}| > 10^{-5}\right] & \text{fallback}
\end{cases}
$$

When the persisted map is supplied it is used as a **soft weight**, not binarised; the fallback
reproduces the pre-Tier-3 threshold exactly.

### Regional grid spectra (`extract_grid_spectra_multi`, stateless)

Both grids are extracted from **one** shared masked product $x'\odot m$ (formed once,
not per grid), background-excluding adaptive pooling:

$$
G^{(g)}_{b,(i,j),c} = \frac{\operatorname{AvgPool}_{g\times g}(x'\odot m)_{b,c,i,j}}
                            {\max(\operatorname{AvgPool}_{g\times g}(m)_{b,i,j},\,10^{-5})},
\qquad g \in \{8, 4\}
$$

Each call also returns the **cell mass** $\omega^{(g)}_{b,(i,j)} = \operatorname{AvgPool}_{g\times g}(m)_{b,i,j}$
— the per-cell foreground coverage, consumed by Branch A's mass-weighted pool (§3.3).
Grid $8\times8$ ($N_a=64$ cells) feeds Branch A; grid $4\times4$ ($N_d=16$ cells) feeds Branch D.
Cells entirely background clamp to $0$.

### Masked mean spectrum (`masked_mean_spectrum`, stateless)

Branch B's entire spectral input — the foreground mean spectrum, nothing higher-order:

$$
\bar{x}_{b,c} = \frac{\sum_p x_{b,c,p}\, m_{b,p}}{\max(\sum_p m_{b,p},\, 1)} \in \mathbb{R}^{B\times40}
$$

The pre-Tier-3 nine-statistic extractor (`masked_spectral_stats`: mean/std/max/skew/kurtosis/
percentiles) still exists in `stats_ops.py` but is **no longer consumed by any branch** — it was
found to be a provably rank-2 tensor under a per-session gain (§2.2.5 of the plan). Only the
mean survives, computed by the cheaper `masked_mean_spectrum` rather than the full nine-way
reduction.

### Physical wavelength — two independent mechanisms (FE-1)

**`SpectralDerivatives`** (Branch A only) — exact λ-derivatives on the irregular band grid via
local Savitzky–Golay operators, fit once from the wavelength vector:

$$
D_1, D_2 \in \mathbb{R}^{C\times C}, \qquad
\text{shape\_channels}(\bar{x}) = \big[\operatorname{snv}(\bar{x}) \,\|\, D_1\operatorname{snv}(\bar{x}) \,\|\, D_2\operatorname{snv}(\bar{x})\big] \in \mathbb{R}^{B\times3\times C}
$$

$D_1$/$D_2$ are built per-band from its $k{=}7$ nearest neighbours in $\lambda$ by a degree-2
weighted-least-squares fit (exact on quadratics regardless of grid irregularity), then globally
rescaled by the **median** band spacing (and its square) so the operators are unitless. Zero
learnable parameters — `d1_op`/`d2_op` are registered buffers, $(40,40)$ each, and travel in the
checkpoint.

**SNV** (`snv`, applied before the derivative operators):

$$
\operatorname{snv}(\bar{x})_{b,c} = \frac{\bar{x}_{b,c} - \operatorname{mean}_c(\bar{x}_b)}{\operatorname{std}_c(\bar{x}_b) + 10^{-5}}
$$

invariant to $r \mapsto a\,r+b$ for $a>0$ — removes exactly the per-pixel/per-session gain that
made the old nine-moment tensor rank-2.

**`LambdaConv1d`** — the continuous-wavelength kernel generator $\kappa_\phi$, replacing a
learned-per-index convolution with one generated from physical $\Delta\lambda$:

$$
(\mathrm{Conv}_\lambda F)_{o,i} = \sum_{j\in\mathcal{N}_k(i)} \big[\kappa_\phi(\lambda_j-\lambda_i)\big]_{o,c}\, F_{c,j},
\qquad
\kappa_\phi(\delta) = \mathrm{MLP}\big(\mathrm{FourierFeatures}(\delta)\big)
$$

$\mathcal{N}_k(i)$ is the $k{=}5$ nearest bands in $\lambda$ (a buffer, precomputed once);
`FourierFeatures` is a zero-parameter $2n_{\text{freq}}$-wide sinusoidal lift of the signed
offset $\delta$ (log-spaced frequencies over the observed $\Delta\lambda$ span,
$n_{\text{freq}} = $ `wl_embed_dim` $= 16$); the MLP (`Linear(32\to32)\to\mathrm{GELU}\to
\mathrm{LayerNorm}\to\mathrm{Linear}(32\to O{\cdot}I)`) generates the convolution weights **once
per forward, shared across the batch** — they depend only on the wavelength grid, not on $x$.
Parameter cost is **independent of band count** (measured $8\text{k}\text{–}12\text{k}$ params
for a $(3\to96, k{=}5)$ instance); this is what makes Branch A's stem transferable across a
band-count ablation. `nbr`/`offsets` are registered buffers ($40\times5$ each) with the default
`persistent=True`, so they travel in the checkpoint.

### Physical-wavelength positional encoding (`PhysicalWavelengthPE`, `wl_pe_cnn`)

Shared by reference between `self.wl_pe_cnn` and `branch_a.wl_pe_module` — **the same module
object**, so both checkpoint keys carry identical values by construction, not by convention.
With $d=96$, $h=d/2=48$:

$$
\omega_j = \exp\!\Big(-j\,\frac{\ln 10^4}{h-1}\Big),\quad j=0,\dots,47,
\qquad
E_{\mathrm{wl}}[i,:] = \big[\sin(\tilde\lambda_i\boldsymbol\omega)\,\|\,\cos(\tilde\lambda_i\boldsymbol\omega)\big] \in \mathbb{R}^{40\times96}
$$

applied additively over the channel axis of Branch A's 1-D tower. Registered buffer, part of
`state_dict()`.

### Reusable blocks

| Block | Structure |
|---|---|
| `SEBlock1D(C, r{=}8)` | $x\odot\sigma(W_2\,\mathrm{GELU}(W_1\,\mathrm{GAP}(x)))$ |
| `LargeKernelBlock1D(d,k)` | ConvNeXt-style: depthwise $\mathrm{Conv}_k\to\mathrm{GN}(1,d)\to1{\times}1(d{\to}4d)\to\mathrm{GELU}\to1{\times}1(4d{\to}d)\to\mathrm{SE}\to{+}x$ |
| `ResBlock2D(C_i,C_o,s)` | bottleneck $1{\times}1\to3{\times}3_{(s)}\to1{\times}1$, GroupNorm, GELU, projected skip |
| `CBAM(c,r{=}8)` | channel-then-spatial gating (Branch C) |
| `_PreLNBlock` | pre-norm Transformer block: $\mathrm{LN}\to\mathrm{MHSA}\to{+}\to\mathrm{LN}\to\mathrm{GELU\text{-}FFN}\to{+}$ (Branch D) |

---

## 3.3 The four branches

Tier 3's controlling constraint: **each branch must see something the others cannot
reconstruct.**

| Branch | Input after Tier 3 | Unique information |
|---|---|---|
| A | $8\times8$ per-cell SNV spectra + $\partial_\lambda,\partial^2_\lambda$ | spectral *shape*, gain-free |
| B | learned NDI bank + continuum-removed depths + morphometry | ratios and physical size |
| C | the $(40,64,64)$ cube through a 3-D stem | spatial texture $\times$ spectral position |
| D | $4\times4$ raw grid spectra, $\lambda$-uniform tokens | long-range band interactions |

### Branch A — Spectral Profile (`SpectralProfileBranch`)

**Input:** `shape_channels` on the $8\times8$ grid, flattened to $(64B, 3, 40)$ — SNV spectrum
plus its two λ-derivatives, one grid cell at a time (permutation-invariant regional
descriptor; cells cost no extra parameters since they're processed independently).

Pipeline: `LambdaConv1d`-based stem ($3\to96$ channels) $\to$ `wl_pe_cnn` added $\to$ three
parallel `LargeKernelBlock1D` towers at kernel widths $\{3,5,7\}$ (two blocks each) $\to$
concat $\to$ fused back to 96 channels $\to$ softmax attention pooling over the band axis:

$$
\mathbf{w} = \operatorname{softmax}_\lambda\big(\mathrm{Conv}_{96\to1}(\mathrm{GELU}(\mathrm{Conv}_{96\to24}(F)))\big),
\qquad
\mathbf{z} = \sum_\lambda F_{:,:,\lambda}\,\mathbf{w}_{:,\lambda} \in \mathbb{R}^{64B\times96}
$$

projected to $D{=}256$ by $\mathrm{Linear}\to\mathrm{LayerNorm}\to\mathrm{GELU}\to
\mathrm{Dropout}(0.15)$. The 64 cell embeddings are then **mass-weighted** (not simply averaged)
back to one vector per patch — this pooling happens in `SpectralQuadNet.forward`, not inside the
branch:

$$
\mathbf{b}_A = \frac{\sum_{n=1}^{64} \mathbf{z}^{(n)}\,\omega^{(8)}_n}{\max(\sum_n \omega^{(8)}_n,\,10^{-5})} \in \mathbb{R}^{B\times256}
$$

so cells with little or no foreground coverage contribute little or nothing to the pooled
descriptor, rather than diluting it as a plain mean would.

**Measured parameters: 603,089** (stem 10,816 · tower$_{3/5/7}$ 153,024/153,408/153,792 ·
fusion 104,352 · attn\_pool 2,353 · proj 25,344). Buffers: 7,456 elements, all persisted —
`wl_pe_module.pe` $(40,96)$ = 3,840; `derivatives.{d1_op,d2_op}` $(40,40)$ each = 3,200;
`LambdaConv1d`'s `stem.0.{nbr,offsets}` $(40,5)$ each and `features.omega` $(16,)$ = 416.

### Branch B — Index Bank (`SpectralStatsBranch`)

**Input:** the foreground mean spectrum $\bar{x}\in\mathbb{R}^{B\times40}$ plus 8 persisted
morphometrics (zeros if absent) — a *global* seed descriptor, no grid.

**`SoftIndexBank`** — $n_{\text{idx}}{=}64$ learned normalised-difference indices, each a
softmax-weighted numerator/denominator band group rather than a fixed pair, exactly invariant to
a per-pixel/per-session gain:

$$
\pi^\pm_k = \operatorname{softmax}(\theta^\pm_k) \in \Delta^{C-1},
\qquad
u_k = \bar{x}\cdot\pi^+_k,\;\; v_k = \bar{x}\cdot\pi^-_k,
\qquad
z_k = \frac{u_k - v_k}{|u_k|+|v_k|+10^{-6}}
$$

$\theta^\pm\in\mathbb{R}^{64\times40}$ are raw `nn.Parameter`s ($\mathcal{N}(0,0.02^2)$ init,
**not** touched by the model's outer weight-init pass), costing $2\times64\times40=5{,}120$
parameters. $|z_k|\le1$ by construction.

**`ContinuumDepths`** — $n_{\text{depth}}{=}16$ deepest continuum-removed absorption features,
via the upper concave hull (piecewise-chord envelope) of the spectrum:

$$
\text{depth}_c = 1 - \frac{\bar{x}_c}{\operatorname{envelope}(\bar{x})_c},
\qquad
\text{output} = \operatorname{topk}_{16}(\text{depth})
$$

Zero learnable parameters; its $O(C^3)$ interpolation-weight buffers ($(40,40,40)$, four of
them) are **non-persistent** — computed once from the wavelength grid, never written to the
checkpoint.

**Assembly:** $[z\,(64) \,\|\, \text{depth}\,(16) \,\|\, \text{morph}\,(8)] \in \mathbb{R}^{B\times88}$
$\to \mathrm{LayerNorm} \to$ MLP $\mathrm{Linear}(88{\to}256)\to\mathrm{LN}\to\mathrm{GELU}\to
\mathrm{Dropout}(0.15)\to\mathrm{Linear}(256{\to}256)\to\mathrm{LN}\to\mathrm{GELU} \to
\mathbf{b}_B\in\mathbb{R}^{B\times256}$.

**Measured parameters: 94,896** — an $88{,}000\times$-smaller reduction than the pre-Tier-3
branch's 686,424 params spent on a provably rank-2 input.

### Branch C — Spatial CNN (`SpatialCNNBranch`)

**Input:** the full gated cube $(B,40,64,64)$ + mask — the only branch that keeps both spatial
axes and the spectral axis jointly.

**`SpectralSpatialStem3D`** — three `Conv3d` stages fold the spectral axis into the channel
dimension while halving spatial resolution, rather than discarding it via $1{\times}1$
convolutions as the pre-Tier-3 stem did:

$$
(B,1,40,64,64) \xrightarrow{\text{Conv3d}(1\to16,\,k=(7,3,3),\,s=(2,1,1))} (B,16,20,64,64)
\xrightarrow{\text{Conv3d}(16\to32,\,k=(5,3,3),\,s=2)} (B,32,10,32,32)
$$
$$
\xrightarrow{\text{Conv3d}(32\to64,\,k=(5,3,3),\,s=2)} (B,64,5,16,16)
\xrightarrow{\text{reshape}} (B,320,16,16)
\xrightarrow{\text{Conv2d}(320\to192,1)} (B,192,16,16)
$$

each `Conv3d` stage followed by `GroupNorm`+`GELU`; the mask is re-pooled to the current spatial
resolution and multiplied in **after every stage**, so the padded region is exactly zero at
every depth. The fold-in channel width (`stem_channels = 192`) is band-count agnostic — derived
from the folded spectral depth ($\lceil\lceil\lceil40/2\rceil/2\rceil/2\rceil=5$) times 64, not
hardcoded. Measured stem parameters: **178,256**.

**Tail** — four `ResBlock2D` stages, stride 2, `CBAM` after the first three:

$$
16{\times}16 \;(192\text{ ch}) \;\to\; 32{\times}32\,(128) \;\to\; 16{\times}16\,(192) \;\to\; 8{\times}8\,(256) \;\to\; 4{\times}4\,(256)
$$

Pooling by signed power normalisation, concatenated mean+max, $\ell_2$-normalised:

$$
\operatorname{pn}(u)=\operatorname{sign}(u)\sqrt{\max(|u|,10^{-8})},
\qquad
\mathbf{b}_C = \mathrm{proj}\Big(\tfrac{\mathbf{v}}{\|\mathbf{v}\|_2}\Big),
\quad
\mathbf{v}=[\operatorname{pn}(\operatorname{mean}_{hw}h)\,\|\,\operatorname{pn}(\max_{hw}h)]\in\mathbb{R}^{512}
$$

with $\mathrm{proj}=\mathrm{Linear}(512{\to}256)\to\mathrm{BatchNorm1d}\to\mathrm{GELU}$.

**Measured parameters: 2,230,646** (stem 178,256 + stages 1,920,550 + proj 131,840) — up from
1,694,158 pre-Tier-3, a deliberate capacity *increase*: the branch now performs the only joint
spectral-spatial convolution in the network. A synthetic band/space-swap test confirms the 3-D
stem (unlike the old $1{\times}1$ reduction) produces different outputs for cubes differing only
in *which band* a spatial blob occupies.

### Branch D — SpecFormer (`SpecFormerBranch`)

**Input:** the $4\times4$ grid spectra $(B,16,40)$, tokenised on a **wavelength-uniform** axis
rather than a raw band-index stride.

**`LambdaWindowPooling`** — pools $C{=}40$ bands into $n_{\text{tok}}=\lfloor
40/(\texttt{specf\_patch}/2)\rfloor=10$ tokens whose edges are equal-width windows over
$\tilde\lambda\in[0,1]$ (**not** the observed min/max of the selected band subset), so token $t$
means the same physical spectral region regardless of which/how-many bands were selected — the
property that lets a checkpoint trained at one band count load `strict=True` into a branch built
for a different one. A window that catches no band takes its nearest band instead of averaging
to zero. Raises `ValueError` at construction if `physical_wl` is not already normalised to
$[0,1]$.

**`RelativeLambdaBias`** — a learned, per-head additive attention bias keyed on the *difference*
in window centre wavelengths:

$$
b_\psi(\bar\lambda_t - \bar\lambda_u) = \mathrm{MLP}\big(\mathrm{FourierFeatures}(\bar\lambda_t-\bar\lambda_u)\big) \in \mathbb{R}^{n_{\text{heads}}}
$$

fed to `nn.MultiheadAttention` as an additive `attn_mask`; CLS-token rows/columns are
zero-padded (no wavelength). $\sim$1.3k parameters.

**Positional codes are a sinusoidal buffer keyed on the token centre wavelengths**
(`spec_pos_embed`, the same free function that builds `PhysicalWavelengthPE`), not a learned
table — zero parameters for token position. Both CLS tokens (`spec_cls`, `spatial_cls`) remain
learned parameters, `trunc_normal_(std=0.02)`.

**Multi-scale tokenisation** — three parallel strided-1-D convs at kernels $\{3,5,7\}$ over the
now-uniform 10-token axis, channel split $\lfloor192/3\rfloor{=}64$ each, concatenated $\to$
`GroupNorm`+`GELU`.

**Two-stage attention**, $d_{\text{model}}=192$ (`specf_dim`), 8 heads, dropout $0.15$
(`specf_drop`, now actually wired to the branch): Stage D1 prepends `spec_cls`, adds the position
buffer, runs `specf_layers // 2 = 2` pre-LN blocks with the λ-bias, and token 0 becomes the
cell's spectral summary. Stage D2 re-assembles the 16 cell summaries, prepends `spatial_cls`,
runs 2 more pre-LN blocks (no bias — cells have no natural spatial ordering), and token 0 is
`LayerNorm`ed and projected $\mathrm{Linear}(192{\to}256)\to\mathrm{BatchNorm1d}\to\mathrm{GELU}$.

**Measured parameters: 1,241,640** (tokenizer 1,536 + λ-bias 1,320 + spectral blocks 594,048 +
spatial blocks 594,048 + norm 384 + proj 49,920) — down from 2,180,866 pre-Tier-3 (`specf_dim`
$256\to192$): with the wavelength axis now carried explicitly by token identity and the
relative-λ bias, the branch needs less brute capacity to rediscover it. A checkpoint trained at
40 bands loads `strict=True` into a 20-band instance and their embeddings agree on a spectrum
both can sample, confirming the transfer property the λ-uniform tokenisation is built for.

### Branch masking (training-time regularisation)

$$
\mathbf{p} = 0.20\times(0.75,\,0.75,\,0.0,\,0.75) = (0.15,\,0.15,\,0,\,0.15)
$$

— the **inverse** of the pre-Tier-3 profile. Before Tier 3, Branches C and D (the only
non-duplicated branches) were dropped hardest and A/B (near-duplicates of each other) never at
all; BR-3 inverts this: **drop the branches another branch can reconstruct, never the one that
sees the full cube** (Branch C, factor $0$).

One keep/drop decision per branch is drawn **per batch** (not per sample), forced to keep at
least one branch via a uniformly-random "safe index":

$$
k_b = \mathbb{1}[u_b > p_b],\; u_b\sim\mathcal{U}(0,1)^4,
\qquad
k \leftarrow \max(k,\,\operatorname{onehot}(s)),\; s\sim\mathcal{U}\{0,1,2,3\}
$$

Only the **fused** path is masked ($\mathbf{b}_x \leftarrow \mathbf{b}_x\cdot k_x$) — the
auxiliary heads always read the unmasked $\mathbf{b}^{\text{raw}}$, so deep supervision keeps
flowing to a dropped branch. An explicit `branch_mask` argument (used by the leave-one-branch-out
influence diagnostic, §5.2) overrides this draw verbatim. In `.eval()` no masking occurs and no
RNG is consumed.

---

## 3.4 Fusion

### `MorphologyEmbed` — the fifth modality

$$
\mathbf{b}_E = \mathrm{Linear}(64{\to}256)\big(\mathrm{GELU}(\mathrm{Linear}(8{\to}64)(\text{morph}))\big) \in \mathbb{R}^{B\times256}
$$

17,216 parameters. The eight persisted morphometrics enter fusion as a genuine fifth token,
alongside — not folded into — Branch B.

### `CrossModalInteraction` — gated low-rank bilinear pool

Replaces the pre-Tier-3 Perceiver-style latent cross-attention entirely (`output_proj`,
`latents` and the attention blocks are gone). With $M=5$ modality tokens
$\{\mathbf{b}_A,\mathbf{b}_B,\mathbf{b}_C,\mathbf{b}_D,\mathbf{b}_E\}$:

**Per-modality normalisation** is a **dataset statistic**, not a per-sample rescale —
`BatchNorm1d`, one per modality, so a low-confidence sample's small-norm branch output stays
small in `.eval()` rather than being renormalised to unit scale:

$$
\hat{\mathbf{b}}_m = \mathrm{BatchNorm1d}_m(\mathbf{b}_m), \qquad m=1,\dots,5
$$

**Confidence vector** — the log-norm of each branch *before* normalisation:

$$
\nu_m = \log(\|\mathbf{b}_m\|_2 + 10^{-6})
$$

**Modality gate** — a **sigmoid**, not a softmax, so two modalities can be fully attended to at
once rather than competing for a probability budget:

$$
\boldsymbol\gamma = \sigma\Big(W_g\big[\hat{\mathbf{b}}_1\|\cdots\|\hat{\mathbf{b}}_5\|\boldsymbol\nu\big]\Big) \in (0,1)^5
$$

**First-order term:**

$$
\mathbf{f}_1 = \sum_{m=1}^5 \gamma_m\,\hat{\mathbf{b}}_m
$$

**Second-order term** — low-rank bilinear pooling over all $\binom{5}{2}=10$ modality pairs,
rank $r=128$ (`fusion_rank`):

$$
\mathbf{f}_2 = \sum_{m<m'} (U_m\hat{\mathbf{b}}_m)\odot(U_{m'}\hat{\mathbf{b}}_{m'}) \in \mathbb{R}^{128}
$$

$U_m\in\mathbb{R}^{128\times256}$, bias-free — a full bilinear pool would cost $10d^2$; this
costs $5dr$.

**Output:**

$$
\mathbf{f} = \mathrm{Dropout}\Big(W_o\big[\mathbf{f}_1 \,\|\, V\mathbf{f}_2\big]\Big), \qquad V:128\to256,\;\; W_o:512\to256
$$

`fusion_gate_hidden = 128` sizes the gate MLP's hidden width; `fusion_drop = 0.10` is the output
dropout.

**Measured parameters: 496,005** (branch\_norms 2,560 · modality\_gate 165,253 · bilinear
163,840 · bilinear\_out 33,024 · output 131,328) — a $\mathbf{-1.64\,M}$ reduction from the
pre-Tier-3 fusion's 2,190,916, the single largest saving in the redesign. `model.fusion_heads`,
which named the deleted Perceiver's attention head count, has **no home in the schema anymore**
(deleted, not merely unwired — §3.8).

### `EmbedNet` — pre-norm residual refinement

$$
\mathbf{e} = \mathrm{LN}_2\Big(\mathbf{f} + \mathrm{Drop}\big(\mathrm{MLP}(\mathrm{LN}_1(\mathbf{f}))\big)\Big),
\qquad \mathrm{MLP}: 256\to512\to256
$$

The only post-fusion refinement block (Tier 3 deleted the Perceiver's duplicate `output_proj`,
which was this same block applied twice to the same vector). $\mathbf{e}$ is $\ell_2$-normalised
to $\hat{\mathbf{e}}$ before the head, and consumed identically by SupCon/ProtoNCE (§4.4) and the
t-SNE figure. Its terminal `LayerNorm` pins $\|\mathbf{e}\|\approx\sqrt{256}=16$ — the reason a
zero-margin ArcFace head (Stage 1) is nearly what a plain linear head already was: the linear
head was implicitly scoring a near-constant-norm embedding all along.

263,936 measured parameters.

---

## 3.5 Classification head — one head, three margins

### Adaptive sub-centre ArcFace head (`AdaptiveSubcenterArcFaceHead`)

Built **once**, shared verbatim by all three stages. Weight
$\mathbf{W}\in\mathbb{R}^{(90\cdot3)\times256}$ (`xavier_uniform_`), i.e. $K{=}3$ sub-centres
per class; a `margins` buffer $(90,)$ initialised to $m_{\text{base}}$ and a `confusion` buffer
$(90,90)$ initialised to zero both travel in the checkpoint.

**Sub-centre cosine with soft-to-hard pooling** — a temperature $\tau$ (set per-stage via
`set_subcentre_tau`, **not itself part of the checkpoint** — it is a plain Python float, not a
buffer) interpolates between a differentiable log-sum-exp pool early in a stage and the exact
hard maximum late in it:

$$
\cos\theta_{i,c,k} = \operatorname{clamp}(\hat{\mathbf{e}}_i^\top\hat{\mathbf{W}}_{c,k},\,-1{+}\varepsilon,\,1{-}\varepsilon),
\qquad \varepsilon = 10^{-3}
$$

$$
\cos\theta_{i,c} =
\begin{cases}
\tau\log\sum_k \exp(\cos\theta_{i,c,k}/\tau), & \tau > 0 \quad\text{(soft, re-clamped)}\\
\max_k \cos\theta_{i,c,k}, & \tau \le 0 \quad\text{(hard, exact)}
\end{cases}
$$

`subcenter_tau_init = 0.20 \to subcenter_tau_final = 0.02` anneals across each stage, so every
sub-centre receives gradient early (none can die before seeing data) and the assignment hardens
back to the $\max_k$ it started as. $\varepsilon=10^{-3}$ (not the pre-Tier-1 $10^{-6}$) bounds
$|d/dc\sqrt{1-c^2}|$ at $22.4$ rather than $707$ at the clamp boundary — a numerical-conditioning
fix, not a gradient-amplification one: both embedding and weight are normalised, so the ArcFace
gradient reaching either is $s\sin(\theta+m)$, bounded by $s$, with no singularity regardless of
the clamp.

**Sub-centre load-balancing** (`subcenter_balance_weight = 0.01`) — a soft assignment
$\pi_{i,k}=\operatorname{softmax}_k(\cos\theta_{i,y_i,k}/\tau)$ (or a one-hot at $\tau\le0$),
averaged per class present in the batch, penalised against uniform:

$$
\mathcal{L}_{\text{balance}} = \sum_{c\in\text{batch}} \mathrm{KL}\big(\boldsymbol\pi_c \,\|\, \mathrm{Uniform}_K\big)
$$

A uniform assignment costs exactly $0$; a fully-collapsed one costs $|\mathcal{C}|\log K$. This
term keeps sub-centres from dying — sub-centres are seeded by spherical $k$-means over real
training embeddings (`init_subcentres_from_embeddings`, run once at Stage-2 entry on both the
live model and the EMA shadow), not by the old "one vector plus $0.01k$ Gaussian jitter"
bootstrap, which under hard-max assignment left decoy sub-centres permanently unreachable.

**Margined target logit**, with the classic "easy-margin" guard generalised to a hard cap on
$\theta+m$ at $\pi/2$ (past which $-s\sin(\theta+m)$ would start decreasing rather than
increasing, breaking ArcFace's whole premise):

$$
\theta_{i,y_i} = \arccos(\cos\theta_{i,y_i}),
\qquad
m_i = \min\!\big(M_i,\; \max(0,\,\tfrac{\pi}{2}-\theta_{i,y_i})\big)
$$

$$
\phi_i = \cos\theta_{i,y_i}\cos m_i - \sin\theta_{i,y_i}\sin m_i = \cos(\theta_{i,y_i}+m_i)
$$

**Pairwise confusion term** — pushes a class away from what it is *actually* confused with,
rather than uniformly from all 89 others, using a row-normalised, zero-diagonal confusion matrix
$\Omega$ fitted once at Stage-2 entry (`set_confusion`):

$$
\mathrm{logit}_{i,c} = s\Big(\mathbb{1}[c{=}y_i]\,\phi_i + \mathbb{1}[c{\ne}y_i]\big(\cos\theta_{i,c} - \delta_{\text{pw}}\,\Omega_{y_i,c}\big)\Big),
\qquad s=48,\;\; \delta_{\text{pw}} = 0.10
$$

**At `global_m = 0` the head takes a fast path** that skips the margin algebra entirely
(bit-identical to running the full computation at a vanishing margin) — this is the code path
Stage 1 always runs.

### The per-class margin — a signed precision/recall rule (HD-3)

Computed **once per stage**, at Stage-2 entry, from precision/recall measured on the fit split
(calibration split if carved, else `val`):

$$
\boxed{\,M(c) = \operatorname{clip}\big(m_{\text{base}} + m_\Delta\,(R_c - P_c),\; m_{\min},\; m_{\max}\big)\,},
\qquad m_{\text{base}}{=}0.35,\; m_\Delta{=}0.20,\; m_{\min}{=}0.20,\; m_{\max}{=}0.50
$$

This **replaces** the pre-Tier-2 F1-driven rule $M(c)=m_{\text{base}}+m_\Delta(1-F_1^{(c)})$, and
the sign is the whole point: an additive angular margin *shrinks* a class's decision region, so
an **over-claiming** class ($R_c > P_c$) should have its margin *raised*, and an
**under-claiming** one ($R_c < P_c$, losing recall) should have it *lowered* — the F1-driven rule
moved every low-F1 class the same direction regardless of which error it was making. Two classes
sharing the same $F_1$ can therefore land on opposite sides of $m_{\text{base}}$.

Stages 1/2/3 read this vector through three different lenses (§4.1–§4.3): Stage 1 always passes
`global_m = 0`, bypassing it; Stage 2 warms a **global scalar** up to `arcface_m = 0.35` over 20
epochs, then hands over to the per-class vector; Stage 3 keeps the vector but scales the *whole*
of it multiplicatively by a per-cycle annealed $\kappa$.

### Auxiliary heads (deep supervision)

Four identical `AuxiliaryHead`s, $\mathrm{Linear}(256{\to}128)\to\mathrm{GELU}\to
\mathrm{Linear}(128{\to}90)$, `trunc_normal_(std=0.02)`, zero bias. Called **only in training
mode**, and always on the **unmasked** branch embeddings $\mathbf{b}^{\text{raw}}$ — so branch
dropout regularises the fusion pathway without ever starving deep supervision. 44,506 parameters
each, 178,024 total.

---

## 3.6 Tensor shape matrix

Input contract: $x\in\mathbb{R}^{B\times40\times64\times64}$, `float32`.

### Shared front-end

| Stage | Module | Input | Output |
|---|---|---|---|
| Spectral gate | `se` | $(B,40,64,64)$ | $(B,40,64,64)$ |
| Grid A | `extract_grid_spectra_multi(\cdot,8)` | $(B,40,64,64)$ | $(B,64,40)$ → flat $(64B,40)$, mass $(B,64)$ |
| Grid D | `extract_grid_spectra_multi(\cdot,4)` | $(B,40,64,64)$ | $(B,16,40)$ |
| Mean spectrum | `masked_mean_spectrum` | $(B,40,64,64)$ | $(B,40)$ |

### Branch A ($64B$ flattened cells)

| Step | Module | Input | Output |
|---|---|---|---|
| A.0 | `shape_channels` (SNV + $D_1,D_2$) | $(64B,40)$ | $(64B,3,40)$ |
| A.1 | `stem` (`LambdaConv1d` 3→96 + GN + GELU) | $(64B,3,40)$ | $(64B,96,40)$ |
| A.2 | `+ wl_pe_cnn` | $(64B,96,40)$ | $(64B,96,40)$ |
| A.3 | `tower_{s,m,l}` ($k{=}3/5/7$) | $(64B,96,40)$ | $3\times(64B,96,40)$ |
| A.4–5 | concat + `fusion` | $(64B,288,40)$ | $(64B,96,40)$ |
| A.6–7 | `attn_pool` + weighted sum | $(64B,96,40)$ | $(64B,96)$ |
| A.8 | `proj` | $(64B,96)$ | $(64B,256)$ |
| A.9 | mass-weighted pool over 64 cells | $(B,64,256),(B,64)$ | $\mathbf{b}_A\,(B,256)$ |

### Branch B

| Step | Module | Input | Output |
|---|---|---|---|
| B.0 | `SoftIndexBank` | $(B,40)$ | $(B,64)$ |
| B.1 | `ContinuumDepths` | $(B,40)$ | $(B,16)$ |
| B.2 | concat with morph | $(B,64),(B,16),(B,8)$ | $(B,88)$ |
| B.3 | MLP | $(B,88)$ | $\mathbf{b}_B\,(B,256)$ |

### Branch C

| Step | Module | Input | Output |
|---|---|---|---|
| C.1 | `stem` (3-D, 3 stages) | $(B,1,40,64,64)$ | $(B,64,5,16,16)\to(B,192,16,16)$ |
| C.2–5 | `stages` (4× `ResBlock2D`, 3× `CBAM`) | $(B,192,16,16)$ | $(B,256,4,4)$ |
| C.6 | pool (pn-mean, pn-max, $\ell_2$) | $(B,256,4,4)$ | $(B,512)$ |
| C.7 | `proj` | $(B,512)$ | $\mathbf{b}_C\,(B,256)$ |

### Branch D ($16B$ flattened cells)

| Step | Module | Input | Output |
|---|---|---|---|
| D.0 | `LambdaWindowPooling` | $(16B,40)$ | $(16B,10)$ |
| D.1 | `tokenizer` ($k{=}3/5/7$ strided) | $(16B,1,10)$ | $(16B,192,10)$ |
| D.2 | `+ spec_pos_embed`, prepend `spec_cls` | $(16B,10,192)$ | $(16B,11,192)$ |
| D.3 | `spectral_blocks` ($\times2$, $+\lambda$-bias) | $(16B,11,192)$ | $(16B,11,192)$ |
| D.4 | token 0, reshape | $(16B,11,192)$ | $(B,16,192)$ |
| D.5 | prepend `spatial_cls` | $(B,16,192)$ | $(B,17,192)$ |
| D.6 | `spatial_blocks` ($\times2$) | $(B,17,192)$ | $(B,17,192)$ |
| D.7 | token 0 → `norm` → `proj` | $(B,192)$ | $\mathbf{b}_D\,(B,256)$ |

### Fusion, embedding and head

| Step | Module | Input | Output |
|---|---|---|---|
| E.0 | `morphology_embed` | $(B,8)$ | $\mathbf{b}_E\,(B,256)$ |
| F.0 | branch masking (train only) | $4\times(B,256)$ | $4\times(B,256)$ |
| F.1 | `branch_norms` (5× BatchNorm1d) | $5\times(B,256)$ | $5\times(B,256)$ |
| F.2 | `modality_gate` (sigmoid) | $(B,1285)$ | $(B,5)$ |
| F.3 | first order + bilinear second order | $5\times(B,256)$ | $(B,256),(B,128)$ |
| F.4 | `output` | $(B,512)$ | $(B,256)$ |
| E.1 | `embed_net` | $(B,256)$ | $\mathbf{e}\,(B,256)$ |
| H.1 | $\ell_2$-normalise | $(B,256)$ | $\hat{\mathbf{e}}\,(B,256)$ |
| H.2 | `arcface_head`: $\hat{\mathbf{e}}\hat{\mathbf{W}}^\top$ | $(B,256)\times(270,256)$ | $(B,270)$ |
| H.3 | reshape + pool over $K{=}3$ | $(B,270)$ | $(B,90,3)\to(B,90)$ |
| X.1–4 | `aux_head_{a,b,c,d}` (train only) | $4\times(B,256)$ | $4\times(B,90)$ |

---

## 3.7 Parameter budget

Measured on the shipped configuration (`configs/model/spectral_quadnet_v4.yaml`):

| Component | Parameters | Share | Pre-Tier-3 |
|---|---:|---:|---:|
| `se` (MaskedSpectralECA) | 6 | 0.00 % | 6 |
| `wl_pe_cnn` (buffer only, shared) | 0 | 0.00 % | 0 |
| `branch_a` — Spectral Profile | 603,089 | 11.6 % | 592,753 |
| `branch_b` — Index Bank | 94,896 | 1.8 % | 686,424 |
| `branch_c` — Spatial CNN | 2,230,646 | 42.9 % | 1,694,158 |
| `branch_d` — SpecFormer | 1,241,640 | 23.9 % | 2,180,866 |
| `morphology_embed` | 17,216 | 0.3 % | — (new) |
| `cross_interaction` — fusion | 496,005 | 9.6 % | 2,190,916 |
| `aux_head_{a,b,c,d}` — $4\times44{,}506$ | 178,024 | 3.4 % | 178,024 |
| `embed_net` | 263,936 | 5.1 % | 263,936 |
| `linear_head` | — (removed) | — | 23,130 |
| `arcface_head` ($270\times256$) | 69,120 | 1.3 % | 69,120 |
| **Total (all trainable)** | **5,194,578** | 100 % | **7,879,333** |

A $-2.68\text{M}$ ($-34\%$) reduction overall, dominated by fusion ($-1.64\text{M}$, the Perceiver
replacement) and Branch B ($-0.59\text{M}$, dropping the rank-2 moment tensor), partially spent
back on Branch C ($+0.54\text{M}$, the new joint spectral-spatial 3-D stem) and a new
`morphology_embed` (+17k). Checkpoint-persisted buffer elements: 25,013 (state\_dict element
total: 5,219,591). A further $\sim$253k buffer elements exist at runtime but are **not**
persisted — dominated by `ContinuumDepths`' four $(40,40,40)$ chord-interpolation tensors,
recomputed from the wavelength grid on every construction rather than checkpointed.

---

## 3.8 Architectural invariants

1. **Attribute names are checkpoint schema.** The 14 top-level names — `se`, `wl_pe_cnn`,
   `branch_{a,b,c,d}`, `morphology_embed`, `cross_interaction`, `aux_head_{a,b,c,d}`,
   `embed_net`, `arcface_head` — are the keys of every schema-v3 checkpoint (`SCHEMA_VERSION =
   3`). There is no `linear_head`.
2. **v1/v2 → v3 checkpoints cannot be migrated — this is a hard refusal, not a partial load.**
   `engine/checkpoint.py::remap_state_dict` raises `SchemaTooOldError` for any bundle at schema
   $\le2$: Branches B, C and D changed what they *consume*, not merely how many parameters they
   consume it with, so no tensor-level remap exists. The only paths are retraining or checking
   out the Tier-2 tree to re-score an archived checkpoint.
3. **Construction order is initialisation.** Every `_init_weights` draws from the same global
   torch RNG stream, in `__init__` order: `se → wl_pe_cnn → branch_a → branch_b → branch_c →
   branch_d → morphology_embed → cross_interaction → aux_head_{a,b,c,d} → embed_net →
   arcface_head`. `train.py` documents the required call order at its `set_seed` site:
   *config → `set_seed` → `DataStore` → `SpectralQuadNet` → `ModelEMA`*.
4. **Weight initialisation policy.** The outer `_init_weights` walks every submodule
   (`self.modules()`) and applies: Conv1d/2d/3d → `kaiming_normal_(mode='fan_out',
   nonlinearity='relu')`; BatchNorm/GroupNorm → $\gamma{=}1,\beta{=}0$; Linear →
   `trunc_normal_(std=0.02)`, zero bias. It does **not** reach raw `nn.Parameter`s outside those
   layer types — `SoftIndexBank.theta_pos/theta_neg` ($\mathcal{N}(0,0.02^2)$),
   `arcface_head.weight` (`xavier_uniform_`), and both SpecFormer CLS tokens
   (`trunc_normal_(std=0.02)`) all keep their own construction-time initialisation for the
   model's whole life. LayerNorm and `LambdaConv1d`'s internal `kernel_mlp` LayerNorm are also
   left at PyTorch defaults, deliberately — it keeps the generated-kernel scale a known constant.
5. **Buffers travel — except where explicitly non-persistent.** `wl_pe_cnn.pe` /
   `branch_a.wl_pe_module.pe` (object-identity-shared, $(40,96)$), `branch_a.derivatives.{d1_op,
   d2_op}` $(40,40)$ each, `branch_a.stem.0.{nbr,offsets,features.omega}`,
   `branch_d.spec_pos_embed`, `branch_d.lambda_bias.features.omega`, and
   `arcface_head.{margins,confusion}` are `register_buffer`s and part of `state_dict()`.
   `branch_b`'s `ContinuumDepths` hull-interpolation weights and `branch_d`'s
   `LambdaWindowPooling.{token_wl,pool}` are registered `persistent=False` — deterministic
   functions of the wavelength grid, recomputed on load rather than stored.
6. **`ModelEMA.state_dict()` is the shadow's**, so the shadow must carry the identical key
   structure — asserted for all three stages. Buffers are copied outright at each EMA update,
   never averaged. `ModelEMA.update` requires the **unwrapped** module — passing a DDP- or
   `torch.compile`-wrapped model raises `RuntimeError` rather than silently failing to track,
   since its parameter-matching is by name.
7. **Aux heads and branch dropout are both training-only.** No auxiliary head is ever invoked in
   `.eval()`, and `branch_drop_prob` never masks a branch there either — the return-type table in
   §3.1 reflects this directly (aux keys only appear in the training-mode dict).
8. **Two modules carry a runtime-selected execution path, and it is not schema.**
   `SpectralSpatialStem3D.decompose_conv3d` (Branch C's `Conv3d` stages evaluated as stacks of
   `Conv2d`) and `SpectralProfileBranch.grad_checkpoint` (Branch A's towers recomputed in the
   backward) are set by `utils/device.py::apply_runtime_optimisations` from the resolved runtime
   plan, before the EMA deep-copy and every wrapper, so all copies of the module agree. Neither
   adds a parameter, a buffer or a state-dict key, neither draws from the RNG, and the
   decomposition is the same arithmetic in a different summation order ($1.9\times10^{-7}$ on the
   logits) — so a checkpoint written under either path loads under the other, and a resumed run
   may legitimately change device and change paths with it.
   `06_EXECUTION_AND_HARDWARE.md` §6.2 has the measurements and the defaults.

### Config keys — full coverage, one call-site-level exception

Unlike the pre-Tier-3 model, `tests/unit/test_config_wiring.py` currently asserts **every**
`cfg.model.*` key is either forward-observable (perturbing it changes eval-mode logits),
train-mode-only observable (a dropout rate), or has an explicit, named reason it cannot be
(`branch_drop_prob`: eval never masks; `subcenter_tau_final`: the endpoint of an epoch-driven
schedule, not a constructor argument; `subcenter_balance_weight`: read by the loss, not the
model). There is no known dead `cfg.model.*` key today. One call-site-level parameter — not a
config key — is unused: `SpectralQuadNet` computes `stride = cfg.model.specf_patch // 2` and
passes it to `SpecFormerBranch.__init__`, but the branch derives its token count directly from
`patch_size` and never reads the `stride` argument it receives.

`model.fusion_heads`, which named the deleted Perceiver fusion's attention head count, is
**deleted from the schema entirely** (not merely unwired) — there is nothing left in the gated
bilinear pool for a head count to parameterise.
