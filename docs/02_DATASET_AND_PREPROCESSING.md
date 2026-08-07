# 2 · Dataset, Preprocessing and the Data Path

Two disjoint code paths live under `spectralquadnet.data`:

| Path | Modules | Runs | Dependencies |
|---|---|---|---|
| **Offline preparation** | `data/prep/{download,segmentation,patch_extraction,band_selection,config}.py` | once, by hand, via `scripts/prepare_dataset.py` and `scripts/select_bands.py` | `[prep]` extra (`opencv`, `scikit-image`, `scipy`, `spectral`, `tqdm`) |
| **Training data path** | `data/{mmap_store,datasets,samplers,loaders}.py` | every epoch | core install only |

Training never imports `data.prep.*`. The boundary is enforced by the dependency split in
`pyproject.toml`.

---

## 2.1 Acquisition

`data/prep/config.py::PrepConfig` fixes the acquisition contract:

| Field | Value | Meaning |
|---|---|---|
| `data_url` | Zenodo record `3241923` | *RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties* |
| `root` | `./dataset` | Working directory for the archive and outputs |
| `patch_size` | `64` | Output patch side, pixels |
| `num_bands` | `256` | Bands in the raw cube |

`download()` fetches the archive with a resumable `curl -L --fail -C -` and short-circuits if
`dataset/rice_hsi.zip` already exists. Cubes are read directly out of the zip; only the four
members needed for one scan (`.hdr` + data + `*black.hdr` + black data) are ever extracted to
a temporary directory, which is torn down after each cube.

**Cube indexing.** Every `.hdr` member that is *not* a `black.hdr` is a scan. The acquisition
session is the path component beginning `Data-VIS`; the variety name is the file stem with
its trailing `-<n>` suffix removed. Class labels are `pandas.factorize` over the variety
column, yielding the $90$ integer classes.

**Wavelength axis.** `wavelengths.csv` inside the archive provides the 256-band axis:
$\lambda \in [383.22,\; 1006.47]$ nm at a mean spacing of $\approx 2.444$ nm.
(The README and abstract quote this nominally as *385–1000 nm*.)

---

## 2.2 Radiometric correction and segmentation

### Dark-current correction

For a raw cube $R \in \mathbb{R}^{H_0 \times W \times 256}$ and its dark reference
$D$ (`*black.hdr`, same geometry), `preprocess_raw` removes the row-averaged dark frame and
crops to the sensor's valid-data band:

$$
\bar{D}_{w,c} \;=\; \frac{1}{H_D}\sum_{h} D_{h,w,c},
\qquad
X_{h,w,c} \;=\; \max\!\big(R_{h,w,c} - \bar{D}_{w,c},\; 0\big),
\qquad
X \leftarrow X_{[0:600]}
$$

Both cubes are loaded as `float32` via `spectral.io.envi` (imported lazily, with
`envi_support_nonlowercase_params = True` set at call time, not import time).

### Otsu seed segmentation

`segment(cube, wl)` reduces the cube to a visible-band intensity image and thresholds it:

$$
I_{h,w} \;=\; \frac{1}{|\mathcal{V}|}\sum_{c \in \mathcal{V}} X_{h,w,c},
\qquad
\mathcal{V} = \{\,c : 450 < \lambda_c < 700\,\}
$$

$$
B_{h,w} \;=\; \mathbb{1}\!\left[\, I_{h,w} > 0.4\,\tau_{\text{Otsu}}(I) \,\right]
$$

The $0.4$ relaxation of Otsu's threshold deliberately over-segments, and the false positives
are removed by morphology and shape filtering rather than by the threshold itself:

1. `binary_fill_holes` — close specular holes inside a seed.
2. `clear_border` — drop objects touching the frame (partially imaged seeds).
3. `remove_small_objects(min_size=150)` — drop debris.
4. Connected-component labelling + region properties.
5. Shape gate, applied to every region:
   $$300 < \text{area} < 800 \;\;\wedge\;\; \text{eccentricity} > 0.6 \;\;\wedge\;\; \text{solidity} > 0.85$$
6. Sort by centroid $(\text{row}, \text{col})$ — top-left first — so patch order is
   deterministic for a given cube.

The gate encodes what a rice seed *is*: a compact (`solidity`), elongated (`eccentricity`),
size-bounded (`area`) blob.

---

## 2.3 Patch extraction

`build_patch_dataset` makes three passes for one reason: the output is multi-gigabyte, so the
exact allocation must be known before any patch is written.

| Pass | Work |
|---|---|
| 1 · count | Walk every cube, segment, accumulate $N = \sum_{\text{cubes}} \lvert\mathcal{R}\rvert$ |
| 2 · allocate | Allocate $X \in \mathbb{R}^{N \times 256 \times 64 \times 64}$ (`float32`) and $y \in \mathbb{Z}^{N}$ once |
| 3 · write | Re-walk, re-segment, and write each patch in place |

Per region $r$ with bounding box $(r_0, c_0, r_1, c_1)$ and label image $L$, write the crop
$P$ and its mask $M = \mathbb{1}[L_{[r_0:r_1,\,c_0:c_1]} = r]$. **Both** are centre-padded to
a square $(S, S, \cdot)$ with $S = \max(h, w)$ and area-resized to $64 \times 64$; the
resized mask $\alpha$ is then the *fill fraction* of each output pixel, and

$$
\tilde P_{p} \;=\;
\begin{cases}
P^{\text{rs}}_{p} / \alpha_p, & \alpha_p > \tfrac12\\[2pt]
0, & \text{otherwise}
\end{cases}
$$

so **every pixel outside the seed's own connected component is exactly zero across all
bands** — the invariant that every masked statistic downstream relies on — and no surviving
pixel carries the partial-coverage attenuation. The patch is finally transposed
$(H, W, C) \to (C, H, W)$.

> **This order is P-3 / T4-3, and it is a correction.** Until Tier 4 the mask was applied
> *before* the resize and the mask itself was never resized, so `INTER_AREA` averaged seed
> pixels with background zeros: a boundary pixel covering a fraction $\alpha_p$ of seed came
> out at $\alpha_p$ times its true radiance, and a pixel covering a thousandth of a seed came
> out as a *non-zero background*. The invariant asserted above held only in the interior
> (IMPROVEMENT_PLAN M-11), which is what made the downstream $>10^{-5}$ foreground test
> fragile. `tests/unit/test_patch_extraction.py` measures both the old failure and the fix.

Two more Tier-4 items happen here. **P-2 / T4-2**: the cube is radiance, not reflectance, so a
per-session illumination gain multiplies every spectrum captured in that session (C-1). With no
white panel in the archive — verified: its only reference cubes are `black.hdr` — the
resolution is per-pixel SNV along $\lambda$, applied *after* masking, which removes that gain
and the per-pixel geometric gain identically. The gain it removes is persisted rather than
discarded, so brightness stays available as an explicit input. **P-4 / T4-4**: the eight
morphometrics `segment` already computed, gated on and threw away are written out, in physical
pixel units — the absolute scale the resize destroys.

Failures on any single cube are caught, printed and skipped, so one unreadable scan does not
abort a multi-hour extraction; a cube counted in pass 1 that fails in pass 2 leaves no
all-zero rows, since the arrays are truncated to what was written.

**Resulting artifacts**, all row-aligned on the patch index except the last:

| File | Shape | Item |
|---|---|---|
| `dataset/patches.npy` | $(8624, 256, 64, 64)$ `float32` | the patches |
| `dataset/labels.npy` | $(8624,)$ `int64` | class index, 90 classes at 91–96 patches each |
| `dataset/groups.npy` | $(8624,)$ `int64` | P-1 — the `scan_id` a grouped split needs |
| `dataset/masks.npy` | $(8624, 64, 64)$ `float16` | P-3 — the fill map $\alpha$ |
| `dataset/gain.npy` | $(8624, 2, 64, 64)$ `float32` | P-2 — per-pixel $(\bar x, \mathrm{sd})$ |
| `dataset/morphology.npy` | $(8624, 8)$ `float32` | P-4 — shape descriptors, unstandardised |
| `dataset/scan_table.csv` | one row per cube | P-1 — what each `scan_id` is |

`morphology.npy` is persisted **unstandardised** on purpose: the mean and scale are fitted on
the training split alone (`data/morphometrics.py`), and no split exists at extraction time.

---

## 2.4 Band selection: $256 \to 40$

`data/prep/band_selection.py` runs a six-step pipeline. The rationale, recorded in the module
docstring, is that CARS is a PLS-regression method needing a continuous target and is
therefore inapplicable to 90-class discrimination; mRMR works from class labels through
mutual information, and SPA complements it geometrically.

### Step 1 — Foreground mean spectra

Patches are read through a memory map in chunks of 2,048 and reduced to one masked mean
spectrum each:

$$
m_p \;=\; \mathbb{1}\Big[\textstyle\sum_c |X_{n,c,p}| > 10^{-5}\Big],
\qquad
\mathbf{x}_{n,c} \;=\; \frac{\sum_{p} X_{n,c,p}\, m_p}{\max\big(\sum_p m_p,\; 10^{-5}\big)}
$$

giving $\mathbf{X} \in \mathbb{R}^{8624 \times 256}$.

### Step 2 — Decorrelation pre-filter

A greedy left-to-right sweep (low $\to$ high $\lambda$) keeps band $i$ only if it is not a
near-duplicate of anything already kept:

$$
\mathcal{K} \leftarrow \mathcal{K} \cup \{i\}
\quad\text{iff}\quad
\max_{j \in \mathcal{K}} \big|r_{ij}\big| \le \tau_{\text{corr}} = 0.995
$$

Both selectors then operate on the surviving candidate set $\mathcal{K}$.

### Step 3 — Fisher Discriminant Ratio (diagnostic only)

$$
\mathrm{FDR}_k \;=\; \frac{\sum_{c} n_c \,(\mu_{ck} - \mu_k)^2}{\sum_{c} n_c\, \sigma^2_{ck} + 10^{-10}}
$$

Printed for the top-10 bands as a sanity reference; it does **not** drive selection.

### Step 4 — mRMR (MID criterion)

Greedy forward selection maximising relevance minus mean redundancy against the already-selected set $S$:

$$
J(x_k) \;=\; \underbrace{\mathrm{MI}(x_k;\, y)}_{\text{relevance}} \;-\;
\underbrace{\frac{1}{|S|}\sum_{x_j \in S} \mathrm{MI}(x_k;\, x_j)}_{\text{redundancy}}
$$

Relevance uses `mutual_info_classif` (k-NN estimator, $k = 5$, `random_state=42`), which makes
no Gaussian assumption. Redundancy uses the closed-form Gaussian identity as a proxy, exact
for bivariate normals and close for spectrally smooth bands, avoiding an $O(P^2)$ k-NN MI matrix:

$$
\mathrm{MI}(x_i;\, x_j) \;\approx\; -\tfrac{1}{2}\log\!\big(1 - r_{ij}^2 + \varepsilon\big),
\qquad \varepsilon = 10^{-10}
$$

The first band is the one with maximum relevance; selection runs to $n_{\max} = 100$.

### Step 5 — SPA (Successive Projections Algorithm)

Columns are $\ell_2$-normalised, the seed is the **highest-MI band** (so the geometric search
is anchored on a known-discriminative direction), and each step Gram-Schmidt-projects the
last selected column out of every remaining column, then takes the largest residual:

$$
\mathbf{x}_j \;\leftarrow\; \mathbf{x}_j - \frac{\mathbf{x}_j^{\top}\mathbf{p}}{\|\mathbf{p}\|^2}\,\mathbf{p},
\qquad
k^{\star} \;=\; \arg\max_{j \notin S} \|\mathbf{x}_j\|_2
$$

with pivot $\mathbf{p}$ the last selected column in its current, already-orthogonalised state.
This minimises multicollinearity in the resulting subset.

### Step 6 — Cross-validated selection and the elbow

Both orderings are evaluated at $k \in \{5,10,15,20,25,30,40,50,70,100\}$ (restricted to
$k \le |\text{ordering}|$) with 5-fold `StratifiedKFold`, using two classifiers on standardised
mean spectra: LDA (`solver='svd'`, pseudo-inverse — safe at 90 classes) and
`LinearSVC(C=0.1, max_iter=3000)`.

Recorded curve — `dataset/band_selection_report.csv`:

| $k$ | mRMR LDA | mRMR SVC | SPA LDA | SPA SVC |
|---:|---:|---:|---:|---:|
| 5 | 0.2490 | 0.1827 | 0.2662 | 0.1982 |
| 10 | 0.3155 | 0.2444 | 0.3858 | 0.3225 |
| 15 | 0.3798 | 0.3122 | 0.4383 | 0.3822 |
| 20 | 0.4149 | 0.3402 | 0.4762 | 0.4112 |
| 25 | 0.4510 | 0.3780 | 0.5003 | 0.4293 |
| 30 | 0.5017 | 0.4123 | 0.5248 | 0.4416 |
| **40** | 0.5908 | 0.4678 | **0.5916** | **0.4755** |

The winner is the method with the higher **peak SVC** accuracy — SPA ($0.4755 > 0.4678$) —
and the band count is the elbow

$$
k^{\star} \;=\; \min\Big\{\,k \;:\; \mathrm{acc}(k) \;\ge\; 0.98 \cdot \max_{k'} \mathrm{acc}(k')\,\Big\}
$$

With $\max = 0.4755$ the threshold is $0.4660$, met only at $k = 40$; hence
$k^{\star} = 40$, a $84.4\%$ reduction of the spectral axis. The final band set is sorted
ascending before writing (selection *order* is discarded; only the *set* matters downstream).

**Outputs** — `dataset/patches_spa_40b.npy` $(8624, 40, 64, 64)$ `float32` (written in
2,048-row chunks off the memory map) and `dataset/wavelengths_spa_40b.csv`, whose 40 centres
span $383.22$–$1006.47$ nm.

---

## 2.5 The zero-RAM data store

`data/mmap_store.py::DataStore` is the single owner of the patch cube. Its invariants are
non-negotiable and are asserted by `tests/unit/test_mmap_store.py`.

| Invariant | Mechanism |
|---|---|
| **Never paged into RAM** | `np.load(path, mmap_mode="r")` — read-only mapping; `mmap_mode=None` is forbidden |
| **Load once** | `load_patches` returns early when `patches is not None` |
| **One handle per process** | `DataStore.__new__` returns a process-wide singleton; constructing it twice yields the *same object* |
| **Copy lives in the Dataset** | `DataStore` exposes the raw `np.memmap`; only `RiceSeedDataset.__getitem__` touches it |
| **Exactly one patch per item** | `np.array(self.patches[ri])` materialises $(40,64,64) = 655{,}360$ B $\approx 0.64$ MiB; no batched `.copy()` anywhere on the loader path |

Labels ($8624 \times 8$ B) are read fully into RAM; wavelengths are read with a sniffed
delimiter (`sep=None`), taken from the **last** CSV column, and min–max normalised to $[0,1]$
before being moved to the training device:

$$
\tilde\lambda_i = \frac{\lambda_i - \min_j \lambda_j}{\max_j \lambda_j - \min_j \lambda_j}
$$

Typed accessors `require_patches()` / `require_labels()` / `require_wavelengths()` raise
rather than return `None`, so a missing load fails at the call site.

**Footprint.** The cube is $8624 \times 40 \times 64 \times 64 \times 4\,\text{B} = 5.65$ GB on
disk. The README records a measured **peak RSS of 1.39 GB, median 0.65 GB** across a full
three-stage run against that file, with mean usage *falling* over the run — clean mapped
pages being reclaimed, not a dataset accumulating. Every `DataLoader` in the system uses
`num_workers=0`, so no worker process duplicates the mapping.

---

## 2.6 Dataset and augmentation profiles

`RiceSeedDataset.__getitem__(idx)` returns
$\big(\text{patch} \in \mathbb{R}^{40\times64\times64},\; \text{label} \in \mathbb{Z}\big)$,
both already on the training device.

Augmentation is selected by a named profile, each a vector of independent Bernoulli trigger
probabilities — a sample can receive several augmentations at once:

| Profile | `band_drop` | `cutout` | `noise` | `warp` | `mult` | intensity $s$ | warp range |
|---|---:|---:|---:|---:|---:|---:|---:|
| `heavy` | 0.08 | 0.06 | 0.04 | 0.03 | 0.05 | 1.00 | 0.05 |
| `medium` | 0.05 | 0.04 | 0.03 | 0.02 | 0.03 | 0.70 | 0.03 |
| `very_light` | 0.05 | 0.04 | 0.05 | 0.01 | 0.04 | 0.25 | 0.00 |
| `light` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.00 |
| `none` | — | — | — | — | — | — | — |

With `none` the profile is `None` and **no augmentation runs at all — including the spatial
transform**; every other profile always applies the spatial transform.

Primitives, with foreground mask $m_{h,w} = \mathbb{1}[\sum_c |x_{c,h,w}| > 10^{-5}]$:

| Name | Operation |
|---|---|
| band dropout | $x_c \leftarrow x_c \cdot \mathbb{1}[u_c > p_{\text{band\_drop}}]$, $u_c \sim \mathcal{U}(0,1)$ |
| band cutout | zero a contiguous run of $n \sim \mathcal{U}\{1,\dots,3\}$ bands at a random start ($3 = $ `data.max_cutout_bands`) |
| spectral noise | $x \leftarrow x + \sigma\, m \odot \epsilon$, $\epsilon\sim\mathcal{N}(0,I)$, $\sigma = 0.02\,s$ ($0.02 = $ `data.noise_std`) |
| spectral warp | resample the band axis by $\alpha \sim \mathcal{U}(1-w,\,1+w)$ (linear interpolation), then centre-crop or centre-pad back to 40 bands |
| multiplicative | $x \leftarrow x \odot (1 + 0.05\,s\,\epsilon_c)\, m$, one scalar per band |
| spatial | random horizontal flip, random vertical flip, random $k\cdot 90^\circ$ rotation — applied **unconditionally** whenever a profile is active |

> **Ordering invariant.** The five `if torch.rand(1) < p[...]` guards consume the global torch
> RNG stream in a fixed order. Reordering them — even when an augmentation is a no-op at
> $p=0$ — changes every subsequent draw and breaks fixed-seed reproducibility. The module
> docstring states this explicitly.

---

## 2.7 Batch sampling

### `ClassBalancedBatchSampler` — Stages 2 and 3

Constructs each batch as $n_{\text{cls}} \times n_{\text{spc}} = 16 \times 8 = 128$ samples.
Given CDWS weights $w_c$ (§4.2), classes are drawn **without replacement** from a normalised
categorical:

$$
p_c \;=\; \frac{w_c}{\sum_{c'} w_{c'}}
\qquad
\mathcal{C}_t \sim \text{Categorical-without-replacement}(p,\; n_{\text{cls}})
$$

then $n_{\text{spc}}$ indices are drawn per chosen class, with replacement only if that
class's pool is smaller than $n_{\text{spc}}$. Epoch length is

$$
T \;=\; \left\lfloor \frac{N_{\text{train}}}{n_{\text{cls}}\, n_{\text{spc}}} \right\rfloor
\;=\; \left\lfloor \frac{6036}{128} \right\rfloor \;=\; 47 \text{ batches}
$$

Passing `class_weights=None` falls back to uniform class selection. **The RNG is unseeded**
(`np.random.default_rng()` per `__iter__`) — a documented source of run-to-run
non-determinism.

### `HardClassOversampledSampler` — Stage 1, Phase 3

An index sampler (not a batch sampler) over the *same* dataset, weighted by inverse class
difficulty measured at the Phase 2 → 3 boundary:

$$
\tilde{w}_c \;=\; \min\!\left(\Big(\frac{1}{F_1^{(c)} + \epsilon}\Big)^{\gamma},\; W_{\max}\right),
\qquad
w_c \;=\; \frac{\tilde{w}_c}{\frac{1}{C}\sum_{c'} \tilde{w}_{c'}}
$$

with $\gamma = 0.65$ (`p3_oversample_power`), $W_{\max} = 7.0$ (`p3_oversample_max_w`),
$\epsilon = 0.05$ (`p3_oversample_eps`). Mean-normalisation keeps the effective epoch
sampling rate unchanged. Indices are drawn by `torch.multinomial(..., replacement=True)` for
`num_samples = len(train_labels)`. A class absent from `class_f1` is treated as $F_1 = 0$
(maximally hard).

`p3_hard_f1_thresh = 0.80` **does not enter the weights** — it only selects which classes are
listed in the sampler's diagnostic print.

### Loader construction

| Loader | Batch | Shuffle | Sampler | Augmentation |
|---|---|---|---|---|
| Stage 1 · P1/P2/P3 datasets | 128 | ✓ (P1/P2) | oversampled (P3) | `heavy` / `medium` / `very_light` |
| Stage 2 train | 16×8 = 128 | — | `ClassBalancedBatchSampler` + Stage-1 CDWS | `very_light` |
| Stage 3 train | 16×8 = 128 | — | `ClassBalancedBatchSampler` + Stage-2 CDWS | `light` |
| Validation | 256 | ✗ | — | `none` |
| Test | 256 | ✗ | — | `none` |

Stage 2 uses `very_light` rather than `none` so ArcFace does not overfit the training split's
exact spectral signatures; Stage 3 uses `light`, which triggers no perturbation but *does*
apply the random dihedral spatial transform.
