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

Applying the mask after resizing (rather than before, or resizing the crop without ever
resizing the mask) is what makes the zero-outside-the-seed invariant exact: resizing an
unmasked crop lets an interpolation kernel blend seed pixels with background zeros, so a
boundary pixel covering a fraction $\alpha_p$ of seed comes out at $\alpha_p$ times its true
radiance and a pixel covering a thousandth of a seed comes out as a non-zero background.
`tests/unit/test_patch_extraction.py` asserts the exact-zero invariant directly.

Two further quantities are written out here. The cube is radiance, not reflectance, so a
per-session illumination gain multiplies every spectrum captured in that session; the full
resolution is `data/prep/radiometry.py`, a dedicated module (below). The eight morphometrics
`segment` already computes as part of its shape gate are additionally written out, in physical
pixel units — the absolute scale the resize destroys.

### Radiometry (`data/prep/radiometry.py`)

`PrepConfig.radiometry: str = "auto"` selects one of four modes, resolved once per archive by
`resolve_radiometry`:

| Mode | Behaviour |
|---|---|
| `auto` | `white` if `find_white_reference()` locates a panel cube in the archive, else `snv` |
| `white` | physically-correct reflectance division against a located white panel |
| `snv` | per-pixel Standard Normal Variate along $\lambda$ (the operative path today) |
| `none` | no radiometric correction — raw dark-corrected radiance |

`find_white_reference()` scans archive members for white-panel filename hints; on the *RGB and
VIS-NIR HSI Data for 90 Rice Seed Varieties* archive this returns nothing — its only reference
cubes are `black.hdr` — so `auto` resolves to `snv`, applied by `apply_radiometry()` **after**
masking:

$$
\tilde x_{c,p} = \frac{x_{c,p} - \bar x_{\cdot,p}}{\operatorname{sd}_c(x_{\cdot,p}) + \varepsilon},
\qquad \varepsilon = \texttt{RADIOMETRY\_EPS} = 10^{-6}
$$

(mean/std taken across bands, per pixel) — this removes the per-pixel geometric gain and the
per-session illumination gain identically, since both multiply every band of a pixel's spectrum
by the same scalar. The gain it removes is persisted rather than discarded (`gain.npy`, order
$(\bar x, \operatorname{sd})$), so brightness stays available as an explicit input rather than
being thrown away.

`white_reference_correct()`/`white_gain()` implement the physically-correct reflectance-division
path — cube-level reflectance division against a located panel, applied **before**
segmentation — fully implemented and unit-tested even though no cube in this archive exercises
it; `radiometry="white"` would select it automatically the moment a compatible archive supplied
one.

Failures on any single cube are caught, printed and skipped, so one unreadable scan does not
abort a multi-hour extraction; a cube counted in pass 1 that fails in pass 2 leaves no
all-zero rows, since the arrays are truncated to what was written.

**Resulting artifacts**, all row-aligned on the patch index except the last:

| File | Shape | Item |
|---|---|---|
| `dataset/patches.npy` | $(8624, 256, 64, 64)$ `float32` | the patches |
| `dataset/labels.npy` | $(8624,)$ `int64` | class index, 90 classes at 91–96 patches each |
| `dataset/groups.npy` | $(8624,)$ `int64` | the `scan_id` a grouped split needs |
| `dataset/masks.npy` | $(8624, 64, 64)$ `float16` | the fill map $\alpha$ |
| `dataset/gain.npy` | $(8624, 2, 64, 64)$ `float32` | per-pixel $(\bar x, \mathrm{sd})$ |
| `dataset/morphology.npy` | $(8624, 8)$ `float32` | shape descriptors, unstandardised |
| `dataset/scan_table.csv` | one row per cube | what each `scan_id` is |

`morphology.npy` is persisted **unstandardised** on purpose: the mean and scale are fitted on
the training split alone (`data/morphometrics.py`), and no split exists at extraction time.

---

## 2.4 Band selection: $256 \to k$ — **not on the primary path**

> **The primary pipeline does not run this step.** `scripts/prepare_dataset.py` writes
> `dataset/patches.npy` at the full 256 bands and `python train.py` trains on it directly;
> nothing goes between them. This section documents the retained band-selection **ablation
> pathway** — its build step, its artifacts, and the reason it is an ablation rather than the
> default. See `07_BAND_SELECTION_PATHWAY.md` for the experiment that surrounds it, and §2.4.1
> below for why $k^\star = 40$ was demoted.

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

Both orderings are evaluated with 5-fold `StratifiedKFold`, using two classifiers on
standardised mean spectra: LDA (`solver='svd'`, pseudo-inverse — safe at 90 classes) and
`LinearSVC(C=0.1, max_iter=3000)`. `BandSelectionConfig.n_candidates` defaults to
$k\in\{5,10,15,20,25,30,40,50,70,100,128,160,192,224,256\}$, restricted to
$k\le|\text{ordering}|$.

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

### 2.4.1 Why this is an ablation and not the default

Two independent defects, either of which is disqualifying for a headline number.

**(i) The elbow is vacuous.** `verify_elbow()` and its
> `ElbowVerdict` explicitly check whether a chosen $k^\star$ is *demonstrable* — the curve must
> extend past $k^\star$ and clear `elbow_pct · peak` there — rather than merely being the
> curve's own last point. The checked-in `dataset/band_selection_report.csv` (the table above)
> terminates at $k=40$, its own chosen value: the 98% criterion is satisfied *vacuously*, not
> demonstrated. `tests/unit/test_band_curve.py::test_the_shipped_curve_cannot_demonstrate_its_elbow`
pins exactly this as a known property of the current artifact. The per-fold 100-band mRMR
selection has the same defect for the same reason: its curve terminates at $k = 100$.

**(ii) The selection saw test labels.** Without `--fold`, mRMR relevance is
`mutual_info_classif(X, y)` over **every** patch, including the ones that become test. The
selected bands are a hyperparameter of the input representation chosen with test labels in
scope, and feature selection outside the resampling loop is a known and quantified source of
optimism — Ambroise & McLachlan, *PNAS* 99(10):6562–6566 (2002), who obtain near-zero apparent
error on *random labels* that way. This is genuine label leakage independent of the split
protocol: it contaminates `grouped` too. `--fold k` restricts every step — the correlation
pre-filter, the FDR diagnostic, mRMR, SPA and the cross-validated curve — to fold $k$'s
training patches.

Neither defect argues that 40 bands is *wrong*; both argue that nobody knows, and that a study
about what the acquired spectrum carries must not open by discarding 84% of it on an
undemonstrated elbow. So the full cube became the default and the question became measurable:
ablation **A2** compares `full_256` (the reference), `spa40_whole_corpus` and
`spa40_within_fold`, and `spectralquadnet.bandstudy` runs the curve to $k = 256$ with two null
methods so an elbow can be demonstrated rather than asserted. `cfg.deployed_curve_path`
additionally lets the proxy curve be overridden by cross-validating the actually-deployed
estimator.

**Outputs** — `dataset/patches_spa_40b.npy` $(8624, 40, 64, 64)$ `float32` (written in
2,048-row chunks off the memory map), `dataset/wavelengths_spa_40b.csv`, whose 40 centres
span $383.22$–$1006.47$ nm, and `dataset/band_selection_elbow.json`, the serialised
`ElbowVerdict`. Read only by `configs/data/ablation/spa40_*.yaml`.

**The cheaper mechanism.** A k-band arm does not need a materialised cube at all:
`data.band_indices_path` names a `.npy` of band indices and `RiceSeedDataset._load_patch`
slices each patch **as it comes off the mmap**, touching only the selected bands' pages. A
40-of-256 run therefore reads ~16% of the bytes a full read would. That is what
`spectralquadnet.bandstudy`'s neural confirmation arms use, and it is the difference between a
20-cell band sweep being a disk-space problem and being a config change.

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
| **Exactly one patch per item** | `np.array(self.patches[ri])` materialises $(256,64,64) = 4{,}194{,}304$ B $= 4$ MiB on the primary path ($0.64$ MiB at $k{=}40$); no batched `.copy()` anywhere on the loader path |
| **The spectral axis is checked, not assumed** | `band_geometry(cfg.data, store)` runs before the model is built and raises `BandGeometryError` if the cube's band axis, the optional `band_indices_path`, the wavelength CSV's row count and `data.num_bands` are not one number |
| **The mapping survives a process boundary** | `RiceSeedDataset.__getstate__` drops `patches`/`masks` before pickling and `__setstate__` re-opens them from the recorded paths — `np.memmap` inherits `ndarray`'s pickling, which *materialises*, so sending one to a worker would copy the whole cube (four workers ≈ 22 GB). `labels` (69 kB) and the standardised `morph` (276 kB) are ordinary in-RAM arrays and do travel. A store built from arrays rather than files raises here by name rather than leaking gigabytes — valid for the unit tests, but it cannot cross into a worker, so that configuration needs `runtime.num_workers=0`. |

Labels ($8624 \times 8$ B) are read fully into RAM; wavelengths are read with a sniffed
delimiter (`sep=None`), taken from the **last** CSV column, and min–max normalised to $[0,1]$
before being moved to the training device:

$$
\tilde\lambda_i = \frac{\lambda_i - \min_j \lambda_j}{\max_j \lambda_j - \min_j \lambda_j}
$$

Typed accessors `require_patches()` / `require_labels()` / `require_wavelengths()` raise
rather than return `None`, so a missing load fails at the call site.

**Footprint.** The primary cube is $8624 \times 256 \times 64 \times 64 \times 4\,\text{B} = 36.2$ GB
on disk; the retained 40-band ablation array is $5.65$ GB on
disk. The README records a measured **peak RSS of 1.39 GB, median 0.65 GB** across a full
three-stage run against that file, with mean usage *falling* over the run — clean mapped
pages being reclaimed, not a dataset accumulating.

Worker processes each carry **their own page table** for that file — one mapping per worker, no
resident bytes — which is why the auto worker count is bounded: `MAX_AUTO_WORKERS = 8` on CUDA,
`NON_CUDA_AUTO_WORKERS = 2` on Metal/CPU, where a worker's pages and the accelerator's
activations come out of the same unified pool (`06_EXECUTION_AND_HARDWARE.md` §6.3). The mapping
is also what dominates the host cost per sample: measured at 1.86 ms, of which 1.41 ms is the
mmap page-in and only ~0.45 ms is augmentation — so the feed is I/O-bound, not CPU-bound, and
more workers buy little.

---

## 2.6 Dataset and augmentation profiles

`RiceSeedDataset.__getitem__(idx)` returns **CPU tensors and never touches an accelerator** —
building tensors directly on the training device would cost one host-to-device copy per sample
per tensor and would make `num_workers > 0` impossible (a worker process cannot hand a CUDA
tensor back through the multiprocessing queue). The batched transfer happens once, in
`engine/batch.py::unpack_batch`. The return shape is also conditional: with neither a persisted
mask nor morphometrics configured it is the 2-tuple
$\big(\text{patch}\in\mathbb{R}^{C\times64\times64},\;\text{label}\in\mathbb{Z}\big)$, $C$ =
`data.num_bands` (256 on the primary path); with
either `data.masks_path` or `data.morphology_path` set, it is a 4-tuple
$(\text{patch},\,\text{label},\,\text{mask\_or\_ABSENT},\,\text{morph\_or\_ABSENT})$, where
`ABSENT = torch.zeros(0)` is a zero-length sentinel signalling "not configured" (a sentinel
rather than `None` because `None` cannot survive PyTorch's default collate). `unpack_batch`
accepts both shapes transparently everywhere in `engine/`.

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
| spectral warp | resample the band axis by $\alpha \sim \mathcal{U}(1-w,\,1+w)$ (linear interpolation), then centre-crop or centre-pad back to $C$ bands |
| multiplicative | $x \leftarrow x \odot (1 + 0.05\,s\,\epsilon_c)\, m$, one scalar per band |
| spatial | random horizontal flip, random vertical flip, random $k\cdot 90^\circ$ rotation — applied **unconditionally** whenever a profile is active |

> **Ordering invariant.** Seven `if torch.rand(1) < p[...]` guards — the five above, plus the two
> CutMix guards below — consume the global torch RNG stream in a fixed order. Reordering them,
> even when an augmentation is a no-op at $p=0$, changes every subsequent draw and breaks
> fixed-seed reproducibility. The module docstring states this explicitly.

### Same-class CutMix

Two more Bernoulli-gated primitives, appended **after** the five above and **before** the spatial
dihedral transform, each guarded so a profile with the probability at $0$ consumes no RNG state
and reproduces the exact stream of the profile with CutMix disabled:

| Profile | `spec_cutmix` | `spat_cutmix` |
|---|---:|---:|
| `heavy` | 0.10 | 0.10 |
| `medium` | 0.08 | 0.08 |
| `very_light` | 0.06 | 0.06 |
| `light` | 0.06 | 0.06 |
| `none` | — | — |

Configured by `data.cutmix_bands` — **51** on the primary path and **8** on the 40-band arm,
which is the same $\approx20\%$ of the spectral axis in both. A band is not a fixed quantity of
spectrum, so both band-expressed widths are derived from a fixed fraction by
`data/datasets.py::band_augmentation_widths`; left as literals they would make the primary
pipeline and every band-selection arm run a *different* augmentation while claiming to differ
only in the band count. `tests/unit/test_cutmix.py` checks the YAML against the rule in both
directions. Also configured by
`data.cutmix_spatial` (shipped $24$, $\approx14\%$ of the $64\times64$ patch area).

**Partner selection** — lazily built per dataset, only if the active profile needs it: a
`{label: positions}` index over *this split's own rows only* (a train dataset can never draw a
val/test partner), drawn uniform-over-pool-minus-anchor in one shot (no rejection sampling); a
class with fewer than 2 members in the split is skipped (CutMix silently no-ops for it). The
partner is used **un-augmented** — no independent augmentation draws stack on it, so the
composite's statistics don't depend on the partner's own luck.

$$
\text{spectral (band window): } x_{[t:t+w]} \leftarrow x^{\text{partner}}_{[t:t+w]}, \quad w=\texttt{cutmix\_bands}
$$

$$
\text{spatial (square, all bands): } x_{[:,\,r:r+s,\,c:c+s]} \leftarrow x^{\text{partner}}_{[:,\,r:r+s,\,c:c+s]}, \quad s=\texttt{cutmix\_spatial}
$$

with the window/region start drawn uniformly at random on each call; for `spat_cutmix` the
partner's fill mask (if present) travels with the pasted region, so the dihedral spatial
transform applied afterwards sees a consistent foreground. Both operators clone before writing
(the anchor tensor is never mutated in place).

**Label-preserving by construction** — the partner is the same class, so no soft target is
produced. Unlike mixup, this is not excluded by a non-zero ArcFace margin: the
`arc_m > 0` guard in `train_epoch.py` checks for mixup specifically, and is untouched by CutMix
firing. This is what lets Stage 2/3 — which run mixup off, at $\sim$67 training samples/class —
still get some intra-class variation without ever changing a label (§4.4).

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

Passing `class_weights=None` falls back to uniform class selection. **The RNG is unseeded by
default** (`np.random.default_rng()` per `__iter__`) on a single-process run — a documented
source of run-to-run non-determinism there. Under DDP, `build_loaders` passes an explicit `seed`
(so every rank composes the *identical* global batch before it is sharded — see below), and the
sampler advances a per-epoch stream `np.random.default_rng([seed, epoch])` via `set_epoch()`,
making it fully deterministic in that configuration.

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

Under DDP, balanced batches are sharded rather than independently redrawn per rank:
`DistributedBatchShardSampler` wraps the (seeded) `ClassBalancedBatchSampler` and gives each rank
a **contiguous slice** of the one globally-identical batch, preserving the $n_{\text{cls}}\times
n_{\text{spc}}$ balance guarantee across shards; `DistributedIndexShardSampler` does the
analogous round-robin split for `HardClassOversampledSampler`'s flat weighted stream. The plain
(non-balanced) train path shards its configured batch size directly
(`batch // world_size`, refused rather than rounded if it doesn't divide evenly). Full DDP
mechanics are in `06_EXECUTION_AND_HARDWARE.md`.

---

## 2.8 Split protocols and the calibration split

`data/loaders.py::build_split_bundle` builds one of two protocols, selected by
`cfg.data.split_scheme`, with a module-level `SPLIT_SEED = 42` deliberately decoupled from
`cfg.seed`.

**`stratified`** (`configs/data/hsi256_stratified.yaml`) — a two-step stratified
`train_test_split` at the **patch** level, $8{,}624\to6{,}036/1{,}294/1{,}294$ (70/15/15). It
puts every one of the dataset's **180 acquisition bundles** in train *and* in val/test — the
executed run measured `180 of 180 scans are in train and in val/test` — so part of the reported
number under this protocol is bundle recognition, not variety recognition.

> **Corrected (CHANGES.md §3.1 / IC-13).** Earlier revisions of this document stated **107**
> capture scans, "measured 107/107". That figure was wrong. The Zenodo record (3241923) images
> each of the 90 varieties as **two bundles of 48 kernels**, each bundle a tray of one single
> variety, giving $90 \times 2 = 180$ acquisition units — which matches the executed run exactly
> ($8{,}624 / 180 = 47.9$ patches per bundle). 107 is not divisible by 90 and contradicts the
> "exactly two scans per class" statement made in the same paragraph.
> `tests/unit/test_splits.py::test_the_documented_bundle_count_is_180` pins it.

**This is no longer the default protocol.** CHANGES IC-3 switched the shipped experiment to
`grouped`, because a number from this split is a claim about acquisition sessions and a number
from `grouped` is a claim about rice varieties. `stratified` is retained as the *contrast* arm of
ablation A1, whose gap `F1_stratified − F1_grouped` quantifies how much of reported performance
on this dataset is acquisition recognition.

**`grouped`** (`configs/data/hsi256_grouped.yaml`, **the default**) — holds out whole scans via
`grouped_split`, rotating which scans are held by `data.split_fold` and targeting
`data.split_eval_frac` of each class's **groups** (not patches) for val∪test. It requires
`groups.npy`. On this archive every variety was captured in exactly **two** scans, so a class has
exactly two groups: full three-way group-disjointness is mathematically impossible, and
`grouped_split` reports this rather than asserting an unmeetable contract. A `SplitReport`
records, per class, whether train/eval and val/test are group-disjoint, which classes have only
one group, and which leak — `data.split_eval_frac=0.30` on a two-scan-per-class dataset realises
close to a 50/50 one-scan-out split rather than 70/15/15, with val and test literally two halves
of the same held-out scan. `single_group_policy` (`"error"`, default — refuses to build the split
and names the offending classes; `"patch_split"` — accepted-leak fallback) governs what happens
when a class has fewer than 2 groups to split. Sweeping `split_fold` over `{0, 1}` is the
leave-one-scan-out cross-validation this dataset can actually support.

`stratified` and `grouped` trade off differently: an ungrouped result is a claim about
acquisition sessions as well as varieties; a grouped result is a claim about varieties alone, at
the cost of a coarser, harder-to-balance split on a dataset with only two scans per class.

**The calibration split** (`data.calib_frac`) carves an inner `calib` split out of `train` (by
group under `grouped`, by patch under `stratified`), never out of val/test. The per-class
margins (§4.2), the CDWS weights, and the Phase-3 oversampling weights are fitted there, so
`val` — the split that also selects the checkpoint — carries no fitted parameter.
`configs/data/ablation/spa40_audited.yaml` — the frozen historical replica — ships
`calib_frac: 0.0` (everything fitted on `val`, so that split carries fitted parameters as well
as selecting the checkpoint); every current config ships
`calib_frac: 0.15`.

`data/morphometrics.py` fits the morphometric standardisation (`MorphometricStats`,
`STD_FLOOR = 10^{-6}`) on `train_idx` alone — once, threaded to every loader that needs it,
rather than refit per stage.
