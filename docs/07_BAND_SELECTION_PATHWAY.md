# 07 — The band-selection research pathway

> How much of the 256-band cube does this task need, which bands are useful, and which
> selection strategy is appropriate?

`python -m spectralquadnet.bandstudy.cli` (or `python scripts/run_band_study.py`).

---

## 0. Where this sits — read this first

**Nothing in this document is on the primary path.** The study's primary methodology is to
train on the **complete 256-band cube with no band selection and no dimensionality reduction
of any kind** (`configs/data/hsi256_grouped.yaml`, `01_ABSTRACT_AND_OVERVIEW.md` §1.1). A
default `python train.py` never runs a selector, never reads a band-index file, and prints
`Spectral: 256 bands — the full acquired cube, no band selection (primary methodology)` on its
first screen.

This pathway is **retained, not deprecated**, and the distinction matters:

- *Retained*, because "how few bands would do?" is a real research question with a real
  deployment consequence — a multispectral instrument costs a fraction of a hyperspectral one —
  and because deleting the machinery that can answer it would turn the primary path's refusal
  to reduce from a **measured choice** into an **assumption**. That is the precise defect this
  whole revision exists to correct, applied to itself.
- *Not the default*, for the two reasons in §1: neither shipped $k$ was demonstrated, and the
  whole-corpus selection saw test labels.

Everything here is reachable only by explicit opt-in:

| Component | Entry point |
|---|---|
| The experiment | `python -m spectralquadnet.bandstudy.cli` |
| The build step | `python scripts/select_bands.py` |
| Reduced data configs | `python train.py data=ablation/spa40_grouped` |
| Index-file slicing | `python train.py data.band_indices_path=… data.num_bands=k` |
| The gateway ablation | `python -m spectralquadnet.experiments.cli ablate A2` |

Ablation **A2** takes the primary path as its **reference arm**, so the question it answers is
"what does reducing cost?" rather than "which reduction is best".

---

## 1. Why this exists

The repository ships two band selections and **neither was chosen by an experiment that could
have returned a different answer**:

| artifact | k | how it was chosen | the problem |
|---|---:|---|---|
| `dataset/patches_spa_40b.npy` | 40 | mRMR/SPA on **all 8,624 patches**, curve validated at k ∈ {5 … 40} | Test labels were in scope (F2 / §4.1). The curve terminates at 40, so the 98 %-of-peak elbow criterion is satisfied *vacuously* (M-14). |
| `dataset/folds/patches_fold{0,1}_100b.npy` | 100 | mRMR/SPA on fold training rows, curve validated at k ∈ {5 … 100} | Leakage fixed (IC-4). But `band_selection_elbow_fold0.json` records `"demonstrable": false` — the curve again terminates at its own chosen k. |

Both elbow files say so themselves. So the 100-band choice is *the largest value that was
evaluated*, and there is no evidence in the repository about what happens at 128, 192 or 256 —
nor about whether mRMR and SPA are the right two methods to have compared.

That is why **the default is now the full cube**: an undemonstrated elbow is not a defensible
place to start a study about what the acquired spectrum carries. The reduced arrays are kept
and remain fully runnable (`configs/data/ablation/`), and this study is the experiment that can
return a different answer. It is designed so that "use all 256 bands" and "no method beats a
random subset" are both reachable conclusions, and the report says so plainly when they are.

### What it is not

It is **not** a replacement for `scripts/select_bands.py`. That script is a *build step*: it
materialises a reduced `.npy` cube at one band count. This is the *experiment* that decides what
that band count should be. It writes band **index** files, not cubes — a 100-band reduced cube
is 14 GB and this study evaluates thousands of band sets.

---

## 2. The design

### 2.1 Three questions, kept separate

1. **How many bands?** A budget sweep from k = 1 to the full k = 256. The full band count is
   *mandatory* — `BandStudyConfig.validate` refuses a grid without it — because an elbow is only
   falsifiable if the curve extends past it.
2. **Which bands?** Selection frequency over folds × replicates × methods, reported in
   nanometres with the redundancy of each set quantified.
3. **Which method?** Twelve strategies, including two nulls.

### 2.2 Leakage discipline

The same discipline the rest of the project runs under, applied one level up. Band selection is
a hyperparameter of the input representation, so choosing it on data that is later scored is
C-1 wearing different clothes.

| partition | may do | enforced by |
|---|---|---|
| `train` | fit the selectors; fit the proxy models | selectors are handed `FoldData.train` and nothing else |
| `calib` | decide the budget and the method | `analysis.load_records` filters to `split == "calib"` and **drops** anything else, loudly |
| `val ∪ test` | confirm an already-fixed choice, once | reachable only through `FoldData.reveal_heldout()`, which logs every call at WARNING |

The `confirm` stage refuses to run before `analyse` has written a recommendation. That ordering
is the entire reason the stage is separate.

The splits come from `spectralquadnet.data.loaders.grouped_split` — the *same* builder the
training runs use, with the same parameters as `configs/data/hsi256_grouped.yaml` — so the
rows a selector may see are exactly the rows a training run would put a gradient through.

### 2.3 Two selection scopes

Each (fold, method) is selected twice over:

* **canonical** (`rep=full`) — on the whole training split. This is what a deployed pipeline
  would produce, it is what gets written to `bands/`, and its calib curve is the primary curve.
* **replicates** (`rep=0…R-1`) — on stratified 80 % subsamples. These measure *selection
  stability* and *metric variance*, which the canonical selection cannot report about itself.

`random` is exempt from the replicates: its uncertainty is the spread across draws, which it
already has.

### 2.4 The methods

| method | family | supervised | nested | reference |
|---|---|---|---|---|
| `uniform` | **null** | no | no | — |
| `random` | **null** | no | no | — |
| `variance` | univariate | no | yes | — |
| `fdr` | univariate | yes | yes | Fisher (1936) |
| `mi` | univariate | yes | yes | Kraskov et al., Phys Rev E 69:066138 (2004) |
| `mrmr` | redundancy | yes | yes | Peng, Long & Ding, IEEE TPAMI 27(8):1226-38 (2005) |
| `spa` | geometric | no | yes | Araújo et al., Chemom Intell Lab Syst 57(2):65-73 (2001) |
| `cluster_ward` | redundancy | yes | no | Ward (1963); Martínez-Usó et al., IEEE TGRS 45(12) (2007) |
| `pca_loading` | geometric | no | yes | Chang et al., IEEE TGRS 37(6):2631-41 (1999) |
| `l1_path` | embedded | yes | yes | Tibshirani, JRSS-B 58(1):267-88 (1996) |
| `tree_importance` | embedded | yes | yes | Geurts, Ernst & Wehenkel, Mach Learn 63(1) (2006) |
| `pls_vip` | chemometric | yes | yes | Wold, Sjöström & Eriksson, Chemom Intell Lab Syst 58(2) (2001) |

`mrmr` and `spa` are the repository's incumbents, reimplemented with the same decorrelation
pre-filter (|r| > 0.995) and the same MI-seeded SPA start, so the arms are the same arms.

**The two nulls are not optional.** With neighbour correlations above 0.99 a random 20-band
subset already spans most of the usable spectrum. `null_margins.csv` reports every method's
advantage over a random subset of the same size, and `method_ranking.csv`'s `effective` column
is a pass/fail on it. A method that never clears the margin is reported as *not selecting* —
it is subsetting.

*Nested* means every budget is a prefix of one ranking, so k does not multiply the method's
selection cost. `uniform`, `cluster_ward` and `random` are per-budget; their k = 20 set need not
overlap their k = 40 set at all, which is why the stability table separates them.

### 2.5 The proxies, and what a proxy conclusion is worth

| proxy | family | note |
|---|---|---|
| `lda` | generative-linear | the repository's most important baseline (CHANGES §19.4), recomputed at every budget |
| `linsvc` | discriminative-linear | C = 0.1 **fixed** across budgets — a per-budget tuned C would make the curve a curve through a tuning procedure |
| `extratrees` | nonlinear-ensemble | the only proxy that can use a band conditionally on another |

All three read the **foreground-masked mean spectrum**, which discards every spatial cue. The
project's own bracket puts ~25 macro-F1 points of the deployed model in exactly what they
discard (0.5916 for LDA vs ~0.845 for the full model, same leaky protocol).

So, and the report repeats this in three places:

> **A proxy plateau is a LOWER bound on the useful band budget, not an upper one.**
> CHANGES F-3 predicts a spatial-spectral network's curve keeps rising past it. The `neural`
> stage is what settles that.

Three families rather than one because "how many bands are needed" is allowed to depend on what
is reading them. Where the three agree the budget is a property of the spectra; where they
disagree it is a property of the estimator, and `analysis.detect_flags` raises
`budget_is_model_dependent` when they differ by more than 4×.

### 2.6 Pre-registered decision rules

Set in `BandStudyConfig`, printed in §1 of the report, and every one can return "no":

| rule | default | flag |
|---|---|---|
| plateau: smallest k within `plateau_tol` of the curve's own peak | 0.01 macro-F1 | `--plateau-tol` |
| stable: mean replicate Jaccard ≥ `stability_floor` | 0.50 | `--stability-floor` |
| effective: beats the random null by ≥ `null_margin` somewhere | 0.01 macro-F1 | `--null-margin` |

A plateau is **demonstrable** only when the curve extends past it. That is `verify_elbow`'s
criterion from `band_selection.py`, applied to every one of the ~1,400 curves.

---

## 3. Running it

### 3.1 First, cost nothing

```bash
python -m spectralquadnet.bandstudy.cli list
```

Prints the grid, the methods with their references, the proxies, and the cell counts for every
stage. Nothing is read and nothing is written.

### 3.2 The complete analysis

```bash
python -m spectralquadnet.bandstudy.cli all
```

Runs `prepare → select → proxy → analyse → report`. It does **not** run `confirm` (which spends
held-out evidence) or `neural` (which spends GPU-days); both are named at the end of its output.

Roughly 2–3 hours on a laptop at the defaults — dominated by ~9,000 proxy fits. It is fully
resumable, so interrupt it freely.

Cheaper variants:

```bash
# every stage, every artifact, minutes instead of hours — for verifying the machinery
python -m spectralquadnet.bandstudy.cli all --quick

# the two linear proxies and three replicates: ~40 % of the cost, most of the evidence
python -m spectralquadnet.bandstudy.cli all --proxies lda linsvc --replicates 3
```

### 3.3 Individual stages

```bash
python -m spectralquadnet.bandstudy.cli prepare   # 36 GB read → cached mean spectra + splits
python -m spectralquadnet.bandstudy.cli select    # every method, training rows only
python -m spectralquadnet.bandstudy.cli proxy     # the budget sweep, scored on calib
python -m spectralquadnet.bandstudy.cli analyse   # trends, stability, flags, recommendation
python -m spectralquadnet.bandstudy.cli report    # REPORT.md + figures
python -m spectralquadnet.bandstudy.cli inspect   # what has run so far
```

Restricting the grid — the flags compose, and every stage takes all of them:

```bash
python -m spectralquadnet.bandstudy.cli select --methods mrmr spa uniform random
python -m spectralquadnet.bandstudy.cli proxy  --budgets 10 20 40 80 160 256
python -m spectralquadnet.bandstudy.cli proxy  --folds 0
python -m spectralquadnet.bandstudy.cli proxy  --proxies lda
```

> **Note.** The grid flags are part of the study's *fingerprint*. Running `select` with four
> methods and then `proxy` with twelve is two different studies, and the manifest check refuses
> it rather than silently building one table out of both. To narrow a run, pass the same flags
> to every stage — or use a separate `--output-root`.

### 3.4 Held-out confirmation — after reading the report

```bash
python -m spectralquadnet.bandstudy.cli confirm
```

Scores a **fixed list** on `val ∪ test`, once, with bootstrap intervals: the recommended
configuration, the aggressive one, the runner-up methods, the full band set, and the
repository's two incumbents where their budgets are on the grid. It refuses to run before
`analyse`. Every reveal is logged to `logs/confirm_*.log`.

### 3.5 Neural confirmation

```bash
python -m spectralquadnet.bandstudy.cli neural              # plan only — prints, writes, runs nothing
python -m spectralquadnet.bandstudy.cli neural --execute    # launch the runs
python -m spectralquadnet.bandstudy.cli neural --seeds 0 1 2 --execute
bash outputs/band_study/neural/commands.sh                  # or run them yourself, anywhere
```

Each arm is a `train.py` run of the repository's default composition, differing **only** in

```
data.patches_data=./dataset/patches.npy          # the FULL cube
data.band_indices_path=outputs/band_study/bands/<method>_f<fold>_k<budget>.npy
data.wavelength_path=outputs/band_study/bands/<method>_f<fold>_k<budget>_wavelengths.csv
data.num_bands=<budget>
data.cutmix_bands=…  data.max_cutout_bands=…     # clamped below k; see below
```

`data.band_indices_path` (new, BS-1) makes the dataset slice each patch to those bands **as it
comes off the mmap**. A k = 100 reduced cube is 14 GB, so materialising one per (method, fold,
budget) would be terabytes for an experiment whose content is a list of integers. It also reads
fewer bytes per patch than a full read, because the fancy index touches only the selected bands'
pages.

The two augmentation widths that are *expressed in bands* are clamped below k, and the clamped
values appear in the printed override list: a `cutmix_bands=8` window on a 5-band input swaps
the whole spectrum, which is a relabelling rather than a CutMix.

---

## 4. Output tree

```
outputs/band_study/
  study.json                       config, fingerprint, per-stage outcomes
  REPORT.md                        ← the deliverable
  logs/<stage>_<timestamp>.log     including every held-out reveal
  cache/
    mean_spectra_<hash>.npy        the 36 GB read, done once
    splits.json                    every fold's sizes and SplitReport
  selections/fold<k>/rep_<full|0..R-1>/<method>.json
                                   ranking, per-budget sets, timings, failures, scope
  bands/
    <method>_f<fold>_k<budget>.npy             ← data.band_indices_path
    <method>_f<fold>_k<budget>_wavelengths.csv ← data.wavelength_path
  proxy/records.jsonl              one row per fit — the raw evidence
  confirm/records.jsonl            the held-out confirmations
  analysis/
    curves.csv                     (fold, method, proxy, budget) → score + spread
    trends.csv                     per-curve shape verdict, plateau, knee
    null_margins.csv               each method's advantage over random
    stability.csv                  replicate agreement per (fold, method, budget)
    cross_fold_agreement.csv       do the two acquisition folds pick the same bands
    redundancy.csv                 correlation, effective rank, spectral coverage
    method_ranking.csv             the method comparison
    wavelength_frequency.csv       selection frequency per band, in nm
    flags.json                     every automated check that fired
    recommendation.json            machine-readable recommendation
    tables/*.md, *.csv             publication tables
    figures/*.png                  publication figures
  neural/
    plan.json, commands.sh         the confirmation grid
    <arm>__f<fold>_s<seed>/        standard run directories
```

### Figures

| file | shows |
|---|---|
| `budget_curves.png` | score vs k, one panel per proxy, **random null as a shaded band** |
| `budget_curves_per_fold.png` | the same, split by acquisition fold — do the folds agree? |
| `plateau_summary.png` | where each curve plateaus; hollow markers = plateau is the endpoint |
| `selection_stability.png` | replicate Jaccard vs k, with chance drawn in |
| `redundancy.png` | rank efficiency and spectral coverage vs k |
| `method_margins.png` | best Δ vs random per method; grey = never cleared the margin |
| `compute_tradeoff.png` | score vs band count, with the Pareto front |
| `wavelength_frequency.png` | selection frequency vs nm, over the corpus mean spectrum |
| `confirm_heldout.png` | held-out scores with bootstrap CIs (after `confirm`) |

---

## 5. Resuming

Resumability is a property of the **artifacts**, not of a checkpoint file. Re-run the same
command:

```bash
python -m spectralquadnet.bandstudy.cli all      # picks up exactly where it stopped
```

* `select` skips a (fold, rep, method) whose JSON exists.
* `proxy` and `confirm` skip a cell whose record key is already in `records.jsonl`. Records are
  appended and flushed per cell, so a kill loses at most the last line — and a truncated final
  line is dropped with a warning rather than being fatal.
* `analyse` and `report` are pure functions of what is on disk and always recompute.

`study.json` records the config fingerprint. Resuming into a directory whose records came from a
**different** configuration is refused, with the differing fields named:

```
outputs/band_study holds results from a different study configuration
(fingerprint 6428f03f…, this run is 2f1cbf00…). Resuming would build one table
out of two experiments. Use a fresh --output-root, or --force to overwrite.
The differing fields are:
  budgets: [5, 256] -> [10, 256]
```

Fields that cannot change a number — `output_root`, `verbose`, `jobs`, `note` — are excluded
from the fingerprint, so resuming with `-v` is still the same study.

`--force` recomputes cells that already have results.

---

## 6. Reading the results

```bash
python -m spectralquadnet.bandstudy.cli inspect     # stages, artifact counts, last recommendation
$PAGER outputs/band_study/REPORT.md                 # the deliverable
open outputs/band_study/analysis/figures/           # the figures
```

`REPORT.md` sections: protocol and leakage discipline → how many bands → which method → which
wavelengths → stability and redundancy → **automated checks** → held-out confirmation → neural
confirmation → conclusions and recommendations → where the evidence lives.

Machine-readable:

```bash
jq . outputs/band_study/analysis/recommendation.json
jq '.[] | select(.severity=="critical")' outputs/band_study/analysis/flags.json
python -c "import pandas as pd; print(pd.read_csv('outputs/band_study/analysis/method_ranking.csv'))"
```

### The automated checks

`analysis.detect_flags` raises these, ordered critical → warning → info. Each exists because the
corresponding mistake is one this project's audit found or one this design makes possible.

| code | severity | fires when |
|---|---|---|
| `no_plateau_anywhere` | critical | **no** curve plateaus inside the range — the study found a lower bound, not an answer |
| `no_method_beats_the_null` | critical | no named method beats a random subset anywhere |
| `implausible_scores` | critical | cells above 0.98 macro-F1 — points at the split or the cache, not at good bands |
| `some_plateaus_not_demonstrable` | warning | some plateaus are the endpoint of their curve (M-14) |
| `more_bands_hurt` | warning | curves peak in the interior and fall — extra bands cost accuracy |
| `budget_is_model_dependent` | warning | plateau differs > 4× across proxy families |
| `budget_is_fold_dependent` | warning | plateau differs ≥ 2× between acquisition folds |
| `methods_no_better_than_random` | warning | named methods that never clear the null margin |
| `null_wins` | warning | the best method at k ≤ 40 *is* a null |
| `worse_than_random` | warning | a method averaging below the null |
| `unstable_selections` | warning | replicate Jaccard below the floor — no wavelength claim may be made |
| `folds_choose_different_bands` | warning | the two folds select nearly disjoint sets |
| `narrow_spectral_coverage` | warning | selections spanning < 35 % of 385–1006 nm |
| `implausible_at_tiny_budgets` | warning | ≤ 2 bands doing > 60 % of the full-budget job |
| `failed_cells` | warning | proxy fits that failed — holes in the tables, named |
| `trend_census`, `redundancy_census`, `chance_stability` | info | the distributions a reader needs to calibrate against |

---

## 7. Tests

```bash
pytest tests/unit/test_band_study.py                       # ~2 s, no dataset needed
pytest tests/smoke/test_band_study_e2e.py --run-slow       # the whole study + a real train.py run
```

The unit tier gates the properties whose failure would be *silent*: selection rows disjoint from
calib/val/test; the analysis dropping any non-calib record; `confirm` refusing before `analyse`;
every method returning exactly k distinct in-range bands at every budget; every ranking reaching
k = C; supervised methods recovering a planted informative window; Kuncheva returning 0 at
chance and `nan` at k = C; single-set stability not being reported as 1.0; each of the four trend
shapes being reachable; the recommendation falling back to `uniform` when nothing beats the
null; the fingerprint ignoring non-semantic fields; the record store resuming; and the
band-slicing data path.

The smoke tier runs the whole study on a miniature cube and — the load-bearing one — a real
`train.py` composition with `data.band_indices_path` set, because `num_bands`, the wavelength
CSV and the index array are three separate keys that have to agree and would otherwise fail deep
inside a branch naming none of them.

---

## 8. Limitations, stated up front

1. **The proxies are not the model.** Mean spectra discard all spatial structure. A proxy
   plateau is a lower bound on the network's budget (CHANGES F-3). The `neural` stage is not
   optional if the budget is going to be treated as settled.
2. **Two folds, one training bundle per class per fold.** There is no third acquisition bundle.
   Fold-to-fold spread is an n = 2 estimate and is reported as a range, never as a standard
   error.
3. **Selection frequency is not importance.** Bands correlated above 0.99 are interchangeable; a
   frequently-chosen band may be the arbitrary representative of a group. `redundancy.csv`'s
   rank-efficiency column is the check.
4. **The budget grid is finite.** A plateau is located to the nearest grid point.
5. **`cluster_ward`'s Ward linkage is O(C²) in memory** — fine at C = 256, not at C = 10,000.

---

## 9. Relation to the rest of the repository

| | |
|---|---|
| `configs/data/hsi256_grouped.yaml` | **the primary path**, and this study's reference point. The protocol is mirrored exactly — same split builder, same `split_eval_frac`, same `calib_frac` — so a budget curve is comparable with the headline runs. |
| `scripts/select_bands.py` | the build step. Materialises one reduced cube at one k. Use it *after* this study has decided k; the neural arms do not need it at all. |
| `configs/data/ablation/` | the shipped reduced arms: `spa40_grouped`, `spa40_stratified`, and the frozen `spa40_audited` replica. |
| **A2** (`cli ablate A2`) | the gateway. Three arms — `full_256` (reference), `spa40_whole_corpus`, `spa40_within_fold` — reading two deltas from one table: what the reduction costs, and what selecting outside the fold leaks. The `bands/` artifacts here supply matched within-fold selections at *every* budget, which turns A2 into a curve. |
| **A12** | run-to-run σ. Every neural delta in §8 of the report is interpreted against it. Run it first. |
| `data.band_indices_path` | the mechanism that makes a k-band training arm a config change rather than a 14 GB cube. The neural stage's arms are built on it, and at k = 256 they reproduce the primary composition exactly — including its augmentation widths, via `band_augmentation_widths`. |
| CHANGES §4.1, §19.3, M-14, F-3, IC-4 | the findings this study answers. |
