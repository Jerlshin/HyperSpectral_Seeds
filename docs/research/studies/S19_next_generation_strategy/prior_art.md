# Current prior art and its implications

Search/verification cutoff: **2026-10-03**. This is a broad, structured primary-source
review focused on decisions for this dataset, not a systematic-review claim of
exhaustive coverage. Thirty-nine records span the exact rice benchmark, adjacent
seed tasks, spectral/spatial models, foundation models, multimodal optimization,
domain shift and evaluation. Each record has source, bibliographic identity, access
level, protocol scope and local implication in [literature.csv](../../evidence/S19_next_generation_strategy/literature.csv).
[references.bib](../../evidence/S19_next_generation_strategy/references.bib) is a
working bibliography; deliberately abbreviated author lists should be expanded
before submission. [Search log](../../evidence/S19_next_generation_strategy/search_log.md)
records search families, exclusions and access limits. A verified abstract does not
mean independently verified code, splits, results or replication.

## 1. The closest rice results do not form a comparable leaderboard

| Study | Verified source result | What is known about comparability |
|---|---|---|
| Fabiyi 2020 [L02] | 90 varieties; RGB shape + HSI-derived features; 78.27% F1 | 4:1 train/test stated; no acquisition-disjoint reconstruction in inspected methods |
| Filipović 2021 [L03] | Morphological-feature paper identity verified | Full original protocol/result unavailable here; 85.65% F1 appears in later tables, not treated as independently confirmed |
| FusedNet 2023 [L31] | 87.27% accuracy, 86.87% average F1; feature SVM + RGB ResNet50 | Publisher abstract verified; split unverified |
| Taheri 2024 [L04] | 15 HSI bands + RGB/deep ensemble; 92.73–96.17% overall precision | Metric is precision; no verified grouped protocol from accessible preview |
| Hu 2025 [L05] | Auxiliary RVHID90 table97.78% accuracy, 97.67% F1 | Main experiment is maize; rice split IDs/calibration/selection scope insufficiently specified to reconstruct here |
| Huang 2026 [L06] | 90-class 2D-RGB/3D-HSI adaptive gate; 93.50% accuracy | August2026 publisher abstract; exact dataset identity and split unverified |

Sources: [Fabiyi full paper](https://strathprints.strath.ac.uk/71055/19/Fabiyi_etal_IEEEAccess_2020_Varietal_classification_of_rice_seeds_using_RGB.pdf),
[Filipović proceedings](https://www.proceedings.com/content/060/060646webtoc.pdf),
[FusedNet](https://link.springer.com/chapter/10.1007/978-3-031-43605-5_3),
[Taheri](https://link.springer.com/article/10.1007/s12652-024-04782-2),
[Hu](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1599231/full),
[Huang](https://xnmz.cbpt.cnki.net/portal/journal/portal/client/paper/8ef266236f95cfbb6a2918e2135d7fcb).

**Implication:** current v5 has not demonstrated superiority over these methods on
matched conditions; nor do these high numbers establish superiority on our grouped
session diagnostic. Missing protocol information is not proof of leakage. Reproduce
at least the closest simple multimodal method and strongest available conventional
competitor on common kernels/splits. Keep published numbers in a contextual table,
with metric and protocol qualifications, not in a bar chart ranking unlike tasks.

A useful bibliographic correction emerged from checking primary sources: FusedNet
is in Agriculture-Centric Computation, CCIS1866, pages28–42; a later rice reference
misstates its venue/pages. Hu's Table4 also has baseline attributions inconsistent
with the original paper identities. Cite originals rather than copying a recent
paper's comparison table.

## 2. Seed-specific lessons beyond this benchmark

[ClusterRiceNet](https://www.sciencedirect.com/science/article/pii/S0950705125010330)
[L07] already combines band clustering, local features and global Swin features on
six/seven-variety datasets. [RiceSeedNet](https://www.sciencedirect.com/science/article/pii/S2666154324000991)
[L08] already uses a ViT for 13 RGB seed varieties and evaluates another eight-class
grain dataset. Neither provides evidence about this90-variety session problem.
The naming collision also argues against presenting another “RiceSeedNet” as unique.

Subregional modeling has direct agricultural precedent: [Zhou 2021 maize](https://scijournals.onlinelibrary.wiley.com/doi/abs/10.1002/jsfa.11095)
[L34] aggregates CNN decisions over seed subregions. [Zhu 2019 soybean](https://mdpi-res.com/d_attachment/sensors/sensors-19-04065/article_deploy/sensors-19-04065-v2.pdf)
[L33] studies pixel spectra. These motivate a pixel-distribution/regional control,
not a first-ever claim. All pixels/views from one seed must stay in one partition.

[Tyagi 2025 spectral band attention](https://link.springer.com/article/10.1007/s10921-025-01215-8)
[L32] is particularly relevant prior art for band selection plus RGB:96 **wheat**
varieties, 900–1700 nm, 25 selected bands and95.75% accuracy. It is not a rice result,
and its acquisition protocol needs full-text verification. This strengthens the
case that simply combining attention, selected spectra and an RGB ensemble is weak
novelty. Our uniform-band controls and measured acquisition failure are essential.

## 3. Spectral-spatial learning: what transfers conceptually

[HybridSN](https://arxiv.org/abs/1902.06701) [L35] establishes the 3D→2D hybrid family
already reflected in SeedNet. [SpectralFormer](https://arxiv.org/abs/2107.02988) [L09]
models grouped spectra and cross-layer information. They are worthwhile baselines,
but remote-sensing pixel classification differs from recognizing an isolated seed:
spatial context, label granularity, scene independence and effective sample count
all change. Random pixels from one scene should not be advertised as equivalent
evidence to unseen acquisitions here.

[MambaHSI](https://arxiv.org/abs/2501.04944) and [SS-Mamba](https://arxiv.org/abs/2404.18401)
[L10/L10b] target efficient spatial/spectral dependency modeling. MambaHSI's journal
record is2024 despite a2025 arXiv submission. Linear sequence complexity matters
when sequences or full scenes are large. With32–215 bands and a small object, a
compact convolution or small attention model may already be inexpensive. The
algorithmic cost argument alone is insufficient to allocate a broad Mamba sweep.

The important local architectural issue is the **physical axis**, not the brand of
sequence model: spectral indices bridge a102.7-nm unmeasured gap, and changing band
count changes both sample locations and stem strides. Wavelength tokens, separate
interval-local filters and explicit distance-aware global mixing address an actual
input property. Their performance benefit remains a testable hypothesis.

[Deep Sets](https://arxiv.org/abs/1703.06114), [attention MIL](https://proceedings.mlr.press/v80/ilse18a.html)
and [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) [L15–L17]
justify symmetric aggregation of variable-size pixel/region sets. They do not justify
discarding arrangement if shape is discriminative. Hence the proposed nested mean/
quantile→pixel-bag→spatial-region experiment. Attention visualizations should be
validated by perturbation and stability; they are not biochemical explanations.

## 4. Foundation models: the best modern opportunity is transfer, initially frozen

[HyperSL](https://github.com/kkweil/HyperSL) [L13] explicitly provides wavelength-aware
spectral tokens and heterogeneous-input spectral pretraining. It is the most direct
low-complexity HSI probe to attempt. [HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA)
[L11] offers a much larger spectral/spatial alternative. [HyperFree](https://arxiv.org/abs/2503.21841)
[L12] supports varying channels using a wavelength dictionary, but its prompt/segmentation
setting and400 nm lower range differ from the complete local axis.

A2026 [remote-to-proximal transfer study](https://arxiv.org/abs/2604.26478) [L14T]
provides empirical reason to test such transfer rather than dismiss remote pretraining.
It does not validate rice or acquisition invariance. [HyperFM](https://arxiv.org/abs/2604.21127)
[L14], in CVPR2026 **Findings**, concerns PACE/cloud properties; its atmospheric
pretraining has a much larger physical mismatch and lower priority here. Frontier
recency alone should not override relevance.

For RGB, [DINOv3](https://arxiv.org/abs/2508.10104) [L18] is a strong frozen-feature
candidate and [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)
[L19] a useful pretrained convolutional control. Their benefit could be shape,
texture, color or background; the proposed mask/resolution/grayscale controls
separate those explanations. Check pretraining overlap, checkpoint and license
before claiming an inductive result. A frozen-probe failure should trigger crop/
domain diagnostics before a single bounded tuning trial.

[MAE](https://arxiv.org/abs/2111.06377) [L20] motivates train-only masked learning,
but the local data has many correlated pixels and few acquisitions. Reconstruction
can learn the nuisance we want to remove. Pretraining on all test scans, even without
labels, changes the experiment to a transductive setting.

The latest verified tabular report is [TabPFN-3.5](https://arxiv.org/abs/2609.17895)
[L30], revised September 22, 2026. It includes grouped/non-iid benchmark claims and
supersedes an automatic choice of TabPFN-3 in the backlog. Treat it as an optional
descriptor baseline only after checking90-class capability, resource limits and
availability. An abstract benchmark claim is neither a guarantee nor a CPU runtime
estimate for this task.

## 5. Robustness and multimodality: assumptions matter more than modules

[DomainBed](https://arxiv.org/abs/2007.01434) [L23] makes model selection part of the
DG comparison and supports strong ERM controls. [DANN](https://jmlr.org/papers/v17/15-239.html)
[L24] uses domain alignment, [Group DRO](https://arxiv.org/abs/1911.08731) [L24g]
optimizes observed group risk, and [DFR](https://arxiv.org/abs/2204.02937) [L24d]
uses balanced last-layer retraining. None generates missing class-session combinations.
Current grouped calibration is not a balanced cross-session development set.

[Accuracy–invariance theory](https://jmlr.org/papers/v23/21-1078.html) [L24t]
reinforces a distribution-dependent tradeoff. S19's own support argument is more
specific: S=g(Y) within training, so perfect class information retains session
information. It would be wrong to infer that every empirical adversarial penalty is
impossible; the problem is unidentified nuisance removal without extra assumptions.

[MixStyle](https://arxiv.org/abs/2104.02008) [L25] randomizes feature statistics;
its local masked variant already failed S14. [SPDDA](https://ojs.aaai.org/index.php/AAAI/article/download/37296/41258)
[L26], verified2026 primary work, offers spectral/spatial-property augmentation.
Its motivation is relevant only where perturbations fit the sensor's actual valid
intervals and measured nuisance range. Standards-derived corrections or consistency
are more defensible than assuming any smooth transformation preserves variety.
[Passos 2026](https://www.sciencedirect.com/science/article/pii/S0165993626003729)
[L27] is a design review, useful context but not independent new experimental proof.

[Wang–Tran–Feiszli](https://arxiv.org/abs/1905.12681) [L21] and
[OGM](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html)
[L22] explain why multimodal optimization can suppress a branch. Diagnose that
failure with unimodal heads and simple fusion before adding balancing. S14 also
shows a local complication: in-session calibration can reward the wrong modality
for session transfer. A sophisticated fusion gate does not fix its supervision.

## 6. Priority matrix

| Mechanism | Local failure addressed | Priority | Required counterexample/control |
|---|---|---|---|
| Frozen RGB + HSI fusion | Missing fine structure; complementary measurements | Highest | RGB-only, HSI-only, simple fusion; masked/downsample controls |
| Wavelength-aware full-spectrum probe | Discarded bands, nonuniform gap, stride confound | Highest | Fixed-capacity index model, nested axes, 195 vs215 |
| Pixel/region distribution encoder | Mean-spectrum loss; spatial pathway carries both signal and nuisance | High | Quantile LDA, pixel bag, geometry-aware grid |
| Standards-supported nuisance consistency | Residual acquisition variation | Conditional high | Measured nuisance family; generic jitter and no-correction controls |
| Compact spectral foundation transfer | Limited acquisition diversity | High probe, conditional tuning | Same shallow head on classical summaries and frozen model |
| Larger transformer/Mamba/KAN/hypergraph | No unique local failure yet | Low | A specific failure or compute constraint not solved by simpler models |
| End-to-end gated fusion | Possible conditional modality complementarity | Later | Better than independent branches and shallow fusion on new sessions |
| Generic domain adversarial loss | Desired domain independence but missing support | Low on current design | Crossed training support or standards/external prior that identifies nuisance |

The strongest prospective contribution is a **measured, acquisition-robust use of
complementary geometry and spectral information**. Existing work rules out novelty
claims based solely on RGB+HSI, attention, band grouping, regional voting, hybrid
3D/2D CNNs or transformers. The proposed architecture is therefore a research
hypothesis, with a novelty claim conditional on its mechanism and validation.
