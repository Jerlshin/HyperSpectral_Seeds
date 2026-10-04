# S19 search and verification log

Date/cutoff: 2026-10-03. Live web search, primary publisher/proceedings pages,
author preprints, university manuscript deposits and official code repositories.
This is a reconstructed thematic query log, not a claim of exhaustive database
export or PRISMA screening. No complete copyrighted papers are republished here.
The ledger contains39 retained primary-source records; access may be abstract-only,
metadata-only or indexed publisher excerpt, explicitly recorded per row.

| Search family / representative query strings | Verification path and outcome |
|---|---|
| `90 rice varieties hyperspectral RGB Fabiyi Taheri`; `RVHID90 classification 2025 2026` | Zenodo, Strathprints full manuscript, Springer, Frontiers; exact task sources L01–L06/L31 |
| `FusedNet Model for Varietal Classification of Rice Seeds` | Springer book TOC→chapter link; DOI suffix_3 confirmed, correct author/venue/pages recorded; full methods paywalled |
| `rice varieties hyperspectral band clustering ClusterRiceNet`; `RiceSeedNet 101062` | Publisher abstracts for six/seven-class HSI and13-class RGB; not pooled into90-class leaderboard |
| `Identification of Soybean Varieties hyperspectral 4065`; `Identification of the variety of maize seeds subregional voting` | Publisher-indexed PDF excerpt / Wiley primary abstract; regional/pixel precedents retained with protocol limits |
| `Spectral Band Attention Networks hyperspectral RGB 2025` | Publisher DOI10.1007/s10921-025-01215-8;96 wheat, not rice; source code linked but not run |
| `SpectralFormer hyperspectral`; `MambaHSI`; `SS-Mamba` | arXiv and journal identifiers; MambaHSI online journal year2024 differs from arXiv2025 |
| `HyperSIGMA HyperFree HyperSL foundation model`; `hyperspectral foundation models 2026 proximal transfer` | Official repositories, arXiv/CVF; wavelength interfaces and domain mismatches recorded, checkpoints not downloaded |
| `HyperFM spectral grouping Tushar` | arXiv and CVPR Findings paper; cloud retrieval scope makes lower priority |
| `DINOv3`; `ConvNeXt A ConvNet for the 2020s`; `Masked Autoencoders` | Author report/CVF primary papers; latest RGB transfer candidate plus established control |
| `TabPFN-3`; `TabPFN-3.5` | Latest verified report is2609.17895v2,2026-09-22;90-class/runtime/license not checked by running code |
| `Deep Sets`; `Attention-based Deep Multiple Instance Learning`; `Set Transformer` | arXiv/PMLR theorem/method abstracts; symmetric aggregation basis, not a rice accuracy proof |
| `What Makes Training Multi-Modal Classification Networks Hard`; `On-the-fly Gradient Modulation` | Author report/CVF; mechanism only conditional on measured branch suppression |
| `In Search of Lost Domain Generalization`; `DANN`; `Group DRO`; `Last Layer Re-Training is Sufficient` | Primary papers; exact DFR ID2204.02937 verified (an initially guessed different ID was unrelated and excluded) |
| `Fundamental Limits and Tradeoffs in Invariant Representation Learning`; `Lost Domain Generalization lack training domains` | JMLR theory retained; avoid transferring generic bounds without their assumptions |
| `MixStyle`; `Spectral Property-Driven Data Augmentation`; `Vis-NIR conditional design Passos` | ICLR/AAAI/arXiv/publisher identities verified; local perturbation validity emphasized |
| `On Over-fitting in Model Selection and Subsequent Selection Bias` | Primary JMLR paper retained to motivate a new locked acquisition test |

## Additional leads and exclusions

- 2025 six-class multi-dimensional CNN rice fusion paper:
  DOI10.1016/j.jfca.2025.108389, publisher
  https://www.sciencedirect.com/science/article/abs/pii/S0889157525012050 .
  Abstract found; full protocol not verified. Not needed for the central recommendation.
- 2025 hybrid indica CNN-Transformer paper:
  https://www.mdpi.com/2079-6374/15/10/647 , DOI10.3390/bios15100647.
  Publisher-indexed methods mention13 related varieties grown under common management;
  direct fetch repeatedly429. Keep as a follow-up comparator lead, not an evidence
  source for90-class transfer.
- 2026/2027 DMR-HGNN rice hypergraph lead: DOI10.1016/j.ins.2026.124077, assigned
  January2027 issue. First-online date before the cutoff was not verified; excluded
  from the current-evidence ledger. No claim of its absence from the field.
- KAN rice-germplasm viability lead DOI10.1016/j.compag.2025.111034 concerns viability,
  not variety recognition; not a reason to add a KAN here.
- SpecAware and very recent HyperSAM leads were not verified in sufficient depth;
  watchlist only. The selected mechanisms already cover the decisions required.
- EPO calibration-transfer classic DOI10.1016/S0169-7439(03)00051-0 was located through
  secondary references/software docs, but original primary article fetch failed.
  No unverified numerical result or theorem from it is used. Standards-grounded
  nuisance proposal is stated as a hypothesis with an explicit measurement model.

## Verification boundaries

The Fabiyi full paper and Hu full publisher text were inspected for evaluation;
most frontier papers were screened from verified primary abstracts/metadata and
relevant official repositories. This is sufficient for mechanism prioritization,
not faithful implementation or a final systematic review. Before reproducing any
method, read its full paper/code, inspect its data split and freeze adaptation choices.
No paywall was bypassed; inaccessible full texts remain marked. JCR/SJR category-year
rankings were not verified; no journal is asserted currently Q1. TGRS official scope
was checked at https://www.grss-ieee.org/publications/transactions-on-geoscience-remote-sensing/ ;
CEA/Information Fusion publisher scope pages returned403. Venue recommendations
are conditional fit judgments; confirm editorial scope and ranking before submission.

All headlines are interpreted as authors' reported results, not independent
replications. Metric differences, class counts, label support and acquisition unit
are retained. Absence of a grouped protocol in accessible text is not proof of
leakage. Bibliographic identity is verified separately from experimental validity.
