# S36 prior-art check for the proposed mechanisms (2026-10-05)

Targeted web searches run before any novelty statement. They extend S19's verified review
(`evidence/S19_next_generation_strategy/literature.csv`, prior_art.md §4–6), whose conclusions
still apply: generic gated 2-D RGB / 3-D HSI fusion for 90 rice varieties exists (L06); domain
adversarial / invariance losses need crossed support (L23–L24t); nuisance augmentation is
defensible only within a measured nuisance range (L25–L26).

| Query (abridged) | What came back | Bearing |
|---|---|---|
| test-time degradation matching, class-conditional rendering, acquisition confounding | test-time adaptation (entropy matching, NeurIPS 2024, arXiv 2408.07511); visual conditioning tokens (2406.19341); class-invariant TTA (2509.14420); HyperTTA for HSI (2509.08436); test-time style shifting (PMLR v202) | TTA adapts the *model* or the *test input globally*; none scores each class under that class's own training acquisition |
| analysis-by-synthesis / counterfactual rendering per class, acquisition shortcut | counterfactual contrastive learning for acquisition shift (CF-SimCLR, arXiv 2403.09605); counterfactual image generation via causal models | counterfactuals are used to *train* invariant representations; closest prior art to the augmentation half, not to the class-conditional scoring rule |
| test-time blurring to match the training distribution | domain-adaptive video deblurring via test-time blurring (arXiv 2407.09059); degradation-aware super-resolution | blur used to adapt restoration networks; not classification under a class-confounded acquisition |
| DINOv2 last-four-layer readout vs class token | DINOv2 paper (arXiv 2304.07193): the linear evaluation concatenates class + pooled patch tokens over the last four layers | S31's readout follows a published recipe; the new evidence is its size here and that it does not transfer |
| rice/seed RGB + HSI fusion 2025–2026 | FusedNet (90 rice varieties, RGB + HSI, SVM + ResNet-50); sorghum/maize image–spectral fusion (PMC12412220; J. Stored Prod. Res. 2026) | RGB+HSI fusion on seeds, and on this 90-variety dataset, is established; acquisition-aware scoring is not reported in these results |

**Conclusion (bounded):** no direct precedent was found for *class-conditional acquisition
rendering* as a scoring rule against session-confounded classes. This is a targeted search, not
an exhaustive review. The claim must be stated as "to our knowledge" and re-verified before
submission. The other components (foreground tokens, multi-layer readout, late fusion,
blur augmentation) are known individually.
