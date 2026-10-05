# Confirmation allocation after development screening

2026-10-05. This queue is a conditional resource policy, not authorization for an
automatic three-seed expansion of S22 or every new candidate.

| Candidate or comparison | Development allocation | Eventual confirmation and reason |
|---|---|---|
| Corrected-fold v5 and fixed equal RGB fusion | S22: seed 0, both folds; two shared HSI fits | Additional encoder seeds 1/2 on both folds when these are controls for a selected final system, or the fixed fusion itself becomes the final system |
| S23/S25 additive frozen-feature head | S23 headseed0 both encoders; S25 headseeds1/2 on encoder0, selective cheap sensitivity complete | If selected after mechanism screening, match encoder and head seeds 0/1/2 across both folds; six head fits on six encoders, rather than three heads on encoder 0 |
| Frozen DINO RGB + deterministic probe | Existing S21 fit per fold | Reuse the same exported probes; three identical refits do not measure initialization uncertainty |
| A trainable RGB control or an adapted multimodal encoder | One seed, both folds first | Independent training seeds for the final relevant control/system; neither is represented by deterministic DINO replication |
| Learned-head modality and mechanism ablations | Bounded seed-0 screens when a positive learned result makes them informative | Confirm the selected ablations that support a paper contribution or a meaningful borderline decision; reject weak intermediates without automatic replication |
| New crossed sessions and lots | Acquisition protocol must be frozen separately | Independent acquisitions and locked external evaluation remain necessary for broad transfer claims; training seeds cannot substitute for them |

For a final frozen-encoder architecture, v5, fixed fusion and learned-head controls
can share the same encoder fits. Seeds 1/2 would require **four additional HSI fits**
across both folds, not separate fits for every probability-fusion arm. Heads and
probability controls are cheap. If the final architecture adapts the encoders,
its independently trained multimodal cells need their own matched confirmation.

S24 already removed each learned correction branch while retaining the multimodal
anchor. No smaller model qualifies for adoption. The strict H42 necessity gate
fails; full-head superiority over HSI-only is .004767, below the .005 threshold.
Controls change capacity, so they do not establish an interaction. S25 then selectively
confirmed head-initialization stability only, at9.76s for four fits; encoder uncertainty
is still unmeasured. Do not expand the rejected smaller branches automatically.

S26's fixed TTA-anchor swap fails calibration without new test scores. S27 is proposed:
train the same head against TTA from the outset, initially one head seed both folds,
after profiling frozen training-row inference and sealing a plan. Only a clear
practical gain should earn further selective head sensitivity or expensive encoder
replication. If a paper claims both feature branches contribute, the retained joint
head and relevant HSI-only learned correction need matched multi-seed confirmation
and a capacity-controlled interpretation. No kernel-specific cross-attention claim
is supported by current pairing controls.

All current screens reuse existing acquisitions. Paired variety intervals describe
heterogeneity of saved class contributions; they do not estimate seed variance,
fresh-session uncertainty, or a locked final-test effect.

## Update after S27–S29 (2026-10-05)

The selected candidate is now the TTA-trained head on frozen DINOv2 **ViT-L/14** (S29, D51).
S28 already covered head-seed sensitivity (SD .0009), so no further head-only replication is needed.
The remaining confirmation is exactly [S30](../S30_final_confirmation/README.md):
- four HSI fits (encoder seeds 1/2 × both folds), exporting embeddings and train-row TTA;
- one head per encoder at head seed = encoder seed, for ViT-L and ViT-S;
- shared controls: HSI TTA, equal ViT-S, equal ViT-L.

Its C2 contrast, head ViT-L − equal ViT-L, decides whether the final system keeps a learned
component. The one-seed margin is only +.0067. ViT-B and the S23 single-anchor head are not
replicated.
