# `configs/data/ablation/` — reduced-band arms

**Nothing here is on the primary path.** The primary data configuration is
`configs/data/hsi256_grouped.yaml`: the complete 256-band cube, no selection.
These files exist so that band reduction stays a *measurable* research question
rather than an inherited assumption.

| Config | Bands | Split | What it is for |
|---|---:|---|---|
| `spa40_grouped.yaml` | 40 | `grouped` | A2's reduced arm — the shipped SPA subset under the primary protocol, so `full 256 − SPA 40` is a clean one-variable delta. |
| `spa40_stratified.yaml` | 40 | `stratified` | The reduced arm's leaky twin. Only needed if A1 is re-run at k = 40. |
| `spa40_audited.yaml` | 40 | `stratified` | Frozen. Reproduces the *audited run's* input and partition exactly, and is what `configs/experiment/quadnet_audited.yaml` and the golden regression gates compose. Do not tidy it. |

Selecting one is explicit at the command line, which is the point:

```bash
python train.py data=ablation/spa40_grouped data.num_bands=40
```

Note that `data.num_bands` is carried by each file — the override above is
redundant and shown only to make the contract visible: the cube, the wavelength
CSV and `num_bands` must agree, and
`spectralquadnet.data.band_geometry` fails the run at startup if they do not.

## The other reduction mechanism

A k-band arm does **not** need a reduced copy of the 36 GB cube. Point
`data.band_indices_path` at a `.npy` of band indices and the dataset slices each
patch as it comes off the mmap:

```bash
python train.py \
  data.band_indices_path=outputs/band_study/select/fold0/mrmr_k64_bands.npy \
  data.wavelength_path=outputs/band_study/select/fold0/mrmr_k64_wavelengths.csv \
  data.num_bands=64
```

That is what `python -m spectralquadnet.bandstudy.cli neural` generates, and it
is the mechanism the band-selection pathway is expected to use.
See [`docs/07_BAND_SELECTION_PATHWAY.md`](../../../docs/07_BAND_SELECTION_PATHWAY.md).
