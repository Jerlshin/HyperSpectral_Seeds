"""S10 — the spectral descriptor and the reflectance-level channel, isolated (CPU, train + calib only).

Two questions the end-to-end probes in checkpoints.py cannot separate:

1. **Which part of the spectral descriptor carries the linear information?** The descriptor is
   [index bank 64 | continuum 16 | SNV 32 | D₁ SNV 32 | D₂ SNV 32 | morph 8]. D₁ and D₂ are fixed
   linear maps of SNV, so for a linear model they add no information; the index bank was found ≈
   constant. Shrinkage LDA (train → calib) on each block and on cumulative unions answers this
   without the spatial path (the descriptor is a function of the mean spectrum only).

2. **Where does absolute reflectance level enter?** By construction the stem (bias-free Conv3d →
   GroupNorm, background included) and every spectral feature except morph (SNV, D₁/D₂ of SNV,
   normalised differences, hull-ratio depths) are invariant to x → a·x. The only level-dependent
   operator is MaskedSpectralECA's gate. Applying a gain *after* the gate (to x′) tests the
   pathways with that side channel held fixed; applying it before (to x) is what the model sees.

Writes descriptor_probe.csv and level_channel.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from checkpoints import BLOCKS, MORPH_RAW, PATCHES, MASKS, LABELS, batch, build, lda_probe
from s10common import run_config, run_dirs, save, split_rows

from spectralquadnet.data.morphometrics import standardise_morphometrics
from spectralquadnet.models.stats_ops import foreground_mask, masked_mean_spectrum

torch.set_grad_enabled(False)


def mean_spectra(model, rows: np.ndarray, gated: bool) -> np.ndarray:
    """Foreground mean spectrum, through the trained ECA gate or not — no spatial path needed."""
    out = []
    rows = np.sort(rows)
    for i in range(0, len(rows), 256):
        r = rows[i:i + 256]
        x = torch.from_numpy(np.asarray(PATCHES[r], dtype=np.float32))
        m = foreground_mask(x, torch.from_numpy(np.asarray(MASKS[r], dtype=np.float32)))
        out.append(masked_mean_spectrum(model.se(x, m) if gated else x, m).numpy())
    return np.concatenate(out)


def main() -> None:
    rows_out, lvl = [], []
    for arm, fold, seed, d in run_dirs():
        cfg = run_config(d)
        tr, ca = split_rows(arm, fold)
        morph, _ = standardise_morphometrics(MORPH_RAW, tr)
        ck = torch.load(d / "best_stage1.pth", map_location="cpu", weights_only=False)
        model = build(cfg, seed)
        model.load_state_dict(ck["model" if ck["best_source"] == "live" else "ema"])
        ytr, yca = LABELS[np.sort(tr)], LABELS[np.sort(ca)]
        feats = {}
        for split, rows in (("train", tr), ("calib", ca)):
            r = torch.from_numpy(mean_spectra(model, rows, gated=True))
            feats[split] = model.spectral.features(r, torch.from_numpy(morph[np.sort(rows)])).numpy()
        sets = {
            "SNV": ["snv"], "SNV + D1 + D2": ["snv", "d1", "d2"], "continuum depths": ["continuum"],
            "index bank": ["index_bank"], "SNV + D1 + D2 + continuum": ["snv", "d1", "d2", "continuum"],
            "descriptor without morph": ["index_bank", "continuum", "snv", "d1", "d2"],
            "morph": ["morph"], "SNV + morph": ["snv", "morph"], "full descriptor": list(BLOCKS),
        }
        for name, blocks in sets.items():
            cols = np.concatenate([np.arange(*BLOCKS[b]) for b in blocks])
            acc, f1 = lda_probe(feats["train"][:, cols], ytr, feats["calib"][:, cols], yca)
            rows_out.append(dict(arm=arm, fold=fold, seed=seed, features=name, n_features=len(cols),
                                 calib_acc=acc, calib_macro_f1=f1))
        for name, gated in (("raw mean reflectance (ungated)", False),):
            Xtr, Xca = mean_spectra(model, tr, gated), mean_spectra(model, ca, gated)
            acc, f1 = lda_probe(Xtr, ytr, Xca, yca)
            rows_out.append(dict(arm=arm, fold=fold, seed=seed, features=name, n_features=Xtr.shape[1], calib_acc=acc, calib_macro_f1=f1))
            acc, f1 = lda_probe(np.hstack([Xtr, morph[np.sort(tr)]]), ytr, np.hstack([Xca, morph[np.sort(ca)]]), yca)
            rows_out.append(dict(arm=arm, fold=fold, seed=seed, features="raw mean reflectance + morph", n_features=Xtr.shape[1] + 8, calib_acc=acc, calib_macro_f1=f1))

        # level channel: gain applied before vs after the ECA gate, 128 calib kernels
        sub = np.sort(ca)[:: max(1, len(ca) // 128)][:128]
        x, mk, mo, _ = batch(sub, morph)
        m = foreground_mask(x, mk)
        xg = model.se(x, m)
        base_sp = model.spatial(xg, m)
        base_f = model.spectral.features(masked_mean_spectrum(xg, m), mo)[:, :BLOCKS["morph"][0]]
        for a in (0.8, 1.25):
            for where, xin in (("after the ECA gate (x' -> a x')", a * xg), ("before the ECA gate (x -> a x)", model.se(a * x, m))):
                sp = model.spatial(xin, m)
                f = model.spectral.features(masked_mean_spectrum(xin, m), mo)[:, :BLOCKS["morph"][0]]
                lvl.append(dict(arm=arm, fold=fold, seed=seed, gain=a, applied=where,
                                rel_change_spatial_output=float((sp - base_sp).norm() / base_sp.norm()),
                                rel_change_spectral_descriptor_wo_morph=float((f - base_f).norm() / base_f.norm())))
        print(f"{arm} f{fold} s{seed}", flush=True)
    save(pd.DataFrame(rows_out), "descriptor_probe.csv")
    save(pd.DataFrame(lvl), "level_channel.csv")


if __name__ == "__main__":
    main()
