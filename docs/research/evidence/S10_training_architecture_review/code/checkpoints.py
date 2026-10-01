"""S10 checkpoint forensics: what the 12 trained SpectralSeedNets actually compute.

CPU only, forward passes only, **train and calib rows only** (s10common.split_rows never
returns held-out rows). For every run of the S08 sweep it loads ``best_stage1.pth`` —
the checkpoint the run selected on calib — rebuilds the run's exact initialisation
(``set_seed(seed)`` then ``build_model`` is the pipeline's order; the reconstruction is
checked against the never-trained dead taps of the last ResBlock, which must equal it up
to weight decay) and writes:

  ckpt_fit.csv           clean fit (eval mode, no augmentation, no mixup, margin 0) on train
                         and calib, live and EMA weights: accuracy, CE, and the angular
                         geometry the ArcFace margin acts on (θ_y, cosine gap, share of
                         training kernels that would satisfy m = 0.30)
  ckpt_probes.csv        linear probes (shrinkage LDA, fit on train, scored on calib) on each
                         pathway's output, the fused embedding, the spectral descriptor and the
                         raw 40 scalars; eval-time knock-outs (spatial, spectral, morphometrics)
  ckpt_gain.csv          response of logits and of each pathway to a global reflectance gain
                         a ∈ {0.8, 1.25} on 128 calib kernels
  ckpt_index_bank.csv    how far the 64 learned normalised-difference indices moved from
                         uniform band weights; their spread across kernels
  ckpt_spectral_scale.csv  magnitude of each block of the spectral descriptor entering the
                         per-sample LayerNorm
  ckpt_weight_change.csv ‖W − W₀‖ / ‖W₀‖ per module (live weights vs reconstructed init)
  ckpt_dead_taps.json    the last ResBlock's never-trained taps: count, gradient, drift

Runtime ≈ 30 min on 4 CPU threads.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from s10common import DS32, CACHE, run_config, run_dirs, save, split_rows
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score

from spectralquadnet.data.morphometrics import standardise_morphometrics
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet
from spectralquadnet.models.stats_ops import foreground_mask, masked_mean_spectrum

torch.set_grad_enabled(False)
BATCH = 128
MARGIN = 0.30
GAINS = (0.8, 1.25)

PATCHES = np.load(DS32 / "patches.npy", mmap_mode="r")
MASKS = np.load(DS32 / "masks.npy", mmap_mode="r")
MORPH_RAW = np.load(DS32 / "morphology.npy").astype(np.float32)
LABELS = np.load(DS32 / "labels.npy").astype(np.int64)
_wl = pd.read_csv(DS32 / "wavelengths.csv").iloc[:, -1].to_numpy(np.float32)
WL = torch.from_numpy((_wl - _wl.min()) / (_wl.max() - _wl.min()))
C = len(_wl)
# Spectral descriptor layout at C bands: [index bank 64 | continuum 16 | snv C | D1 C | D2 C | morph 8]
BLOCKS = {"index_bank": (0, 64), "continuum": (64, 80), "snv": (80, 80 + C),
          "d1": (80 + C, 80 + 2 * C), "d2": (80 + 2 * C, 80 + 3 * C), "morph": (80 + 3 * C, 88 + 3 * C)}


def build(cfg, seed: int) -> SpectralSeedNet:
    torch.manual_seed(seed)  # the pipeline: set_seed(cfg.seed) -> DataStore -> build_model
    return SpectralSeedNet.from_config(cfg, WL).eval()


def batch(rows: np.ndarray, morph: np.ndarray):
    rows = np.sort(rows)
    x = torch.from_numpy(np.asarray(PATCHES[rows], dtype=np.float32))
    m = torch.from_numpy(np.asarray(MASKS[rows], dtype=np.float32))
    return x, m, torch.from_numpy(morph[rows]), torch.from_numpy(LABELS[rows])


def parts(model: SpectralSeedNet, x, mask, morph, keep=(1.0, 1.0)) -> dict[str, torch.Tensor]:
    """SpectralSeedNet.forward, unrolled so every intermediate is visible."""
    m = foreground_mask(x, mask)
    xs = model.se(x, m)
    b_sp = model.spatial(xs, m) * keep[0]
    feat = model.spectral.features(masked_mean_spectrum(xs, m), morph)
    b_spec = model.spectral.mlp(model.spectral.in_norm(feat)) * keep[1]
    emb = F.normalize(model.embed_net(model.fuse(torch.cat([b_sp, b_spec], 1))), dim=1)
    head = model.arcface_head
    cos = head.pool_subcentres(head.subcentre_cosines(emb))
    return {"cos": cos, "aux": model.aux_head_spatial(b_sp), "b_spatial": b_sp, "b_spectral": b_spec,
            "emb": emb, "feat": feat, "r0": masked_mean_spectrum(x, m)}


def collect(model, rows, morph, keep=(1.0, 1.0), want=("cos", "aux")) -> dict[str, np.ndarray]:
    out: dict[str, list] = {k: [] for k in (*want, "y")}
    rows = np.sort(rows)
    for i in range(0, len(rows), BATCH):
        x, m, mo, y = batch(rows[i:i + BATCH], morph)
        p = parts(model, x, m, mo, keep)
        for k in want:
            out[k].append(p[k].numpy())
        out["y"].append(y.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def fit_stats(cos: np.ndarray, aux: np.ndarray, y: np.ndarray, s: float) -> dict[str, float]:
    n = len(y)
    cy = cos[np.arange(n), y]
    other = cos.copy()
    other[np.arange(n), y] = -np.inf
    cmax = other.max(1)
    theta = np.arccos(np.clip(cy, -1, 1))
    # what the shipped margin asks for: cos(θ_y + m) > max_{j≠y} cos θ_j (with the π/2 cap)
    m_eff = np.minimum(MARGIN, np.maximum(math.pi / 2 - theta, 0))
    logits = torch.from_numpy(cos * s)
    yt = torch.from_numpy(y)
    pen = cos.copy()
    pen[np.arange(n), y] = np.cos(theta + m_eff)
    pen_t = torch.from_numpy(pen * s)
    return {
        "acc": float((cos.argmax(1) == y).mean()),
        "macro_f1": float(f1_score(y, cos.argmax(1), average="macro")),
        "ce": float(F.cross_entropy(logits, yt)),
        "ce_ls04_m30": float(F.cross_entropy(pen_t, yt, label_smoothing=0.04)),
        "ce_ls04_m0": float(F.cross_entropy(logits, yt, label_smoothing=0.04)),
        "aux_acc": float((aux.argmax(1) == y).mean()),
        "aux_ce": float(F.cross_entropy(torch.from_numpy(aux), yt)),
        "theta_y_deg_median": float(np.degrees(np.median(theta))),
        "cos_y_median": float(np.median(cy)),
        "cos_gap_median": float(np.median(cy - cmax)),
        "margin_satisfied": float((np.cos(theta + m_eff) > cmax).mean()),
        "margin_capped_frac": float((m_eff < MARGIN).mean()),
    }


def lda_probe(Xtr, ytr, Xca, yca) -> tuple[float, float]:
    est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    p = est.predict(Xca)
    return float((p == yca).mean()), float(f1_score(yca, p, average="macro"))


def main() -> None:
    fit_rows, probe_rows, gain_rows, idx_rows, scale_rows, wc_rows, dead = [], [], [], [], [], [], {}
    for arm, fold, seed, d in run_dirs():
        t0 = time.time()
        cfg = run_config(d)
        s = float(cfg.single.arcface_s)
        tr, ca = split_rows(arm, fold)
        morph, _ = standardise_morphometrics(MORPH_RAW, tr)  # train-only stats, as the loader does
        ck = torch.load(d / "best_stage1.pth", map_location="cpu", weights_only=False)
        init = build(cfg, seed)
        live = build(cfg, seed)
        live.load_state_dict(ck["model"])
        ema = build(cfg, seed)
        ema.load_state_dict(ck["ema"])
        tag = dict(arm=arm, fold=fold, seed=seed, best_epoch=int(ck["epoch"]), best_source=ck["best_source"])

        # ── the dead taps of the last ResBlock (2×2 input, 3×3 kernel, stride 2) ──
        w, w0 = live.spatial.stages[6].c2.weight, init.spatial.stages[6].c2.weight
        dead_mask = torch.ones(3, 3, dtype=torch.bool)
        dead_mask[1:, 1:] = False
        ratio = (w[:, :, dead_mask] / w0[:, :, dead_mask]).median().item()
        dead[f"{arm}_f{fold}_s{seed}"] = {
            "dead_params": int(dead_mask.sum()) * w.shape[0] * w.shape[1],
            "dead_over_init_median": ratio,
            "dead_max_abs_dev_from_init_x_ratio": float((w[:, :, dead_mask] - ratio * w0[:, :, dead_mask]).abs().max()),
            "live_rel_change": float((w[:, :, ~dead_mask] - w0[:, :, ~dead_mask]).norm() / w0[:, :, ~dead_mask].norm()),
        }

        # ── per-module weight change, live vs init ──
        sd0 = init.state_dict()
        agg: dict[str, list[float]] = {}
        for name, p in live.named_parameters():
            mod = ".".join(name.split(".")[:2]) if name.startswith(("spatial", "spectral")) else name.split(".")[0]
            a = agg.setdefault(mod, [0.0, 0.0, 0])
            a[0] += float((p - sd0[name]).pow(2).sum())
            a[1] += float(sd0[name].pow(2).sum())
            a[2] += p.numel()
        for mod, (dn, n0, n) in agg.items():
            wc_rows.append({**tag, "module": mod, "params": n, "rel_change": math.sqrt(dn / max(n0, 1e-30))})

        # ── clean fit, live and EMA, train and calib ──
        cache = {}
        for wname, model in (("live", live), ("ema", ema)):
            for split, rows in (("train", tr), ("calib", ca)):
                want = ("cos", "aux", "b_spatial", "b_spectral", "emb", "feat", "r0") if wname == tag["best_source"] else ("cos", "aux")
                out = collect(model, rows, morph, want=want)
                if wname == tag["best_source"]:
                    cache[split] = out
                fit_rows.append({**tag, "weights": wname, "split": split, "n": len(rows),
                                 **fit_stats(out["cos"], out["aux"], out["y"], s)})

        # ── linear probes on the selected weights' representations (train → calib) ──
        best = live if tag["best_source"] == "live" else ema
        trc, cac = cache["train"], cache["calib"]
        morph_tr = trc["feat"][:, BLOCKS["morph"][0]:BLOCKS["morph"][1]]
        morph_ca = cac["feat"][:, BLOCKS["morph"][0]:BLOCKS["morph"][1]]
        reps = {
            "network (argmax cos)": None,
            "fused embedding": (trc["emb"], cac["emb"]),
            "spatial pathway output": (trc["b_spatial"], cac["b_spatial"]),
            "spectral pathway output": (trc["b_spectral"], cac["b_spectral"]),
            "spectral descriptor (MLP input)": (trc["feat"], cac["feat"]),
            "spectral descriptor without morph": (trc["feat"][:, :BLOCKS["morph"][0]], cac["feat"][:, :BLOCKS["morph"][0]]),
            "raw mean spectrum + morph (40 scalars)": (np.hstack([trc["r0"], morph_tr]), np.hstack([cac["r0"], morph_ca])),
            "raw mean spectrum (32)": (trc["r0"], cac["r0"]),
            "SNV mean spectrum (32)": (trc["feat"][:, BLOCKS["snv"][0]:BLOCKS["snv"][1]], cac["feat"][:, BLOCKS["snv"][0]:BLOCKS["snv"][1]]),
        }
        for name, xy in reps.items():
            if xy is None:
                p = cac["cos"].argmax(1)
                acc, f1 = float((p == cac["y"]).mean()), float(f1_score(cac["y"], p, average="macro"))
            else:
                acc, f1 = lda_probe(xy[0], trc["y"], xy[1], cac["y"])
            probe_rows.append({**tag, "probe": "linear (LDA train→calib)", "representation": name, "calib_acc": acc, "calib_macro_f1": f1})
        for name, keep, zero_morph in (("knock out spatial", (0.0, 1.0), False), ("knock out spectral", (1.0, 0.0), False),
                                       ("morphometrics set to train mean", (1.0, 1.0), True)):
            mo = np.zeros_like(morph) if zero_morph else morph
            out = collect(best, ca, mo, keep=keep, want=("cos",))
            p = out["cos"].argmax(1)
            probe_rows.append({**tag, "probe": "eval-time knock-out", "representation": name,
                               "calib_acc": float((p == out["y"]).mean()), "calib_macro_f1": float(f1_score(out["y"], p, average="macro"))})

        # ── global-gain response on 128 calib kernels ──
        rows = np.sort(ca)[:: max(1, len(ca) // 128)][:128]
        x, m, mo, y = batch(rows, morph)
        ref = parts(best, x, m, mo)
        ref_nm = parts(best, x, m, torch.zeros_like(mo))
        for a in GAINS:
            for label, mo_used, base in (("with morph", mo, ref), ("morph zeroed", torch.zeros_like(mo), ref_nm)):
                g = parts(best, x * a, m, mo_used)
                rel = lambda k: float((g[k] - base[k]).norm() / base[k].norm())  # noqa: E731
                gain_rows.append({**tag, "gain": a, "morph": label,
                                  "rel_change_spatial": rel("b_spatial"), "rel_change_spectral": rel("b_spectral"),
                                  "rel_change_spectral_descriptor_wo_morph": float((g["feat"][:, :BLOCKS["morph"][0]] - base["feat"][:, :BLOCKS["morph"][0]]).norm() / base["feat"][:, :BLOCKS["morph"][0]].norm()),
                                  "rel_change_logits": rel("cos"),
                                  "argmax_agreement": float((g["cos"].argmax(1) == base["cos"].argmax(1)).float().mean()),
                                  "rel_change_raw_mean_spectrum": float((g["r0"] - base["r0"]).norm() / base["r0"].norm())})

        # ── the learned index bank ──
        pos, neg = (t.detach() for t in best.spectral.index_bank.selectors())
        ent = lambda p: -(p * p.clamp_min(1e-12).log()).sum(1)  # noqa: E731
        z = cac["feat"][:, BLOCKS["index_bank"][0]:BLOCKS["index_bank"][1]]
        idx_rows.append({**tag,
                         "entropy_pos_over_lnC_median": float((ent(pos) / math.log(C)).median()),
                         "entropy_neg_over_lnC_median": float((ent(neg) / math.log(C)).median()),
                         "effective_bands_pos_median": float(ent(pos).exp().median()),
                         "max_weight_pos_median": float(pos.max(1).values.median()),
                         "max_weight_any": float(torch.cat([pos, neg]).max()),
                         "cos_pos_neg_median": float(F.cosine_similarity(pos, neg, dim=1).median()),
                         "index_value_abs_median": float(np.median(np.abs(z))),
                         "index_sd_across_kernels_median": float(np.median(z.std(0))),
                         "uniform_weight": 1.0 / C})

        # ── scale of each descriptor block entering the per-sample LayerNorm ──
        f = cac["feat"]
        ss_total = (f - f.mean(1, keepdims=True)) ** 2
        for blk, (a0, a1) in BLOCKS.items():
            scale_rows.append({**tag, "block": blk, "width": a1 - a0,
                               "abs_mean": float(np.abs(f[:, a0:a1]).mean()),
                               "sd_across_kernels_mean": float(f[:, a0:a1].std(0).mean()),
                               "share_of_layernorm_variance": float(ss_total[:, a0:a1].sum(1).mean() / ss_total.sum(1).mean())})
        print(f"{arm} f{fold} s{seed}: {time.time() - t0:.0f}s")

    save(pd.DataFrame(fit_rows), "ckpt_fit.csv")
    save(pd.DataFrame(probe_rows), "ckpt_probes.csv")
    save(pd.DataFrame(gain_rows), "ckpt_gain.csv")
    save(pd.DataFrame(idx_rows), "ckpt_index_bank.csv")
    save(pd.DataFrame(scale_rows), "ckpt_spectral_scale.csv")
    save(pd.DataFrame(wc_rows), "ckpt_weight_change.csv")
    save(dead, "ckpt_dead_taps.json")
    (CACHE / "checkpoints.done").write_text(time.ctime())


if __name__ == "__main__":
    main()
