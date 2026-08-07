#!/usr/bin/env python3
"""Phase 0 · actions 0-D, 0-E, 0-F, 0-G, 0-I, 0-J — the cheap probes.

Each probe corresponds to one row of ``IMPROVEMENT_PLAN.md`` §4.1 and, where
applicable, one falsifiable prediction from Appendix B:

======  ====================================================  ==========
Action  Question                                              Prediction
======  ====================================================  ==========
0-D     Rank of the ``(9, 40)`` masked-statistics tensor      F-2
0-E     Pairwise cosine between the fusion latents            F-4
0-F     Wavelength-spacing spread of the 40 selected bands    (C-5)
0-G     Precision vs recall for the hardest classes           F-5
0-I     Sub-centre win rates — dead ArcFace centres           F-8
0-J     Config keys that never reach the module they name     N-1a…f
======  ====================================================  ==========

Nothing here modifies model code: every probe either reads a checkpoint
tensor, reads a CSV, reads a saved prediction array, or runs the model
forward in eval mode.

Usage::

    python scripts/phase0_probes.py                  # all probes
    python scripts/phase0_probes.py --only D,E,J     # a subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import build_loaders, build_splits
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.engine.checkpoint import load_ckpt, stage_ckpt_path
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.models.stats_ops import masked_spectral_stats
from spectralquadnet.utils.device import resolve_device
from spectralquadnet.utils.seed import set_seed

#: The five classes §2.4.1 names as the hardest; F-5 predicts >= 3 of them
#: show recall far below precision.
_F5_CLASSES = [49, 52, 41, 51, 37]

_RULE = "═" * 74


def _hdr(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


# ══════════════════════════════════════════════════════════════════════
#  0-D · rank of the masked-statistics tensor  (F-2)
# ══════════════════════════════════════════════════════════════════════


def probe_D(ctx: dict[str, Any]) -> dict[str, Any]:
    """Singular-value spectrum of each training seed's ``(9, 40)`` statistics matrix.

    §2.2.5 models a seed's foreground pixels as ``x_{c,p} = a_p · r_c`` and
    derives ``rank(S) <= 2``: seven of the nine rows are ``r`` up to a scalar
    and the remaining two (skew, kurtosis) are constant in ``c``. F-2 predicts
    ``sigma_3 / sigma_1 < 0.05`` for more than 90 % of seeds.

    The matrix is built twice: from the raw patch (checkpoint-independent) and
    from ``se(x)``, which is what ``SpectralQuadNet.forward`` actually hands to
    ``masked_spectral_stats``. ``MaskedSpectralECA`` returns ``x + x*gate`` with
    a per-band gate, i.e. a diagonal column scaling, so it cannot change the
    rank — computing both makes that a measurement rather than an assertion.
    """
    _hdr("0-D · SVD of the (9, 40) masked-statistics tensor   [F-2]")
    device, model, store = ctx["device"], ctx["model"], ctx["store"]
    train_idx = ctx["train_idx"]

    load_ckpt(stage_ckpt_path(ctx["cfg"], 2), model, ctx["ema"], device)
    se = ctx["ema"].shadow.se
    se.eval()

    patches = store.require_patches()
    rows: dict[str, list[npt.NDArray[Any]]] = {"raw": [], "se": []}

    with torch.no_grad():
        for start in range(0, len(train_idx), 128):
            idx = np.sort(train_idx[start : start + 128])
            x = torch.from_numpy(np.array(patches[idx])).to(device)
            for tag, cube in (("raw", x), ("se", se(x))):
                stats = torch.stack(masked_spectral_stats(cube), dim=1)  # (B, 9, 40)
                sv = torch.linalg.svdvals(stats.float().cpu())
                rows[tag].append(sv.numpy())

    out: dict[str, Any] = {}
    for tag, chunks in rows.items():
        sv = np.concatenate(chunks, axis=0)  # (N, 9)
        s1 = sv[:, 0]
        r2 = sv[:, 1] / np.maximum(s1, 1e-30)
        r3 = sv[:, 2] / np.maximum(s1, 1e-30)
        # Effective rank at a 1 % relative tolerance on the singular values.
        eff_rank = (sv > 0.01 * s1[:, None]).sum(axis=1)
        energy2 = (sv[:, :2] ** 2).sum(1) / np.maximum((sv**2).sum(1), 1e-30)
        frac_below = float((r3 < 0.05).mean())
        out[tag] = {
            "n_seeds": int(sv.shape[0]),
            "sigma2_over_sigma1": {
                "mean": float(r2.mean()),
                "median": float(np.median(r2)),
            },
            "sigma3_over_sigma1": {
                "mean": float(r3.mean()),
                "median": float(np.median(r3)),
                "p90": float(np.percentile(r3, 90)),
                "max": float(r3.max()),
            },
            "frac_seeds_sigma3_over_sigma1_below_0.05": frac_below,
            "energy_in_top2_singular_values": {
                "mean": float(energy2.mean()),
                "min": float(energy2.min()),
            },
            "effective_rank_at_1pct": {
                "mean": float(eff_rank.mean()),
                "median": float(np.median(eff_rank)),
                "max": int(eff_rank.max()),
            },
        }
        d = out[tag]
        print(
            f"[{tag:>3}] n={d['n_seeds']:,}  "
            f"σ2/σ1 med={d['sigma2_over_sigma1']['median']:.4f}  "
            f"σ3/σ1 med={d['sigma3_over_sigma1']['median']:.2e}  "
            f"p90={d['sigma3_over_sigma1']['p90']:.2e}  max={d['sigma3_over_sigma1']['max']:.2e}"
        )
        print(
            f"      seeds with σ3/σ1 < 0.05 : {frac_below:.2%}   "
            f"energy in top-2 SVs: mean {d['energy_in_top2_singular_values']['mean']:.6f}  "
            f"min {d['energy_in_top2_singular_values']['min']:.6f}"
        )
        print(
            f"      effective rank @1%      : mean {d['effective_rank_at_1pct']['mean']:.2f}  "
            f"median {d['effective_rank_at_1pct']['median']:.0f}  "
            f"max {d['effective_rank_at_1pct']['max']}"
        )

    verdict = (
        "F-2 CONFIRMED"
        if out["se"]["frac_seeds_sigma3_over_sigma1_below_0.05"] > 0.90
        else "F-2 REFUTED"
    )
    out["verdict"] = (
        f"{verdict} — prediction was σ3/σ1 < 0.05 for >90% of seeds; measured "
        f"{out['se']['frac_seeds_sigma3_over_sigma1_below_0.05']:.2%} on the tensor the branch "
        f"actually receives."
    )
    print(f"\n{out['verdict']}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  0-E · fusion latent collapse  (F-4)
# ══════════════════════════════════════════════════════════════════════


def probe_E(ctx: dict[str, Any]) -> dict[str, Any]:
    """Pairwise cosine between the four fusion latents, per checkpoint.

    Two measurements, because they answer different halves of M-1:

    * the **parameter** ``cross_interaction.latents`` — what 0-E names, and what
      F-4's ``max_{n != n'} cos(L_n, L_n') > 0.95`` is stated over;
    * the **post-block latents** on a real validation batch, captured with a
      forward hook on the last fusion block's feed-forward. M-1's actual claim
      is that ``f = mean_n L_n ≈ L_1``, i.e. that the latents are still
      near-copies *after* the two cross/self-attention blocks — which the
      parameter alone cannot show. Since the block computes
      ``latents = latents + ff(latents)``, the hook's ``input[0] + output``
      reconstructs the exact tensor that ``latents.mean(dim=1)`` then pools.
    """
    _hdr("0-E · fusion latent cosine similarity   [F-4]")
    device, model, ema, cfg = ctx["device"], ctx["model"], ctx["ema"], ctx["cfg"]
    out: dict[str, Any] = {}

    # Historical since Tier 3: FU-1(b) / T3-4 deletes the latents rather than
    # fixing their initialisation scale, so there is nothing left to measure the
    # collapse of — which is §4.2's own wording for T3-4's criterion ("0-E
    # collapse metric becomes moot"). The recorded result is in
    # MIGRATION_PROGRESS under 0-E; the live check on what replaced it is
    # `tests/unit/test_fusion.py`.
    if not hasattr(model.cross_interaction, "latents"):
        raise RuntimeError(
            "0-E cannot run against the Tier-3 architecture: FU-1(b) deleted the Perceiver "
            "latents whose pairwise cosine this probe measures. Result recorded in "
            "MIGRATION_PROGRESS (0-E); the successor check is tests/unit/test_fusion.py."
        )

    for stage in (1, 2, 3):
        path = stage_ckpt_path(cfg, stage)
        if not Path(path).is_file():
            continue
        load_ckpt(path, model, ema, device)
        shadow = ema.shadow
        shadow.eval()

        lat = shadow.cross_interaction.latents.detach().float().cpu()  # (4, 256)
        ln = F.normalize(lat, dim=1)
        cos = (ln @ ln.T).numpy()
        off = cos[~np.eye(len(cos), dtype=bool)]

        # Post-block latents. `blocks[-1]` computes `latents = latents + ff(latents)`,
        # so `input[0] + output` on an `ff` forward hook is the final latent state.
        captured: list[torch.Tensor] = []

        def _hook(
            _m: Any,
            i: tuple[torch.Tensor, ...],
            o: torch.Tensor,
            sink: list[torch.Tensor] = captured,
        ) -> None:
            sink.append((i[0] + o).detach().float().cpu())

        last_block = shadow.cross_interaction.blocks[-1]
        handle = last_block["ff"].register_forward_hook(_hook)
        with torch.no_grad():
            x, _ = next(iter(ctx["val_ldr"]))
            shadow(x.to(device))
        handle.remove()

        post = captured[-1]  # (B, 4, 256)
        pl = F.normalize(post, dim=2)
        pcos = torch.einsum("bnd,bmd->bnm", pl, pl).numpy()
        poff = pcos[:, ~np.eye(pcos.shape[1], dtype=bool)]
        # How close the pooled token is to a single latent — M-1's `f ≈ L_1`.
        pooled = post.mean(dim=1, keepdim=True)
        cos_pool = F.cosine_similarity(post, pooled, dim=2).numpy()  # (B, 4)

        rec: dict[str, Any] = {
            "param_max_offdiag_cos": float(off.max()),
            "param_min_offdiag_cos": float(off.min()),
            "param_mean_offdiag_cos": float(off.mean()),
            "param_latent_norms": [float(v) for v in lat.norm(dim=1)],
            "postblock_max_offdiag_cos_mean": float(poff.max(axis=1).mean()),
            "postblock_mean_offdiag_cos": float(poff.mean()),
            "postblock_min_cos_to_pooled_mean": float(cos_pool.min(axis=1).mean()),
            "f4_holds": bool(off.max() > 0.95),
            "postblock_collapsed": bool(poff.max(axis=1).mean() > 0.95),
        }
        out[str(stage)] = rec
        print(
            f"stage {stage}: latents param   max cos={rec['param_max_offdiag_cos']:+.4f}  "
            f"mean={rec['param_mean_offdiag_cos']:+.4f}  "
            f"‖L_n‖={[round(v, 2) for v in rec['param_latent_norms']]}"
        )
        print(
            f"         post-block Lₙ  max cos={rec['postblock_max_offdiag_cos_mean']:+.4f}  "
            f"mean={rec['postblock_mean_offdiag_cos']:+.4f}  "
            f"min cos(Lₙ, mean)={rec['postblock_min_cos_to_pooled_mean']:+.4f}"
        )
        print(
            f"         F-4 (param max cos > 0.95): "
            f"{'HOLDS' if rec['f4_holds'] else 'FAILS'}   "
            f"post-block collapse: {'YES' if rec['postblock_collapsed'] else 'NO'}"
        )

    all_hold = all(v["f4_holds"] for v in out.values())
    collapsed = [k for k, v in out.items() if v["postblock_collapsed"]]
    out["verdict"] = (
        (
            "F-4 CONFIRMED — max pairwise cosine of the latent parameter exceeds 0.95 in every "
            "checkpoint"
            if all_hold
            else "F-4 REFUTED on the parameter — max pairwise cosine of "
            "`cross_interaction.latents` stays below 0.95 in "
            f"{sum(1 for v in out.values() if not v['f4_holds'])}/{len(out)} checkpoints"
        )
        + ". Post-block latent state on real data: "
        + (
            f"collapsed (>0.95) in checkpoints {collapsed}"
            if collapsed
            else "not collapsed in any checkpoint"
        )
        + "."
    )
    print(f"\n{out['verdict']}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  0-F · wavelength spacing
# ══════════════════════════════════════════════════════════════════════


def probe_F(ctx: dict[str, Any]) -> dict[str, Any]:
    """Δλ histogram over the 40 selected bands; reports max/min.

    §2.2.6 (C-5): every 1-D convolution in branches A/B/D treats the band axis
    as uniformly spaced. It is not — SPA picked 40 bands out of 256, so a
    ``kernel=3`` conv is a finite difference over a step that varies by however
    much this ratio says.
    """
    _hdr("0-F · wavelength spacing of the 40 selected bands   [C-5]")
    df = pd.read_csv(ctx["cfg"].data.wavelength_path, sep=None, engine="python")
    wl = df.iloc[:, -1].to_numpy(dtype=float)
    d = np.diff(wl)

    counts, edges = np.histogram(d, bins=10)
    print(f"bands={len(wl)}  span={wl.min():.2f}–{wl.max():.2f} nm")
    print(
        f"Δλ  min={d.min():.3f}  median={np.median(d):.3f}  mean={d.mean():.3f}  "
        f"max={d.max():.3f} nm"
    )
    print(f"Δλ  max/min ratio = {d.max() / d.min():.2f}×")
    print("\nΔλ histogram:")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:], strict=True):
        print(f"  [{lo:7.2f}, {hi:7.2f}) nm : {'█' * c}{'' if c else '·'} ({c})")

    # A kernel=3 conv centred on band i spans wl[i+1]-wl[i-1].
    span3 = wl[2:] - wl[:-2]
    print(
        f"\nkernel=3 receptive span: min={span3.min():.2f}  max={span3.max():.2f} nm  "
        f"({span3.max() / span3.min():.2f}× spread)"
    )
    out = {
        "n_bands": int(len(wl)),
        "wl_min_nm": float(wl.min()),
        "wl_max_nm": float(wl.max()),
        "delta_lambda": {
            "min": float(d.min()),
            "median": float(np.median(d)),
            "mean": float(d.mean()),
            "max": float(d.max()),
            "max_over_min": float(d.max() / d.min()),
        },
        "kernel3_span_nm": {
            "min": float(span3.min()),
            "max": float(span3.max()),
            "max_over_min": float(span3.max() / span3.min()),
        },
        "histogram": {
            "counts": [int(c) for c in counts],
            "edges": [float(e) for e in edges],
        },
    }
    out["verdict"] = (
        f"C-5 magnitude: the band grid is irregular by {d.max() / d.min():.1f}×, so a shared "
        f"1-D kernel applies the same finite-difference weights across steps from "
        f"{d.min():.2f} nm to {d.max():.2f} nm."
    )
    print(f"\n{out['verdict']}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  0-G · hard-class precision/recall  (F-5)
# ══════════════════════════════════════════════════════════════════════


def probe_G(ctx: dict[str, Any]) -> dict[str, Any]:
    """Precision and recall for the hardest classes, per checkpoint.

    §2.4.1 (M-6): the margin rule ``m_c = m_base + m_delta·(1 − F1_c)`` widens
    the angular margin for low-F1 classes. Widening a margin *suppresses* that
    class's logit, which raises precision and lowers recall — so if a class is
    hard because its recall is already low, the rule pushes it the wrong way.
    F-5 predicts at least 3 of ``{49, 52, 41, 51, 37}`` show ``R_c << P_c``.
    """
    _hdr("0-G · precision vs recall for the hardest classes   [F-5]")
    preds_dir = Path(ctx["out_root"]) / "preds"
    out: dict[str, Any] = {"sources": {}}

    sources: list[tuple[str, Path, Path]] = []
    for stage in (1, 2, 3):
        p = preds_dir / f"test_stage{stage}_tta12.npy"
        t = preds_dir / "test_targets.npy"
        if p.is_file() and t.is_file():
            sources.append((f"stage{stage}_tta12", p, t))
    ref = Path(ctx["cfg"].output_dir)
    if (ref / "test_preds_TTA.npy").is_file():
        sources.append(("reference_run_TTA", ref / "test_preds_TTA.npy", ref / "test_targets.npy"))

    if not sources:
        print("No prediction arrays found — run scripts/phase0_eval_checkpoints.py first.")
        return {"error": "no prediction arrays"}

    n_wrong_sign_any = 0
    for name, ppath, tpath in sources:
        p, t = np.load(ppath), np.load(tpath)
        labels = list(range(ctx["cfg"].data.num_classes))
        prec, rec, f1, sup = precision_recall_fscore_support(t, p, labels=labels, zero_division=0)
        order = np.argsort(f1, kind="stable")
        hardest = [int(c) for c in order[:5]]

        print(f"\n── {name} ──  (5 hardest by F1: {hardest})")
        print(f"{'class':>6} {'P':>7} {'R':>7} {'F1':>7} {'sup':>5}  {'R−P':>7}  note")
        rows = []
        wrong_sign = 0
        for c in sorted(set(hardest) | set(_F5_CLASSES)):
            gap = rec[c] - prec[c]
            tag = ""
            if c in _F5_CLASSES:
                tag += "F-5 "
            if c in hardest:
                tag += "hardest "
            if gap < -0.05:
                tag += "← R≪P: margin rule has the WRONG SIGN"
                if c in _F5_CLASSES:
                    wrong_sign += 1
            elif gap > 0.05:
                tag += "(R>P: margin rule directionally right)"
            print(
                f"{c:>6} {prec[c]:>7.3f} {rec[c]:>7.3f} {f1[c]:>7.3f} {sup[c]:>5}  "
                f"{gap:>+7.3f}  {tag}"
            )
            rows.append(
                {
                    "class": c,
                    "precision": float(prec[c]),
                    "recall": float(rec[c]),
                    "f1": float(f1[c]),
                    "support": int(sup[c]),
                    "recall_minus_precision": float(gap),
                    "in_f5_set": c in _F5_CLASSES,
                    "in_hardest5": c in hardest,
                }
            )
        f5_gaps = {c: float(rec[c] - prec[c]) for c in _F5_CLASSES}
        print(f"   F-5 set {_F5_CLASSES}: {wrong_sign}/5 have R−P < −0.05 " f"(prediction: >= 3)")
        n_wrong_sign_any = max(n_wrong_sign_any, wrong_sign)
        out["sources"][name] = {
            "hardest5_by_f1": hardest,
            "rows": rows,
            "f5_recall_minus_precision": f5_gaps,
            "f5_count_recall_far_below_precision": wrong_sign,
            "macro_precision": float(prec.mean()),
            "macro_recall": float(rec.mean()),
        }

    per_source = ", ".join(
        f"{k}: {v['f5_count_recall_far_below_precision']}/5" for k, v in out["sources"].items()
    )
    out["verdict"] = (
        f"F-5 {'CONFIRMED' if n_wrong_sign_any >= 3 else 'REFUTED'} — prediction was that >= 3 of "
        f"{{49,52,41,51,37}} show recall more than 0.05 below precision; measured {per_source}."
    )
    print(f"\n{out['verdict']}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  0-I · dead sub-centres  (F-8)
# ══════════════════════════════════════════════════════════════════════


def probe_I(ctx: dict[str, Any]) -> dict[str, Any]:
    """Win rate of each ``(class, sub-centre)`` pair over the training split.

    ``AdaptiveSubcenterArcFaceHead.forward`` reduces the ``(B, C, K)`` cosine
    tensor with ``max(dim=2)``, so on every sample exactly one sub-centre per
    class carries that class's logit — and therefore its gradient. Two win
    rates are reported because they answer different questions:

    * **global** — does ``(c, k)`` ever win, over *any* training sample? A
      sub-centre that never wins receives zero gradient forever. This is the
      criterion F-8 is stated over.
    * **own-class** — does ``(c, k)`` win for samples whose label *is* ``c``?
      This is what decides whether the sub-centre represents a real
      sub-population of its class, which is the feature's stated purpose.

    Stage 1's ArcFace head was never bootstrapped (``arcface_init_done: false``
    in its meta), so its numbers describe a Xavier-random head and are reported
    only as a control.
    """
    _hdr("0-I · ArcFace sub-centre win rates   [F-8]")
    device, model, ema, cfg = ctx["device"], ctx["model"], ctx["ema"], ctx["cfg"]
    out: dict[str, Any] = {}

    for stage in (1, 2, 3):
        path = stage_ckpt_path(cfg, stage)
        if not Path(path).is_file():
            continue
        ck = load_ckpt(path, model, ema, device)
        shadow = ema.shadow
        shadow.eval()
        head = shadow.arcface_head
        K, C = head.K, head.C

        embs, targs = [], []
        with torch.no_grad():
            for x, y in ctx["train_eval_ldr"]:
                _, e = shadow(x.to(device, non_blocking=True), return_embed=True)
                embs.append(e.float().cpu())
                targs.append(y.cpu())
        emb = torch.cat(embs)
        tgt = torch.cat(targs)

        w_n = F.normalize(head.weight.detach().float().cpu(), dim=1)
        cos = (F.normalize(emb, dim=1) @ w_n.T).view(-1, C, K)  # (N, C, K)
        argk = cos.argmax(dim=2).numpy()  # (N, C)

        # Global win counts: (c, k) wins on sample i if it is class c's argmax.
        global_wins = np.zeros((C, K), dtype=np.int64)
        for k in range(K):
            global_wins[:, k] = (argk == k).sum(axis=0)
        dead_global = int((global_wins == 0).sum())
        cls_with_dead_global = int((global_wins == 0).any(axis=1).sum())

        # Own-class win counts.
        own_wins = np.zeros((C, K), dtype=np.int64)
        for c in range(C):
            sel = tgt.numpy() == c
            if sel.any():
                for k in range(K):
                    own_wins[c, k] = int((argk[sel, c] == k).sum())
        dead_own = int((own_wins == 0).sum())
        cls_with_dead_own = int((own_wins == 0).any(axis=1).sum())

        share = global_wins / np.maximum(global_wins.sum(axis=1, keepdims=True), 1)
        top_share = share.max(axis=1)
        note = " (head never bootstrapped — control only)" if not ck.get("use_arcface") else ""
        print(f"\nstage {stage}  K={K}  C={C}  N_train={len(emb):,}{note}")
        print(
            f"  global : dead (c,k) pairs {dead_global}/{C * K} "
            f"({dead_global / (C * K):.1%})   classes with >=1 dead sub-centre "
            f"{cls_with_dead_global}/{C}"
        )
        print(
            f"  own-cls: dead (c,k) pairs {dead_own}/{C * K} "
            f"({dead_own / (C * K):.1%})   classes with >=1 dead sub-centre "
            f"{cls_with_dead_own}/{C}"
        )
        print(
            f"  dominant sub-centre share (global): mean {top_share.mean():.3f}  "
            f"min {top_share.min():.3f}  max {top_share.max():.3f}   "
            f"(1/K = {1 / K:.3f} would be balanced)"
        )
        out[str(stage)] = {
            "arcface_live": bool(ck.get("use_arcface", False)),
            "K": int(K),
            "C": int(C),
            "n_train": int(len(emb)),
            "global": {
                "dead_pairs": dead_global,
                "dead_pair_fraction": dead_global / (C * K),
                "classes_with_dead": cls_with_dead_global,
            },
            "own_class": {
                "dead_pairs": dead_own,
                "dead_pair_fraction": dead_own / (C * K),
                "classes_with_dead": cls_with_dead_own,
            },
            "dominant_share": {
                "mean": float(top_share.mean()),
                "min": float(top_share.min()),
                "max": float(top_share.max()),
            },
        }

    live = [v for k, v in out.items() if k.isdigit() and v["arcface_live"]]
    hits = [v for v in live if v["global"]["classes_with_dead"] >= 1]
    out["verdict"] = (
        f"F-8 {'CONFIRMED' if hits else 'REFUTED'} — prediction was that >= 1 sub-centre per "
        f"class has zero win-rate over the training split; measured "
        + ", ".join(
            f"stage {k}: {v['global']['classes_with_dead']}/{v['C']} classes with a dead "
            f"sub-centre (own-class criterion: {v['own_class']['classes_with_dead']}/{v['C']})"
            for k, v in out.items()
            if k.isdigit() and v["arcface_live"]
        )
    )
    print(f"\n{out['verdict']}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  0-J · config-key wiring  (N-1a…f)
# ══════════════════════════════════════════════════════════════════════


def probe_J(ctx: dict[str, Any]) -> dict[str, Any]:
    """Compare every ``configs/model/*.yaml`` key against the module it names.

    §2.7 lists five dead control paths. ``scripts/check_config_roundtrip.py``
    already asserts that every key "has a home" in the schema — which is a
    different claim from the key reaching the module it is named after. This
    probe reads the value off the *constructed, checkpoint-loaded* module and
    reports the pairs that disagree.

    **Historical since Tier 3.** All four keys it found dead are now wired or
    deleted (T3-3, T3-4, T3-5), the ``cross_interaction.blocks`` it reads no
    longer exist (FU-1(b) deleted the Perceiver), and the schema-v1 checkpoint
    it loads is refused by ``remap_state_dict``. The probe raises rather than
    silently reporting a wiring audit of a model that is not the one in the
    tree; its recorded result lives in MIGRATION_PROGRESS under 0-J, and the
    live equivalent is ``tests/unit/test_config_wiring.py``, which perturbs each
    key and requires the forward pass to react.
    """
    _hdr("0-J · config-key wiring audit   [N-1a…f]")
    cfg, model, ema, device = ctx["cfg"], ctx["model"], ctx["ema"], ctx["device"]
    if not hasattr(model.cross_interaction, "blocks"):
        raise RuntimeError(
            "0-J cannot run against the Tier-3 architecture. The keys it audits are wired "
            "(T3-3, T3-5) or deleted (T3-4), the fusion has no attention blocks to read a "
            "head count off, and the schema-v1 checkpoint it loads is refused. Its result is "
            "recorded in MIGRATION_PROGRESS (0-J); the live check is "
            "tests/unit/test_config_wiring.py."
        )
    load_ckpt(stage_ckpt_path(cfg, 2), model, ema, device)
    m = ema.shadow

    blk = m.cross_interaction.blocks[0]
    specf_blk = m.branch_d.spectral_blocks[0]

    checks: list[tuple[str, Any, Any, str]] = [
        (
            "model.fusion_heads",
            cfg.model.fusion_heads,
            blk["cross_attn"].num_heads,
            "CrossModalInteraction(...) is built without `heads=`, so it keeps its "
            "signature default of 8.",
        ),
        (
            "model.fusion_drop",
            cfg.model.fusion_drop,
            float(blk["cross_attn"].dropout),
            "passed through as `drop=cfg.model.fusion_drop`.",
        ),
        (
            "model.specf_drop",
            cfg.model.specf_drop,
            float(specf_blk.attn.dropout),
            "SpecFormerBranch(...) is built with a literal `dropout=0.10`.",
        ),
        (
            "model.specf_dim",
            cfg.model.specf_dim,
            int(specf_blk.ln1.normalized_shape[0]),
            "passed through as `d_model=`.",
        ),
        (
            "model.specf_heads",
            cfg.model.specf_heads,
            int(specf_blk.attn.num_heads),
            "passed through as `n_heads=`.",
        ),
        (
            "model.specf_layers",
            cfg.model.specf_layers,
            len(m.branch_d.spectral_blocks) + len(m.branch_d.spatial_blocks),
            "split as n_layers//2 spectral + n_layers//2 spatial blocks.",
        ),
        (
            "model.specf_patch",
            cfg.model.specf_patch,
            "unused",
            "SpecFormerBranch's own docstring records `patch_size` as accepted "
            "but unused; tokenisation is stride-based.",
        ),
        (
            "model.subcenter_K",
            cfg.model.subcenter_K,
            int(m.arcface_head.K),
            "passed through as `K=`.",
        ),
        (
            "model.aux_head_hidden",
            cfg.model.aux_head_hidden,
            int(m.aux_head_a.net[0].out_features),
            "passed through as the AuxiliaryHead hidden width.",
        ),
        (
            "model.wl_embed_dim",
            cfg.model.wl_embed_dim,
            "unused",
            "accepted by SpectralQuadNet.__init__ and never read in its body.",
        ),
        (
            "model.branch_drop_prob",
            cfg.model.branch_drop_prob,
            "stored, not used",
            "stored as `self.branch_drop_prob`; forward() builds its own literal "
            "`[0.0, 0.0, 0.30, 0.20]` drop vector instead.",
        ),
        (
            "stage2.arcface_m0",
            cfg.stage2.arcface_m0,
            "see stage2 loop",
            "margin warm-up start; only reachable through stage2's arc_m schedule.",
        ),
    ]

    print(f"{'config key':<26} {'yaml':>10} {'module':>16}  status")
    rows, n_dead = [], 0
    for key, yaml_v, mod_v, why in checks:
        if isinstance(mod_v, str):
            status = "DEAD" if mod_v in ("unused", "stored, not used") else "n/a"
        else:
            status = "wired" if float(yaml_v) == float(mod_v) else "DEAD (overridden)"
        if status.startswith("DEAD"):
            n_dead += 1
        print(f"{key:<26} {str(yaml_v):>10} {str(mod_v):>16}  {status}")
        rows.append(
            {
                "key": key,
                "yaml_value": yaml_v if not isinstance(yaml_v, str) else str(yaml_v),
                "module_value": mod_v if not isinstance(mod_v, str) else str(mod_v),
                "status": status,
                "why": why,
            }
        )

    print("\nNotes:")
    for r in rows:
        if r["status"].startswith("DEAD"):
            print(f"  · {r['key']}: {r['why']}")

    # The two the plan asks for explicitly.
    print(
        f"\nAsked for by 0-J → cross_interaction heads = {blk['cross_attn'].num_heads} "
        f"(config says {cfg.model.fusion_heads});  "
        f"SpecFormer dropout = {float(specf_blk.attn.dropout)} "
        f"(config says {cfg.model.specf_drop})"
    )
    out = {
        "rows": rows,
        "n_dead": n_dead,
        "cross_interaction_heads_actual": int(blk["cross_attn"].num_heads),
        "cross_interaction_heads_config": int(cfg.model.fusion_heads),
        "specformer_dropout_actual": float(specf_blk.attn.dropout),
        "specformer_dropout_config": float(cfg.model.specf_drop),
    }
    out["verdict"] = (
        f"N-1a CONFIRMED (fusion_heads={cfg.model.fusion_heads} in YAML, "
        f"{blk['cross_attn'].num_heads} in the module) and N-1b CONFIRMED "
        f"(specf_drop={cfg.model.specf_drop} in YAML, "
        f"{float(specf_blk.attn.dropout)} in the module). "
        f"{n_dead} of {len(rows)} audited keys never reach the module they name."
    )
    print(f"\n{out['verdict']}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════

_PROBES = {"D": probe_D, "E": probe_E, "F": probe_F, "G": probe_G, "I": probe_I, "J": probe_J}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="Override cfg.device.")
    ap.add_argument("--out", default="outputs/phase0", help="Directory for Phase-0 artifacts.")
    ap.add_argument("--only", default=None, help="Comma-separated probe letters, e.g. D,E,J.")
    args = ap.parse_args()

    wanted = [s.strip().upper() for s in args.only.split(",")] if args.only else list(_PROBES)

    cfg = load_experiment_config()
    set_seed(cfg.seed)
    device = resolve_device(args.device or cfg.device)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    store = DataStore.from_config(cfg.data, device)
    all_labels, train_idx, val_idx, test_idx = build_splits(cfg)
    model = SpectralQuadNet.from_config(cfg, store.require_wavelengths()).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    _, val_ldr, _ = build_loaders(cfg, store, device, train_idx, val_idx, test_idx, 256)

    # An un-augmented, un-shuffled, non-dropping pass over the *training*
    # indices — 0-I needs every training sample, which build_loaders' train
    # loader (shuffle=True, drop_last=True) would not give.
    train_eval_ldr = DataLoader(
        RiceSeedDataset(
            train_idx, aug_strength="none", store=store, data_cfg=cfg.data, device=device
        ),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    ctx: dict[str, Any] = {
        "cfg": cfg,
        "device": device,
        "store": store,
        "model": model,
        "ema": ema,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "val_ldr": val_ldr,
        "train_eval_ldr": train_eval_ldr,
        "out_root": out_root,
    }

    print(f"Device: {device}   Checkpoints: {cfg.output_dir}")
    results: dict[str, Any] = {}
    for letter in wanted:
        if letter not in _PROBES:
            print(f"Unknown probe {letter!r} — skipping")
            continue
        results[f"0-{letter}"] = _PROBES[letter](ctx)

    path = out_root / (
        "probe_results.json"
        if len(wanted) == len(_PROBES)
        else f"probe_results_{''.join(wanted)}.json"
    )
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
