"""S10 — the shipped objective and schedule, worked out analytically (no data, no training).

Writes:
  objective_geometry.csv  for the cosine-softmax head (C = 90 classes): the target-vs-rival cosine
                          gap at which label smoothing is optimal, the loss floor it implies, and
                          the plain-cosine gap an additive angular margin then requires
  schedule_budget.csv     share of the cumulative learning rate, and the LR multiplier, that each
                          phase of a regime receives (shipped; X1 as frozen, which is S10's candidate R1)
  horizons.csv            EMA and Adam-second-moment memory in epochs, at the sweep's steps/epoch
  mixup_strength.csv      how much a Beta(α, α) mixup actually mixes
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from s10common import save

C = 90


def ls_gap(eps: float, s: float) -> float:
    """Logit-space optimum of CE with label smoothing ε: q_y/q_o = (1-ε+ε/C)/(ε/C)."""
    if eps == 0:
        return math.inf
    return math.log((1 - eps + eps / C) / (eps / C)) / s


def ls_floor(eps: float) -> float:
    qy, qo = 1 - eps + eps / C, eps / C
    return -(qy * math.log(qy) + (C - 1) * qo * math.log(qo)) if eps > 0 else 0.0


def geometry() -> pd.DataFrame:
    rows = []
    for s in (16.0, 32.0, 48.0, 64.0):
        for eps in (0.10, 0.056, 0.04, 0.0):
            g = ls_gap(eps, s)
            for m in (0.0, 0.30):
                for th_deg in (30, 45, 60):
                    th = math.radians(th_deg)
                    extra = math.cos(th) - math.cos(min(th + m, math.pi / 2))
                    rows.append(dict(s=s, label_smoothing=eps, margin=m, theta_y_deg=th_deg,
                                     ls_optimal_gap_penalised=g, plain_gap_required=g + extra,
                                     margin_cost_in_cosine=extra, ls_loss_floor=ls_floor(eps),
                                     # loss of one kernel whose penalised target sits 0.1 below its rival
                                     ce_if_penalised_target_0p1_below_rival=math.log(1 + math.exp(0.1 * s)) ))
    return pd.DataFrame(rows)


def lr_mult(ep: int, total: int, warm: int = 5, floor: float = 0.01) -> float:
    if ep <= warm:
        return ep / warm
    p = min(max((ep - warm) / max(total - warm, 1), 0.0), 1.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def budget() -> pd.DataFrame:
    regimes = {
        # name: (total epochs, [(phase label, first, last), ...])
        "shipped (150 ep; mixup 1-110; margin ramp 111-130)": (150, [("mixup", 1, 110), ("clean, margin ramp", 111, 130), ("clean, margin 0.30", 131, 150)]),
        # S10's candidate regime R1 is X1 exactly (S10 §7), so it has no row of its own.
        "X1 as frozen = S10 R1 (200 ep; mixup 1-30; no margin)": (200, [("mixup", 1, 30), ("clean", 31, 200)]),
    }
    rows = []
    for name, (total, phases) in regimes.items():
        lr = np.array([lr_mult(e, total) for e in range(1, total + 1)])
        for label, a, b in phases:
            rows.append(dict(regime=name, phase=label, first_epoch=a, last_epoch=b,
                             share_of_cumulative_lr=lr[a - 1:b].sum() / lr.sum(),
                             lr_mult_at_phase_start=lr[a - 1], lr_mult_at_phase_end=lr[b - 1]))
    return pd.DataFrame(rows)


def horizons() -> pd.DataFrame:
    rows = []
    for arm, steps in (("grouped (3,683 train, batch 128, drop_last)", 28), ("stratified (5,130 train)", 40)):
        for name, beta in (("EMA decay 0.999", 0.999), ("EMA decay 0.995", 0.995), ("EMA decay 0.998", 0.998), ("Adam β2 0.999", 0.999)):
            tau = 1 / (1 - beta)
            rows.append(dict(arm=arm, steps_per_epoch=steps, quantity=name,
                             time_constant_steps=tau, time_constant_epochs=tau / steps,
                             half_life_epochs=math.log(2) / (1 - beta) / steps))
    return pd.DataFrame(rows)


def mixup_strength() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for a in (0.2, 0.35, 0.4, 1.0):
        lam = rng.beta(a, a, 1_000_000)
        mn = np.minimum(lam, 1 - lam)
        rows.append(dict(alpha=a, mean_minor_share=mn.mean(), p_minor_gt_0p2=(mn > 0.2).mean(),
                         p_minor_gt_0p4=(mn > 0.4).mean(),
                         # train/acc as logged under mixup is scored against ya only: a perfect
                         # dominant-label predictor scores P(λ ≥ 0.5) + P(λ < 0.5)/C
                         ceiling_of_logged_mixup_train_acc=(lam >= 0.5).mean() + (lam < 0.5).mean() / C))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    save(geometry(), "objective_geometry.csv")
    save(budget(), "schedule_budget.csv")
    save(horizons(), "horizons.csv")
    save(mixup_strength(), "mixup_strength.csv")
