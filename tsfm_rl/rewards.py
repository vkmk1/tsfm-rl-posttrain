"""Losses and rewards for post-training a quantile forecaster.

All functions take q (B, H, 9) quantile forecasts (levels 0.1 .. 0.9) and y (B, H) realizations and return a
per-episode value (B,). Losses are minimized; rewards are maximized (reward = -loss unless stated). Per-step
variants return (B, H) so that policy-gradient methods can assign credit along the horizon (TimeRFT-style).

    point losses      mse, mae                                   what TS-GRPO / TimeRFT / TPO reward
    proper scores     pinball (WQL), crps (from quantiles or samples), interval_score
    relative          skill(score, reference score)               fev-bench's skill score, scale-free
    shape             directional_accuracy, soft_dtw (differentiable), shape_reward
    calibration       coverage(alpha), width
    decision          newsvendor
    composite         composite(q, y, ref_q, weights)             the reward under test
"""
from __future__ import annotations

import torch

QLEVELS = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def median(q):
    return q[..., 4]


# ----------------------------------------------------------------------------- point losses
def mse(q, y, per_step=False):
    e = (median(q) - y) ** 2; return e if per_step else e.mean(-1)


def mae(q, y, per_step=False):
    e = (median(q) - y).abs(); return e if per_step else e.mean(-1)


def mase(q, y, history, m=1):
    """MAE scaled by the in-sample naive-m forecast error (GIFT-Eval convention)."""
    scale = (history[:, m:] - history[:, :-m]).abs().mean(-1).clamp_min(1e-6)
    return mae(q, y) / scale


# ----------------------------------------------------------------------------- proper scoring rules
def pinball(q, y, per_step=False):
    """Mean pinball over the nine levels (= weighted quantile loss up to normalization)."""
    lv = QLEVELS.to(q.device, q.dtype); d = y[..., None] - q
    p = torch.maximum(lv * d, (lv - 1) * d).mean(-1)
    return p if per_step else p.mean(-1)


def crps_from_quantiles(q, y, per_step=False):
    """CRPS approximated as 2 x the average pinball loss over the levels (the standard quantile-CRPS)."""
    return 2 * pinball(q, y, per_step)


def crps_from_samples(samples, y, per_step=False):
    """Energy-form CRPS from K samples (K, B, H): E|X - y| - 0.5 E|X - X'|."""
    K = samples.shape[0]
    t1 = (samples - y[None]).abs().mean(0)
    t2 = (samples[:, None] - samples[None]).abs().sum((0, 1)) / (K * (K - 1)) if K > 1 else torch.zeros_like(t1)
    c = t1 - 0.5 * t2
    return c if per_step else c.mean(-1)


def interval_score(q, y, alpha=0.2, per_step=False):
    """Winkler / interval score for the central (1 - alpha) interval (alpha = 0.2 -> quantiles 0.1 and 0.9)."""
    lo, hi = q[..., 0], q[..., -1]
    s = (hi - lo) + (2 / alpha) * (lo - y).clamp_min(0) + (2 / alpha) * (y - hi).clamp_min(0)
    return s if per_step else s.mean(-1)


# ----------------------------------------------------------------------------- calibration
def coverage(q, y, alpha=0.2):
    """Empirical coverage of the central (1 - alpha) interval, (B,)."""
    return ((y >= q[..., 0]) & (y <= q[..., -1])).float().mean(-1)


def width(q):
    return (q[..., -1] - q[..., 0]).mean(-1)


# ----------------------------------------------------------------------------- shape
def directional_accuracy(q, y, per_step=False):
    """Fraction of horizon steps where the median's first difference has the sign of the realized one."""
    dm = torch.sign(median(q)[:, 1:] - median(q)[:, :-1]); dy = torch.sign(y[:, 1:] - y[:, :-1])
    a = (dm == dy).float()
    return a if per_step else a.mean(-1)


def soft_dtw(x, y, gamma=0.1):
    """Soft-DTW (Cuturi & Blondel 2017) between x, y (B, H) with squared cost; differentiable. O(H^2), fine for H = 64."""
    B, H = x.shape; D = (x[:, :, None] - y[:, None, :]) ** 2
    R = torch.full((B, H + 1, H + 1), float("inf"), device=x.device, dtype=x.dtype); R[:, 0, 0] = 0
    for i in range(1, H + 1):
        for j in range(1, H + 1):
            r = torch.stack([R[:, i - 1, j], R[:, i, j - 1], R[:, i - 1, j - 1]], -1)
            R[:, i, j] = D[:, i - 1, j - 1] - gamma * torch.logsumexp(-r / gamma, -1)
    return R[:, H, H]


def shape_reward(q, y, gamma=0.1):
    """1 - softDTW(median, y) / softDTW(constant last value, y): 1 = perfect shape, 0 = no better than flat."""
    m = median(q); flat = torch.zeros_like(y) + y[:, :1] * 0 + m[:, :1]
    d = soft_dtw(m, y, gamma); d0 = soft_dtw(flat, y, gamma).clamp_min(1e-6)
    return 1 - d / d0


# ----------------------------------------------------------------------------- relative and decision
def skill(score, ref_score):
    """1 - score / ref_score, clipped to [-1, 1]: positive = better than the reference."""
    return (1 - score / ref_score.clamp_min(1e-6)).clamp(-1, 1)


def newsvendor(q, y, cu=1.0, co=0.5):
    """Cost of ordering the critical-fractile quantile (cu / (cu + co) = 2/3 -> the 0.7 quantile)."""
    order = q[..., 6]; return (cu * (y - order).clamp_min(0) + co * (order - y).clamp_min(0)).mean(-1)


# ----------------------------------------------------------------------------- the reward under test
def composite(q, y, ref_q=None, w_skill=1.0, w_shape=0.25, w_cov=0.25, alpha=0.2, per_step=False):
    """r = w_skill * skill(CRPS vs reference) + w_shape * directional accuracy + w_cov * (1 - |coverage - (1 - alpha)| / (1 - alpha)).
    With ref_q = None the skill term is the negative CRPS itself. per_step gives (B, H) credit using per-step CRPS and direction."""
    if per_step:
        c = crps_from_quantiles(q, y, per_step=True)
        base = -c if ref_q is None else skill(c, crps_from_quantiles(ref_q, y, per_step=True))
        da = torch.nn.functional.pad(directional_accuracy(q, y, per_step=True), (1, 0), value=0.5)
        inside = ((y >= q[..., 0]) & (y <= q[..., -1])).float()
        return w_skill * base + w_shape * da + w_cov * inside
    c = crps_from_quantiles(q, y)
    base = -c if ref_q is None else skill(c, crps_from_quantiles(ref_q, y))
    cov_term = 1 - (coverage(q, y, alpha) - (1 - alpha)).abs() / (1 - alpha)
    return w_skill * base + w_shape * directional_accuracy(q, y) + w_cov * cov_term


REWARDS = {
    "mse": lambda q, y, ref=None: -mse(q, y),
    "mae": lambda q, y, ref=None: -mae(q, y),
    "pinball": lambda q, y, ref=None: -pinball(q, y),
    "crps": lambda q, y, ref=None: -crps_from_quantiles(q, y),
    "skill": lambda q, y, ref=None: skill(crps_from_quantiles(q, y), crps_from_quantiles(ref, y)) if ref is not None else -crps_from_quantiles(q, y),
    "interval": lambda q, y, ref=None: -interval_score(q, y),
    "newsvendor": lambda q, y, ref=None: -newsvendor(q, y),
    "composite": lambda q, y, ref=None: composite(q, y, ref),
}
LOSSES = {"mse": mse, "mae": mae, "pinball": pinball, "crps": crps_from_quantiles, "interval": interval_score}


def sample_forecasts(q, K, g=None):
    """K samples (K, B, H) from the piecewise-linear quantile function (inverse CDF; u clipped to [0.05, 0.95])."""
    B, H, _ = q.shape; u = torch.rand(K, B, H, generator=g, device=q.device).clamp(0.05, 0.95)
    qs, _ = q.sort(-1); lv = QLEVELS.to(q.device, q.dtype); idx = torch.clamp(((u - 0.1) / 0.1).floor().long(), 0, 7)
    q0 = qs.gather(-1, idx.permute(1, 2, 0)).permute(2, 0, 1); q1 = qs.gather(-1, (idx + 1).permute(1, 2, 0)).permute(2, 0, 1)
    return q0 + (u - lv[idx]) / 0.1 * (q1 - q0)


def logpdf(q, y):
    """log density of y (B, H) under the piecewise-linear quantile function q (B, H, 9); differentiable in q.
    Density 0.1 / gap between adjacent quantiles inside, exponential tails outside."""
    qs, _ = q.sort(-1); gaps = (qs[..., 1:] - qs[..., :-1]).clamp_min(1e-4)
    inside = (y[..., None] >= qs[..., :-1]) & (y[..., None] < qs[..., 1:])
    lp_in = (inside.float() * torch.log(0.1 / gaps)).sum(-1)
    lo, hi = qs[..., 0], qs[..., -1]; scale = (hi - lo).clamp_min(1e-3) / 4
    lp_lo = torch.log(torch.tensor(0.1, device=q.device)) - torch.log(scale) - (lo - y).clamp_min(0) / scale
    lp_hi = torch.log(torch.tensor(0.1, device=q.device)) - torch.log(scale) - (y - hi).clamp_min(0) / scale
    return torch.where(inside.any(-1), lp_in, torch.where(y < lo, lp_lo, lp_hi))
