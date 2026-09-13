"""Held-out evaluation: the leaderboard metrics plus calibration and shape, and the audit cells on synthetic laws.

    metrics = evaluate(policy, env_heldout, n=128)      # dict of means over held-out episodes
"""
from __future__ import annotations

import torch

from . import rewards as R
from .env import ContextAction


@torch.no_grad()
def evaluate(policy, env, n=128, batch=16, ref_state=None, actions_fn=None):
    """policy: a Forecaster. ref_state: a graft state to score as the reference (e.g. policy.frozen_state()) for skill."""
    agg = {k: [] for k in ("mase", "crps", "pinball", "mse", "coverage80", "width", "dir_acc", "interval", "newsvendor", "skill_vs_ref", "composite", "r2_cond")}
    seen = 0; pits = []; env.g.manual_seed(12345); torch.manual_seed(12345)     # also pins the token sampler's generation noise, so frozen-vs-policy comparisons are paired
    while seen < n:
        eps = env.reset(min(batch, n - seen)); acts = [actions_fn(e) for e in eps] if actions_fn else [ContextAction.native(e) for e in eps]
        q, y, info = env.forecast(policy, acts); ref_q = None
        if ref_state is not None:
            with policy.swap(ref_state): ref_q = env.forecast(policy, acts, ctx=info["ctx"])[0]
        agg["mase"].append(R.mase(q, y, info["history"])); agg["crps"].append(R.crps_from_quantiles(q, y)); agg["pinball"].append(R.pinball(q, y)); agg["mse"].append(R.mse(q, y))
        agg["coverage80"].append(R.coverage(q, y)); agg["width"].append(R.width(q)); agg["dir_acc"].append(R.directional_accuracy(q, y)); agg["interval"].append(R.interval_score(q, y)); agg["newsvendor"].append(R.newsvendor(q, y))
        agg["skill_vs_ref"].append(R.skill(R.crps_from_quantiles(q, y), R.crps_from_quantiles(ref_q, y)) if ref_q is not None else torch.zeros(len(eps)))
        agg["composite"].append(R.composite(q, y, ref_q))
        if info["cond_mean"] is not None:
            m = info["cond_mean"]; f = R.median(q); f = f - f.mean(1, keepdim=True); mc = m - m.mean(1, keepdim=True)
            agg["r2_cond"].append(1 - ((f - mc) ** 2).sum(1) / (mc ** 2).sum(1).clamp_min(1e-9))
        pits.append(R.pit(q, y).flatten())
        seen += len(eps)
    out = {k: float(torch.cat(v).mean()) for k, v in agg.items() if v}
    out.update({"cal_" + k: v for k, v in calibration_tests(torch.cat(agg["coverage80"]), torch.cat(pits)).items()})
    return out


def calibration_tests(cov_per_window, pit=None, nominal=0.8):
    """Statistical calibration checks instead of a fixed tolerance.
    binomial: two-sided exact-normal test that the mean 80 % coverage over windows equals the nominal level;
    pit_ks: Kolmogorov-Smirnov statistic of the PIT values against uniform (if PIT values are given).
    Returns p-values; a forecast 'stays calibrated' when it is not rejected at the same level as the reference is not."""
    import math
    c = torch.as_tensor(cov_per_window, dtype=torch.float32); n = c.numel(); m = float(c.mean())
    se = math.sqrt(nominal * (1 - nominal) / max(n, 1)); z = (m - nominal) / max(se, 1e-9)
    p_binom = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    out = {"coverage_mean": m, "coverage_z": z, "coverage_p": p_binom}
    if pit is not None:
        u = torch.as_tensor(pit, dtype=torch.float32).flatten().sort().values; k = u.numel()
        d = float(torch.maximum(torch.arange(1, k + 1) / k - u, u - torch.arange(0, k) / k).max())
        out["pit_ks"] = d; out["pit_ks_p"] = min(1.0, 2 * math.exp(-2 * k * d * d))
    return out
