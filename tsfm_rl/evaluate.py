"""Held-out evaluation: the leaderboard metrics plus calibration and shape, and the audit cells on synthetic laws.

    metrics = evaluate(policy, env_heldout, n=128)      # dict of means over held-out episodes
"""
from __future__ import annotations

import torch

from . import rewards as R
from .env import ContextAction


@torch.no_grad()
def evaluate(policy, env, n=128, batch=16, ref_policy=None, actions_fn=None):
    agg = {k: [] for k in ("mase", "crps", "pinball", "mse", "coverage80", "width", "dir_acc", "interval", "newsvendor", "skill_vs_ref", "composite", "r2_cond")}
    seen = 0; env.g.manual_seed(12345)
    while seen < n:
        eps = env.reset(min(batch, n - seen)); acts = [actions_fn(e) for e in eps] if actions_fn else [ContextAction.native(e) for e in eps]
        q, y, info = env.forecast(policy, acts); ref_q = env.forecast(ref_policy, acts)[0] if ref_policy is not None else None
        agg["mase"].append(R.mase(q, y, info["history"])); agg["crps"].append(R.crps_from_quantiles(q, y)); agg["pinball"].append(R.pinball(q, y)); agg["mse"].append(R.mse(q, y))
        agg["coverage80"].append(R.coverage(q, y)); agg["width"].append(R.width(q)); agg["dir_acc"].append(R.directional_accuracy(q, y)); agg["interval"].append(R.interval_score(q, y)); agg["newsvendor"].append(R.newsvendor(q, y))
        agg["skill_vs_ref"].append(R.skill(R.crps_from_quantiles(q, y), R.crps_from_quantiles(ref_q, y)) if ref_q is not None else torch.zeros(len(eps)))
        agg["composite"].append(R.composite(q, y, ref_q))
        if info["cond_mean"] is not None:
            m = info["cond_mean"]; f = R.median(q); f = f - f.mean(1, keepdim=True); mc = m - m.mean(1, keepdim=True)
            agg["r2_cond"].append(1 - ((f - mc) ** 2).sum(1) / (mc ** 2).sum(1).clamp_min(1e-9))
        seen += len(eps)
    return {k: float(torch.cat(v).mean()) for k, v in agg.items() if v}
