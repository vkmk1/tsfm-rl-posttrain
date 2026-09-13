"""M6: PPO on the forecaster with a learned value baseline.

One trajectory per episode is sampled from the forecaster (quantile CDF or token sampler); a value head on history
statistics predicts the expected reward; advantages are reward minus value; the clipped surrogate is optimized for a
few epochs over the batch with the trajectory's log-probability under the current weights versus the old ones.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..rewards import REWARDS, pinball
from ..env import ContextAction
from .common import Run


def history_stats(h):
    z = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
    ac = torch.stack([(z[:, l:] * z[:, :-l]).mean(1) for l in range(1, 9)], 1)
    return torch.cat([ac, h.std(1, keepdim=True), (h[:, -1] - h[:, 0])[:, None] / (h.std(1, keepdim=True) + 1e-6)], 1)


class Value(nn.Module):
    def __init__(self, n=10, hidden=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, f): return self.net(f).squeeze(-1)


def train_ppo(run: Run):
    cfg, env, pol = run.cfg, run.env, run.policy
    value = Value().to(env.device); opt_v = torch.optim.Adam(value.parameters(), lr=1e-3)
    for step in range(cfg.steps):
        env.reset(cfg.batch); actions = [ContextAction.native(e) for e in env.eps]; ctx = env.build(actions); y = env.gold()
        q_ref = run.anchor_q(ctx); info = env.info(ctx)
        with torch.no_grad():
            if cfg.kl > 0 and cfg.kl_dir == "forward": y_anchor = run.sample_under(run.anchor_state(), ctx, cfg.K)[0]
            q_old = env.detrend_fix(pol.quantiles(ctx), ctx).float(); traj, lp_old = pol.sample(ctx, 1, run.g, q=q_old); traj = traj[0]; lp_old = lp_old[0].float().mean(-1)
            r = REWARDS[cfg.reward](traj[..., None].expand(-1, -1, 9), y, q_ref); feats = history_stats(info["history"])
        v = value(feats); loss_v = ((v - r) ** 2).mean(); opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        adv = (r - v.detach())
        if cfg.adv_norm == "std": adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(cfg.ppo_epochs):
            with run.autocast():
                q = None if pol.stochastic else env.detrend_fix(pol.quantiles(ctx), ctx).float()
                lp = pol.logprob(ctx, traj[None], q=q)[0].float().mean(-1); ratio = torch.exp(lp - lp_old)
            surr = torch.minimum(ratio * adv, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv); loss = -surr.mean()
            if cfg.kl > 0:
                if cfg.kl_dir == "forward": loss = loss - cfg.kl * pol.logprob(ctx, y_anchor, q=q).float().mean()
                elif cfg.kl_dir == "k3":
                    lp_a = run.logprob_under(run.anchor_state(), ctx, traj[None])[0].mean(-1); d = lp_a - lp; loss = loss + cfg.kl * (torch.exp(d) - d - 1).mean()
                else: assert q is not None, "kl_dir=sq needs a quantile head"; loss = loss + cfg.kl * ((q - q_ref) ** 2).mean()
            run.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(), 1.0); run.opt.step(); run.update_ema()
        qd = q_old if q is None else q.detach()
        run.record(step, loss=loss, value_loss=loss_v, r_mean=r.mean(), reward_q=env.reward(qd, y, q_ref).mean(), pinball=pinball(qd, y).mean())
    run.save()
