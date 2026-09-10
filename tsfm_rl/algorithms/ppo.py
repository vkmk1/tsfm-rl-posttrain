"""M6: PPO on the forecaster with a learned value baseline.

One trajectory per episode is sampled from the quantile distribution; a value head on history statistics
predicts the expected reward; advantages are reward minus value; the clipped surrogate is optimized for a few
epochs over the batch with the trajectory's log-density under the current quantiles versus the old ones.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..rewards import sample_forecasts, logpdf, REWARDS, pinball
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
        env.reset(cfg.batch); actions = [ContextAction.native(e) for e in env.eps]
        inputs, roles, cpm, n_ctx = env.build(actions)
        q_ref = run.reference(inputs, roles, cpm)["logits"][:, 0, n_ctx - 1, :env.bank.H, :]
        with torch.no_grad():
            q_old, y, info = env.forecast(pol, actions); traj = sample_forecasts(q_old, 1, run.g)[0]
            r = REWARDS[cfg.reward](traj[..., None].expand(-1, -1, 9), y, q_ref)
            lp_old = logpdf(q_old, traj).mean(-1); feats = history_stats(info["history"])
        v = value(feats); loss_v = ((v - r) ** 2).mean(); opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        adv = (r - v.detach()); adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(cfg.ppo_epochs):
            q, _, _ = env.forecast(pol, actions); lp = logpdf(q, traj).mean(-1); ratio = torch.exp(lp - lp_old)
            surr = torch.minimum(ratio * adv, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv)
            loss = -surr.mean() + cfg.kl * ((q - q_ref) ** 2).mean()
            run.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(), 1.0); run.opt.step()
        run.record(step, loss=loss, value_loss=loss_v, r_mean=r.mean(), reward_q=env.reward(q.detach(), y, q_ref).mean(), pinball=pinball(q.detach(), y).mean())
    run.save()
