"""M5 and the LeReT analog.

train_preference   trajectory-pair preference optimization on the forecaster (TPO analog): sample K trajectories,
                   rank them by reward, form pairs, IPO on the log-density ratio versus the frozen reference.
train_context      a small policy chooses among the environment's proposed context actions for a frozen forecaster;
                   IPO on within-episode preference pairs (or best-of-K cross-entropy with sft=True).
"""
from __future__ import annotations

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rewards import REWARDS, pinball
from ..env import ContextAction
from .common import Run


def train_preference(run: Run, tau=0.05):
    cfg, env, pol = run.cfg, run.env, run.policy
    for step in range(cfg.steps):
        env.reset(cfg.batch); actions = [ContextAction.native(e) for e in env.eps]
        ctx = env.build(actions); y = env.gold(); q_ref = run.reference_q(ctx)
        with run.autocast(): q = env.detrend_fix(pol.quantiles(ctx), ctx).float()
        with torch.no_grad():
            samples, _ = pol.sample(ctx, cfg.K, run.g, q=q); r = torch.stack([REWARDS[cfg.reward](samples[k][..., None].expand(-1, -1, 9), y, q_ref) for k in range(cfg.K)])
        lp_ref = run.logprob_under(run.frozen_state, ctx, samples).mean(-1)                       # (K, B)
        with run.autocast(): lp = pol.logprob(ctx, samples, q=q if not pol.stochastic else None).float().mean(-1)
        losses = []
        for i, j in itertools.combinations(range(cfg.K), 2):
            sgn = torch.sign(r[i] - r[j]); h = (lp[i] - lp_ref[i]) - (lp[j] - lp_ref[j])
            losses.append(((sgn * h - 0.5 / tau) ** 2)[sgn != 0])
        loss = torch.cat(losses).mean()
        if cfg.kl > 0 and not pol.stochastic: loss = loss + cfg.kl * ((q - q_ref) ** 2).mean()
        run.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(), 1.0); run.opt.step(); run.update_ema()
        run.record(step, loss=loss, r_mean=r.mean(), reward_q=env.reward(q.detach(), y, q_ref).mean(), pinball=pinball(q.detach(), y).mean())
    run.save()


class ContextPolicy(nn.Module):
    def __init__(self, n_feat=13, hidden=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, f): return self.net(f).squeeze(-1)


def proposal_features(env, proposals):
    feats = []
    for e, props in zip(env.eps, proposals):
        rows = []
        for a in props:
            if a.rows:
                z = (e.cov_hist[a.rows] - e.cov_hist[a.rows].mean(1, keepdim=True)) / (e.cov_hist[a.rows].std(1, keepdim=True) + 1e-6); t = (e.target - e.target.mean()) / (e.target.std() + 1e-6)
                xc = torch.stack([(t[l:] * z[:, :z.shape[1] - l]).mean(1).mean() for l in range(0, 8)]); kf = e.known_future[a.rows].float().mean()
            else: xc = torch.zeros(8); kf = torch.tensor(0.0)
            rows.append(torch.cat([xc, torch.tensor([len(a.rows), kf, float(a.estimator_row), float(a.detrend), a.hist_len / 256])]))
        feats.append(torch.stack(rows))
    return torch.stack(feats)


def train_context(run: Run, sft=False, tau=0.05):
    cfg, env, pol = run.cfg, run.env, run.policy
    ctx = ContextPolicy().to(env.device); opt = torch.optim.Adam(ctx.parameters(), lr=1e-3)
    for step in range(cfg.steps):
        env.reset(cfg.batch); proposals = [env.propose(e) for e in env.eps]; P = min(len(p) for p in proposals); proposals = [p[:P] for p in proposals]
        feats = proposal_features(env, proposals).to(env.device); rewards = torch.zeros(cfg.batch, P, device=env.device)
        with torch.no_grad(), pol.swap(run.frozen_state):
            for j in range(P):
                q, y, _ = env.forecast(pol, [p[j] for p in proposals]); rewards[:, j] = env.reward(q, y)
        logits = ctx(feats); logp = F.log_softmax(logits, -1)
        if sft: loss = F.cross_entropy(logits, rewards.argmax(1))
        else:
            diff = logp[:, :, None] - logp[:, None, :]; pref = rewards[:, :, None] - rewards[:, None, :]; mask = (pref.abs() > 1e-3).float()
            loss = (mask * ((torch.sign(pref) * diff - 0.5 / tau) ** 2)).sum() / mask.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step(); chosen = logits.argmax(1)
        run.record(step, loss=loss, r_native=rewards[:, 0].mean(), r_chosen=rewards.gather(1, chosen[:, None]).mean(), r_oracle=rewards.max(1).values.mean())
    torch.save(ctx.state_dict(), f"{run.out}/context_policy.pt")
