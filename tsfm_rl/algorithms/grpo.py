"""M2 / M3 / M4 / M7: group-relative policy optimization on the forecaster.

For each episode, K trajectories are sampled from the quantile head's predictive distribution; each is scored by
the reward (a point forecast is scored by placing the trajectory at every quantile); advantages are
group-relative (mean/std over the K samples); the policy gradient uses the differentiable log-density of each
trajectory under the quantile function. Options:
    anchor > 0     GTN-R-like: extra term pulling the log-density of the ground truth up (mass toward the truth)
    kl > 0         penalty on quantile drift from the frozen reference
    per_step       TimeRFT-like: per-step rewards and advantages (credit along the horizon)
With reward="mse" and anchor=0 this is TS-GRPO; with anchor>0 it is the GTN-R variant; with reward="composite"
it is the method under test.
"""
from __future__ import annotations

import torch

from ..rewards import sample_forecasts, logpdf, REWARDS, composite, mse, pinball
from ..env import ContextAction
from .common import Run


def _reward_traj(env, traj, y, ref_q, per_step, name):
    q_like = traj[..., None].expand(-1, -1, 9)               # point trajectory as a degenerate quantile set
    if per_step:
        if name == "composite": return composite(q_like, y, ref_q, per_step=True)
        if name in ("mse",): return -mse(q_like, y, per_step=True)
        return -pinball(q_like, y, per_step=True)
    return REWARDS[name](q_like, y, ref_q)


def train_grpo(run: Run):
    cfg, env, pol = run.cfg, run.env, run.policy
    for step in range(cfg.steps):
        env.reset(cfg.batch)
        inputs, roles, cpm, n_ctx = env.build([ContextAction.native(e) for e in env.eps])
        q_ref = run.reference(inputs, roles, cpm)["logits"][:, 0, n_ctx - 1, :env.bank.H, :]
        q, y, info = env.forecast(pol)
        with torch.no_grad():
            samples = sample_forecasts(q.detach(), cfg.K, run.g)                                  # (K, B, H)
            r = torch.stack([_reward_traj(env, samples[k], y, q_ref, cfg.per_step, cfg.reward) for k in range(cfg.K)])   # (K, B[, H])
            adv = (r - r.mean(0, keepdim=True)) / (r.std(0, keepdim=True) + 1e-6)
        lp = torch.stack([logpdf(q, samples[k]) for k in range(cfg.K)])                            # (K, B, H)
        loss_pg = -(adv * (lp if cfg.per_step else lp.mean(-1))).mean()
        loss = loss_pg
        if cfg.anchor > 0: loss = loss - cfg.anchor * logpdf(q, y).mean()                          # GTN-R-like: raise density at the truth
        if cfg.kl > 0: loss = loss + cfg.kl * ((q - q_ref) ** 2).mean()
        run.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(), 1.0); run.opt.step()
        run.record(step, loss=loss, pg=loss_pg, r_mean=r.mean(), r_best=r.max(0).values.mean(), reward_q=env.reward(q.detach(), y, q_ref).mean(), pinball=pinball(q.detach(), y).mean())
    run.save()
