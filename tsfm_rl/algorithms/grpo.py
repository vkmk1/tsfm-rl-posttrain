"""M2 / M3 / M4 / M7 and the estimator variants of D1: group policy optimization on the forecaster.

For each episode, K trajectories are sampled from a quantile CDF (the policy's, the EMA policy's, or a mix); each is
scored by the reward; advantages are group-relative; the policy gradient uses the differentiable log-density of each
trajectory under the policy's quantiles. Options (TrainConfig):
    adv_norm     std (GRPO: divides by the group std, which for a forecaster is the predictive spread; Turtel 2505.17989 and
                 Bereket 2508.11800 show this alone induces overconfidence) | loo (RLOO) | remax (minus the median forecast's
                 reward) | none (minus the group mean)
    set_reward   score the K samples as a set with the fair sample-CRPS (keeps the -0.5 E|X-X'| spread term, so the optimum
                 is the calibrated distribution); baseline = the same score for the reference's samples; one advantage per episode
    sampler      policy | ema | mix: off-policy samples carry truncated importance weights min(p_policy / p_behaviour, 2)
    kl_dir       sq: squared quantile shift from the anchor (quantile heads) | forward: E_{y~anchor}[-log p_policy(y)], mass-covering
                 | k3: KL(policy || anchor) at the sampled trajectories (the LLM estimator; works for the token sampler)
    kl_ref       base (frozen model) | ema (slow trust region)
    anchor > 0   GTN-R-like: raise the log-density of the ground truth
    per_step     TimeRFT-like per-horizon credit
With reward="mse", adv_norm="std", sampler="policy", kl_dir="sq": TS-GRPO (M2). With anchor>0: GTN-R (M3). With per_step: M4.
With reward="composite": M7. With adv_norm="loo", set_reward, sampler="ema", kl_dir="forward": the D1 cell.
"""
from __future__ import annotations

import torch

from ..rewards import sample_forecasts, REWARDS, composite, mse, pinball, crps_from_samples, median
from ..env import ContextAction
from .common import Run


def _reward_traj(env, traj, y, ref_q, per_step, name):
    q_like = traj[..., None].expand(-1, -1, 9)               # point trajectory as a degenerate quantile set
    if per_step:
        if name == "composite": return composite(q_like, y, ref_q, per_step=True)
        if name in ("mse",): return -mse(q_like, y, per_step=True)
        return -pinball(q_like, y, per_step=True)
    return REWARDS[name](q_like, y, ref_q)


def _advantages(r, cfg, q=None, env=None, y=None, ref_q=None):
    """r: (K, B[, H]) per-sample rewards -> advantages of the same shape."""
    if cfg.adv_norm == "std": return (r - r.mean(0, keepdim=True)) / (r.std(0, keepdim=True) + 1e-6)
    if cfg.adv_norm == "loo":
        K = r.shape[0]; return r - (r.sum(0, keepdim=True) - r) / max(K - 1, 1)
    if cfg.adv_norm == "remax":
        base = _reward_traj(env, median(q), y, ref_q, cfg.per_step, cfg.reward); return r - base[None]
    return r - r.mean(0, keepdim=True)


def train_grpo(run: Run):
    cfg, env, pol = run.cfg, run.env, run.policy
    for step in range(cfg.steps):
        env.reset(cfg.batch); actions = [ContextAction.native(e) for e in env.eps]
        ctx = env.build(actions); y = env.gold(); H = env.bank.H
        # ---- everything that swaps weights runs before the policy graph exists -----------------------------------
        q_ref = run.reference_q(ctx)                                                                  # frozen model: reward baseline / skill
        q_anchor = q_ref if cfg.kl_ref == "base" else run.anchor_q(ctx)
        with torch.no_grad():
            if cfg.sampler == "policy": beh = None
            elif cfg.sampler == "ema": beh = run.sample_under(run.ema_state, ctx, cfg.K)
            else: beh = run.sample_under(run.ema_state, ctx, cfg.K - cfg.K // 2)                       # mix: the rest from the policy below
            if cfg.kl > 0 and cfg.kl_dir == "forward": y_anchor = run.sample_under(run.anchor_state(), ctx, cfg.K)[0]
            if cfg.set_reward: ref_samples = run.sample_under(run.frozen_state, ctx, cfg.K)[0]
        # ---- policy forward ------------------------------------------------------------------------------------------
        with run.autocast(): q = env.detrend_fix(pol.quantiles(ctx), ctx).float()                     # grad for quantile heads; empirical for the token sampler
        with torch.no_grad():
            if beh is None: samples, lp_beh = pol.sample(ctx, cfg.K, run.g, q=q)
            elif cfg.sampler == "ema": samples, lp_beh = beh
            else:
                s1, l1 = pol.sample(ctx, cfg.K // 2, run.g, q=q); samples = torch.cat([s1, beh[0]]); lp_beh = torch.cat([l1, beh[1]])
            lp_beh = lp_beh.float()
            # ---- rewards and advantages ------------------------------------------------------------------------------
            if cfg.set_reward:
                r_set = -crps_from_samples(samples, y); r_base = -crps_from_samples(ref_samples, y)     # (B,) proper score of the set
                adv_ep = r_set - r_base
                if cfg.adv_norm == "std": adv_ep = (adv_ep - adv_ep.mean()) / (adv_ep.std() + 1e-6)
                elif cfg.adv_norm == "loo": B = adv_ep.shape[0]; adv_ep = adv_ep - (adv_ep.sum() - adv_ep) / max(B - 1, 1)
                adv = adv_ep[None].expand(cfg.K, -1); r = r_set[None].expand(cfg.K, -1)
            else:
                r = torch.stack([_reward_traj(env, samples[k], y, q_ref, cfg.per_step, cfg.reward) for k in range(cfg.K)])   # (K, B[, H])
                adv = _advantages(r, cfg, q.detach(), env, y, q_ref)
        # ---- policy gradient -------------------------------------------------------------------------------------
        with run.autocast(): lp = pol.logprob(ctx, samples, q=q if not pol.stochastic else None).float()   # (K, B, H) under the policy
        w = torch.ones_like(lp.mean(-1)) if cfg.sampler == "policy" else torch.exp((lp - lp_beh).mean(-1)).detach().clamp(max=2.0)
        loss_pg = -(adv * w[..., None] * lp).mean() if (cfg.per_step and not cfg.set_reward) else -(adv * w * lp.mean(-1)).mean()
        loss = loss_pg
        if cfg.anchor > 0: loss = loss - cfg.anchor * pol.logprob(ctx, y[None], q=q if not pol.stochastic else None).mean()   # GTN-R-like: raise density at the truth
        if cfg.kl > 0:
            if cfg.kl_dir == "forward": loss = loss - cfg.kl * pol.logprob(ctx, y_anchor, q=q if not pol.stochastic else None).float().mean()   # mass-covering
            elif cfg.kl_dir == "k3":
                lp_a = run.logprob_under(run.anchor_state(), ctx, samples); d = lp_a - lp; loss = loss + cfg.kl * (torch.exp(d) - d - 1).mean()
            else:
                assert not pol.stochastic, "kl_dir=sq needs a differentiable quantile head; use forward or k3"
                loss = loss + cfg.kl * ((q - q_anchor) ** 2).mean()
        run.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(), 1.0); run.opt.step(); run.update_ema()
        with torch.no_grad():
            qd = q.detach()
            run.record(step, loss=loss, pg=loss_pg, r_mean=r.mean(), r_best=r.max(0).values.mean(), reward_q=env.reward(qd, y, q_ref).mean(), pinball=pinball(qd, y).mean(),
                       width=(qd[..., 8] - qd[..., 0]).mean(), is_w=w.mean(),
                       group_std=r.float().std(0).mean() if r.shape[0] > 1 else torch.tensor(0.0), zero_std_frac=(r.float().std(0) < 1e-4).float().mean() if r.shape[0] > 1 else torch.tensor(0.0))
    run.save()
