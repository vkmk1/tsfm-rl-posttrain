"""Training the agent around a frozen oracle.

    train_agent(run, stage="sft" | "rl" | "sft_rl", reward="crps" | "impratio" | "newsvendor_tau" | "random")
Stage 1 (sft): hindsight labels, cross-entropy on the best (proposal, trust) action; windows where "native, full trust" wins are
fallback examples by construction (that action is index 0 of the joint space with w = 1 at the last grid point).
Stage 2 (rl): full-information policy optimisation: loss = -E_{a~pi}[R(a) - R(native)] computed exactly over all actions
(no sampling, no baseline variance); optional entropy bonus. `grpo` samples actions with a leave-one-out baseline instead,
for action spaces too large to enumerate.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from ..agent import TrustAgent, Fallback, enumerate_actions, deliver
from ..rewards import crps_from_quantiles
from .common import Run


def train_agent(run: Run, stage="sft_rl", reward="crps", K=8, entropy=0.01, sampled=False):
    cfg, env, fc = run.cfg, run.env, run.policy
    agent = TrustAgent().to(env.device); opt = torch.optim.Adam(agent.parameters(), lr=1e-3); fb = Fallback()
    stages = {"sft": [("sft", cfg.steps)], "rl": [("rl", cfg.steps)], "sft_rl": [("sft", cfg.steps // 2), ("rl", cfg.steps - cfg.steps // 2)]}[stage]
    step = 0
    with fc.swap(fc.frozen_state()):                                                      # the oracle is frozen throughout
        for mode, n in stages:
            for _ in range(n):
                env.reset(cfg.batch); props = [env.propose(e, K=K) for e in env.eps]
                q_all, q_fb, R, feats, props = enumerate_actions(env, fc, props, fb, reward=reward)          # R (B, A)
                logits = agent(feats); logp = F.log_softmax(logits, -1); W = len(agent.w_grid); native = W - 1   # action 0*W + (W-1): native context, w = 1
                if mode == "sft": loss = F.cross_entropy(logits, R.argmax(1))
                else:
                    adv = R - R[:, native:native + 1]                                        # improvement over the native oracle forecast
                    if sampled:
                        a = torch.multinomial(logp.exp(), cfg.K, replacement=True); r = adv.gather(1, a)     # (B,K)
                        base = (r.sum(1, keepdim=True) - r) / max(cfg.K - 1, 1); loss = -((r - base).detach() * logp.gather(1, a)).mean()
                    else: loss = -(logp.exp() * adv).sum(1).mean()                           # exact expected improvement
                    loss = loss - entropy * (-(logp.exp() * logp).sum(1)).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    a = logits.argmax(1); j, w = agent.decode(a, q_all.shape[0]); d = deliver(q_all, q_fb, j, w); y = env.gold()
                    run.record(step, stage=mode, loss=loss, r_chosen=R.gather(1, a[:, None]).mean(), r_native=R[:, native].mean(), r_oracle=R.max(1).values.mean(),
                               crps_delivered=crps_from_quantiles(d, y).mean(), crps_native=crps_from_quantiles(q_all[0], y).mean(), w_mean=w.mean(), p_native=(j == 0).float().mean())
                step += 1
    torch.save({"agent": agent.state_dict(), "cfg": {"stage": stage, "reward": reward, "K": K}}, os.path.join(run.out, "agent.pt"))
    return agent, fb


def best_fixed_context(run: Run, K=8, n_batches=16):
    """The regret reference: the single proposal index (with w = 1) that scores best on training windows, in hindsight."""
    env, fc = run.env, run.policy; fb = Fallback(); tot = None
    with torch.no_grad(), fc.swap(fc.frozen_state()):
        for _ in range(n_batches):
            env.reset(run.cfg.batch); props = [env.propose(e, K=K) for e in env.eps]
            _, _, R, _, _ = enumerate_actions(env, fc, props, fb, reward="crps"); W = len(TrustAgent().w_grid)
            r = R.reshape(R.shape[0], -1, W)[:, :, W - 1].sum(0); tot = r if tot is None else tot + r
    return int(tot.argmax())
