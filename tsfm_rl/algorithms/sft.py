"""M1: supervised post-training with the model's native differentiable loss (pinball by default; mse / mae / crps /
interval for quantile heads; token cross-entropy for Chronos-T5). The control every RL method must beat."""
from __future__ import annotations

import torch

from ..env import ContextAction
from .common import Run


def train_sft(run: Run, loss_name="pinball"):
    cfg, env, pol = run.cfg, run.env, run.policy
    for step in range(cfg.steps):
        env.reset(cfg.batch); ctx = env.build([ContextAction.native(e) for e in env.eps]); y = env.gold()
        with run.autocast(): loss, q = pol.sft_loss(ctx, y, loss_name)
        run.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(), 1.0); run.opt.step(); run.update_ema()
        kv = {"loss": loss}
        if q is not None: kv["reward"] = env.reward(env.detrend_fix(q.detach().float(), ctx), y).mean()
        run.record(step, **kv)
    run.save()
