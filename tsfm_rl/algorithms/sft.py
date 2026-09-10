"""M1: supervised post-training. Minimizes a differentiable loss (pinball by default; mse / mae / crps / interval)
on the gold future. The control every RL method must beat."""
from __future__ import annotations

import torch

from ..rewards import LOSSES
from .common import Run


def train_sft(run: Run, loss_name="pinball"):
    cfg, env, pol = run.cfg, run.env, run.policy; loss_fn = LOSSES[loss_name]
    for step in range(cfg.steps):
        env.reset(cfg.batch); q, y, _ = env.forecast(pol)
        loss = loss_fn(q, y).mean()
        run.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(), 1.0); run.opt.step()
        run.record(step, loss=loss, reward=env.reward(q.detach(), y).mean())
    run.save()
