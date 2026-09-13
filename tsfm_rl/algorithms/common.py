"""Shared training scaffolding: config, logging, checkpoint, the frozen reference forecaster, EMA policy, AMP."""
from __future__ import annotations

import contextlib, dataclasses, json, os, time
import torch

from ..env import ForecastEnv


@dataclasses.dataclass
class TrainConfig:
    algo: str = "grpo"                 # sft | grpo | ppo | preference | context
    reward: str = "composite"          # mse | mae | pinball | crps | skill | interval | newsvendor | composite
    steps: int = 300
    batch: int = 8
    K: int = 4                         # sampled trajectories per episode (grpo / preference / ppo groups)
    lr: float = 1e-4
    layers: tuple = (10, 11, 12, 13, 14)
    lora_r: int = 8
    head_adapter: bool = False
    anchor: float = 0.0                # GTN-R-like: penalty pulling sampled mass toward the ground-truth neighborhood
    kl: float = 0.05                   # penalty on the shift from the reference
    kl_dir: str = "sq"                 # sq: squared quantile shift | forward: E_{y~ref}[-log p_policy(y)], mass-covering
    kl_ref: str = "base"               # base: frozen model | ema: the EMA policy (a slow trust region instead of an anchor)
    per_step: bool = False             # TimeRFT-like per-horizon-step credit
    adv_norm: str = "std"              # std: group mean/std (GRPO) | loo: leave-one-out mean (RLOO) | remax: minus the median forecast's reward | none: minus group mean
    set_reward: bool = False           # score the K samples as a set with the fair CRPS (proper), one advantage per episode
    sampler: str = "policy"            # policy | ema | mix: which weights generate the K samples (off-policy samples get truncated importance weights)
    ema_decay: float = 0.0             # 0 = no EMA; else EMA of the trainable state after every step
    eval_ema: bool = False             # evaluate / save the EMA weights as the policy
    ppo_clip: float = 0.2
    ppo_epochs: int = 2
    amp: bool = False                  # bf16 autocast on CUDA
    save_every: int = 0                # checkpoint every N steps (0 = only at the end)
    seed: int = 0
    log_every: int = 10


class Run:
    def __init__(self, cfg: TrainConfig, forecaster, env: ForecastEnv, out: str):
        self.cfg, self.env, self.out = cfg, env, out
        os.makedirs(out, exist_ok=True); self.log = open(os.path.join(out, "log.jsonl"), "a"); self.t0 = time.time()
        torch.manual_seed(cfg.seed); self.g = torch.Generator().manual_seed(cfg.seed)
        self.policy = forecaster                                                                  # a tsfm_rl.forecasters.Forecaster with zero grafts
        self.opt = torch.optim.AdamW(self.policy.trainable_parameters(), lr=cfg.lr, weight_decay=0.01)
        self.frozen_state = self.policy.frozen_state()
        self.ema_state = {k: v.detach().clone() for k, v in self.policy.state().items()} if cfg.ema_decay > 0 else None
        self.use_amp = bool(cfg.amp) and env.device.type == "cuda"

    # ---- weight swapping (object substitution; safe at any point of the step) --------------------------------------
    def swap(self, state): return self.policy.swap(state)

    def anchor_state(self):
        if self.cfg.kl_ref == "ema": assert self.ema_state is not None, "ema_decay must be > 0"; return self.ema_state
        return self.frozen_state

    def quantiles_under(self, state, ctx):
        with torch.no_grad(), self.swap(state), self.autocast(): return self.env.detrend_fix(self.policy.quantiles(ctx), ctx).float()

    def sample_under(self, state, ctx, K):
        with torch.no_grad(), self.swap(state), self.autocast(): return self.policy.sample(ctx, K, self.g)

    def logprob_under(self, state, ctx, samples):
        with torch.no_grad(), self.swap(state), self.autocast(): return self.policy.logprob(ctx, samples).float()

    def reference_q(self, ctx): return self.quantiles_under(self.frozen_state, ctx)
    def anchor_q(self, ctx): return self.quantiles_under(self.anchor_state(), ctx)

    def update_ema(self):
        if self.ema_state is None: return
        d = self.cfg.ema_decay
        with torch.no_grad():
            for k, v in self.policy.state().items(): self.ema_state[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.use_amp else contextlib.nullcontext()

    # ---- bookkeeping -------------------------------------------------------------------------------------------------
    def record(self, step, **kv):
        rec = {"step": step, **{k: (float(v.detach()) if torch.is_tensor(v) else v) for k, v in kv.items()}, "elapsed": time.time() - self.t0}
        self.log.write(json.dumps(rec) + "\n"); self.log.flush()
        if step % self.cfg.log_every == 0 or step == self.cfg.steps - 1:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
        if self.cfg.save_every and step and step % self.cfg.save_every == 0: self.save(tag=f"step{step}")

    def save(self, tag=None):
        obj = {"state": self.policy.state(), "ema_state": self.ema_state, "cfg": dataclasses.asdict(self.cfg)}
        torch.save(obj, os.path.join(self.out, "policy.pt" if tag is None else f"policy_{tag}.pt"))

    def eval_state(self):
        """The weights to evaluate: EMA if requested, else the live policy."""
        return self.ema_state if (self.cfg.eval_ema and self.ema_state is not None) else None
