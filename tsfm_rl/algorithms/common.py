"""Shared training scaffolding: config, logging, checkpoint, the frozen reference forecaster."""
from __future__ import annotations

import dataclasses, json, os, time
import torch

from ..adapters import Policy
from ..env import ForecastEnv, base_fn


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
    kl: float = 0.05                   # penalty on the quantile shift from the frozen reference
    per_step: bool = False             # TimeRFT-like per-horizon-step credit
    ppo_clip: float = 0.2
    ppo_epochs: int = 2
    seed: int = 0
    log_every: int = 10


class Run:
    def __init__(self, cfg: TrainConfig, base, env: ForecastEnv, out: str):
        self.cfg, self.base, self.env, self.out = cfg, base, env, out
        os.makedirs(out, exist_ok=True); self.log = open(os.path.join(out, "log.jsonl"), "a"); self.t0 = time.time()
        torch.manual_seed(cfg.seed); self.g = torch.Generator().manual_seed(cfg.seed)
        self.ref = base_fn(base)                                   # frozen reference (evaluated before grafts act: grafts are zero at init and the ref call runs with policy context off)
        self.policy = Policy(base, layers=cfg.layers, lora_r=cfg.lora_r, head_adapter=cfg.head_adapter).to(env.device)
        self.opt = torch.optim.AdamW(self.policy.trainable_parameters(), lr=cfg.lr, weight_decay=0.01)
        self.ref_state = {k: v.detach().clone() for k, v in self.policy.state().items()}

    def reference(self, inputs, roles, cpm):
        """Frozen-model output: run the policy with its grafts temporarily zeroed (LoRA B and head alpha)."""
        saved = {k: v.detach().clone() for k, v in self.policy.state().items()}
        with torch.no_grad():
            for k, v in self.policy.state_dict().items():
                if k.endswith(".B") or k.endswith("head.alpha"): v.zero_()
            out = self.policy(inputs, roles, cpm)
            self.policy.load_state_dict({**self.policy.state_dict(), **saved}, strict=False)
        return out

    def record(self, step, **kv):
        rec = {"step": step, **{k: (float(v) if torch.is_tensor(v) else v) for k, v in kv.items()}, "elapsed": time.time() - self.t0}
        self.log.write(json.dumps(rec) + "\n"); self.log.flush()
        if step % self.cfg.log_every == 0 or step == self.cfg.steps - 1:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

    def save(self):
        torch.save({"state": self.policy.state(), "cfg": dataclasses.asdict(self.cfg)}, os.path.join(self.out, "policy.pt"))
