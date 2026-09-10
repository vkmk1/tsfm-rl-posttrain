#!/usr/bin/env python3
"""Run one cell of the benchmark matrix (method x reward x environment) and evaluate on held-out episodes.

    python scripts/run_matrix.py --algo grpo --reward composite --bank synth --steps 300 --out runs/synth/grpo_composite
    python scripts/run_matrix.py --algo sft  --reward pinball   --bank csv:data/ETTh1.csv --steps 300 --out runs/etth1/sft_pinball
    python scripts/run_matrix.py --algo grpo --reward mse --anchor 0.1 ...     # GTN-R variant
    python scripts/run_matrix.py --algo grpo --reward mse --per_step ...       # TimeRFT-like credit
    python scripts/run_matrix.py --algo none --bank synth --out runs/synth/native   # M0: evaluate the frozen model
    (--tiny for the CPU test model)

Writes <out>/log.jsonl, <out>/policy.pt and <out>/eval.json (held-out metrics, plus the frozen reference's metrics).
"""
from __future__ import annotations

import argparse, dataclasses, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm_rl.model import load_model, tiny_model  # noqa: E402
from tsfm_rl.data import EpisodeBank  # noqa: E402
from tsfm_rl.env import ForecastEnv, base_fn  # noqa: E402
from tsfm_rl.evaluate import evaluate  # noqa: E402
from tsfm_rl.algorithms.common import TrainConfig, Run  # noqa: E402
from tsfm_rl.algorithms.sft import train_sft  # noqa: E402
from tsfm_rl.algorithms.grpo import train_grpo  # noqa: E402
from tsfm_rl.algorithms.ppo import train_ppo  # noqa: E402
from tsfm_rl.algorithms.preference import train_preference, train_context  # noqa: E402


def make_bank(spec, T, H, n, seed):
    if spec == "synth": return EpisodeBank.synthetic(n, T, H, seed=seed)
    if spec.startswith("csv:"): return EpisodeBank.from_csv(spec[4:], T, H, n, seed=seed)
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="grpo", choices=["none", "sft", "grpo", "ppo", "preference", "context", "context_sft"])
    ap.add_argument("--reward", default="composite"); ap.add_argument("--bank", default="synth"); ap.add_argument("--episodes", type=int, default=640)
    ap.add_argument("--T", type=int, default=256); ap.add_argument("--H", type=int, default=64); ap.add_argument("--heldout", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--K", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 11, 12, 13, 14]); ap.add_argument("--lora_r", type=int, default=8); ap.add_argument("--head_adapter", action="store_true")
    ap.add_argument("--anchor", type=float, default=0.0); ap.add_argument("--kl", type=float, default=0.05); ap.add_argument("--per_step", action="store_true")
    ap.add_argument("--tiny", action="store_true"); ap.add_argument("--model", default="google/timesfm-3.0-pytorch"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--log_every", type=int, default=10)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = tiny_model() if a.tiny else load_model(a.model, device)
    bank = make_bank(a.bank, a.T, a.H, a.episodes, a.seed); train, held = bank.split(1 - a.heldout / max(len(bank), 1))
    env = ForecastEnv(train, a.reward, device, a.seed); env_h = ForecastEnv(held, a.reward, device, a.seed + 1)
    cfg = TrainConfig(algo=a.algo, reward=a.reward, steps=a.steps, batch=a.batch, K=a.K, lr=a.lr, layers=tuple(a.layers), lora_r=a.lora_r, head_adapter=a.head_adapter,
                      anchor=a.anchor, kl=a.kl, per_step=a.per_step, seed=a.seed, log_every=a.log_every)
    os.makedirs(a.out, exist_ok=True)
    ref_metrics = evaluate(base_fn(base), env_h, n=a.heldout)                     # frozen reference on the held-out set (before any graft)
    if a.algo == "none":
        json.dump({"reference": ref_metrics, "policy": ref_metrics, "cfg": dataclasses.asdict(cfg)}, open(os.path.join(a.out, "eval.json"), "w"), indent=1); print(json.dumps(ref_metrics, indent=1)); return
    run = Run(cfg, base, env, a.out)
    print(f"{a.algo} · reward {a.reward} · bank {a.bank} · trainable {sum(p.numel() for p in run.policy.trainable_parameters())/1e6:.3f}M", flush=True)
    {"sft": lambda: train_sft(run, a.reward if a.reward in ("mse", "mae", "pinball", "crps", "interval") else "pinball"), "grpo": lambda: train_grpo(run), "ppo": lambda: train_ppo(run),
     "preference": lambda: train_preference(run), "context": lambda: train_context(run), "context_sft": lambda: train_context(run, sft=True)}[a.algo]()
    pol_metrics = evaluate(run.policy, env_h, n=a.heldout, ref_policy=run.reference) if a.algo not in ("context", "context_sft") else ref_metrics
    res = {"reference": ref_metrics, "policy": pol_metrics, "cfg": dataclasses.asdict(cfg)}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print("held-out:", {k: (round(ref_metrics[k], 4), round(pol_metrics[k], 4)) for k in ("mase", "crps", "coverage80", "dir_acc")})


if __name__ == "__main__":
    main()
