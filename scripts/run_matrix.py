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
from tsfm_rl.forecasters import make_forecaster, MODELS  # noqa: E402
from tsfm_rl.data import EpisodeBank  # noqa: E402
from tsfm_rl.env import ForecastEnv  # noqa: E402
from tsfm_rl.evaluate import evaluate  # noqa: E402
from tsfm_rl.algorithms.common import TrainConfig, Run  # noqa: E402
from tsfm_rl.algorithms.sft import train_sft  # noqa: E402
from tsfm_rl.algorithms.grpo import train_grpo  # noqa: E402
from tsfm_rl.algorithms.ppo import train_ppo  # noqa: E402
from tsfm_rl.algorithms.preference import train_preference, train_context  # noqa: E402
from tsfm_rl.algorithms.agent import train_agent, best_fixed_context  # noqa: E402
from tsfm_rl.agent import AgentForecaster, TrustAgent, Fallback  # noqa: E402


def make_bank(spec, T, H, n, seed, region=(0.0, 1.0), cols=None):
    if spec == "synth": return EpisodeBank.synthetic(n, T, H, seed=seed)
    if spec.startswith("csv:"): return EpisodeBank.from_csv(spec[4:], T, H, n, seed=seed, region=region, cols=cols)
    raise ValueError(spec)


def walkforward_regions(N, T, H, folds=5, first_origin=0.5):
    """Rolling-origin folds. Origin o_k splits the timeline: training windows end before o_k (a T+H gap), held-out windows
    start in [o_k, o_{k+1}). Expanding training window, no leakage, every fold is a genuine out-of-time test."""
    gap = (T + H) / N; step = (1.0 - first_origin) / folds
    return [((0.0, first_origin + k * step - gap), (first_origin + k * step, first_origin + (k + 1) * step)) for k in range(folds)]


def make_banks(spec, T, H, n_train, n_held, seed, split="chrono", ood_frac=0.0, regions=None):
    """train / held-out banks. csv + chrono: train windows start in the first 70 % of the timeline, held-out windows in the
    last 30 % minus a T+H gap (no overlap). ood_frac > 0 additionally holds out that fraction of the columns entirely
    (PostTime's ID / OOD variable split, arXiv 2605.29401): held-out targets are series never seen in training.
    random (the toy-matrix protocol) or synth: one bank split by index."""
    if spec.startswith("csv:") and split in ("chrono", "walkforward"):
        import pandas as pd; df = pd.read_csv(spec[4:]); N = len(df); gap = (T + H) / N; cols_tr = cols_te = None
        tr_region, te_region = regions if regions is not None else ((0.0, 0.7), (0.7 + gap, 1.0))
        if ood_frac > 0:
            C = len([c for c in df.columns if c != "date"]); g = torch.Generator().manual_seed(seed); perm = torch.randperm(C, generator=g).tolist()
            n_ood = max(1, int(round(ood_frac * C))); cols_te, cols_tr = perm[:n_ood], perm[n_ood:]
        return make_bank(spec, T, H, n_train, seed, tr_region, cols_tr), make_bank(spec, T, H, n_held, seed + 1, te_region, cols_te)
    bank = make_bank(spec, T, H, n_train, seed); return bank.split(1 - n_held / max(len(bank), 1))          # --episodes is the whole bank; the last n_held are held out (toy-matrix semantics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="grpo", choices=["none", "sft", "grpo", "ppo", "preference", "context", "context_sft", "agent_sft", "agent_rl", "agent_sft_rl", "agent_random", "fixed_context"])
    ap.add_argument("--agent_reward", default="crps", choices=["crps", "impratio", "newsvendor_tau"]); ap.add_argument("--proposals", type=int, default=8)
    ap.add_argument("--reward", default="composite"); ap.add_argument("--bank", default="synth"); ap.add_argument("--episodes", type=int, default=640)
    ap.add_argument("--T", type=int, default=256); ap.add_argument("--H", type=int, default=64); ap.add_argument("--heldout", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--K", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 11, 12, 13, 14]); ap.add_argument("--lora_r", type=int, default=8); ap.add_argument("--head_adapter", action="store_true")
    ap.add_argument("--anchor", type=float, default=0.0); ap.add_argument("--kl", type=float, default=0.05); ap.add_argument("--per_step", action="store_true")
    ap.add_argument("--kl_dir", default="sq", choices=["sq", "forward", "k3"]); ap.add_argument("--kl_ref", default="base", choices=["base", "ema"])
    ap.add_argument("--adv_norm", default="std", choices=["std", "loo", "remax", "none"]); ap.add_argument("--set_reward", action="store_true")
    ap.add_argument("--sampler", default="policy", choices=["policy", "ema", "mix"]); ap.add_argument("--ema_decay", type=float, default=0.0); ap.add_argument("--eval_ema", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA"); ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--tiny", action="store_true"); ap.add_argument("--model", default="timesfm3", help="timesfm3 | chronos-t5-{tiny,mini,small,base} | chronos-bolt-{tiny,mini,small,base} | chronos-2 | HF repo id")
    ap.add_argument("--split", default="walkforward", choices=["walkforward", "chrono", "random"], help="csv banks: rolling-origin folds (default), one chronological split, or random windows")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--first_origin", type=float, default=0.5, help="walk-forward: fraction of the timeline at the first forecast origin"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ood_frac", type=float, default=0.0, help="csv + chrono: fraction of columns held out entirely (held-out targets are unseen series)")
    ap.add_argument("--data_seed", type=int, default=None, help="seed for the episode bank / held-out split (default: --seed); vary --seed alone for a training-noise bound")
    ap.add_argument("--out", required=True); ap.add_argument("--log_every", type=int, default=10)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda": torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    fc = make_forecaster(a.model, device, layers=tuple(a.layers), lora_r=a.lora_r, head_adapter=a.head_adapter, tiny=a.tiny)
    if not fc.supports_covariates: print(f"note: {a.model} is univariate; candidate covariate rows are ignored", flush=True)
    if fc.stochastic and a.kl_dir == "sq": a.kl_dir = "k3"; print("note: token sampler has no differentiable quantiles; --kl_dir sq -> k3", flush=True)
    ds = a.seed if a.data_seed is None else a.data_seed
    cfg = TrainConfig(algo=a.algo, reward=a.reward, steps=a.steps, batch=a.batch, K=a.K, lr=a.lr, layers=tuple(a.layers), lora_r=a.lora_r, head_adapter=a.head_adapter,
                      anchor=a.anchor, kl=a.kl, kl_dir=a.kl_dir, kl_ref=a.kl_ref, per_step=a.per_step, adv_norm=a.adv_norm, set_reward=a.set_reward, sampler=a.sampler,
                      ema_decay=a.ema_decay if (a.ema_decay > 0 or a.sampler != "policy" or a.kl_ref == "ema" or a.eval_ema) else 0.0, eval_ema=a.eval_ema, amp=a.amp,
                      save_every=a.save_every, seed=a.seed, log_every=a.log_every)
    if cfg.ema_decay == 0.0 and (a.sampler != "policy" or a.kl_ref == "ema" or a.eval_ema): cfg.ema_decay = 0.99   # EMA needed but no decay given
    os.makedirs(a.out, exist_ok=True)
    if a.bank.startswith("csv:") and a.split == "walkforward":
        import pandas as pd; N = len(pd.read_csv(a.bank[4:])); folds = walkforward_regions(N, a.T, a.H, a.folds, a.first_origin)
    else: folds = [None]
    results = []
    for k, regions in enumerate(folds):
        out_k = a.out if regions is None else os.path.join(a.out, f"fold{k}")
        train, held = make_banks(a.bank, a.T, a.H, a.episodes, a.heldout, ds, a.split, a.ood_frac, regions)
        env = ForecastEnv(train, a.reward, device, a.seed); env_h = ForecastEnv(held, a.reward, device, ds + 1)
        fc.load_grafts(fc.frozen_state())                                                     # every fold starts from the released model
        meta = {"model": a.model, "split": a.split, "ood_frac": a.ood_frac, "fold": k, "regions": regions}
        ref_metrics = evaluate(fc, env_h, n=a.heldout)                                        # frozen reference on this fold's held-out block
        if a.algo == "none": pol_metrics = ref_metrics
        elif a.algo.startswith("agent") or a.algo == "fixed_context":
            run = Run(cfg, fc, env, out_k); fb = Fallback()
            if a.algo == "fixed_context":
                j = best_fixed_context(run, K=a.proposals); print(f"best fixed context on train: proposal {j}", flush=True)
                class _Fixed:
                    stochastic = False
                    def quantiles(self, ctx):
                        props = [env_h.propose(e, K=a.proposals) for e in env_h.eps]; P = min(len(p) for p in props); jj = min(j, P - 1)
                        return env_h.forecast(fc, [p[jj] for p in props])[0]
                    def swap(self, st): return fc.swap(st)
                with fc.swap(fc.frozen_state()): pol_metrics = evaluate(_Fixed(), env_h, n=a.heldout)
            else:
                stage = {"agent_sft": "sft", "agent_rl": "rl", "agent_sft_rl": "sft_rl", "agent_random": "rl"}[a.algo]
                agent, fb = train_agent(run, stage=stage, reward="random" if a.algo == "agent_random" else a.agent_reward, K=a.proposals)
                with fc.swap(fc.frozen_state()): pol_metrics = evaluate(AgentForecaster(agent, fc, env_h, fb, K=a.proposals), env_h, n=a.heldout)
        else:
            run = Run(cfg, fc, env, out_k)
            print(f"{a.algo} · reward {a.reward} · bank {a.bank} · fold {k} {regions} · trainable {sum(p.numel() for p in run.policy.trainable_parameters())/1e6:.3f}M", flush=True)
            {"sft": lambda: train_sft(run, "ce" if fc.stochastic else (a.reward if a.reward in ("mse", "mae", "pinball", "crps", "interval") else "pinball")), "grpo": lambda: train_grpo(run), "ppo": lambda: train_ppo(run),
             "preference": lambda: train_preference(run), "context": lambda: train_context(run), "context_sft": lambda: train_context(run, sft=True)}[a.algo]()
            if a.algo in ("context", "context_sft"): pol_metrics = ref_metrics
            elif run.eval_state() is not None:
                with run.swap(run.eval_state()): pol_metrics = evaluate(run.policy, env_h, n=a.heldout, ref_state=run.frozen_state)
            else: pol_metrics = evaluate(run.policy, env_h, n=a.heldout, ref_state=run.frozen_state)
        res = {"reference": ref_metrics, "policy": pol_metrics, "cfg": {**dataclasses.asdict(cfg), **meta}}
        if regions is not None: os.makedirs(out_k, exist_ok=True); json.dump(res, open(os.path.join(out_k, "eval.json"), "w"), indent=1)
        results.append(res); print(f"fold {k} held-out:", {kk: (round(ref_metrics[kk], 4), round(pol_metrics[kk], 4)) for kk in ("mase", "crps", "coverage80", "dir_acc")}, flush=True)
    agg = lambda key: {m: float(sum(r[key][m] for r in results) / len(results)) for m in results[0][key]}
    ref_metrics, pol_metrics = agg("reference"), agg("policy")
    json.dump({"reference": ref_metrics, "policy": pol_metrics, "cfg": {**dataclasses.asdict(cfg), "model": a.model, "split": a.split, "ood_frac": a.ood_frac, "folds": len(results)},
               "folds": [{"fold": r["cfg"]["fold"], "regions": r["cfg"]["regions"], "reference": r["reference"], "policy": r["policy"]} for r in results]},
              open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print("held-out (mean over folds):", {k: (round(ref_metrics[k], 4), round(pol_metrics[k], 4)) for k in ("mase", "crps", "coverage80", "dir_acc")})


if __name__ == "__main__":
    main()
