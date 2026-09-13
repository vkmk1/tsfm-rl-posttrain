#!/usr/bin/env python3
"""Score a saved policy (or the frozen model) through the official harnesses on the TEST splits, once.

    python3 scripts/eval_benchmarks.py --policy runs/gpu/grpo_d1_full/policy.pt --gift --configs electricity/short solar/short --out results/gift/grpo_d1_full
    python3 scripts/eval_benchmarks.py --frozen --fev --out results/fev/frozen
Requires the gift-eval package and $GIFT_EVAL (data root) for --gift, the `fev` library for --fev. --ema uses the EMA weights.
"""
from __future__ import annotations

import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm_rl.forecasters import make_forecaster  # noqa: E402
from tsfm_rl.benchmarks import make_policy_fn, evaluate_gift_eval, evaluate_fev  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy"); ap.add_argument("--frozen", action="store_true"); ap.add_argument("--ema", action="store_true")
    ap.add_argument("--model", default="timesfm3"); ap.add_argument("--gift", action="store_true"); ap.add_argument("--fev", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None); ap.add_argument("--tasks", default="fev_bench"); ap.add_argument("--out", required=True); ap.add_argument("--name", default=None)
    a = ap.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.frozen: model = make_forecaster(a.model, device, lora_r=0); name = a.name or f"{a.model}-frozen"
    else:
        ck = torch.load(a.policy, map_location=device); cfg = ck["cfg"]
        model = make_forecaster(cfg.get("model", a.model), device, layers=tuple(cfg["layers"]), lora_r=cfg["lora_r"], head_adapter=cfg["head_adapter"])
        st = ck["ema_state"] if (a.ema and ck.get("ema_state")) else ck["state"]; assert model.load_grafts(st) > 0, "no graft keys matched"; model.eval()
        name = a.name or os.path.basename(os.path.dirname(a.policy)) + ("-ema" if a.ema else "")
    fn = make_policy_fn(model, device)
    if a.gift: evaluate_gift_eval(fn, os.path.join(a.out, "gift_eval"), configs=a.configs, model_name=name)
    if a.fev: evaluate_fev(fn, os.path.join(a.out, "fev"), tasks=a.tasks, model_name=name)


if __name__ == "__main__":
    main()
