#!/usr/bin/env python3
"""Re-score finished cells on their held-out episodes with per-episode metrics and paired bootstrap CIs vs the frozen model.

    python3 scripts/reeval_ci.py runs/synth --bank synth
    python3 scripts/reeval_ci.py runs/electricity --bank csv:data/electricity.csv

Writes <run>/eval_ci.json: per-episode arrays, mean deltas (policy - frozen) with 95 % paired-bootstrap intervals,
and quantile fans for a few fixed held-out windows (for the forecast figures). Runs on CPU next to a live matrix
(2 threads).
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm_rl.forecasters import make_forecaster  # noqa: E402
from tsfm_rl.env import ForecastEnv, ContextAction  # noqa: E402
from tsfm_rl import rewards as R  # noqa: E402
from scripts.run_matrix import make_banks  # noqa: E402

KEYS = ("mase", "crps", "pinball", "coverage80", "width", "dir_acc", "interval", "newsvendor")


@torch.no_grad()
def per_episode(policy, env, n=128, batch=16, keep=4):
    agg = {k: [] for k in KEYS}; fans = []; seen = 0; env.g.manual_seed(12345)
    while seen < n:
        eps = env.reset(min(batch, n - seen)); q, y, info = env.forecast(policy, [ContextAction.native(e) for e in eps])
        agg["mase"].append(R.mase(q, y, info["history"])); agg["crps"].append(R.crps_from_quantiles(q, y)); agg["pinball"].append(R.pinball(q, y))
        agg["coverage80"].append(R.coverage(q, y)); agg["width"].append(R.width(q)); agg["dir_acc"].append(R.directional_accuracy(q, y))
        agg["interval"].append(R.interval_score(q, y)); agg["newsvendor"].append(R.newsvendor(q, y))
        if not fans: fans = [{"history": info["history"][i, -128:].tolist(), "truth": y[i].tolist(), "q": q[i].tolist()} for i in range(min(keep, len(eps)))]
        seen += len(eps)
    return {k: torch.cat(v) for k, v in agg.items()}, fans


def bootstrap(delta, n_boot=4000, seed=0):
    g = torch.Generator().manual_seed(seed); n = delta.shape[0]
    idx = torch.randint(0, n, (n_boot, n), generator=g); means = delta[idx].mean(1)
    return float(delta.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--bank", required=True); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--model", default=None, help="default: the model recorded in each cell's eval.json"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(a.threads); cache = {}; fcs = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*"))):
        ev = os.path.join(d, "eval.json"); out = os.path.join(d, "eval_ci.json")
        if not os.path.exists(ev) or (os.path.exists(out) and not a.force): continue
        top = json.load(open(ev)); cfg = top["cfg"]; seed = cfg["seed"]; model = a.model or cfg.get("model", "timesfm3"); split = cfg.get("split", "random")
        key = (model, tuple(cfg["layers"]), cfg["lora_r"], cfg["head_adapter"])
        if key not in fcs: fcs[key] = make_forecaster(model, "cpu", layers=tuple(cfg["layers"]), lora_r=cfg["lora_r"], head_adapter=cfg["head_adapter"])
        fc = fcs[key]
        folds = [(os.path.join(d, f"fold{f['fold']}"), tuple(tuple(r) for r in f["regions"])) for f in top["folds"]] if top.get("folds") else [(d, None)]
        refs, pols, ref_fans, fans = [], [], None, None
        for fd, regions in folds:                                                           # one paired scoring per fold, concatenated over folds
            ck_key = (seed, split, regions)
            if ck_key not in cache:
                _, held = make_banks(a.bank, 256, 64, 640 - 128, 128, seed, split, cfg.get("ood_frac", 0.0), regions); env = ForecastEnv(held, cfg["reward"], "cpu", seed + 1)
                with fc.swap(fc.frozen_state()): r_, rf_ = per_episode(fc, env)
                cache[ck_key] = (env, r_, rf_)
            env, r_, rf_ = cache[ck_key]; ref_fans = ref_fans or rf_
            if cfg["algo"] in ("none", "context", "context_sft") or not os.path.exists(os.path.join(fd, "policy.pt")): p_, f_ = r_, rf_
            else:
                ck = torch.load(os.path.join(fd, "policy.pt"), map_location="cpu"); st = ck["ema_state"] if (cfg.get("eval_ema") and ck.get("ema_state")) else ck["state"]
                assert fc.load_grafts(st) > 0, f"no graft keys matched for {fd}"
                p_, f_ = per_episode(fc, env); fc.load_grafts(fc.frozen_state())
            refs.append(r_); pols.append(p_); fans = fans or f_
        ref = {k: torch.cat([r[k] for r in refs]) for k in KEYS}; pol = {k: torch.cat([p[k] for p in pols]) for k in KEYS}
        res = {"cfg": cfg, "n": int(ref["mase"].shape[0]), "reference": {k: float(v.mean()) for k, v in ref.items()}, "policy": {k: float(v.mean()) for k, v in pol.items()},
               "delta": {k: bootstrap(pol[k] - ref[k]) for k in KEYS},
               "rel_skill": {k: bootstrap(1 - pol[k] / ref[k].mean()) for k in ("mase", "crps", "pinball")},   # 1 - policy/frozen, bootstrapped over episodes
               "per_episode": {k: {"policy": pol[k].tolist(), "reference": ref[k].tolist()} for k in KEYS}, "fans": fans, "ref_fans": ref_fans}
        json.dump(res, open(out, "w")); print(os.path.basename(d), {k: tuple(round(x, 4) for x in res["delta"][k]) for k in ("mase", "crps", "coverage80")}, flush=True)


if __name__ == "__main__":
    main()
