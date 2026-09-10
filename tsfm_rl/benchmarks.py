"""Public benchmarks as training banks and as the scored evaluation.

Training side (episode banks from TRAIN splits only):
    bank = gift_eval_bank(names=["electricity", "solar", ...], T=256, H=64, n=2000)      # needs the gift-eval package and $GIFT_EVAL
    bank = fev_bank(task_yaml="fev_bench", T=256, H=64, n=2000)                          # needs the fev library
Scoring side (official harnesses on TEST splits, once):
    evaluate_gift_eval(policy_fn, out_dir, configs=None)      # writes all_results.csv in the leaderboard layout
    evaluate_fev(policy_fn, out_dir, tasks="fev_bench")       # writes summaries.csv; fev.leaderboard gives skill scores + CIs

policy_fn(target (B,U,T), past_only (B,K1,T)|None, future (B,K2,T+H)|None, H) -> (B, U, H, 9) quantiles in the
original units. `make_policy_fn(policy, device)` builds it from a tsfm_rl Policy (or the frozen base).
The harness calls are import-guarded: they run on the cluster where gluonts / gift-eval / fev are installed.
"""
from __future__ import annotations

import os
import numpy as np
import torch

from .model import build_inputs, horizon_quantiles, PATCH
from .data import Episode, EpisodeBank


def make_policy_fn(policy, device="cpu", context=2048):
    @torch.no_grad()
    def fn(target, past_only, future, H):
        B, U, T = target.shape; ctx = min(context, (T // PATCH) * PATCH)
        mu = target[..., -ctx:].mean(-1, keepdim=True); sd = target[..., -ctx:].std(-1, keepdim=True) + 1e-6
        tgt = (target[..., -ctx:] - mu) / sd
        po = None if past_only is None else (past_only[..., -ctx:] - past_only[..., -ctx:].mean(-1, keepdim=True)) / (past_only[..., -ctx:].std(-1, keepdim=True) + 1e-6)
        fu = None
        if future is not None:
            fh = future[..., :T][..., -ctx:]; m2 = fh.mean(-1, keepdim=True); s2 = fh.std(-1, keepdim=True) + 1e-6
            fu = torch.cat([(fh - m2) / s2, (future[..., T:T + H] - m2) / s2], -1)
        inputs, roles, cpm, n_ctx = build_inputs(tgt.to(device), None if po is None else po.to(device), None if fu is None else fu.to(device), min(H, 64))
        out = policy(inputs, roles, cpm); q = horizon_quantiles(out, n_ctx, min(H, 64))[:, :U]
        if H > 64:   # stitch with the model's own decode for long horizons
            raise NotImplementedError("use TimesFM3Torch.decode (stitching) for H > 64; wire through Policy.base.decode")
        return (q.cpu() * sd[..., None] + mu[..., None])
    return fn


# ----------------------------------------------------------------------------- GIFT-Eval
def gift_eval_bank(names, T=256, H=64, n=2000, seed=0, term="short", max_candidates=8):
    """Episodes from GIFT-Eval TRAIN data (never the test windows). Multivariate datasets provide candidate rows."""
    from gift_eval.data import Dataset  # type: ignore
    g = torch.Generator().manual_seed(seed); eps = []
    for name in names:
        ds = Dataset(name=name, term=term, to_univariate=False)
        series = [np.asarray(e["target"], dtype=np.float32) for e in ds.training_dataset]
        for s in series:
            s = s[None] if s.ndim == 1 else s
            C, L = s.shape
            if L < T + H + 1: continue
            for _ in range(max(1, n // max(len(series) * len(names), 1))):
                st = int(torch.randint(0, L - T - H, (1,), generator=g)); tgt = int(torch.randint(0, C, (1,), generator=g))
                w = torch.tensor(s[:, st:st + T + H]).T; mu = w[:T].mean(0); sd = w[:T].std(0) + 1e-6; w = (w - mu) / sd
                others = [c for c in range(C) if c != tgt][:max_candidates]
                eps.append(Episode(w[:T, tgt], w[T:, tgt], w[:T, others].T.contiguous(), w[T:, others].T.contiguous(), torch.zeros(len(others), dtype=torch.bool), None, {"src": name, "col": tgt}))
                if len(eps) >= n: return EpisodeBank(eps, T, H)
    return EpisodeBank(eps, T, H)


def evaluate_gift_eval(policy_fn, out_dir, configs=None, model_name="timesfm3-rl"):
    """Official GIFT-Eval scoring on the test splits; writes <out_dir>/all_results.csv."""
    from gift_eval.data import Dataset  # type: ignore
    from gluonts.model.forecast import QuantileForecast  # type: ignore
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss  # type: ignore
    from gluonts.model.evaluation import evaluate_forecasts  # type: ignore
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True); qlv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; rows = []
    configs = configs or [f"{n}/{t}" for n in sorted(os.listdir(os.environ["GIFT_EVAL"])) for t in ("short", "medium", "long")]
    for cfg in configs:
        name, term = cfg.split("/")[0], cfg.split("/")[-1]
        try: ds = Dataset(name=name, term=term, to_univariate=False)
        except Exception: continue
        H = ds.prediction_length; fcs = []
        for entry in ds.test_data.input:
            tgt = torch.tensor(np.asarray(entry["target"], dtype=np.float32)); tgt = tgt[None] if tgt.ndim == 1 else tgt
            q = policy_fn(tgt[None], None, None, H)[0]                                          # (U, H, 9)
            arr = np.transpose(q.numpy(), (2, 1, 0)); arr = arr[..., 0] if arr.shape[-1] == 1 else arr
            fcs.append(QuantileForecast(forecast_arrays=arr, forecast_keys=[str(l) for l in qlv], start_date=entry["start"] + tgt.shape[1], item_id=entry.get("item_id")))
        m = evaluate_forecasts(fcs, test_data=ds.test_data, metrics=[MASE(), MeanWeightedSumQuantileLoss(qlv)], axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=ds.seasonality)
        rows.append({"dataset": cfg, "model": model_name, **{k: float(v) for k, v in m.iloc[0].items()}}); print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out_dir, "all_results.csv"), index=False); return df


# ----------------------------------------------------------------------------- fev-bench
def evaluate_fev(policy_fn, out_dir, tasks="fev_bench", model_name="timesfm3-rl"):
    """fev-bench scoring (MASE, SQL; skill scores and win rates with bootstrap CIs via fev.leaderboard)."""
    import fev  # type: ignore
    import pandas as pd, time
    bench = fev.Benchmark.from_yaml(tasks if tasks.endswith(".yaml") else "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml")
    os.makedirs(out_dir, exist_ok=True); summaries = []; qlv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for task in bench.tasks:
        t0 = time.time(); preds_all = []
        for window in task.iter_windows():
            past, future = window.get_input_data(); preds = []
            for i in range(len(past)):
                rec = past[i]; tgt = torch.tensor(np.asarray(rec[task.target], dtype=np.float32))[None, None]
                po = torch.stack([torch.tensor(np.asarray(rec[c], dtype=np.float32)) for c in task.past_dynamic_columns])[None] if task.past_dynamic_columns else None
                fu = None
                if task.known_dynamic_columns:
                    fr = future[i]; fu = torch.stack([torch.cat([torch.tensor(np.asarray(rec[c], dtype=np.float32)), torch.tensor(np.asarray(fr[c], dtype=np.float32))]) for c in task.known_dynamic_columns])[None]
                q = policy_fn(tgt, po, fu, task.horizon)[0, 0].numpy()
                preds.append({"predictions": q[:, 4], **{str(l): q[:, j] for j, l in enumerate(qlv)}})
            preds_all.append(preds)
        summaries.append(task.evaluation_summary(preds_all, model_name=model_name, inference_time_s=time.time() - t0))
    df = pd.DataFrame(summaries); df.to_csv(os.path.join(out_dir, "summaries.csv"), index=False); print(fev.leaderboard(df)); return df
