"""The environment: turn an episode and a context action into TimesFM-3 inputs, run a policy, score the forecast.

    env = ForecastEnv(bank, reward="composite", device="cpu")
    eps = env.reset(batch_size)                       # list of Episode
    actions = [ContextAction.native(e) for e in eps]  # or env.propose(e) for the retrieval strategies
    q, y, info = env.forecast(policy, actions)        # q (B, H, 9) target-row quantiles, y gold (B, H)
    r = env.reward(q, y, ref_q)                       # (B,) from tsfm_rl.rewards

Context actions choose which candidate rows enter as past-only or known-future covariates, the history length,
detrending, and whether the estimator row is appended. Retrieval strategies (shape, spectrum, seasonality,
random, lag-regression rank) propose diverse actions; a context policy (algorithms/preference.py) can learn to
choose among them. With native actions the environment reduces to plain post-training of the forecaster.
"""
from __future__ import annotations

import dataclasses
import torch

from .model import build_inputs, horizon_quantiles, PATCH
from .rewards import REWARDS
from .adapters import InContextLagRegression


@dataclasses.dataclass
class ContextAction:
    rows: list
    hist_len: int = 256
    detrend: bool = False
    estimator_row: bool = False

    @staticmethod
    def native(ep): return ContextAction(rows=list(range(ep.cov_hist.shape[0])))


def _z(x, dim=-1): return (x - x.mean(dim, keepdim=True)) / (x.std(dim, keepdim=True) + 1e-6)


def strat_shape(ep, m): return ((_z(ep.cov_hist[:, -128:]) - _z(ep.target[-128:])[None]) ** 2).mean(1).argsort()[:m].tolist()
def strat_spectrum(ep, m):
    P = lambda x: torch.log1p(torch.fft.rfft(_z(x), dim=-1).abs()); return ((P(ep.cov_hist) - P(ep.target)[None]) ** 2).mean(1).argsort()[:m].tolist()
def strat_seasonality(ep, m):
    acf = lambda x: torch.stack([(_z(x)[..., l:] * _z(x)[..., :-l]).mean(-1) for l in range(1, 25)], -1)
    return ((acf(ep.cov_hist) - acf(ep.target)[None]) ** 2).mean(1).argsort()[:m].tolist()
def strat_random(ep, m, g=None): return torch.randperm(ep.cov_hist.shape[0], generator=g)[:m].tolist()
_ICR = InContextLagRegression(lags=32)
def strat_lagreg(ep, m):
    T, C = ep.target.shape[0], ep.cov_hist.shape[0]
    x = torch.cat([torch.cat([ep.target, torch.zeros(ep.future.shape[0])])[None], torch.cat([ep.cov_hist, ep.cov_future], 1)], 0)[None]
    obs = torch.ones_like(x); obs[:, 0, T:] = 0; roles = torch.cat([torch.zeros(1, dtype=torch.long), torch.full((C,), 2)])[None]
    with torch.no_grad(): _, _, r2 = _ICR(x, roles, obs, torch.tensor([T]))
    return r2[0, 1:].argsort(descending=True)[:m].tolist()


STRATEGIES = {"shape": strat_shape, "spectrum": strat_spectrum, "seasonality": strat_seasonality, "random": strat_random, "lagreg": strat_lagreg}


class ForecastEnv:
    def __init__(self, bank, reward="composite", device="cpu", seed=0):
        self.bank, self.reward_name, self.device = bank, reward, torch.device(device); self.g = torch.Generator().manual_seed(seed); self.icr = InContextLagRegression(32)

    def reset(self, batch_size):
        self.eps = self.bank.sample(batch_size, self.g); return self.eps

    def propose(self, ep, K=8, m=3):
        acts = [ContextAction.native(ep), ContextAction(rows=[])]
        for name, fn in STRATEGIES.items():
            rows = fn(ep, m, self.g) if name == "random" else fn(ep, m)
            acts += [ContextAction(rows=rows), ContextAction(rows=rows, estimator_row=True)]
        return acts[:K]

    def build(self, actions):
        T, H = self.bank.T, self.bank.H; B = len(self.eps); L = max(PATCH, (min(a.hist_len for a in actions) // PATCH) * PATCH)
        tgt = torch.stack([e.target[-L:] for e in self.eps])[:, None].clone()
        n_po = max([sum(1 for r in a.rows if not e.known_future[r]) for a, e in zip(actions, self.eps)] + [0])
        n_kf = max([sum(1 for r in a.rows if e.known_future[r]) + int(a.estimator_row) for a, e in zip(actions, self.eps)] + [0])
        po = torch.zeros(B, n_po, L); kf = torch.zeros(B, n_kf, L + H); po_pad = torch.ones(B, n_po, dtype=torch.bool); kf_pad = torch.ones(B, n_kf, dtype=torch.bool); self._slope = torch.zeros(B)
        for i, (a, e) in enumerate(zip(actions, self.eps)):
            ip = ik = 0
            for r in a.rows:
                if e.known_future[r]: kf[i, ik, :L] = e.cov_hist[r, -L:]; kf[i, ik, L:] = e.cov_future[r]; kf_pad[i, ik] = False; ik += 1
                else: po[i, ip] = e.cov_hist[r, -L:]; po_pad[i, ip] = False; ip += 1
            if a.estimator_row:
                rows = [r for r in a.rows if e.known_future[r]]
                if rows:
                    x = torch.cat([torch.cat([e.target[-L:], torch.zeros(H)])[None], torch.cat([e.cov_hist[rows][:, -L:], e.cov_future[rows]], 1)], 0)[None]
                    obs = torch.ones_like(x); obs[:, 0, L:] = 0; roles = torch.cat([torch.zeros(1, dtype=torch.long), torch.full((len(rows),), 2)])[None]
                    with torch.no_grad(): per_cov, _, r2 = self.icr(x, roles, obs, torch.tensor([L]))
                    g = torch.sigmoid(6 * r2[0, 1:] - 3); kf[i, ik] = (per_cov[0][:, 1:] * g[None]).sum(-1); kf_pad[i, ik] = False; ik += 1
            if a.detrend:
                t = torch.arange(L, dtype=torch.float32); tc = t - t.mean(); slope = (tc * (tgt[i, 0] - tgt[i, 0].mean())).sum() / (tc ** 2).sum()
                tgt[i, 0] = tgt[i, 0] - slope * tc; self._slope[i] = slope
        inputs, roles, cpm, n_ctx = build_inputs(tgt.to(self.device), po.to(self.device) if n_po else None, kf.to(self.device) if n_kf else None, H)
        if n_po or n_kf:
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), po_pad, kf_pad], 1).to(self.device); inputs["masks"] = inputs["masks"] | pad[:, :, None, None]
        return inputs, roles, cpm, n_ctx

    def forecast(self, policy, actions=None):
        """policy: callable(inputs, roles, cpm) -> output dict (a Policy, or the frozen base via base_fn)."""
        actions = actions or [ContextAction.native(e) for e in self.eps]
        inputs, roles, cpm, n_ctx = self.build(actions)
        out = policy(inputs, roles, cpm); q = horizon_quantiles(out, n_ctx, self.bank.H)[:, 0]
        if self._slope.abs().sum() > 0:
            L = inputs["values"].shape[2] * PATCH - ((self.bank.H + 63) // 64) * 64; t = torch.arange(1, self.bank.H + 1, dtype=torch.float32, device=q.device)
            q = q + (self._slope.to(q.device)[:, None] * (t[None] + (L - 1) / 2))[:, :, None]
        y = torch.stack([e.future for e in self.eps]).to(self.device)
        info = {"inputs": inputs, "roles": roles, "cpm": cpm, "n_ctx": n_ctx, "history": torch.stack([e.target for e in self.eps]).to(self.device),
                "cond_mean": torch.stack([e.cond_mean for e in self.eps]).to(self.device) if self.eps[0].cond_mean is not None else None}
        return q, y, info

    def reward(self, q, y, ref_q=None, name=None):
        return REWARDS[name or self.reward_name](q, y, ref_q)


def base_fn(base):
    """Wrap the frozen base model as a policy-like callable."""
    return lambda inputs, roles, cpm: base(inputs, patch_cpm_mask=cpm)
