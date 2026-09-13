"""The agent around a frozen oracle: chooses a context and a trust weight, delivers a pooled forecast, trained by
hindsight supervision (stage 1) and full-information policy optimisation (stage 2). The oracle receives no gradient.

    fb = Fallback()                                   # calibrated local forecaster (seasonal naive + residual quantiles from the history)
    agent = TrustAgent(n_prop_feat=13, n_diag=5, w_grid=(0, .25, .5, .75, 1))
    props = [env.propose(e) for e in env.eps]         # K context proposals per episode (native first)
    q_all, q_fb, R, feats = enumerate_actions(env, forecaster, props, fb, reward="crps")   # every action scored in hindsight
    logits = agent(feats)                              # (B, P*|W|) joint action logits
    delivered = deliver(q_all, q_fb, action)           # linear pool of the oracle's fan under context c with the fallback, weight w
Reward is a proper score of the delivered distribution (or PostTime's improvement ratio over the native forecast, or a
randomised-fractile decision cost), never of a sample, so no action is rewarded for shrinking a band (T1 in docs/plan).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rewards import QLEVELS, crps_from_quantiles, newsvendor, mae
from .env import ContextAction


# ----------------------------------------------------------------------------- fallback forecaster
class Fallback:
    """Seasonal-naive (period m) point forecast plus per-horizon residual quantiles estimated by rolling origins inside the
    history: a conformal-style calibrated local forecaster with no learned parameters."""

    def __init__(self, periods=(24, 12, 7, 1), n_origins=16):
        self.periods, self.n_origins = periods, n_origins

    def _period(self, h):                                  # pick the period with the largest lag autocorrelation (per batch element)
        z = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6); best = torch.ones(h.shape[0], dtype=torch.long, device=h.device); score = torch.full((h.shape[0],), -2.0, device=h.device)
        for m in self.periods:
            if m >= h.shape[1] // 2: continue
            ac = (z[:, m:] * z[:, :-m]).mean(1); upd = ac > score; score = torch.where(upd, ac, score); best = torch.where(upd, torch.full_like(best, m), best)
        return best

    def _naive(self, h, m, H):                              # (B,T),(B,),int -> (B,H)
        B, T = h.shape; t = torch.arange(H, device=h.device)[None]
        idx = T - m[:, None] + (t % m[:, None]); return h.gather(1, idx.clamp(0, T - 1))

    def __call__(self, history, H):
        B, T = history.shape; m = self._period(history); dev = history.device
        origins = torch.linspace(max(2 * int(m.max()), T // 4), T - H, self.n_origins).long()
        res = []
        for o in origins.tolist():
            res.append(history[:, o:o + H] - self._naive(history[:, :o], m, H))            # (B,H) residuals at origin o
        res = torch.stack(res, 0)                                                            # (O,B,H)
        qr = torch.quantile(res, QLEVELS.to(dev), dim=0).permute(1, 2, 0)                   # (B,H,9)
        return self._naive(history, m, H)[..., None] + qr


# ----------------------------------------------------------------------------- pooling
def _tail_points(q):
    """Outer breakpoints where the linearly extended CDF reaches 0 and 1: q1 - (q2 - q1), q9 + (q9 - q8)."""
    qs, _ = q.sort(-1); return qs[..., 0] - (qs[..., 1] - qs[..., 0]).clamp_min(1e-6), qs[..., 8] + (qs[..., 8] - qs[..., 7]).clamp_min(1e-6)


def _cdf_eval(q, x):
    """CDF of the piecewise-linear distribution through quantiles q (B,H,9) at points x (B,H,G) -> (B,H,G). Between quantiles
    the CDF is linear; beyond the outer quantiles it continues with the adjacent slope until it reaches 0 / 1 (continuous)."""
    qs, _ = q.sort(-1); lo, hi = _tail_points(qs); pts = torch.cat([lo[..., None], qs, hi[..., None]], -1)          # (B,H,11)
    lv = torch.cat([torch.zeros(1), QLEVELS, torch.ones(1)]).to(q.device, q.dtype)                                  # 0, .1..., .9, 1
    below = (x[..., None] >= pts[..., None, :]).float().sum(-1)                                                      # 0..11
    k = below.clamp(1, 10).long(); x0 = pts.gather(-1, k - 1); x1 = pts.gather(-1, k); f0 = lv[k - 1]; f1 = lv[k]
    frac = ((x - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
    return (f0 + (f1 - f0) * frac).clamp(0, 1)


def pool_quantiles(qa, qb, w):
    """Quantiles of the linear pool w*Pa + (1-w)*Pb at QLEVELS, (B,H,9); w (B,) in [0,1]. Exact: the pooled CDF is piecewise
    linear with breakpoints at the union of both quantile sets, so it is evaluated there and inverted by interpolation."""
    B, H, Q = qa.shape; ww = w[:, None, None]; la, ha = _tail_points(qa); lb, hb = _tail_points(qb)
    bp = torch.cat([qa, qb, la[..., None], ha[..., None], lb[..., None], hb[..., None]], -1).sort(-1).values          # (B,H,2Q+4) breakpoints incl. tail ends
    Fp = ww * _cdf_eval(qa, bp) + (1 - ww) * _cdf_eval(qb, bp)                              # exact pooled CDF at the breakpoints
    out = []
    for l in QLEVELS.tolist():
        k = (Fp < l - 1e-9).float().sum(-1).clamp(1, bp.shape[-1] - 1).long()              # first breakpoint with F >= l
        x1 = bp.gather(-1, k[..., None])[..., 0]; x0 = bp.gather(-1, (k - 1)[..., None])[..., 0]
        f1 = Fp.gather(-1, k[..., None])[..., 0]; f0 = Fp.gather(-1, (k - 1)[..., None])[..., 0]
        t = ((l - f0) / (f1 - f0).clamp_min(1e-9)).clamp(0, 1); out.append(x0 + (x1 - x0) * t)
    q = torch.stack(out, -1)
    q = torch.where(ww.expand_as(q) >= 1 - 1e-6, qa, torch.where(ww.expand_as(q) <= 1e-6, qb, q))   # endpoints return the component fans untouched (exact pairing with the oracle)
    return q


def deliver(q_all, q_fb, prop_idx, w):
    """q_all (P,B,H,9), q_fb (B,H,9), prop_idx (B,), w (B,) -> delivered (B,H,9)."""
    B = q_fb.shape[0]; qc = q_all[prop_idx, torch.arange(B, device=q_fb.device)]
    return pool_quantiles(qc, q_fb, w)


# ----------------------------------------------------------------------------- features and the agent
def oracle_diagnostics(q_all, q_fb, history):
    """Per-proposal diagnostics (B,P,5): band width, median shift from native, disagreement across proposals, width vs fallback,
    history volatility."""
    P, B, H, _ = q_all.shape; med = q_all[..., 4]; width = (q_all[..., 8] - q_all[..., 0]).mean(-1)                      # (P,B)
    shift = (med - med[0:1]).abs().mean(-1); disagree = med.std(0).mean(-1)[None].expand(P, -1)
    wfb = (width / ((q_fb[..., 8] - q_fb[..., 0]).mean(-1)[None] + 1e-6)); vol = history.std(1)[None].expand(P, -1)
    return torch.stack([width, shift, disagree, wfb, vol], -1).permute(1, 0, 2)


class TrustAgent(nn.Module):
    def __init__(self, n_prop_feat=13, n_diag=5, hidden=64, w_grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        super().__init__(); self.w_grid = torch.tensor(w_grid); self.net = nn.Sequential(nn.Linear(n_prop_feat + n_diag, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(w_grid)))

    def forward(self, feats):                               # (B,P,F) -> joint logits (B, P*|W|) over (proposal, trust)
        return self.net(feats).reshape(feats.shape[0], -1)

    def decode(self, action, P):                            # joint index -> (proposal, w)
        W = len(self.w_grid); return action // W, self.w_grid.to(action.device)[action % W]


def enumerate_actions(env, forecaster, proposals, fallback, reward="crps", y=None, tau=None):
    """Score every (proposal, trust) action in hindsight. Returns q_all (P,B,H,9), q_fb (B,H,9), R (B, P*|W|), feats (B,P,F).
    Full information: the realisation scores all actions, so the policy objective E_pi[R] is computable exactly."""
    from .algorithms.preference import proposal_features
    P = min(len(p) for p in proposals); proposals = [p[:P] for p in proposals]
    with torch.no_grad():
        q_all = torch.stack([env.forecast(forecaster, [p[j] for p in proposals])[0] for j in range(P)])              # (P,B,H,9)
        hist = torch.stack([e.target for e in env.eps]).to(q_all.device); y = env.gold() if y is None else y
        q_fb = fallback(hist, env.bank.H)
        feats = torch.cat([proposal_features(env, proposals).to(q_all.device), oracle_diagnostics(q_all, q_fb, hist)], -1)   # (B,P,13+5)
        W = torch.tensor(TrustAgent().w_grid.tolist(), device=q_all.device); B = q_fb.shape[0]; R = torch.zeros(B, P, len(W), device=q_all.device)
        for j in range(P):
            for i, w in enumerate(W.tolist()):
                d = pool_quantiles(q_all[j], q_fb, torch.full((B,), w, device=q_all.device))
                if reward == "crps": R[:, j, i] = -crps_from_quantiles(d, y)
                elif reward == "impratio": R[:, j, i] = (0.5 + 0.5 * (1 - crps_from_quantiles(d, y) / (crps_from_quantiles(q_all[0], y) + 1e-6))).clamp(0, 1)
                elif reward == "newsvendor_tau":                                                                      # randomised critical fractile (utility-proper, T4)
                    t = tau if tau is not None else torch.rand(B, device=q_all.device); cu = t; co = 1 - t
                    R[:, j, i] = -newsvendor(d, y, cu=cu[:, None] if torch.is_tensor(cu) else cu, co=co[:, None] if torch.is_tensor(co) else co)
                elif reward == "random": R[:, j, i] = torch.rand(B, device=q_all.device)
                else: raise ValueError(reward)
    return q_all, q_fb, R.reshape(B, -1), feats, proposals


class AgentForecaster:
    """Wraps (agent, oracle, env) so evaluate() can score the delivered forecast like any forecaster."""

    def __init__(self, agent, forecaster, env, fallback, K=8):
        self.agent, self.f, self.env, self.fb, self.K = agent, forecaster, env, fallback, K; self.stochastic = False

    def quantiles(self, ctx):
        props = [self.env.propose(e, K=self.K) for e in self.env.eps]
        q_all, q_fb, _, feats, props = enumerate_actions(self.env, self.f, props, self.fb, reward="random")
        with torch.no_grad(): a = self.agent(feats).argmax(1); j, w = self.agent.decode(a, q_all.shape[0])
        return deliver(q_all, q_fb, j, w)

    def swap(self, state): return self.f.swap(state)
    def frozen_state(self): return self.f.frozen_state()
