"""Episodes: one forecasting instance each (target history, gold future, candidate covariate rows).

    bank = EpisodeBank.from_csv("data/ETTh1.csv", T=256, H=64)        # Time-Series-Library layout (date + columns)
    bank = EpisodeBank.synthetic(n=512)                                # known laws, conditional mean available
    train, heldout = bank.split(0.8)

Synthetic laws (the audit's generator, widened): y_t = b x_{s,t-l} + e ("lag"), two-driver sum, product,
regime switch, saturation; AR(1) drivers with separate history / future coefficients; distractor drivers.
"""
from __future__ import annotations

import dataclasses, os
import numpy as np
import torch

LAWS = ("lag", "sum", "product", "regime", "saturate")


@dataclasses.dataclass
class Episode:
    target: torch.Tensor           # (T,)
    future: torch.Tensor           # (H,) gold
    cov_hist: torch.Tensor         # (C, T) candidate rows, history
    cov_future: torch.Tensor       # (C, H) candidate rows, future (used only for rows flagged known-future)
    known_future: torch.Tensor     # (C,) bool
    cond_mean: torch.Tensor | None = None
    meta: dict = dataclasses.field(default_factory=dict)


def _ar1(B, T, phi, g, burn=32):
    e = torch.randn(B, T + burn, generator=g) * (1 - phi ** 2) ** 0.5; x = torch.zeros(B, T + burn)
    for t in range(1, T + burn): x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x[:, burn:]


def make_synthetic(B=32, T=256, H=64, K=2, laws=("lag",), lags=(0, 1, 2, 4, 8), phi_h=(0.0, 0.8, 0.95), phi_f=(0.0, 0.8), noise=0.1, n_distractors=0, seed=0, fixed=None):
    g = torch.Generator().manual_seed(seed)
    ph = torch.tensor(phi_h)[torch.randint(len(phi_h), (B,), generator=g)]; pf = torch.tensor(phi_f)[torch.randint(len(phi_f), (B,), generator=g)]
    if fixed:
        if "phi_h" in fixed: ph = torch.full((B,), float(fixed["phi_h"]))
        if "phi_f" in fixed: pf = torch.full((B,), float(fixed["phi_f"]))
    Kd = K + n_distractors; x = torch.zeros(B, Kd, T + H)
    for b in range(B):
        gb = torch.Generator().manual_seed(seed * 100003 + b)
        for k in range(Kd):
            x[b, k, :T] = _ar1(1, T, float(ph[b]), gb)[0]; x[b, k, T:] = _ar1(1, H, float(pf[b]), gb)[0]
    law_idx = torch.randint(len(laws), (B,), generator=g); lag = torch.tensor(lags)[torch.randint(len(lags), (B,), generator=g)]
    if fixed and "lag" in fixed: lag = torch.full((B,), int(fixed["lag"]))
    src = torch.randint(K, (B,), generator=g); src2 = (src + 1 + torch.randint(max(K - 1, 1), (B,), generator=g)) % K if K > 1 else src
    sign = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
    if fixed and "sign" in fixed: sign = torch.full((B,), float(fixed["sign"]))
    if fixed and "src" in fixed: src = torch.full((B,), int(fixed["src"]))
    m = torch.zeros(B, T + H)
    for b in range(B):
        law = laws[law_idx[b]]; l = int(lag[b]); s, s2 = int(src[b]), int(src2[b]); bb = float(sign[b])
        xs = torch.nn.functional.pad(x[b, s], (l, 0))[: T + H]
        if law == "lag": m[b] = bb * xs
        elif law == "sum": m[b] = bb * xs + 0.5 * torch.nn.functional.pad(x[b, s2], (l // 2, 0))[: T + H]
        elif law == "product": m[b] = bb * xs * x[b, s2]
        elif law == "regime": m[b] = torch.where(x[b, s2] > 0, bb * xs, -bb * xs)
        elif law == "saturate": m[b] = bb * torch.tanh(1.5 * xs)
    y = m + noise * torch.randn(B, T + H, generator=g)
    return {"target": y[:, None, :T], "future": x, "cond_mean": m[:, T:], "y_future": y[:, T:], "rho": {"law": law_idx, "lag": lag, "src": src, "sign": sign, "phi_h": ph, "phi_f": pf}}


class EpisodeBank:
    def __init__(self, episodes, T, H):
        self.episodes, self.T, self.H = episodes, T, H

    @classmethod
    def from_csv(cls, path, T=256, H=64, n=512, known_future_frac=0.5, seed=0, max_candidates=8):
        """The target is a random column per window; the other columns are candidate rows. A fraction of candidates
        is declared known-future (their gold future is available as a covariate), the rest past-only."""
        import pandas as pd
        df = pd.read_csv(path); X = torch.tensor(df[[c for c in df.columns if c != "date"]].to_numpy(np.float32))
        N, C = X.shape; g = torch.Generator().manual_seed(seed); eps = []
        for _ in range(n):
            s = int(torch.randint(0, N - T - H, (1,), generator=g)); tgt = int(torch.randint(0, C, (1,), generator=g))
            w = X[s:s + T + H]; mu = w[:T].mean(0); sd = w[:T].std(0) + 1e-6; w = (w - mu) / sd
            others = [c for c in range(C) if c != tgt][:max_candidates]; known = torch.rand(len(others), generator=g) < known_future_frac
            eps.append(Episode(w[:T, tgt], w[T:, tgt], w[:T, others].T.contiguous(), w[T:, others].T.contiguous(), known, None, {"src": os.path.basename(path), "col": tgt, "start": s}))
        return cls(eps, T, H)

    @classmethod
    def synthetic(cls, n=512, T=256, H=64, K=2, laws=LAWS, lags=(0, 1, 2, 4, 8, 16), distractors=2, seed=0):
        b = make_synthetic(n, T, H, K, laws, lags, n_distractors=distractors, seed=seed); eps = []
        for i in range(n):
            eps.append(Episode(b["target"][i, 0], b["y_future"][i], b["future"][i, :, :T], b["future"][i, :, T:], torch.ones(b["future"].shape[1], dtype=torch.bool), b["cond_mean"][i], {k: v[i].item() for k, v in b["rho"].items()}))
        return cls(eps, T, H)

    def split(self, frac=0.8):
        k = int(len(self.episodes) * frac); return EpisodeBank(self.episodes[:k], self.T, self.H), EpisodeBank(self.episodes[k:], self.T, self.H)

    def sample(self, batch_size, g):
        idx = torch.randint(0, len(self.episodes), (batch_size,), generator=g); return [self.episodes[i] for i in idx.tolist()]

    def __len__(self):
        return len(self.episodes)
