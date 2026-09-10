"""Trainable parameters grafted onto the frozen TimesFM-3: the policy's parameters for every post-training method.

    policy = Policy(base, layers=(10..14), lora_r=8, head_adapter=True)
    out = policy(inputs, roles, cpm)        # same output dict as the base model
    policy.trainable_parameters()

LoRA       low-rank updates on the query / key / value projections of variate attention at the chosen layers.
HeadAdapter an in-context lag regression of the target on the known-future covariates (fitted on the history,
            parameter-free) whose prior forecast is added at the output head through a learned gate and scale.
            It removes the delay boundary the frozen model has; off by default so the RL-vs-SFT comparison
            is about the algorithm, on for the "our policy class" arm.
All grafts are zero at initialization: the released model is recovered exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import TimesFM3Torch, ROLE_TARGET, ROLE_FUTURE


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01); self.B = nn.Parameter(torch.zeros(base.out_features, r))
        for p in self.base.parameters(): p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


class InContextLagRegression(nn.Module):
    """Ridge regression of the target on lagged covariate values over the observed history; prior forecast for all
    steps (covariate futures known), per-covariate held-out R^2 (chronological 70/30 split) for the gate."""

    def __init__(self, lags=32, ridge=1e-2):
        super().__init__(); self.L, self.ridge = lags, ridge

    def forward(self, x, roles, observed, t_hist):     # x (B,C,T) normalized values, observed (B,C,T), t_hist (B,)
        B, C, T = x.shape; is_t = roles == ROLE_TARGET
        y = (x * is_t[:, :, None]).sum(1); xo = x * observed
        cov = (~is_t).to(x.dtype)[:, :, None]
        cols = [torch.nn.functional.pad(xo, (l, 0))[..., :T] for l in range(self.L)]
        Phi = (torch.stack(cols, -1) * cov[..., None]).permute(0, 2, 1, 3).reshape(B, T, C * self.L)
        t = torch.arange(T, device=x.device)[None]
        w_all = ((t < t_hist[:, None]) & (t >= self.L)).to(x.dtype)
        w_fit = w_all * (t < (self.L + 0.7 * (t_hist[:, None] - self.L)).floor()).to(x.dtype); w_val = w_all - w_fit
        eye = torch.eye(C * self.L, device=x.device, dtype=x.dtype)

        def solve(w):
            G = torch.einsum("bti,btj->bij", Phi * w[..., None], Phi); g = torch.einsum("bti,bt->bi", Phi * w[..., None], y)
            lam = (self.ridge * G.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]).clamp_min(1e-3)
            return torch.nan_to_num(torch.linalg.solve(G + lam * eye, g[..., None])[..., 0])

        pv = (Phi * solve(w_fit)[:, None]).view(B, T, C, self.L).sum(-1)
        r2 = []
        for c in range(C):
            res = (((pv[..., c] - y) ** 2) * w_val).sum(-1); mu = (y * w_val).sum(-1, keepdim=True) / w_val.sum(-1, keepdim=True).clamp_min(1)
            r2.append(1 - res / (((y - mu) ** 2) * w_val).sum(-1).clamp_min(1e-6))
        r2 = torch.nan_to_num(torch.stack(r2, -1), nan=-1.0) * (~is_t).to(x.dtype)
        R = solve(w_all).view(B, C, self.L)
        prior_per_cov = (Phi * R.reshape(B, 1, -1)).view(B, T, C, self.L).sum(-1)     # (B, T, C)
        return prior_per_cov, R, r2


class HeadAdapter(nn.Module):
    def __init__(self, lags=32):
        super().__init__()
        self.icr = InContextLagRegression(lags); self.alpha = nn.Parameter(torch.zeros(1))
        self.gate = nn.Linear(1, 1); nn.init.constant_(self.gate.weight, 6.0); nn.init.constant_(self.gate.bias, -3.0)
        self.last_table = None; self.last_gate = None

    def prior(self, inputs, roles):
        """(B, n, p) prior for the target row in the model's normalized units, gate (B,) from the best covariate's held-out R^2."""
        vals, masks = inputs["values"], inputs["masks"]; B, V, n, p = vals.shape
        obs = (~masks).to(vals.dtype); x = (vals * obs).reshape(B, V, n * p); o = obs.reshape(B, V, n * p)
        t_hist = (~masks[:, 0].all(-1)).sum(1) * p                                   # observed target steps (unmasked patches of row 0)
        th = (torch.arange(n * p, device=vals.device)[None, None] < t_hist[:, None, None]); w = o * th.to(vals.dtype)
        mu = (x * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp_min(1); sd = (((x - mu) ** 2 * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp_min(1)).sqrt() + 1e-6
        xn = ((x - mu) / sd) * o
        with torch.no_grad():
            per_cov, R, r2 = self.icr(xn, roles, o, t_hist)
        valid = o.sum(-1) > 8; r2 = torch.where(valid, r2, torch.full_like(r2, -1.0))
        g = torch.sigmoid(self.gate(r2.unsqueeze(-1))).squeeze(-1) * (roles == ROLE_FUTURE).to(vals.dtype)   # (B, V)
        prior = (per_cov * g[:, None, :]).sum(-1)                                                          # (B, n*p)
        self.last_table, self.last_gate = R.detach(), g.detach()
        return torch.nan_to_num(prior).view(B, n, p), g.max(1).values


class Policy(nn.Module):
    """Frozen TimesFM-3 + trainable grafts. forward(inputs, roles, cpm) -> the base model's output dict."""

    def __init__(self, base: TimesFM3Torch, layers=(10, 11, 12, 13, 14), lora_r=8, head_adapter=False):
        super().__init__()
        self.base = base
        for p in base.parameters(): p.requires_grad_(False)
        self.layers = [i for i in layers if i < len(base.transformer_stack.layers)]
        self.lora = nn.ModuleList()
        for i in self.layers:
            mha = base.transformer_stack.layers[i].var_attn
            for name in ("query_proj", "key_proj", "value_proj"):
                if lora_r > 0 and not isinstance(getattr(mha, name), LoRALinear):
                    lr = LoRALinear(getattr(mha, name), lora_r); setattr(mha, name, lr); self.lora.append(lr)
        self.head = HeadAdapter() if head_adapter else None
        self._ctx = None
        if self.head is not None:
            self.base.output_head.register_forward_hook(self._head_hook)

    def _head_hook(self, mod, args, out):
        if self._ctx is None: return out
        prior, gmax, is_t, o, qn = self._ctx
        B, n, p = prior.shape; flat = prior.reshape(B, n * p)
        nxt = torch.stack([torch.nn.functional.pad(flat[:, (i + 1) * p:(i + 1) * p + o], (0, max(0, o - flat[:, (i + 1) * p:(i + 1) * p + o].shape[1]))) for i in range(n)], 1)
        add = (self.head.alpha * gmax)[:, None, None] * nxt
        add = add[:, :, :, None].expand(B, n, o, qn).reshape(B, 1, n, o * qn) * is_t[:, :, None, None].to(out.dtype)
        return out + add

    def forward(self, inputs, roles, cpm):
        if self.head is not None:
            prior, gmax = self.head.prior(inputs, roles)
            self._ctx = (prior, gmax, roles == ROLE_TARGET, self.base.output_patch_len, self.base.num_quantiles)
        out = self.base(inputs, patch_cpm_mask=cpm); self._ctx = None
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("base.") or k.endswith(".A") or k.endswith(".B")}
