"""One interface over several frozen time-series foundation models, so every trainer, reward and evaluator runs unchanged
on TimesFM-3, Chronos-Bolt, Chronos-2 and the token-sampling Chronos-T5.

    f = make_forecaster("timesfm3" | "chronos-t5-small" | "chronos-bolt-small" | "chronos-2", device, layers, lora_r, head_adapter)
    ctx = env.build(actions)                      # {"target": (B,L), "po": (B,n,L)|None, "kf": (B,m,L+H)|None, "po_pad", "kf_pad", "H"}
    q = f.quantiles(ctx)                          # (B,H,9) at QLEVELS; differentiable for quantile heads, empirical (no grad) for Chronos-T5
    samples, lp_beh = f.sample(ctx, K, g)         # (K,B,H) numeric trajectories and their per-step log-prob under the current weights (no grad)
    lp = f.logprob(ctx, samples)                  # (K,B,H) differentiable per-step log-density / token log-prob
    loss = f.sft_loss(ctx, y, "pinball")          # the model's native supervised loss (pinball / CE on tokens)
    with f.swap(f.frozen_state()): ...            # the released model (grafts zeroed)
Grafts: LoRA on attention q/k/v of the chosen blocks (all models), optional head adapter (TimesFM-3 only). Zero at init.
"""
from __future__ import annotations

import contextlib, math, re
import torch
import torch.nn as nn

from .model import load_model, build_inputs, horizon_quantiles, PATCH, QLEVELS
from .adapters import Policy, LoRALinear
from .rewards import sample_forecasts, logpdf, LOSSES

MODELS = {"timesfm3": "google/timesfm-3.0-pytorch",
          **{f"chronos-t5-{s}": f"amazon/chronos-t5-{s}" for s in ("tiny", "mini", "small", "base", "large")},
          **{f"chronos-bolt-{s}": f"amazon/chronos-bolt-{s}" for s in ("tiny", "mini", "small", "base")},
          "chronos-2": "amazon/chronos-2"}


class Forecaster(nn.Module):
    name = "base"; stochastic = False; supports_covariates = False

    # ---- to implement --------------------------------------------------------------------------------------------
    def quantiles(self, ctx): raise NotImplementedError
    def sft_loss(self, ctx, y, loss_name="pinball"): raise NotImplementedError

    # ---- defaults for quantile heads: sampling through the piecewise-linear CDF -----------------------------------
    def sample(self, ctx, K, g=None, q=None):
        with torch.no_grad():
            q = self.quantiles(ctx) if q is None else q.detach(); s = sample_forecasts(q, K, g)
            return s, torch.stack([logpdf(q, s[k]) for k in range(K)])

    def logprob(self, ctx, samples, q=None):
        q = self.quantiles(ctx) if q is None else q
        return torch.stack([logpdf(q, samples[k]) for k in range(samples.shape[0])])

    # ---- grafts ----------------------------------------------------------------------------------------------------
    def trainable_parameters(self): return [p for p in self.parameters() if p.requires_grad]

    def state(self):
        """The trainable grafts, keyed by canonical parameter name (aliases of the same tensor are deduplicated)."""
        return {k: v for k, v in self.named_parameters() if v.requires_grad}

    def frozen_state(self):
        return {k: (torch.zeros_like(v) if k.endswith((".B", ".alpha")) else v) for k, v in self.state().items()}

    def load_grafts(self, state):
        """Load a saved state() (also accepts the older Policy.state() keys without the 'policy.' prefix)."""
        own = self.state(); out = {}
        for k, v in state.items():
            kk = k if k in own else ("policy." + k if "policy." + k in own else None)
            if kk is not None: out[kk] = v
        with torch.no_grad():
            for k, v in out.items(): own[k].copy_(v.to(own[k].device, own[k].dtype))
        return len(out)

    @contextlib.contextmanager
    def swap(self, state):
        """Temporarily replace graft tensors by *object substitution* (not in-place copy), so a swapped forward is safe
        even after the policy's autograd graph has been built and the optimizer keeps its parameter references."""
        saved = []
        for k, v in state.items():
            mod = self.get_submodule(k.rsplit(".", 1)[0]); attr = k.rsplit(".", 1)[1]
            if attr in mod._parameters:
                old = mod._parameters[attr]; saved.append((mod, attr, old, True)); mod._parameters[attr] = nn.Parameter(v.detach().clone().to(old.device, old.dtype), requires_grad=False)
            elif attr in mod._buffers:
                old = mod._buffers[attr]; saved.append((mod, attr, old, False)); mod._buffers[attr] = v.detach().clone().to(old.device, old.dtype)
        try: yield
        finally:
            for mod, attr, old, is_param in reversed(saved):
                if is_param: mod._parameters[attr] = old
                else: mod._buffers[attr] = old


def add_lora(root: nn.Module, layers, r=8, block_re=r"\.blocks?\.(\d+)\."):
    """Wrap attention q/k/v Linear modules of the chosen block indices with LoRA (HF T5 / Chronos-2 naming)."""
    wrapped = []
    for name, mod in list(root.named_modules()):
        if not isinstance(mod, nn.Linear) or not name.split(".")[-1] in ("q", "k", "v"): continue
        m = re.search(block_re, "." + name + "."); idx = int(m.group(1)) if m else None
        if idx is None or idx not in layers: continue
        parent = root.get_submodule(name.rsplit(".", 1)[0]); lr = LoRALinear(mod, r); setattr(parent, name.split(".")[-1], lr); wrapped.append(lr)
    return wrapped


def _pick_layers(layers, n_blocks):
    ls = [l for l in layers if l < n_blocks]
    return ls or list(range(max(0, n_blocks - max(1, n_blocks // 3)), n_blocks))


# ----------------------------------------------------------------------------- TimesFM-3
class TimesFMForecaster(Forecaster):
    name = "timesfm3"; supports_covariates = True

    def __init__(self, base, layers=(10, 11, 12, 13, 14), lora_r=8, head_adapter=False):
        super().__init__(); self.policy = Policy(base, layers=layers, lora_r=lora_r, head_adapter=head_adapter)

    def _inputs(self, ctx):
        B, L = ctx["target"].shape; po, kf = ctx["po"], ctx["kf"]
        inputs, roles, cpm, n_ctx = build_inputs(ctx["target"][:, None], po, kf, ctx["H"])
        if po is not None or kf is not None:
            pads = [torch.zeros(B, 1, dtype=torch.bool, device=ctx["target"].device)]
            if po is not None: pads.append(ctx["po_pad"])
            if kf is not None: pads.append(ctx["kf_pad"])
            inputs["masks"] = inputs["masks"] | torch.cat(pads, 1)[:, :, None, None]
        return inputs, roles, cpm, n_ctx

    def quantiles(self, ctx):
        inputs, roles, cpm, n_ctx = self._inputs(ctx); out = self.policy(inputs, roles, cpm)
        return horizon_quantiles(out, n_ctx, ctx["H"])[:, 0].float()

    def sft_loss(self, ctx, y, loss_name="pinball"):
        q = self.quantiles(ctx); return LOSSES[loss_name](q, y).mean(), q


# ----------------------------------------------------------------------------- Chronos-Bolt (quantile head, univariate)
class ChronosBoltForecaster(Forecaster):
    name = "chronos-bolt"

    def __init__(self, repo, device="cpu", layers=(), lora_r=8):
        super().__init__()
        from chronos import ChronosBoltPipeline
        self.pipe = ChronosBoltPipeline.from_pretrained(repo, device_map=device, dtype=torch.float32); self.model = self.pipe.model
        for p in self.model.parameters(): p.requires_grad_(False)
        n = len(self.model.encoder.block); self.layers = _pick_layers(layers, n); self.lora = nn.ModuleList(add_lora(self.model, self.layers, lora_r) if lora_r > 0 else [])
        self.qidx = _level_index(self.pipe.quantiles); self.H_model = self.pipe.model_prediction_length

    def quantiles(self, ctx):
        H = ctx["H"]; assert H <= self.H_model, f"H={H} > model prediction length {self.H_model}: use stitching"
        q = self.model(context=ctx["target"]).quantile_preds[:, :, :H]                   # (B, Q, H) in input units
        return q[:, self.qidx].permute(0, 2, 1).float()

    def sft_loss(self, ctx, y, loss_name="pinball"):
        q = self.quantiles(ctx); return LOSSES[loss_name](q, y).mean(), q


# ----------------------------------------------------------------------------- Chronos-2 (quantile head, covariates)
class Chronos2Forecaster(Forecaster):
    name = "chronos-2"; supports_covariates = True

    def __init__(self, repo="amazon/chronos-2", device="cpu", layers=(), lora_r=8):
        super().__init__()
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained(repo, device_map=device, dtype=torch.float32); self.model = self.pipe.model
        for p in self.model.parameters(): p.requires_grad_(False)
        n = len(self.model.encoder.blocks) if hasattr(self.model.encoder, "blocks") else len(self.model.encoder.block)
        self.layers = _pick_layers(layers, n); self.lora = nn.ModuleList(add_lora(self.model, self.layers, lora_r) if lora_r > 0 else [])
        self.qidx = _level_index(self.pipe.quantiles); self.P = self.pipe.model_output_patch_size

    def quantiles(self, ctx):
        tgt, po, kf, H = ctx["target"], ctx["po"], ctx["kf"], ctx["H"]; B, L = tgt.shape; dev = tgt.device
        rows, fut, gid, is_t = [], [], [], []
        for b in range(B):
            rows.append(tgt[b]); fut.append(torch.full((H,), float("nan"), device=dev)); gid.append(b); is_t.append(True)
            if po is not None:
                for j in range(po.shape[1]):
                    if not ctx["po_pad"][b, j]: rows.append(po[b, j]); fut.append(torch.full((H,), float("nan"), device=dev)); gid.append(b); is_t.append(False)
            if kf is not None:
                for j in range(kf.shape[1]):
                    if not ctx["kf_pad"][b, j]: rows.append(kf[b, j, :L]); fut.append(kf[b, j, L:L + H]); gid.append(b); is_t.append(False)
        n_out = math.ceil(H / self.P)
        out = self.model(context=torch.stack(rows), group_ids=torch.tensor(gid, device=dev), future_covariates=torch.stack(fut), num_output_patches=n_out)
        q = out.quantile_preds[torch.tensor(is_t, device=dev)][:, :, :H]                  # (B, Q, H)
        return q[:, self.qidx].permute(0, 2, 1).float()

    def sft_loss(self, ctx, y, loss_name="pinball"):
        q = self.quantiles(ctx); return LOSSES[loss_name](q, y).mean(), q


# ----------------------------------------------------------------------------- Chronos-T5 (token sampler: a real stochastic policy)
class ChronosT5Forecaster(Forecaster):
    name = "chronos-t5"; stochastic = True

    def __init__(self, repo, device="cpu", layers=(), lora_r=8, n_eval_samples=20):
        super().__init__()
        from chronos import ChronosPipeline
        self.pipe = ChronosPipeline.from_pretrained(repo, device_map=device, dtype=torch.float32)
        self.t5 = self.pipe.model.model; self.tok = self.pipe.tokenizer; self.cfg = self.pipe.model.config; self.n_eval = n_eval_samples
        for p in self.t5.parameters(): p.requires_grad_(False)
        n = len(self.t5.encoder.block); self.layers = _pick_layers(layers, n); self.lora = nn.ModuleList(add_lora(self.t5, self.layers, lora_r) if lora_r > 0 else [])

    def _encode(self, ctx):
        ids, am, scale = self.tok.context_input_transform(ctx["target"].detach().cpu()); dev = ctx["target"].device
        return ids.to(dev), am.to(dev), scale.to(dev)

    def _tokens(self, values, scale):                    # numeric (N, H) -> token ids (N, H) with the context scale
        ids, _, _ = self.tok._input_transform(context=values.detach().cpu(), scale=scale.detach().cpu()); return ids.to(values.device)

    @torch.no_grad()
    def _generate(self, ctx, K):
        ids, am, scale = self._encode(ctx); H = ctx["H"]
        toks = self.pipe.model(ids, am, prediction_length=H, num_samples=K, temperature=1.0, top_k=0, top_p=1.0)   # (B, K, H) exact sampling
        vals = self.tok.output_transform(toks.cpu(), scale.cpu()).to(ctx["target"].device)                             # (B, K, H)
        return vals.permute(1, 0, 2).contiguous(), scale

    def quantiles(self, ctx):
        s, _ = self._generate(ctx, self.n_eval); return torch.quantile(s, QLEVELS.to(s.device), dim=0).permute(1, 2, 0).float()   # (B,H,9)

    def sample(self, ctx, K, g=None, q=None):
        s, _ = self._generate(ctx, K)
        with torch.no_grad(): lp = self.logprob(ctx, s)
        return s, lp

    def logprob(self, ctx, samples, q=None):
        K, B, H = samples.shape; ids, am, scale = self._encode(ctx)
        lab = self._tokens(samples.reshape(K * B, H), scale.repeat(K))                                                 # (K*B, H)
        dec_in = torch.cat([torch.full((K * B, 1), self.t5.config.decoder_start_token_id, device=lab.device, dtype=lab.dtype), lab[:, :-1]], 1)
        logits = self.t5(input_ids=ids.repeat(K, 1), attention_mask=am.repeat(K, 1), decoder_input_ids=dec_in).logits    # (K*B, H, V)
        return torch.log_softmax(logits.float(), -1).gather(-1, lab[..., None])[..., 0].reshape(K, B, H)

    def sft_loss(self, ctx, y, loss_name="ce"):
        ids, am, scale = self._encode(ctx); lab = self._tokens(y, scale)
        return self.t5(input_ids=ids, attention_mask=am, labels=lab).loss, None


def _level_index(levels):
    lv = torch.tensor(levels, dtype=torch.float32); idx = [int((lv - l).abs().argmin()) for l in QLEVELS.tolist()]
    assert all(abs(float(lv[i]) - l) < 1e-3 for i, l in zip(idx, QLEVELS.tolist())), f"model quantile levels {levels} do not contain {QLEVELS.tolist()}"
    return idx


def make_forecaster(name="timesfm3", device="cpu", layers=(10, 11, 12, 13, 14), lora_r=8, head_adapter=False, tiny=False):
    if tiny or name == "tiny":
        from .model import tiny_model; return TimesFMForecaster(tiny_model(), layers=layers, lora_r=lora_r, head_adapter=head_adapter).to(device)
    repo = MODELS.get(name, name)
    if name == "timesfm3" or "timesfm" in repo: return TimesFMForecaster(load_model(repo, device), layers=layers, lora_r=lora_r, head_adapter=head_adapter).to(device)
    if "chronos-bolt" in repo: return ChronosBoltForecaster(repo, device, layers, lora_r)
    if "chronos-2" in repo: return Chronos2Forecaster(repo, device, layers, lora_r)
    if "chronos-t5" in repo: return ChronosT5Forecaster(repo, device, layers, lora_r)
    raise ValueError(name)
