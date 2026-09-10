"""TimesFM-3 access: load the released checkpoint (or a tiny random twin for tests), build inputs the way
`TimesFM3Torch.decode` does but differentiably, read the horizon, and expose the quantile levels.

    base = load_model()                          # google/timesfm-3.0-pytorch, 331M, frozen
    base = tiny_model()                          # 2 layers, d 64, for CPU tests
    inputs, roles, cpm, n_ctx = build_inputs(target, past_only, future, H)
    q = horizon_quantiles(base(inputs, patch_cpm_mask=cpm), n_ctx, H)      # (B, V, H, 9)
"""
from __future__ import annotations

import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party"))
from timesfm3 import configs  # noqa: E402
from timesfm3.model import TimesFM3Torch  # noqa: E402

PATCH = 32
QLEVELS = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ROLE_TARGET, ROLE_PAST, ROLE_FUTURE = 0, 1, 2


def tiny_model(n_layers=2, d=64, heads=2, seed=0) -> TimesFM3Torch:
    torch.manual_seed(seed)
    rb = configs.ResidualBlockConfig(hidden_dims=d, output_dims=d, use_bias=False, activation="relu")
    tc = configs.StackedTransformersConfig(num_layers=n_layers, transformer=configs.TransformerConfig(
        model_dims=d, hidden_dims=d, num_heads=heads, attention_norm="rms", feedforward_norm="rms", qk_norm="rms",
        use_bias=False, use_rope_seq=True, use_rope_var=False, ff_activation="relu", deterministic=True))
    return TimesFM3Torch(residual_block_config=rb, transformer_config=tc)


def load_model(repo_or_path="google/timesfm-3.0-pytorch", device="cpu") -> TimesFM3Torch:
    return TimesFM3Torch.from_pretrained(repo_or_path).to(device).eval()


def build_inputs(target, past_only, future, horizon, patch=PATCH):
    """target (B,U,T); past_only (B,K1,T) | None; future (B,K2,T+H) | None -> inputs, roles (B,V), cpm (B,n), n_ctx.
    Target-like rows are masked over the horizon; known-future rows are visible; context left-padded to a patch multiple."""
    B, U, T = target.shape; dev = target.device
    pad = (-T) % patch
    if pad:
        target = torch.nn.functional.pad(target, (pad, 0))
        if past_only is not None: past_only = torch.nn.functional.pad(past_only, (pad, 0))
        if future is not None: future = torch.nn.functional.pad(future, (pad, 0))
    T += pad; Hp = ((horizon + 63) // 64) * 64
    rows, masks, roles = [], [], []
    hor_mask = torch.ones(B, 1, Hp, dtype=torch.bool, device=dev); ctx_mask = torch.zeros(B, 1, T, dtype=torch.bool, device=dev); ctx_mask[:, :, :pad] = True
    for u in range(U):
        rows.append(torch.cat([target[:, u:u + 1], torch.zeros(B, 1, Hp, device=dev)], -1)); masks.append(torch.cat([ctx_mask, hor_mask], -1)); roles.append(ROLE_TARGET)
    for k in range(0 if past_only is None else past_only.shape[1]):
        rows.append(torch.cat([past_only[:, k:k + 1], torch.zeros(B, 1, Hp, device=dev)], -1)); masks.append(torch.cat([ctx_mask, hor_mask], -1)); roles.append(ROLE_PAST)
    for k in range(0 if future is None else future.shape[1]):
        f = future[:, k:k + 1]; fpad = Hp - (f.shape[-1] - T)
        f = torch.nn.functional.pad(f, (0, fpad)) if fpad > 0 else f[..., :T + Hp]
        fm = torch.zeros(B, 1, T + Hp, dtype=torch.bool, device=dev); fm[:, :, :pad] = True
        if fpad > 0: fm[:, :, -fpad:] = True
        rows.append(f); masks.append(fm); roles.append(ROLE_FUTURE)
    vals = torch.cat(rows, 1); msk = torch.cat(masks, 1); V = vals.shape[1]; n = (T + Hp) // patch; n_ctx = T // patch
    roles_t = torch.tensor(roles, device=dev)[None].expand(B, -1)
    inputs = {"values": vals.reshape(B, V, n, patch), "masks": msk.reshape(B, V, n, patch), "patch_is_target": (roles_t != ROLE_FUTURE)[:, :, None].expand(B, V, n)}
    cpm = torch.zeros(B, n, dtype=torch.bool, device=dev); cpm[:, n_ctx:] = True
    return inputs, roles_t, cpm, n_ctx


def horizon_quantiles(out, n_ctx, horizon):
    """(B, V, horizon, 9) from the last context patch's 64-step head (horizon <= 64; use decode() beyond)."""
    assert horizon <= 64
    return out["logits"][:, :, n_ctx - 1, :horizon, :]
