#!/usr/bin/env python3
"""Figures from runs/<env>/*/eval_ci.json (or eval.json):  python3 scripts/make_figures.py runs/synth runs/electricity --out figures
  <env>_deltas.png     held-out MASE / CRPS / coverage / width per cell, policy minus frozen, paired-bootstrap 95 % CIs
  <env>_calibration.png   80 % coverage vs interval width per cell (nominal line)
  <env>_curves.png     training curves (held-in pinball or reward vs step): the sample-efficiency view
  <env>_fans.png       held-out windows: truth, frozen quantile fan, SFT fan, best RL fan
"""
from __future__ import annotations

import glob, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["native", "sft_pinball", "sft_pinball_seed1", "grpo_mse", "grpo_mse_anchor", "grpo_mse_perstep", "pref_mse", "ppo_mse", "ppo_composite",
         "grpo_crps", "grpo_skill", "grpo_composite", "grpo_composite_head", "context_ipo"]
LABEL = {"native": "M0 frozen", "sft_pinball": "M1 SFT pinball", "sft_pinball_seed1": "M1 SFT (seed 1)", "grpo_mse": "M2 GRPO −MSE (TS-GRPO)", "grpo_mse_anchor": "M3 +anchor (GTN-R)",
         "grpo_mse_perstep": "M4 per-step (TimeRFT)", "pref_mse": "M5 pref. IPO (TPO)", "ppo_mse": "M6 PPO −MSE", "ppo_composite": "M6 PPO composite",
         "grpo_crps": "GRPO −CRPS", "grpo_skill": "GRPO skill", "grpo_composite": "M7 GRPO composite (ours)", "grpo_composite_head": "M7 + head adapter", "context_ipo": "M8 context policy"}
COLOR = {"sft": "#6c757d", "grpo_mse": "#c0392b", "grpo_mse_anchor": "#c0392b", "grpo_mse_perstep": "#c0392b", "pref": "#e67e22", "ppo": "#8e44ad", "ours": "#1f77b4", "native": "#2a9d8f"}


def color(run):
    if run.startswith("sft"): return COLOR["sft"]
    if run in ("grpo_crps", "grpo_skill", "grpo_composite", "grpo_composite_head"): return COLOR["ours"]
    if run.startswith("grpo"): return COLOR["grpo_mse"]
    if run.startswith("ppo"): return COLOR["ppo"]
    if run.startswith("pref"): return COLOR["pref"]
    return COLOR["native"]


def load(root):
    cells = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        f = os.path.join(d, "eval_ci.json"); f2 = os.path.join(d, "eval.json")
        if os.path.exists(f): cells[os.path.basename(d)] = json.load(open(f))
        elif os.path.exists(f2):
            e = json.load(open(f2)); cells[os.path.basename(d)] = {"cfg": e["cfg"], "reference": e["reference"], "policy": e["policy"], "delta": None}
    return {k: cells[k] for k in ORDER if k in cells}


def fig_deltas(env, cells, out):
    keys = [("mase", "MASE"), ("crps", "CRPS"), ("coverage80", "80 % coverage"), ("width", "80 % interval width")]
    runs = [r for r in cells if r != "native"]; fig, axes = plt.subplots(1, 4, figsize=(17, 0.45 * len(runs) + 2.2), sharey=True)
    for ax, (k, title) in zip(axes, keys):
        for i, r in enumerate(runs):
            c = cells[r]; ref = c["reference"][k]
            if c.get("delta"): m, lo, hi = c["delta"][k]; ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=color(r), capsize=3)
            else: ax.plot(c["policy"][k] - ref, i, "o", color=color(r), mfc="none")
        ax.axvline(0, color="k", lw=0.8); ax.set_title(f"{title}\nfrozen = {cells[next(iter(cells))]['reference'][k]:.3f}"); ax.grid(alpha=0.3)
        ax.set_xlabel("policy − frozen (held-out)")
    axes[0].set_yticks(range(len(runs))); axes[0].set_yticklabels([LABEL.get(r, r) for r in runs]); axes[0].invert_yaxis()
    fig.suptitle(f"{env}: held-out change vs frozen TimesFM-3 after 300 CPU steps (paired bootstrap 95 % CI over 128 windows; left of 0 = better for MASE/CRPS/width; coverage should stay at 0)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_calibration(env, cells, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, c in cells.items():
        ax.scatter(c["policy"]["width"], c["policy"]["coverage80"], color=color(r), s=60 if r == "native" else 40, marker="*" if r == "native" else "o", zorder=3)
        ax.annotate(LABEL.get(r, r), (c["policy"]["width"], c["policy"]["coverage80"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.8, color="k", ls="--", lw=0.8, label="nominal 80 %"); ax.set_xlabel("mean 80 % interval width (normalized units)"); ax.set_ylabel("empirical 80 % coverage (held-out)")
    ax.set_title(f"{env}: calibration of the post-trained models"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_curves(env, root, cells, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for r in cells:
        f = os.path.join(root, r, "log.jsonl")
        if not os.path.exists(f): continue
        recs = [json.loads(l) for l in open(f)]; steps = np.array([x["step"] for x in recs])
        for ax, key in zip(axes, ("pinball", "reward")):
            key2 = key if key in recs[0] else ("loss" if key == "pinball" and "loss" in recs[0] and cells[r]["cfg"]["algo"] == "sft" else ("r_mean" if key == "reward" and "r_mean" in recs[0] else None))
            if key2 is None: continue
            v = np.array([x.get(key2, np.nan) for x in recs], dtype=float); w = 20; sm = np.convolve(v, np.ones(w) / w, mode="valid")
            ax.plot(steps[w - 1:], sm, color=color(r), label=LABEL.get(r, r), lw=1.2, alpha=0.9)
    axes[0].set_title("training-batch pinball loss (20-step moving average)"); axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3)
    axes[1].set_title("training-batch reward under each cell's own reward (not comparable across rewards)"); axes[1].set_xlabel("step"); axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=7); fig.suptitle(f"{env}: sample efficiency during post-training"); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_fans(env, cells, out, picks=("sft_pinball", "grpo_mse", "grpo_composite")):
    have = [r for r in picks if r in cells and cells[r].get("fans")]
    if not have or not cells.get("native", {}).get("ref_fans"): return
    nwin = len(cells["native"]["ref_fans"]); fig, axes = plt.subplots(nwin, len(have) + 1, figsize=(4.2 * (len(have) + 1), 2.6 * nwin), squeeze=False)
    cols = [("native", cells["native"]["ref_fans"])] + [(r, cells[r]["fans"]) for r in have]
    for j, (r, fans) in enumerate(cols):
        for i in range(nwin):
            ax = axes[i, j]; f = fans[i]; h = np.array(f["history"]); y = np.array(f["truth"]); q = np.array(f["q"]); T = len(h); t = np.arange(T, T + len(y))
            ax.plot(np.arange(T), h, color="k", lw=0.8); ax.plot(t, y, color="k", lw=1.2, label="truth")
            ax.fill_between(t, q[:, 0], q[:, 8], color=color(r), alpha=0.18, label="10–90 %"); ax.fill_between(t, q[:, 2], q[:, 6], color=color(r), alpha=0.3)
            ax.plot(t, q[:, 4], color=color(r), lw=1.3, label="median"); ax.axvline(T, color="gray", lw=0.6, ls=":")
            if i == 0: ax.set_title(LABEL.get(r, r), fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7); fig.suptitle(f"{env}: held-out windows (last 128 steps of history, 64-step forecast)"); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    roots = [a for a in sys.argv[1:] if not a.startswith("--")]; out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures"; os.makedirs(out, exist_ok=True)
    for root in roots:
        env = os.path.basename(root.rstrip("/")); cells = load(root)
        if not cells: print("no cells in", root); continue
        fig_deltas(env, cells, f"{out}/{env}_deltas.png"); fig_calibration(env, cells, f"{out}/{env}_calibration.png"); fig_curves(env, root, cells, f"{out}/{env}_curves.png"); fig_fans(env, cells, f"{out}/{env}_fans.png")
        print("wrote", env, "figures ->", out)


if __name__ == "__main__":
    main()
