"""Chronos adapters on the tiny checkpoints (downloads ~40 MB on first run):  python3 tests/test_chronos.py"""
import os, subprocess, sys, tempfile
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tsfm_rl.forecasters import make_forecaster  # noqa: E402
from tsfm_rl.data import EpisodeBank  # noqa: E402
from tsfm_rl.env import ForecastEnv, ContextAction  # noqa: E402
from tsfm_rl.evaluate import evaluate  # noqa: E402


def check(name):
    fc = make_forecaster(name, "cpu", layers=(2, 3), lora_r=2); bank = EpisodeBank.synthetic(8); env = ForecastEnv(bank, "crps"); env.reset(2)
    ctx = env.build([ContextAction.native(e) for e in env.eps]); y = env.gold()
    q = fc.quantiles(ctx); assert q.shape == (2, 64, 9), q.shape
    s, lp = fc.sample(ctx, 3); assert s.shape == (3, 2, 64) and lp.shape == (3, 2, 64)
    lp2 = fc.logprob(ctx, s); assert lp2.shape == (3, 2, 64) and torch.isfinite(lp2).all()
    if not fc.stochastic: assert torch.allclose(lp2, lp, atol=1e-4)
    else: assert torch.allclose(lp2, lp, atol=1e-4), "teacher-forced log-prob must reproduce the sampling log-prob"
    loss, _ = fc.sft_loss(ctx, y, "ce" if fc.stochastic else "pinball"); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fc.trainable_parameters()), "no gradient reached the LoRA grafts"
    with fc.swap(fc.frozen_state()): q0 = fc.quantiles(ctx)
    n = len(fc.state()); m = evaluate(fc, env, n=4, batch=2, ref_state=fc.frozen_state()); assert "crps" in m
    print(f"{name}: ok  quantiles {tuple(q.shape)}  graft tensors {n}  layers {fc.layers}  stochastic {fc.stochastic}")


def cells(name):
    d = tempfile.mkdtemp(); env = {**os.environ, "PYTHONPATH": ROOT}
    for algo, extra in (("sft", []), ("grpo", ["--reward", "mse"]), ("grpo", ["--set_reward", "--adv_norm", "loo", "--kl_dir", "forward"]), ("ppo", ["--reward", "crps", "--kl_dir", "k3"])):
        r = subprocess.run([sys.executable, "scripts/run_matrix.py", "--model", name, "--algo", algo, "--bank", "synth", "--episodes", "16", "--heldout", "4", "--steps", "2", "--batch", "2", "--K", "3",
                            "--layers", "2", "3", "--lora_r", "2", "--log_every", "1", "--out", f"{d}/{algo}_{len(extra)}", *extra], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (name, algo, extra, r.stderr[-2000:])
    print(f"{name}: cells ok")


if __name__ == "__main__":
    for name in ("chronos-bolt-tiny", "chronos-t5-tiny"): check(name); cells(name)
    print("CHRONOS TESTS PASSED")
