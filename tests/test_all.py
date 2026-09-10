"""CPU test of every component on the tiny model (< 2 minutes):  python tests/test_all.py"""
import os, subprocess, sys, tempfile
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tsfm_rl.model import tiny_model, build_inputs, horizon_quantiles  # noqa: E402
from tsfm_rl import rewards as R  # noqa: E402
from tsfm_rl.data import EpisodeBank, make_synthetic  # noqa: E402
from tsfm_rl.env import ForecastEnv, ContextAction, base_fn, STRATEGIES  # noqa: E402
from tsfm_rl.adapters import Policy  # noqa: E402
from tsfm_rl.evaluate import evaluate  # noqa: E402


def test_rewards():
    torch.manual_seed(0); q = torch.sort(torch.randn(4, 64, 9), -1).values; y = torch.randn(4, 64)
    for name, fn in R.REWARDS.items():
        r = fn(q, y, q); assert r.shape == (4,) and torch.isfinite(r).all(), name
    assert R.composite(q, y, q, per_step=True).shape == (4, 64)
    s = R.sample_forecasts(q, 5); assert s.shape == (5, 4, 64)
    qg = q.clone().requires_grad_(); lp = R.logpdf(qg, s[0]); lp.sum().backward(); assert torch.isfinite(qg.grad).all()
    assert (R.coverage(q, q[..., 4]) == 1).all() and (R.soft_dtw(y, y) < R.soft_dtw(y, y + 1.0)).all()   # soft-min: identical paths score lowest, not zero
    print("rewards ok")


def test_env_policy():
    m = tiny_model(); bank = EpisodeBank.synthetic(16); env = ForecastEnv(bank, "composite")
    eps = env.reset(4); q, y, info = env.forecast(base_fn(m)); assert q.shape == (4, 64, 9) and info["cond_mean"] is not None
    props = env.propose(eps[0]); assert len(props) == 8 and all(k in STRATEGIES for k in ("shape", "lagreg"))
    q2, _, _ = env.forecast(base_fn(m), [p[3] for p in [env.propose(e) for e in eps]]); assert q2.shape == q.shape
    pol = Policy(m, layers=(0, 1), lora_r=2, head_adapter=True); q3, _, _ = env.forecast(pol)
    assert torch.allclose(q3, q, atol=1e-5), "grafts must be zero at init"
    R.pinball(q3, y).mean().backward(); assert pol.lora[0].B.grad.abs().sum() > 0 and pol.head.alpha.grad is not None
    csv = os.path.join(ROOT, "data", "ETTh1.csv")
    if os.path.exists(csv):
        b2 = EpisodeBank.from_csv(csv, n=8); e2 = ForecastEnv(b2); e2.reset(4); qq, yy, _ = e2.forecast(base_fn(m)); assert qq.shape == (4, 64, 9)
    met = evaluate(base_fn(m), env, n=8, batch=4, ref_policy=base_fn(m)); assert "mase" in met and "r2_cond" in met
    print("env + policy + evaluate ok")


def test_algorithms():
    d = tempfile.mkdtemp(); env = {**os.environ, "PYTHONPATH": ROOT}
    for algo, extra in (("none", []), ("sft", ["--reward", "pinball"]), ("grpo", ["--reward", "composite"]), ("grpo", ["--reward", "mse", "--anchor", "0.1", "--per_step"]),
                        ("ppo", ["--reward", "crps"]), ("preference", ["--reward", "skill"]), ("context", []), ("context_sft", [])):
        r = subprocess.run([sys.executable, "scripts/run_matrix.py", "--tiny", "--algo", algo, "--bank", "synth", "--episodes", "24", "--heldout", "8", "--steps", "3", "--batch", "4", "--K", "3",
                            "--layers", "0", "1", "--lora_r", "2", "--log_every", "1", "--out", f"{d}/{algo}_{len(extra)}", *extra], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (algo, extra, r.stderr[-1500:])
        assert os.path.exists(f"{d}/{algo}_{len(extra)}/eval.json")
    print("algorithms ok")


if __name__ == "__main__":
    test_rewards(); test_env_policy(); test_algorithms(); print("ALL TESTS PASSED")
