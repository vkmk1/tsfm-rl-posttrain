"""CPU test of every component on the tiny model (< 2 minutes):  python tests/test_all.py"""
import os, subprocess, sys, tempfile
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tsfm_rl.model import tiny_model, build_inputs, horizon_quantiles  # noqa: E402
from tsfm_rl import rewards as R  # noqa: E402
from tsfm_rl.data import EpisodeBank, make_synthetic  # noqa: E402
from tsfm_rl.env import ForecastEnv, ContextAction, STRATEGIES  # noqa: E402
from tsfm_rl.forecasters import TimesFMForecaster  # noqa: E402
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
    m = tiny_model(); bank = EpisodeBank.synthetic(16); env = ForecastEnv(bank, "composite"); fc = TimesFMForecaster(m, layers=(0, 1), lora_r=2, head_adapter=True)
    eps = env.reset(4)
    with fc.swap(fc.frozen_state()): q, y, info = env.forecast(fc)
    assert q.shape == (4, 64, 9) and info["cond_mean"] is not None
    props = env.propose(eps[0]); assert len(props) == 8 and all(k in STRATEGIES for k in ("shape", "lagreg"))
    q2, _, _ = env.forecast(fc, [p[3] for p in [env.propose(e) for e in eps]]); assert q2.shape == q.shape
    q3, _, _ = env.forecast(fc); assert torch.allclose(q3, q, atol=1e-5), "grafts must be zero at init"
    R.pinball(q3, y).mean().backward(); assert fc.policy.lora[0].B.grad.abs().sum() > 0 and fc.policy.head.alpha.grad is not None
    s, lp = fc.sample(env.build([ContextAction.native(e) for e in eps]), 3); assert s.shape == (3, 4, 64) and lp.shape == (3, 4, 64)
    csv = os.path.join(ROOT, "data", "ETTh1.csv")
    if os.path.exists(csv):
        b2 = EpisodeBank.from_csv(csv, n=8, region=(0.7, 1.0)); e2 = ForecastEnv(b2); e2.reset(4); qq, yy, _ = e2.forecast(fc); assert qq.shape == (4, 64, 9)
    met = evaluate(fc, env, n=8, batch=4, ref_state=fc.frozen_state()); assert "mase" in met and "r2_cond" in met
    print("env + policy + evaluate ok")


def test_agent_parts():
    from tsfm_rl.agent import Fallback, pool_quantiles, _cdf_eval
    torch.manual_seed(0); h = torch.sin(torch.arange(256).float() / 24 * 6.283)[None].repeat(3, 1) + 0.1 * torch.randn(3, 256)
    fb = Fallback(); q = fb(h, 64); assert q.shape == (3, 64, 9) and (q[..., 1:] >= q[..., :-1]).all()
    qa = torch.sort(torch.randn(3, 64, 9), -1).values; qb = qa + 1.0
    p1 = pool_quantiles(qa, qb, torch.ones(3)); p0 = pool_quantiles(qa, qb, torch.zeros(3))
    assert (p1 - qa).abs().max() < 1e-5 and (p0 - qb).abs().max() < 1e-5, "pool at w=1 / w=0 must recover the components exactly"
    pm = pool_quantiles(qa, qb, torch.full((3,), 0.5)); assert ((pm >= torch.minimum(qa, qb) - 1e-5) & (pm <= torch.maximum(qa, qb) + 1e-5)).all()
    # the pooled median of two shifted copies at w=0.5 lies between the two medians and the pooled CDF is monotone
    assert (pm[..., 1:] >= pm[..., :-1] - 1e-6).all()
    from tsfm_rl.agent import TrustAgent, AgentForecaster
    from tsfm_rl.forecasters import make_forecaster
    fc = make_forecaster("tiny"); bank2 = EpisodeBank.synthetic(32); env2 = ForecastEnv(bank2, "crps")
    class Forced(TrustAgent):
        def forward(self, feats):
            B, P, _ = feats.shape; W = len(self.w_grid); z = torch.zeros(B, P * W); z[:, W - 1] = 10.0; return z
    ref = evaluate(fc, env2, n=16, batch=8); pol = evaluate(AgentForecaster(Forced(), fc, env2, fb, K=6), env2, n=16, batch=8)
    assert all(abs(ref[k] - pol[k]) < 1e-4 for k in ("mase", "crps", "coverage80")), "native/full-trust agent must reproduce the oracle exactly through the evaluation path"
    print("agent parts ok")


def test_algorithms():
    d = tempfile.mkdtemp(); env = {**os.environ, "PYTHONPATH": ROOT}
    for algo, extra in (("none", []), ("sft", ["--reward", "pinball"]), ("grpo", ["--reward", "composite"]), ("grpo", ["--reward", "mse", "--anchor", "0.1", "--per_step"]),
                        ("ppo", ["--reward", "crps"]), ("preference", ["--reward", "skill"]), ("context", []), ("context_sft", []),
                        ("grpo", ["--set_reward", "--adv_norm", "loo", "--sampler", "ema", "--ema_decay", "0.9", "--kl_dir", "forward", "--eval_ema", "--save_every", "2"]),
                        ("grpo", ["--reward", "mse", "--adv_norm", "remax", "--sampler", "mix", "--kl_ref", "ema"]), ("grpo", ["--reward", "random", "--adv_norm", "std"]),
                        ("ppo", ["--reward", "crps", "--adv_norm", "none", "--kl_dir", "forward", "--ema_decay", "0.9", "--eval_ema"]),
                        ("grpo", ["--reward", "mse", "--kl_dir", "k3", "--bank", "csv:data/ETTh1.csv", "--split", "chrono"]),
                        ("sft", ["--reward", "crps", "--bank", "csv:data/ETTh1.csv", "--split", "walkforward", "--folds", "2"]),
                        ("agent_sft_rl", ["--agent_reward", "crps", "--proposals", "4"]), ("agent_rl", ["--agent_reward", "newsvendor_tau", "--proposals", "4"]),
                        ("agent_random", ["--proposals", "4"]), ("fixed_context", ["--proposals", "4"])):
        out = f"{d}/{algo}_{abs(hash(tuple(extra)))}"
        bank = [] if "--bank" in extra else ["--bank", "synth"]
        r = subprocess.run([sys.executable, "scripts/run_matrix.py", "--tiny", "--algo", algo, *bank, "--episodes", "24", "--heldout", "8", "--steps", "3", "--batch", "4", "--K", "3",
                            "--layers", "0", "1", "--lora_r", "2", "--log_every", "1", "--out", out, *extra], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (algo, extra, r.stderr[-1500:])
        assert os.path.exists(f"{out}/eval.json"), (algo, extra)
        if "--save_every" in extra: assert os.path.exists(f"{out}/policy_step2.pt")
    print("algorithms ok")


if __name__ == "__main__":
    test_rewards(); test_env_policy(); test_agent_parts(); test_algorithms(); print("ALL TESTS PASSED")
