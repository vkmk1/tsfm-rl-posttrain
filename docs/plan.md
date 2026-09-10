# RL post-training of TimesFM-3: plan v2 (9 Sept 2026)

Supersedes v1 (the five-reward / four-policy sketch). Decisions so far: no pretraining; the frozen TimesFM-3 is the environment's simulator and the agent (LoRA + grafts) at once; context construction is the environment's action space; the SAE toolkit is the instrument that says what changed.

## 1. The idea under test: a reward that is not MSE/MAE
Every RL-for-TSFM paper so far (TimeRFT, TPO, TS-GRPO/GTN-R) rewards −MSE or −MAE of sampled point trajectories against the ground truth. That reward has three problems for RL specifically: it is scale-dependent (advantages are dominated by high-variance series unless normalized within a group), it is a point score on a model whose output is a distribution (the nine quantiles), and it is exactly the loss supervised fine-tuning already minimizes, so RL can only recover SFT more slowly. The candidate reward is relative, distributional, and partly non-differentiable, which is where RL has a reason to exist:

    r = r_skill + λ_shape · r_shape + λ_cov · r_cov [+ λ_dec · r_decision]

- r_skill = 1 − CRPS(forecast distribution, y) / CRPS(reference, y): the fev-bench skill score against a reference forecaster (frozen native TimesFM-3, or seasonal naive). Scale-free, distributional, comparable across series. CRPS from the nine quantiles (or from K samples of the quantile CDF).
- r_shape: 1 − normalized soft-DTW (or DILATE-style shape + temporal distortion) between the median path and y; directional accuracy of first differences as the cheap variant. Non-differentiable in its hard forms.
- r_cov: −|coverage of the 10–90 % interval − 0.8| − width penalty (an interval score). Non-differentiable.
- r_decision: newsvendor / imbalance cost, the decision ablation.
Per-step credit (TimeRFT's idea): rewards can be given per horizon step and aggregated with a horizon weight; GRPO then uses per-step advantages.

Hypothesis: under this reward, RL improves MASE/CRPS/coverage on held-out windows beyond SFT-pinball and beyond GRPO-with-MSE at equal steps, because it optimizes what the benchmarks measure and what SFT cannot express. Falsified if GRPO-with-MSE or SFT matches it.

## 2. Benchmark matrix
Methods (all on the same frozen TimesFM-3, same trainable parameters, same data, same steps):
| # | Method | Reward / loss | Literature anchor |
|---|---|---|---|
| M0 | inference-only: native · history-locked routing · estimator row | — | the draft; our audit |
| M1 | SFT: LoRA + grafts on pinball | pinball | LoRA-for-TSFMs, TimesFM-FT |
| M2 | GRPO, MSE reward on sampled trajectories, no anchor | −MSE | TS-GRPO (2608.08010) |
| M3 | M2 + ground-truth-neighborhood anchor | −MSE + anchor | GTN-R (2608.08010) |
| M4 | GRPO with per-step temporal credit | −MSE per step | TimeRFT (2605.00015), as far as the abstract specifies |
| M5 | preference optimization between sampled trajectories (IPO/DPO on reward-ranked pairs) | ranked −MSE | TPO (Qi 2025) |
| M6 | PPO with a learned value baseline (clipped surrogate, one sample per step) | our reward | standard PPO |
| M7 | GRPO with our reward | r_skill + shape + cov | ours |
| M8 | M7 + context policy (joint) | our reward | ours + LeReT analog |
Rewards are crossed with methods where meaningful (M2/M6/M7 each run with MSE, pinball and our reward), so the table answers "is it the algorithm or the reward".

## 3. Metrics (held-out windows, never the training windows)
MASE and WQL/CRPS as GIFT-Eval computes them; 80 % coverage and interval width; soft-DTW; directional accuracy; on synthetic laws additionally response R² and the audit suite (delay, shift, suffix). Two-level: what the reward optimizes vs what the leaderboard measures.

## 4. CPU toy protocol (before any GPU run)
The 331 M model runs at ~1 s per training step at batch 8 on this laptop, so a 300-step run is ~5 minutes.
- Environments: E1 synthetic laws (ρ known); E2 electricity windows (321 series, real cross-sectional bank; to download, 95 MB); E3 ETTh1 as the low-headroom control.
- Grid: methods M1, M2, M3, M6, M7 × rewards {MSE, pinball, ours} where applicable, 300 steps, batch 8, K = 4, one seed; evaluation on 128 held-out windows per environment. About 2 hours of CPU per environment.
- Decision rule: if M7 beats M1 and M2 by more than the seed noise (measured by one repeat of M1) on E2's held-out MASE/CRPS, the reward idea goes to the GPU campaign; if only on E1, the reward is a synthetic artifact; if nowhere, the paper's contribution is the environment + benchmark + audit, and the reward becomes a negative result.

## 5. GPU campaign (after the toy)
Same matrix at 4,000 steps, batch 256, K = 8, on GIFT-Eval and fev-bench episode banks, two seeds; TimesFM-3 and Chronos-2; SAE audit before and after; the audit suite; iterative resampling once.

## 6. What is implemented (tfm3/)
env.py (episode banks, five retrieval strategies, context actions, rewards pinball/MASE/newsvendor, quantile-CDF sampling, differentiable log-density), rl_train.py (GRPO with KL anchor, context IPO/best-of-K, joint), adapter.py (LoRA, value-level estimator adapter, head adapter, sparse bottleneck), posttrain.py (SFT), audit_suite.py, sae_audit.py, hooks.py (attention capture/replay), condition_policy.py. To add for the matrix: CRPS/shape/coverage rewards and the composite; PPO with a value head; per-step credit; trajectory-pair preference (TPO analog); the held-out evaluator; the electricity bank.
