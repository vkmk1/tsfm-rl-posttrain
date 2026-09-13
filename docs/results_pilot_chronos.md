# Pilot: the reward question on two Chronos families (CPU, 12 Sept 2026)

ETTh1, chronological split (train windows in the first 70 % of the timeline, held-out windows in the last 30 %), 128 held-out
windows, 150 steps, batch 8, K = 4, LoRA r = 4 on attention q/k/v of blocks 2–3, one seed. Frozen → post-trained.
`runs/pilot_chronos-t5-tiny`, `runs/pilot_chronos-bolt-tiny`. Token-sampler metrics carry ~0.02–0.03 MASE of sampling noise
(20 sample paths per window; the evaluation RNG is pinned from the set-CRPS cell onward).

| model | cell | MASE | CRPS | 80 % coverage | 80 % width |
|---|---|---|---|---|---|
| Chronos-T5-tiny (token sampler) | SFT, token cross-entropy | 2.609 → 2.549 | 0.692 → 0.683 | 0.64 → 0.66 | 1.80 → 1.95 |
| | GRPO, −MSE, std-normalised (TS-GRPO recipe) | 2.585 → 2.568 | 0.689 → **0.699** | 0.63 → **0.55** | 1.76 → 1.38 |
| | GRPO, −MSE, leave-one-out baseline | 2.610 → **2.540** | 0.698 → **0.679** | 0.63 → 0.61 | 1.75 → 1.63 |
| | GRPO, set-level fair CRPS, LOO, forward KL | 2.610 → 2.585 | 0.698 → 0.686 | 0.63 → **0.63** | 1.75 → 1.76 |
| Chronos-Bolt-tiny (quantile head) | SFT, pinball | 2.250 → 2.238 | 0.601 → 0.597 | 0.75 → 0.75 | 1.94 → 1.93 |
| | GRPO, −MSE, std-normalised | 2.250 → 2.237 | 0.601 → 0.609 | 0.75 → **0.65** | 1.94 → 1.50 |
| | GRPO, −MSE, leave-one-out | 2.250 → 2.237 | 0.601 → 0.603 | 0.75 → 0.70 | 1.94 → 1.71 |
| | GRPO, set-level fair CRPS, LOO, forward KL | 2.250 → 2.239 | 0.601 → 0.611 | 0.75 → **0.64** | 1.94 → 1.50 |

## Reading
1. The literature recipe (MSE reward, std-normalised advantages) damages calibration on both families: coverage −9 and −10 points,
   CRPS worse, for a point-error gain within noise. Third model family showing this (TimesFM-3, Chronos-T5, Chronos-Bolt).
2. Removing the std division alone (RLOO) halves the damage on both models and gives the best point metrics on the token
   sampler (Turtel 2505.17989, Bereket 2508.11800 confirmed for forecasters).
3. The set-level proper score preserves calibration **only on the genuinely stochastic policy** (the token sampler: coverage
   0.63 → 0.63, CRPS improved). On the quantile head it collapses exactly like the MSE reward (0.75 → 0.64). The quantile
   head's "sampling" is a device (inverse-CDF draws clipped to [0.05, 0.95] with a piecewise-linear density); the
   score-function identity E[∇ log p] = 0 does not hold for it, so the estimator sharpens regardless of the reward.
   Consequence for the paper: RL on the predictive distribution is well-posed for token/flow samplers, not for quantile heads;
   for quantile heads the differentiable proper loss (SFT-CRPS) is the correct object, and it wins here too.
4. SFT matches or beats every RL cell on the quantile head and is within noise of the best RL cell on the token sampler.
   The direction stands: the forecaster stays frozen; RL goes to the choices around it.
Caveats: one seed, 150 steps, tiny checkpoints, 128 windows; paired-bootstrap CIs via `scripts/reeval_ci.py` pending.
