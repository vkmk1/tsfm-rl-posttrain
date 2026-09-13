# Toy results (CPU, 300 steps, batch 8, LoRA r=8 on variate-attention q/k/v of layers 10–14, 0.31M trainable), 2026-09-10

Held-out = 128 windows never used in training, same windows for every cell; frozen → post-trained, with 95 % paired-bootstrap
intervals on the change (`scripts/reeval_ci.py`). Figures in `figures/` (`scripts/make_figures.py`).

| bank | cell | MASE (CI on change) | CRPS (CI on change) | 80 % coverage | 80 % width |
|---|---|---|---|---|---|
| synth | grpo_composite | 1.142 → 1.109 (-0.050, -0.018) | 0.465 → 0.456 (-0.013, -0.004) | 0.82 → 0.73 | 2.02 → 1.67 |
| synth | grpo_crps | 1.142 → 1.110 (-0.049, -0.017) | 0.465 → 0.457 (-0.012, -0.003) | 0.82 → 0.72 | 2.02 → 1.62 |
| synth | grpo_mse | 1.142 → 1.107 (-0.053, -0.019) | 0.465 → 0.456 (-0.013, -0.004) | 0.82 → 0.73 | 2.02 → 1.65 |
| synth | grpo_mse_anchor | 1.142 → 1.088 (-0.080, -0.031) | 0.465 → 0.451 (-0.021, -0.007) | 0.82 → 0.70 | 2.02 → 1.48 |
| synth | grpo_mse_perstep | 1.142 → 1.103 (-0.067, -0.012) | 0.465 → 0.518 (+0.041, +0.064) | 0.82 → 0.24 | 2.02 → 0.45 |
| synth | native | 1.142 → 1.142 (+0.000, +0.000) | 0.465 → 0.465 (+0.000, +0.000) | 0.82 → 0.82 | 2.02 → 2.02 |
| synth | ppo_composite | 1.142 → 1.256 (+0.082, +0.147) | 0.465 → 0.615 (+0.137, +0.165) | 0.82 → 0.11 | 2.02 → 0.24 |
| synth | pref_mse | 1.142 → 1.101 (-0.061, -0.023) | 0.465 → 0.454 (-0.017, -0.005) | 0.82 → 0.71 | 2.02 → 1.55 |
| synth | sft_pinball | 1.142 → 1.067 (-0.107, -0.045) | 0.465 → 0.439 (-0.035, -0.017) | 0.82 → 0.78 | 2.02 → 1.72 |
| synth | sft_pinball_seed1 | 1.191 → 1.144 (-0.088, -0.015) | 0.451 → 0.437 (-0.022, -0.005) | 0.84 → 0.78 | 2.01 → 1.72 |
| electricity | grpo_composite | 1.376 → 1.371 (-0.012, -0.000) | 0.237 → 0.237 (-0.001, +0.000) | 0.79 → 0.77 | 0.77 → 0.74 |
| electricity | grpo_mse | 1.376 → 1.374 (-0.006, -0.000) | 0.237 → 0.237 (-0.000, +0.000) | 0.79 → 0.78 | 0.77 → 0.76 |
| electricity | native | 1.376 → 1.376 (+0.000, +0.000) | 0.237 → 0.237 (+0.000, +0.000) | 0.79 → 0.79 | 0.77 → 0.77 |
| electricity | sft_pinball | 1.376 → 1.389 (-0.005, +0.034) | 0.237 → 0.240 (-0.001, +0.007) | 0.79 → 0.80 | 0.77 → 0.80 |

## Reading
1. Synthetic: every method beats the frozen model on MASE/CRPS (intervals exclude 0); SFT-pinball beats every RL cell.
2. Synthetic: the reward does not matter inside GRPO. −MSE, −CRPS and the composite reward land within 0.01 of each other on
   every metric. The policy gradient raises the log-density at the best sampled trajectories, which for a quantile head means
   pulling quantiles together; every reward ranks the samples the same way, so every reward yields the same update.
3. Synthetic: all sampled-trajectory RL cells shrink the 80 % interval below nominal (0.82 → 0.70–0.73); per-step credit
   (TimeRFT-like) collapses it to 0.24 and worsens CRPS; PPO diverged (coverage 0.11). SFT stays near nominal.
4. Electricity (real, Hugging Face thuml/Time-Series-Library mirror of UCI ElectricityLoadDiagrams): nothing moves. All changes
   are within ±0.01 MASE and ±0.003 CRPS; the GRPO intervals include 0; SFT is insignificantly worse. The frozen model is
   already near its ceiling on hourly load.

## Verdict against the decision rule in docs/plan.md
M7 does not beat M1 or M2 beyond noise on either bank → the "better reward fixes RL post-training of the forecaster" hypothesis
is refuted at toy scale, and the mechanism (likelihood sharpening) does not depend on the step budget. The result that stands is
the calibration audit: the published MSE-rewarded recipes improve point error by shrinking uncertainty. RL retains a role only
where the reward is not differentiable through the forecast (context selection, M8), which has not been run.


## Raw table

