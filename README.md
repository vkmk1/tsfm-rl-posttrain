# tsfm-rl-posttrain

Post-training the frozen TimesFM-3 forecaster with rewards that are not MSE/MAE, benchmarked against the standard
RL recipes (GRPO, PPO) and the RL-for-time-series methods in the literature (TS-GRPO, GTN-R, TimeRFT, TPO) under
one environment, one policy parameterization and one evaluation.

## The question
Every RL post-training paper for time-series foundation models rewards `-MSE` or `-MAE` of sampled point
trajectories. That reward is scale-dependent, a point score on a model whose output is a distribution (nine
quantiles), and the loss supervised fine-tuning already minimizes. We test proper scoring rules (CRPS, pinball,
interval score), a scale-free skill score against the frozen reference, shape and calibration terms, and a
composite of them, and ask whether RL under such rewards beats SFT and MSE-rewarded RL on held-out MASE / CRPS /
coverage at equal steps.

## Layout
```
tsfm_rl/
  model.py         load TimesFM-3 (google/timesfm-3.0-pytorch) or a tiny twin; differentiable input construction; horizon readout
  adapters.py      the policy's parameters: LoRA on variate-attention q/k/v; optional head adapter (in-context lag regression, gated)
  rewards.py       mse, mae, mase, pinball, crps (quantile and sample forms), interval score, coverage/width, directional accuracy,
                   soft-DTW, skill, newsvendor, composite; per-step variants; sampling from the quantile CDF and its log-density
  data.py          episode banks: Time-Series-Library CSV windows (target + candidate covariate rows), synthetic laws with known ρ
  env.py           the environment: context actions, five retrieval strategies, input construction, forecast, reward
  evaluate.py      held-out metrics (MASE, CRPS, coverage, width, direction, interval, newsvendor, skill, R² vs the known law)
  algorithms/
    sft.py         M1  supervised (pinball / mse / mae / crps / interval)
    grpo.py        M2 TS-GRPO (mse), M3 + GTN-R-like anchor, M4 per-step credit (TimeRFT-like), M7 ours (composite)
    ppo.py         M6  PPO with a learned value baseline
    preference.py  M5  trajectory-pair IPO (TPO-like); M8 context policy over retrieval proposals (LeReT-like)
scripts/run_matrix.py   one cell: train, then evaluate on held-out episodes against the frozen reference
scripts/make_report.py  collect eval.json files into a table
configs/toy_matrix.sh   the CPU toy matrix (14 runs, ~1 s/step for the 331M model on a laptop)
tests/test_all.py       every component on the tiny model, < 2 min
third_party/timesfm3/   Google's TimesFM-3 torch code (Apache-2.0); one edit: a variance floor before a sqrt so gradients are finite
```

## Run
```bash
pip install -r requirements.txt
python tests/test_all.py
python scripts/download_data.py electricity                     # optional real bank (95 MB); data/ETTh1.csv is included
bash configs/toy_matrix.sh synth                                # methods x rewards on synthetic laws, held-out metrics
bash configs/toy_matrix.sh csv:data/electricity.csv
python scripts/run_matrix.py --algo grpo --reward composite --bank csv:data/electricity.csv --steps 4000 --batch 16 --K 8 --out runs/gpu/grpo_composite
```
The checkpoint (1.3 GB) downloads on first use into the Hugging Face cache.

## Method map
| # | method | reward / loss | literature anchor |
|---|---|---|---|
| M0 | frozen model | — | — |
| M1 | SFT (LoRA) | pinball | LoRA for TSFMs |
| M2 | GRPO on sampled trajectories | −MSE | TS-GRPO (arXiv 2608.08010) |
| M3 | M2 + anchor toward the ground truth | −MSE | GTN-R (arXiv 2608.08010) |
| M4 | GRPO with per-step credit | −MSE per step | TimeRFT (arXiv 2605.00015) |
| M5 | trajectory-pair preference (IPO) | ranked −MSE | TPO (Qi 2025) |
| M6 | PPO with value baseline | composite / mse | standard PPO |
| M7 | GRPO | composite (skill + shape + coverage) | ours |
| M8 | context policy on retrieval proposals, frozen forecaster | composite | LeReT analog |

Decision rule for the toy matrix: M7 beats M1 and M2 beyond the seed noise on held-out MASE and CRPS on a real bank
→ the reward goes to the GPU campaign; only on synthetic laws → an artifact; nowhere → reported as a negative result.
