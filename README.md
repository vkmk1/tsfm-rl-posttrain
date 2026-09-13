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
python3 tests/test_all.py
python3 scripts/download_data.py electricity                     # optional real bank (95 MB); data/ETTh1.csv is included
bash configs/toy_matrix.sh synth                                # methods x rewards on synthetic laws, held-out metrics
bash configs/toy_matrix.sh csv:data/electricity.csv
python3 scripts/run_matrix.py --algo grpo --reward composite --bank csv:data/electricity.csv --steps 4000 --batch 16 --K 8 --out runs/gpu/grpo_composite
```
The checkpoint (1.3 GB) downloads on first use into the Hugging Face cache.

## GPU campaign
One cell per GPU, no DDP (TimesFM-3 at context 256 needs under 10 GB at batch 32 in bf16):
```bash
python3 scripts/download_data.py electricity
python3 scripts/launch_gpu.py configs/gpu_matrix.txt --gpus 0-15 --out runs/gpu \
    --extra "--bank csv:data/electricity.csv --steps 4000 --batch 32 --K 8 --episodes 4096 --heldout 512 --amp --save_every 1000"
python3 scripts/reeval_ci.py runs/gpu --bank csv:data/electricity.csv      # paired-bootstrap CIs vs frozen
python3 scripts/make_figures.py runs/gpu --out figures                      # deltas, calibration, fans, curves
python3 scripts/eval_benchmarks.py --policy runs/gpu/grpo_d1_full/policy.pt --ema --gift --fev --out results/grpo_d1_full   # official harnesses, test splits, once
```
`configs/gpu_matrix.txt` lists the cells (edit freely): baselines, the literature recipes (M2–M6), the estimator fixes of
D1/D8 one at a time and combined, the random-reward control, and the context policy (M8). The launcher skips cells that
already have an `eval.json`, so it can be re-run after a crash.

### Estimator options (D1 / D8), all in `scripts/run_matrix.py`
| flag | what it changes | why |
|---|---|---|
| `--adv_norm loo\|remax\|none` | group baseline without the std division | GRPO's std normalisation alone induces overconfidence on stochastic targets (Turtel 2505.17989, Bereket 2508.11800) |
| `--set_reward` | reward = fair CRPS of the K samples as a set, one advantage per episode | proper in the empirical distribution: the optimum is the calibrated fan, not a point mass |
| `--kl_dir forward` | KL(anchor ‖ policy) estimated with anchor samples | mass-covering; the squared quantile shift and reverse KL are mode-seeking |
| `--kl_ref ema --ema_decay d` | anchor = EMA of the policy (slow trust region) | ProRL-style moving reference; changes speed, not the fixed point |
| `--sampler ema\|mix` | K samples from the EMA policy with truncated importance weights | good trajectories the live policy made unlikely get positive advantage ("Rewarding the Unlikely" 2506.02355) |
| `--eval_ema` | evaluate and benchmark the EMA weights | Polyak averaging; WiSE-FT-style dispersion recovery |
| `--reward random` | control | if it reproduces the collapse the estimator is the cause (Spurious Rewards 2506.10947) |

## Several foundation models, one interface
`tsfm_rl/forecasters.py` wraps each model behind `quantiles / sample / logprob / sft_loss / swap`, so every trainer, reward and
evaluator runs unchanged: `--model timesfm3` (default; covariates, 9-quantile head), `chronos-2` (covariates, 21-quantile head
mapped to the 9 levels), `chronos-bolt-{tiny,mini,small,base}` (univariate quantile head) and `chronos-t5-{tiny,mini,small,base}`
(a token sampler: a genuinely stochastic policy with per-token log-probs, the LLM setting; its native SFT loss is token
cross-entropy and its KL is estimated at the samples, `--kl_dir k3`). LoRA on attention q/k/v of `--layers`; indices beyond a
model's depth fall back to its last third. `configs/gpu_matrix_models.txt` runs the same cells on all of them;
`tests/test_chronos.py` exercises the tiny checkpoints on CPU.

Real-data protocol: `--split chrono` (default for CSV banks) draws training windows from the first 70 % of the timeline and
held-out windows from the last 30 % with a gap, so no held-out window overlaps a training window. The September toy matrix
used `--split random`, which is kept for reproducibility.

## The agent around a frozen oracle (H2 / H3)
`tsfm_rl/agent.py` and `tsfm_rl/algorithms/agent.py`: a small policy chooses a context proposal (which covariate rows, history
length, transforms, estimator row; native first) and a trust weight on a grid, and delivers the linear pool of the oracle's
fan under that context with a calibrated seasonal-naive fallback (`Fallback`: residual quantiles from rolling origins inside
the history). Stage 1 (`--algo agent_sft`): hindsight labels, every action scored against the realisation. Stage 2
(`--algo agent_rl`, `agent_sft_rl`): full-information policy optimisation, the expected improvement over the native forecast
computed exactly over all actions (`--agent_reward crps | impratio | newsvendor_tau`); `agent_random` is the control and
`fixed_context` the regret reference. Rewards are proper scores of the delivered distribution, never of samples, so no action is
rewarded for shrinking a band. `configs/gpu_agent_matrix.txt` holds the arms.

## Review Radar (separate dashboard)
`python3 scripts/review_radar.py [--refresh]` pulls the public mirror of OpenReview scores (Paper Copilot paperlists, ICLR
2025–2026), ranks the corpus by similarity to the abstract in `paper/main.tex`, and writes `docs/review_radar.html` +
`.json`: nearest neighbours with decisions, per-reviewer ratings and sub-scores; the empirical P(accept | mean rating)
curves overall and in our primary area; text-only rating priors (similarity-weighted kNN, a ridge model with its CV error,
a logistic acceptance model with its CV AUC); the sub-score profile of accepted vs rejected neighbours; and the gap list
distilled from the comparables' reviews (`docs/acceptance_assessment.md`). Re-run after every abstract change; the
corpus cache lives in `~/.cache/tsfm_radar`.

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
