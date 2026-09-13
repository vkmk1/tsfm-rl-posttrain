# Evaluation protocol: how a post-trained TimesFM-3 is compared to everything else

## The analog of AutoResearchExam for us
AutoResearchExam scores an agent on a hidden test set after it optimized a validation metric, time-weighted, with
anti-overfitting controls. The time-series analog is not one benchmark but a protocol built from three public
benchmarks that already separate "what you may train on" from "what you are scored on":

| Benchmark | Tasks | Point / probabilistic metric | Aggregation | Covariates | Role for us |
|---|---|---|---|---|---|
| GIFT-Eval (Salesforce, arXiv 2410.10393) | 97 configs (23 datasets × short/medium/long) | MASE / CRPS (WQL) | geometric mean of the ratio to seasonal naive; mean rank by CRPS and by MASE | multivariate configs only | primary leaderboard; train splits are the training/backtest windows, test splits are scored once |
| fev-bench (Amazon, arXiv 2509.26468) | 100 tasks, 46 with covariates | MASE / SQL (scaled quantile loss) | skill score (1 − clipped geometric-mean relative error) and win rates, both with bootstrapped confidence intervals | yes (past-only and known-future) | the covariate result; the CI machinery we report everywhere |
| TIME (arXiv 2602.12147) | 98 tasks, 50 datasets, 8 domains | MASE / CRPS | rolling evaluation | limited | generalization check, secondary |

Hidden-test discipline: the model is post-trained only on GIFT-Eval train splits, fev-bench training windows and synthetic laws; the official test splits are evaluated once, through the official harnesses (`gift_eval` package, `fev` library), with the leaderboard's own aggregation code. The leakage statement follows GIFT-Eval's rules (TimesFM-3's pretraining corpus is already leakage-free against GIFT-Eval; our post-training adds only train-split windows). On the leaderboards the entry is tagged "fine-tuned", not "pretrained".

## What "better" means, in order of weight for a reviewer
1. Same-model deltas with intervals. Frozen TimesFM-3 vs post-trained TimesFM-3 (ours) vs the controls (SFT-pinball, TS-GRPO, GTN-R, TimeRFT-style, TPO-style, PPO) on identical test splits, identical trainable parameters, identical steps and data. Report the skill score against the frozen model with bootstrapped 95 % intervals over tasks, and the win rate. A method whose interval excludes zero on GIFT-Eval CRPS and fev-bench SQL wins; one whose interval includes zero does not, whatever the mean says.
2. The reward decides, not the algorithm. The matrix crosses algorithms with rewards, so the paper can state whether the gain comes from the CRPS-type reward (M7 vs M2 at equal algorithm) or from the algorithm (M7 vs M6 at equal reward).
3. Metrics the training loss cannot express: 80 % coverage, interval width, directional accuracy, newsvendor cost. These separate "RL found a better point forecast" from "RL improved the distribution".
4. Leaderboard placement. Where the post-trained model lands among Chronos-2, TimesFM-3, Toto 2.0, TiRex-2 on GIFT-Eval and fev-bench. This is context, not the claim: a 2–3 % CRPS gain moves one or two places.
5. Sample efficiency (the time-weighted analog). Held-out CRPS versus post-training steps (or GPU-hours) for every method; report the area under the improvement curve as AutoResearchExam reports AUARC. Cheap: every run logs it.
6. Generality. The same protocol on a second frozen model (Chronos-2). Without it the result is about one checkpoint.
7. Mechanism. The audit suite (delay, texture shift, suffix dependence) before and after, so the paper says what changed, not only how much.

## What the existing RL-for-TSFM papers did, and why we do not copy it
TimeRFT, TS-GRPO and GTN-R report MSE/MAE on ETT, Electricity, Weather and similar Time-Series-Library datasets with
their own splits. That protocol has no probabilistic metric, no covariates, no confidence intervals, no leaderboard
comparability, and the datasets overlap with common pretraining corpora. Reporting on it as a secondary table lets
reviewers compare directly; it cannot carry the paper.

## Concretely, per run
`scripts/run_matrix.py` writes `eval.json` with the internal held-out metrics (the toy protocol). For the paper:
`tsfm_rl/benchmarks.py` loads GIFT-Eval train splits and fev-bench training windows as episode banks, and evaluates
a policy through the official harnesses on the test splits (`evaluate_gift_eval`, `evaluate_fev`). Results tables:
skill score and win rate with bootstrap CIs (fev's `fev.leaderboard`), GIFT-Eval's geometric-mean MASE / CRPS and
ranks (the harness' `all_results.csv`), coverage and width from our evaluator, and the sample-efficiency curve.


## Amendments (12 Sept 2026)
- Walk-forward is the primary real-data protocol: `--split walkforward --folds 5 --first_origin 0.5` gives rolling forecast
  origins; at each origin the policy is trained only on windows ending before it (a T+H gap) and scored on the block that
  follows; `eval.json` carries per-fold and pooled metrics; `scripts/reeval_ci.py` pairs frozen and post-trained per window and
  bootstraps over all held-out windows across folds. A single chronological split (`--split chrono`) is one fold of this.
  Holding out a fraction of the series (`--ood_frac`) is a secondary robustness axis, reported separately, never the main split.
- No fixed tolerances. "Calibration intact" means: the paired-bootstrap interval on the coverage change includes zero, and the
  binomial test of 80 % coverage against nominal and the KS test of PIT uniformity (`tsfm_rl.evaluate.calibration_tests`,
  reported as `cal_*` in every eval.json) are not rejected where the frozen model's are not.
