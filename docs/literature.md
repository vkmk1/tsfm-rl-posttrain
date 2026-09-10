# Post-training of time-series models: literature (checked 9 Sept 2026)
Literature: post-training of time-series models

## A. RL post-training where the numeric forecaster is the policy (the thin core)


| Paper | Model post-trained | How the policy is stochastic | Algorithm · reward | Finding / gap for us |
|---|---|---|---|---|
| TimeRFT, Li et al., arXiv 2605.00015 (Apr 2026, rev. Jul) | Moirai-MoE (token-by-token) | sample each segment's distribution, feed back as context | reinforcement fine-tuning; quality-aware temporal reward with per-step credit; difficulty-aware data selection | beats SFT under distribution shift on TSLib-style benchmarks; Moirai-MoE specific; no covariates |
| TPO, Qi et al. 2025 (cited in GTN-R; primary not located) | one-shot TSFMs (Moirai) | point-wise predictive distribution | RLHF-style preference optimization | applies only to one-shot forecasters; MSE-type rewards |
| TS-GRPO + GTN-R, Zhang et al., arXiv 2608.08010 (Aug 2026) | Moirai-small/base, Moirai-MoE, Toto, UniTS | sample from point-wise or per-segment distributions | GRPO; ground-truth-neighborhood regularization against "suboptimal collapse" (mass drifting away from the truth) | GTN-R gives 2–4 % MSE over TS-GRPO, 3–8 % over SFT on ETT/ECL/Weather/Loop Seattle/ENTSO-e; the baseline we must reimplement on TimesFM-3 |
| Post-Training in TSFMs: A Unifying Framework, Xie et al., arXiv 2607.20002 (Jul 2026) | survey | — | taxonomy: parameter adaptation · context augmentation · model composition · output processing / uncertainty · compression | RL sits inside "parameter adaptation"; our environments cut across four of the five categories |
| Decision-focused fine-tuning of TSFMs, arXiv 2503.01936 (2025) | a TSFM on a dispatchable-feeder problem | — | differentiable decision loss through an optimizer, not RL | the decision-cost reward R4 is the RL version of this |



## B. RL where an LLM reasons about series or revises a forecaster (larger, different object)


| Paper | Policy | Algorithm · reward | Relevance |
|---|---|---|---|
| PostTime, Liu, Zhou, Sen, Prakash, Das (Google), arXiv 2605.29401 (May 2026) | Gemma-3-4B as a revisor over frozen TimesFM-2.5 (revise / preserve / ignore) | SFT + RLVR; TimesX multimodal benchmark | closest cousin: RL over a frozen TimesFM, but the policy is a text model and the reward is text-conditioned |
| Time-R1, Luo et al., arXiv 2506.10630 (2025) | LLM slow-thinking forecaster | SFT warm-up + GRIP (GRPO variant with non-uniform sampling); fine-grained multi-objective reward | reward design and two-stage recipe transfer; model class does not |
| TimeMaster, Feng et al., arXiv 2506.13705 (NeurIPS 2025) | multimodal LLM over plotted series | SFT + token-level GRPO; format + accuracy + insight reward | classification/reasoning, not forecasting |
| Cast-R1, Tao et al., arXiv 2602.13802 (Feb 2026) | tool-using agent (feature extraction, light forecasters, reflection) | SFT + multi-turn RL with curriculum | template for our tool environment E5 |
| Outcome-based RL to predict the future, arXiv 2505.17989 (2025) | LLM event forecaster | RL with Brier-score reward | proper-scoring-rule rewards (our R5) |
| STReasoner 2601.03248 · TS-Reasoner 2510.03519 · CoT for time series via RL 2510.01116 · LangTime 2503.08271 (PPO) · Hindsight preference optimization for financial advisory 2604.23988 | LLM / VLM reasoners | GRPO / PPO / DPO variants | context only |



## C. Non-RL post-training of TSFMs (what the SFT controls look like)


| Paper | Method | Relevance |
|---|---|---|
| In-context fine-tuning (TimesFM-ICF), Das et al., arXiv 2410.24087 (ICML 2025) | continued pretraining with in-context examples; matches supervised fine-tuning without gradients at test time | the Google recipe for adapting TimesFM; our context-augmentation baseline |
| LoRA for TSFMs, arXiv 2405.10216 · TRACE, arXiv 2503.16991 | parameter-efficient fine-tuning (Lag-Llama, Moirai, Chronos); gated LoRA module selection | the PEFT baseline; "little work on PEFT for TSFMs" is stated in the literature itself |
| Adapting TSFMs through data mixtures, arXiv 2603.02840 · black-box online adaptation, arXiv 2606.14222 · EIDOS latent predictive learning, arXiv 2602.14024 | data-side, online and latent post-training | context; none uses a reward |
| LLM as forecasting planner, arXiv 2607.24892 (Jul 2026) | training-free text conditioning of TSFMs | inference-only comparator |

Search protocol: arXiv via web search on 9 Sept 2026 with the queries "reinforcement learning post-training time series foundation model", "GRPO time series forecasting", "preference optimization time series forecasting", "TS-GRPO / TPO / TimeRFT", "PostTime", plus the TimesFM / Chronos / Moirai fine-tuning literature. Two gaps remain: TPO's primary source was not located, and TS-GRPO's exact reward is not stated in the abstract; both to be fixed when the PDFs are read in full.
