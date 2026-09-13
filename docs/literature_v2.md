# Literature v2 (10 Sept 2026): LLM post-training → TSFM transfer, gap map, directions

Full curated tables live in the dashboard tab "Brainstorm: LLM RL → TSFM" (artifact a2762577; source
~/tsfm-research/dashboard/brainstorm_tab.html). This file is the index.

## The theorem behind the toy result
Policy gradient maximises J(θ) = E_{τ~p_θ}[r(τ)], linear in p_θ ⇒ optimum is a point mass. CRPS(p,y) = E|Y−y| − ½E|Y−Y'|;
a per-sample reward drops the spread term, so "CRPS as reward" = MAE at the median (our M2 ≈ M7 ≈ GRPO-CRPS).
KL-anchored optimum p ∝ p_ref·exp(r/β) is a tempered p_ref (Korbak 2205.11275). ΔH ∝ −Cov(log p_θ, A) (Cui 2505.22617).

## Direct precedents for our collapse
- Turtel et al. 2505.17989: GRPO std-normalised advantage ⇒ extreme overconfidence on Brier-rewarded forecasts; ReMax fixed it.
- Bereket et al. 2508.11800: GRPO induces overconfidence on stochastic outcomes; PPO/RLOO stay calibrated.
- GTN-R 2608.08010: "suboptimal collapse" in Moirai/Toto under TS-GRPO/TPO/TimeRFT; regulariser; no calibration reported.

## Gap map (published P / empty —)
| action \ reward | point error | proper score | calibration | utility |
|---|---|---|---|---|
| distribution, RL | P (TimeHF 2501.15942, TimeRFT 2605.00015, GTN-R 2608.08010; TSLib MSE only) | — | — | — |
| distribution, SFT | P many | P Toto-2 2605.20119, Laglil 2607.23146 | — | P DFL-Moirai 2503.01936 |
| context / exemplars | P TATO 2603.00629 (search), TS-RAG 2503.07649 | — | — | — |
| decision from forecast | — | — | — | — (LLM only) |
| online / test-time | P ELF 2502.12920, ORCA 2606.14222 | — | — | — |
| flow sampler (Sundial 2502.00816) | — | — | — | — |
No RL-post-trained TSFM on GIFT-Eval or fev-bench. Survey 2607.20002 cites zero RL works.

## Directions (ranked; details on the dashboard)
D1 fix the estimator (LOO/ReMax, set-level fair CRPS, forward-KL anchor, random-reward control) — 4 CPU cells.
D2 context/retrieval policy over frozen forecaster (LeReT 2410.23214, Search-R1 2503.09516, ToolRL 2504.13958) — cell M8.
D3 utility-proper decision environments (randomised fractile ⇒ integrated regret ∝ CRPS; HDPO 2306.11246; 2606.16790).
D4 test-time RL with rolling-origin verifier (2411.07279, TTRL 2504.16084).
D5 stochastic-sampler RL on Sundial (Flow-GRPO 2505.05470, Adjoint Matching 2409.08861).
D6 ReST-EM with diversity (2312.06585, 2308.01825) as the cheap baseline.
D7 SAE-feature steering as action space (2603.10071).
Anti-patterns: entropy/self-certainty rewards (2505.15134, 2505.22660, 2505.19590); per-sample proper scores as reward;
DPO on densities without pinball anchor (2404.19733); learned forecast reward models (2210.10760); returns as target
(2511.18578, 2606.27100, 2607.12248).

## Finance targets with documented signal
Realised volatility (HAR is the bar: Brini 2607.05291, Goel 2505.11163); intraday volume/VWAP (Cucuringu 2505.08180,
IVE 2411.10956; no TSFM paper); short-horizon OFI (Cont 2014, 2112.13213). CRPS-best ≠ profit-best (Nitka 2308.15443, M6).
