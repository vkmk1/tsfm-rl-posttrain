# ICLR acceptance assessment (13 Sept 2026)

Sources: Paper Copilot mirror of OpenReview scores (iclr2025.json, iclr2026.json); HF review datasets (3Liz22/iclr2026_real_reviews,
smallari/openreview-iclr2025-peer-reviews-RAW, Vidushee/iclr-rejected-papers-with-code-1k); ICLR 2026 retrospective blog.
OpenReview itself was bot-blocked; meta-reviews unverified.

## Comparable papers (decision, ratings, what reviewers said)
| paper | decision | ratings | praised | criticised |
|---|---|---|---|---|
| LeReT (ICLR 2025) | Poster | 6;6;6;6 | RL beats SFT for query generation; compatible with retrievers | "incremental within an existing framework"; only basic baselines; datasets dated |
| Beyond Accuracy: TSFM calibration (ICLR 2026) | Poster | 4;4;6;8 | first systematic study; metrics; comprehensive | "no technical contribution"; no *why*; wants GIFT-Eval scale |
| TATO (ICLR 2026) | Poster | 2;4;6;6 | data-adaptation paradigm; fast; 6 LTMs × 8 datasets | no fine-tuned-LTM baseline; no TTA baselines; "no theory"; writing |
| RL Squeezes, SFT Expands (ICLR 2026) | Poster | 4;4;4;6 | timely, solid analysis | "scope and novelty not sufficient"; narrow domain |
| Model Portfolios / Chroma (ICLR 2026) | Poster | 2;4;6;8 | comprehensive, reproducible | test-time validation labels; no OOD; one architecture; no CIs |
| Time-R1 (ICLR 2026, RL LLM forecaster) | **Reject** | 2;4;4;4 | principled two-stage RFT; ablations | "marginal over GRPO"; rewards ≈ MSE loss, "merely rewards numerical accuracy"; ad-hoc reward; inference cost |
| ZooCast (13 TSFMs on GIFT-Eval) | **Reject** | 2;4;6;6 | fair GIFT-Eval comparison | ensemble loses to best single model; "not theoretically grounded" |
| Accepted audit/negative-result papers 2025–26 | Posters | 4–10 | mechanism/derivation + broad model coverage | "position paper", "no formal statistical testing", over-reach |

No ICLR 2025/2026 submission doing GRPO/PPO post-training of a numeric TSFM was found: the audit target has not been refereed.

## Thresholds (ICLR 2026, decided papers)
27.4 % accepted of valid submissions; P(accept | avg rating): 4.0 → 13 %, 4.5 → 30 %, 5.0 → 57 %, 5.5 → 82 %, 6.0 → 93 %.
Time-series area: 30 % acceptance; 4.5 → 23 %, 5.0 → 51 %, 5.5 → 85 %.

## Five objections and what neutralises each
1. Trivial theorem, straw-man targets → reproduce each recipe from its code, replicate its point gains, then show the hidden cost with CIs on ≥3 oracles × ≥20 GIFT-Eval tasks; the sampler-vs-quantile-head distinction is the non-obvious part.
2. Limited novelty (context selection + pooling + Hedge are known) → ablations; head-to-head vs TATO, COSA/TAFAS, best fixed context, conformalised oracle; cross-oracle transfer as a headline.
3. Scale (CPU pilots, no leaderboards) → full GIFT-Eval and fev-bench, four oracle families incl. a current SOTA, compute table.
4. Test-time labels / leakage → feedback lag = horizon; results vs lag; fallback under zero feedback; pre-registered windows.
5. "Why RL at all?" → show policy-gradient beats Hedge on the non-differentiable decision, or retitle honestly and let the audit carry the RL claim; beat the conformalised oracle on CRPS and utility.

## Probability
(i) current evidence: 10–20 %. (ii) significant transferable agent gain on fev-bench covariate tasks with the baselines above: 35–50 %, up to ~55 % with clean transfer and SOTA oracles.
