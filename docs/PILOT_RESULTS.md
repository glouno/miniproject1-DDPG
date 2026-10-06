# LunarLander pilot results

Local working analysis; the shared report remains on Samskrift.

Measured pilot: 12 runs, 3 training seeds per condition, 50,000 transitions per run.
All ± values below are **sample standard deviations across training seeds**, not confidence intervals.

| Algorithm | Critic LN | Final return | Signed Q1 error | Mean episode Q1 RMSE | Normalized learning-curve area |
|---|---|---:|---:|---:|---:|
| DDPG | off | -34.5 ± 53.2 | 92.4 ± 22.6 | 94.3 ± 22.5 | -35.8 ± 21.1 |
| DDPG | on | -28.4 ± 46.6 | 121.2 ± 10.8 | 124.0 ± 11.9 | -68.5 ± 11.6 |
| TD3 | off | -53.2 ± 83.8 | 18.8 ± 15.5 | 36.5 ± 8.2 | -94.0 ± 48.7 |
| TD3 | on | -2.6 ± 68.2 | 10.3 ± 17.8 | 34.9 ± 1.4 | -55.0 ± 55.1 |

Performance is undiscounted reward up to 1,000 steps. Signed error compares Q1 with the discounted
return of the same visited state/action pair under the frozen policy, using a 10,000-step diagnostic
horizon. For natural termination that is a complete return; otherwise it is a finite approximation.

## Paired normalization effects

| Algorithm | Metric | Mean on-minus-off | SD of differences | Differences for seeds 0, 1, 2 |
|---|---|---:|---:|---|
| DDPG | eval_return | 6.1 | 51.1 | 55.9, -46.1, 8.5 |
| DDPG | bias_q1 | 28.8 | 17.0 | 38.0, 39.2, 9.1 |
| DDPG | rmse_q1 | 29.7 | 16.8 | 39.0, 39.8, 10.4 |
| TD3 | eval_return | 50.6 | 83.9 | 132.2, 55.0, -35.4 |
| TD3 | bias_q1 | -8.5 | 30.5 | -41.0, -4.0, 19.5 |
| TD3 | rmse_q1 | -1.6 | 8.0 | -9.3, 6.7, -2.0 |

## Measurement audit

The original final 5,000-step diagnostic censored 23/120 episodes. Complete-only bias
therefore excludes some hovering policies. Its checkpoint curves are retained as a diagnostic
and explicitly marked as conditional, not used as the main final comparison.
The supplemental 10,000-step diagnostic still censored 23/120 episodes but retains all of them.
Across all sampled final pairs, the largest 2,500→5,000 return change was 8.06e-06;
the largest 5,000→10,000 change was 0. This checks observed horizon sensitivity,
not a mathematical bound on all possible future rewards. No critic prediction fills the missing tail.

## What we can conclude

These measurements describe early learning under one fixed configuration. A smaller signed error
can mean less optimism or more pessimism; RMSE helps distinguish calibration quality from a sign shift.
Layer normalization can affect optimization, actions and visited states as well as critic errors.
Even when lower optimism accompanies better returns, that does not establish a causal effect of bias.
Three training seeds provide little precision. Do not use 120 episodes or thousands of correlated
trajectory pairs as if they were 120 or thousands of independent training runs.

## Next experiment

Use this pilot to choose a sufficient training budget. Freeze settings before collecting fresh
training seeds (preferably 8–10) and held-out evaluation seeds. Retain the four-condition design.
Extend the return-horizon sensitivity diagnostic to all checkpoints in the final study. Consider
one min-versus-mean TD3 target ablation only after the main experiment is stable.

## Execution verification

All 12 final policies passed independent replay audits: 6,841 stored state/action/Q pairs matched and 1,080 direct discounted-return checks passed. The walkthrough notebook executed all 10 code cells without errors using the Master MIND RL kernel.
