# Long-study interpretation

All 40 runs completed, 500,000 training transitions each. Main batch: 93.98 minutes; total elapsed including restart/analysis: 97.40 minutes.

Times are wall-clock elapsed times, not measured CPU time. Runtime and per-seed details are in review.json.


| Condition | Final return ± training-seed SD | Return at 300k | On-policy Q1−MC | Mean episode RMSE | Common-probe Q1−MC | Common-probe pooled RMSE |
|---|---:|---:|---:|---:|---:|---:|
| DDPG LN off | 229.0 ± 35.8 | 123.5 | 46.5 | 55.1 | 54.2 | 61.7 |
| DDPG LN on | -40.9 ± 107.4 | -31.1 | 187.6 | 193.3 | 156.2 | 162.2 |
| TD3 LN off | 259.7 ± 50.5 | 220.1 | -11.8 | 19.8 | -42.2 | 56.0 |
| TD3 LN on | 278.9 ± 11.7 | 229.7 | -6.7 | 15.1 | -1.1 | 52.2 |

DDPG normalization lowered final performance and raised optimism/calibration error in every paired seed. The same direction persists on common probes, so changed initial state/action visitation alone does not explain it. Continuation policies still differ.

TD3 normalization has the best mean final return and lower between-seed spread, but the paired performance interval includes zero. Seven seeds improve; one large rescued-seed gain drives most of the mean difference. The other nine seeds average only 3.5 points improvement.

TD3 critics are slightly pessimistic on policy. Normalization raises their predictions toward the observed returns: this reduces underestimation, rather than reducing positive overestimation.

Near-zero common-probe mean signed error for TD3 LN does not mean all predictions are accurate: positive and negative errors can cancel, while pooled RMSE remains about 52. These probes also include deliberately off-policy actions.

The 500k budget matters: DDPG without LN gains about 105 return points from 300k; both TD3 settings gain about 40–49. Extending this unchanged DDPG+LN configuration is not supported by its flat/poor mean curve.

The study supports an algorithm-dependent effect of critic LN under this fixed configuration. It does not identify a causal effect of bias on performance, nor a universal effect of normalization. A separately frozen target-bias ablation is the next step for that causal question.


Ten training seeds determine uncertainty. Bootstrap intervals are descriptive, pointwise and not adjusted for multiple comparisons. Evaluation episodes and trajectory pairs are not independent training replicates.

All trajectories were retained. Final censor counts and maximum observed horizon sensitivity are in review.json; small empirical sensitivity does not prove a universal bound on unobserved future rewards. Raw returns use up to 10k diagnostic steps.


This local working analysis is not a replacement for the live shared Samskrift report.


Reproduce this review with `python miniproject/review_study.py --input miniproject/outputs/study-20261005-500k`. The raw 1.2 GiB study directory is ignored by Git and is needed to regenerate the figures. All 40 final models passed a reload audit: 25,114 saved Q pairs matched and 720 independent discounted-return checks passed.
