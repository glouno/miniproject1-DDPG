# Bias sensitivity to 1,000 versus 10,000 diagnostic steps

Checked on 9 October 2026. For the final Mac Stage A DDPG policy
`ddpg-a0-c1-seed21` (actor LayerNorm off, critic LayerNorm on, training seed 21,
1,000,000 training transitions), the longer diagnostic increases the estimated
on-policy overestimation bias by **1.17 native reward units, or 0.42%**.
This is statistically detectable within this policy's evaluation episodes,
but small compared with its roughly 282-unit overestimation. It does not change
the conclusion that this critic strongly overestimates the policy's returns.

| Episode cutoff | Equal-episode mean Q1 minus discounted return |
| --- | ---: |
| 1,000 steps | 281.093043 |
| 10,000 steps | 282.264386 |
| Change, 10,000 minus 1,000 | +1.171343 |

The paired 95% percentile-bootstrap interval for the change is **[+0.756373,
+1.588250]**, using 10,000 episode resamples and RNG seed 419. A two-sided paired
t-test gives **p = 0.0000413** (t-based 95% interval [+0.708349, +1.634337]).
As a directional robustness check, an exact two-sided sign test on the 13
nonzero episode changes gives **p = 0.000244**; this tests sign balance rather
than the mean change. The seven episodes terminating before 1,000 steps have
exactly zero change. All 13 remaining episodes have positive changes.

The policy was selected from available final checkpoints as an informative
case with trajectories extending beyond the performance cutoff, rather than
requiring new training. It is a poor-performing policy (mean 1,000-step
undiscounted return −190.52) with persistent flight on many evaluation seeds.
The result is an exploratory sensitivity check for this one policy, not an
algorithm-wide finding across independent training seeds.

The comparison reuses the **same 20 saved evaluation reset seeds 40000–40019**,
the same 1,433 sampled state/action pairs, and the same saved Q1 predictions.
For each episode and sampled timestep t, calculate

`G_H(t) = sum_{k=t}^{min(H, episode_length)-1} gamma^(k-t) r_k`, with gamma = 0.99.

The bias is `Q1(s_t, a_t) - G_H(t)`. Average first over sampled pairs within each
episode, then give all episodes equal weight, matching the study protocol.
For the paired difference, Q cancels exactly: `delta = G_1000 - G_10000`.
Only the return cutoff changes; no additional rollout randomness is introduced.
Episode reset seeds are the uncertainty units conditional on the frozen policy;
correlated timesteps within an episode are not independent replicates.

Thirteen of the 20 episodes reach the 10,000-step cap, so their longer returns
remain finite approximations. No critic value is inserted as a tail estimate.
All recalculated 10,000-step returns and episode biases match the stored data,
and all 1,433 Q predictions match scalar inference from the final checkpoint
exactly. Independent direct discounted sums also verify early and late sampled
returns in every episode. Input SHA-256 hashes are saved in the generated summary.

Cutting an episode at step 1,000 leaves only ten future rewards for a state
sampled at step 990. Consequently, the maximum individual pair's bias change
is **26.84 units**, even though the overall mean changes little. This explains
why gamma^1000 being small does not make the episode cutoff harmless for late
sampled states. As a separate check, using 1,000 future rewards **after each
sampled state**, instead of stopping the whole episode at step 1,000, changes
the mean bias by only +0.000521 units relative to the saved 10,000-step episode
cutoff. That is a different return definition from the project's current
episode-level diagnostic horizon.

Reproduce from the repository root (choose a fresh output path):

```bash
uv run --locked python -m lunarlander_rl.horizon_sensitivity \
  --run outputs/mac-stage-a-failure-input/mac-stage-a-20261008/A/ddpg-a0-c1-seed21 \
  --output generated/horizon-sensitivity-new
uv run --locked python -m unittest discover -s tests -p 'test_horizon_sensitivity.py' -v
```

The original run and final checkpoint are preserved. Generated results are in
`generated/horizon-sensitivity-20261009-ddpg-seed21-verified/`: `summary.json`,
`episodes.csv`, and `horizon_sensitivity.png`/`.pdf`. Raw runs and generated
artifacts are ignored by Git and must be retained separately for reproducibility.

![Episode and state-level cutoff sensitivity](../generated/horizon-sensitivity-20261009-ddpg-seed21-verified/horizon_sensitivity.png)

## Check of several other variants

Four additional final Mac Stage A policies were checked, using training seed 21
and the 1,000,000-transition checkpoint in each case. The variants and common
training seed were chosen before computing their horizon differences. Each
comparison pairs the same 20 held-out episodes under the same frozen policy.

| Variant | Bias, 1,000 | Bias, 10,000 | Change | Paired 95% bootstrap interval | Episodes past 1,000 |
| --- | ---: | ---: | ---: | --- | ---: |
| DDPG, no LN | 26.874 | 26.874 | 0.000 | [0.000, 0.000] | 0/20 |
| DDPG, critic LN (original policy) | 281.093 | 282.264 | +1.171 | [+0.756, +1.588] | 13/20 |
| DDPG, actor + critic LN | 121.350 | 121.657 | +0.307 | [−1.203, +1.989] | 12/20 |
| TD3, no LN | −18.629 | −18.629 | 0.000 | [0.000, 0.000] | 0/20 |
| TD3, critic LN | −0.694 | −1.019 | −0.325 | [−0.975, 0.000] | 1/20 |

Only the original DDPG critic-LN policy has a statistically significant mean
change in this selected set. Its paired t-test remains significant after Holm
correction across the five policies (adjusted p = 0.000206). The other four
adjusted p-values are 1.0. The descriptive bootstrap intervals above are not
adjusted for multiplicity. The baseline DDPG and TD3 policies have exactly equal
estimates because every evaluated episode terminates before step 1,000.

These policies confirm **small average changes**, but do not confirm that a
significant increase occurs in every variant. Changes can have either sign.
The actor+critic-LN DDPG policy has an individual pair difference up to 111.59
units, and TD3 with critic LN up to 62.17 units. Late-state discrepancies can
therefore be substantial even when episode and overall averaging cancels them.
This supports retaining the longer diagnostic for state-level calibration.
Only the original policy has censored trajectories at 10,000 in this selected
set; all other selected trajectories terminate naturally by that horizon.

The same horizon change also applies to TD3 Q2 and min(Q1, Q2) signed biases:
only the Monte Carlo return changes, so each critic's fixed Q prediction cancels
from the paired difference. Their short/long estimates are saved in the CSV.
All selected Q1 predictions match their final checkpoint exactly, and all
recalculated long-horizon episode biases match their original journals.

This is a check of five specific trained policies, conditional on those policies,
not an inference across all variants or independent training seeds. Before
the scope was narrowed to several policies, a preliminary check of ten seeds
of the original DDPG critic-LN variant gave a mean change of +0.558 units,
with training-seed 95% bootstrap interval [−0.018, +1.065]; this reinforces
that the significant result for seed 21 should not be generalized.

Generated comparisons, episode CSVs, input hashes and per-policy figures are in
`generated/horizon-sensitivity-selected-20261009/`. The final diagnostic inputs
were extracted from SHA-256-verified Mac release archives into
`outputs/horizon-sensitivity-input-20261009/`; existing research runs were
not changed. To reproduce the selected analysis, use a fresh output path:

```bash
uv run --locked python scripts/check_selected_bias_horizons.py \
  --run outputs/horizon-sensitivity-input-20261009/mac-stage-a-20261008/A/ddpg-a0-c0-seed21 \
  --run outputs/horizon-sensitivity-input-20261009/mac-stage-a-20261008/A/ddpg-a0-c1-seed21 \
  --run outputs/horizon-sensitivity-input-20261009/mac-stage-a-20261008/A/ddpg-a1-c1-seed21 \
  --run outputs/horizon-sensitivity-input-20261009/mac-stage-a-20261008/A/td3-a0-c0-seed21 \
  --run outputs/horizon-sensitivity-input-20261009/mac-stage-a-20261008/A/td3-a0-c1-seed21 \
  --output generated/horizon-sensitivity-selected-new
```
