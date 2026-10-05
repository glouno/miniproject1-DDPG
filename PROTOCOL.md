# LunarLander normalization and overestimation study

This protocol was written before launching the 12-run pilot. The pilot is for
method validation and effect-size exploration; it is not the final report.
The live team report remains [the shared Samskrift project](https://samskrift.com/project/7ce99927-c689-46cf-b78a-6ea0d8ecc0d8).

## Questions and controlled comparison

1. Does critic layer normalization change signed value-estimation error?
2. Does that change accompany a change in learning performance?
3. Is the relationship similar for DDPG and TD3?

Use four conditions: DDPG/TD3, each with critic normalization off/on. Normalize
after each hidden critic linear layer, before ReLU, with PyTorch LayerNorm
`eps=1e-5`, learned affine parameters. Do not normalize the actor or output
layer. This isolates critic normalization; it does not test every possible
placement of normalization. Initial linear weights match within paired seeds.

Both algorithms use a target actor and target critic(s), unlike the simplified
DDPG exercise. TD3 additionally uses two independently initialized critics,
a minimum target, target action smoothing, and delayed actor/target updates.
Thus an algorithm comparison changes multiple mechanisms. A later minimum-target
ablation would be needed to better isolate a mechanism affecting bias.

## Pilot configuration

Three training seeds (0, 1, 2) x four conditions = 12 runs, each with 50,000
real transitions. Each run has one training environment and one CPU thread.
`configs/pilot.json` holds the common parameters; complete resolved configs,
library versions, source snapshots/hashes and the dependency lock hash are
stored with every run. The pilot manifest fixes all conditions before training.

| Setting | Pilot value |
|---|---|
| Environment | `LunarLander-v3`, `continuous=True`, gravity -10, wind off |
| Actor/critic hidden layers | 128, 128; ReLU; actor output tanh scaled to bounds |
| Discount / Polyak coefficient | 0.99 / 0.005 |
| Actor/critic optimizer | Adam, learning rate 0.0003; betas (0.9, 0.999), eps 1e-8 |
| Batch / replay capacity | 128 / 100,000 |
| Warmup | 5,000 uniformly random actions; learning starts at transition 5,000 |
| Update-to-data ratio | One critic-update round per new transition after warmup |
| Critic-to-actor ratio | DDPG 1; TD3 2 (both critics updated in each TD3 round) |
| Exploration noise | Gaussian sigma 0.1 in normalized action units; clip actions |
| TD3 smoothing noise / clip | 0.2 / 0.5 in normalized units |
| Performance horizon | 1,000 transitions or natural termination |
| Evaluation | At 0 and every 10,000 transitions; ten fixed seeds 10,000–10,009 |
| Bias diagnostic horizon | Continue beyond 1,000 if needed, up to 5,000 |
| Diagnostic sampling | Every tenth visited pair within the first 1,000 transitions |

No reward/observation normalization, reward clipping, gradient clipping,
prioritized replay, or hyperparameter search is used in this initial comparison.
The CPU runs seed Python, NumPy, PyTorch and the environment. Small floating-point
differences across hardware/library versions remain possible.

## Q versus Monte Carlo

At each checkpoint freeze the current actor and contemporaneous critics.
Collect separate deterministic-policy trajectories from fixed evaluation seeds.
For each sampled visited pair `(s_t, a_t)`, compute the critic prediction and
`G_t = sum_k gamma**k * r_(t+k+1)` along that same trajectory, following the
frozen actor thereafter. Never pair a best actor with critics from a later time.
Never try to reconstruct Box2D state from the observation vector.

Training bootstraps through time-limit truncation but not natural termination.
To prevent an omitted time-limit tail from being labeled overestimation,
diagnostic episodes continue past the standard 1,000-step limit. Bias is only
computed for episodes that naturally terminate within the 5,000-step diagnostic
limit. Censored episodes remain in performance evaluation and their number is
reported. If censoring occurs, bias is conditional on completion and may suffer
selection bias; those results must not be presented as unbiased calibration
over all evaluation episodes. Zero censored diagnostics removes this particular
missing-tail issue, not other sampling limitations.

Primary signed bias is `Q1 - G`, averaged over sampled pairs within each
episode, then over valid episodes within each run. This compares the critic
optimized by the actor in both algorithms. TD3's second critic and minimum
are secondary diagnostics. Report RMSE, initial-state bias and the fraction
of optimistic predictions too: a signed mean can cancel large opposite errors.

Performance is undiscounted reward over the first at most 1,000 transitions,
even when diagnostics continue longer. Calibration returns are discounted and
can therefore differ substantially from performance scores.

## Analysis and limits

The independent training seed is the inferential unit. Evaluation episodes
and trajectory pairs are repeated measurements, not extra trained agents.
Show individual seed traces and mean ± one sample standard deviation across
training seeds. Final-checkpoint comparisons are predefined at 50,000 steps;
the area under the evaluation curve is a secondary sample-efficiency summary.
Within each algorithm, calculate paired on-minus-off differences using the
same training-seed identifiers. Three paired runs have very low precision:
describe effects, do not claim algorithm-wide statistical significance.

Bias/performance association is descriptive. Pooled checkpoint correlations
are confounded by training progress and repeated measurements. Prefer plots
within algorithm/checkpoint and per-run change summaries, and do not infer
causation from a correlation. Normalization also affects optimization and state
visitation; it may affect performance through paths other than bias. Every
condition's diagnostic uses its own policy's visited states, so differences
in state distributions are another limitation.

After examining the pilot, choose the final budget and settings, freeze them,
then run fresh independent training seeds (preferably 8–10) and held-out
evaluation seeds. Do not combine pilot tuning runs with the final confirmatory
sample. If bias appears important, add one focused TD3 min-versus-mean target
ablation with everything else fixed. Avoid expanding all architecture/update
ratios at once.

## Sources

- [DDPG paper](https://arxiv.org/abs/1509.02971)
- [TD3 paper](https://proceedings.mlr.press/v80/fujimoto18a.html)
- [Layer Normalization paper](https://arxiv.org/abs/1607.06450)
- [LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Gymnasium time-limit handling](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

## Pilot diagnostic amendment

During the first-seed pilot, some policies were observed to hover beyond the
5,000-step diagnostic limit. Complete-only calibration is therefore retained
as conditional evidence, not used alone to compare final agents. The added
`calibration.py` evaluates the saved final models on the same initial seeds,
retains every trajectory, and compares discounted suffix returns truncated
at 2,500, 5,000 and 10,000 steps of the same rollout. Naturally terminated
trajectories have full returns. Nonterminating trajectories have explicitly
labeled finite-return approximations. Report the observed horizon sensitivity
and censoring counts; no universal future-reward bound is claimed. This is a
pilot-stage method correction before a future confirmatory study, not a
post-hoc change to a preregistered confirmatory result.
