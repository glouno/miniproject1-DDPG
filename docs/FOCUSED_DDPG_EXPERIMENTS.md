# Focused DDPG follow-up

Exploratory study of the original critic-LN failure. All conditions use **DDPG,
critic LN on, actor LN off**, with other settings fixed to Stage A.
Reuse its resolved configuration: networks, optimizers, replay, warmup,
exploration, discount, horizons, evaluation/probes, and RNG initialization.
Match the platform/backend; use the same implementation across all new conditions.

| Experiment | Training reward multiplier | Critic:actor updates |
|---|---:|---:|
| Reward-scale intervention | 0.1 | 1:1 |
| Actor-delay intervention | 1 | 2:1 |
| Stronger actor-delay intervention | 1 | 3:1 |

- Keep actor LN off to isolate interventions on the original failing condition;
  this does not imply actor LN is generally harmful or unnecessary.
- Keep one critic update per training transition after warmup. Delay only actor
  updates; update both target networks every critic update with unchanged tau.
- Keep environment rewards and saved replay rewards raw. Apply the multiplier
  once to immediate rewards when building critic training targets, never during
  evaluation. Score raw returns; compare raw discounted MC with Q divided by
  the training reward multiplier.
- Reuse Stage A's DDPG actor-LN-off/critic-LN-on baseline (scale 1, ratio 1:1).
  Budget: three new conditions, each trained from scratch with Stage A's ten
  training seeds **20–29** and 1,000,000 transitions per seed: **30 training runs,
  30 million transitions**. Pair seeds across conditions and with Stage A;
  identical seed numbers do not guarantee identical trajectories after interventions.
  Use the same evaluation seeds and checkpoint schedule, with dense early
  evaluations; 1M is the primary endpoint and 300k an intermediate checkpoint.
  No new baseline or additional LN-off control is scheduled.

Reward scaling follows [Henderson et al. (AAAI 2018)](https://doi.org/10.1609/aaai.v32i1.11694).
Save resolved configurations; label this follow-up separately from frozen Stage B.

## Local execution

Use `python -m lunarlander_rl.focused_runner`: `--prepare` freezes the three
conditions; `--launch --label org.glouno.rl.ddpg-focused-20261009` starts the
prepared queue under macOS launchd with sleep prevention. Keep the Mac plugged
in and the lid open. No reward-based early stopping or time-budget cutoff.

On the M1 Air (8 GB), a 9 October short timing check of identical 12-job batches
took 33.6 / 29.5 / 32.5 seconds with 4 / 6 / 8 CPU workers. Choose **six workers,
one Torch/BLAS thread each**; eight increased swap usage and did not improve
throughput. Budget roughly **4–6 hours** for 30 full runs including evaluations;
this is an extrapolation, not a measured full-study duration.

Retain the existing saving scheme: 49 inference/evaluation checkpoints per seed,
full diagnostic trajectories, and rolling latest/previous recovery states every
5k. Frozen sources/configs/probes and baseline CSVs live in the new output folder;
the historical baseline is reused, not retrained.
