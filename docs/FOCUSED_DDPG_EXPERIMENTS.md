# Focused DDPG follow-up

Exploratory study of the original critic-LN failure. All conditions use **DDPG,
critic LN on, actor LN off**, with other settings fixed to Stage A.
Reuse its resolved configuration: networks, optimizers, replay, warmup,
exploration, discount, horizons, evaluation/probes, and RNG initialization.
Match the platform/backend; use the same implementation across all new conditions.

| Experiment | Training reward multiplier | Critic:actor updates |
|---|---:|---:|
| Fresh baseline | 1 | 1:1 |
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
- Budget: four conditions, each trained from scratch with Stage A's ten training
  seeds **20–29** and 1,000,000 transitions per seed: **40 training runs,
  40 million transitions**. Pair seeds across conditions and with Stage A;
  identical seed numbers do not guarantee identical trajectories after interventions.
  Use the same evaluation seeds and checkpoint schedule, with dense early
  evaluations; 1M is the primary endpoint and 300k an intermediate checkpoint.
  Optional fifth condition: critic LN off at scale 1 and ratio 1:1
  (50 training runs / 50 million transitions if included).

Reward scaling follows [Henderson et al. (AAAI 2018)](https://doi.org/10.1609/aaai.v32i1.11694).
Save resolved configurations; label this follow-up separately from frozen Stage B.