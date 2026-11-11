# Proposed follow-up: where LayerNorm changes actor-critic learning

Prepared 7 October 2026. This is a planning document, not a frozen or launched
experiment. Freeze the source, lockfile, configurations, seeds and diagnostics
after validation, before substantive training. The existing 5 October study
remains unchanged. This local protocol does not replace the shared Samskrift report.

## 1. Questions and prior evidence

Primary question: how do actor normalization, critic normalization and their
interaction affect performance and critic calibration in DDPG and TD3?
Secondary question: can the failing DDPG + critic-LN behavior be changed by
slowing actor learning, critic learning, or both? Which observations support
specific mechanisms, rather than just identifying a higher-scoring configuration?

The previous experiment used critic-only LN, deliberately isolating the critic
intervention. It did not test actor LN or establish that actor LN is unnecessary.
The project implementation uses both a target actor and target critic in DDPG;
the simplified TME implementation is not the project implementation.

Verified old-run means across ten training seeds (on-policy episode aggregation):

| DDPG setting | Step | Evaluation return | Mean Q1 | Mean discounted MC | Q1 - MC |
|---|---:|---:|---:|---:|---:|
| Critic LN off | 50,000 | -62.27 | 70.77 | -16.78 | 87.54 |
| Critic LN on | 50,000 | -17.60 | 115.10 | -15.48 | 130.58 |
| Critic LN off | 200,000 | -16.69 | 83.54 | -14.23 | 97.77 |
| Critic LN on | 200,000 | 51.10 | 144.03 | 2.08 | 141.96 |
| Critic LN off | 500,000 | 228.99 | 110.60 | 64.10 | 46.50 |
| Critic LN on | 500,000 | -40.89 | 167.90 | -19.73 | 187.62 |

Bias already starts around +92 for both at step zero because untrained Q is
near zero while the untrained policy earns negative discounted returns. Thus a
positive bias alone is not proof of learned value divergence. The notable
finding is its persistence/increase and the mismatch with realized behavior.
LN initially improves mean performance, then fails to sustain improvement.

Exploratory final-state stratification of saved pairs also shows elevated error
in multiple regions: mean per-seed signed error in airborne samples is 182.11
with LN versus 61.52 without; for samples with either leg in contact, 135.56
versus 26.46. All ten seeds contributed to both subsets. These statistics pool
pairs within a seed, then average seeds, unlike the primary episode aggregation.
They are post-hoc, correlated, visitation-dependent diagnostics, not significance
tests or causal results. The failure is not visibly confined to leg-contact states.

## 2. Mechanistic hypotheses and discriminating observations

H1: the actor exploits inaccurate high-valued actions. Examine whether the
critic prefers actions whose matched MC returns are poor, and whether large
action gradients/action saturation precede performance deterioration. Slower
actor learning is an intervention on this feedback loop, not proof of mediation.

H2: critic learning/bootstrapping becomes poorly calibrated. Examine online and
target Q, target labels, TD residuals, terminal versus nonterminal samples and
weight growth. Slower critic learning tests sensitivity to optimization speed.
A small TD loss is compatible with a wrong self-consistent bootstrap solution.

H3: actor normalization changes control behavior. Examine pre-tanh outputs,
saturation, effective engine firing, action/state distributions and trajectories.
Actor LN could improve or worsen optimization or control; neither is presumed.

H4: learned scales or representation loss matter. Examine LN gains/biases,
pre-LN variance, post-LN activations, ReLU activity, feature variability and final
head weights. LN removes common feature offset/scale, changes action derivatives,
and introduces learned affine parameters. It does not bound learned scalar Q.
An affine=False or normalization-placement ablation is deferred unless these
measurements motivate it; observed weight growth alone does not establish cause.

H5: measurement/implementation errors imitate bias. Check same physical (s,a),
same contemporaneous actor/critic, reward indexing, terminal masks, action scaling,
diagnostic cutoff sensitivity and source/environment identity before interpretation.

## 3. Stage A: complete actor/critic normalization factorial

Use the full 2 x 2 x 2 design, all on the same WSL/backend:

| Algorithm | Actor LN | Critic LN | Interpretation |
|---|---|---|---|
| DDPG | off | off | Original unnormalized control |
| DDPG | off | on | Original failing condition |
| DDPG | on | off | New actor-only intervention |
| DDPG | on | on | New combined intervention |
| TD3 | off | off | Original unnormalized control |
| TD3 | off | on | Original critic-only intervention |
| TD3 | on | off | New actor-only intervention |
| TD3 | on | on | New combined intervention |

The colleague's four additional configurations are actor-on versions of each
original algorithm/critic-LN condition. Rerun the original four as contemporaneous
WSL controls: do not treat historical Mac runs as exchangeable with new WSL runs.

Ten fresh training seeds 20-29 in every condition. Budget 1,000,000 transitions
per run: 80 runs, 80 million training transitions. Final 1M is primary; 500k is a
prespecified secondary endpoint aligned with the previous study. No reward-based
early stopping or best-checkpoint selection. Plot the full learning trajectory.

LN specification: after each hidden Linear and before ReLU; eps=1e-5,
elementwise_affine=True, default gain 1/bias 0 initialization. Actor and critic
flags are independent. No LN on the actor's final action logits, the tanh output,
or scalar Q output. Target copies have the corresponding online architecture.
Do not normalize raw observations or rewards as part of this intervention.

All other numerical settings match the original main study:
- LunarLander-v3, continuous actions, gravity -10, no wind; performance horizon 1000.
- Actor/critic two hidden layers of 128; float32; no mixed precision or compilation.
- Gamma .99, tau .005; actor/critic Adam lr 3e-4, betas (.9,.999), eps 1e-8;
  no weight decay, reward clipping or gradient clipping.
- Replay 100,000, batch 128, 5,000 initial uniform actions; one critic round per
  collected transition thereafter. Keep replay sampling/collection unchanged.
- Exploration Gaussian sigma .1 in normalized action units, clipped to action bounds.
- DDPG: one critic; actor and both targets updated each critic round.
- TD3: independent twin critics, minimum target, smoothing sigma .2 clipped .5,
  actor/target updates every second critic round.

Use independent seeded streams for network initialization, exploration, replay,
training environment and diagnostics. In Stage A, actor-on changes should leave
initial critic weights identical within a matched seed, and vice versa. Use
isolated RNG contexts for initialization; LN affine initialization draws no random
numbers. Test this property. Shared seed indices improve experimental blocking
but do not make later trajectories identical. Use fixed independent evaluation
and diagnostic streams so instrumentation consumes no training randomness.

## 4. Evaluation schedule and diagnostic data

Evaluate at 0; every 5k through 50k; then every 25k through 1M (49 checkpoints).
At each checkpoint freeze contemporaneous actor and critic(s), and evaluate
20 held-out reset seeds 40000-40019, deterministic actions without exploration.
Use the same evaluation seeds across all conditions/checkpoints. Report arithmetic
mean undiscounted return within the first <=1000 steps, length, return distribution,
termination/truncation, episode-level action and state summaries.

Diagnostics follow the frozen policy beyond the 1000-step performance window to
natural termination or 10,000 steps. Save every tenth (s,a) in the first 1000
steps, critic predictions, full reward sequence, episode offsets and termination
reason. Retain censored trajectories; compare 2500/5000/10000 observed suffixes.
Finite censored returns are approximations; never fill tails with critic Q.
Compute Q1, Q2 and min errors as applicable. Keep separate:
- Signed error Q - MC, RMSE, MAE, positive-error fraction, mean Q and mean MC.
- Equal-episode aggregation for on-policy metrics, then equal training-seed weight.
- Pooled RMSE within each fixed probe bank, then equal training-seed weight.
- Histograms/scatter plots, quantiles, and conditional errors; averages can cancel.

A fixed common bank of 24 pairs uses reset seeds 41000-41023, 30 seeded uniform
prefix actions and one seeded fixed first action. Serialize and hash the bank
before launch; reconstruct the physical state via reset/prefix and assert exact
observation equality. Continue with each evaluated actor afterwards. This holds
initial (s,a) fixed, not continuation policy or true Q^pi.

Common random-prefix states have limited coverage. Add a secondary realistic
bank offline AFTER Stage A from deterministic, prespecified held-out trajectories of
the actor-off/critic-off DDPG and TD3 controls: select seed 20, checkpoint 500k,
evaluation reset seeds 42000-42003 and t=25,75,150 where those times exist;
include all available records, no selection for interesting errors. Record absent
times and physical action prefixes. This is a deferred secondary diagnostic:
freeze the bank before evaluating Stage B checkpoints on it, rather than blocking
Stage B training. Reference trajectories do not enter training/replay.

On-policy phase/error plots: first 100 steps vs later; airborne vs either leg
contact; abs(angle) <=.2 vs >.2; descending vs ascending; near-bound actions
(abs(normalized action) >.95). Report counts and seeds represented in each bin.
Do not infer a phase's contribution from conditional means without its frequency.

Record engine behavior using the actual pinned environment's firing rules
(verify these in its source): main/side activation fractions, normalized action
histograms, effective thrust summaries, velocity/angle/contact histories.
Do not equate a negative main-engine action with reverse thrust.
Classify natural termination from actual environment state/info: grounded/resting,
crash, out of bounds; keep hovering/time-limit separate. Do not invent a landing
classifier from total reward or assume every natural termination is success.

## 5. Training/optimization instrumentation

Every 1000 transitions, log bounded summaries from actual training batches:
critic loss, actor loss when updated, TD residual quantiles, predicted Q and
target-label quantiles, mean immediate reward, bootstrap contribution, terminal
fraction and replay action/actor action distance. Specify last-batch summaries
as such; do not label them whole-run averages. Log gradient norms before optimizer
steps, parameter/update norms, and online-target distances. In TD3 include critic
disagreement; distinguish individual critic loss from sum of two losses.

At evaluation checkpoints, on a separate frozen audit batch/state bank, collect:
- Layer-wise pre/post-LN mean, variance, near-zero variance frequency, LN gains
  and offsets, ReLU zero fraction and fraction of features always zero in the bank.
- Actor pre-tanh distribution, abs(action)>.95 fraction, action derivatives and
  critic action-gradient norm ||dQ/da||. Report units and normalized-action scaling.
- Final Q-head norm/bias and feature variability/effective rank (descriptive only).
- Online/target Q disagreement and policy changes on fixed observations.

Histograms are diagnostic, not independent statistical samples. Hooks must be
disabled outside audit points, removed afterwards, and never modify activations.
Use frozen copies and restore modes; use inference_mode only where no derivative
is needed. Gradient diagnostics need autograd but must not alter optimizer states,
parameter .grad fields, training RNGs or the training environment.

Action-value maps at 50k, 200k, 500k, 1M: choose six available fixed probe states
by bank index, never by error. Evaluate a 9 x 9 grid of legal continuous actions.
Plot critic Q contours and the actor's chosen action. For five predefined candidate
actions (current actor, zero, two opposite interior diagonal actions, and critic
grid argmax), reset/replay to the identical state, force that action, then follow
the same frozen actor. Compare predicted Q differences to MC return differences.
The grid argmax's error is selected, so report it explicitly as an exploitation
diagnostic, not an unbiased population error estimate. This tests whether Q's
preferred actions are actually better under that policy.

Save deterministic trajectory plots and render videos OFFLINE at 50k, 200k, 500k,
1M for seeds 20,21,22 in all eight conditions using evaluation seed 40000. Videos
are deferred to offline analysis; raw trajectories and checkpoints are retained. Also
show best/median/worst final training seeds only as labeled post-hoc examples;
keep all seed-level tables. Rendering must not slow or change training.

## 6. Stage B: focused optimization study of original failing DDPG

Freeze both stages before launching. Stage B may automatically follow successful
completion and data-integrity checks of Stage A; interpreting A is not a launch
gate because B's configurations, endpoints and seeds are fixed in advance.
Any new configurations chosen after inspecting A are exploratory and require
a separate protocol and fresh confirmation seeds. Focus stays on actor-LN off, critic-LN on,
even if actor LN rescues it; that asks why the original condition fails.

Use a small 2 x 2 learning-rate factorial, with everything else unchanged:

| Critic lr | Actor lr | Question |
|---:|---:|---|
| 3e-4 | 3e-4 | Fresh baseline |
| 1e-4 | 3e-4 | Does slower critic optimization change failure? |
| 3e-4 | 1e-4 | Does a slower actor reduce exploitation of critic error? |
| 1e-4 | 1e-4 | Does reducing both rates help; is there an interaction? |

Add two critic-LN-off controls: both rates 3e-4 and both rates 1e-4. They help
distinguish a general optimizer effect from an LN-associated effect for the
joint-rate change; single-rate LN-specific effects would need matched additional
controls and cannot be claimed from this reduced design.

Ten fresh seeds 30-39 per configuration, 1M transitions, same diagnostics:
60 runs, 60 million transitions. Same fixed evaluation/probe seeds as Stage A.
Independent training seeds prevent reusing Stage A's tuning outcomes as confirmation.
No selected best seed/checkpoint. Stage B is a prespecified mechanistic follow-up,
not a claim that the selected rate is globally optimal.

Stage A + B: 140 runs / 140 million training transitions, plus diagnostic rollouts.
The queue prepares both stages without starting them. If compute is limited, shorten
the number of questions/stages, not the number of independent seeds silently.

## 7. Deferred interventions

Do not initially vary depth, batch size, discount, reward scaling, replay size,
exploration noise or UTD. They change multiple mechanisms and make explanation
harder. Three hidden layers/UTD>1 are scaling questions, not the first failure test.

If Stage A/B supports actor-critic feedback, test actor delay 1 vs 2 in DDPG
as an explicitly named DDPG-delay ablation. Hold target-update cadence fixed
per critic round to avoid inadvertently changing target lag with actor delay.
The current engine ignores policy_delay in DDPG and requires an explicit change.

If learned LN affine scales appear problematic, test affine=True/False with a
separately declared parameterization intervention. If target bias is the main
remaining question, use a separately controlled twin-critic mean-vs-min target
experiment with actor delay, noise and architecture fixed. TD3-vs-DDPG changes
three mechanisms and cannot identify which mechanism caused a difference.

No observational correlation between bias and return, including its temporal
ordering, proves bias causes performance. Interventions can narrow explanations
but learning-rate changes also affect exploration, fitting and policy dynamics.

## 8. Statistical analysis

Training seed is the experimental unit (n=10 per cell); episodes/steps/pairs
are nested observations, never additional training replicates. Common seed
indices block initialization/randomness; pairing is declared before training.

Primary endpoint: final return at 1M. Primary contrasts within each algorithm:
actor-LN effect with critic LN on, and critic-LN effect with actor LN off
(four contrasts total). Test two-sided paired seed differences, report effect
sizes, paired 95% seed-bootstrap intervals (10,000 resamples), raw p values
and Holm-adjusted p values across these four primary return contrasts. Use
paired sign-flip randomization tests on mean differences with all 2^10 signs;
state their symmetry/exchangeability assumptions. With ten seeds, intervals
remain approximate and power for small effects is limited.

Report the factorial interaction per algorithm:
(both-on - critic-only) - (actor-only - neither), computed within each seed;
supporting descriptive interval, not an extra unadjusted significance claim.
Show every seed, mean/SD, median/IQR and paired effect plots. On-policy signed
error/RMSE, common-probe metrics, AUC and 500k are prespecified secondary metrics;
label their intervals descriptive unless a separate multiplicity family is frozen.
Do not use band overlap as a significance test. Trajectory/bootstrap plots are
pointwise; no claim of a significant difference at every checkpoint.

For Stage B, critic-rate and actor-rate main effects and their interaction are
the three prespecified primary return contrasts, within seed; Holm correction
within that family. LN-off controls and mechanistic metrics support interpretation.
Do not declare a higher-scoring grid cell a mechanism or reuse its best seed.

All failures are reported. Numeric NaNs count as training failures and are not
silently excluded or converted into fabricated returns. Separate failure rate
from valid-return summaries; define the analysis of any failed cell before
unblinding comparisons. Infrastructure interruptions resume unchanged states.

## 9. WSL execution preparation (next implementation phase)

Read-only verification on 7 October: ssh paris-devbox reaches paris-devbox-wsl;
checkout /home/glouno/src/Sorbonne/RL is clean at 301a716, behind the Mac;
i5-13500, 20 logical CPUs, ~23 GiB RAM; RTX 3060 12 GiB; installed torch
2.14.0+cu130 reports CUDA available. nvidia-smi is available at
/usr/lib/wsl/lib/nvidia-smi, not on the noninteractive PATH. GPU already has
an unrelated process using memory; do not stop it. Recheck resource availability.

Before launch: sync source without overwriting unrelated files; uv sync --locked;
record environment versions/lock hashes; run scientific and instrumentation tests.
The Mac now records miniproject as a separate private Git submodule at
https://github.com/glouno/miniproject1-DDPG.git. The old WSL checkout predates
that extraction. Preserve/verify its ignored outputs before migrating the parent,
initialize the submodule recursively and verify its pinned revision; a simple
parent git pull alone is not a sufficient checkout migration plan. The training
workspace will be /home/glouno/src/Sorbonne/RL/miniproject, with its standalone
environment/pyproject.toml and uv.lock, or an explicitly verified parent environment.
Add independent actor_layer_norm/critic_layer_norm fields with backward-compatible
loading of old layer_norm checkpoints. Preserve old source/locks for old-run resume.

Benchmark the final instrumented workload on CPU with 4/8/12/16 concurrent runs
(one Torch/BLAS thread each) and a representative CUDA alternative. Use balanced
condition mixes, same training/diagnostic budgets, warm-up and repeated timings;
measure aggregate transitions/s, diagnostic time, memory, swap and checkpoint I/O.
Choose throughput for completed experiments, not maximal CPU/GPU utilization.
Historical small-network timing favored CPU; changing batch/architecture to fill
the GPU would change the experiment. Keep one backend for each complete stage.
Do not treat hardware tuning runs as scientific seeds.

Run from the Linux filesystem, not /mnt/c. Pin OMP/MKL/OpenBLAS threads to one;
measure available CPU quota/load and peak memory before selecting workers. Start
8 as a measured baseline, not a claimed optimum. CPU-internal 20-thread models
plus many workers would oversubscribe. CUDA multi-process tiny runs need measurement,
not an assumption of speedup. CPU checkpoint evaluation can dominate runtime.

Use a user systemd service with persistent logs, bounded workers and resumable
runner state, independent of SSH. Verify user lingering/startup behavior and Windows
host sleep/reboot/update behavior separately; Linux service persistence alone does
not prevent Windows sleep. Do not change Windows power policy without discussion.
No five-hour scientific deadline; administrative shutdowns checkpoint and resume.
Test SIGTERM, process kill, partial journal recovery and exact continuation.

Use a frozen checkout/source snapshot and immutable manifest. Do not upgrade
packages or modify source in active runs. Atomic latest/previous resume checkpoints
retain optimizers/replay/RNG/physical episode prefix. Save progress/heartbeat and
per-job exceptions, avoid unlimited automatic restarts, detect disk pressure.
Verify storage estimates from representative instrumented runs before allocating
140 jobs; raw data is ignored by Git and needs a separately verified backup.
No automatic deletion of raw diagnostics or previous study data.

TensorBoard: live metrics plus offline regenerated event logs, local bind only;
view over SSH tunnel (for example local port 6007 to remote 6006 if 6006 is in use).
JSON/NPZ are authoritative; export timestamps are not training wall time. Aggregate
scientific intervals come from the analysis scripts, not dashboard smoothing.

## 10. Launch gates and deliverables

Required before freezing:
1. LN placement/output bounds/target copies and independent initialization tests.
2. Terminal vs truncation, manual Bellman targets, gradient routing tests.
3. Instrumented vs uninstrumented weights/RNG/trajectories match on short CPU runs.
4. Exact interrupted-vs-uninterrupted training for all eight conditions.
5. Reload checks of saved Q(s,a), independently summed MC returns and common-state replay.
6. CPU/CUDA portability smoke tests; choose backend after throughput measurement.
7. Estimated runtime/disk/RAM, restart demonstration, immutable manifest and backup plan.

Deliver: manifest/lock/source hashes, all seed-level learning/calibration curves,
Q-versus-MC and phase plots, action-value maps, trajectory examples/videos,
activation/gradient/weight diagnostics, paired factorial contrasts and interaction,
runtime/failure/censoring table, and an explanation that separates evidence,
hypotheses, interventions and remaining uncertainty. Update the live shared report
only through its established Samskrift workflow in a later writing task.

## Implementation status for the prepared queue

`followup_runner.py` implements both fixed stages, frozen sources/configs/probes,
bounded independent CPU workers and resumable scheduling. `study_train.py` saves
full state every 5k, journals optimization summaries every 1k, and writes network
and phase audits at each evaluation. `followup_analyse.py` produces seed-level
return/bias/RMSE curves and prespecified primary return contrasts.

`followup_examples.py --maps` derives the prespecified action maps and trajectory
examples offline from saved checkpoints. Secondary realistic-state probes,
effective-thrust plots, videos, seed-level phase aggregation and mediation-style
exploration remain offline analysis tasks, not automated launch gates. Conditional
phase summaries saved during training pool pairs within a seed and are descriptive;
their counts must accompany any later cross-seed aggregation. No causal explanation
is inferred automatically from a successful queue or a significant contrast.

Full resume is validated on CPU; CUDA is tested only with fresh benchmark runs.
No scientific job is launched by preparation or service installation.

## References

- Layer Normalization: https://arxiv.org/abs/1607.06450
- Addressing Function Approximation Error in Actor-Critic Methods (TD3):
  https://proceedings.mlr.press/v80/fujimoto18a.html
- Bigger, Regularized, Optimistic (LN is part of a combined architecture/training recipe):
  https://arxiv.org/html/2405.16158v3
- Original local study: STUDY_PROTOCOL.md, STUDY_RESULTS.md;
  outputs/study-20261005-500k/manifest.json and per-run metrics.jsonl/diagnostics.
