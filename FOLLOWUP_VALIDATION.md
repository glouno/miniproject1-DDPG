# Follow-up implementation validation (7 October 2026)

The long scientific queue is prepared separately from these timing and validation
runs. No timing seed contributes to the main study.

## WSL worker measurements

Host: Paris WSL, Intel i5-13500, 20 logical processors, 23 GiB RAM, RTX 3060
12 GiB. Environment: locked Python 3.11; Linux PyTorch CUDA wheel works for both
CPU and CUDA. Existing unrelated services were retained.

Preliminary development batches: 16 jobs (all eight LN/algorithm conditions,
two timing seeds), 15,000 transitions each, 2×128 networks, batch 128, one
Torch/BLAS thread per process. Evaluation, instrumentation, checkpoint writes,
process startup and analysis are included.

| Workers | Wall time (seconds) | Training transitions / second |
|---:|---:|---:|
| 4 | 184.2 | 1,303 |
| 8 | 130.1 | 1,844 |
| 12 | 140.4 | 1,709 |
| 16 | 134.7 | 1,782 |

These are one batch per count, measured while the implementation was being
completed: later snapshots include extra data-integrity checks/logging. They
are useful engineering measurements, not controlled statistical comparisons
or proof of an optimal worker count. An eight-worker repeat of the frozen validated queue took **133.55 seconds**
for 240,000 transitions (**1,797 transitions/s**). Subsequently, target-network
state was also added to inference checkpoints; this adds archive writes without
changing training updates. The final scientific tests were rerun after that change.

Choose **8 independent CPU workers** provisionally: competitive throughput with
less concurrent resource use than 12/16. Do not increase UTD, batch size, network
width or environments per seed simply to increase GPU/CPU utilization: those
changes alter the scientific protocol. Revisit concurrency using identical frozen
snapshots and repeated measurements if the first full runs show a bottleneck.

A mechanical 140M / 1,797 extrapolation is about 22 hours, **not a promised
completion time**. Full runs use 20 evaluation episodes and 24 probes instead of
2/4; learned policies can also require much longer diagnostic trajectories.
Refine runtime and storage estimates from full-study progress rather than
silently shorten the scientific budget. The free disk space was approximately
664 GiB before these short runs; actual retained bytes are in benchmark JSON.

## Fresh CPU/CUDA comparison

Both actor and critic LN were enabled. Each backend/algorithm collected 6,000
transitions with 5,001 critic rounds; two endpoint evaluations were separate.
One run per pair; this is a timing measurement, not a learning-quality comparison.

| Algorithm | CPU training seconds | CUDA training seconds | CUDA / CPU |
|---|---:|---:|---:|
| DDPG | 12.104 | 29.523 | 2.44 |
| TD3 | 12.436 | 29.198 | 2.35 |

The existing GPU workload remained active. These numbers describe available
throughput on this machine, not exclusive-GPU performance. The CPU route is the
chosen route and supports the tested exact recovery. GPUs may become useful for
larger networks/batches, but changing those would change this study's protocol.

## Data and lifecycle checks

CPU tests cover all eight conditions: uninterrupted versus interrupted/resumed
weights and audits match exactly; instrumentation leaves training weights
unchanged. Tests also check LN placement, independent actor/critic initialization,
manual discounted-return sums, terminal masks, corruption fallback and rejected
changes to frozen inputs. Exact CPU continuation was verified separately on both
macOS and WSL.

The actual WSL user-service smoke test stopped with a saved checkpoint, restarted,
finished both stages (14 small runs), and passed raw-data coverage/finite-value
checks. It used distinct timing seeds. Its temporary unit was removed afterwards. The full-study
unit is installed inactive and is not enabled or started by preparation.

Windows AC automatic sleep is already disabled (`powercfg` STANDBYIDLE AC index
0). User lingering is enabled. Neither setting guarantees uptime through a
Windows reboot/update or explicit WSL shutdown.

## Action-map example from the old scientific study

`outputs/action-map-demo-20261007/probe0.png`: old DDPG, critic LN on, actor LN
off, training seed 10, checkpoint 500k, held-out probe seed 30000. State recreated
by replaying its saved physical action prefix. Predicted Q is around 142–144;
five matched first-action rollouts estimate discounted return around 0–4 under
the frozen actor. Three reach the 10,000-step diagnostic cutoff; two terminate
naturally. The figure marks censored estimates rather than labeling them full MC.
The heatmap's narrow color scale amplifies differences under 2; the much larger
issue in this example is the predicted-value offset. This is one state/seed,
not a population-level mechanism or evidence that all actions have identical
true value. Only five candidates have MC values; the full heatmap is prediction.
Raw candidate rewards, grid values and checkpoint provenance accompany the plot.
