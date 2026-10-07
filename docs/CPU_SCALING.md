# CPU scaling for the LayerNorm follow-up

Measured on Paris WSL, 7 October 2026. This is an engineering throughput study;
its timing seeds never enter the scientific report as training replicates.

## Available compute

The i5-13500 has 6 performance cores and 8 efficiency cores, with 20 hardware
threads in total. WSL exposes CPUs 0–19 and has no CPU quota on the user slice.
The guest reports a synthetic 10-core/2-thread topology; it does not expose the
host's hybrid-core layout faithfully. Do not assume a guest CPU number maps to
a particular host physical core or pin workers based on that assumption.

WSL2 runs a Linux kernel in a lightweight virtual machine. No dedicated CPU is
reserved here for instruction emulation. Windows, Linux services and training
share the host's compute; select concurrency from measured throughput and memory
headroom rather than simply equating 20 logical CPUs with 20 full-speed cores.
Existing services were retained; Windows power policy and WSL CPU allocation were
not changed.

Increasing workers divides each core's resources among more processes. SMT
threads share execution units and caches, efficiency cores have lower per-core
throughput, and every run also consumes memory bandwidth. Consequently, a run
may take longer even when the whole batch finishes sooner. Compare batch wall
time, not an individual run's speed or CPU utilization alone. The short batch's
last partially occupied wave also affects the estimate.

Sources:
- [Intel i5-13500 specifications](https://www.intel.com/content/www/us/en/products/sku/230580/intel-core-i513500-processor-24m-cache-up-to-4-80-ghz/specifications.html)
- [Microsoft WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

## Controlled timing protocol

Run 40 jobs per worker count: the eight DDPG/TD3 × actor-LN × critic-LN cells,
five timing seeds 9000–9004, 15,000 transitions each. One Torch/BLAS thread per
worker, 2×128 networks, batch 128, replay capacity 100k, warmup 1k, UTD 1.
Evaluate at 0 and 15k using the full 20 held-out episodes and 24 common probes
with 30-action prefixes and 10k diagnostic horizon. Save resume state every 5k.
This preserves the scientific model/update settings; shorter warmup, training
budget and evaluation cadence belong only to the timing workload.

All batch inputs and source snapshots are prepared before the first timed batch.
Run 8, 12, 16 and 20 workers sequentially, measuring startup, training, diagnostics,
checkpoint writes, raw-data validation and analysis. Forty jobs allow 20 workers
to be active; the previous 16-job benchmark could not do that. The same seeded
jobs occur at every count. Compare final weights afterwards to verify that
concurrency has not altered learning.

Resource samples every second record guest CPU utilization, available memory,
swap usage and paging, and summed child RSS. RSS double-counts shared library
mappings; available RAM and swapping are the memory-pressure checks. The command
also supports repeated batches with reversed worker-count order to reduce order
effects. A single sweep is a throughput estimate, not a significance test.

```sh
uv run --locked lander-benchmark-followup --output outputs/cpu-scaling-new \
  --workers 8 12 16 20 --jobs 40 --steps 15000 --full-evaluation
```

The JSON result and per-batch resource samples are the measured records. Learning
curves are timing diagnostics only. Retain old prepared scientific queues and
frozen snapshots; if dependency-lock or source requirements change before launch,
prepare a fresh queue rather than rewriting an existing manifest/hash.

The CPU-only sweep uses committed source `9337b53`, synchronized before launch,
in `outputs/cpu-scaling-cpu-only-20261007`. Its CPU worker subprocesses set
`CUDA_VISIBLE_DEVICES=` before importing Torch, and metadata records that value.
This fixes a reliability issue found in the preceding sweep: the CUDA-enabled
Torch 2.14 Adam implementation queried an accelerator stream even for CPU
parameters, and a 12-worker job failed during CUDA initialization. The failed
sweep and its checkpoints are retained; those timings are not used to choose
the final CPU-only configuration. Other GPU workloads were left running.

## Results and launch configuration

| Workers | Batch time (s) | Transitions/s | Guest CPU | Min available RAM | Mean job time (s) |
|---:|---:|---:|---:|---:|---:|
| 8 | 339.4 | 1,768 | 37.6% | 11.4 GiB | 57.7 |
| 12 | 345.7 | 1,736 | 53.1% | 9.9 GiB | 84.2 |
| 16 | 348.8 | 1,720 | 69.4% | 8.4 GiB | 111.8 |
| 20 | 350.4 | 1,712 | 89.8% | 7.1 GiB | 149.9 |

Select **8 workers**, each with one Torch/BLAS thread. Twelve, sixteen and twenty
workers are respectively 1.8%, 2.7% and 3.1% slower in this sweep. This is a
small difference from a single sequential sweep, not evidence of a statistically
established global optimum. The practical result is a plateau: using 2.5 times
as many workers does not yield a useful speedup and raises CPU activity from
37.6% to 89.8%, while average job time rises from 57.7 to 149.9 seconds.

All 160 timing jobs completed and validated. The final actor, critics and both
target networks are bitwise identical for every matching job at 12/16/20 versus
8 workers (120 comparisons). Frozen source/config/probe hashes also match. Saved
records: `benchmark.json`, each batch's `resources.json`, manifests and
`concurrency_validation.json`. Small amounts of swap-out were observed (about
4/64/148/95 MiB across the four batches); minimum available memory remained
7.1 GiB or more. A live guest pressure check during the 16-worker batch showed
negligible memory/I/O pressure and no user-slice CPU or memory quota. This does
not identify the precise mix of host core sharing, cache/memory stalls and power
limits; no hardware-counter profile was collected.

The earlier 1,844/1,709/1,782 transitions/s figures used a different shorter
workload/source version. They suggested the same plateau but must not be mixed
with this table. Guest CPU utilization excludes Windows applications; other host
and Linux workloads were retained, so fixed-order timing can be affected by
background load. For a close comparison use repeated/reversed-order batches.

A 15k-step workload does not model a full replay buffer or all later policy
regimes. Do not extrapolate it into a precise full-study completion time. Refine
estimates from the first million-step runs. Worker count changes may be made
without changing scientific hyperparameters, provided resource settings and
continuation checks are recorded.

The final queue is `outputs/ln-followup-20261007-cpu-scaled`, 140 million-step
jobs with eight workers. Its frozen manifest records the final documentation
commit rather than the earlier timing commit. All tracked changes are pushed and
synchronized before preparing it. The user authorized launching the scientific
study with eight workers after this sweep; no additional timing tests are needed.
The older prepared queue and benchmark failures are retained.
