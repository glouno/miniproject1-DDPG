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

## Results and launch configuration

Results are filled from the completed timing files before selecting launch
concurrency. A 15k-step workload does not model a full replay buffer or all later
policy regimes: refine completion estimates from the first full runs. Worker
count changes may be made without changing scientific hyperparameters, provided
resource settings and continuation checks are recorded.
