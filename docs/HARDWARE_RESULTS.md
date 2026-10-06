# Measured hardware and worker comparison — 5 October 2026

Recommendation for the current small-network study: **M1 Max CPU, eight independent
training workers, one Torch/BLAS thread per worker**. WSL works, including CUDA,
but did not outperform the Mac in these exploratory timings.

## Worker scaling

Each row processes the same eight seeded TD3 + critic LayerNorm jobs, each with
6,000 transitions, 1,000 random warm-up steps, 128×128 networks, batch 128 and
initial/final two-episode evaluations. Wall time includes child Python startup,
training, evaluation and checkpoint saves. Jobs run without rendering.

| Machine | Workers | Batch wall time | Transitions/second |
|---|---:|---:|---:|
| M1 Max | 1 | 78.13 s | 614 |
| M1 Max | 4 | 21.71 s | 2,211 |
| M1 Max | 8 | 13.06 s | 3,676 |
| WSL / i5-13500 | 1 | 131.08 s | 366 |
| WSL / i5-13500 | 4 | 40.84 s | 1,175 |
| WSL / i5-13500 | 8 | 33.51 s | 1,433 |

On the Mac, eight workers gave **1.66×** the throughput of four and **5.98×** the
throughput of one. With eight workers on both hosts, the Mac's throughput was
**2.57×** WSL's. Individual jobs took longer under contention, while total batch
throughput improved. Eight was the highest worker count measured; it is not a
proven global optimum. Increasing threads *inside* each tiny-network run is a
different choice and can cause oversubscription when many processes are active.

## CPU versus accelerator

Separate device tests run the same three seeded jobs sequentially, each with
2,000 transitions and 1,000 warm-up steps. Architecture/batch/evaluation match
above. These times must not be compared directly with the 6,000-step table.

| Machine / backend | Batch wall time | Mean per-job training-loop time |
|---|---:|---:|
| M1 Max CPU | 9.82 s | 1.49 s |
| M1 Max MPS | 39.06 s | 11.15 s |
| WSL CPU | 26.04 s | 4.95 s |
| WSL RTX 3060 CUDA | 28.86 s | 6.91 s |

MPS took **3.98×** the end-to-end CPU time on the Mac; CUDA took **1.11×** the CPU
time on WSL. Accelerator startup contributes to wall time, but the training-loop
times also favor CPU. GPU work is synchronized at timing boundaries. These
measurements show that this implementation's tiny operations and CPU/GPU
handoffs do not benefit from accelerators here. They do not establish an exact
breakdown between physics, replay, Python, transfers and neural-network work.
Larger batches/networks or different update ratios might change the result;
those changes would also change the scientific experiment.

## Evidence and limits

These are **one timing batch per worker count/backend**, on working machines
whose background load was not isolated. Warm-up, OS scheduling, thermals, power
settings and concurrent applications can change results. The training seeds are
not repeated timing trials. Repeat longer batches before treating these ratios
as precise predictions for 300k–500k-step runs. No learning-quality conclusions
are drawn from these short workloads.

Mac: Apple M1 Max, 10 visible CPU cores, 32 GiB RAM, Python 3.11.13.
WSL: i5-13500, 20 visible logical CPUs, about 23 GiB RAM, Python 3.11.16.
GPU: RTX 3060 12 GiB, Windows driver 616.56, PyTorch 2.14.0+cu130 / CUDA 13.0.
Both numerical environments used rl-mind 0.3.0, Gymnasium 1.3.0, NumPy 2.4.6
and Box2D 2.3.10. Floating-point and trajectory results need not be bit-identical
across backends/OSes; use one backend for the final controlled study.

[hardware-benchmarks.json](hardware-benchmarks.json) preserves configurations,
source hashes, timing summaries and per-job outcomes. Full raw runs and source
snapshots remain in ignored `outputs/benchmark-*` directories. Reproduce with
[benchmark.py](../src/lunarlander_rl/benchmark.py); commands are in [PORTABILITY.md](PORTABILITY.md).
The original scientific pilot is unchanged.

The WSL environment passed dependency checks, all four scientific unit checks,
the analysis notebook, real DDPG/TD3 CPU training and real TD3 CUDA training.
Mac CPU/MPS training passed; a CUDA checkpoint from WSL was loaded on Mac CPU
and used to step LunarLander successfully. Eight-worker batches completed all
jobs without errors.
