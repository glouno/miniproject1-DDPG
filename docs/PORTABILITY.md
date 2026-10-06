# macOS and WSL execution

Clone `miniproject1-DDPG` on each machine and run commands from its root.
The root `pyproject.toml` defines the installable `lunarlander_rl` package,
with `uv.lock` and a separate root `.venv` on each machine. `uv sync`
installs the package in editable mode; notebooks import it without `sys.path` edits.
Notebook dependencies are in the default `dev` dependency group.
The lock records the runtime versions used in the pilot:

```toml
dependencies = [
    # other notebook and analysis packages remain in the actual manifest
    "gymnasium[classic-control,box2d]==1.3.0",
    "rl-mind==0.3.0",
    "numpy==2.4.6",
    "torch==2.14.0",
]
```

The previous environment layout's locked installation succeeded on Apple
Silicon macOS and x86-64 WSL. The src-layout migration preserves every third-party
locked version and has been checked on macOS. Default
PyPI PyTorch wheels provided macOS CPU/MPS and Linux CUDA 13.0 builds; WSL's
Windows NVIDIA driver 616.56 ran a real CUDA tensor operation successfully.
No platform-specific index override is required for these tested machines.
On a machine with an older driver, choose a compatible CUDA wheel using
[uv's explicit PyTorch indexes and platform markers](https://docs.astral.sh/uv/guides/integration/pytorch/)
and regenerate/test the lock; do not install an unrecorded replacement into the venv.

Use the Windows NVIDIA driver for WSL GPU access. Do not install a Linux NVIDIA
GPU driver inside WSL. On this host, `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi`.
See [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Setup and checks

From this repository root:

```bash
uv sync --locked
uv pip check --python .venv/bin/python
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m ipykernel install --user \
  --name miniproject1-ddpg --display-name "Python (miniproject1-DDPG)"
uv run --locked jupyter lab notebooks/
```

The analysis notebook needs the ignored pilot data in addition to Git sources.
It was copied separately to WSL and executed successfully. No virtual environment
or authentication files were copied. Generated data and figures remain ignored.
If an old environment claims Box2D is installed but imports fail, restore the
locked wheel with `uv sync --locked --reinstall-package box2d` and rerun the checks.
A normal clean WSL installation did not require that repair.

## Training devices

```bash
uv run --locked lander-train \
  --config configs/pilot.json --algorithm td3 --layer-norm 1 \
  --seed 10 --device cpu --output outputs/new-cpu-run
```

Change `--device` to `cuda` on WSL or `mps` on the Mac. CPU is the default;
unavailable devices raise an error. Training models and sampled minibatches
use the selected backend. Box2D and replay storage stay on CPU. Environment
actions are copied back to CPU. Evaluation uses a frozen CPU copy of the current
networks on all backends; saved checkpoints can be reloaded on CPU across hosts.
Accelerated backends and OSes are not promised to reproduce bit-identical runs.
Keep the final scientific study on one backend and record it in run metadata.

A full factorial batch accepts worker count, training seeds and a custom config:

```bash
uv run --locked lander-pilot \
  --output outputs/new-study --workers 8 --device cpu \
  --seeds 10 11 12 13 14 15 16 17
```

Each worker is an independent training subprocess, with one Torch/BLAS CPU thread.
This parallelizes independent seeds, not data collection within a single agent.
Changing environment parallelism, update ratio or batching would change the
learning procedure and requires a separately specified experiment.

## Timing

```bash
uv run --locked lander-benchmark \
  --output outputs/new-cpu-benchmark \
  --device cpu --workers 1 4 8 --jobs 8 --steps 6000
```

The timing workload fixes TD3, critic LayerNorm, 128×128 networks, batch 128,
1,000 random warm-up transitions, and two evaluation episodes at initial/final
checkpoints. Every worker-count batch uses the same eight seeds. End-to-end
wall time includes child interpreter startup, training, evaluation and saves;
per-run summaries separate training-loop time from checkpoints. GPU operations
are synchronized at timing boundaries. Short timing runs do not support learning
quality conclusions. One batch per worker count is an exploratory comparison,
not a precise repeated performance benchmark.

Use `--device cuda` or `mps` with `--workers 1` first. Many CPU workers can improve
throughput; many GPU workers may contend for one accelerator. Prefer the measured
best CPU worker count for this small-network workload unless GPU timing proves
otherwise. Do not alter scientific hyperparameters just to make a GPU busy.

## Measured recommendation

See [HARDWARE_RESULTS.md](HARDWARE_RESULTS.md). For the current configuration,
use Mac CPU with `--workers 8`. Four-to-eight workers improved short-batch
throughput by 1.66×; CUDA and MPS were slower than CPU on their respective hosts.
Eight is the largest tested count, not a guaranteed optimum for every workload.
