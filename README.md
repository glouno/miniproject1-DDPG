# miniproject1-DDPG

LunarLander experiments with DDPG and TD3, plus notebooks that explain the
continuous-action environment and analyse the saved results. The shared team
report is [RL - miniproject1 on Samskrift](https://samskrift.com/project/7ce99927-c689-46cf-b78a-6ea0d8ecc0d8).

## Layout

```text
src/lunarlander_rl/   Models, training, evaluation, analysis and command-line tools
notebooks/           Environment walkthrough and experiment analysis
configs/             Pilot, benchmark and long-study configurations
tests/               Scientific correctness, recovery and packaging checks
docs/                Protocols, measured results and portability notes
pyproject.toml       Package, dependencies, editable build and CLI entry points
uv.lock              Locked dependencies
.python-version      Python version used by uv
outputs/             Local runs, checkpoints and logs (ignored by Git)
generated/           Regenerated figures (ignored by Git)
```

## Setup and notebooks

Run from this repository root:

```bash
uv sync --locked
uv run --locked python -m ipykernel install --user \
  --name miniproject1-ddpg --display-name "Python (miniproject1-DDPG)"
uv run --locked jupyter lab notebooks/
```

`uv sync` creates the root `.venv` and installs `lunarlander_rl` in editable
mode. Notebook tools belong to the default `dev` dependency group. Select
**Python (miniproject1-DDPG)** in Jupyter. Imports work from the notebook directory
or any other working directory, with no `sys.path` edits:

```python
from lunarlander_rl.experiment import Actor, Critic, Config
from lunarlander_rl.paths import project_root
```

- [Environment walkthrough](notebooks/environment_walkthrough.ipynb): all eight
  observations, both engine commands, rewards and episode endings; a tiny 32×32
  DDPG agent trained for 20,000 transitions; flight playback and state/action/reward
  plots against an untrained actor and a hand-written controller. Saved outputs
  are included. Run All starts a fresh training run under
  `outputs/environment-walkthrough/`; no research data is required.
- [Pilot analysis](notebooks/analysis.ipynb): Q-values, matched Monte Carlo returns,
  normalization, learning curves and measured results. Run All reads saved data
  under `outputs/pilot-20261005/` and writes figures under `generated/`.
  The raw pilot data is required to regenerate its saved outputs.

Edits in `src/lunarlander_rl/` are available through the editable install; restart
the notebook kernel after changing modules that are already imported.
See [PORTABILITY.md](docs/PORTABILITY.md) for macOS/WSL and device details.

## Run and verify

Run the scientific, resume and packaging checks:

```bash
uv run --locked python -m unittest discover -s tests -v
```

Train a single agent:

```bash
uv run --locked lander-train \
  --config configs/pilot.json --algorithm ddpg --layer-norm 0 --seed 10 \
  --output outputs/new-ddpg-seed10
```

Run a fresh pilot, its long-return calibration and replay audit, then regenerate
its figures and tables:

```bash
uv run --locked lander-pilot --output outputs/pilot-new --workers 4
uv run --locked lander-calibrate --pilot outputs/pilot-new --workers 4
uv run --locked lander-audit --pilot outputs/pilot-new --workers 4
uv run --locked lander-analyse-pilot \
  --input outputs/pilot-new --output generated/pilot-new
```

Existing run data is not silently overwritten. CPU is the default; `lander-train`
and `lander-pilot` accept `--device cpu`, `cuda` or `mps`. See
[HARDWARE_RESULTS.md](docs/HARDWARE_RESULTS.md) for the measured comparisons.
Every tool also supports `uv run --locked python -m lunarlander_rl.<module>`.
Use `--help` on a command for its arguments.

## Long studies and TensorBoard

[STUDY_PROTOCOL.md](docs/STUDY_PROTOCOL.md) defines the frozen 40-run comparison,
held-out evaluations, common probes, seed-level analysis and recovery methods.
Prepare a new study, then run it:

```bash
uv run --locked lander-study --prepare --config configs/study.json \
  --output outputs/study-new --workers 8 --max-hours 5
uv run --locked lander-study --output outputs/study-new
uv run --locked lander-study-status --input outputs/study-new
uv run --locked lander-study-review --input outputs/study-new
```

The runner uses the **frozen package** under `<study-dir>/source/lunarlander_rl/`
for workers and analysis. It records source/config/lock/probe hashes and verifies
them before continuing. New inference-only pilot checkpoints cannot resume
training; the study trainer saves replay, optimizers and recovery state.

For a new frozen study, the runner itself can also be launched directly from its
snapshot using the absolute path to this project's interpreter:

```bash
cd outputs/study-new/source
/absolute/path/to/miniproject1-DDPG/.venv/bin/python \
  -m lunarlander_rl.study_runner --output /absolute/path/to/outputs/study-new
```

Legacy flat snapshots remain supported by the launcher, but old studies need
their original source **and original lock/environment** for exact continuation.
The packaging migration changes source and lock hashes; it does not rewrite old
runs to make them appear compatible. `lander-launch-study` installs a macOS
launchd job with caffeinate for an explicitly prepared study.

Export saved study curves without retraining:

```bash
uv run --locked lander-tensorboard-export --input outputs/study-20261005-500k
uv run --locked tensorboard \
  --logdir outputs/study-20261005-500k/tensorboard --host 127.0.0.1 --port 6006
```

Open http://127.0.0.1:6006. `eval/on_policy/return` is the performance score;
`bias_q1` and `rmse_q1` describe calibration. Training return includes exploration
noise. JSON/NPZ records remain authoritative; TensorBoard logs are derived outputs.
Historical runs have no saved optimizer-loss/gradient/activation histories.

## Results and collaboration

Read [PROTOCOL.md](docs/PROTOCOL.md), [PILOT_RESULTS.md](docs/PILOT_RESULTS.md),
and [STUDY_RESULTS.md](docs/STUDY_RESULTS.md) for the completed experiments and
their limitations. These working notes are separate from the shared report.

Clone and work in this repository directly. Raw outputs, checkpoints, TensorBoard
logs and generated figures are ignored and must be retained separately when
sharing or backing up a run. Source and locked dependencies can regenerate runs,
but cannot preserve an irreplaceable existing experiment.

## Actor/critic LayerNorm follow-up

See [NEXT_STUDY_PROTOCOL.md](docs/NEXT_STUDY_PROTOCOL.md),
[WSL_RUNBOOK.md](docs/WSL_RUNBOOK.md) and
[FOLLOWUP_VALIDATION.md](docs/FOLLOWUP_VALIDATION.md).
`lander-followup --prepare` freezes the 140-run queue without launching it.
`lander-prepare-wsl` installs an inactive, resumable user service.
Scientific outputs are incremental JSON/NPZ records with TensorBoard views.
