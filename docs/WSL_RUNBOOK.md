# Frozen LayerNorm follow-up on Paris WSL

The queue is prepared independently of launch. Stage A is the 8-cell algorithm ×
actor-LN × critic-LN factorial, seeds 20–29. Stage B is 6 fixed DDPG learning-rate
conditions, seeds 30–39. Each run collects 1,000,000 training transitions; total
140 runs / 140 million transitions, plus evaluation and diagnostic rollouts.
Stage B automatically follows successful completion and analysis/integrity checks
of A. No human interpretation or outcome-based configuration selection is needed.
Changing B after examining A creates a separate exploratory experiment.

Use `/home/glouno/src/Sorbonne/RL/miniproject` on the native Linux filesystem.
Its locked editable-package environment is the root `.venv`; `uv sync --locked`
recreates it. Do not copy the Mac venv. The package manifest works on macOS and Linux; no CUDA dependency split is needed for this
CPU study. Fresh CUDA benchmarks are separate; exact CUDA continuation is not
validated, so the scientific queue uses CPU throughout.

## Freeze and install, without launching

Run from the mini-project directory. The output directory must be new:

```sh
uv run --locked lander-followup --prepare --preset full --workers 8 --output outputs/ln-followup-20261007
uv run --locked lander-prepare-wsl --output outputs/ln-followup-20261007
```

The manifest hashes all frozen Python sources, the dependency lock, protocol,
probe bank and individual configs. Workers run those copies, not the editable
checkout. The active lock must match. Don't upgrade the environment during a run.
Git revision and content hashes are separate provenance records.

## Launch, monitor, stop, resume

```sh
systemctl --user enable --now rl-ln-followup-20261007.service
systemctl --user status rl-ln-followup-20261007.service
journalctl --user -u rl-ln-followup-20261007.service -f
```

`status.json` records queue outcomes. Each job's `progress.json` updates every
1,000 transitions; `logs/` contains worker exceptions and evaluation summaries.
No automatic unlimited retry: a failed stage blocks the next stage. Inspect the
log before restarting. A controlled stop checkpoints each active worker:

```sh
systemctl --user stop rl-ln-followup-20261007.service
systemctl --user start rl-ln-followup-20261007.service
```

Completed jobs are reused. Interrupted jobs resume; pending jobs start fresh.
The runner lock prevents duplicate queues. The user service works without an SSH
session; lingering is enabled on this machine. WSL must still be running and
Windows must stay awake. Windows shutdown, sleep, updates and WSL startup are
separate host policies; the Linux service does not prevent those events.

## Incremental records and recovery

- Full resume state every **5,000 transitions**, plus evaluation checkpoints and
  graceful-stop checkpoints: replay, online/target models, optimizers, RNGs,
  counters, episode prefix and environment RNG reconstruction state.
- Latest and previous full checkpoints, written to a temporary file, fsynced and
  atomically renamed; containing directories are fsynced too. A corrupt latest
  file falls back to the previous checkpoint. An abrupt kill can lose at most
  the work since the last successful save, normally fewer than 5,000 transitions.
- Evaluation records at 0, every 5k through 50k, then every 25k through 1M:
  append-only JSON metrics, raw Q/state/action/reward/MC NPZ files, inference
  checkpoints including target networks, network and phase audits. Episode journals are flushed and fsynced.
- Last-minibatch optimization records every 1k after warmup, flushed and fsynced.
  TensorBoard is a derived view; it is rebuilt from recovered journals on resume.
- Recovery trims partial/stale journal tails and archives diagnostics beyond the
  restored checkpoint. It does not silently use stale samples twice.
- Job launch requires 20 GiB free disk. This is a launch guard, not a continuous
  disk reservation: monitor free space and stop gracefully before it runs out.

Checkpoints protect against process interruption, not loss of the disk. Keep a
second copy off the WSL disk. For live backups copy small journals and completed
inference/diagnostic files first; avoid copying a mutable `resume.pt` directly.
Use a graceful stop before a complete replay/checkpoint backup, then restart.
A backup is only verified after reading its records/checkpoints successfully.

## TensorBoard and offline analysis

On WSL:

```sh
uv run --locked tensorboard --logdir outputs/ln-followup-20261007 --host 127.0.0.1 --port 6006
```

On the Mac, use local 6007 if the old TensorBoard already occupies 6006:

```sh
ssh -N -L 6007:127.0.0.1:6006 paris-devbox
```

Open `http://127.0.0.1:6007`. Each run has its own curves; dashboard smoothing is
not statistical inference. Completed-seed analyses are saved after each stage:
seed bootstrap intervals, paired contrasts and Holm-adjusted sign-flip p-values.
For monitoring only, the analysis also supports completed subsets:

```sh
cd /home/glouno/src/Sorbonne/RL/miniproject/outputs/ln-followup-20261007/source
/home/glouno/src/Sorbonne/RL/miniproject/.venv/bin/python -m lunarlander_rl.followup_analyse --input /home/glouno/src/Sorbonne/RL/miniproject/outputs/ln-followup-20261007 --stage A
```

Derive prespecified trajectory examples and action maps offline from frozen
checkpoints (training seeds 20–22 for A, 30–32 for B; four fixed checkpoints):

```sh
cd /home/glouno/src/Sorbonne/RL/miniproject/outputs/ln-followup-20261007/source
/home/glouno/src/Sorbonne/RL/miniproject/.venv/bin/python -m lunarlander_rl.followup_examples --input /home/glouno/src/Sorbonne/RL/miniproject/outputs/ln-followup-20261007 --stage A --maps
```

The map holds a replayable state fixed and evaluates a 9×9 continuous-action
sampling grid. Its heatmap is predicted Q; measured MC is collected for five
specified candidate actions, following that checkpoint's actor thereafter.
The grid does not change the policy's continuous action space. Full MC grids,
additional realistic-state banks and offline videos are later diagnostics; they
are not required to launch B or included in training throughput measurements.

## Throughput measurement

`lander-benchmark-followup` supports configurable job counts, steps, repeated
batches and full evaluation. For 20-worker measurement use at least 40 balanced
jobs; the default 16-job batch cannot occupy 20 workers. All sources/configs are
frozen before the first batch. CPU and memory-pressure samples accompany the
results. See [CPU_SCALING.md](CPU_SCALING.md) for the measured settings and choice.
Timing seeds never enter scientific analysis. `lander-benchmark-devices` measures
fresh CPU and CUDA runs separately. Short timings are not full-study duration
forecasts: diagnostic episode lengths change with learning. Use actual full-run
progress to refine completion and storage estimates.
