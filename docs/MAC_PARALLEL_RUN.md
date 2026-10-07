# Parallel Mac and WSL study

The Mac repeats the WSL queue with eight CPU workers and one Torch/BLAS thread
per worker. Copy the WSL queue's `source/`, `configs/`, `probes.json`,
`manifest.json` and `manifest.sha256` into a new output directory. Do not copy
training outputs or regenerate probes using Mac physics. Verify all recorded
hashes and package versions using the frozen runner before launch.

The scientific revision is `be965b9` on both machines. The Mac manifest is a
verbatim copy of the WSL manifest, so its `interpreter` field records the origin
on WSL. The Mac's `supervisor.json` records the actual local interpreter, platform,
time budget, runner PID, caffeinate PID and supervisor script hash. Per-job
metadata records the actual runtime platform and package versions too.

Mac output: `outputs/ln-followup-20261008-mac`. WSL output:
`outputs/ln-followup-20261007-cpu-scaled`. These are platform repetitions with the
same seeds, not 20 independent seeds to pool into the main study.

## Resilience and eight-hour limit

A user LaunchAgent named `fr.sorbonne.rl-ln-followup-mac-20261008` owns the
supervisor, rather than a terminal or Codex. It starts the frozen runner and
`caffeinate -i -s -w <runner-pid>`. The display can sleep; idle sleep is prevented,
and system sleep is prevented while on AC power. Keep the Mac plugged in and
the lid open. This does not prevent shutdown, reboot or logout.

After eight hours of the supervisor session, it sends SIGTERM to the runner.
The runner forwards the signal to its workers, which checkpoint and pause. The
supervisor waits for that checkpointing to finish before releasing caffeinate.
Eight hours is a graceful-stop request, not a hard kill; the stop may take longer
while an evaluation finishes. KeepAlive is false: failures and the budget pause
are not automatically retried. Resuming starts another explicit time budget.

The plist uses `RunAtLoad=false`; bootstrap registers it and an explicit kickstart
launches it. It does not automatically start again at the next login.

```sh
launchctl print gui/$(id -u)/fr.sorbonne.rl-ln-followup-mac-20261008
tail -f outputs/ln-followup-20261008-mac/supervisor.log
pmset -g assertions
```

For an early graceful pause, send SIGTERM to the supervisor PID recorded in
`supervisor.json`. To resume after it has exited:

```sh
launchctl kickstart gui/$(id -u)/fr.sorbonne.rl-ln-followup-mac-20261008
```

Do not use `kickstart -k` on an active study. Latest/previous resume checkpoints
and journal recovery are the same as WSL; exact continuation is checked within a
platform, not guaranteed across ARM/macOS and x86/Linux.

## Comparing timing and learning

- `progress.json`: current training step and cumulative active elapsed seconds.
- `metrics.jsonl`: evaluation checkpoints with step and cumulative elapsed time.
- `summary.json`: completed job's `elapsed_seconds`, `checkpoint_seconds` and
  `training_seconds`. Checkpoint time includes evaluation, calibration diagnostics,
  audits and inference writes. Training time also includes collection, resume
  writes and other non-evaluation work; it is not pure gradient-compute time.
- `supervisor.json`: total Mac session wall time, including subprocess startup
  and stage analysis; `machine_timing.jsonl`: CPU/memory/load samples each minute.
- WSL `status.json` and systemd timestamps: queue-session wall time. Resuming the
  queue resets its session timer; per-job elapsed time accumulates across resumes.

Compare matching (stage, condition, seed) at equal checkpoints or completed
one-million-step budgets, not just how far each machine has reached at the same
clock time. Report speed ratios for individual paired jobs and aggregate machine
throughput, separating evaluation overhead where possible. Long diagnostic
episodes can differ because the learned policies differ.

Compare each platform's condition effects separately, then paired-seed differences
in rewards, signed bias and calibration error. Identical package versions, seeds
and code do not imply bitwise-identical cross-platform calculations: floating
point kernels and Box2D physics may diverge. This compares complete machine/OS
stacks, not a causal effect of CPU architecture alone. Background applications
and thermal/power state also affect timing; retain resource logs and report them.
Keep this hardware repetition separate from the prespecified within-WSL study.
