# LunarLander mini-project

This workspace contains the experiment implementation, a completed exploratory
pilot and a student-facing analysis notebook. It is separate from the TME
corrections. The live team report is
[RL - miniproject1 on Samskrift](https://samskrift.com/project/7ce99927-c689-46cf-b78a-6ea0d8ecc0d8);
the local pilot analysis is a working note, not a replacement for that report.

Start with [analysis.ipynb](analysis.ipynb), then [PROTOCOL.md](PROTOCOL.md).
A compact measured-results note is [PILOT_RESULTS.md](PILOT_RESULTS.md).
The notebook explains Q-values, matched Monte Carlo returns, normalization,
the measured results and their limitations. It only reads saved experiment
data and regenerates figures; it does not start training.

## Completed pilot

Four conditions: DDPG/TD3, each with critic LayerNorm off/on. Three training
seeds per condition (0, 1, 2), 50,000 transitions per run: 600,000 training
transitions total. All 12 runs completed. Ten fixed evaluation initial seeds
are used per checkpoint. Full resolved configurations, source snapshots,
dependency versions, model checkpoints, episode scores and state/action
diagnostic pairs are stored under `outputs/pilot-20261005/`.

The original diagnostic excluded trajectories that had not naturally ended by
5,000 steps. Some policies hovered, so a supplemental final-checkpoint check
retains all trajectories and compares 2,500/5,000/10,000-step discounted returns.
Its finite approximations and observed horizon sensitivity are explained in
the protocol and notebook. Bias plots based only on completed trajectories
remain explicitly marked as conditional.

Figures, tables and the generated working analysis are in
`../report/generated/lunarlander-pilot-20261005/`. Source figure code is tracked
at `../report/scripts/lunarlander_pilot.py`. The shaded bands and ± values show
**sample standard deviations across three independent training seeds**.
The three-seed pilot does not establish statistically significant effects.

## Reproduce

Commands below run from `/Users/glouno/Documents/Sorbonne` and use the verified
existing **Python (Master MIND RL)** environment. For a fresh machine, the exact
dependency manifest and lock are copied into `environment/`; run
`uv sync --locked --project RL/miniproject/environment`, then use that directory's
`.venv/bin/python` in place of the interpreter below. The run metadata records
the actual Python patch version and CPU platform.

Run scientific correctness checks:

```bash
RL/master-mind-rl/.venv/bin/python -m unittest discover -s RL/miniproject -v
```

Rebuild plots and tables from completed data:

```bash
RL/master-mind-rl/.venv/bin/python RL/report/scripts/lunarlander_pilot.py \
  --input RL/miniproject/outputs/pilot-20261005 \
  --output RL/report/generated/lunarlander-pilot-20261005
```

Independently replay final policies and verify saved state/action/Q alignment and
direct discounted reward sums (writes new audit artifacts, once per run):

```bash
RL/master-mind-rl/.venv/bin/python RL/miniproject/audit_calibration.py \
  --pilot RL/miniproject/outputs/pilot-new --workers 4
```

Run another complete pilot into a **new** directory, then its final long-return
check. Existing run data is never silently overwritten:

```bash
RL/master-mind-rl/.venv/bin/python RL/miniproject/run_pilot.py \
  --output RL/miniproject/outputs/pilot-new --workers 4
RL/master-mind-rl/.venv/bin/python RL/miniproject/calibration.py \
  --pilot RL/miniproject/outputs/pilot-new --workers 4
```

To run a single different configuration, copy/edit a configuration JSON, then:

```bash
RL/master-mind-rl/.venv/bin/python RL/miniproject/experiment.py \
  --config RL/miniproject/configs/pilot.json \
  --algorithm td3 --layer-norm 1 --seed 10 \
  --output RL/miniproject/outputs/new-td3-ln1-seed10
```

Checkpoints are for inference and diagnostics; they do not contain the replay
buffer or a resumable training state. Training outputs are ignored by Git, so
retain the actual data when sharing the notebook or report. Source and locked
dependencies alone can regenerate it, but do not preserve an irreplaceable run.

## Next study

First extend training enough to assess stable learning; the 50,000-step pilot
is still early and its condition means remain below the 200-point reference.
Keep the four-condition comparison. Freeze settings and use fresh training
seeds, preferably 8–10, plus held-out evaluation seeds. Track calibration at
several checkpoints using the improved horizon-sensitive diagnostic. Separate
critic accuracy (signed error and RMSE) from performance (undiscounted reward).

If the main comparison is stable, one focused min-versus-mean TD3 target
ablation can probe the role of pessimistic targets. A common-state/action probe
set, reconstructed by replaying seeded action prefixes rather than restoring
an observation vector, can help distinguish calibration changes from changed
state visitation. Do not expand architecture, update ratio, actor delay and
Optuna search all at once.
