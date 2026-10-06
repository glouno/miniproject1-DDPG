# Repository guide

This repository studies DDPG and TD3 on continuous `LunarLander-v3`.
Read [README.md](README.md) for setup and commands. Work within this repository;
the sibling TME and RL project directories have separate code and instructions.

## Where to look

- `src/lunarlander_rl/experiment.py` defines the shared models, environment,
  Bellman target and pilot trainer.
- `study_train.py`, `study_runner.py` and `study_diagnostics.py` in that package
  handle resumable studies, frozen source snapshots and evaluation.
- `analyse.py`, `study_analyse.py` and `review_study.py` generate result tables
  and figures. `tensorboard_logging.py` exports saved journals.
- `paths.py` locates the checkout and copies importable source snapshots.
- `notebooks/environment_walkthrough.ipynb` teaches the environment and trains
  a small agent. `notebooks/analysis.ipynb` reads saved pilot data.
- `configs/` contains experiment and smoke configurations. `tests/` covers
  scientific correctness, exact training recovery and package imports.
- `docs/` contains protocols, results and portability notes. Run data belongs
  in `outputs/`; generated figures belong in `generated/`. Git ignores both.

## Environment and checks

Use uv from the repository root. `uv sync --locked` creates `.venv` and installs
`lunarlander_rl` in editable mode. Import the package directly; do not add
`sys.path` edits to notebooks. Use `lunarlander_rl.paths.project_root()` for
repository paths. The notebook kernel is `miniproject1-ddpg`.

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked jupyter lab notebooks/
uv run --locked lander-train --help
```

`pyproject.toml` lists the CLI entry points. Each tool also runs as
`uv run --locked python -m lunarlander_rl.<module>`.
Use `configs/smoke.json` or `configs/study-smoke.json` for short training checks.
After notebook edits, execute the affected notebook with the project kernel.
Run All in the walkthrough trains for 20,000 transitions by default.
The analysis notebook needs the ignored pilot data to regenerate its outputs.

## Experiment rules

Read `docs/PROTOCOL.md` or `docs/STUDY_PROTOCOL.md` before changing an experiment.
Keep configuration, seeds and evaluation settings explicit. Bootstrap through
time-limit truncation; zero future value only at natural termination.

Choose a fresh output directory for a new run. Preserve existing checkpoints,
journals and frozen source snapshots. Resume requires matching source and lock
hashes; do not change recorded hashes to bypass a mismatch. Pilot checkpoints
support inference; study checkpoints contain the state needed to resume.
Report missing raw data when it prevents verification. The local results notes
are separate from the shared Samskrift report linked in the README.
