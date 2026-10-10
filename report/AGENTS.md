# Shared Samskrift report

- Project: **RL - miniproject1**
- Project ID: `7ce99927-c689-46cf-b78a-6ea0d8ecc0d8`
- URL: https://samskrift.com/project/7ce99927-c689-46cf-b78a-6ea0d8ecc0d8
- Main document: `main.tex`; compiler: `pdflatex`.
- Mapping verified through Samskrift on 2026-10-09 (owner access).

Samskrift is the source of truth. Read the current remote files before editing
and compile through Samskrift after changes. Apply the report-format skill.
Keep reproducible report scripts in `scripts/` and intermediate output and PDF
review files in `generated/`, ignored by `.gitignore`. Existing training data,
checkpoints, configurations and frozen sources must remain unchanged.

`scripts/build_results_tables.py` reads the Mac Stage A, WSL Stage B and Mac
focused follow-up records. It writes a generated LaTeX fragment for the shared
report, with explicit hyperparameter columns and final seed-averaged results.

`scripts/build_target_update_figures.py` uses the published
`results/wsl/target-updates-20261010/` metrics to match Figure 8's reward/bias
panels. It validates all 20 intervention runs and uses Mac Stage A `ddpg-a0-c1`
(critic LN on, actor LN off) for the plotted baseline and table improvements in raw units,
matching the existing learning-rate/update-ratio reference. Outputs are in
`generated/ddpg-target-updates/`. The release's significance tests compare
against a separate, unplotted historical WSL baseline; do not relabel those tests
as comparisons against the plotted Mac baseline.
