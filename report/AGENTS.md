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
