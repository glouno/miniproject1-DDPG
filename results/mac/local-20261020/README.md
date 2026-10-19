# Mac local runs and diagnostics — archive of 20 October 2026

[Download the release](https://github.com/glouno/miniproject1-DDPG/releases/tag/results-mac-local-20261020).
This preserves the local experiment files needed before removing the original Mac checkout.
The original [Mac Stage A release](https://github.com/glouno/miniproject1-DDPG/releases/tag/results-mac-stage-a-20261008)
remains available separately.

## Contents

- **30 focused DDPG runs**, completed 9 October 2026 on M1 Air: actor delay 2,
  actor delay 3, and reward scale 0.1, each with seeds 20–29 and 1,000,000 transitions.
  Three condition archives contain every run file, including all 49 inference
  checkpoints per run, recovery states, raw diagnostics, journals and TensorBoard logs.
- **Other outputs and provenance**: focused-study manifest, configurations, frozen
  source, locked environment, baseline tables, analysis and logs; focused timing runs;
  horizon-sensitivity inputs; environment walkthrough data; Stage A failure inputs;
  checkpoint-animation inputs and their archive provenance; layout-migration records.
- **Generated artifacts**: all local `generated/` and `report/generated/` files,
  including full checkpoint-animation captures and offline HTML viewers, figures,
  report tables, diagnostic results and build artifacts.

The focused study is exploratory and post-selection. Its historical baseline used
M1 Max; interventions used M1 Air with the same CPU backend and package versions.
Keep this provenance when interpreting its descriptive paired bootstrap intervals.
Some diagnostic inputs are recovered subsets of the previously released Stage A
runs, rather than additional independent experiments.

Original measurements, frozen source hashes and historical absolute paths are
preserved unchanged. Restoring on a different path may require supplying updated
input paths to analysis commands. Exact training recovery still requires the
recorded environment and platform; checkpoints alone do not guarantee portability.

`publication.json` describes the archive assets. `file-inventory.json` records the
path, size, SHA-256 and containing archive of every preserved file. `SHA256SUMS`
verifies the archives and metadata. All archived file contents were checked against
the inventory before upload; uploaded assets are checked against GitHub's SHA-256
asset digests.

## Restore

Clone the repository, then download the release assets into a separate directory:

```sh
git clone git@github.com:glouno/miniproject1-DDPG.git
mkdir mac-local-assets
cd mac-local-assets
gh release download results-mac-local-20261020 --repo glouno/miniproject1-DDPG
shasum -a 256 -c SHA256SUMS
for archive in mac-local-20261020-*.tar.zst; do
  zstd -dc "$archive" | tar -xf - -C ../miniproject1-DDPG
done
```

Archives merge into the checkout's original `outputs/`, `generated/` and
`report/generated/` directories. Use a fresh clone to avoid overwriting local runs.
Recreate the package environment with `uv sync --locked`; the installed `.venv`,
Python bytecode caches and macOS `.DS_Store` files are excluded. Tracked code,
notebooks, report scripts and locked dependencies are preserved in Git.
