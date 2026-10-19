# Interactive fixed-state checkpoint animation

Build a self-contained offline HTML explorer from **all** inference checkpoints
in one run directory. Supply an existing `state_action_evaluation` directory to
reuse its exact reconstructed physical state. No training is performed.

```bash
uv run --locked python -m lunarlander_rl.checkpoint_animation \
  --run outputs/checkpoint-animation-input-20261014/mac-stage-a-20261008/A/ddpg-a0-c1-seed21 \
  --state-source generated/stage-a-ln-on-seed21-eval40009-state125-20261009 \
  --output generated/checkpoint-animation-state125-ln-on-all-20261014 \
  --workers 3
```

Open `index.html` in a modern browser (gzip `DecompressionStream` support is
required). It contains its own data; no server or network access is needed.

- Play or scrub saved training checkpoints, with no interpolation. Every frame
  has equal screen time; the history plot uses actual training transitions.
- Compare Q, five-seed mean G, ΔQ and ΔG together. Colour limits are fixed across
  checkpoints by default; a local-scale option reveals smaller differences.
- Follow the exact actor action, grid Q maximum, grid mean-G maximum, or click
  any grid action. The selected action persists across checkpoints; candidate
  selections follow their checkpoint-specific action.
- See all five seeded MC paths simultaneously, scrub their continuation time,
  and compare Q and realized discounted remaining G along those same paths.
  Each rollout's individual G, initial Q−G, length and ending are listed.
- Inspect actor-action Q and mean G across training. The band is ± one sample
  SD over five MC continuations, not a confidence interval across training seeds.

## Definitions and reproducibility

All checkpoints use the same reset seed and replayed reference prefix. At each
action, force that action once and then follow the current frozen checkpoint
actor. Five paired future engine-dispersion seeds (55000–55004) are reused for
every action and checkpoint. The 9 × 9 command grid and the exact actor action
are evaluated separately. Q uses the first online critic in native reward units.

ΔQ(s,a) = Q(s,a) − Q(s,π(s)). ΔG subtracts the exact actor-action return **within
each paired seed**, then averages. The displayed paired SD describes those
five differences; it is not the SD of the raw G samples. Delta does not mean
a difference from the previous training checkpoint.

Returns include the forced action's reward, use the checkpoint gamma and stop
at natural termination or the diagnostic horizon. Censored returns are labelled
finite observed sums, with no critic-filled tail. Along-path G is the discounted
remaining reward on that individual realized trajectory, not the five-seed mean
map value. Earlier policies need not visit the reference state. Changing the
continuation policy across checkpoints also changes what G measures.

Each `capture-STEP.npz` stores all states before actions, actions, rewards,
critic Q, realized suffix G, initial-action MC samples and rollout offsets.
The JSON sidecar stores seeds, final observations, endings and censoring.
The viewer alone samples paths and curves to at most 160 pre-action states
plus the terminal endpoint, keeping the exact diagnostic values in the tables.
`manifest.json` records checkpoint hashes, physical-state inputs, code hashes
and evaluation settings. Existing outputs can resume only with matching inputs;
use a fresh directory when adding checkpoints or changing evaluation settings.
To rebuild the viewer from completed captures without rerunning evaluations:

```bash
uv run --locked python -m lunarlander_rl.checkpoint_animation \
  --output generated/checkpoint-animation-state125-ln-on-all-20261014 --render-only
```

Rendering loads each compressed trajectory array once per checkpoint.
`render_provenance.json` records the renderer, template, HTML and capture-manifest
hashes separately, preserving the original measurement provenance after rendering
changes.

When source diagnostic measurements match the checkpoint hashes/grid/seeds,
regenerated MC samples must agree within 1e-9 before the viewer is written.

The full seed-21 Stage A series was recovered from the original Mac release:
49 checkpoints at zero, every 5k through 50k, and every 25k from 75k through 1M.
The archive SHA-256 matches `results/mac/stage-a-20261008/SHA256SUMS`; the four
previously used checkpoint files also match byte for byte. The extracted inputs
and archive provenance are under `outputs/checkpoint-animation-input-20261014/`.
The program discovers steps rather than hard-coding the four original figures.
The full evaluation contains 49 × 82 × 5 = 20,090 MC continuations. Generated
data stays under the Git-ignored `generated/` directory.
