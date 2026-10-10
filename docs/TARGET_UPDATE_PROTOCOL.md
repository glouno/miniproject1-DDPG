# WSL DDPG target-update follow-up — 10 October 2026

Two exploratory interventions requested after Stage A/B, separately labelled WSL.
DDPG critic LN on, actor LN off; all numerical settings match the resolved Mac
Stage A baseline, except target schedule. Compare to the existing WSL Stage A
`ddpg-a0-c1` baseline, not pooled Mac/WSL seeds. No new baseline training.

| Condition | Target schedule for BOTH actor and critic |
|---|---|
| soft-tau-0.001 | Polyak update every critic round, tau=0.001 |
| hard-k400 | Exact online-to-target copy every 400 critic rounds; no updates in between |

Hard-copy interval is counted after warmup in completed critic optimizer rounds,
not environment steps, evaluation steps or wall time. Both targets initially
equal online models. Copy follows that round's critic and actor optimizer steps.
Persisted critic-round counter preserves schedule across resume. Tau is unused
in hard mode. Actor updates each round; reward scale=1; no further interventions.

K=400 approximately matches the average parameter-history age of the original
tau=.005 baseline: (1-tau)/tau=199 rounds versus (K-1)/2=199.5 for periodic copies.
This is an average-lag heuristic, not equivalence of the update dynamics. It does
not match tau=.001 (which would suggest K about 2000); only K=400 is scheduled.

Ten seeds 20–29 per condition, 1,000,000 transitions each; 20 new runs / 20M
transitions, CPU, eight workers, one Torch/BLAS thread per worker. Frozen native
WSL common probes must exactly match WSL Stage A. Preserve original runs.

Keep baseline settings: two 128-unit hidden layers, gamma=.99, actor/critic Adam
lr=3e-4, replay=100k, batch=128, warmup=5k, one critic round per transition,
Gaussian exploration sigma=.1, continuous LunarLander-v3, no wind, gravity=-10.
Twenty held-out reset seeds 40000–40019, diagnostic horizon=10k, 24 common probes.
Retain 49 model checkpoints: zero, every 5k through 50k, every 25k to 1M.
Rolling full recovery every 5k plus evaluation/graceful-stop checkpoints.

Primary endpoint final 1M return; full learning/calibration trajectories retained.
Use equal seed weighting, show seed SD and pointwise bootstrap bands, paired
exploratory return differences against historical WSL baseline with Holm correction
across the two contrasts. Historical reuse and post-Stage-A selection limit inference.
No reward-based early stopping or best checkpoint selection.

`python -m lunarlander_rl.target_runner --prepare --output outputs/target-updates-wsl-20261010`
freezes the queue without launch. Run the frozen package using a separate user
service. Successful completion automatically generates baseline-comparison learning
and common-probe calibration PNG/PDF figures. Exact report Figure 7 panel/layout
matching remains a report task because that live figure is not in this checkout.
