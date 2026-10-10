"""Audit transient DDPG reward drops using saved evaluation episode journals.

Run from the repository root:
    uv run --locked python report/scripts/analyse_reward_drops.py

Inputs: published Mac Stage A learning_metrics.csv and the previously extracted
40 DDPG metrics.jsonl files under outputs/horizon-sensitivity-input-20261009/.
Outputs: report/generated/reward-drops/{summary.json,episodes.csv,README.md},
plus separate reward-drop-0500000 and reward-drop-0675000 PDF/PNG figures.
No training or remote report is modified.
The -1000 cutoff is a descriptive screen, not a prespecified statistical test.
"""

import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "outputs/horizon-sensitivity-input-20261009/mac-stage-a-20261008/A"
PUBLISHED = ROOT / "results/mac/stage-a-20261008/learning_metrics.csv"
OUT = ROOT / "report/generated/reward-drops"
CONDITIONS = [f"ddpg-a{a}-c{c}" for a in (0, 1) for c in (0, 1)]
STEPS = [0, *range(5000, 50001, 5000), *range(75000, 1000001, 25000)]
SCREEN = -1000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    published = {(r["condition"], int(r["seed"]), int(r["step"])): r
                 for r in csv.DictReader(PUBLISHED.open()) if r["algorithm"] == "ddpg"}
    journals, episode_rows, sources = {}, [], [PUBLISHED, Path(__file__).resolve()]
    for condition in CONDITIONS:
        for seed in range(20, 30):
            path = RAW / f"{condition}-seed{seed}" / "metrics.jsonl"
            sources.append(path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r["step"] for r in records] == STEPS
            for m in records:
                key = condition, seed, m["step"]
                op = m["on_policy"]
                assert [e["seed"] for e in op["episodes"]] == list(range(40000, 40020))
                assert m["seed"] == seed and m["algorithm"] == "ddpg"
                np.testing.assert_allclose(statistics.mean(e["performance_return"] for e in op["episodes"]), op["return"], rtol=1e-12)
                for journal_key, column in [("return", "on_policy_return"), ("bias_q1", "on_policy_bias_q1")]:
                    np.testing.assert_allclose(op[journal_key], float(published[key][column]), rtol=1e-12)
                journals[key] = op
                for e in op["episodes"]:
                    episode_rows.append(dict(condition=condition, training_seed=seed, step=m["step"],
                                             evaluation_seed=e["seed"], reward=e["performance_return"],
                                             length=e["performance_length"], diagnostic_length=e["length"],
                                             diagnostic_ending=e["ending"], bias=e["bias_q1"]))
    assert len(journals) == 40 * 49 and len(episode_rows) == 39200

    def mean(condition, step, omit=None):
        return statistics.mean(journals[condition, seed, step]["return"]
                               for seed in range(20, 30) if seed != omit)

    # Exactly two seed/checkpoint means cross this descriptive screen.
    selected = sorted([key for key, op in journals.items() if op["return"] < SCREEN], key=lambda key: key[2])
    assert selected == [("ddpg-a1-c0", 26, 500000), ("ddpg-a0-c1", 28, 675000)]
    events = []
    for condition, seed, step in selected:
        previous, following = step - 25000, step + 25000
        event = dict(condition=condition, training_seed=seed, step=step, checkpoints=[])
        for checkpoint in [previous, step, following]:
            op = journals[condition, seed, checkpoint]
            ep = op["episodes"]
            all_ep = [e for s in range(20, 30) for e in journals[condition, s, checkpoint]["episodes"]]
            event["checkpoints"].append(dict(
                step=checkpoint, seed_reward=op["return"], condition_reward=mean(condition, checkpoint),
                other_nine_reward=mean(condition, checkpoint, omit=seed),
                seed_bias=op["bias_q1"], seed_mean_q=op["mean_q1"], seed_mean_mc=op["mean_mc"],
                severe_episodes=sum(e["performance_return"] < SCREEN for e in ep),
                severe_episodes_all_seeds=sum(e["performance_return"] < SCREEN for e in all_ep),
                median_episode_reward=statistics.median(e["performance_return"] for e in ep),
                reward_min=min(e["performance_return"] for e in ep), reward_max=max(e["performance_return"] for e in ep),
                diagnostic_endings=dict(collections.Counter(e["ending"] for e in ep)),
                performance_length_min=min(e["performance_length"] for e in ep),
                performance_length_max=max(e["performance_length"] for e in ep),
            ))
        event["condition_drop"] = mean(condition, step) - mean(condition, previous)
        event["culprit_contribution_to_drop"] = (journals[condition, seed, step]["return"] - journals[condition, seed, previous]["return"]) / 10
        events.append(event)
    severe = [r for r in episode_rows if r["reward"] < SCREEN]
    summary = dict(description="Exploratory audit of evaluation returns, not training-episode returns",
                   conditions=CONDITIONS, training_seeds=list(range(20, 30)), evaluation_seeds=list(range(40000, 40020)),
                   checkpoints=len(STEPS), total_episodes=len(episode_rows), severe_cutoff=SCREEN,
                   severe_episodes=len(severe), severe_episodes_at_or_after_10000=sum(r["step"] >= 10000 for r in severe),
                   episodes_at_or_after_10000=sum(r["step"] >= 10000 for r in episode_rows),
                   severe_groups=[dict(condition=c, training_seed=s, step=t, episodes=n)
                                  for (c, s, t), n in collections.Counter((r["condition"], r["training_seed"], r["step"]) for r in severe).items()],
                   events=events, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                   limitation="Intermediate trajectory diagnostics, model weights and optimization journals for the two collapses are not extracted locally. Episode records establish widespread failure within one policy at each checkpoint, but do not identify the physical behavior or causal optimizer mechanism. Diagnostic endings can occur after the 1000-step performance cutoff.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
        writer.writeheader(); writer.writerows(episode_rows)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    for event in events:
        fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 2.8), layout="constrained")
        c, seed, step = event["condition"], event["training_seed"], event["step"]
        window = [s for s in STEPS if step-50000 <= s <= step+50000]
        for s in range(20, 30):
            left.plot(np.array(window)/1000, [journals[c, s, t]["return"] for t in window],
                      color="#8c2d91" if s == seed else "0.7", linewidth=2 if s == seed else 0.8,
                      label=f"Training seed {seed}" if s == seed else None)
        left.plot(np.array(window)/1000, [mean(c, t) for t in window], color="black", linestyle="--", label="Mean of all 10 seeds")
        left.axvline(step/1000, color="0.5", linestyle=":", linewidth=0.8)
        left.set(title=f"Drop at {step:,} transitions", xlabel=r"Training transitions ($10^3$)", ylabel="Mean evaluation reward\n(20 episodes)")
        left.legend(frameon=False, fontsize=8, loc="lower left")
        for checkpoint, color, label in [(step-25000,"#0072b2","Before"),(step,"#8c2d91","At the drop"),(step+25000,"#009e73","After")]:
            right.plot(range(20), [e["performance_return"] for e in journals[c, seed, checkpoint]["episodes"]],
                       marker="o", markersize=3, linewidth=1, color=color, label=f"{label}: {checkpoint:,}")
        right.set(title=f"Training seed {seed}: all 20 evaluation episodes", xlabel="Evaluation reset seed", ylabel="Individual episode reward",
                  xticks=[0,5,10,15,19], xticklabels=["40000","40005","40010","40015","40019"])
        low, high = right.get_ylim()
        right.set_ylim(low - 0.16 * (high - low), high)
        right.legend(frameon=False, fontsize=8, loc="lower center", ncols=3, columnspacing=0.7, handlelength=1.5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in (left, right):
            for artist in (ax.title, ax.xaxis.label, ax.yaxis.label, ax.get_legend()):
                box = artist.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, "Clipped figure label"
        basename = f"reward-drop-{step:07d}"
        fig.savefig(OUT / f"{basename}.png", dpi=160)
        fig.savefig(OUT / f"{basename}.pdf")
        plt.close(fig)

    lines = ["# Transient DDPG evaluation collapses", "",
             "Audited all 40 Mac Stage A DDPG runs: 49 checkpoints and 20 fixed evaluation episodes per checkpoint (39,200 episodes). Every episode mean and bias agrees with the published checkpoint table. These are evaluation episodes; training episode rewards do not enter these curves.", "",
             "The two largest isolated collapses are driven by one training seed each, but involve many episodes of that policy. A return below -1,000 is used only as a descriptive screen.", "",
             "| Configuration | Checkpoint | Training seed | Before / at drop / after (policy means) | Episodes below -1,000 at drop | Other nine policies at drop |",
             "|---|---:|---:|---|---:|---:|"]
    for event in events:
        before, at, after = event["checkpoints"]
        lines.append(f"| {event['condition']} | {event['step']:,} | {event['training_seed']} | {before['seed_reward']:.1f} / {at['seed_reward']:.1f} / {after['seed_reward']:.1f} | {at['severe_episodes']}/20 | {at['other_nine_reward']:.1f} |")
    lines += ["", "`a` denotes actor LayerNorm; `c` denotes critic LayerNorm (0 off, 1 on). Before and after are the neighboring checkpoints, 25,000 training transitions apart.", ""]
    for event in events:
        before, at, after = event["checkpoints"]
        lines += [f"## {event['condition']}, seed {event['training_seed']}, step {event['step']:,}", "",
                  f"All-seed mean changes from {before['condition_reward']:.2f} to {at['condition_reward']:.2f}. The affected seed contributes {event['culprit_contribution_to_drop']:.2f} to this change; the total change is {event['condition_drop']:.2f}. Excluding it, the other nine seeds change from {before['other_nine_reward']:.2f} to {at['other_nine_reward']:.2f}.", "",
                  f"Episode rewards range from {at['reward_min']:.2f} to {at['reward_max']:.2f} (median {at['median_episode_reward']:.2f}). Saved diagnostic endings: {at['diagnostic_endings']}. Performance lengths: {at['performance_length_min']}–{at['performance_length_max']} transitions. Diagnostic endings describe the longer rollout and need not occur within the 1,000-transition reward budget.", "",
                  f"Signed bias rises from {before['seed_bias']:.2f} to {at['seed_bias']:.2f}, then changes to {after['seed_bias']:.2f}. At the drop, mean Q is {at['seed_mean_q']:.2f} and mean observed discounted suffix return is {at['seed_mean_mc']:.2f}. These quantities average sampled states within episodes and then episodes; they are not the undiscounted episode rewards.", ""]
    lines += ["## Scope and limits", "",
              f"Across all checkpoints, {len(severe)} of 39,200 individual evaluation episodes score below -1,000; {summary['severe_episodes_at_or_after_10000']} of {summary['episodes_at_or_after_10000']:,} do so at or after 10,000 training transitions. The two headline collapses account for 36 such episodes. Other isolated episodes and broader, smaller drops exist; the headline events should not be generalized to every drop.", "",
              "The same reset seeds and noise-free, frozen policies are used at the neighboring checkpoints, so the collapse reflects a change in the learned policy's behavior, rather than the evaluation drawing an unlucky new batch of seeds. Both policies recover by the next recorded checkpoint. This does not prove what caused the actor change, nor that critic overestimation caused it.", "",
              summary['limitation'], "",
              "The release is accessible in the signed-in browser, but attempts to download the intermediate archive timed out. GitHub CLI is not authenticated. No raw trajectories were fabricated or inferred from episode summaries.", "",
              "In the comparison plot, grey curves show individual training-seed episode means, purple highlights the affected seed, and black dashed curves show the ten-seed mean. The right panels show each of the 20 individual evaluation episodes before, during and after the drop; points are not averages.", "",
              "![Actor LN on, critic LN off: drop at 500,000](reward-drop-0500000.png)", "",
              "![Actor LN off, critic LN on: drop at 675,000](reward-drop-0675000.png)", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print(json.dumps({"validated_runs":40,"validated_checkpoints":1960,"episodes":39200,"severe_episodes":len(severe),"events":events}, indent=2))


if __name__ == "__main__":
    main()
