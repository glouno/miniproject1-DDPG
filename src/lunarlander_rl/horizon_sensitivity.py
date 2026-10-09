"""Paired episode-level sensitivity of saved Q bias to the diagnostic cutoff.

Reuses full saved trajectories so only the return cutoff changes. A horizon is
an episode cutoff, as in evaluate_study, not a new rollout from each sampled state.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

from .experiment import discounted_returns, load_checkpoint


def compare(data, gamma, short=1000, long=10000):
    if not 0 < short < long:
        raise ValueError("Require 0 < short < long")
    q = np.asarray(data["q"])[:, 0]
    episodes, pair_deltas, pair_times = [], [], []
    offsets = data["offsets"]
    if offsets[0] != 0 or offsets[-1] != len(data["rewards"]):
        raise ValueError("Reward offsets must cover the entire saved reward array")
    for index, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        rewards = data["rewards"][lo:hi]
        mask = data["episode"] == index
        times = data["t"][mask].astype(int)
        if not len(times) or np.any(times < 0) or np.any(times >= min(short, len(rewards))):
            raise ValueError("Every sampled pair must precede the short cutoff")
        if len(rewards) > long:
            raise ValueError("Saved trajectory exceeds the specified long cutoff")
        g_short = discounted_returns(rewards[:short], gamma)[times]
        g_long = discounted_returns(rewards[:long], gamma)[times]
        if f"mc{long}" in data:
            np.testing.assert_allclose(g_long, data[f"mc{long}"][mask], rtol=0, atol=1e-10)
        # Delta is calculated directly from returns: the identical Q cancels.
        delta = g_short - g_long
        # Independent direct discounted sum checks a sampled early and late state.
        for position in {0, len(times) - 1}:
            suffix = rewards[times[position]:long]
            direct = np.dot(suffix, gamma ** np.arange(len(suffix)))
            np.testing.assert_allclose(direct, g_long[position], rtol=0, atol=1e-10)
        # Supporting check: a full short-step window after EACH state is a
        # different estimand from cutting the whole episode at the short horizon.
        per_state_short = np.array([
            np.dot(rewards[t:t + short], gamma ** np.arange(len(rewards[t:t + short])))
            for t in times
        ])
        episodes.append(dict(index=index, length=len(rewards), sampled_pairs=len(times),
            bias_short=float((q[mask] - g_short).mean()),
            bias_long=float((q[mask] - g_long).mean()),
            delta=float(delta.mean()), max_abs_pair_delta=float(np.abs(delta).max()),
            per_state_short_delta=float((per_state_short - g_long).mean())))
        pair_deltas.extend(delta)
        pair_times.extend(times)
    return episodes, np.asarray(pair_times), np.asarray(pair_deltas)


def paired_statistics(deltas, resamples=10000, seed=419):
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 2:
        raise ValueError("At least two independent units are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), (resamples, len(deltas)))
    boot = deltas[indices].mean(axis=1)
    nonzero = deltas[deltas != 0]
    sign_p = stats.binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) else 1.0
    sd = deltas.std(ddof=1)
    if sd == 0:
        t_p = 1.0 if deltas.mean() == 0 else 0.0
        t_ci = [float(deltas.mean())] * 2
    else:
        t_p = float(stats.ttest_1samp(deltas, 0).pvalue)
        margin = stats.t.ppf(.975, len(deltas) - 1) * sd / np.sqrt(len(deltas))
        t_ci = [float(deltas.mean() - margin), float(deltas.mean() + margin)]
    return dict(mean_delta=float(deltas.mean()),
        bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
        paired_t_ci95=t_ci, paired_t_p=t_p, exact_two_sided_sign_p=float(sign_p),
        positive_episodes=int((deltas > 0).sum()), negative_episodes=int((deltas < 0).sum()),
        unchanged_episodes=int((deltas == 0).sum()), bootstrap_resamples=resamples,
        bootstrap_seed=seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_run(run, step=1000000, short=1000, long=10000):
    """Validate a final checkpoint and matched returns without writing artifacts."""
    run = Path(run).resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    cfg = metadata["config"]
    metric = next(json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
                  if json.loads(line)["step"] == step)
    path = run / f"diagnostics/on_policy-{step:07d}.npz"
    torch.set_num_threads(1)
    checkpoint_cfg, _, critics, checkpoint_step = load_checkpoint(run / f"checkpoint-{step:07d}.pt")
    assert checkpoint_step == step and checkpoint_cfg.gamma == cfg["gamma"]
    with np.load(path) as data:
        episodes, times, pair_deltas = compare(data, cfg["gamma"], short, long)
        # Use the original one-pair inference shape to avoid batched float32
        # rounding differences. Predictions were stored in native reward units.
        with torch.inference_mode():
            predictions = np.array([
                critics[0](torch.from_numpy(obs).unsqueeze(0),
                           torch.from_numpy(action).unsqueeze(0)).item() / checkpoint_cfg.reward_scale
                for obs, action in zip(data["obs"], data["action"], strict=True)
            ])
        np.testing.assert_allclose(predictions, data["q"][:, 0], rtol=0, atol=1e-10)
        checkpoint_q_max_difference = float(np.max(np.abs(predictions - data["q"][:, 0])))
    for episode, saved in zip(episodes, metric["on_policy"]["episodes"], strict=True):
        assert episode["index"] == saved["index"] and episode["length"] == saved["length"]
        np.testing.assert_allclose(episode["bias_long"], saved["bias_q1"], rtol=0, atol=1e-10)
        episode.update(seed=saved["seed"], censored_long=saved["censored"], ending=saved.get("ending"))
    np.testing.assert_allclose(np.mean([e["bias_long"] for e in episodes]),
                               metric["on_policy"]["bias_q1"], rtol=0, atol=1e-10)
    result = dict(run=str(run), checkpoint_step=step, config=cfg,
        short_horizon=short, long_horizon=long, evaluation_episodes=len(episodes),
        sampled_pairs=sum(e["sampled_pairs"] for e in episodes),
        checkpoint_q_max_difference=checkpoint_q_max_difference,
        bias_short=float(np.mean([e["bias_short"] for e in episodes])),
        bias_long=float(np.mean([e["bias_long"] for e in episodes])),
        episodes_past_short=sum(e["length"] > short for e in episodes),
        censored_long=sum(e["censored_long"] for e in episodes),
        statistics=paired_statistics([e["delta"] for e in episodes]),
        max_abs_pair_delta=float(np.abs(pair_deltas).max()),
        per_state_short_mean_delta=float(np.mean([e["per_state_short_delta"] for e in episodes])),
        source_sha256={name: sha256(run / name) for name in (
            "metadata.json", "metrics.jsonl", f"diagnostics/on_policy-{step:07d}.npz",
            f"checkpoint-{step:07d}.pt")},
        inference_unit="Evaluation reset seed, conditional on this one frozen trained policy",
        aggregation="Mean within episode, then equal mean across episodes",
        delta_definition="(Q - G_long) - (Q - G_short) = G_short - G_long",
        episodes=episodes)
    for key in ("bias_q2", "bias_min"):
        saved = metric["on_policy"].get(key)
        if saved is not None:
            result[f"{key}_long"] = saved
            result[f"{key}_short"] = saved - result["statistics"]["mean_delta"]
    return result, times, pair_deltas


def analyse(run, output, step=1000000, short=1000, long=10000):
    result, times, pair_deltas = read_run(run, step, short, long)
    output = Path(output).resolve()
    episodes = result["episodes"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "episodes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader(); writer.writerows(episodes)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    colors = ["#b14c30" if e["length"] > short else "#6f7b85" for e in episodes]
    seed_start = min(e["seed"] for e in episodes)
    axes[0].bar([e["seed"] - seed_start for e in episodes], [e["delta"] for e in episodes], color=colors)
    axes[0].set(xlabel=f"Evaluation reset seed − {seed_start}", ylabel="Bias increase (10,000 − 1,000)",
                title="Paired episode means")
    axes[0].set_xticks(range(0, len(episodes), 5))
    axes[1].scatter(times, pair_deltas, s=6, alpha=.25, color="#b14c30")
    axes[1].set(xlabel="Sampled state timestep", ylabel="Bias increase per state/action pair",
                title="Sensitivity rises near the episode cutoff")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=.7)
        ax.grid(axis="y", alpha=.15)
    fig.savefig(output / "horizon_sensitivity.png", dpi=180)
    fig.savefig(output / "horizon_sensitivity.pdf")
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "config", "source_sha256")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=1000000)
    parser.add_argument("--short", type=int, default=1000)
    parser.add_argument("--long", type=int, default=10000)
    args = parser.parse_args()
    analyse(args.run, args.output, args.step, args.short, args.long)


if __name__ == "__main__":
    main()
