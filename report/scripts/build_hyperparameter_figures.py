"""Compare DDPG learning rates and critic:actor update ratios from saved data.

Run from the repository root:
    uv run --locked --with seaborn==0.13.2 python report/scripts/build_hyperparameter_figures.py

Inputs: published Mac Stage A and WSL Stage B CSVs/manifests, and completed
outputs/ddpg-focused-20261009-m1air journals. Outputs: two PDF/150 dpi PNG figures,
captions, plotted statistics, endpoint summaries and provenance in
report/generated/ddpg-hyperparameters/. Original experiment files are read only.
"""

import csv
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from build_stage_a_figures import ROOT, STEPS, METRICS, RESAMPLES, BOOTSTRAP_SEED


MAC = ROOT / "results/mac/stage-a-20261008"
WSL = ROOT / "results/wsl/stages-a-b-20261008"
FOCUSED = ROOT / "outputs/ddpg-focused-20261009-m1air"
OUT = ROOT / "report/generated/ddpg-hyperparameters"
SOURCES = set()


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


def read_csv(path):
    SOURCES.add(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(config):
    return dict(config, reward_scale=config.get("reward_scale", 1),
                ddpg_actor_delay=config.get("ddpg_actor_delay", 1))


def matrix_group(rows, configs, condition, platform, stage):
    seeds = sorted({int(r["seed"]) for r in rows})
    assert len(seeds) == 10 and len(rows) == 10 * len(STEPS)
    assert seeds == sorted(c["seed"] for c in configs)
    assert all(c["algorithm"] == "ddpg" and not c["actor_layer_norm"] and
               c["critic_layer_norm"] and c["max_steps"] == 1000000 for c in configs)
    runs = []
    for seed in seeds:
        run = sorted((r for r in rows if int(r["seed"]) == seed), key=lambda r: int(r["step"]))
        np.testing.assert_array_equal([int(r["step"]) for r in run], STEPS)
        runs.append(run)
    values = {metric: np.array([[float(r[metric]) for r in run] for run in runs])
              for metric, _, _ in METRICS}
    assert all(np.isfinite(v).all() for v in values.values())
    return dict(condition=condition, platform=platform, stage=stage, seeds=seeds,
                configs=[normalized(c) for c in configs], values=values)


def published_group(root, condition, platform, stage):
    rows = [r for r in read_csv(root / "learning_metrics.csv") if r["condition"] == condition]
    assert all(r["platform"] == platform and r["stage"] == stage and r["algorithm"] == "ddpg"
               and r["actor_layer_norm"] == "False" and r["critic_layer_norm"] == "True" for r in rows)
    jobs = [j for j in read_json(root / "provenance/manifest.json")["jobs"]
            if j["cell"] == condition and j["stage"] == stage]
    group = matrix_group(rows, [j["config"] for j in jobs], condition, platform, stage)
    summary_path = root / ("analysis/summary.json" if platform == "mac" else f"analysis/{stage}/summary.json")
    summary = read_json(summary_path)["conditions"][condition]
    for metric, _, published in METRICS:
        np.testing.assert_allclose(group["values"][metric][:, -1].mean(), summary[published], rtol=1e-12)
    return group


def focused_group(condition):
    manifest = read_json(FOCUSED / "manifest.json")
    assert read_json(FOCUSED / "status.json")["status"] == "complete"
    jobs = [j for j in manifest["jobs"] if j["cell"] == condition]
    rows = []
    endpoints = read_csv(FOCUSED / "seed_summary.csv")
    for job in jobs:
        directory = FOCUSED / job["directory"]
        summary = read_json(directory / "summary.json")
        assert summary["status"] == "complete" and summary["transitions"] == 1000000
        path = directory / "metrics.jsonl"
        SOURCES.add(path)
        metrics = [json.loads(line) for line in path.read_text().splitlines()]
        ratio = job["config"]["ddpg_actor_delay"]
        for metric in metrics:
            assert metric["algorithm"] == "ddpg" and not metric["actor_layer_norm"] and metric["critic_layer_norm"]
            assert abs(metric["critic_rounds"] - ratio * metric["actor_updates"]) <= ratio
            rows.append(dict(seed=metric["seed"], step=metric["step"],
                             **{key: metric["on_policy"][key.removeprefix("on_policy_")] for key, _, _ in METRICS}))
        endpoint = next(r for r in endpoints if r["condition"] == condition and int(r["seed"]) == job["config"]["seed"])
        np.testing.assert_allclose(metrics[-1]["on_policy"]["return"], float(endpoint["return_raw"]), rtol=1e-12)
    return matrix_group(rows, [j["config"] for j in jobs], condition, "mac", "focused")


def check_settings(group, baseline, critic_lr, actor_lr, ratio):
    reference = baseline["configs"][0]
    allowed = {"seed", "actor_lr", "critic_lr", "ddpg_actor_delay"}
    for config in group["configs"]:
        assert config["critic_lr"] == critic_lr and config["actor_lr"] == actor_lr
        assert config["ddpg_actor_delay"] == ratio and config["reward_scale"] == 1
        assert {k: v for k, v in config.items() if k not in allowed} == {
            k: v for k, v in reference.items() if k not in allowed
        }


def draw(name, groups, labels, ncols, statistics):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), layout="constrained")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 10, (RESAMPLES, 10))
    for group, label, color in zip(groups, labels, sns.color_palette("colorblind", len(groups))):
        for ax, (metric, ylabel, _) in zip(axes, METRICS):
            values = group["values"][metric]
            mean = values.mean(axis=0)
            low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975], axis=0)
            ax.plot(STEPS / 1e6, mean, color=color, label=label)
            ax.fill_between(STEPS / 1e6, low, high, color=color, alpha=0.15, linewidth=0)
            ax.set(xlabel=r"Training transitions ($10^6$)", ylabel=ylabel.replace("Q_1", "Q"),
                   xlim=(0, 1), xticks=[0, 0.5, 1])
            statistics.extend(dict(figure=name, condition=group["condition"], platform=group["platform"],
                                   stage=group["stage"], step=int(step), metric=metric,
                                   mean=float(m), ci95_low=float(lo), ci95_high=float(hi))
                              for step, m, lo, hi in zip(STEPS, mean, low, high))
    axes[0].axhline(200, color="0.4", linestyle="--", linewidth=0.8, zorder=0)
    lower, upper = axes[0].get_ylim()
    axes[0].set(ylim=(lower, max(upper, 450)), yticks=[-1500, -1000, -500, 0, 200, 400])
    axes[1].axhline(0, color="0.4", linestyle="--", linewidth=0.8, zorder=0)
    sns.despine(fig=fig)
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=ncols, frameon=False)
    fig.canvas.draw()
    for artist in [legend, *(a for ax in axes for a in (ax.xaxis.label, ax.yaxis.label))]:
        box = artist.get_window_extent(fig.canvas.get_renderer())
        assert box.x0 >= -0.01 and box.y0 >= -0.01 and box.x1 <= fig.bbox.width + 0.01 and box.y1 <= fig.bbox.height + 0.01
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=150)
    plt.close(fig)


def main():
    baseline = published_group(MAC, "ddpg-a0-c1", "mac", "A")
    check_settings(baseline, baseline, 0.0003, 0.0003, 1)
    rates = [(0.0003, 0.0001), (0.0001, 0.0003), (0.0001, 0.0001)]
    learning = [baseline]
    for critic, actor in rates:
        condition = f"ddpg-a0-c1-clr{critic}-alr{actor}"
        group = published_group(WSL, condition, "wsl", "B")
        check_settings(group, baseline, critic, actor, 1)
        assert group["seeds"] == list(range(30, 40))
        learning.append(group)
    updates = [baseline, focused_group("actor-delay-2"), focused_group("actor-delay-3")]
    for ratio, group in enumerate(updates, 1):
        check_settings(group, baseline, 0.0003, 0.0003, ratio)
        assert group["seeds"] == baseline["seeds"] == list(range(20, 30))
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="ticks", palette="colorblind", font_scale=1.5)
    statistics = []
    rate_labels = [r"Baseline: $\eta_c=0.0003$, $\eta_a=0.0003$"] + [
        rf"$\eta_c={critic}$, $\eta_a={actor}$" for critic, actor in rates]
    ratio_labels = ["Critic:actor = 1:1 (baseline)", "Critic:actor = 2:1", "Critic:actor = 3:1"]
    draw("ddpg-learning-rates", learning, rate_labels, 2, statistics)
    draw("ddpg-update-ratios", updates, ratio_labels, 3, statistics)
    common = (
        "DDPG with actor LayerNorm off and critic LayerNorm on. Lines are means across ten training seeds per setting; "
        "shading shows pointwise 95% percentile-bootstrap confidence intervals over seeds (10,000 resamples). "
        "Every checkpoint evaluates 20 deterministic held-out episodes. Reward is undiscounted over at most 1,000 "
        "transitions; the dashed reward line marks the 200-reward success criterion. Bias is Q minus discounted "
        "Monte Carlo return G (gamma=0.99), with positive values indicating overestimation. RMSE is averaged over "
        "episodes, then seeds. Observed Monte Carlo continuations are capped at 10,000 steps. "
    )
    tests = read_json(WSL / "analysis/B/summary.json")["primary_return_contrasts"]
    captions = {
        "ddpg-learning-rates": common +
        "The Mac Stage A baseline uses critic and actor learning rates of 0.0003 and seeds 20–29; "
        "the three WSL Stage B variants use seeds 30–39 and the rates shown in the legend (eta_c: critic, eta_a: actor). "
        "The comparison with the Mac baseline also changes platform and seed cohort. Reported WSL-only final-reward "
        "factorial tests use the full four-cell WSL design, including its WSL baseline not plotted here: Holm-adjusted "
        f"p={tests['critic_lr_lower']['p_holm']:.4f} for lower critic LR, "
        f"p={tests['actor_lr_lower']['p_holm']:.4f} for lower actor LR, "
        f"p={tests['interaction']['p_holm']:.4f} for their interaction. These are not tests against the Mac baseline.",
        "ddpg-update-ratios": common +
        "The reused Mac Stage A baseline performs one critic update per actor update (1:1); the Mac focused "
        "variants delay the actor to obtain 2:1 and 3:1. All settings use learning rates of 0.0003, reward scale 1, "
        "and seeds 20–29. Both target networks are updated every critic round. This is a descriptive comparison; "
        "no new significance tests are performed.",
    }
    (OUT / "captions.md").write_text("\n\n".join(f"### {name}\n\n{text}" for name, text in captions.items()) + "\n")
    for name, text in captions.items():
        latex = (text.replace("%", r"\%").replace("gamma=0.99", r"$\gamma=0.99$")
                 .replace("eta_c", r"$\eta_c$").replace("eta_a", r"$\eta_a$"))
        (OUT / f"{name}.tex").write_text(
            "\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{name}.pdf}}\n\\caption{{{latex}}}\n"
            f"\\label{{fig:{name}}}\n\\end{{figure}}\n")
    with (OUT / "plotted_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    endpoints = [{**{k: v for k, v in r.items() if k not in {"step", "figure"}}}
                 for r in statistics if r["step"] == 1000000]
    (OUT / "endpoints.json").write_text(json.dumps(endpoints, indent=2) + "\n")
    SOURCES.add(ROOT / "report/scripts/build_hyperparameter_figures.py")
    SOURCES.add(ROOT / "report/scripts/build_stage_a_figures.py")
    provenance = dict(bootstrap_resamples=RESAMPLES, bootstrap_rng_seed=BOOTSTRAP_SEED,
                      checkpoints_per_run=len(STEPS), no_smoothing=True, png_dpi=150,
                      baseline=dict(condition=baseline["condition"], platform="mac", stage="A", seeds=baseline["seeds"]),
                      learning_rate_conditions=[dict(condition=g["condition"], platform=g["platform"], seeds=g["seeds"]) for g in learning],
                      update_ratio_conditions=[dict(condition=g["condition"], platform=g["platform"], seeds=g["seeds"]) for g in updates],
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)})
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Validated 60 distinct runs, 49 checkpoints each; generated two figures in {OUT}")


if __name__ == "__main__":
    main()
