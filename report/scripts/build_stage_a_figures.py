"""Generate report figures from the published Mac Stage A checkpoint metrics.

Run from the repository root:
    uv run --locked --with seaborn==0.13.2 python report/scripts/build_stage_a_figures.py

Inputs: results/mac/stage-a-20261008/{learning_metrics.csv,analysis/summary.json}.
Outputs: report/generated/stage-a-mac/, with PDF/150 dpi PNG figures, captions,
plotted statistics and provenance. No training or original analyses are modified.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/mac/stage-a-20261008"
OUT = ROOT / "report/generated/stage-a-mac"
SETTINGS = ((False, False), (False, True), (True, False), (True, True))
METRICS = (
    ("on_policy_return", "Episode reward", "return_mean"),
    ("on_policy_bias_q1", r"Bias ($Q_1-G$)", "bias_mean"),
    ("on_policy_rmse_q1", r"Critic RMSE ($Q_1$)", "rmse_mean"),
)
SEEDS = tuple(range(20, 30))
STEPS = np.array([0, *range(5000, 50001, 5000), *range(75000, 1000001, 25000)])
RESAMPLES = 10000
BOOTSTRAP_SEED = 419


def load():
    with (DATA / "learning_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((DATA / "analysis/summary.json").read_text())
    assert len(rows) == 80 * len(STEPS)
    assert summary["completed"] == summary["expected"] == 80 and not summary["partial"]
    assert all(r["platform"] == "mac" and r["stage"] == "A" for r in rows)
    groups = {}
    for algorithm in ("ddpg", "td3"):
        for actor_ln, critic_ln in SETTINGS:
            condition = f"{algorithm}-a{int(actor_ln)}-c{int(critic_ln)}"
            selected = [r for r in rows if r["condition"] == condition]
            assert {int(r["seed"]) for r in selected} == set(SEEDS)
            ordered = []
            for seed in SEEDS:
                run = sorted((r for r in selected if int(r["seed"]) == seed), key=lambda r: int(r["step"]))
                np.testing.assert_array_equal([int(r["step"]) for r in run], STEPS)
                assert all(r["algorithm"] == algorithm and
                           r["actor_layer_norm"] == str(actor_ln) and
                           r["critic_layer_norm"] == str(critic_ln) for r in run)
                ordered.append(run)
            groups[condition] = {}
            for metric, _, published in METRICS:
                values = np.array([[float(r[metric]) for r in run] for run in ordered])
                assert np.isfinite(values).all()
                np.testing.assert_allclose(values[:, -1].mean(), summary["conditions"][condition][published], rtol=1e-12)
                groups[condition][metric] = values
    return groups, summary


def caption(algorithm, summary):
    critic_symbol = "Q" if algorithm == "ddpg" else "Q1"
    tests = summary["primary_return_contrasts"]
    critic = tests[f"{algorithm}_critic_LN_actor_off"]["p_holm"]
    actor = tests[f"{algorithm}_actor_LN_critic_on"]["p_holm"]
    finding = (
        "Critic LayerNorm accompanies poorer DDPG reward and greater overestimation; actor LayerNorm does not restore performance."
        if algorithm == "ddpg" else
        "TD3 attains similar final rewards across LayerNorm settings, with slightly negative final signed bias."
    )
    return (
        f"{algorithm.upper()} on continuous LunarLander-v3, varying actor and critic LayerNorm (LN). "
        "Lines show means across ten training seeds (20–29); shaded bands are pointwise 95% percentile-bootstrap "
        "confidence intervals over training seeds (10,000 resamples). Each checkpoint uses 20 deterministic held-out episodes. "
        "Reward is undiscounted over at most 1,000 transitions; the dashed reward line marks the project's "
        f"200-reward success threshold. Signed bias is {critic_symbol} minus discounted Monte Carlo return "
        "(gamma=0.99); positive values mean overestimation. RMSE is averaged over episodes, then seeds. "
        "Monte Carlo continuations are observed to termination or 10,000 steps; capped trajectories remain included. "
        f"{finding} Final-reward paired sign-flip tests, Holm-adjusted across four primary contrasts: "
        f"critic LN on versus off with actor LN off, p={critic:.4f}; "
        f"actor LN on versus off with critic LN on, p={actor:.4f}. "
        "Bands do not establish checkpoint-wise significance or a causal effect of bias on reward."
    )


def main():
    groups, summary = load()
    OUT.mkdir(parents=True, exist_ok=True)
    # A wide canvas with adequate height leaves room for curves and reward ticks.
    # Scale text so it remains readable when included at 170 mm report width.
    sns.set_theme(context="paper", style="ticks", palette="colorblind", font_scale=1.5)
    colors = sns.color_palette("colorblind", 4)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(SEEDS), (RESAMPLES, len(SEEDS)))
    statistics, captions, artifacts = [], [], []
    for algorithm in ("ddpg", "td3"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), layout="constrained")
        for (actor_ln, critic_ln), color in zip(SETTINGS, colors):
            condition = f"{algorithm}-a{int(actor_ln)}-c{int(critic_ln)}"
            label = f"Actor LN {'on' if actor_ln else 'off'}, Critic LN {'on' if critic_ln else 'off'}"
            for ax, (metric, ylabel, _) in zip(axes, METRICS):
                values = groups[condition][metric]
                mean = values.mean(axis=0)
                low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975], axis=0)
                ax.plot(STEPS / 1e6, mean, color=color, label=label)
                ax.fill_between(STEPS / 1e6, low, high, color=color, alpha=0.15, linewidth=0)
                ax.set(xlabel=r"Training transitions ($10^6$)", ylabel=ylabel,
                       xlim=(0, 1), xticks=[0, 0.5, 1])
                if algorithm == "ddpg":
                    ax.set_ylabel(ylabel.replace("Q_1", "Q"))
                statistics.extend(dict(algorithm=algorithm, condition=condition, step=int(step), metric=metric,
                                       mean=float(m), ci95_low=float(lo), ci95_high=float(hi))
                                  for step, m, lo, hi in zip(STEPS, mean, low, high))
        axes[0].axhline(200, color="0.4", linestyle="--", linewidth=0.8, zorder=0)
        lower, upper = axes[0].get_ylim()
        if algorithm == "ddpg":
            axes[0].set(ylim=(lower, max(upper, 450)), yticks=[-1500, -1000, -500, 0, 200, 400])
        else:
            axes[0].set(ylim=(lower, max(upper, 350)), yticks=[-400, -200, 0, 100, 200, 300])
        axes[1].axhline(0, color="0.4", linestyle="--", linewidth=0.8, zorder=0)
        sns.despine(fig=fig)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
        fig.canvas.draw()
        # Check visible labels at the native report dimensions before exporting.
        renderer = fig.canvas.get_renderer()
        for artist in [legend, *(a for ax in axes for a in (ax.xaxis.label, ax.yaxis.label))]:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -0.01 and box.y0 >= -0.01 and box.x1 <= fig.bbox.width + 0.01 and box.y1 <= fig.bbox.height + 0.01, f"Clipped label: {artist}, {box.bounds}, {fig.bbox.bounds}"
        basename = f"stage-a-mac-{algorithm}"
        for extension in ("pdf", "png"):
            path = OUT / f"{basename}.{extension}"
            fig.savefig(path, dpi=150)
            artifacts.append(dict(path=path.name, bytes=path.stat().st_size))
        plt.close(fig)
        text = caption(algorithm, summary)
        captions.append(f"### {algorithm.upper()}\n\n{text}\n")
        # A fragment to paste into Samskrift after uploading the matching PDFs.
        latex = text.replace("%", r"\%").replace("Q1", "$Q_1$").replace("gamma=0.99", "$\\gamma=0.99$")
        (OUT / f"{basename}.tex").write_text(
            "\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{basename}.pdf}}\n"
            f"\\caption{{{latex}}}\n\\label{{fig:{basename}}}\n\\end{{figure}}\n"
        )
    with (OUT / "plotted_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    (OUT / "captions.md").write_text("\n".join(captions))
    provenance = dict(
        platform="mac", stage="A", completed_runs=80, seeds=list(SEEDS), checkpoints_per_run=len(STEPS),
        bootstrap_resamples=RESAMPLES, bootstrap_rng_seed=BOOTSTRAP_SEED,
        uncertainty="Pointwise 95% percentile-bootstrap CI over training seeds; no smoothing",
        theme=dict(context="paper", style="ticks", palette="colorblind", font_scale=1.5),
        figure_width_mm=304.8, figure_height_mm=124.46, intended_report_width_mm=170,
        reward_success_threshold=200,
        png_dpi=150, artifacts=artifacts,
        versions=dict(matplotlib=matplotlib.__version__, seaborn=sns.__version__, numpy=np.__version__),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [DATA / "learning_metrics.csv", DATA / "analysis/summary.json", Path(__file__).resolve()]},
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified 80 runs × {len(STEPS)} checkpoints; wrote {len(artifacts)} figure files to {OUT}")


if __name__ == "__main__":
    main()
