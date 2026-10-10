"""Match Figure 8's reward/bias style for the WSL target-update release.

Run from the repository root:
    uv run --locked --with seaborn==0.13.2 python report/scripts/build_target_update_figures.py

Inputs: results/wsl/target-updates-20261010 CSVs, frozen manifests and summary.
Outputs: PDF and 150 dpi PNG, plotted statistics, endpoint audit, LaTeX table
rows/caption and hashed provenance in report/generated/ddpg-target-updates/.
The plotted reference is Mac Stage A ddpg-a0-c1, as in Figures 8a/8b.
Release tests retain their separate historical WSL reference. Seeds are not pooled.
All experiment records are read only. Uses the existing Figure 8 drawing code.
"""

import csv
import hashlib
import json

import numpy as np
import seaborn as sns

import build_hyperparameter_figures as figures
from build_hyperparameter_comparison_table import indicator


ROOT = figures.ROOT
DATA = ROOT / "results/wsl/target-updates-20261010"
OUT = ROOT / "report/generated/ddpg-target-updates"
CONDITIONS = ("baseline-soft-tau-0.005", "soft-tau-0.001", "hard-k400")


def normalized(config):
    return dict(figures.normalized(config),
                hard_target_interval=config.get("hard_target_interval", 0))


def load():
    manifest = figures.read_json(DATA / "provenance/manifest.json")
    baseline_manifest = figures.read_json(DATA / "provenance/baseline_manifest.json")
    summary = figures.read_json(DATA / "analysis/summary.json")
    status = figures.read_json(DATA / "provenance/status.json")
    verification = figures.read_json(DATA / "verification.json")
    assert summary["completed"] == summary["expected"] == 20 and not summary["partial"]
    assert status["status"] == "complete" and status["total_jobs"] == 20
    assert all(job["status"] == "complete" for job in status["jobs"].values())
    assert verification["status"] == "passed" and verification["checkpoint_rows"] == 980
    assert manifest["baseline_platform"] == manifest["platform"] == "wsl"
    rows = figures.read_csv(DATA / "learning_metrics.csv")
    baseline_rows = [r for r in figures.read_csv(DATA / "baseline_learning_metrics.csv")
                     if r["stage"] == "A" and r["condition"] == "ddpg-a0-c1"]
    assert len(rows) == 980
    groups = []
    for condition in CONDITIONS:
        historical = condition == CONDITIONS[0]
        selected = baseline_rows if historical else [r for r in rows if r["condition"] == condition]
        jobs = ([j for j in baseline_manifest["jobs"] if j["stage"] == "A" and j["cell"] == "ddpg-a0-c1"]
                if historical else [j for j in manifest["jobs"] if j["cell"] == condition])
        assert all(r["platform"] == "wsl" and r["algorithm"] == "ddpg"
                   and r["actor_layer_norm"] == "False" and r["critic_layer_norm"] == "True"
                   for r in selected)
        group = figures.matrix_group(selected, [j["config"] for j in jobs], condition, "wsl", "A" if historical else "T")
        assert group["seeds"] == list(range(20, 30))
        for metric, _, published in figures.METRICS:
            np.testing.assert_allclose(group["values"][metric][:, -1].mean(), summary["conditions"][condition][published], rtol=1e-12)
        for config in group["configs"]:
            assert config["actor_lr"] == config["critic_lr"] == 0.0003
            assert config["ddpg_actor_delay"] == config["reward_scale"] == 1
            assert config["eval_episodes"] == 20 and config["gamma"] == 0.99
            expected_tau = 0.001 if condition == "soft-tau-0.001" else 0.005
            expected_interval = 400 if condition == "hard-k400" else 0
            assert config["tau"] == expected_tau
            assert config.get("hard_target_interval", 0) == expected_interval
            allowed = {"seed", "tau", "hard_target_interval"}
            assert {k: v for k, v in normalized(config).items() if k not in allowed} == {
                k: v for k, v in normalized(groups[0]["configs"][0] if groups else jobs[0]["config"]).items() if k not in allowed}
        if not historical:
            assert all(r["target_mode"] == ("hard" if condition == "hard-k400" else "soft")
                       and float(r["tau"]) == expected_tau and int(r["hard_target_interval"]) == expected_interval
                       for r in selected)
        groups.append(group)
    # Verify each endpoint against the separately exported seed and condition tables.
    seeds = figures.read_csv(DATA / "seed_summary.csv")
    for group in groups[1:]:
        records = sorted((r for r in seeds if r["condition"] == group["condition"]), key=lambda r: int(r["seed"]))
        assert len(records) == 10 and all(int(r["transitions"]) == 1000000 for r in records)
        for metric, column in (("on_policy_return", "final_return"), ("on_policy_bias_q1", "final_bias")):
            np.testing.assert_allclose(group["values"][metric][:, -1], [float(r[column]) for r in records], rtol=1e-12)
    for row in figures.read_csv(DATA / "conditions.csv"):
        assert int(row["n"]) == 10 and row["platform"] == "wsl"
        for key in ("return_mean", "bias_mean"):
            np.testing.assert_allclose(float(row[key]), summary["conditions"][row["condition"]][key], rtol=1e-12)
    wsl_baseline = groups[0]
    mac_baseline = figures.published_group(figures.MAC, "ddpg-a0-c1", "mac", "A")
    figures.check_settings(mac_baseline, mac_baseline, 0.0003, 0.0003, 1)
    assert mac_baseline["seeds"] == wsl_baseline["seeds"]
    for config in mac_baseline["configs"]:
        assert normalized(config) == normalized(next(c for c in wsl_baseline["configs"] if c["seed"] == config["seed"]))
    return [mac_baseline, *groups[1:]], summary, wsl_baseline


def main():
    groups, summary, wsl_baseline = load()
    OUT.mkdir(parents=True, exist_ok=True)
    figures.OUT = OUT
    sns.set_theme(context="paper", style="ticks", palette="colorblind", font_scale=1.5)
    statistics = []
    figures.draw("ddpg-target-updates", groups,
                 [r"Baseline: soft $\tau=0.005$", r"Soft $\tau=0.001$", "Hard copies every 400 critic updates"],
                 1, statistics)
    with (OUT / "plotted_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    reward0 = float(groups[0]["values"]["on_policy_return"][:, -1].mean())
    bias0 = float(groups[0]["values"]["on_policy_bias_q1"][:, -1].mean())
    audit = []
    latex = []
    for condition, label, change in [
        ("soft-tau-0.001", r"Soft target coefficient $\tau$", r"$0.005\rightarrow0.001$"),
        ("hard-k400", "Target update rule", r"Soft $\rightarrow$ hard ($K=400$)"),
    ]:
        reward = summary["conditions"][condition]["return_mean"]
        bias = summary["conditions"][condition]["bias_mean"]
        reward_improvement = reward - reward0
        bias_improvement = abs(bias0) - abs(bias)
        reward_percent = 100 * reward_improvement / abs(reward0)
        bias_percent = 100 * bias_improvement / abs(bias0)
        latex.append(" & ".join([label, change, rf"\textcolor{{black}}{{${reward:.2f}$}} " + indicator(reward_improvement),
                                 rf"\textcolor{{black}}{{${bias:+.2f}$}} " + indicator(bias_improvement)]) + r" \\")
        contrast = summary["exploratory_paired_return_contrasts"][condition]
        difference = groups[CONDITIONS.index(condition)]["values"]["on_policy_return"][:, -1] - wsl_baseline["values"]["on_policy_return"][:, -1]
        np.testing.assert_allclose(difference, contrast["differences"], rtol=1e-12)
        np.testing.assert_allclose(difference.mean(), contrast["mean"], rtol=1e-12)
        audit.append(dict(condition=condition, platform="wsl", baseline_condition="ddpg-a0-c1", baseline_platform="mac", baseline_reward=reward0,
                          baseline_bias=bias0, variant_reward=reward, variant_bias=bias,
                          reward_improvement=reward_improvement, bias_improvement=bias_improvement,
                          reward_improvement_percent=reward_percent, bias_improvement_percent=bias_percent,
                          release_test_baseline_platform="wsl", release_test_baseline_condition=CONDITIONS[0],
                          release_paired_return_difference=contrast["mean"], release_paired_ci95=contrast["ci95"], release_p_holm=contrast["p_holm"]))
    (OUT / "table-rows.tex").write_text("\n".join(latex) + "\n")
    (OUT / "endpoints.json").write_text(json.dumps(audit, indent=2) + "\n")
    caption = (r"Target-update schedules, actor LN off and critic LN on: soft baseline $\tau=0.005$, "
               r"slower soft updates $\tau=0.001$, and exact copies of both targets every $K=400$ critic updates "
               r"(no interpolation between copies). "
               r"Ten training seeds (20--29), 20 deterministic held-out episodes per checkpoint, "
               r"one million transitions. Lines are seed means; shading gives pointwise 95\% percentile-bootstrap intervals "
               r"(10,000 resamples). Reward is undiscounted within 1,000 transitions; the dashed line marks $R=200$. "
               r"Bias is $Q-G$ with discounted observed Monte Carlo return ($\gamma=0.99$, continuation capped at 10,000 steps). "
               r"The plotted comparison is descriptive; hard updates improve the final means while slower soft updates worsen them.")
    (OUT / "caption.tex").write_text(caption + "\n")
    scripts = [ROOT / "report/scripts/build_target_update_figures.py", ROOT / "report/scripts/build_hyperparameter_figures.py",
               ROOT / "report/scripts/build_stage_a_figures.py", ROOT / "report/scripts/build_hyperparameter_comparison_table.py"]
    figures.SOURCES.update(scripts)
    provenance = dict(release_url="https://github.com/glouno/miniproject1-DDPG/releases/tag/results-wsl-target-updates-20261010",
                      release_tag="results-wsl-target-updates-20261010", intervention_platform="wsl", new_runs=20,
                      plotted_baseline=dict(platform="mac", stage="A", condition="ddpg-a0-c1", reused_runs=10),
                      release_test_baseline=dict(platform="wsl", condition=CONDITIONS[0], plotted=False),
                      conditions=["ddpg-a0-c1", *CONDITIONS[1:]], seeds=list(range(20, 30)), checkpoints_per_run=len(figures.STEPS),
                      bootstrap_resamples=figures.RESAMPLES, bootstrap_rng_seed=figures.BOOTSTRAP_SEED, no_smoothing=True, png_dpi=150,
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(figures.SOURCES)})
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Validated 20 new WSL runs; plotted 10 Mac Stage A baseline runs; separately checked the release's WSL test baseline.")
    print("\n".join(latex))


if __name__ == "__main__":
    main()
