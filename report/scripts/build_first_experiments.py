"""Build concise setup, metric definitions and Mac Stage A tables for Samskrift.

Run from the repository root: python3 report/scripts/build_first_experiments.py
Reads the published configuration, checkpoint metrics and condition summary;
verifies final seed aggregation and writes report/generated/first_experiments.tex.
"""
import csv
import json
import math
from pathlib import Path
import statistics

from build_results_tables import table

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/mac/stage-a-20261008"


def main():
    manifest = json.loads((DATA / "provenance/manifest.json").read_text())
    jobs = [job for job in manifest["jobs"] if job["stage"] == "A"]
    assert len(jobs) == 80
    cfg = jobs[0]["config"]
    varying = {"algorithm", "layer_norm", "actor_layer_norm", "critic_layer_norm", "seed"}
    for job in jobs:
        assert {k: v for k, v in job["config"].items() if k not in varying} == {
            k: v for k, v in cfg.items() if k not in varying
        }
    expected = dict(env_id="LunarLander-v3", continuous=True, gravity=-10.0,
                    enable_wind=False, train_horizon=1000, actor_hidden=[128, 128],
                    critic_hidden=[128, 128], gamma=0.99, tau=0.005,
                    actor_lr=0.0003, critic_lr=0.0003, batch_size=128,
                    replay_capacity=100000, learning_starts=5000,
                    gradient_steps_per_transition=1, exploration_noise=0.1,
                    policy_delay=2, target_noise=0.2, target_noise_clip=0.5,
                    max_steps=1000000, eval_episodes=20, eval_seed_start=40000,
                    diagnostic_horizon=10000, diagnostic_stride=10,
                    early_eval_interval=5000, early_eval_until=50000,
                    eval_interval=25000, device="cpu", torch_threads=1)
    assert all(cfg[key] == value for key, value in expected.items())
    with (DATA / "learning_metrics.csv").open(newline="") as handle:
        learning = list(csv.DictReader(handle))
    summary = json.loads((DATA / "analysis/summary.json").read_text())["conditions"]
    assert len(summary) == 8
    for condition, metrics in summary.items():
        rows = [row for row in learning if row["condition"] == condition and int(row["step"]) == cfg["max_steps"]]
        assert len(rows) == metrics["n"] == 10
        assert {int(row["seed"]) for row in rows} == set(range(20, 30))
        for target, column in [("return_mean", "on_policy_return"), ("bias_mean", "on_policy_bias_q1"), ("rmse_mean", "on_policy_rmse_q1")]:
            assert math.isclose(statistics.mean(float(row[column]) for row in rows), metrics[target], abs_tol=1e-8)
        assert math.isclose(statistics.stdev(float(row["on_policy_return"]) for row in rows), metrics["return_sd"], abs_tol=1e-8)
    schedules = {tuple(int(row["step"]) for row in learning if row["condition"] == condition and int(row["seed"]) == seed)
                 for condition in summary for seed in range(20, 30)}
    expected_schedule = tuple([0, *range(5000, 50001, 5000), *range(75000, 1000001, 25000)])
    assert schedules == {expected_schedule} and len(expected_schedule) == 49

    text = r"""\subsection{Environment}
\qquad Table~\ref{tab:environment} specifies the environment; physics and rewards retain their defaults.\footnote{\href{https://gymnasium.farama.org/environments/box2d/lunar_lander/}{Gymnasium LunarLander documentation}.}

"""
    text += table("lp{11cm}", ["Parameter", "Setting"], [
        ["Environment", r"\texttt{LunarLander-v3}, continuous control; Gymnasium 1.3.0 / Box2D 2.3.10"],
        ["Observation", r"8 components: position, velocity, angle, angular velocity, and two leg-contact indicators"],
        ["Action", r"2 engine controls (main and lateral), each in $[-1,1]$"],
        ["Gravity", r"$-10\,\mathrm{m\,s^{-2}}$ (default)"],
        ["Wind / turbulence", r"Disabled; default powers 15 / 1.5 are inactive"],
        ["Episode limit", r"1,000 transitions for training and performance evaluation"],
        ["Preprocessing", r"No additional observation/reward normalization or clipping"],
    ], "Environment settings shared by all experiments.", "tab:environment")
    text += r"\FloatBarrier" + "\n\n"
    text += r"""\subsection{Hyperparameters}
\qquad Tables~\ref{tab:networks}--\ref{tab:runs} give the baseline settings. Layer normalization (LN) is switched independently in actor and critic, giving four combinations per algorithm.

"""
    text += table("lp{11cm}", ["Parameter", "Setting"], [
        ["Actor", r"$8\rightarrow128\rightarrow128\rightarrow2$; ReLU hidden layers, tanh output"],
        ["Critic", r"Concatenated state/action input: $10\rightarrow128\rightarrow128\rightarrow1$; ReLU hidden layers, linear output"],
        ["LayerNorm", r"When enabled, Linear $\rightarrow$ LN $\rightarrow$ ReLU in both hidden layers; $\epsilon=10^{-5}$, learned gain/offset; outputs unnormalized"],
        ["Optimizer / rates", r"Adam for both networks; $\eta_a=\eta_c=3\times10^{-4}$, $\beta=(0.9,0.999)$, $\epsilon=10^{-8}$; no weight decay or gradient clipping"],
        ["Discount / soft targets", r"$\gamma=0.99$, $\tau=0.005$; target networks initialized from online networks"],
        ["Replay / batch", r"100,000 transitions / 128 uniformly sampled transitions"],
        ["Warm-up / updates", r"5,000 uniform random actions; one critic update round per transition from transition 5,000"],
        ["Exploration", r"Gaussian action noise, $\sigma=0.1$; actions clipped to $[-1,1]$"],
        ["Reward scale / precision", r"1 / float32 on CPU, one Torch/BLAS thread per run"],
    ], "Baseline network and training hyperparameters; later interventions are stated separately.", "tab:networks")
    text += table("lp{5cm}p{7cm}", ["Parameter", "DDPG", "TD3"], [
        ["Critics / target value", "1 / its prediction", "2 independent / minimum prediction"],
        ["Actor / target updates", "Every critic round", "Every second critic round"],
        ["Target-action noise", "None", r"Gaussian $\sigma=0.2$, clipped to $\pm0.5$; actions clipped to bounds"],
    ], "Algorithm-specific baseline settings.", "tab:algorithms")
    text += table("lp{11cm}", ["Parameter", "Setting"], [
        ["Training budget", r"1,000,000 transitions per run; 8 conditions $\times$ 10 seeds = 80 runs"],
        ["Training seeds", r"20--29, matched across all LN combinations and algorithms"],
        ["Evaluation", r"20 deterministic episodes per checkpoint; fixed reset seeds 40000--40019; no exploration noise"],
        ["Checkpoints", r"0; every 5,000 transitions through 50,000; then every 25,000 (49 total); final checkpoint reported"],
        ["Calibration sampling", r"Every tenth state/action pair within the first 1,000 transitions; continuation to termination or 10,000 transitions"],
        ["Common probes", r"24 fixed state/action pairs; reset seeds 41000--41023, 30 uniform prefix actions; continuation with the evaluated actor"],
        ["Horizon check", r"Discounted suffixes compared at 2,500 / 5,000 / 10,000 transitions"],
    ], "Training and evaluation protocol for the Mac LN comparison. Common probes are supporting diagnostics; the results below use on-policy trajectories.", "tab:runs")
    text += r"\FloatBarrier" + "\n\n"
    text += r"""\subsection{Evaluation}
\begin{align*}
R_{je} &= \sum_{t=0}^{\min(L_{je},1000)-1} r_{jet},
&G_{jet} &= \sum_{u=t}^{L_{je}-1}\gamma^{u-t}r_{jeu},\\
\delta_{jet} &= Q_{1,j}(s_{jet},\pi_j(s_{jet}))-G_{jet},\\
B_{je} &= \frac{1}{n_{je}}\sum_{t\in\mathcal I_{je}}\delta_{jet},
&M_{je} &= \sqrt{\frac{1}{n_{je}}\sum_{t\in\mathcal I_{je}}\delta_{jet}^{2}},\\
(\bar R,\bar B,\overline{\mathrm{RMSE}})
&=\frac{1}{NE}\sum_{j=1}^{N}\sum_{e=1}^{E}(R_{je},B_{je},M_{je}).
\end{align*}
\qquad Here $r_{jet}$ is the raw transition reward, $N=10$ training seeds, $E=20$ episodes, $L_{je}\leq10{,}000$ the observed trajectory length, $\mathcal I_{je}=\{0,10,\ldots\}\cap[0,\min(L_{je},1000))$, and $n_{je}=|\mathcal I_{je}|$. The actor and critic are frozen during evaluation; TD3 calibration uses $Q_1$. Table~\ref{tab:metrics} interprets the metrics.

"""
    text += table("lp{11cm}", ["Metric", "Meaning"], [
        [r"Cumulative reward $\bar R$", r"Undiscounted episode performance within 1,000 transitions; higher is better"],
        [r"Signed bias $\bar B$", r"Predicted minus observed discounted return; positive means overestimation, negative underestimation"],
        [r"$\overline{\mathrm{RMSE}}$", r"Magnitude of critic prediction error; lower is better. Averaged over episodes, then seeds"],
    ], "Performance and on-policy critic-calibration metrics. Episodes receive equal weight; RMSE is not pooled across all sampled pairs.", "tab:metrics")
    text += r"""\qquad Trajectories capped at 10,000 steps remain included as finite-return approximations; missing tails are not filled with critic predictions. Training bootstraps through time limits and stops bootstrapping only at natural termination.

\qquad Table~\ref{tab:ln-results} reports the final Mac results. Reward is mean $\pm$ sample SD of the ten seed-level episode means; bias and RMSE are seed means.

"""
    result_rows = []
    for condition, metrics in summary.items():
        algorithm, actor, critic = condition.split("-")
        result_rows.append([
            algorithm.upper(), "On" if actor == "a1" else "Off", "On" if critic == "c1" else "Off",
            rf"${metrics['return_mean']:.2f}\pm{metrics['return_sd']:.2f}$",
            f"{metrics['bias_mean']:+.2f}", f"{metrics['rmse_mean']:.2f}",
        ])
    text += table("lccrrr", ["Algorithm", "Actor LN", "Critic LN", r"Reward (mean $\pm$ SD)", "Bias", "RMSE"], result_rows,
                  "Final results after 1,000,000 transitions, ten training seeds per condition. Critic LN coincides with lower DDPG reward and larger positive bias; TD3 performance remains similar across LN settings.", "tab:ln-results")
    text += r"""\FloatBarrier
\qquad Actor LN does not restore DDPG performance when critic LN is enabled. These comparisons do not establish that overestimation causes poor performance.

"""
    out = ROOT / "report/generated/first_experiments.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Verified 80 runs, all final metrics and 49 checkpoints; wrote {out}")


if __name__ == "__main__":
    main()
