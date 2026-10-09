"""Generate the shared report's experiment tables without changing training data.

Run from the repository root:
    python3 report/scripts/build_results_tables.py

Inputs: published Mac Stage A and WSL Stage B manifests, learning tables and
analysis summaries; the completed local Mac focused manifest and seed table.
Output: report/generated/results.tex (an intermediate fragment for Samskrift).
"""
import collections
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads((ROOT / path).read_text())


def read_csv(path):
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def lr(value):
    return {0.0003: r"$3\times10^{-4}$", 0.0001: r"$10^{-4}$"}[value]


def ln(value):
    return "On" if value else "Off"


def score(mean, sd):
    return rf"${mean:.2f}\pm{sd:.2f}$"


def table(columns, header, rows, caption, label):
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.2}",
        r"\rowcolors{2}{reportblue}{white}",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", "",
    ])


def published_rows(folder, stage):
    manifest = read_json(f"{folder}/provenance/manifest.json")
    summary_path = "analysis/summary.json" if stage == "A" and "/mac/" in folder else f"analysis/{stage}/summary.json"
    summary = read_json(f"{folder}/{summary_path}")
    grouped = collections.defaultdict(list)
    for row in read_csv(f"{folder}/learning_metrics.csv"):
        if row["stage"] == stage and int(row["step"]) == 1_000_000:
            grouped[row["condition"]].append(row)
    result = []
    for condition, metrics in summary["conditions"].items():
        records = grouped[condition]
        assert len(records) == metrics["n"] == 10
        assert {int(row["seed"]) for row in records} == set(range(20, 30) if stage == "A" else range(30, 40))
        for target, column in [("return_mean", "on_policy_return"), ("bias_mean", "on_policy_bias_q1"), ("rmse_mean", "on_policy_rmse_q1")]:
            assert math.isclose(statistics.mean(float(row[column]) for row in records), metrics[target], abs_tol=1e-8)
        cfg = next(job["config"] for job in manifest["jobs"] if job["stage"] == stage and job["cell"] == condition)
        assert cfg["max_steps"] == 1_000_000
        result.append((cfg, metrics))
    return result


def main():
    mac = published_rows("results/mac/stage-a-20261008", "A")
    wsl = published_rows("results/wsl/stages-a-b-20261008", "B")
    common = mac[0][0]
    fm = read_json("outputs/ddpg-focused-20261009-m1air/manifest.json")
    focused = collections.defaultdict(list)
    for row in read_csv("outputs/ddpg-focused-20261009-m1air/seed_summary.csv"):
        assert int(row["transitions"]) == 1_000_000
        focused[row["condition"]].append(row)
    assert read_json("outputs/ddpg-focused-20261009-m1air/status.json")["status"] == "complete"
    for cfg, _ in mac + wsl:
        for key in ["env_id", "continuous", "gravity", "enable_wind", "train_horizon", "max_steps", "actor_hidden", "critic_hidden", "gamma", "tau", "batch_size", "replay_capacity", "learning_starts", "gradient_steps_per_transition", "exploration_noise", "eval_episodes", "eval_seed_start", "diagnostic_horizon", "diagnostic_stride"]:
            assert cfg[key] == common[key]
    for job in fm["jobs"]:
        for key in common:
            if key != "seed":
                reference = next(cfg for cfg, _ in mac if cfg["algorithm"] == "ddpg" and not cfg["actor_layer_norm"] and cfg["critic_layer_norm"])
                assert job["config"][key] == reference[key]

    text = r"""\section{Experimental setup and evaluation}
\qquad We report three completed studies: the actor/critic LayerNorm comparison on Mac (Stage A, 80 runs), the DDPG learning-rate study on WSL (Stage B, 60 runs), and an exploratory DDPG reward-scale/actor-delay follow-up on Mac (30 new runs). Every research run started from scratch and collected 1,000,000 training transitions. Each condition has ten training seeds. The fixed training settings are listed in Table~\ref{tab:shared-settings}; the result tables state the parameters that distinguish conditions explicitly.

"""
    text += table("lp{10.5cm}", ["Parameter", "Shared setting"], [
        ["Environment", r"Continuous \texttt{LunarLander-v3}; gravity $-10$; wind disabled; 1,000-step training episode limit"],
        ["Networks", r"Two 128-unit ReLU hidden layers in actor and critic; tanh actor output; linear Q output"],
        ["Optimizer", r"Adam, $\beta=(0.9,0.999)$, $\epsilon=10^{-8}$, weight decay zero"],
        ["Discount / targets", r"$\gamma=0.99$; soft-update coefficient $\tau=0.005$"],
        ["Replay / minibatches", r"Capacity 100,000; batch size 128; one critic round per transition, starting at transition 5,000"],
        ["Exploration", r"Uniform random actions for the first 5,000 transitions; then Gaussian action noise with $\sigma=0.1$, clipped actions"],
        ["LayerNorm", r"When enabled: hidden Linear $\rightarrow$ LayerNorm $\rightarrow$ ReLU; affine parameters enabled, $\epsilon=10^{-5}$"],
        ["Backend", r"CPU, one Torch/BLAS thread per worker; isolated actor/critic initialization streams"],
    ], "Fixed settings for all three research studies. Learning rates, LayerNorm switches, reward multipliers and update cadence are specified in the following result tables.", "tab:shared-settings")
    text += r"""\qquad All results refer to the final checkpoint at 1,000,000 transitions. Evaluation reward is the mean undiscounted total over the first at most 1,000 steps of each of 20 deterministic episodes (reset seeds 40000--40019), then averaged equally across the ten training seeds. The displayed $\pm$ value is the sample standard deviation across training-seed means, not a confidence interval. Seed-level variability matters in deep reinforcement learning~\cite{henderson2018deep}.

\qquad Signed bias is $Q_1(s,a)-G_{\mathrm{MC}}(s,a)$, with $G_{\mathrm{MC}}$ discounted using $\gamma=0.99$: positive values indicate overestimation and negative values underestimation. Bias and RMSE are measured on the evaluated policy's trajectories, first within episodes and then averaged equally across episodes and seeds. RMSE is an average of episode RMSEs, not a pooled RMSE. Diagnostic suffixes extend to natural termination or 10,000 steps; capped, censored trajectories remain in the recorded finite-horizon estimates. Bias and RMSE are therefore comparisons with the observed discounted suffix, while evaluation reward is undiscounted.

\qquad Evaluation checkpoints occur at step zero, every 5,000 steps through 50,000, and every 25,000 steps thereafter (49 checkpoints). The studies also retain 24 common probes (seeds 41000--41023), diagnostics at horizons 2,500/5,000/10,000, optimization journals every 1,000 transitions, and recovery states every 5,000 transitions. The tables below use on-policy calibration, rather than common-probe calibration. Time-limit truncation retains the bootstrap term; only natural termination removes it.
\FloatBarrier
\clearpage

\section{Stage A: actor and critic LayerNorm on Mac}
\qquad Stage A independently switched LayerNorm in the actor and critic for both DDPG and TD3. Ten matched training seeds (20--29) were used for each of the eight conditions; both learning rates were fixed at $3\times10^{-4}$. Table~\ref{tab:mac-stage-a} reports the completed Mac results.

\qquad DDPG has one critic and updates its actor and both target networks every critic round. TD3 has two critics, uses their minimum in the Bellman target, adds target-action smoothing ($\sigma=0.2$, noise clipped to $\pm0.5$), and updates its actor and target networks every second critic round. The algorithm comparison therefore changes these mechanisms together. All other settings in Table~\ref{tab:shared-settings} remain fixed.

"""
    header = ["Algorithm", r"\shortstack{Actor\\LN}", r"\shortstack{Critic\\LN}", r"\shortstack{Actor\\LR}", r"\shortstack{Critic\\LR}", r"\shortstack{Reward\\mean $\pm$ SD}", "Bias", "RMSE"]
    rows = [[cfg["algorithm"].upper(), ln(cfg["actor_layer_norm"]), ln(cfg["critic_layer_norm"]), lr(cfg["actor_lr"]), lr(cfg["critic_lr"]), score(m["return_mean"], m["return_sd"]), f'{m["bias_mean"]:+.2f}', f'{m["rmse_mean"]:.2f}'] for cfg, m in mac]
    text += table("lccccrrr", header, rows, "Mac Stage A: final on-policy results, ten seeds per condition. Reward scaling is one throughout. Critic LayerNorm coincides with lower DDPG reward and larger positive Q bias; TD3 remains strong across all four LayerNorm combinations. Higher reward and smaller absolute bias/RMSE are preferable.", "tab:mac-stage-a")
    text += r"""\qquad With actor LayerNorm off, adding critic LayerNorm changes DDPG mean reward from 135.18 to $-79.79$, and mean bias from 67.25 to 198.34. Adding actor LayerNorm with critic LayerNorm already on does not rescue the mean reward in this dataset. TD3 means remain between 277.07 and 280.99. These comparisons describe joint learning outcomes and do not establish that Q bias causes poor performance.
\FloatBarrier

\section{Stage B: DDPG learning rates on WSL}
\qquad Stage B used DDPG with actor LayerNorm off and ten fresh training seeds (30--39) per condition. Four critic-LayerNorm-on conditions form a $2\times2$ actor/critic learning-rate factorial; two critic-LayerNorm-off conditions provide controls at the original and jointly reduced rates. Table~\ref{tab:wsl-stage-b} reports the 60 completed WSL runs. Reward scaling remains one, and actor and target networks update every critic round.

"""
    rows = [[ln(cfg["actor_layer_norm"]), ln(cfg["critic_layer_norm"]), lr(cfg["actor_lr"]), lr(cfg["critic_lr"]), score(m["return_mean"], m["return_sd"]), f'{m["bias_mean"]:+.2f}', f'{m["rmse_mean"]:.2f}'] for cfg, m in wsl]
    text += table("ccccrrr", header[1:], rows, "WSL Stage B: final DDPG results, ten fresh seeds per condition. The separate actor and critic learning-rate columns expose the factorial interventions. Lower actor learning rate improves the critic-LayerNorm-on means descriptively, while the LayerNorm-off controls retain higher mean rewards.", "tab:wsl-stage-b")
    text += r"""\qquad With critic learning rate fixed at $3\times10^{-4}$, reducing actor learning rate to $10^{-4}$ changes mean reward from $-18.04$ to 59.66 and bias from 206.41 to 106.59. Reducing both rates gives mean reward 32.86 with critic LayerNorm on, compared with 197.53 with it off. The prespecified actor-rate main effect has Holm-adjusted $p=0.0645$ across the three primary Stage B contrasts, so these means alone do not establish a confirmed rescue or an optimal rate. Stage B uses a different platform and fresh seeds from Mac Stage A; its controls provide the within-study comparison.
\FloatBarrier

\clearpage
\section{Exploratory DDPG reward scaling and actor delay on Mac}
\qquad The focused study ran three new conditions with DDPG, actor LayerNorm off, critic LayerNorm on, and both learning rates $3\times10^{-4}$. Each condition reused the Stage A seed indices 20--29, with all agents newly initialized. The historical Mac Stage A condition with these same LayerNorm settings supplies the reference in the first row of Table~\ref{tab:mac-focused}; no additional baseline was trained.

\qquad Reward scaling multiplies only the immediate reward in the Bellman target; replay retains raw rewards. Evaluation reward stays raw and Q is divided by the multiplier before calibration. Actor-delay conditions keep both target-network updates at every critic round, so only actor update frequency changes. They retain one critic and no target-action smoothing. The critic:actor ratio specifies update frequency, rather than learning-rate scaling.

"""
    baseline = next((cfg, m) for cfg, m in mac if cfg["algorithm"] == "ddpg" and not cfg["actor_layer_norm"] and cfg["critic_layer_norm"])
    cfg, m = baseline
    rows = [["Off", "On", lr(cfg["actor_lr"]), lr(cfg["critic_lr"]), "1", "1:1", score(m["return_mean"], m["return_sd"]), f'{m["bias_mean"]:+.2f}', f'{m["rmse_mean"]:.2f}']]
    for condition in ["reward-scale-0.1", "actor-delay-2", "actor-delay-3"]:
        records = focused[condition]
        assert len(records) == 10 and {int(row["seed"]) for row in records} == set(range(20, 30))
        cfg = next(j["config"] for j in fm["jobs"] if j["cell"] == condition)
        reward = [float(row["return_raw"]) for row in records]
        bias = statistics.mean(float(row["bias_raw"]) for row in records)
        rmse = statistics.mean(float(row["rmse_raw"]) for row in records)
        rows.append([ln(cfg["actor_layer_norm"]), ln(cfg["critic_layer_norm"]), lr(cfg["actor_lr"]), lr(cfg["critic_lr"]), f'{cfg["reward_scale"]:g}', f'{cfg["ddpg_actor_delay"]}:1', score(statistics.mean(reward), statistics.stdev(reward)), f'{bias:+.2f}', f'{rmse:.2f}'])
    text += table("ccccccrrr", [*header[1:5], r"\shortstack{Reward\\scale}", r"\shortstack{Critic:actor\\updates}", *header[5:]], rows, "Mac focused DDPG study: final means over ten seeds. The first row is the reused historical reference; the other rows are new reward-scale and actor-delay interventions. Target networks update every critic round in every row. All metrics use raw reward units. The historical reference ran on M1 Max and the interventions on M1 Air, with the same CPU backend and package versions.", "tab:mac-focused")
    text += r"""\qquad Relative to the historical reference, reward scaling to 0.1 increases mean bias from 198.34 to 270.23 and RMSE from 204.36 to 277.30. Actor delay of three critic rounds reduces them to 94.33 and 101.31, respectively, while mean reward rises from $-79.79$ to 33.49. Reward variation across seeds remains large. The paired descriptive 95\% bootstrap interval for the delay-three reward change is $[-20.46,236.29]$, which includes zero. These are exploratory comparisons selected after observing Stage A, with a historical reference on a different Mac; they do not establish a causal mechanism or a confirmed reward improvement.
\FloatBarrier
"""
    output = ROOT / "report/generated/results.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(f"Verified 80 Mac Stage A, 60 WSL Stage B and 30 focused records; wrote {output}")


if __name__ == "__main__":
    main()
