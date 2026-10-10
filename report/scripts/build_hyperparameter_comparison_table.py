"""Build the compact hyperparameter table without modifying experiment data.

Run from the repository root:
    python3 report/scripts/build_hyperparameter_comparison_table.py

Inputs: published Mac Stage A / WSL Stage B conditions.csv and the completed
Mac focused study's seed_summary.csv and the WSL target-update release's
conditions.csv. Outputs: a LaTeX table and an exact-value
CSV audit under report/generated/ddpg-hyperparameters/. All seven variants use
the historical Mac baseline plotted in the report's hyperparameter figure.
"""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/generated/ddpg-hyperparameters"


def rows(path):
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def endpoint(records, condition, stage=None):
    matches = [r for r in records if r["condition"] == condition
               and (stage is None or r["stage"] == stage)]
    assert len(matches) == 1 and int(matches[0]["n"]) == 10
    return float(matches[0]["return_mean"]), float(matches[0]["bias_mean"])


def indicator(value):
    color, arrow = ("green!45!black", r"\uparrow") if value >= 0 else ("red!75!black", r"\downarrow")
    return rf"\textcolor{{{color}}}{{$({arrow}\,{abs(value):.2f})$}}"


def main():
    mac = rows("results/mac/stage-a-20261008/conditions.csv")
    wsl = rows("results/wsl/stages-a-b-20261008/conditions.csv")
    focused_path = ROOT / "outputs/ddpg-focused-20261009-m1air"
    assert json.loads((focused_path / "status.json").read_text())["status"] == "complete"
    focused = rows("outputs/ddpg-focused-20261009-m1air/seed_summary.csv")
    reward0, bias0 = endpoint(mac, "ddpg-a0-c1")
    assert abs(reward0) > 0 and abs(bias0) > 0
    variants = []
    rate_change = r"$3\times10^{-4}\rightarrow10^{-4}$"
    for label, condition in [
        (r"Actor learning rate $\eta_a$", "ddpg-a0-c1-clr0.0003-alr0.0001"),
        (r"Critic learning rate $\eta_c$", "ddpg-a0-c1-clr0.0001-alr0.0003"),
        (r"Both learning rates $\eta_a,\eta_c$", "ddpg-a0-c1-clr0.0001-alr0.0001"),
    ]:
        reward, bias = endpoint(wsl, condition, "B")
        variants.append((label, rate_change, condition, reward, bias))
    for delay in (2, 3):
        condition = f"actor-delay-{delay}"
        records = [r for r in focused if r["condition"] == condition]
        assert len(records) == 10 and {int(r["seed"]) for r in records} == set(range(20, 30))
        assert all(int(r["transitions"]) == 1_000_000 for r in records)
        reward = statistics.mean(float(r["return_raw"]) for r in records)
        bias = statistics.mean(float(r["bias_raw"]) for r in records)
        variants.append(("Critic:actor update ratio", rf"$1:1\rightarrow{delay}:1$", condition, reward, bias))
    target = rows("results/wsl/target-updates-20261010/conditions.csv")
    for label, change, condition in [
        (r"Soft target coefficient $\tau$", r"$0.005\rightarrow0.001$", "soft-tau-0.001"),
        ("Target update rule", r"Soft $\rightarrow$ hard ($K=400$)", "hard-k400"),
    ]:
        reward, bias = endpoint(target, condition)
        variants.append((label, change, condition, reward, bias))
    audit = [dict(condition="ddpg-a0-c1 (baseline)", baseline_reward=reward0,
                  baseline_bias=bias0, variant_reward=reward0, variant_bias=bias0,
                  reward_improvement=0.0, bias_improvement=0.0,
                  reward_improvement_percent=0.0, bias_improvement_percent=0.0)]
    latex = [
        r"\begin{table}[htbp]", r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.1}",
        r"\rowcolors{2}{reportblue}{white}",
        r"\begin{tabular}{lcrr}", r"\toprule",
        r"Hyperparameter & Change & \shortstack{Reward $\uparrow$\\(improvement)} & \shortstack{Estimation bias $\rightarrow0$\\(improvement)} \\",
        r"\midrule",
        rf"Baseline & No change & \textcolor{{black}}{{${reward0:.2f}$}} & \textcolor{{black}}{{${bias0:+.2f}$}} \\",
    ]
    for label, change, condition, reward, bias in variants:
        reward_improvement = reward - reward0
        bias_improvement = abs(bias0) - abs(bias)
        reward_percent = 100 * reward_improvement / abs(reward0)
        bias_percent = 100 * bias_improvement / abs(bias0)
        latex.append(" & ".join([
            label, change,
            rf"\textcolor{{black}}{{${reward:.2f}$}} " + indicator(reward_improvement),
            rf"\textcolor{{black}}{{${bias:+.2f}$}} " + indicator(bias_improvement),
        ]) + r" \\")
        audit.append(dict(condition=condition, baseline_reward=reward0, baseline_bias=bias0,
                          variant_reward=reward, variant_bias=bias,
                          reward_improvement=reward_improvement, bias_improvement=bias_improvement,
                          reward_improvement_percent=reward_percent,
                          bias_improvement_percent=bias_percent))
    latex += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Final ten-seed means after one million transitions, with improvements in raw units relative to the Mac Stage A baseline (critic LN on, actor LN off): reward $R-R_0$, bias $|B_0|-|B|$. Final values are black; the parenthesized arrows give the magnitude of improvement (green $\uparrow$) or deterioration (red $\downarrow$). Learning-rate and target-update variants are WSL runs; update-ratio variants are Mac runs.}",
        r"\label{tab:ddpg-hyperparameter-improvements}", r"\end{table}", "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison-table.tex").write_text("\n".join(latex))
    with (OUT / "comparison-table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print("\n".join(latex))


if __name__ == "__main__":
    main()
