"""Compare a small explicit list of final policies, preserving episode pairing."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np

from lunarlander_rl.horizon_sensitivity import analyse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, results = [], []
    for run in args.run:
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyse(run, args.output / run.name)
        results.append(result)
        stat = result["statistics"]
        rows.append(dict(policy=run.name, evaluation_episodes=result["evaluation_episodes"],
            bias_1000=result["bias_short"], bias_10000=result["bias_long"],
            delta=stat["mean_delta"], ci95_low=stat["bootstrap_ci95"][0],
            ci95_high=stat["bootstrap_ci95"][1], paired_t_p=stat["paired_t_p"],
            episodes_past_1000=result["episodes_past_short"], censored_10000=result["censored_long"],
            max_abs_pair_delta=result["max_abs_pair_delta"],
            bias_min_1000=result.get("bias_min_short"), bias_min_10000=result.get("bias_min_long"),
            bias_q2_1000=result.get("bias_q2_short"), bias_q2_10000=result.get("bias_q2_long")))
    # Control the familywise error rate for this explicit small set of policies.
    order = np.argsort([r["paired_t_p"] for r in rows])
    running = 0.
    for rank, index in enumerate(order):
        running = min(1., max(running, (len(rows) - rank) * rows[index]["paired_t_p"]))
        rows[index]["holm_paired_t_p"] = running
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(dict(
        inference_unit="Independent evaluation reset seeds within each one frozen policy",
        scope="Explicit selected final policies; no algorithm-wide inference across training seeds",
        confidence_intervals="Unadjusted paired episode-bootstrap intervals, 10000 resamples, seed 419",
        multiplicity="Holm correction over the selected policies' paired t-tests",
        comparison=rows, policies=results), indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
