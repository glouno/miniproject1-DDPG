"""Compare local worker counts with identical disposable intervention jobs."""
import argparse
import json
from pathlib import Path
from .benchmark_followup import measure
from .focused_runner import prepare
from .paths import frozen_command
from .study_diagnostics import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 6, 8])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for count in args.workers:
        directory = out / f"workers-{count}"
        manifest = prepare(directory, count, "benchmark")
        seconds, resources = measure(frozen_command(directory/"source", "focused_runner") +
                                     ["--output", str(directory)], directory/"source")
        summaries = [json.loads((directory/j["directory"]/"summary.json").read_text())
                     for j in manifest["jobs"]]
        atomic_json(directory/"resources.json", resources)
        result = dict(workers=count, wall_seconds=seconds,
            transitions_per_second=manifest["total_transitions"]/seconds,
            mean_training_seconds=sum(s["training_seconds"] for s in summaries)/len(summaries),
            mean_checkpoint_seconds=sum(s["checkpoint_seconds"] for s in summaries)/len(summaries),
            resources={k: v for k, v in resources.items() if k != "samples"})
        results.append(result)
        atomic_json(out/"benchmark.json", dict(batches=results,
            note="12 identical jobs per worker count; 6000 steps, 1000 warmup, reduced evaluation. Not scientific data."))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
