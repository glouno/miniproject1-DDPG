"""Measure end-to-end training throughput without changing scientific runs.

Independent subprocesses, one Torch/BLAS thread each; identical seeded jobs for
all worker counts. These are timing workloads, not learning-quality evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 4, 8])
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args()
    assert args.steps > 1000 and args.jobs > 0 and all(w > 0 for w in args.workers)
    base = Path(__file__).resolve().parent
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((base/'configs/pilot.json').read_text())
    config.update(max_steps=args.steps, learning_starts=1000, eval_interval=args.steps,
                  eval_episodes=2, diagnostic_horizon=1000, device=args.device)
    config_path = out/'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  platform=platform.platform(), python=platform.python_version(),
                  device=args.device, jobs=args.jobs, config=config,
                  source_sha256=hashlib.sha256((base/'experiment.py').read_bytes()).hexdigest(),
                  note='One timing batch per worker count; short timing workloads, not scientific results.',
                  batches=[])
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for workers in args.workers:
        batch_dir = out/f'workers-{workers}'
        batch_dir.mkdir()
        def launch(index):
            name = f'job-{index}'
            command = [sys.executable, str(base/'experiment.py'), '--config', str(config_path),
                       '--algorithm', 'td3', '--layer-norm', '1', '--seed', str(index),
                       '--output', str(batch_dir/name)]
            with (batch_dir/f'{name}.log').open('w') as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'Failed {name}; read {batch_dir/name}.log')
            return json.loads((batch_dir/name/'summary.json').read_text())
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(launch, range(args.jobs)))
        seconds = time.monotonic() - started
        batch = dict(workers=workers, wall_seconds=seconds,
                     transitions_per_second=args.jobs*args.steps/seconds, runs=summaries)
        result['batches'].append(batch)
        (out/'benchmark.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k:v for k,v in batch.items() if k!='runs'}), flush=True)


if __name__ == '__main__':
    main()
