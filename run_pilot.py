"""Run the predefined 2 algorithms x 2 normalization settings x 3 seeds pilot.

Usage: <locked-python> run_pilot.py --output outputs/pilot-20261005 --workers 4
Each child uses one CPU thread. Existing run directories are never overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = (args.config or base / 'configs' / 'pilot.json').resolve()
    assert args.workers > 0 and len(set(args.seeds)) == len(args.seeds)
    jobs = [(algo, ln, seed) for seed in args.seeds for algo in ('ddpg', 'td3') for ln in (0, 1)]
    manifest = dict(stage='exploratory pilot', expected_runs=len(jobs), seeds=args.seeds, device=args.device,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=json.loads(config.read_text()),
        experiment_sha256=hashlib.sha256((base/'experiment.py').read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        jobs=[dict(algorithm=a, layer_norm=bool(ln), seed=s,
                   directory=f'{a}-ln{ln}-seed{s}') for a, ln, s in jobs])
    if (output/'manifest.json').exists():
        raise FileExistsError('Existing pilot manifest; choose a new output directory.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output/'logs').mkdir()

    def run(job):
        algo, ln, seed = job
        name = f'{algo}-ln{ln}-seed{seed}'
        command = [sys.executable, str(base/'experiment.py'), '--config', str(config),
                   '--algorithm', algo, '--layer-norm', str(ln), '--seed', str(seed),
                   '--output', str(output/name), '--device', args.device]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        print('START '+name, flush=True)
        with (output/'logs'/f'{name}.log').open('w') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        return name, completed.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            name, returncode = future.result()
            print(f'FINISHED {name}: exit {returncode}', flush=True)
            if returncode:
                failures.append(name)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}; read logs. No results were replaced.')
    print(f'PILOT COMPLETE: {len(jobs)}/{len(jobs)} runs', flush=True)


if __name__ == '__main__':
    main()
