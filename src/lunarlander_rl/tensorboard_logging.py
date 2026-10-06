"""Derived TensorBoard views; JSON journals remain the experiment records."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def log_metric(writer, metric):
    step = metric['step']
    for kind in ('on_policy', 'common_probe'):
        data = metric[kind]
        for key, value in data.items():
            if key == 'episodes' or value is None or not isinstance(value, (int, float)):
                continue
            if key == 'rmse_q1' and kind == 'common_probe':
                # Same pooled-probe definition as study_analyse.py.
                value = float(np.sqrt(np.mean([e['rmse_q1']**2 for e in data['episodes']])))
            if math.isfinite(value):
                writer.add_scalar(f'eval/{kind}/{key}', value, step)
    for key in ('elapsed_seconds', 'critic_rounds', 'actor_updates'):
        if key in metric:
            writer.add_scalar(f'run/{key}', metric[key], step)
    writer.flush()


def log_episode(writer, episode):
    writer.add_scalar('train/episode_return', episode['reward'], episode['step'])
    writer.add_scalar('train/episode_length', episode['length'], episode['step'])


def create_writer(run, destination=None):
    """Rebuild valid history after journal recovery, purging stale event steps."""
    run = Path(run)
    writer = SummaryWriter(str(destination or run/'tensorboard'), purge_step=0, flush_secs=10)
    metadata = run/'metadata.json'
    if metadata.exists():
        writer.add_text('reproducibility/metadata', '```json\n'+metadata.read_text()+'\n```', 0)
    for filename, callback in (('metrics.jsonl', log_metric), ('training_episodes.jsonl', log_episode)):
        journal = run/filename
        if journal.exists():
            for line in journal.read_text().splitlines():
                callback(writer, json.loads(line))
    writer.flush()
    return writer


def export_study(root, destination):
    root, destination = Path(root), Path(destination)
    manifest = json.loads((root/'manifest.json').read_text())
    for job in manifest['jobs']:
        writer = create_writer(root/job['directory'], destination/job['directory'])
        writer.add_text('reproducibility/study_manifest', '```json\n'+json.dumps(manifest, indent=2)+'\n```', 0)
        writer.close()
    print(f"Exported {len(manifest['jobs'])} runs to {destination.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    export_study(args.input, args.output or args.input/'tensorboard')


if __name__ == '__main__':
    main()
