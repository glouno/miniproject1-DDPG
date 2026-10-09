"""Export the completed Paris WSL A+B study, preserving all frozen run data.

Tables and figures go in Git; lossless raw-data archives go in a GitHub Release.
Run with the study's locked interpreter. Output directories must be fresh.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import sys
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    root, out, assets = (p.resolve() for p in (args.input, args.output, args.assets))
    assert 'microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower()
    assert socket.gethostname() == 'paris-devbox-wsl'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert digest(root / 'manifest.json') == (root / 'manifest.sha256').read_text().strip()
    for name, expected in manifest['sha256'].items():
        assert digest(root / name) == expected, name
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == 'complete'
    assert Counter(j['stage'] for j in manifest['jobs']) == {'A': 80, 'B': 60}
    for job in manifest['jobs']:
        directory = root / job['directory']
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['status'] == 'complete'
        assert summary['transitions'] == job['config']['max_steps'] == 1_000_000
        assert json.loads((directory / 'completion_validation.json').read_text())['status'] == 'valid'
        assert status['jobs'][job['directory']]['status'] == 'complete'
    out.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    label = 'wsl-stages-a-b-20261008'
    release_url = f'https://github.com/glouno/miniproject1-DDPG/releases/tag/results-{label}'

    # Re-render with the frozen analysis implementation and a visible WSL label.
    # Redirect its writes into the publication; originals remain untouched.
    sys.path.insert(0, str(root / 'source'))
    import lunarlander_rl.followup_analyse as analysis_module
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    original_savefig = Figure.savefig
    current_stage = None

    def labelled_savefig(fig, filename, *a, **kw):
        filename = Path(filename)
        destination = out / filename.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not getattr(fig, '_wsl_label_added', False):
            fig.text(.99, .005, f'WSL (Paris) | Stage {current_stage} | CPU | completed 8 October 2026',
                     ha='right', va='bottom', fontsize=9)
            fig._wsl_label_added = True
        return original_savefig(fig, destination, *a, **kw)

    original_atomic_json = analysis_module.atomic_json

    def labelled_json(path, value):
        original = json.loads(Path(path).read_text())
        assert value == original, f'Regenerated analysis differs: {path}'
        value = dict(platform='wsl', host='paris-devbox-wsl', **value)
        destination = out / Path(path).relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return original_atomic_json(destination, value)

    Figure.savefig = labelled_savefig
    analysis_module.atomic_json = labelled_json
    for current_stage in ('A', 'B'):
        analysis_module.analyse(root, current_stage)
    Figure.savefig = original_savefig
    analysis_module.atomic_json = original_atomic_json

    learning, final, conditions = [], [], []
    for job in manifest['jobs']:
        directory, cfg = root / job['directory'], job['config']
        identity = dict(platform='wsl', stage=job['stage'], condition=job['cell'],
                        algorithm=cfg['algorithm'], actor_layer_norm=cfg['actor_layer_norm'],
                        critic_layer_norm=cfg['critic_layer_norm'], actor_lr=cfg['actor_lr'],
                        critic_lr=cfg['critic_lr'], seed=cfg['seed'])
        metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        assert len(metrics) == 49 and metrics[-1]['step'] == 1_000_000
        assert [m['step'] for m in metrics] == sorted({m['step'] for m in metrics})
        for metric in metrics:
            row = dict(identity, **{k: metric[k] for k in
                       ('step', 'elapsed_seconds', 'critic_rounds', 'actor_updates')})
            for kind in ('on_policy', 'common_probe'):
                for key, value in metric[kind].items():
                    if isinstance(value, (int, float)) or value is None:
                        row[f'{kind}_{key}'] = value
            row['common_probe_analysis_rmse'] = math.sqrt(sum(
                e['rmse_q1'] ** 2 for e in metric['common_probe']['episodes']) /
                len(metric['common_probe']['episodes']))
            learning.append(row)
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['final_return'] == metrics[-1]['on_policy']['return']
        final.append(dict(identity, **summary))
    for stage in ('A', 'B'):
        analysis = json.loads((out / f'analysis/{stage}/summary.json').read_text())
        assert not analysis['partial'] and analysis['completed'] == analysis['expected']
        conditions.extend(dict(platform='wsl', stage=stage, condition=name, **values)
                          for name, values in analysis['conditions'].items())
    write_csv(out / 'learning_metrics.csv', learning)
    write_csv(out / 'seed_summary.csv', final)
    write_csv(out / 'conditions.csv', conditions)
    (out / 'provenance').mkdir()
    for name in ('manifest.json', 'manifest.sha256', 'probes.json', 'launch_record.json', 'status.json'):
        shutil.copy2(root / name, out / 'provenance' / name)
    write_json(out / 'provenance/platform.json', dict(
        platform='wsl', host=socket.gethostname(), kernel=Path('/proc/sys/kernel/osrelease').read_text().strip(),
        source_run=str(root), device='cpu', workers=8, threads_per_worker=1,
        scientific_preparation_revision=manifest['git_revision'],
        started_at=status['started_at'], finished_at=status['finished_at'],
        recorded_versions=manifest['versions'],
        note='These are WSL repetitions. Mac data are separate and must not be pooled as extra independent seeds.'))

    # Include every original file except the ephemeral runner lock.
    files = sorted(f for f in root.rglob('*') if f.is_file() and f != root / 'runner.lock')
    inventory = {str(f.relative_to(root)): dict(bytes=f.stat().st_size, sha256=digest(f)) for f in files}
    with gzip.open(out / 'provenance/file_inventory.json.gz', 'wt') as stream:
        json.dump(inventory, stream, separators=(',', ':'))
    archives, verified_files = [], set()

    def archive(name, selected):
        destination = assets / f'{label}-{name}.tar.gz'
        with tarfile.open(destination, 'w:gz', compresslevel=1) as tar:
            for f in selected:
                tar.add(f, arcname=f'{label}/{f.relative_to(root)}', recursive=False)
        assert destination.stat().st_size < 2 * 1024 ** 3, 'Split archive before GitHub upload'
        with tarfile.open(destination, 'r:gz') as tar:
            for member in tar:
                assert member.isfile()
                relative = str(Path(member.name).relative_to(label))
                expected = inventory[relative]
                h = hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                assert member.size == expected['bytes'] and h.hexdigest() == expected['sha256'], relative
                assert relative not in verified_files
                verified_files.add(relative)
        record = dict(name=destination.name, bytes=destination.stat().st_size, sha256=digest(destination))
        archives.append(record)
        print(json.dumps(record), flush=True)

    assigned = set()
    for stage in ('A', 'B'):
        cells = sorted({j['cell'] for j in manifest['jobs'] if j['stage'] == stage})
        for cell in cells:
            jobs = [j for j in manifest['jobs'] if j['stage'] == stage and j['cell'] == cell]
            prefixes = [j['directory'] + '/' for j in jobs]
            logs = {f"logs/{stage}-{cell}-seed{j['config']['seed']}.log" for j in jobs}
            selected = [f for f in files if any(str(f.relative_to(root)).startswith(p) for p in prefixes)
                        or str(f.relative_to(root)) in logs]
            assigned.update(selected)
            archive(f'{stage.lower()}-{cell}', selected)
    archive('provenance', [f for f in files if f not in assigned])
    assert verified_files == set(inventory)
    checksums = ''.join(f"{a['sha256']}  {a['name']}\n" for a in archives)
    (out / 'SHA256SUMS').write_text(checksums)
    (assets / 'SHA256SUMS').write_text(checksums)
    publication = dict(platform='wsl', stages=['A', 'B'], label=label, release_url=release_url,
        exported_at=datetime.now(timezone.utc).isoformat(), completed_runs=140,
        training_steps=140_000_000, scientific_preparation_revision=manifest['git_revision'],
        raw_bytes=sum(v['bytes'] for v in inventory.values()),
        compressed_archive_bytes=sum(a['bytes'] for a in archives), archives=archives,
        exporter_sha256=digest(Path(__file__).resolve()))
    write_json(out / 'publication.json', publication)
    write_json(out / 'verification.json', dict(status='passed', platform='wsl', completed_runs=140,
        stage_a_runs=80, stage_b_runs=60, checkpoint_table_rows=len(learning),
        verified_at=datetime.now(timezone.utc).isoformat(), archived_source_files=len(verified_files),
        archives=len(archives), method='Frozen hashes checked; all completion validations valid; regenerated statistics exactly match originals; every archived file size and SHA-256 matches the full inventory.'))
    shutil.copy2(out / 'publication.json', assets / 'publication.json')
    print(json.dumps(publication), flush=True)


if __name__ == '__main__':
    main()
