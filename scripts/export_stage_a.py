"""Publishable, platform-labelled Stage A tables and complete release archives.

Reads completed runs without altering checkpoints or frozen provenance.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--platform', choices=('mac', 'wsl'), required=True)
    p.add_argument('--date', required=True)
    a = p.parse_args()
    root = a.input.resolve(); out = a.output.resolve(); assets = a.assets.resolve()
    out.mkdir(parents=True, exist_ok=False); assets.mkdir(parents=True, exist_ok=False)
    m = json.loads((root/'manifest.json').read_text())
    assert digest(root/'manifest.json') == (root/'manifest.sha256').read_text().strip()
    for name, expected in m['sha256'].items():
        assert digest(root/name) == expected, name
    jobs = [j for j in m['jobs'] if j['stage'] == 'A']
    assert len(jobs) == 80
    for j in jobs:
        d = root/j['directory']; summary = json.loads((d/'summary.json').read_text())
        assert summary['status'] == 'complete' and summary['transitions'] == j['config']['max_steps'] == 1_000_000
        assert json.loads((d/'completion_validation.json').read_text())['status'] == 'valid'
    analysis = json.loads((root/'analysis/A/summary.json').read_text())
    assert analysis['completed'] == analysis['expected'] == 80 and not analysis['partial']
    shutil.copytree(root/'analysis/A', out/'analysis')
    (out/'provenance').mkdir()
    for name in ('manifest.json', 'manifest.sha256', 'probes.json', 'platform_comparison.json',
                 'launch_record.json', 'launch_verification.json', 'supervisor.json'):
        if (root/name).exists(): shutil.copy2(root/name, out/'provenance'/name)
    (out/'provenance/stage-a-jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    learning = []; final = []
    for j in jobs:
        d = root/j['directory']; cfg = j['config']
        identity = dict(platform=a.platform, stage='A', condition=j['cell'],
            algorithm=cfg['algorithm'], actor_layer_norm=cfg['actor_layer_norm'],
            critic_layer_norm=cfg['critic_layer_norm'], seed=cfg['seed'])
        for line in (d/'metrics.jsonl').read_text().splitlines():
            metric = json.loads(line)
            row = dict(identity, step=metric['step'], elapsed_seconds=metric['elapsed_seconds'],
                       critic_rounds=metric['critic_rounds'], actor_updates=metric['actor_updates'])
            for kind in ('on_policy', 'common_probe'):
                for key, value in metric[kind].items():
                    if isinstance(value, (int, float)) or value is None:
                        row[f'{kind}_{key}'] = value
            row['common_probe_analysis_rmse'] = math.sqrt(sum(e['rmse_q1']**2 for e in metric['common_probe']['episodes'])/len(metric['common_probe']['episodes']))
            learning.append(row)
        final.append(dict(identity, **json.loads((d/'summary.json').read_text())))
    write_csv(out/'learning_metrics.csv', learning)
    write_csv(out/'seed_summary.csv', final)
    write_csv(out/'conditions.csv', [dict(platform=a.platform, condition=name, **values)
                                    for name, values in analysis['conditions'].items()])

    label = f'{a.platform}-stage-a-{a.date}'
    release_tag = f'results-{label}'
    release_url = f'https://github.com/glouno/miniproject1-DDPG/releases/tag/{release_tag}'
    dataset_files = sorted(f for j in jobs for f in (root/j['directory']).rglob('*') if f.is_file())
    analysis_files = sorted(f for f in (root/'analysis/A').rglob('*') if f.is_file())
    raw_bytes = sum(f.stat().st_size for f in dataset_files+analysis_files)
    inventory = [dict(path=str(f.relative_to(root)), bytes=f.stat().st_size, sha256=digest(f))
                 for f in dataset_files+analysis_files]
    with gzip.open(out/'provenance/file_inventory.json.gz', 'wt') as stream:
        json.dump(inventory, stream, separators=(',', ':'))

    def archive(name, entries):
        destination = assets/f'{label}-{name}.tar.zst'
        with destination.open('wb') as stream:
            process = subprocess.Popen(['zstd', '-3', '-T2', '-q', '-c'], stdin=subprocess.PIPE, stdout=stream)
            try:
                with tarfile.open(fileobj=process.stdin, mode='w|') as tar:
                    for source, relative in entries:
                        tar.add(source, arcname=f'{label}/{relative}')
            finally:
                process.stdin.close()
            if process.wait() != 0: raise RuntimeError(f'Compression failed: {destination}')
        assert destination.stat().st_size < 2*1024**3, 'Split this archive before upload'
        subprocess.run(['zstd', '-q', '-t', str(destination)], check=True)
        record = dict(name=destination.name, bytes=destination.stat().st_size, sha256=digest(destination))
        print(json.dumps(record), flush=True)
        return record

    archives = []
    for cell in analysis['conditions']:
        selected = [j for j in jobs if j['cell'] == cell]
        entries = [(root/j['directory'], j['directory']) for j in selected]
        entries += [(root/'logs'/f"A-{j['cell']}-seed{j['config']['seed']}.log",
                     f"logs/A-{j['cell']}-seed{j['config']['seed']}.log") for j in selected]
        archives.append(archive(cell, entries))
    entries = [(root/name, name) for name in ('source', 'configs', 'manifest.json', 'manifest.sha256', 'probes.json', 'analysis/A')]
    entries += [(root/name, name) for name in ('platform_comparison.json', 'launch_record.json', 'launch_verification.json',
                                             'supervisor.json', 'machine_timing.jsonl') if (root/name).exists()]
    entries.append((out/'provenance/file_inventory.json.gz', 'file_inventory.json.gz'))
    archives.append(archive('provenance', entries))
    checksums = ''.join(f"{item['sha256']}  {item['name']}\n" for item in archives)
    (assets/'SHA256SUMS').write_text(checksums); (out/'SHA256SUMS').write_text(checksums)
    publication = dict(platform=a.platform, stage='A', label=label, release_url=release_url,
        exported_at=datetime.now(timezone.utc).isoformat(), scientific_preparation_revision=m['git_revision'],
        exporter_sha256=digest(Path(__file__).resolve()), completed_runs=80, training_steps=80_000_000,
        raw_stage_a_bytes=raw_bytes, compressed_archive_bytes=sum(x['bytes'] for x in archives),
        archives=archives, original_scope='Frozen manifest/configs describe A+B; uploaded run data contains A only')
    (out/'publication.json').write_text(json.dumps(publication, indent=2)+'\n')
    rows = '\n'.join(f"| {name} | {v['return_mean']:.1f} | {v['return_sd']:.1f} | {v['bias_mean']:.1f} |" for name,v in analysis['conditions'].items())
    readme = f'''# {a.platform.upper()} — Stage A — {a.date}

Platform-labelled results: **80 completed runs**, eight DDPG/TD3 × actor-LN ×
critic-LN conditions, ten training seeds (20–29) per condition, **1,000,000 steps
per run**. CPU, eight independent workers, one Torch/BLAS thread per worker.
Scientific preparation revision: `{m['git_revision']}`. Stage B is not included.

## Results

![Learning and on-policy calibration](analysis/learning.png)

![Common-probe calibration](analysis/calibration.png)

Final held-out episode returns and signed Q1-minus-Monte-Carlo bias:

| Condition (`a`: actor LN, `c`: critic LN) | Mean return | Return SD across seeds | Mean signed bias |
|---|---:|---:|---:|
{rows}

Each mean averages ten training seeds. Returns use 20 held-out evaluation episodes;
positive bias means overestimation. SD describes seed variation, not confidence
interval width. Plots show pointwise seed-bootstrap intervals. Prespecified paired
contrasts, bootstrap intervals and Holm-adjusted p-values are in
[analysis/summary.json](analysis/summary.json). These results do not establish that
bias causes poor performance.

## Included here

- [Learning/checkpoint scalar metrics](learning_metrics.csv): every saved checkpoint
  for every seed, explicitly labelled `{a.platform}`. `common_probe_analysis_rmse`
  matches the analysis's root-mean-square of episode RMSEs.
- [Per-seed outcomes and active timing](seed_summary.csv), [condition table](conditions.csv).
- PDF/PNG figures and complete aggregate statistical analysis under `analysis/`.
- Original manifest, probe bank, platform comparison and SHA-256 file inventory
  under `provenance/`. The original manifest retains the full planned A+B study;
  this publication exports only the 80 completed A jobs.

## Complete raw data on GitHub

[Download the **{a.platform.upper()} Stage A release**]({release_url}). Eight
condition archives each contain all ten seed directories, including raw Q/state/
action/return diagnostics, journals, inference checkpoints, latest/previous resume
states, TensorBoard records and job logs. A ninth archive contains frozen source,
locked environment/configs, analysis and provenance. No Stage B run data is uploaded.

Original Stage A runs plus analysis: **{raw_bytes:,} bytes**
({raw_bytes/1e9:.2f} GB / {raw_bytes/1024**3:.2f} GiB).
Compressed archives: **{publication['compressed_archive_bytes']:,} bytes**
({publication['compressed_archive_bytes']/1e9:.2f} GB).
These files are release assets rather than ordinary Git history. Repository and
release access remain private.

Download all nine archives and `SHA256SUMS` into one directory, then:

```sh
shasum -a 256 -c SHA256SUMS
for archive in {label}-*.tar.zst; do
  zstd -dc "$archive" | tar -xf -
done
```

They merge into `{label}/`. To regenerate the analysis without training:

```sh
cd {label}/source
/absolute/path/to/the/locked/.venv/bin/python -m lunarlander_rl.followup_analyse \\
  --input /absolute/path/to/{label} --stage A
```

Use the recorded package versions and frozen source; this is an analysis command,
not a training launch. Resume checkpoints are platform-specific. Mac Box2D probe
observations differ slightly from WSL; seeds/action prefixes match, while exact
physical replay is enforced within each platform. Do not pool these runs with
the WSL repetitions as additional independent training seeds.
'''
    (out/'README.md').write_text(readme)
    shutil.copy2(out/'publication.json', assets/'publication.json')
    print(json.dumps(publication), flush=True)


if __name__ == '__main__': main()
