"""Frozen Paris WSL DDPG target-update interventions and baseline comparisons."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from .experiment import Config
from .focused_runner import run as run_queue
from .followup_runner import sha, timestamp, verify
from .paths import dependency_lock, project_root, snapshot_source
from .study_diagnostics import atomic_json, build_probes


def jobs_for(preset='full'):
    original = json.loads((project_root() /
        'results/mac/stage-a-20261008/provenance/manifest.json').read_text())
    reference = next(j['config'] for j in original['jobs']
                     if j['stage'] == 'A' and j['cell'] == 'ddpg-a0-c1')
    jobs = []
    for seed in range(20, 30) if preset == 'full' else (9000,):
        for cell, tau, interval in (('soft-tau-0.001', .001, 0), ('hard-k400', .005, 400)):
            cfg = asdict(Config(**dict(reference, seed=seed, tau=tau,
                                      hard_target_interval=interval)))
            if preset == 'smoke':
                cfg.update(max_steps=1100, learning_starts=100, batch_size=32,
                           replay_capacity=1200, actor_hidden=[16,16], critic_hidden=[16,16],
                           eval_interval=1100, early_eval_interval=0, eval_episodes=1,
                           train_horizon=100, diagnostic_horizon=300, resume_interval=250)
            Config(**cfg).validate()
            jobs.append(dict(stage='T', cell=cell, config=cfg,
                             directory=f'T/{cell}-seed{seed}'))
    return jobs


def prepare(out, workers=8, preset='full'):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = project_root()
    source = out / 'source'
    source.mkdir(); (out / 'logs').mkdir(); (out / 'configs').mkdir()
    snapshot_source(source)
    for path in (dependency_lock(), root/'pyproject.toml', root/'docs/TARGET_UPDATE_PROTOCOL.md'):
        shutil.copy2(path, source/path.name)
    baseline = root/'results/wsl/stages-a-b-20261008'
    shutil.copy2(baseline/'seed_summary.csv', out/'baseline_seed_summary.csv')
    shutil.copy2(baseline/'learning_metrics.csv', out/'baseline_learning_metrics.csv')
    shutil.copy2(baseline/'provenance/manifest.json', out/'baseline_manifest.json')
    jobs = jobs_for(preset)
    for index, job in enumerate(jobs):
        job['config_path'] = f'configs/job-{index:03d}.json'
        atomic_json(out/job['config_path'], job['config'])
    cfg = Config(**jobs[0]['config'])
    bank = build_probes(cfg, 24 if preset == 'full' else 2, 41000, 30 if preset == 'full' else 5)
    if preset == 'full':
        original = json.loads((baseline/'provenance/probes.json').read_text())
        assert bank == original, 'Native WSL probes differ from the baseline'
    atomic_json(out/'probes.json', bank)
    files = [p for p in out.rglob('*') if p.is_file()]
    manifest = dict(created_at=timestamp(), preset=preset, workers=workers,
        interpreter=sys.executable, platform='wsl', host=socket.gethostname(),
        study='Exploratory DDPG target updates; WSL Stage A baseline reused',
        versions={n:importlib.metadata.version(n) for n in ('rl-mind','torch','gymnasium','numpy','Box2D')},
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
        jobs=jobs, total_transitions=sum(j['config']['max_steps'] for j in jobs),
        baseline_platform='wsl', baseline_condition='A/ddpg-a0-c1', baseline_reused=True,
        target_schedule='Both targets after online updates: soft tau=.001 every round OR exact copy every 400 critic rounds; tau unused in hard mode',
        sha256={str(p.relative_to(out)):sha(p) for p in files})
    atomic_json(out/'manifest.json', manifest)
    (out/'manifest.sha256').write_text(sha(out/'manifest.json')+'\n')
    print(f'PREPARED {len(jobs)} runs / {manifest["total_transitions"]:,} transitions at {out}', flush=True)
    return manifest


def analyse(out):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .followup_analyse import paired, holm
    manifest = verify(out)
    groups = {}
    baseline = {}
    with (out/'baseline_learning_metrics.csv').open() as f:
        for row in csv.DictReader(f):
            if row['platform']=='wsl' and row['stage']=='A' and row['condition']=='ddpg-a0-c1':
                baseline.setdefault(int(row['seed']), []).append(dict(
                    step=int(row['step']), **{k:float(row[k]) for k in
                    ('on_policy_return','on_policy_bias_q1','on_policy_rmse_q1',
                     'common_probe_bias_q1','common_probe_analysis_rmse')}))
    groups['baseline-soft-tau-0.005'] = baseline
    for job in manifest['jobs']:
        directory = out/job['directory']
        if not (directory/'summary.json').exists():
            continue
        records = []
        for line in (directory/'metrics.jsonl').read_text().splitlines():
            m = json.loads(line)
            records.append(dict(step=m['step'], on_policy_return=m['on_policy']['return'],
                on_policy_bias_q1=m['on_policy']['bias_q1'], on_policy_rmse_q1=m['on_policy']['rmse_q1'],
                common_probe_bias_q1=m['common_probe']['bias_q1'],
                common_probe_analysis_rmse=float(np.sqrt(np.mean([e['rmse_q1']**2 for e in m['common_probe']['episodes']])))))
        groups.setdefault(job['cell'], {})[job['config']['seed']] = records
    destination = out/'analysis'; destination.mkdir(exist_ok=True)
    conditions = {}
    for name, runs in groups.items():
        values = [r[-1]['on_policy_return'] for r in runs.values()]
        conditions[name] = dict(n=len(values), return_mean=float(np.mean(values)),
            return_sd=float(np.std(values, ddof=1)) if len(values)>1 else None,
            bias_mean=float(np.mean([r[-1]['on_policy_bias_q1'] for r in runs.values()])))
    for figure, keys in [('learning', ('on_policy_return','on_policy_bias_q1','on_policy_rmse_q1')),
                         ('calibration', ('common_probe_bias_q1','common_probe_analysis_rmse'))]:
        fig, axes = plt.subplots(1,3, figsize=(15,4))
        for name, runs in groups.items():
            for ax, key in zip(axes, keys):
                steps = [r['step'] for r in next(iter(runs.values()))]
                values = np.array([[r[key] for r in run] for run in runs.values()])
                line = ax.plot(steps, values.mean(axis=0), label=f'{name} (n={len(runs)})')[0]
                idx = np.random.default_rng(419).integers(0,len(runs),(10000,len(runs)))
                lo,hi = np.quantile(values[idx].mean(axis=1), [.025,.975], axis=0)
                ax.fill_between(steps,lo,hi,color=line.get_color(),alpha=.15)
                ax.set(title=key, xlabel='Training transitions'); ax.grid(alpha=.2)
            if figure=='calibration':
                axes[2].scatter([r[-1]['on_policy_bias_q1'] for r in runs.values()],
                                [r[-1]['on_policy_return'] for r in runs.values()], label=name)
        axes[0].legend(fontsize=7)
        if figure=='calibration':
            axes[2].set(xlabel='Final signed Q−MC bias',ylabel='Final return',title='One point per seed; association only')
        fig.suptitle('WSL (Paris) — DDPG critic LN on / actor LN off — target-update interventions')
        fig.tight_layout()
        for extension in ('png','pdf'):
            fig.savefig(destination/f'{figure}.{extension}',dpi=150)
        plt.close(fig)
    contrasts = {}
    for name, runs in groups.items():
        if name=='baseline-soft-tau-0.005': continue
        seeds = sorted(set(runs)&set(baseline))
        if seeds:
            contrasts[name] = paired([runs[s][-1]['on_policy_return']-baseline[s][-1]['on_policy_return'] for s in seeds])
    holm(contrasts)
    completed = sum(len(r) for name,r in groups.items() if name!='baseline-soft-tau-0.005')
    atomic_json(destination/'summary.json',dict(platform='wsl',completed=completed,
        expected=len(manifest['jobs']),partial=completed!=len(manifest['jobs']),
        conditions=conditions,exploratory_paired_return_contrasts=contrasts,
        note='Historical WSL baseline reused; exploratory paired comparisons, Holm correction across two contrasts. Seeds are units; bootstrap bands pointwise.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--analyse', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--preset', choices=('full','smoke'), default='full')
    args = parser.parse_args()
    if args.workers < 1: parser.error('Workers must be positive')
    out = args.output.resolve()
    if args.prepare: prepare(out,args.workers,args.preset)
    elif args.analyse: analyse(out)
    else:
        code = run_queue(out, args.workers, min_free_gib=20)
        if not code: analyse(out)
        sys.exit(code)


if __name__ == '__main__': main()
