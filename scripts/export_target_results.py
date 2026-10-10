"""Publish completed WSL target interventions, including verified raw archives."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import shutil
import tarfile
from export_wsl_study import digest, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    root, out, assets = (p.resolve() for p in (args.input,args.output,args.assets))
    m = json.loads((root/'manifest.json').read_text())
    assert m['platform']=='wsl' and m['host']=='paris-devbox-wsl'
    assert digest(root/'manifest.json')==(root/'manifest.sha256').read_text().strip()
    for name, expected in m['sha256'].items(): assert digest(root/name)==expected, name
    status = json.loads((root/'status.json').read_text())
    assert status['status']=='complete' and len(m['jobs'])==20
    analysis = json.loads((root/'analysis/summary.json').read_text())
    assert analysis['platform']=='wsl' and analysis['completed']==analysis['expected']==20 and not analysis['partial']
    out.mkdir(parents=True,exist_ok=False); assets.mkdir(parents=True,exist_ok=False)
    shutil.copytree(root/'analysis',out/'analysis')
    (out/'provenance').mkdir()
    for name in ('manifest.json','manifest.sha256','status.json','probes.json',
                 'baseline_manifest.json','launch_verification.json'):
        shutil.copy2(root/name,out/'provenance'/name)
    learning, final = [], []
    for job in m['jobs']:
        d = root/job['directory']; c = job['config']
        s = json.loads((d/'summary.json').read_text())
        assert s['status']=='complete' and s['transitions']==1_000_000
        assert json.loads((d/'completion_validation.json').read_text())['status']=='valid'
        assert json.loads((d/'metadata.json').read_text())['config']==c
        identity = dict(platform='wsl',stage='T',condition=job['cell'],seed=c['seed'],
            algorithm=c['algorithm'],critic_layer_norm=c['critic_layer_norm'],
            actor_layer_norm=c['actor_layer_norm'],target_mode='hard' if c['hard_target_interval'] else 'soft',
            tau=c['tau'],hard_target_interval=c['hard_target_interval'])
        metrics = [json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]
        assert len(metrics)==49 and metrics[-1]['step']==1_000_000
        for metric in metrics:
            assert (d/f"checkpoint-{metric['step']:07d}.pt").is_file()
            row = dict(identity,step=metric['step'],elapsed_seconds=metric['elapsed_seconds'])
            for kind in ('on_policy','common_probe'):
                row.update({f'{kind}_{k}':v for k,v in metric[kind].items()
                            if isinstance(v,(int,float)) or v is None})
            row['common_probe_analysis_rmse']=math.sqrt(sum(e['rmse_q1']**2 for e in metric['common_probe']['episodes'])/len(metric['common_probe']['episodes']))
            learning.append(row)
        final.append(dict(identity,**s))
    write_csv(out/'learning_metrics.csv',learning)
    write_csv(out/'seed_summary.csv',final)
    write_csv(out/'conditions.csv',[dict(platform='wsl',condition=k,**v) for k,v in analysis['conditions'].items()])
    for name in ('baseline_seed_summary.csv','baseline_learning_metrics.csv'):
        shutil.copy2(root/name,out/name)
    files = sorted(f for f in root.rglob('*') if f.is_file() and f.name!='runner.lock')
    inventory = {str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=digest(f)) for f in files}
    with gzip.open(out/'provenance/file_inventory.json.gz','wt') as f:json.dump(inventory,f,separators=(',',':'))
    label = 'wsl-target-updates-20261010'
    archives, verified, assigned = [], set(), set()
    selections=[]
    for cell in sorted({j['cell'] for j in m['jobs']}):
        prefixes=[j['directory']+'/' for j in m['jobs'] if j['cell']==cell]
        logs={f"logs/{j['directory'].replace('/','-')}.log" for j in m['jobs'] if j['cell']==cell}
        selected=[f for f in files if any(str(f.relative_to(root)).startswith(p) for p in prefixes) or str(f.relative_to(root)) in logs]
        assigned.update(selected); selections.append((cell,selected))
    selections.append(('provenance',[f for f in files if f not in assigned]))
    for name, selected in selections:
        dest=assets/f'{label}-{name}.tar.gz'
        with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
            for f in selected:tar.add(f,arcname=f'{label}/{f.relative_to(root)}',recursive=False)
        assert dest.stat().st_size<2*1024**3
        with tarfile.open(dest,'r:gz') as tar:
            import hashlib
            for member in tar:
                relative=str(Path(member.name).relative_to(label)); h=hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                assert member.size==inventory[relative]['bytes'] and h.hexdigest()==inventory[relative]['sha256']
                assert relative not in verified; verified.add(relative)
        archives.append(dict(name=dest.name,bytes=dest.stat().st_size,sha256=digest(dest)))
        print('Verified '+dest.name,flush=True)
    assert verified==set(inventory)
    model_count=sum(Path(n).name.startswith('checkpoint-') and n.endswith('.pt') for n in inventory)
    recovery_count=sum(Path(n).name in ('resume.pt','resume.previous.pt') for n in inventory)
    assert model_count==980 and recovery_count==40
    checksums=''.join(f"{a['sha256']}  {a['name']}\n" for a in archives)
    for directory in (out,assets):(directory/'SHA256SUMS').write_text(checksums)
    publication=dict(platform='wsl',label=label,release_url=f'https://github.com/glouno/miniproject1-DDPG/releases/tag/results-{label}',
        completed_runs=20,training_steps=20_000_000,model_checkpoints=model_count,recovery_files=recovery_count,
        source_revision=m['git_revision'],finished_at=status['finished_at'],exported_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=sum(v['bytes'] for v in inventory.values()),compressed_archive_bytes=sum(a['bytes'] for a in archives),
        archives=archives,exporter_sha256=digest(Path(__file__).resolve()))
    write_json(out/'publication.json',publication);write_json(assets/'publication.json',publication)
    write_json(out/'verification.json',dict(platform='wsl',status='passed',completed_runs=20,checkpoint_rows=len(learning),
        archived_files=len(verified),model_checkpoints=model_count,recovery_files=recovery_count,
        method='Frozen source/config/probe hashes and completion validations checked; every archived file read back and matched against original size and SHA-256.'))


if __name__=='__main__':main()
