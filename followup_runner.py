"""Immutable two-stage study queue. Preparation never launches training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import numpy as np

from experiment import Config
from study_diagnostics import atomic_json, build_probes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def jobs_for(preset):
    base=asdict(Config(max_steps=1_000_000, eval_interval=25000,early_eval_interval=5000,
        resume_interval=5000,eval_episodes=20,eval_seed_start=40000,diagnostic_horizon=10000,
        isolated_initialization=True,instrumentation=True))
    jobs=[]
    seeds=range(20,30) if preset=='full' else (9000,9001) if preset=='benchmark' else (9000,)
    for seed in seeds:
        for algo in ('ddpg','td3'):
            for actor_ln in (False,True):
                for critic_ln in (False,True):
                    cfg=dict(base,algorithm=algo,seed=seed,actor_layer_norm=actor_ln,
                             critic_layer_norm=critic_ln,layer_norm=critic_ln)
                    jobs.append(dict(stage='A',cell=f'{algo}-a{int(actor_ln)}-c{int(critic_ln)}',config=cfg))
    if preset!='benchmark':
        for seed in range(30,40) if preset=='full' else (9002,):
            for cln,clr,alr in [(True,c,a) for c in (3e-4,1e-4) for a in (3e-4,1e-4)]+[(False,r,r) for r in (3e-4,1e-4)]:
                cfg=dict(base,algorithm='ddpg',seed=seed,critic_layer_norm=cln,layer_norm=cln,critic_lr=clr,actor_lr=alr)
                cell=f'ddpg-a0-c{int(cln)}-clr{clr:g}-alr{alr:g}'
                jobs.append(dict(stage='B',cell=cell,config=cfg))
    if preset!='full':
        for job in jobs:
            job['config'].update(max_steps=15000 if preset=='benchmark' else 400,
                learning_starts=1000 if preset=='benchmark' else 100,
                eval_interval=5000 if preset=='benchmark' else 200,
                early_eval_interval=0,resume_interval=5000 if preset=='benchmark' else 100,
                eval_episodes=2,train_horizon=1000 if preset=='benchmark' else 100,
                diagnostic_horizon=10000 if preset=='benchmark' else 300,
                replay_capacity=100000 if preset=='benchmark' else 500,
                batch_size=128 if preset=='benchmark' else 32,
                actor_hidden=[128,128] if preset=='benchmark' else [16,16],
                critic_hidden=[128,128] if preset=='benchmark' else [16,16])
    for j in jobs:
        j['directory']=f"{j['stage']}/{j['cell']}-seed{j['config']['seed']}"
    return jobs


def prepare(out,preset,workers):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    base=Path(__file__).resolve().parent
    source=out/'source';source.mkdir();(out/'logs').mkdir()
    for p in base.glob('*.py'):
        shutil.copy2(p,source/p.name)
    lock=Path(sys.prefix).parent/'uv.lock'
    if not lock.exists(): lock=base/'environment/uv.lock'
    shutil.copy2(lock,source/'uv.lock')
    shutil.copy2(lock.with_name('pyproject.toml'),source/'pyproject.toml')
    shutil.copy2(base/'NEXT_STUDY_PROTOCOL.md',source/'NEXT_STUDY_PROTOCOL.md')
    jobs=jobs_for(preset)
    (out/'configs').mkdir()
    for index,job in enumerate(jobs):
        Config(**job['config']).validate()
        job['config_path']=f'configs/job-{index:03d}.json'
        atomic_json(out/job['config_path'],job['config'])
    cfg=Config(**jobs[0]['config'])
    bank=build_probes(cfg,24 if preset=='full' else 4 if preset=='benchmark' else 2,41000,30 if preset=='full' else 5)
    atomic_json(out/'probes.json',bank)
    files=[*source.iterdir(),out/'probes.json',*(out/'configs').iterdir()]
    manifest=dict(created_at=timestamp(),preset=preset,workers=workers,interpreter=sys.executable,
        versions={name:importlib.metadata.version(name) for name in ('rl-mind','torch','gymnasium','numpy','Box2D')},
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip(),
        stage_B_policy='Fixed before A; run B only after all A training jobs complete successfully',
        jobs=jobs,sha256={str(p.relative_to(out)):sha(p) for p in files},
        total_transitions=sum(j['config']['max_steps'] for j in jobs))
    atomic_json(out/'manifest.json',manifest)
    (out/'manifest.sha256').write_text(sha(out/'manifest.json')+'\n')
    print(f'PREPARED {len(jobs)} runs, {manifest["total_transitions"]:,} transitions at {out}',flush=True)
    return manifest


def verify(out):
    if sha(out/'manifest.json')!=(out/'manifest.sha256').read_text().strip():
        raise RuntimeError('Frozen manifest changed')
    m=json.loads((out/'manifest.json').read_text())
    for name,digest in m['sha256'].items():
        if sha(out/name)!=digest:raise RuntimeError(f'Frozen file changed: {name}')
    lock=Path(sys.prefix).parent/'uv.lock'
    if not lock.exists():lock=Path(__file__).parent/'environment/uv.lock'
    if sha(lock)!=m['sha256']['source/uv.lock']:raise RuntimeError('Active lock differs from frozen lock')
    for name,version in m['versions'].items():
        if importlib.metadata.version(name)!=version:raise RuntimeError(f'Active package changed: {name}')
    for job in m['jobs']:
        if job['config']!=json.loads((out/job['config_path']).read_text()):raise RuntimeError('Manifest/config mismatch')
    return m


def validate_completed(directory, config):
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='complete' or summary['transitions']!=config['max_steps']:
        raise RuntimeError(f'Incomplete summary: {directory}')
    metadata=json.loads((directory/'metadata.json').read_text())
    if metadata['config']!=config:raise RuntimeError(f'Configuration mismatch: {directory}')
    metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    early=config['early_eval_interval'];until=config['early_eval_until'] if early else 0
    expected=sorted({0,config['max_steps'],
        *(range(early,min(until,config['max_steps'])+1,early) if early else ()),
        *(step for step in range(config['eval_interval'],config['max_steps']+1,config['eval_interval']) if step>until)})
    if [m['step'] for m in metrics]!=expected:raise RuntimeError(f'Checkpoint coverage mismatch: {directory}')
    for metric in metrics:
        step=metric['step']
        if not (directory/f'checkpoint-{step:07d}.pt').exists():raise RuntimeError('Missing inference checkpoint')
        if config['instrumentation']:
            json.loads((directory/'audits'/f'audit-{step:07d}.json').read_text())
        for kind in ('on_policy','common_probe'):
            with np.load(directory/'diagnostics'/f'{kind}-{step:07d}.npz') as pairs:
                for key in pairs.files:
                    if not np.isfinite(pairs[key]).all():raise RuntimeError(f'Nonfinite {key}: {directory}')
                if len(pairs['q'])!=metric[kind]['sampled_pairs']:raise RuntimeError('Pair count mismatch')
    atomic_json(directory/'completion_validation.json',dict(status='valid',checked_at=timestamp(),
        checkpoints=len(metrics),transitions=summary['transitions']))


def run_job(out,job,stop,processes,guard,min_free_gib):
    name=job['directory'];directory=out/name
    if (directory/'summary.json').exists():
        validate_completed(directory,job['config'])
        return dict(status='complete',reused=True)
    if stop.is_set():return dict(status='pending')
    if shutil.disk_usage(out).free<min_free_gib*2**30:
        stop.set();return dict(status='blocked_disk_space')
    command=[sys.executable,str(out/'source/followup_runner.py'),'--worker','--output',str(out),
             '--job',name]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    log_path=out/'logs'/f'{name.replace("/","-")}.log'
    with log_path.open('a') as log:
        log.write(f'\nSTART {timestamp()}\n');log.flush()
        with guard:
            if stop.is_set():return dict(status='pending')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            processes[name]=process
        code=process.wait()
        with guard:processes.pop(name,None)
    completed=(directory/'summary.json').exists() and code==0
    if completed:validate_completed(directory,job['config'])
    return dict(status='complete' if completed else 'paused' if stop.is_set() else 'failed',
                exit_code=code,log=str(log_path))


def worker(out,name):
    from study_train import train_study
    m=verify(out);job=next(j for j in m['jobs'] if j['directory']==name)
    cfg=Config(**job['config']);directory=out/name
    bank=json.loads((out/'probes.json').read_text())
    horizons=(2500,5000,10000) if cfg.diagnostic_horizon==10000 else (150,300)
    restoring=(directory/'resume.pt').exists() or (directory/'resume.previous.pt').exists()
    train_study(cfg,directory,bank,horizons= horizons,resume=restoring)


def run(out,workers=None,min_free_gib=20):
    # A process lock prevents two services/shells from running the same queue.
    import fcntl
    lock_handle=(out/'runner.lock').open('a')
    fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=verify(out);workers=workers or m['workers']
    stop=threading.Event();guard=threading.Lock();processes={}
    def interrupt(signum,frame):
        stop.set()
        with guard:
            for p in processes.values():p.terminate()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    status=dict(status='running',started_at=timestamp(),runner_pid=os.getpid(),workers=workers,jobs={})
    # Restore known statuses; actual summary/resume files are authoritative.
    if (out/'status.json').exists():status['jobs']=json.loads((out/'status.json').read_text()).get('jobs',{})
    start=time.monotonic()
    for stage in ('A','B'):
        selected=[j for j in m['jobs'] if j['stage']==stage]
        if not selected:continue
        stage_ok=True;status['stage']=stage
        atomic_json(out/'status.json',status)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(run_job,out,j,stop,processes,guard,min_free_gib):j for j in selected}
            for f in as_completed(pending):
                job=pending[f]
                try: result=f.result()
                except Exception as error:result=dict(status='failed',error=repr(error))
                status['jobs'][job['directory']]=result
                stage_ok &= result['status']=='complete'
                status.update(updated_at=timestamp(),elapsed_seconds=time.monotonic()-start)
                atomic_json(out/'status.json',status)
                print(job['directory'],result['status'],flush=True)
        if not stage_ok or stop.is_set():
            status['status']='paused' if stop.is_set() else 'requires_attention'
            atomic_json(out/'status.json',status);return 1
        subprocess.run([sys.executable,str(out/'source/followup_analyse.py'),'--input',str(out),'--stage',stage],check=True)
    status.update(status='complete',elapsed_seconds=time.monotonic()-start,finished_at=timestamp())
    atomic_json(out/'status.json',status)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--preset',choices=['full','smoke','benchmark'],default='full')
    p.add_argument('--workers',type=int);p.add_argument('--min-free-gib',type=float,default=20)
    p.add_argument('--worker',action='store_true');p.add_argument('--job')
    a=p.parse_args();out=a.output.resolve()
    if a.workers is not None and a.workers<1:p.error('workers must be positive')
    if a.prepare:prepare(out,a.preset,a.workers or 8)
    elif a.worker:worker(out,a.job)
    else:sys.exit(run(out,a.workers,a.min_free_gib))


if __name__=='__main__':main()
