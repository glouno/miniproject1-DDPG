"""Freeze and run a bounded, resumable study; safe across Codex disconnection."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from experiment import Config
from study_diagnostics import atomic_json, build_probes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out, config, seeds, workers, max_hours, probes_count=12, prefix_steps=30):
    out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir();(out/'logs').mkdir()
    base=Path(__file__).resolve().parent
    for p in base.glob('*.py'):
        shutil.copy2(p,source/p.name)
    settings=json.loads(config.read_text());cfg=Config(**settings);cfg.validate();assert cfg.device=='cpu'
    shutil.copy2(config,out/'config.json')
    (source/'environment').mkdir();lock=Path(sys.prefix).parent/'uv.lock';assert lock.exists()
    shutil.copy2(lock,source/'environment/uv.lock')
    for p in (base/'STUDY_PROTOCOL.md',base.parent/'pyproject.toml'):
        shutil.copy2(p,out/p.name)
    atomic_json(out/'probes.json',build_probes(cfg,probes_count,30000,prefix_steps))
    jobs=[dict(algorithm=a,layer_norm=bool(ln),seed=s,directory=f'{a}-ln{ln}-seed{s}')
          for s in seeds for a in ('ddpg','td3') for ln in (0,1)]
    manifest=dict(git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=base).strip(),stage='frozen main study',created_at=datetime.now(timezone.utc).isoformat(),
                  seeds=seeds,workers=workers,max_hours=max_hours,expected_runs=len(jobs),config=settings,jobs=jobs,
                  source_sha256={p.name:sha(p) for p in source.glob('*.py')},lock_sha256=sha(lock),
                  config_sha256=sha(out/'config.json'),probes_sha256=sha(out/'probes.json'),
                  interpreter=sys.executable,protocol_sha256=sha(out/'STUDY_PROTOCOL.md'))
    atomic_json(out/'manifest.json',manifest)
    print('PREPARED',out,'runs',len(jobs),'transitions',cfg.max_steps*len(jobs),flush=True)


def run(out):
    manifest=json.loads((out/'manifest.json').read_text())
    source=out/'source'
    for name,h in manifest['source_sha256'].items():
        assert sha(source/name)==h, f'Frozen source changed: {name}'
    assert sha(out/'config.json')==manifest['config_sha256']
    assert sha(out/'probes.json')==manifest['probes_sha256']
    assert sha(Path(sys.prefix).parent/'uv.lock')==manifest['lock_sha256']
    started=time.monotonic();deadline=started+manifest['max_hours']*3600
    status=dict(status='running',runner_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),jobs={})
    lock=threading.Lock()
    def update(name,value):
        with lock:
            status['jobs'][name]=value
            status['elapsed_seconds']=time.monotonic()-started
            atomic_json(out/'status.json',status)
    def launch(job):
        name=job['directory'];directory=out/name
        if (directory/'summary.json').exists():
            summary=json.loads((directory/'summary.json').read_text())
            assert summary['status']=='complete' and summary['transitions']==manifest['config']['max_steps']
            update(name,dict(status='complete',reused=True));return name,True
        if time.monotonic()>=deadline:
            update(name,dict(status='not_started_deadline'));return name,False
        command=[sys.executable,str(source/'study_train.py'),'--config',str(out/'config.json'),
                 '--probes',str(out/'probes.json'),'--algorithm',job['algorithm'],
                 '--layer-norm',str(int(job['layer_norm'])),'--seed',str(job['seed']),'--output',str(directory)]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for attempt in (1,2):
            cmd=command+(['--resume'] if (directory/'resume.pt').exists() or (directory/'resume.previous.pt').exists() else [])
            with (out/'logs'/f'{name}.log').open('a') as log:
                log.write(f'\nATTEMPT {attempt} {datetime.now(timezone.utc).isoformat()}\n');log.flush()
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                update(name,dict(status='running',pid=process.pid,attempt=attempt))
                print('START',name,'PID',process.pid,flush=True)
                try:
                    code=process.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait()
                    update(name,dict(status='paused_deadline'));return name,False
            if code==0 and (directory/'summary.json').exists():
                update(name,dict(status='complete'));print('COMPLETE',name,flush=True);return name,True
            if time.monotonic()>=deadline:
                break
            if not ((directory/'resume.pt').exists() or (directory/'resume.previous.pt').exists()):
                break
        update(name,dict(status='failed',exit_code=code));print('FAILED',name,code,flush=True);return name,False
    with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
        results=[future.result() for future in as_completed([pool.submit(launch,j) for j in manifest['jobs']])]
    successful=all(ok for _,ok in results)
    status['status']='training_complete' if successful else 'incomplete_requires_attention'
    status['elapsed_seconds']=time.monotonic()-started
    atomic_json(out/'status.json',status)
    if successful:
        result=subprocess.run([sys.executable,str(source/'study_analyse.py'),'--input',str(out)],env=dict(os.environ,MPLBACKEND='Agg'))
        status['status']='complete' if result.returncode==0 else 'training_complete_analysis_failed'
        atomic_json(out/'status.json',status)
    print('STUDY STATUS',status['status'],flush=True)
    return 0 if status['status']=='complete' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10,20)))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-hours',type=float,default=5)
    parser.add_argument('--probe-count',type=int,default=12)
    parser.add_argument('--prefix-steps',type=int,default=30)
    args=parser.parse_args();out=args.output.resolve()
    assert args.workers>0 and args.max_hours>0 and len(set(args.seeds))==len(args.seeds)
    if args.prepare:
        prepare(out,args.config.resolve(),args.seeds,args.workers,args.max_hours,args.probe_count,args.prefix_steps)
    else:
        sys.exit(run(out))


if __name__=='__main__':
    main()
