"""Sequential worker-count measurements of the instrumented frozen workload."""
import argparse
import copy
import json
import subprocess
import sys
import time
import threading
import statistics
from pathlib import Path
from .followup_runner import prepare, jobs_for
from .study_diagnostics import atomic_json
from .paths import frozen_command


def timing_jobs(count=16,steps=15000,full_evaluation=False):
    if count<8 or count%8:raise ValueError('jobs must be a positive multiple of eight')
    if steps<=1000:raise ValueError('steps must exceed warmup')
    template=jobs_for('benchmark')[:8];jobs=[]
    for seed in range(9000,9000+count//8):
        for original in template:
            job=copy.deepcopy(original);cfg=job['config']
            cfg.update(seed=seed,max_steps=steps,eval_interval=steps,resume_interval=5000)
            if full_evaluation:cfg['eval_episodes']=20
            job['directory']=f"A/{job['cell']}-seed{seed}"
            jobs.append(job)
    return jobs


def measure(command,cwd):
    import psutil
    samples=[];stop=threading.Event()
    def observe():
        psutil.cpu_percent()
        while not stop.wait(1):
            vm=psutil.virtual_memory();swap=psutil.swap_memory()
            children=psutil.Process().children(recursive=True)
            rss=0
            for child in children:
                try:rss+=child.memory_info().rss
                except psutil.Error:pass
            samples.append(dict(cpu_percent=psutil.cpu_percent(),available_bytes=vm.available,
                swap_used_bytes=swap.used,swap_in_bytes=swap.sin,swap_out_bytes=swap.sout,child_rss_sum=rss))
    thread=threading.Thread(target=observe,daemon=True);thread.start();start=time.monotonic()
    try:subprocess.run(command,check=True,cwd=cwd)
    finally:stop.set();thread.join()
    wall=time.monotonic()-start
    return wall,dict(samples=samples,mean_cpu_percent=statistics.mean(s['cpu_percent'] for s in samples) if samples else None,
        minimum_available_gib=min((s['available_bytes']/2**30 for s in samples),default=None),
        peak_child_rss_sum_gib=max((s['child_rss_sum']/2**30 for s in samples),default=None),
        swap_increase_bytes=samples[-1]['swap_used_bytes']-samples[0]['swap_used_bytes'] if samples else None,
        swap_out_bytes=samples[-1]['swap_out_bytes']-samples[0]['swap_out_bytes'] if samples else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,nargs='+',default=[4,8,12,16])
    p.add_argument('--jobs',type=int,default=16);p.add_argument('--steps',type=int,default=15000)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--full-evaluation',action='store_true')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    if a.repeats<1 or any(w<1 for w in a.workers) or len(set(a.workers))!=len(a.workers):p.error('Invalid workers/repeats')
    try:jobs=timing_jobs(a.jobs,a.steps,a.full_evaluation)
    except ValueError as error:p.error(str(error))
    results=[]
    batches=[(repeat,workers) for repeat in range(a.repeats)
             for workers in (a.workers if repeat%2==0 else list(reversed(a.workers)))]
    manifests={(repeat,workers):prepare(a.output/f'repeat-{repeat}-workers-{workers}','benchmark',workers,benchmark_jobs=copy.deepcopy(jobs))
               for repeat,workers in batches}
    for repeat,workers in batches:
        out=a.output/f'repeat-{repeat}-workers-{workers}';manifest=manifests[repeat,workers]
        wall,resources=measure(frozen_command(out/'source','followup_runner')+['--output',str(out)],out/'source')
        summaries=[json.loads((out/j['directory']/'summary.json').read_text()) for j in manifest['jobs']]
        size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        atomic_json(out/'resources.json',resources)
        result=dict(repeat=repeat,workers=workers,wall_seconds=wall,transitions_per_second=manifest['total_transitions']/wall,
            resources={key:value for key,value in resources.items() if key!='samples'},
            bytes=size,mean_run_seconds=sum(s['elapsed_seconds'] for s in summaries)/len(summaries),
            mean_evaluation_seconds=sum(s['checkpoint_seconds'] for s in summaries)/len(summaries))
        results.append(result);atomic_json(a.output/'benchmark.json',dict(batches=results,
            jobs=a.jobs,steps=a.steps,full_evaluation=a.full_evaluation,
            note='All inputs frozen before timing. Identical seed/config jobs at each count; repeated order reversed. Startup, diagnostics and validation included. RSS sum double-counts shared mappings. Not a full-study runtime guarantee.'))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
