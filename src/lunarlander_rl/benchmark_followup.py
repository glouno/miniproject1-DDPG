"""Sequential worker-count measurements of the instrumented frozen workload."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from .followup_runner import prepare
from .study_diagnostics import atomic_json
from .paths import frozen_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,nargs='+',default=[4,8,12,16])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    manifests={workers:prepare(a.output/f'workers-{workers}','benchmark',workers) for workers in a.workers}
    for workers in a.workers:
        out=a.output/f'workers-{workers}'
        manifest=manifests[workers]
        start=time.monotonic()
        subprocess.run(frozen_command(out/'source','followup_runner')+['--output',str(out)],check=True,cwd=out/'source')
        wall=time.monotonic()-start
        summaries=[json.loads((out/j['directory']/'summary.json').read_text()) for j in manifest['jobs']]
        size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        result=dict(workers=workers,wall_seconds=wall,transitions_per_second=manifest['total_transitions']/wall,
            bytes=size,mean_run_seconds=sum(s['elapsed_seconds'] for s in summaries)/len(summaries),
            mean_evaluation_seconds=sum(s['checkpoint_seconds'] for s in summaries)/len(summaries))
        results.append(result);atomic_json(a.output/'benchmark.json',dict(batches=results,
            note='One short instrumented batch per worker count; warmup/imports and analysis included. Not a full-study runtime guarantee.'))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
