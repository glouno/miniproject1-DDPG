"""Read saved study progress without importing Torch or disturbing training."""
import argparse
import json
from pathlib import Path
from .study_completion import summary_error


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    root=args.input;manifest=json.loads((root/'manifest.json').read_text())
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    completed=0;total_steps=0;active=[];unresolved=[]
    for job in manifest['jobs']:
        directory=root/job['directory']
        progress=json.loads((directory/'progress.json').read_text()) if (directory/'progress.json').exists() else {}
        reason=summary_error(directory,manifest['config']['max_steps'])
        completed+=int(reason is None);total_steps+=progress.get('step',0)
        if reason is not None:
            unresolved.append(dict(run=job['directory'],reason=reason))
        entry=status.get('jobs',{}).get(job['directory'],{})
        if entry.get('status')=='running':
            active.append(dict(run=job['directory'],pid=entry.get('pid'),step=progress.get('step',0)))
    print(json.dumps(dict(status=status.get('status','starting'),completed=completed,expected_runs=manifest['expected_runs'],
        saved_progress_transitions=total_steps,planned_transitions=manifest['expected_runs']*manifest['config']['max_steps'],
        active=active,unresolved_summaries=unresolved,analysis=str(root/'analysis/results.md')),indent=2))


if __name__=='__main__':
    main()
