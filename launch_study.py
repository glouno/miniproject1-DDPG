"""Launch a prepared study as a supervised macOS job with idle-sleep inhibition."""
import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--label',required=True)
    args=parser.parse_args();root=args.output.resolve()
    assert sys.platform=='darwin' and (root/'manifest.json').exists()
    assert args.label.startswith('org.glouno.rl.') and '/' not in args.label
    agent=Path.home()/'Library/LaunchAgents'/f'{args.label}.plist'
    agent.parent.mkdir(parents=True,exist_ok=True)
    if agent.exists():
        raise FileExistsError(f'Existing job: {agent}; use launchctl kickstart to resume it')
    plist=dict(Label=args.label,ProgramArguments=['/usr/bin/caffeinate','-i','-s',sys.executable,'-u',
                str(root/'source/study_runner.py'),'--output',str(root)],
               WorkingDirectory=str(Path(__file__).resolve().parent.parent),
               RunAtLoad=False,KeepAlive=False,ProcessType='Standard',
               StandardOutPath=str(root/'supervisor.log'),StandardErrorPath=str(root/'supervisor-error.log'),
               EnvironmentVariables=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                         PYTHONUNBUFFERED='1',MPLBACKEND='Agg'))
    with agent.open('xb') as f:
        plistlib.dump(plist,f)
    domain=f'gui/{os.getuid()}'
    subprocess.run(['/bin/launchctl','bootstrap',domain,str(agent)],check=True)
    subprocess.run(['/bin/launchctl','kickstart',f'{domain}/{args.label}'],check=True)
    print('LAUNCHED',args.label,'configuration',agent,flush=True)


if __name__=='__main__':
    main()
