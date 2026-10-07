"""Install an inactive user service for an already-frozen queue; never start it."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
from .followup_runner import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--name',default='rl-ln-followup-20261007');a=p.parse_args()
    if sys.platform!='linux':p.error('Run this installer on WSL/Linux')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.name):p.error('Invalid service name')
    out=a.output.resolve();manifest=verify(out)
    cpu_environment=' CUDA_VISIBLE_DEVICES=' if all(j['config']['device']=='cpu' for j in manifest['jobs']) else ''
    if any(c.isspace() for c in str(out)+sys.executable):p.error('Service paths must contain no whitespace')
    unit=Path.home()/'.config/systemd/user'/f'{a.name}.service';unit.parent.mkdir(parents=True,exist_ok=True)
    if unit.exists():p.error(f'Existing service preserved: {unit}')
    unit.write_text(f'''[Unit]
Description=Frozen RL actor/critic LayerNorm follow-up study

[Service]
Type=simple
WorkingDirectory={out}/source
ExecStart={sys.executable} -m lunarlander_rl.followup_runner --output {out}
Environment=OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1{cpu_environment}
KillMode=mixed
TimeoutStopSec=180
Restart=no
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
''')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    print(f'Installed inactive unit: {unit}; start explicitly when ready.')


if __name__=='__main__':main()
