"""Offline, prespecified trajectory examples and action maps; no training changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .mechanism_diagnostics import action_maps


def examples(root,stage,maps=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=json.loads((root/'manifest.json').read_text());bank=json.loads((root/'probes.json').read_text())
    seeds=(20,21,22) if stage=='A' else (30,31,32)
    for job in m['jobs']:
        if job['stage']!=stage or job['config']['seed'] not in seeds:continue
        directory=root/job['directory']
        for step in (50_000,200_000,500_000,1_000_000):
            checkpoint=directory/f'checkpoint-{step:07d}.pt'
            if not checkpoint.exists():continue
            out=root/'examples'/job['directory']/f'{step:07d}';out.mkdir(parents=True,exist_ok=True)
            with np.load(directory/'diagnostics'/f'on_policy-{step:07d}.npz') as pairs:
                stop=int(pairs['trajectory_offsets'][1]);obs=pairs['trajectory_obs'][:stop]
                actions=pairs['trajectory_actions'][:stop]
                fig,axes=plt.subplots(1,3,figsize=(13,4))
                axes[0].plot(obs[:,0],obs[:,1]);axes[0].scatter(obs[0,0],obs[0,1],label='Start');axes[0].legend()
                axes[0].set(xlabel='Horizontal position (normalized)',ylabel='Height (normalized)')
                axes[1].plot(obs[:,3],label='Vertical velocity');axes[1].plot(obs[:,4],label='Angle');axes[1].legend()
                axes[2].plot(actions[:,0],label='Main');axes[2].plot(actions[:,1],label='Side');axes[2].legend()
                for ax in axes[1:]:ax.set_xlabel('Episode transition')
                fig.suptitle(f'{job["cell"]}, training seed {job["config"]["seed"]}, {step:,}; eval seed 40000')
                fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=150);plt.close(fig)
            if maps:action_maps(checkpoint,bank,out/'action_maps')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--stage',choices=['A','B'],required=True);p.add_argument('--maps',action='store_true')
    a=p.parse_args();examples(a.input,a.stage,a.maps)
