"""Fresh-run backend comparison; CUDA interruption recovery is not claimed."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from experiment import Config
from study_diagnostics import atomic_json,build_probes
from study_train import train_study


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for device in ('cpu','cuda'):
        for algo in ('ddpg','td3'):
            cfg=Config(algorithm=algo,actor_layer_norm=True,critic_layer_norm=True,device=device,
                isolated_initialization=True,instrumentation=True,max_steps=a.steps,
                learning_starts=1000,eval_interval=a.steps,eval_episodes=2,diagnostic_horizon=10000)
            bank=build_probes(cfg,4,41000,5)
            summary=train_study(cfg,a.output/f'{device}-{algo}',bank)
            results.append(dict(algorithm=algo,**summary))
            atomic_json(a.output/'benchmark.json',dict(runs=results,note='Single fresh run per backend and algorithm; short benchmark, not scientific outcomes.'))
            print(json.dumps(results[-1]),flush=True)


if __name__=='__main__':main()
