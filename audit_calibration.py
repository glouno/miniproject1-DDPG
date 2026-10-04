"""Replay saved final policies and independently verify stored Q/MC pair alignment.

Usage: python audit_calibration.py --pilot outputs/pilot-20261005 --workers 4
Writes reward traces for inspection without replacing any metrics/checkpoints.
Checks direct discounted dot products at sampled first/middle/last states.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from experiment import load_checkpoint,make_env


def audit(run):
    run=Path(run)
    cfg,actor,critics,step=load_checkpoint(run/'checkpoint-0050000.pt')
    destination=run/'calibration-long/reward-traces-audit.npz'
    if destination.exists():
        raise FileExistsError(destination)
    pairs=np.load(run/'calibration-long/pairs.npz',allow_pickle=False)
    all_rewards=[]
    offsets=[0]
    ended=[]
    with torch.inference_mode():
        for episode in range(cfg.eval_episodes):
            env=make_env(cfg,10000)
            obs,_=env.reset(seed=cfg.eval_seed_start+episode)
            rewards=[]
            finished=False
            sample_indices=np.flatnonzero(pairs['episode']==episode)
            sample_by_t={int(pairs['t'][i]):int(i) for i in sample_indices}
            while not finished:
                tensor=torch.as_tensor(obs).unsqueeze(0)
                action=actor(tensor)
                if len(rewards) in sample_by_t:
                    i=sample_by_t[len(rewards)]
                    np.testing.assert_allclose(obs,pairs['obs'][i],rtol=0,atol=1e-6)
                    np.testing.assert_allclose(action[0].numpy(),pairs['action'][i],rtol=0,atol=1e-6)
                    prediction=np.array([float(q(tensor,action)) for q in critics])
                    np.testing.assert_allclose(prediction,pairs['q'][i],rtol=1e-5,atol=1e-5)
                obs,reward,terminated,truncated,_=env.step(action[0].numpy())
                rewards.append(reward)
                finished=terminated or truncated
            env.close()
            # Independent direct sums, rather than the backwards-return helper.
            for i in np.unique(sample_indices[[0,len(sample_indices)//2,-1]]):
                t=int(pairs['t'][i])
                for horizon,key in ((2500,'mc2500'),(5000,'mc5000'),(10000,'mc')):
                    suffix=np.asarray(rewards[t:horizon],dtype=np.float64)
                    actual=np.dot(suffix,cfg.gamma**np.arange(len(suffix)))
                    np.testing.assert_allclose(actual,pairs[key][i],rtol=1e-9,atol=1e-9)
            all_rewards.extend(rewards)
            offsets.append(len(all_rewards))
            ended.append(terminated)
    np.savez_compressed(destination,rewards=np.asarray(all_rewards),offsets=np.asarray(offsets),
                        terminated=np.asarray(ended,dtype=bool),
                        episode_seed=np.arange(cfg.eval_seed_start,cfg.eval_seed_start+cfg.eval_episodes))
    result=dict(status='passed',episodes=cfg.eval_episodes,matched_pairs=len(pairs['mc']),
                direct_return_checks=cfg.eval_episodes*3*3)
    (run/'calibration-long/audit.json').write_text(json.dumps(result,indent=2))
    print(run.name,'independent state/action/Q/return replay audit passed',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.run:
        audit(args.run)
        return
    manifest=json.loads((args.pilot/'manifest.json').read_text())
    runs=[args.pilot/job['directory'] for job in manifest['jobs']]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def launch(run):
        return subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run',str(run)],env=env).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(launch,runs))
    assert all(code==0 for code in results),'Some audits failed'


if __name__=='__main__':
    main()
