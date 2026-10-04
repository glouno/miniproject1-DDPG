"""Supplemental final-checkpoint calibration including hovering trajectories.

Run after the pilot: python calibration.py --pilot outputs/pilot-20261005
All trajectories are retained. Compare 2,500/5,000/10,000-step return horizons
from the SAME rollout, sampling only states in its first 1,000 transitions.
For nonterminating trajectories these are finite-return approximations, not
exact infinite-horizon returns. No critic estimate is inserted into the tail.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from experiment import discounted_returns, load_checkpoint, make_env


def calibrate(run):
    run = Path(run)
    cfg, actor, critics, step = load_checkpoint(run/'checkpoint-0050000.pt')
    out = run/'calibration-long'
    if out.exists():
        raise FileExistsError(f'Existing calibration: {out}')
    out.mkdir()
    q_pairs, g_pairs, ep_pairs, time_pairs, obs_pairs, act_pairs = [], [], [], [], [], []
    horizon_pairs = {2500: [], 5000: [], 10000: []}
    episodes = []
    with torch.inference_mode():
        for index in range(cfg.eval_episodes):
            env = make_env(cfg, 10000)
            obs, _ = env.reset(seed=cfg.eval_seed_start+index)
            rewards, predictions, observations, actions = [], [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                tensor = torch.as_tensor(obs).unsqueeze(0)
                action = actor(tensor)
                if len(rewards) < cfg.train_horizon:
                    predictions.append([float(q(tensor, action)) for q in critics])
                    observations.append(obs.copy())
                    actions.append(action[0].numpy().copy())
                obs, reward, terminated, truncated, _ = env.step(action[0].numpy())
                rewards.append(reward)
            env.close()
            indices = np.arange(0, min(len(rewards), cfg.train_horizon), cfg.diagnostic_stride)
            estimates = np.asarray(predictions)[indices]
            returns = {h: discounted_returns(rewards[:h], cfg.gamma)[indices]
                       for h in (2500, 5000, 10000)}
            actual = returns[10000]
            errors = estimates - actual[:, None]
            record = dict(episode=index, seed=cfg.eval_seed_start+index,
                performance_return=float(sum(rewards[:cfg.train_horizon])),
                diagnostic_length=len(rewards), censored_10000=not bool(terminated),
                bias_q1=float(errors[:,0].mean()),
                bias_min=float((estimates.min(axis=1)-actual).mean()),
                rmse_q1=float(np.sqrt(np.mean(errors[:,0]**2))),
                initial_bias_q1=float(errors[0,0]),
                mean_q1=float(estimates[:,0].mean()), mean_mc=float(actual.mean()),
                delta_2500_5000_max=float(np.max(np.abs(returns[5000]-returns[2500]))),
                delta_5000_10000_max=float(np.max(np.abs(returns[10000]-returns[5000]))))
            if len(critics)==2:
                record['bias_q2']=float(errors[:,1].mean())
            episodes.append(record)
            q_pairs.extend(estimates)
            g_pairs.extend(actual)
            ep_pairs.extend([index]*len(indices))
            time_pairs.extend(indices)
            obs_pairs.extend(np.asarray(observations)[indices])
            act_pairs.extend(np.asarray(actions)[indices])
            for h in horizon_pairs:
                horizon_pairs[h].extend(returns[h])
    result = dict(algorithm=cfg.algorithm, layer_norm=cfg.layer_norm, seed=cfg.seed, step=step,
        episodes=episodes, eval_return=float(np.mean([ep['performance_return'] for ep in episodes])),
        censored_10000=sum(ep['censored_10000'] for ep in episodes),
        sampled_pairs=len(g_pairs),
        delta_2500_5000_max=max(ep['delta_2500_5000_max'] for ep in episodes),
        delta_5000_10000_max=max(ep['delta_5000_10000_max'] for ep in episodes),
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((run/'checkpoint-0050000.pt').read_bytes()).hexdigest(),
        return_definition='observed discounted suffix; 10000-step approximation if censored')
    for key in ('bias_q1','bias_min','rmse_q1','initial_bias_q1','mean_q1','mean_mc','bias_q2'):
        values=[ep[key] for ep in episodes if key in ep]
        result[key]=float(np.mean(values)) if values else None
    original=json.loads((run/'metrics.jsonl').read_text().splitlines()[-1])
    np.testing.assert_allclose(result['eval_return'],original['eval_return'],rtol=0,atol=1e-6)
    np.savez_compressed(out/'pairs.npz',q=np.asarray(q_pairs),mc=np.asarray(g_pairs),
        mc2500=np.asarray(horizon_pairs[2500]),mc5000=np.asarray(horizon_pairs[5000]),
        episode=np.asarray(ep_pairs),t=np.asarray(time_pairs),
        obs=np.asarray(obs_pairs),action=np.asarray(act_pairs))
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(run.name,'long calibration complete; bias',result['bias_q1'],
          'censored',result['censored_10000'],'max horizon change',result['delta_5000_10000_max'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.run:
        calibrate(args.run)
        return
    manifest=json.loads((args.pilot/'manifest.json').read_text())
    runs=[args.pilot/j['directory'] for j in manifest['jobs']]
    assert all((run/'summary.json').exists() for run in runs), 'Pilot is incomplete'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def launch(run):
        return subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run',str(run)],env=env).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(launch,runs))
    assert all(result==0 for result in results), 'Read failed calibration output'


if __name__=='__main__':
    main()
