"""All-trajectory calibration and fixed, replayable state/action probes."""
from pathlib import Path
import json
import os
import numpy as np
import torch
from .experiment import discounted_returns, make_env


def sync_directory(path):
    descriptor = os.open(str(Path(path)), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def build_probes(cfg, count=12, seed_start=30000, prefix_steps=30):
    bank = []
    for index in range(count):
        seed = seed_start+index
        rng = np.random.default_rng(seed+100000)
        env = make_env(cfg, prefix_steps+cfg.diagnostic_horizon)
        obs, _ = env.reset(seed=seed)
        prefix = []
        for _ in range(prefix_steps):
            action = rng.uniform(-1, 1, 2).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated), 'Choose a shorter fixed prefix before launch'
            prefix.append(action.tolist())
        bank.append(dict(seed=seed, prefix=prefix, obs=obs.tolist(),
                         action=rng.uniform(-1, 1, 2).astype(np.float32).tolist()))
        env.close()
    return bank


def record_return(q, rewards, gamma, indices, horizons, terminated):
    estimates = {h: discounted_returns(rewards[:h], gamma)[indices] for h in horizons}
    actual = estimates[max(horizons)]
    errors = q - actual[:, None]
    record = dict(length=len(rewards), censored=not bool(terminated),
                  sampled_pairs=len(indices), bias_q1=float(errors[:, 0].mean()),
                  rmse_q1=float(np.sqrt(np.mean(errors[:, 0]**2))),
                  mae_q1=float(np.abs(errors[:, 0]).mean()),
                  bias_min=float((q.min(axis=1)-actual).mean()),
                  mean_q1=float(q[:, 0].mean()), mean_mc=float(actual.mean()),
                  positive_error_fraction=float((errors[:, 0]>0).mean()),
                  horizon_delta_max=float(np.max(np.abs(actual-estimates[min(horizons)]))))
    if q.shape[1] == 2:
        record['bias_q2'] = float(errors[:, 1].mean())
    return record, estimates


@torch.inference_mode()
def evaluate_study(cfg, actor, critics, step, out, bank, horizons=(2500, 5000, 10000)):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    assert max(horizons) == cfg.diagnostic_horizon and min(horizons) >= cfg.train_horizon
    metric = dict(step=step, algorithm=cfg.algorithm, layer_norm=cfg.layer_norm, seed=cfg.seed,
                  actor_layer_norm=cfg.actor_layer_norm,
                  critic_layer_norm=cfg.layer_norm if cfg.critic_layer_norm is None else cfg.critic_layer_norm)
    for kind in ('on_policy', 'common_probe'):
        records, qs, obs_list, acts, ep_list, times, reward_list, offsets = [], [], [], [], [], [], [], [0]
        trajectory_obs,trajectory_actions,trajectory_offsets=[],[],[0]
        gs = {h: [] for h in horizons}
        count = cfg.eval_episodes if kind == 'on_policy' else len(bank)
        for index in range(count):
            probe = bank[index] if kind == 'common_probe' else None
            prefix = probe['prefix'] if probe else []
            seed = probe['seed'] if probe else cfg.eval_seed_start+index
            env = make_env(cfg, len(prefix)+cfg.diagnostic_horizon)
            obs, _ = env.reset(seed=seed)
            for action in prefix:
                obs, _, term, trunc, _ = env.step(np.asarray(action, dtype=np.float32))
                assert not (term or trunc)
            if probe:
                np.testing.assert_array_equal(obs, np.asarray(probe['obs'], dtype=np.float32))
            rewards, predictions, observations, actions = [], [], [], []
            term = trunc = False
            while not (term or trunc):
                tensor = torch.as_tensor(obs).unsqueeze(0)
                action = torch.tensor(probe['action'], dtype=torch.float32).unsqueeze(0) if probe and not rewards else actor(tensor)
                if cfg.instrumentation and len(rewards)<cfg.train_horizon:
                    trajectory_obs.append(obs.copy());trajectory_actions.append(action[0].numpy().copy())
                if (not probe and len(rewards) < cfg.train_horizon) or (probe and not rewards):
                    # Match raw discounted MC: trained Q is in scaled reward units.
                    predictions.append([q(tensor, action).item() / cfg.reward_scale for q in critics])
                    observations.append(obs.copy()); actions.append(action[0].numpy().copy())
                obs, reward, term, trunc, _ = env.step(action[0].numpy())
                rewards.append(float(reward))
            ending = ('out_of_bounds' if abs(obs[0]) >= 1
                      else 'game_over' if env.unwrapped.game_over
                      else 'lander_asleep' if not env.unwrapped.lander.awake
                      else 'time_limit' if trunc else 'other')
            env.close()
            indices = np.array([0]) if probe else np.arange(0, min(len(rewards), cfg.train_horizon), cfg.diagnostic_stride)
            q = np.asarray(predictions)[indices]
            record, estimates = record_return(q, rewards, cfg.gamma, indices, horizons, term)
            record.update(index=index, seed=seed, performance_return=float(sum(rewards[:cfg.train_horizon])))
            if cfg.instrumentation:
                record.update(ending=ending, performance_length=min(len(rewards),cfg.train_horizon))
            records.append(record); qs.extend(q); obs_list.extend(np.asarray(observations)[indices]); acts.extend(np.asarray(actions)[indices])
            trajectory_offsets.append(len(trajectory_obs))
            ep_list.extend([index]*len(indices)); times.extend(indices)
            reward_list.extend(rewards); offsets.append(len(reward_list))
            for h in horizons:
                gs[h].extend(estimates[h])
        result = dict(episodes=records, censored=sum(r['censored'] for r in records),
                      sampled_pairs=len(qs), horizon_delta_max=max(r['horizon_delta_max'] for r in records))
        for key in ('bias_q1', 'bias_q2', 'bias_min', 'rmse_q1', 'mae_q1', 'mean_q1', 'mean_mc', 'positive_error_fraction'):
            values = [r[key] for r in records if key in r]
            result[key] = float(np.mean(values)) if values else None
        result['return'] = float(np.mean([r['performance_return'] for r in records]))
        metric[kind] = result
        atomic_npz(out/f'{kind}-{step:07d}.npz', q=np.asarray(qs), obs=np.asarray(obs_list), action=np.asarray(acts),
                   episode=np.asarray(ep_list), t=np.asarray(times), rewards=np.asarray(reward_list), offsets=np.asarray(offsets),
                   trajectory_obs=np.asarray(trajectory_obs),trajectory_actions=np.asarray(trajectory_actions),
                   trajectory_offsets=np.asarray(trajectory_offsets),
                   **{f'mc{h}':np.asarray(gs[h]) for h in horizons})
    return metric
