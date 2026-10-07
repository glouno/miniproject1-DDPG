"""CPU long-study training with atomic, independently tested resumable state.

The gradient-update loop follows experiment.py; shared models, buffers, target
helpers and initial RNG setup are unchanged. Evaluations retain all trajectories.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import pickle
from pathlib import Path
from .paths import dependency_lock, snapshot_source
import platform
import random
import shutil
import signal
import time
import numpy as np
import torch
from torch import nn
from rl_mind.core import Action
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.nn import soft_update
from .experiment import Config, initialize, bellman_target, resolve_device, synchronize
from .study_diagnostics import atomic_json, evaluate_study, sync_directory
from .tensorboard_logging import create_writer, log_metric, log_episode, log_optimization
from .mechanism_diagnostics import network_audit, norm, distribution, phase_errors


def atomic_torch(path, state, rotate=False):
    path = Path(path); tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    if rotate and path.exists():
        os.replace(path, path.with_name('resume.previous.pt'))
    os.replace(tmp, path)
    sync_directory(path.parent)


def snapshot_replay(replay):
    data = replay._storage[:len(replay)]._map(lambda t:t.clone()) if len(replay) else None
    return dict(data=data,size=len(replay),pos=replay._pos,capacity=replay.capacity)


def restore_replay(replay,state):
    assert replay.capacity==state['capacity']
    replay._size, replay._pos = state['size'],state['pos']
    if state['data'] is not None:
        replay._storage=state['data']._map(lambda t:torch.zeros((replay.capacity,*t.shape[1:]),dtype=t.dtype))
        replay._storage.set_(torch.arange(replay._size),state['data'])


def recover_files(output,step,sizes):
    archive=output/f'recovery-{time.time_ns()}'
    archive.mkdir()
    for name,size in sizes.items():
        path=output/name
        if path.exists():
            assert path.stat().st_size>=size
            shutil.copy2(path,archive/name)
            with path.open('r+b') as f:
                f.truncate(size)
    for pattern in ('checkpoint-*.pt','diagnostics/*.npz','audits/audit-*.json'):
        for path in output.glob(pattern):
            if int(path.stem.rsplit('-',1)[1])>step:
                dest=archive/path.relative_to(output);dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,dest)


def train_study(cfg, output, bank, horizons=(2500, 5000, 10000), resume=False, stop_after=None):
    cfg.validate()
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    device = resolve_device(cfg.device)
    output = Path(output)
    restoring = resume and ((output/'resume.pt').exists() or (output/'resume.previous.pt').exists())
    if output.exists() and any(output.iterdir()) and not restoring:
        raise FileExistsError(f'Refusing to overwrite run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    snapshot_source(output / 'source')
    active_lock = dependency_lock()
    versions = {name: importlib.metadata.version(name)
                for name in ('rl-mind', 'gymnasium', 'torch', 'numpy', 'Box2D')}
    metadata = dict(config=asdict(cfg), versions=versions, python=platform.python_version(),
        platform=platform.platform(), started_at=datetime.now(timezone.utc).isoformat(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lock_sha256=hashlib.sha256(active_lock.read_bytes()).hexdigest(),
        lock_path=str(active_lock.resolve()),
        device=cfg.device, accelerator_name=(torch.cuda.get_device_name(0) if cfg.device == 'cuda' else None), adam=dict(betas=[0.9, 0.999], eps=1e-8, weight_decay=0.),
        layer_norm=dict(actor=cfg.actor_layer_norm,
            critic=cfg.layer_norm if cfg.critic_layer_norm is None else cfg.critic_layer_norm,
            location='hidden Linear -> LayerNorm -> ReLU; outputs unnormalized', eps=1e-5),
        targets='target actor in BOTH algorithms; TD3 uses min and smoothing',
        aggregation='pairs -> episode means -> run means; training seed is inference unit')
    if restoring:
        old = json.loads((output/'metadata.json').read_text())
        for key in ('config', 'source_sha256', 'lock_sha256', 'versions'):
            assert old[key] == json.loads(json.dumps(metadata[key])), f'Resume mismatch: {key}'
    else:
        atomic_json(output/'metadata.json', metadata)
    env, actor, critics = initialize(cfg)
    target_actor = copy.deepcopy(actor).requires_grad_(False)
    target_critics = copy.deepcopy(critics).requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=cfg.critic_lr)
    replay = ReplayBuffer(cfg.replay_capacity)
    rng = np.random.default_rng(cfg.seed + 17)
    updates = actor_updates = 0
    started = time.monotonic()
    elapsed_base = 0.
    histories = []
    checkpoint_seconds = 0.
    action_low, action_high = env.action_space.low, env.action_space.high
    action_scale = (action_high - action_low) / 2

    def checkpoint(step):
        nonlocal checkpoint_seconds
        synchronize(device)
        checkpoint_started = time.monotonic()
        # Diagnose a frozen CPU copy, consistently across training backends.
        eval_actor = copy.deepcopy(actor).to('cpu') if device.type != 'cpu' else actor
        eval_critics = copy.deepcopy(critics).to('cpu') if device.type != 'cpu' else critics
        metric = evaluate_study(cfg, eval_actor, eval_critics, step, output/'diagnostics', bank, horizons)
        if cfg.instrumentation:
            audit=network_audit(actor,critics,target_actor,target_critics,bank)
            for kind in ('on_policy','common_probe'):
                with np.load(output/'diagnostics'/f'{kind}-{step:07d}.npz') as pairs:
                    audit[kind+'_phases']=phase_errors(pairs)
            (output/'audits').mkdir(exist_ok=True)
            atomic_json(output/'audits'/f'audit-{step:07d}.json',audit)
        metric.update(critic_rounds=updates, actor_updates=actor_updates,
                      elapsed_seconds=round(elapsed_base + time.monotonic() - started, 3))
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metric, allow_nan=False) + '\n')
            f.flush(); os.fsync(f.fileno())
        atomic_torch(output/f'checkpoint-{step:07d}.pt', dict(config=asdict(cfg), actor=actor.state_dict(),
            critics=critics.state_dict(),target_actor=target_actor.state_dict(),
            target_critics=target_critics.state_dict(),step=step,critic_rounds=updates,actor_updates=actor_updates))
        histories.append(metric)
        log_metric(writer, metric)
        print(f'{cfg.algorithm} actorLN={cfg.actor_layer_norm} criticLN={metadata["layer_norm"]["critic"]} S{cfg.seed} step={step:,} '
              f'return={metric["on_policy"]["return"]:.1f} bias={metric["on_policy"]["bias_q1"]:.1f} '
              f'censored={metric["on_policy"]["censored"]} common_bias={metric["common_probe"]["bias_q1"]:.1f}', flush=True)
        checkpoint_seconds += time.monotonic() - checkpoint_started

    obs, _ = env.reset(seed=cfg.seed)
    episode_start_rng = None
    episode_actions = []
    episode_return = 0.
    episode_length = 0
    first_step = 1
    if restoring:
        # Only load our own locally generated training state, never external checkpoints.
        state=None
        for path in (output/'resume.pt',output/'resume.previous.pt'):
            if path.exists():
                try:
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    break
                except (OSError,RuntimeError,EOFError,pickle.UnpicklingError,ValueError):
                    print(f'Unreadable checkpoint {path}; trying previous',flush=True)
        if state is None:
            raise RuntimeError('No readable resume checkpoint')
        assert state['config'] == asdict(cfg)
        actor.load_state_dict(state['actor']); critics.load_state_dict(state['critics'])
        target_actor.load_state_dict(state['target_actor']); target_critics.load_state_dict(state['target_critics'])
        actor_optimizer.load_state_dict(state['actor_optimizer']); critic_optimizer.load_state_dict(state['critic_optimizer'])
        restore_replay(replay, state['replay'])
        episode_start_rng = state['episode_start_rng']
        if episode_start_rng is not None:
            env.unwrapped.np_random.bit_generator.state = episode_start_rng
            obs, _ = env.reset()
        episode_actions = state['episode_actions']
        for action in episode_actions:
            obs, _, term, trunc, _ = env.step(action)
            assert not (term or trunc), 'Saved prefix must be inside a live episode'
        np.testing.assert_array_equal(obs, state['obs'])
        episode_return, episode_length = state['episode_return'], state['episode_length']
        updates, actor_updates = state['updates'], state['actor_updates']
        checkpoint_seconds, elapsed_base = state['checkpoint_seconds'], state['elapsed_seconds']
        rng.bit_generator.state = state['rng']
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
        recover_files(output, state['step'], state['journal_sizes'])
        histories = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        first_step = state['step']+1
        print(f'RESUMED at step {first_step-1}; reconstructed physical state exactly', flush=True)

    def save_resume(step):
        sizes = {name:(output/name).stat().st_size if (output/name).exists() else 0
                 for name in ('metrics.jsonl','training_episodes.jsonl','optimization.jsonl')}
        state = dict(config=asdict(cfg), step=step, actor=actor.state_dict(), critics=critics.state_dict(),
            target_actor=target_actor.state_dict(), target_critics=target_critics.state_dict(),
            actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
            replay=snapshot_replay(replay), rng=rng.bit_generator.state,
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
            episode_start_rng=episode_start_rng, episode_actions=episode_actions, obs=obs,
            episode_return=episode_return, episode_length=episode_length, updates=updates, actor_updates=actor_updates,
            checkpoint_seconds=checkpoint_seconds, elapsed_seconds=elapsed_base+time.monotonic()-started,
            journal_sizes=sizes)
        atomic_torch(output/'resume.pt', state, rotate=True)

    def is_evaluation(step):
        interval=cfg.early_eval_interval if cfg.early_eval_interval and step<=cfg.early_eval_until else cfg.eval_interval
        return step % interval == 0 or step == cfg.max_steps

    writer = create_writer(output)
    if not restoring:
        checkpoint(0)
        save_resume(0)
    for step in range(first_step, cfg.max_steps + 1):
        if step <= cfg.learning_starts:
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        else:
            with torch.no_grad():
                action = actor(torch.as_tensor(obs, device=device).unsqueeze(0))[0].cpu().numpy()
            noise = rng.normal(0., cfg.exploration_noise, size=action.shape) * action_scale
            action = np.clip(action + noise, env.action_space.low, env.action_space.high).astype(np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(next_obs).all() and np.isfinite(reward)
        replay.add(Transitions(obs=torch.as_tensor(obs.copy()).unsqueeze(0),
            action=Action(value=torch.as_tensor(action.copy()).unsqueeze(0)),
            reward=torch.tensor([reward], dtype=torch.float32),
            next_obs=torch.as_tensor(next_obs.copy()).unsqueeze(0),
            terminated=torch.tensor([terminated], dtype=torch.bool)))
        episode_actions.append(action.copy())
        episode_return += reward
        episode_length += 1
        obs = next_obs
        if terminated or truncated:
            log_episode(writer, dict(step=step, reward=episode_return, length=episode_length))
            with (output/'training_episodes.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step, reward=episode_return, length=episode_length,
                                        terminated=bool(terminated), truncated=bool(truncated)))+'\n')
                f.flush();os.fsync(f.fileno())
            episode_start_rng = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)
            obs, _ = env.reset()
            episode_actions = []
            episode_return = 0.
            episode_length = 0

        if step >= cfg.learning_starts:
            audit_update = cfg.instrumentation and step % 1000 == 0
            optimization = dict(step=step, note='Last training minibatch at this transition')
            for _ in range(cfg.gradient_steps_per_transition):
                raw = replay.sample(cfg.batch_size)
                batch = Transitions(obs=raw.obs.to(device),
                    action=Action(value=raw.action.value.to(device)),
                    reward=raw.reward.to(device), next_obs=raw.next_obs.to(device),
                    terminated=raw.terminated.to(device))
                with torch.no_grad():
                    next_action = target_actor(batch.next_obs)
                    if cfg.algorithm == 'td3':
                        noise = (torch.randn_like(next_action) * cfg.target_noise).clamp(
                            -cfg.target_noise_clip, cfg.target_noise_clip) * actor.scale
                        next_action = torch.clamp(next_action + noise, min=actor.low, max=actor.high)
                    values = torch.stack([q(batch.next_obs, next_action) for q in target_critics])
                    next_q = values.min(dim=0).values
                    target = bellman_target(batch.reward, batch.terminated, next_q, cfg.gamma)
                critic_loss = sum(nn.functional.mse_loss(q(batch.obs, batch.action.value), target)
                                  for q in critics)
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError(f'Non-finite critic loss at {step}')
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if audit_update:
                    with torch.no_grad():
                        critic_before=[p.detach().clone() for p in critics.parameters()]
                        prediction=critics[0](batch.obs,batch.action.value)
                        optimization.update(critic_loss_sum=float(critic_loss.detach()), critic_gradient_norm=norm(critics.parameters(),True),
                            td_residual=distribution(prediction-target),q=distribution(prediction),target=distribution(target),
                            immediate_reward=distribution(batch.reward), bootstrap=distribution(target-batch.reward),
                            terminal_fraction=float(batch.terminated.float().mean()),
                            replay_actor_action_distance=float((actor(batch.obs)-batch.action.value).square().mean().sqrt()),
                            critic_losses=[float(nn.functional.mse_loss(q(batch.obs,batch.action.value),target)) for q in critics])
                critic_optimizer.step()
                if audit_update:
                    optimization.update(critic_parameter_norm=norm(critics.parameters()),
                        critic_update_norm=norm([p.detach()-old for p,old in zip(critics.parameters(),critic_before)]))
                updates += 1
                delay = cfg.policy_delay if cfg.algorithm == 'td3' else 1
                if updates % delay == 0:
                    critics[0].requires_grad_(False)
                    actor_loss = -critics[0](batch.obs, actor(batch.obs)).mean()
                    if not torch.isfinite(actor_loss):
                        raise FloatingPointError(f'Non-finite actor loss at {step}')
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if audit_update:
                        actor_before=[p.detach().clone() for p in actor.parameters()]
                        optimization.update(actor_loss=float(actor_loss.detach()),actor_gradient_norm=norm(actor.parameters(),True))
                    actor_optimizer.step()
                    if audit_update:
                        optimization.update(actor_parameter_norm=norm(actor.parameters()),
                            actor_update_norm=norm([p.detach()-old for p,old in zip(actor.parameters(),actor_before)]))
                    critics[0].requires_grad_(True)
                    actor_updates += 1
                    soft_update(actor, target_actor, cfg.tau)
                    soft_update(critics, target_critics, cfg.tau)
            if audit_update:
                with (output/'optimization.jsonl').open('a') as f:
                    f.write(json.dumps(optimization,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                log_optimization(writer,optimization)
        if is_evaluation(step):
            checkpoint(step)
            save_resume(step)
        elif cfg.resume_interval and step % cfg.resume_interval == 0:
            save_resume(step)
        if step % 1000 == 0 or step == cfg.max_steps:
            atomic_json(output/'progress.json', dict(status='training',step=step,max_steps=cfg.max_steps,
                updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=elapsed_base+time.monotonic()-started))
        if stop_requested or (stop_after is not None and step >= stop_after):
            save_resume(step)
            writer.close()
            env.close()
            signal.signal(signal.SIGTERM, previous_handler)
            atomic_json(output/'progress.json',dict(status='interrupted',step=step,max_steps=cfg.max_steps))
            return dict(status='interrupted',step=step)
    writer.close()
    env.close()
    expected_rounds = max(0, cfg.max_steps - cfg.learning_starts + 1) * cfg.gradient_steps_per_transition
    assert updates == expected_rounds
    assert actor_updates == updates // (cfg.policy_delay if cfg.algorithm == 'td3' else 1)
    synchronize(device)
    elapsed = elapsed_base + time.monotonic() - started
    summary = dict(device=cfg.device, checkpoint_seconds=round(checkpoint_seconds, 3),
        training_seconds=round(elapsed - checkpoint_seconds, 3), status='complete', elapsed_seconds=round(elapsed, 3),
        transitions=cfg.max_steps, critic_rounds=updates, actor_updates=actor_updates,
        final_return=histories[-1]['on_policy']['return'], final_bias=histories[-1]['on_policy']['bias_q1'])
    atomic_json(output/'summary.json', summary)
    atomic_json(output/'progress.json',dict(status='complete',step=cfg.max_steps,max_steps=cfg.max_steps))
    signal.signal(signal.SIGTERM, previous_handler)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--probes',type=Path,required=True)
    parser.add_argument('--algorithm',choices=['ddpg','td3'],required=True)
    parser.add_argument('--layer-norm',type=int,choices=[0,1],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    settings=json.loads(args.config.read_text());settings.update(algorithm=args.algorithm,layer_norm=bool(args.layer_norm),seed=args.seed)
    cfg=Config(**settings);assert cfg.device=='cpu','CLI study supports exact CPU resume; CUDA benchmarks use fresh runs'
    train_study(cfg,args.output,json.loads(args.probes.read_text()),resume=args.resume)


if __name__=='__main__':
    main()
