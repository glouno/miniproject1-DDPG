"""Reproducible LunarLander DDPG/TD3 experiment and matched-return diagnostics.

Run with the locked rl-mind environment; see README.md. One environment step
records one actual transition. There are no vector auto-reset placeholder steps.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from rl_mind.core import Action
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.nn import soft_update


@dataclass(frozen=True)
class Config:
    algorithm: str = "td3"
    layer_norm: bool = False
    seed: int = 0
    env_id: str = "LunarLander-v3"
    continuous: bool = True
    gravity: float = -10.0
    enable_wind: bool = False
    train_horizon: int = 1_000
    max_steps: int = 50_000
    actor_hidden: tuple[int, ...] = (128, 128)
    critic_hidden: tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 5_000
    gradient_steps_per_transition: int = 1
    exploration_noise: float = 0.1
    policy_delay: int = 2  # TD3 only; DDPG always updates its actor every round.
    target_noise: float = 0.2  # TD3 only.
    target_noise_clip: float = 0.5  # TD3 only.
    eval_interval: int = 10_000
    eval_episodes: int = 10
    eval_seed_start: int = 10_000
    diagnostic_horizon: int = 5_000
    diagnostic_stride: int = 10
    torch_threads: int = 1
    device: str = "cpu"

    def validate(self):
        assert self.device in {"cpu", "cuda", "mps"}
        assert self.torch_threads > 0
        assert self.algorithm in {"ddpg", "td3"}
        assert self.continuous and 0 < self.gamma < 1 and 0 < self.tau <= 1
        assert self.max_steps > 0 and self.eval_interval > 0
        assert self.replay_capacity >= max(self.learning_starts, self.batch_size)
        assert self.learning_starts >= self.batch_size
        assert self.diagnostic_horizon >= self.train_horizon
        assert self.eval_episodes > 0 and self.diagnostic_stride > 0
        assert self.policy_delay > 0 and self.gradient_steps_per_transition > 0


def mlp(input_size, hidden, output_size, layer_norm=False):
    layers = []
    previous = input_size
    for width in hidden:
        layers.append(nn.Linear(previous, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.ReLU())
        previous = width
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden, low, high):
        super().__init__()
        self.model = mlp(obs_dim, hidden, len(low))
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

    @property
    def scale(self):
        return (self.high - self.low) / 2

    def forward(self, obs):
        return (self.high + self.low) / 2 + self.scale * torch.tanh(self.model(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden, layer_norm):
        super().__init__()
        self.model = mlp(obs_dim + action_dim, hidden, 1, layer_norm)

    def forward(self, obs, action):
        return self.model(torch.cat((obs, action), dim=-1)).squeeze(-1)


def discounted_returns(rewards, gamma):
    """Observed discounted suffixes; valid full MC targets at natural termination."""
    returns = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


def bellman_target(rewards, terminated, next_q, gamma):
    return rewards + gamma * (~terminated).float() * next_q.detach()


def make_env(cfg, horizon):
    return gym.make(cfg.env_id, continuous=cfg.continuous, gravity=cfg.gravity,
                    enable_wind=cfg.enable_wind, max_episode_steps=horizon)


def evaluate(cfg, actor, critics, step, destination):
    """Freeze current networks, collect full trajectories, match Q and MC pairs.

    Performance is the undiscounted reward of the first <=1,000 transitions.
    Diagnostics continue beyond that artificial cutoff to natural termination.
    If still censored at 5,000, that episode's bias is missing (never filled
    using the critic). Performance is retained and censoring is reported.
    Within each naturally terminated trajectory sample every tenth pair from
    the standard evaluation window; aggregate pairs within episodes first.
    """
    pair_q, pair_g, pair_obs, pair_action, pair_episode, pair_t = [], [], [], [], [], []
    episodes = []
    with torch.inference_mode():
        for index in range(cfg.eval_episodes):
            episode_seed = cfg.eval_seed_start + index
            env = make_env(cfg, cfg.diagnostic_horizon)
            obs, _ = env.reset(seed=episode_seed)
            rewards, predictions, observations, actions = [], [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = actor(tensor)
                if len(rewards) < cfg.train_horizon:
                    predictions.append([float(q(tensor, action).item()) for q in critics])
                    observations.append(obs.copy())
                    actions.append(action[0].numpy().copy())
                obs, reward, terminated, truncated, _ = env.step(action[0].numpy())
                rewards.append(float(reward))
            env.close()
            record = dict(episode=index, seed=episode_seed,
                return_1000=float(sum(rewards[:cfg.train_horizon])),
                performance_length=min(len(rewards), cfg.train_horizon),
                standard_time_limit=len(rewards) > cfg.train_horizon or
                    (len(rewards) == cfg.train_horizon and not terminated),
                diagnostic_length=len(rewards), diagnostic_censored=not bool(terminated))
            if terminated:
                g = discounted_returns(rewards, cfg.gamma)
                indices = np.arange(0, min(len(rewards), cfg.train_horizon), cfg.diagnostic_stride)
                q = np.asarray(predictions, dtype=np.float64)[indices]
                actual = g[indices]
                error = q - actual[:, None]
                record.update(bias_q1=float(error[:, 0].mean()),
                    bias_min=float((q.min(axis=1) - actual).mean()),
                    rmse_q1=float(np.sqrt(np.mean(error[:, 0] ** 2))),
                    initial_bias_q1=float(q[0, 0] - actual[0]),
                    mean_q1=float(q[:, 0].mean()), mean_mc=float(actual.mean()),
                    positive_error_fraction=float((error[:, 0] > 0).mean()),
                    sampled_pairs=len(indices))
                if len(critics) == 2:
                    record['bias_q2'] = float(error[:, 1].mean())
                pair_q.extend(q)
                pair_g.extend(actual)
                pair_obs.extend(np.asarray(observations)[indices])
                pair_action.extend(np.asarray(actions)[indices])
                pair_episode.extend([index] * len(indices))
                pair_t.extend(indices)
            episodes.append(record)

    valid = [ep for ep in episodes if not ep['diagnostic_censored']]
    metric = dict(step=step, algorithm=cfg.algorithm, layer_norm=cfg.layer_norm, seed=cfg.seed,
        eval_return=float(np.mean([ep['return_1000'] for ep in episodes])),
        eval_episode_sd=float(np.std([ep['return_1000'] for ep in episodes], ddof=1))
            if len(episodes) > 1 else 0.,
        standard_time_limit_fraction=float(np.mean([ep['standard_time_limit'] for ep in episodes])),
        diagnostic_censored=len(episodes) - len(valid), bias_episodes=len(valid),
        sampled_pairs=len(pair_g), episodes=episodes)
    for key in ('bias_q1', 'bias_q2', 'bias_min', 'rmse_q1', 'initial_bias_q1',
                'mean_q1', 'mean_mc', 'positive_error_fraction'):
        values = [ep[key] for ep in valid if key in ep]
        metric[key] = float(np.mean(values)) if values else None
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / f'pairs-{step:07d}.npz',
            q=np.asarray(pair_q).reshape(-1, len(critics)), mc=np.asarray(pair_g),
            obs=np.asarray(pair_obs).reshape(-1, 8), action=np.asarray(pair_action).reshape(-1, 2),
            episode=np.asarray(pair_episode, dtype=np.int64), t=np.asarray(pair_t, dtype=np.int64))
    return metric


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the PyTorch build and driver.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable on this machine.")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def initialize(cfg, device=None):
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    env = make_env(cfg, cfg.train_horizon)
    actor = Actor(8, cfg.actor_hidden, env.action_space.low, env.action_space.high)
    critics = nn.ModuleList([Critic(8, 2, cfg.critic_hidden, cfg.layer_norm)
                            for _ in range(2 if cfg.algorithm == 'td3' else 1)])
    device = resolve_device(cfg.device) if device is None else torch.device(device)
    return env, actor.to(device), critics.to(device)


def save_checkpoint(path, cfg, actor, critics, step):
    # Inference checkpoints, deliberately not advertised as resumable training.
    torch.save(dict(config=asdict(cfg), actor=actor.state_dict(), critics=critics.state_dict(),
                    step=step), path)


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    cfg = Config(**data['config'])
    env, actor, critics = initialize(cfg, device="cpu")
    env.close()
    actor.load_state_dict(data['actor'])
    critics.load_state_dict(data['critics'])
    return cfg, actor.eval(), critics.eval(), data['step']


def train(cfg, output):
    cfg.validate()
    device = resolve_device(cfg.device)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / 'source'
    source_dir.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, source_dir / source.name)
    active_lock = Path(sys.prefix).parent / 'uv.lock'
    if not active_lock.exists():
        active_lock = Path(__file__).parent/'environment'/'uv.lock'
    versions = {name: importlib.metadata.version(name)
                for name in ('rl-mind', 'gymnasium', 'torch', 'numpy', 'Box2D')}
    metadata = dict(config=asdict(cfg), versions=versions, python=platform.python_version(),
        platform=platform.platform(), started_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lock_sha256=hashlib.sha256(active_lock.read_bytes()).hexdigest(),
        lock_path=str(active_lock.resolve()),
        device=cfg.device, accelerator_name=(torch.cuda.get_device_name(0) if cfg.device == 'cuda' else None), adam=dict(betas=[0.9, 0.999], eps=1e-8, weight_decay=0.),
        layer_norm=dict(location='critics only: Linear -> LayerNorm -> ReLU', eps=1e-5),
        targets='target actor in BOTH algorithms; TD3 uses min and smoothing',
        aggregation='pairs -> episode means -> run means; training seed is inference unit')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    env, actor, critics = initialize(cfg)
    target_actor = copy.deepcopy(actor).requires_grad_(False)
    target_critics = copy.deepcopy(critics).requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=cfg.critic_lr)
    replay = ReplayBuffer(cfg.replay_capacity)
    rng = np.random.default_rng(cfg.seed + 17)
    updates = actor_updates = 0
    started = time.monotonic()
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
        metric = evaluate(cfg, eval_actor, eval_critics, step, output / 'diagnostics')
        metric.update(critic_rounds=updates, actor_updates=actor_updates,
                      elapsed_seconds=round(time.monotonic() - started, 3))
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metric, allow_nan=False) + '\n')
        save_checkpoint(output / f'checkpoint-{step:07d}.pt', cfg, actor, critics, step)
        histories.append(metric)
        print(f'{cfg.algorithm} LN={cfg.layer_norm} S{cfg.seed} step={step:,} '
              f'return={metric["eval_return"]:.1f} bias={metric["bias_q1"]} '
              f'censored={metric["diagnostic_censored"]}', flush=True)
        checkpoint_seconds += time.monotonic() - checkpoint_started

    checkpoint(0)
    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.
    episode_length = 0
    for step in range(1, cfg.max_steps + 1):
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
        episode_return += reward
        episode_length += 1
        obs = next_obs
        if terminated or truncated:
            with (output/'training_episodes.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step, reward=episode_return, length=episode_length,
                                        terminated=bool(terminated), truncated=bool(truncated)))+'\n')
            obs, _ = env.reset()
            episode_return = 0.
            episode_length = 0

        if step >= cfg.learning_starts:
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
                critic_optimizer.step()
                updates += 1
                delay = cfg.policy_delay if cfg.algorithm == 'td3' else 1
                if updates % delay == 0:
                    critics[0].requires_grad_(False)
                    actor_loss = -critics[0](batch.obs, actor(batch.obs)).mean()
                    if not torch.isfinite(actor_loss):
                        raise FloatingPointError(f'Non-finite actor loss at {step}')
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_optimizer.step()
                    critics[0].requires_grad_(True)
                    actor_updates += 1
                    soft_update(actor, target_actor, cfg.tau)
                    soft_update(critics, target_critics, cfg.tau)
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            checkpoint(step)
    env.close()
    expected_rounds = max(0, cfg.max_steps - cfg.learning_starts + 1) * cfg.gradient_steps_per_transition
    assert updates == expected_rounds
    assert actor_updates == updates // (cfg.policy_delay if cfg.algorithm == 'td3' else 1)
    synchronize(device)
    elapsed = time.monotonic() - started
    summary = dict(device=cfg.device, checkpoint_seconds=round(checkpoint_seconds, 3),
        training_seconds=round(elapsed - checkpoint_seconds, 3), status='complete', elapsed_seconds=round(elapsed, 3),
        transitions=cfg.max_steps, critic_rounds=updates, actor_updates=actor_updates,
        final_return=histories[-1]['eval_return'], final_bias=histories[-1]['bias_q1'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--algorithm', choices=['ddpg', 'td3'], required=True)
    parser.add_argument('--layer-norm', type=int, choices=[0, 1], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    settings.update(algorithm=args.algorithm, layer_norm=bool(args.layer_norm), seed=args.seed)
    if args.device is not None:
        settings["device"] = args.device
    train(Config(**settings), args.output)


if __name__ == '__main__':
    main()
