"""Build an offline checkpoint explorer with paired, fully captured MC rollouts.

The physical state comes from an existing state_action_evaluation diagnostic.
Every checkpoint-*.pt in --run is evaluated; no training or source data is changed.
"""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch

from .experiment import load_checkpoint, discounted_returns
from .state_action_evaluation import action_grid, landing_geometry, rollout
from .study_diagnostics import atomic_json, atomic_npz


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_values(q, samples):
    """Exact actor baseline is the final action; subtract within paired seeds."""
    g = np.asarray(samples).mean(axis=1)
    paired = samples - samples[-1]
    return dict(q=q.tolist(), g=g.tolist(), delta_q=(q-q[-1]).tolist(),
                delta_g=paired.mean(axis=1).tolist(), sd=np.std(samples, axis=1, ddof=1).tolist(),
                delta_sd=np.std(paired, axis=1, ddof=1).tolist())


def capture_checkpoint(task):
    path, probe, seeds, grid_size, destination = task
    torch.set_num_threads(1)
    cfg, actor, critics, step = load_checkpoint(path)
    axis, grid = action_grid(grid_size)
    obs = torch.tensor(probe['obs'], dtype=torch.float32)[None]
    with torch.inference_mode():
        actions = np.vstack((grid, actor(obs)[0].numpy()))
        q = critics[0](obs.repeat(len(actions), 1), torch.from_numpy(actions)).numpy().reshape(-1) / cfg.reward_scale
    offsets, observations, rewards, qs, gs, all_actions, samples, meta = [0], [], [], [], [], [], [], []
    for action in actions:
        values = []
        for seed in seeds:
            result = rollout(cfg, actor, probe, action, seed, capture=True)
            trajectory_obs, trajectory_actions = result.pop('observations'), result.pop('actions')
            trajectory_rewards = result.pop('rewards')
            with torch.inference_mode():
                trajectory_q = critics[0](torch.from_numpy(trajectory_obs[:-1]), torch.from_numpy(trajectory_actions)).numpy().reshape(-1) / cfg.reward_scale
            trajectory_g = discounted_returns(trajectory_rewards, cfg.gamma)
            # Store one post-terminal state per rollout separately; arrays at offsets
            # otherwise align states, actions, rewards, Q and realized suffix G.
            result['final_observation'] = trajectory_obs[-1].tolist()
            values.append(result['mc']); meta.append(result)
            observations.append(trajectory_obs[:-1]); all_actions.append(trajectory_actions)
            rewards.append(trajectory_rewards); qs.append(trajectory_q); gs.append(trajectory_g)
            offsets.append(offsets[-1] + len(trajectory_rewards))
        samples.append(values)
    destination = Path(destination)
    atomic_npz(destination, axis=axis, actions=actions, q=q, samples=np.asarray(samples),
               offsets=np.asarray(offsets), observations=np.concatenate(observations),
               rollout_actions=np.concatenate(all_actions), rewards=np.concatenate(rewards),
               rollout_q=np.concatenate(qs), rollout_g=np.concatenate(gs))
    atomic_json(destination.with_suffix('.json'), dict(step=step, rollouts=meta,
                checkpoint_sha256=sha256(path)))
    return step


def compact_rollouts(data, metadata, repeats, max_points=160):
    """Only the viewer is sampled; the NPZ retains every transition and reward."""
    # NpzFile decompresses the entire array on each access. Load each field once,
    # especially for checkpoints containing millions of captured transitions.
    data = {key: data[key] for key in ('offsets', 'observations', 'rollout_q', 'rollout_g')}
    result = []
    for index, record in enumerate(metadata['rollouts']):
        lo, hi = data['offsets'][index:index+2]
        length = int(hi-lo)
        indices = np.unique(np.rint(np.linspace(0, length-1, min(length, max_points))).astype(int))
        obs = data['observations'][lo:hi]
        points = np.vstack((obs[indices, :2], np.asarray(record['final_observation'])[:2]))
        result.append(dict(seed=record['future_seed'], length=length, ending=record['ending'],
                           censored=record['censored'], g=record['mc'], reward=record['performance_return'],
                           t=np.r_[indices, length].tolist(), xy=np.round(points, 6).tolist(),
                           q=np.round(data['rollout_q'][lo:hi][indices], 5).tolist(),
                           suffix_g=np.round(data['rollout_g'][lo:hi][indices], 5).tolist()))
    return [result[i:i+repeats] for i in range(0, len(result), repeats)]


def render(output, probe, paths, seeds, grid_size, cfg, source):
    frames = []
    for path in paths:
        step = int(path.stem.split('-')[-1])
        stem = output / f'capture-{step:07d}'
        metadata = json.loads(stem.with_suffix('.json').read_text())
        with np.load(stem.with_suffix('.npz')) as d:
            samples, q = d['samples'], d['q']
            frame = dict(step=step, actions=d['actions'].tolist(), **map_values(q, samples),
                         samples=samples.tolist(), trajectories=compact_rollouts(d, metadata, len(seeds)))
            frame.update(q_best=int(q[:-1].argmax()), g_best=int(samples.mean(axis=1)[:-1].argmax()))
            frames.append(frame)
    terrain, pad_x, pad_y = landing_geometry(cfg, probe)
    reference_file = source / 'reference_trajectory.npz'
    if reference_file.exists():
        with np.load(reference_file) as reference:
            prefix = reference['observations'][:len(probe['prefix'])+1, :2].tolist()
    else:
        # Older state-75 diagnostics did not store the reference trajectory.
        from .state_action_evaluation import reconstruct
        from .experiment import make_env
        env = make_env(cfg, cfg.diagnostic_horizon + len(probe['prefix']))
        try:
            obs, _ = env.reset(seed=probe['seed']); prefix = [obs[:2].tolist()]
            for action in probe['prefix']:
                obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32)); prefix.append(obs[:2].tolist())
        finally:
            env.close()
    payload = dict(frames=frames, axis=action_grid(grid_size)[0].tolist(), seeds=seeds,
                   state=probe, terrain=terrain.tolist(), pad_x=pad_x.tolist(), pad_y=float(pad_y),
                   prefix=prefix, gamma=cfg.gamma, horizon=cfg.diagnostic_horizon,
                   condition=dict(algorithm=cfg.algorithm, seed=cfg.seed,
                       critic_ln=cfg.layer_norm if cfg.critic_layer_norm is None else cfg.critic_layer_norm))
    packed = base64.b64encode(gzip.compress(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode())).decode()
    template = Path(__file__).with_name('checkpoint_animation.html').read_text()
    (output / 'index.html').write_text(template.replace('__PAYLOAD__', packed))
    atomic_json(output / 'render_provenance.json', dict(
        capture_manifest_sha256=sha256(output / 'manifest.json'),
        renderer_sha256=sha256(__file__),
        template_sha256=sha256(Path(__file__).with_name('checkpoint_animation.html')),
        html_sha256=sha256(output / 'index.html')))


def render_saved(output):
    """Re-render captured measurements, preserving their original provenance."""
    output = Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    settings = manifest['settings']
    paths = [Path(settings['run']) / name for name in settings['checkpoints']]
    paths.sort(key=lambda path: int(path.stem.split('-')[-1]))
    for path in paths:
        if sha256(path) != settings['checkpoints'][path.name]:
            raise ValueError(f'Checkpoint changed: {path}')
        step = int(path.stem.split('-')[-1])
        metadata = json.loads((output / f'capture-{step:07d}.json').read_text())
        if metadata['checkpoint_sha256'] != settings['checkpoints'][path.name]:
            raise ValueError(f'Capture/checkpoint mismatch: {path}')
    cfg = load_checkpoint(paths[0])[0]
    render(output, settings['state'], paths, settings['seeds'], settings['grid_size'],
           cfg, Path(settings['source']))
    print(f'Rendered {len(paths)} saved checkpoints: {output / "index.html"}', flush=True)


def build(run, source, output, workers=3, grid_size=9, seeds=tuple(range(55000, 55005))):
    run, source, output = (Path(p).resolve() for p in (run, source, output))
    if workers < 1 or grid_size < 2 or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Need positive workers, grid size >=2 and at least two distinct future seeds')
    paths = sorted(run.glob('checkpoint-[0-9]*.pt'), key=lambda p: int(p.stem.split('-')[-1]))
    if not paths:
        raise FileNotFoundError(f'No inference checkpoints in {run}')
    probe = json.loads((source / 'state.json').read_text())
    reference_provenance = json.loads((source / 'provenance.json').read_text())
    cfg = load_checkpoint(paths[0])[0]
    # A cache is reusable only with exactly the same models, physical state,
    # implementation and evaluation settings. New checkpoints require a new output.
    settings = dict(run=str(run), source=str(source), state=probe, seeds=list(seeds), grid_size=grid_size,
                    checkpoints={p.name:sha256(p) for p in paths},
                    source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
                                   ('checkpoint_animation.py', 'state_action_evaluation.py', 'experiment.py')})
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['settings'] != settings:
        raise ValueError('Existing output has different inputs; choose a fresh --output directory')
    if not manifest.exists() and any(output.iterdir()):
        raise ValueError('Refusing an existing output directory without a manifest')
    for path in paths:
        other, _, _, recorded_step = load_checkpoint(path)
        if recorded_step != int(path.stem.split('-')[-1]):
            raise ValueError(f'Checkpoint filename/step mismatch: {path}')
        for field in ('env_id', 'continuous', 'gravity', 'enable_wind', 'gamma', 'train_horizon', 'diagnostic_horizon', 'seed', 'algorithm', 'reward_scale'):
            if getattr(other, field) != getattr(cfg, field):
                raise ValueError(f'Inconsistent {field}: {path}')
    if cfg.gamma != reference_provenance['gamma'] or cfg.diagnostic_horizon != reference_provenance['diagnostic_cutoff']:
        raise ValueError('Reference and checkpoint return definitions differ')
    atomic_json(manifest, dict(settings=settings, reference=reference_provenance,
        definition='delta subtracts exact actor-action value within checkpoint; paired future seeds',
        trajectory_sampling='NPZ: every transition; HTML: up to 160 evenly spaced states plus terminal endpoint',
        note='G follows each frozen checkpoint actor after the selected first action; finite censored sums have no critic tail'))
    tasks = []
    for path in paths:
        step = int(path.stem.split('-')[-1]); destination = output / f'capture-{step:07d}.npz'
        if not (destination.exists() and destination.with_suffix('.json').exists()):
            tasks.append((str(path), probe, seeds, grid_size, str(destination)))
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for count, step in enumerate(pool.map(capture_checkpoint, tasks), 1):
            print(f'Captured {step:,}: {count}/{len(tasks)} new checkpoints', flush=True)
    # When the original four measurements exist, require numerical agreement.
    if list(seeds) == reference_provenance['future_seeds'] and grid_size == reference_provenance['grid_size']:
        for path in paths:
            step = int(path.stem.split('-')[-1]); original = source / f'values-{step:07d}.npz'
            if original.exists() and reference_provenance['checkpoints'].get(str(step), {}).get('sha256') == sha256(path):
                with np.load(original) as old, np.load(output / f'capture-{step:07d}.npz') as new:
                    np.testing.assert_allclose(old['samples'], new['samples'], rtol=0, atol=1e-9)
                    np.testing.assert_allclose(old['q'], new['q'], rtol=1e-6, atol=1e-4)
    render(output, probe, paths, seeds, grid_size, cfg, source)
    print(f'Viewer: {output / "index.html"} ({len(paths)} checkpoints, {len(seeds)} MC seeds/action)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--state-source', type=Path, help='Existing fixed-state diagnostic directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true', help='Render existing captures without new evaluations')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--grid-size', type=int, default=9)
    args = parser.parse_args()
    if args.render_only:
        render_saved(args.output)
        return
    if args.run is None or args.state_source is None:
        parser.error('--run and --state-source are required unless --render-only is set')
    build(args.run, args.state_source, args.output, args.workers, args.grid_size)


if __name__ == '__main__':
    main()
