"""Matched physical-state Q/Monte Carlo maps across frozen checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

from .experiment import load_checkpoint, make_env
from .study_diagnostics import atomic_json, atomic_npz


def action_grid(size=9):
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    main, lateral = np.meshgrid(axis, axis, indexing="xy")
    return axis, np.column_stack((main.ravel(), lateral.ravel()))


def reconstruct(cfg, probe, future_seed=None):
    env = make_env(cfg, len(probe["prefix"]) + cfg.diagnostic_horizon)
    obs, _ = env.reset(seed=probe["seed"])
    for action in probe["prefix"]:
        obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            env.close()
            raise AssertionError("Reference prefix must end inside a live episode")
    try:
        np.testing.assert_array_equal(obs, np.asarray(probe["obs"], dtype=np.float32))
    except AssertionError:
        env.close()
        raise
    # Preserve the physical state and prefix; vary only future engine dispersion.
    if future_seed is not None:
        env.unwrapped.np_random.bit_generator.state = np.random.default_rng(future_seed).bit_generator.state
    return env, obs


def landing_geometry(cfg, probe):
    """Seed-specific terrain in the same normalized frame as saved observations."""
    from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
    env, _ = reconstruct(cfg, probe)
    try:
        world = env.unwrapped
        half_width, half_height = VIEWPORT_W / SCALE / 2, VIEWPORT_H / SCALE / 2
        origin_y = world.helipad_y + LEG_DOWN / SCALE
        # Match the terrain extraction used by report/scripts/build_stage_a_failures.py.
        terrain = np.asarray([p[0] for p in world.sky_polys] + [world.sky_polys[-1][1]], dtype=float)
        terrain = (terrain - [half_width, origin_y]) / [half_width, half_height]
        pad_x = (np.asarray([world.helipad_x1, world.helipad_x2]) - half_width) / half_width
        pad_y = (world.helipad_y - origin_y) / half_height
        return terrain, pad_x, pad_y
    finally:
        env.close()


@torch.inference_mode()
def rollout(cfg, actor, probe, action, future_seed=None, capture=False):
    env, obs = reconstruct(cfg, probe, future_seed)
    rewards, observations, actions = [], [], []
    terminated = truncated = False
    try:
        if capture:
            observations.append(obs.copy())
        while not (terminated or truncated):
            chosen = np.asarray(action, dtype=np.float32) if not rewards else actor(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
            obs, reward, terminated, truncated, _ = env.step(chosen)
            rewards.append(float(reward))
            if capture:
                observations.append(obs.copy()); actions.append(chosen.copy())
        ending = ("out_of_bounds" if abs(obs[0]) >= 1 else "game_over" if env.unwrapped.game_over
                  else "lander_asleep" if not env.unwrapped.lander.awake else "time_limit" if truncated else "other")
    finally:
        env.close()
    rewards = np.asarray(rewards, dtype=np.float64)
    remaining = cfg.train_horizon - len(probe["prefix"])
    result = dict(future_seed=future_seed, length=len(rewards), ending=ending,
        censored=not bool(terminated), mc=float(np.dot(rewards, cfg.gamma**np.arange(len(rewards)))),
        mc2500=float(np.dot(rewards[:2500], cfg.gamma**np.arange(min(2500, len(rewards))))),
        mc5000=float(np.dot(rewards[:5000], cfg.gamma**np.arange(min(5000, len(rewards))))),
        performance_timeout=len(rewards) > remaining,
        performance_return=float(probe["prefix_return"] + rewards[:remaining].sum()))
    if capture:
        result.update(observations=np.asarray(observations, dtype=np.float32),
                      actions=np.asarray(actions, dtype=np.float32), rewards=rewards)
    return result


_MODELS, _PROBE = {}, None


def initialize_worker(paths, probe):
    global _MODELS, _PROBE
    torch.set_num_threads(1)
    _MODELS = {int(step): load_checkpoint(path)[:3] for step, path in paths.items()}
    _PROBE = probe


def worker(task):
    step, index, action, future_seed = task
    cfg, actor, _ = _MODELS[step]
    return step, index, rollout(cfg, actor, _PROBE, action, future_seed)


def execute(pool, tasks, message):
    results = []
    pending = [pool.submit(worker, task) for task in tasks]
    for count, future in enumerate(as_completed(pending), 1):
        results.append(future.result())
        if count % 200 == 0 or count == len(pending):
            print(f"{message}: {count}/{len(pending)} rollouts", flush=True)
    return results


def paired_interval(values, reference):
    difference = np.asarray(values) - np.asarray(reference)
    rng = np.random.default_rng(419)
    indices = rng.integers(0, len(difference), (10000, len(difference)))
    low, high = np.quantile(difference[indices].mean(axis=1), [.025, .975])
    return dict(mean=float(difference.mean()), ci95=[float(low), float(high)])


def evaluate(run, out, eval_seed=40009, flight_step=75, steps=(20000, 50000, 300000, 1000000),
             grid_size=9, repeats=5, validation_repeats=20, workers=6):
    run, out = Path(run).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = {int(step): str(run / f"checkpoint-{step:07d}.pt") for step in steps}
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    reference_metric = next(r for r in rows if r["step"] == 1000000)
    episode = next(r for r in reference_metric["on_policy"]["episodes"] if r["seed"] == eval_seed)
    with np.load(run / "diagnostics/on_policy-1000000.npz") as data:
        index = episode["index"]
        lo, hi = (int(data["trajectory_offsets"][index+i]) for i in (0, 1))
        reference_obs = data["trajectory_obs"][lo:hi].copy()
        reference_actions = data["trajectory_actions"][lo:hi].copy()
        rlo, rhi = (int(data["offsets"][index+i]) for i in (0, 1))
        reference_rewards = data["rewards"][rlo:rhi].copy()
    probe = dict(seed=eval_seed, reference_checkpoint=1000000, flight_step=flight_step,
        prefix=reference_actions[:flight_step].tolist(), obs=reference_obs[flight_step].tolist(),
        prefix_return=float(reference_rewards[:flight_step].sum()))
    atomic_json(out / "state.json", probe)
    models = {step: load_checkpoint(path)[:3] for step, path in paths.items()}
    cfg, actor, _ = models[1000000]
    original = rollout(cfg, actor, probe, reference_actions[flight_step], capture=True)
    np.testing.assert_allclose(original["rewards"], reference_rewards[flight_step:], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(original["observations"][:-1], reference_obs[flight_step:])
    print("Verified exact physical replay and original reward suffix", flush=True)
    axis, grid = action_grid(grid_size)
    records = {}
    for step, (cfg, actor, critics) in models.items():
        obs = torch.tensor(probe["obs"], dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            actor_action = actor(obs)[0].numpy()
            actions = np.vstack((grid, actor_action))
            q = critics[0](obs.repeat(len(actions), 1), torch.from_numpy(actions)).numpy() / cfg.reward_scale
        records[step] = dict(actions=actions, q=q, actor_index=len(grid),
                             critic_index=int(q[:len(grid)].argmax()))
    seeds = list(range(55000, 55000 + repeats))
    validation_seeds = list(range(56000, 56000 + validation_repeats))
    tasks = [(step, i, action.tolist(), seed) for step, record in records.items()
             for i, action in enumerate(record["actions"]) for seed in seeds]
    raw_records = []
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(paths, probe)) as pool:
        results = execute(pool, tasks, "Heatmaps")
        for step, record in records.items():
            matrix = np.zeros((len(record["actions"]), repeats))
            censored = np.zeros_like(matrix, dtype=bool)
            for checkpoint, index, sample in results:
                if checkpoint == step:
                    rep = seeds.index(sample["future_seed"])
                    matrix[index, rep] = sample["mc"]; censored[index, rep] = sample["censored"]
                    raw_records.append(dict(checkpoint=step, action_index=index, **sample))
            record.update(g=matrix.mean(axis=1), sd=matrix.std(axis=1, ddof=1), samples=matrix,
                          censored=censored, mc_index=int(matrix[:len(grid)].mean(axis=1).argmax()))
            # Equivalent engine commands must have identical paired returns.
            for side in range(grid_size):
                idx = [side*grid_size+i for i, value in enumerate(axis) if value <= 0]
                np.testing.assert_array_equal(matrix[idx], np.repeat(matrix[idx[0]][None], len(idx), axis=0))
            atomic_npz(out/f"values-{step:07d}.npz", axis=axis, actions=record["actions"], q=record["q"],
                       g=record["g"], sd=record["sd"], samples=matrix, censored=censored)
        tasks = [(step, index, record["actions"][index].tolist(), seed) for step, record in records.items()
                 for index in sorted({record["actor_index"], record["critic_index"], record["mc_index"]})
                 for seed in validation_seeds]
        validation = execute(pool, tasks, "Independent candidate check")
    summary = []
    for step, record in records.items():
        candidate_results = {}
        for label, key in (("actor", "actor_index"), ("critic_best", "critic_index"), ("mc_best", "mc_index")):
            index = record[key]
            samples = sorted([s for st, i, s in validation if st == step and i == index], key=lambda s:s["future_seed"])
            returns = np.array([s["mc"] for s in samples])
            candidate_results[label] = dict(index=index, action=record["actions"][index].tolist(),
                q=float(record["q"][index]), g=float(returns.mean()), sd=float(returns.std(ddof=1)),
                returns=returns.tolist(), censored=sum(s["censored"] for s in samples),
                performance_timeouts=sum(s["performance_timeout"] for s in samples),
                endings={kind:sum(s["ending"] == kind for s in samples) for kind in sorted({s["ending"] for s in samples})})
            # Predefined illustration: original continuation randomness, shared across candidates.
            cfg, actor, _ = models[step]
            branch = rollout(cfg, actor, probe, record["actions"][index], capture=True)
            atomic_npz(out/f"branch-{step:07d}-{label}.npz", observations=branch["observations"],
                       actions=branch["actions"], rewards=branch["rewards"])
            candidate_results[label]["illustration"] = {k:v for k,v in branch.items() if k not in ("observations", "actions", "rewards")}
        for label in ("critic_best", "mc_best"):
            candidate_results[label]["delta_g_vs_actor"] = paired_interval(candidate_results[label]["returns"], candidate_results["actor"]["returns"])
            candidate_results[label]["delta_q_vs_actor"] = candidate_results[label]["q"] - candidate_results["actor"]["q"]
        summary.append(dict(checkpoint=step, **candidate_results,
            grid_censored=int(record["censored"].sum()), grid_rollouts=int(record["censored"].size)))
    atomic_json(out/"summary.json", dict(state=probe, checkpoints=summary,
        elapsed_seconds=time.perf_counter()-started,
        note="Single selected physical state, not a population analysis. G heatmaps: 5 paired future seeds; selected candidates checked on 20 separate seeds. No critic-filled tails. Earlier policies may not visit this state."))
    atomic_json(out/"rollouts.json", dict(grid=raw_records, validation=[dict(checkpoint=s, action_index=i, **r) for s,i,r in validation]))
    atomic_json(out/"provenance.json", dict(checkpoints={str(s):dict(path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for s,p in paths.items()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), eval_seed=eval_seed,
        training_seed=cfg.seed, future_seeds=seeds, validation_seeds=validation_seeds,
        grid_size=grid_size, workers=workers, physical_replay_verified=True, original_reward_suffix_verified=True,
        native_reward_units=True, diagnostic_cutoff=cfg.diagnostic_horizon, gamma=cfg.gamma))
    plot(out, summary, probe, reference_obs)
    print(f"Completed in {time.perf_counter()-started:.1f}s: {out}", flush=True)
    return summary


def plot(out, summary, probe, reference_obs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    steps = [r["checkpoint"] for r in summary]
    values = {step:dict(np.load(out/f"values-{step:07d}.npz")) for step in steps}
    markers = (("actor", "*", "white", 125), ("critic_best", "^", "#44dbf5", 62), ("mc_best", "o", "#ed68ce", 45))
    checkpoint_label = lambda step: '1M' if step == 1000000 else f'{step/1000:g}k'
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.1), layout="constrained", sharex=True, sharey=True)
    for column, row in enumerate(summary):
        v = values[row["checkpoint"]];axis=v["axis"];n=len(axis)
        for r, (data, name) in enumerate(((v["q"][:-1], "Predicted Q"), (v["q"][:-1]-v["g"][:-1], "Q − G"))):
            ax=axes[r,column]
            lo, hi = float(data.min()), float(data.max())
            if lo == hi:
                lo -= .5; hi += .5
            if r == 1 and lo < 0 < hi:
                im=ax.pcolormesh(axis,axis,data.reshape(n,n),shading="nearest",cmap="RdBu_r",norm=TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi))
            else:
                im=ax.pcolormesh(axis,axis,data.reshape(n,n),shading="nearest",cmap="viridis" if r == 0 else "magma",vmin=lo,vmax=hi)
            # Each map exposes small spatial differences; numeric bars retain absolute values.
            bar=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.035,fraction=.055,aspect=30,
                             ticks=[lo,(lo+hi)/2,hi],format='%.1f')
            bar.ax.tick_params(labelsize=8,pad=2)
            bar.set_label(f'{name}, raw units · range width {float(data.max()-data.min()):.1f}',fontsize=8,labelpad=2)
            ax.axvline(0,color="white",ls=":",lw=.7,alpha=.6)
            for boundary in (-.5,.5):ax.axhline(boundary,color="white",ls=":",lw=.7,alpha=.6)
            for label, marker, color, size in markers:
                a=row[label]["action"];ax.scatter(*a,marker=marker,s=size,c=color,edgecolors="black",linewidths=.8,zorder=5,clip_on=False)
            ax.set(xlim=(-1,1),ylim=(-1,1),aspect="equal",xticks=[-1,-.5,0,.5,1],yticks=[-1,-.5,0,.5,1])
            if column==0:ax.set_ylabel(f"{name}\nLateral command")
            if r==0:ax.set_title(f"{checkpoint_label(row['checkpoint'])} training transitions")
            if r==1:ax.set_xlabel("Main-engine command",labelpad=2)
    handles=[Line2D([],[],marker=m,color="none",markerfacecolor=c,markeredgecolor="black",markersize=9,label=l.replace('_',' ')) for l,m,c,_ in markers]
    fig.suptitle(f"DDPG · critic LN on · actor LN off · training seed 21 · eval seed {probe['seed']}\nFixed state at step {probe['flight_step']} · independent color ranges per panel · G: five paired continuations",fontsize=12)
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False)
    save(fig,out/"heatmaps",plt)
    # Relative values make ranking differences visible despite the large Q offset.
    limits={key:max(float(np.abs(v[key][:-1]-v[key][-1]).max()) for v in values.values()) for key in ('q','g')}
    preference_images={}
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout="constrained",sharex=True,sharey=True)
    for col,row in enumerate(summary):
        v=values[row['checkpoint']];n=len(v['axis'])
        for r,key in enumerate(('q','g')):
            ax=axes[r,col];delta=v[key][:-1]-v[key][-1]
            im=ax.pcolormesh(v['axis'],v['axis'],delta.reshape(n,n),shading="nearest",cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-limits[key],vcenter=0,vmax=limits[key]))
            preference_images[key]=im
            for label,marker,color,size in markers:ax.scatter(*row[label]['action'],marker=marker,c=color,s=size,edgecolors="black",linewidths=.8,clip_on=False)
            ax.set(xlim=(-1,1),ylim=(-1,1),aspect="equal",xticks=[-1,0,1],yticks=[-1,0,1])
            if r==0:ax.set_title(f"{row['checkpoint']/1000:g}k")
            if col==0:ax.set_ylabel(('ΔQ versus actor' if r==0 else 'ΔG versus actor')+'\nLateral command')
            if r==1:ax.set_xlabel('Main-engine command')
    for r,key in enumerate(('q','g')):
        fig.colorbar(preference_images[key],ax=axes[r,:].tolist(),label=f'Δ{key.upper()} in raw units; red = higher')
    fig.suptitle('Action preferences relative to the actor; separate scales for ΔQ and ΔG\nSame scale across checkpoints within each row; G uses five paired future seeds')
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    save(fig,out/'action_preferences',plt)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained');x=np.arange(len(summary))
    labels=[f'{s/1000:g}k' for s in steps]
    axes[0].plot(x,[r['actor']['q'] for r in summary],'o-',color='#ad3f36',label='Predicted Q')
    axes[0].plot(x,[r['actor']['g'] for r in summary],'o-',color='#246c80',label='Measured G (20 new seeds)')
    axes[0].set(title='Value of the actor’s chosen action',ylabel='Raw discounted return');axes[0].legend(fontsize=9)
    for label,color,offset in [('critic_best','#207f9c',-.06),('mc_best','#a74186',.06)]:
        means=np.array([r[label]['delta_g_vs_actor']['mean'] for r in summary]);ci=np.array([r[label]['delta_g_vs_actor']['ci95'] for r in summary])
        axes[1].errorbar(x+offset,means,yerr=np.stack((means-ci[:,0],ci[:,1]-means)),fmt='o',capsize=4,color=color,label=label.replace('_',' '))
    axes[1].axhline(0,color='grey',lw=.8);axes[1].set(title='Measured improvement over actor',ylabel='Paired ΔG; descriptive bootstrap 95% CI');axes[1].legend(fontsize=9)
    for ax in axes[:2]:ax.set(xticks=x,xticklabels=labels,xlabel='Saved checkpoint');ax.grid(axis='y',alpha=.2)
    positions=np.array([r['actor']['action'] for r in summary]);axes[2].plot(positions[:,0],positions[:,1],color='#555555',lw=1)
    for i,(position,label) in enumerate(zip(positions,labels)):
        axes[2].scatter(*position,s=55,color=plt.cm.viridis(i/3));axes[2].annotate(label,position,xytext=(6,6),textcoords='offset points',fontsize=9)
    axes[2].set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),xlabel='Main command',ylabel='Lateral command',title='Actor action at the fixed state');axes[2].grid(alpha=.2)
    fig.suptitle('Evolution at one reconstructed state; continuation policy changes with each checkpoint')
    save(fig,out/'evolution',plt)
    provenance = json.loads((out/'provenance.json').read_text())
    cfg = load_checkpoint(provenance['checkpoints']['1000000']['path'])[0]
    terrain, pad_x, pad_y = landing_geometry(cfg, probe)
    atomic_npz(out/'landing_geometry.npz', terrain=terrain, pad_x=pad_x, pad_y=np.asarray(pad_y))
    ground_floor=min(float(terrain[:,1].min())-.06,-.2)
    fig,axes=plt.subplots(1,4,figsize=(14,4.5),layout='constrained',sharex=True,sharey=True)
    for ax,row in zip(axes,summary):
        ax.fill_between(terrain[:,0],ground_floor,terrain[:,1],color='0.94',zorder=0)
        ax.plot(terrain[:,0],terrain[:,1],color='0.65',lw=.9,zorder=1)
        ax.plot(pad_x,[pad_y,pad_y],color='0.2',lw=4,solid_capstyle='butt',zorder=2)
        with np.load(out/f"branch-{row['checkpoint']:07d}-actor.npz") as b:
            remaining=1000-probe['flight_step']
            suffix=b['observations'][:remaining+1]
            reward=float(probe['prefix_return']+b['rewards'][:remaining].sum())
        obs=np.vstack((reference_obs[:probe['flight_step']],suffix))
        ending=row['actor']['illustration']
        outcome='timeout' if ending['performance_timeout'] else {'game_over':'collision','out_of_bounds':'boundary exit','lander_asleep':'rest'}.get(ending['ending'],ending['ending'])
        ax.plot(obs[:,0],obs[:,1],color='#ad3f36',lw=1.5,zorder=3)
        ax.scatter(obs[0,0],obs[0,1],marker='o',facecolors='none',edgecolors='#555555',s=25,zorder=6)
        ax.scatter(obs[-1,0],obs[-1,1],color='#ad3f36',marker='X',s=45,zorder=6)
        ax.scatter(probe['obs'][0],probe['obs'][1],marker='D',color='black',s=35,zorder=6)
        ax.axvline(-1,color='grey',ls='--',lw=.7);ax.axvline(1,color='grey',ls='--',lw=.7)
        ax.set(title=f"{checkpoint_label(row['checkpoint'])} · {outcome}\nCumulative reward = {reward:.1f}",xlabel='Normalized horizontal position');ax.grid(alpha=.15)
        ax.set(ylim=(ground_floor,max(float(reference_obs[:,1].max()),1.4)+.08))
    axes[0].annotate('Landing pad',xy=(float(pad_x.mean()),pad_y),xytext=(.45,ground_floor+.025),
                     fontsize=8,ha='center',arrowprops=dict(arrowstyle='->',lw=.8,color='0.25'))
    axes[0].set_ylabel('Normalized height')
    fig.suptitle(f"One actor trajectory per checkpoint · reconstructed terrain and landing pad (dark bar)\nShared reference prefix through step {probe['flight_step']} (◆), then checkpoint actor; undiscounted reward, ≤1,000 steps",fontsize=12)
    save(fig,out/'branches',plt)
    lines=['# Stage A: one-state action evaluation','',
        'DDPG, actor LN off / critic LN on; training seed 21, evaluation seed 40009.',
        f"State: before action {probe['flight_step']} in the 1M reference flight. Reset and replay the same {probe['flight_step']} actions for every checkpoint.",
        '','| Checkpoint | Actor Q | Actor mean G | Q − G | Critic-best ΔQ | Critic-best ΔG (95% CI) | MC-selected ΔG (95% CI) |',
        '|---|---:|---:|---:|---:|---|---|']
    for r in summary:
        a=r['actor'];c=r['critic_best'];best=r['mc_best']
        def interval(item):
            d=item['delta_g_vs_actor'];return f"{d['mean']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]"
        lines.append(f"| {r['checkpoint']/1000:g}k | {a['q']:.1f} | {a['g']:.1f} | {a['q']-a['g']:+.1f} | {c['delta_q_vs_actor']:+.1f} | {interval(c)} | {interval(best)} |")
    lines += ['', 'Heatmaps use a fixed 9×9 grid plus the exact actor action, five paired future-randomness seeds per action. Highlighted candidates are independently checked with 20 new paired seeds. Intervals describe continuation randomness at this one state, not training-seed uncertainty.',
              '','Diagnostic continuation ends at natural termination or 10,000 steps; censored returns remain finite approximations. Rewards and Q are in original units. No critic prediction fills missing tails.',
              '','The final actor reproduces the original stored reward suffix and states exactly with original future randomness. Branch plots use that original randomness; heatmaps and candidate summaries use repeated future randomness.',
              '','This selected failed-flight state is a case study. Earlier policies may not visit it; each G map follows its own checkpoint policy. Different maps do not by themselves prove a causal training feedback loop.',
              '', 'Report figures: Q and Q − G maps below use independent linear color ranges with numerical colorbars. Compare numbers, not colors, across panels. Sequential colors expose within-map differences; all prediction errors in this case are positive. G estimates still use five paired continuations and can be noisy.',
              '', 'The trajectory panels show only each checkpoint actor after the same reference prefix. Each displayed cumulative reward is the raw, undiscounted sum over that prefix and actor continuation, through termination or the 1,000-step performance boundary. Earlier panels are continuations from the fixed state, not full episodes generated by the earlier actor from reset.',
              '', 'Terrain and landing-pad bounds are reconstructed from evaluation seed 40009, using the same extraction as report/scripts/build_stage_a_failures.py. Geometry is converted into observation coordinates: the pad surface is at y = −0.09, because normalized y = 0 includes the nominal leg offset above the surface.',
              '', '![Q and prediction-error maps](heatmaps.png)', '', '![Actor trajectories](branches.png)',
              '', 'Additional diagnostic figures (excluded from the report): [Action preferences](action_preferences.png), [Evolution](evolution.png).']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def save(fig, path, plt):
    fig.savefig(path.with_suffix('.png'),dpi=165)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--flight-step',type=int,default=75)
    args=parser.parse_args()
    evaluate(args.run,args.output,workers=args.workers,flight_step=args.flight_step)


if __name__=='__main__':main()
