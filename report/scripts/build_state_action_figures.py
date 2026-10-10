"""Build the two report figures for the seed-21 fixed-state case study.

Run from the repository root:
    uv run --locked python report/scripts/build_state_action_figures.py
Inputs: generated/stage-a-seed21-eval40009-state75-20261009/{state.json,
summary.json,values-*.npz,branch-*-actor.npz,landing_geometry.npz} and the
original final on-policy diagnostic file and the four checkpoints listed in the
evaluation provenance. Outputs: report/generated/state-action/.
No training, checkpoints or rollout measurements are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import torch

from lunarlander_rl.experiment import discounted_returns, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'generated/stage-a-seed21-eval40009-state75-20261009'
OUT = ROOT / 'report/generated/state-action'
STEPS = (20000, 50000, 300000, 1000000)
MARKERS = (('actor', '*', 'white', 38), ('critic_best', '^', '#44dbf5', 22),
           ('mc_best', 'o', '#00e676', 18))


def export(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    global DATA, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--name-prefix',default='ddpg-state75')
    parser.add_argument('--bias-limit',type=float,
                        help='Shared symmetric range for comparing trajectory figures')
    args=parser.parse_args()
    DATA, OUT=args.data.resolve(),args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'xtick.labelsize':7,
                         'ytick.labelsize':7, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    study = json.loads((DATA / 'summary.json').read_text())
    probe = study['state']
    summary = {r['checkpoint']:r for r in study['checkpoints']}
    values = {step:dict(np.load(DATA / f'values-{step:07d}.npz')) for step in STEPS}
    stats = {'checkpoints':[], 'heatmap_panel_ranges':[], 'sd_measure':'sample SD of five paired Monte Carlo continuations',
             'trajectory_frame':'physical metres; horizontal origin pad centre, vertical origin pad surface'}
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 4.05), layout='constrained', sharex=True, sharey=True)
    for col, step in enumerate(STEPS):
        v = values[step]; n = len(v['axis'])
        for row, (data, quantity, cmap) in enumerate(((v['q'][:-1], 'Q', 'viridis'),
                (v['q'][:-1]-v['g'][:-1], 'Q − G', 'magma'))):
            ax=axes[row,col]; low,high=float(data.min()),float(data.max())
            image=ax.pcolormesh(v['axis'],v['axis'],data.reshape(n,n),shading='nearest',
                               cmap=cmap,vmin=low,vmax=high,rasterized=False,
                               edgecolors='none',antialiased=False)
            for name, marker, color, size in MARKERS:
                point=ax.scatter(*summary[step][name]['action'],marker=marker,c=color,s=size,
                           edgecolors='black',linewidths=.55,zorder=5,clip_on=False)
                if name == 'mc_best':
                    point.set_path_effects([pe.Stroke(linewidth=2.3,foreground='white'),pe.Normal()])
            ax.grid(False)
            ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xticks=[-1,0,1],yticks=[-1,0,1])
            ax.tick_params(length=2,pad=2)
            checkpoint='1M' if step == 1000000 else f'{step/1000:g}k'
            ax.set_title(f'({chr(97+4*row+col)}) {checkpoint} transitions',fontsize=8,pad=5)
            bar=fig.colorbar(image,ax=ax,orientation='horizontal',pad=.045,fraction=.06,aspect=22,
                             ticks=[low,(low+high)/2,high],format='%.1f')
            bar.ax.tick_params(labelsize=6.5,length=2,pad=2)
            label=f'{quantity}, range width {high-low:.1f}'
            bar.set_label(label,fontsize=6.8,labelpad=2)
            stats['heatmap_panel_ranges'].append({'checkpoint':step,'quantity':quantity,'min':low,'max':high,
                'mc_sd_min':float(v['sd'][:-1].min()),'mc_sd_max':float(v['sd'][:-1].max())})
    fig.supxlabel('Main-engine command',fontsize=8)
    fig.supylabel('Lateral-engine command',fontsize=8)
    labels={'actor':'Actor','critic_best':'Q best','mc_best':'MC best'}
    handles=[]
    for name,marker,color,_ in MARKERS:
        handle=Line2D([],[],marker=marker,linestyle='none',markerfacecolor=color,
                      markeredgecolor='black',markeredgewidth=.6,markersize=5,label=labels[name])
        if name == 'mc_best':
            handle.set_path_effects([pe.Stroke(linewidth=2.3,foreground='white'),pe.Normal()])
        handles.append(handle)
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,
               fontsize=8,handletextpad=.5,columnspacing=1.8)
    export(fig,f'{args.name_prefix}-heatmaps')

    if (DATA/'reference_trajectory.npz').exists():
        with np.load(DATA/'reference_trajectory.npz') as d:
            prefix=d['observations'][:probe['flight_step']].copy()
    else:
        with np.load(ROOT / 'outputs/mac-stage-a-failure-input/mac-stage-a-20261008/A/ddpg-a0-c1-seed21/diagnostics/on_policy-1000000.npz') as d:
            a,b=d['trajectory_offsets'][9:11]
            prefix=d['trajectory_obs'][int(a):int(b)][:probe['flight_step']].copy()
    with np.load(DATA / 'landing_geometry.npz') as g:
        # Invert the observation scaling, using metres relative to the pad surface.
        terrain=g['terrain'] * [10,20/3] + [0,.6]
        pad=g['pad_x'] * 10
        pad_height=float(g['pad_y'] * (20/3) + .6)
    np.testing.assert_allclose(pad_height,0,atol=1e-12)
    provenance=json.loads((DATA/'provenance.json').read_text())
    branches={}
    torch.set_num_threads(1)
    for step in STEPS:
        with np.load(DATA / f'branch-{step:07d}-actor.npz') as b:
            observations=b['observations'].copy()
            actions=b['actions'].copy()
            rewards=b['rewards'].copy()
        cfg,actor,critics,_=load_checkpoint(provenance['checkpoints'][str(step)]['path'])
        with torch.inference_mode():
            # Replay used one observation per actor call; match that floating-point path.
            actor_actions=np.stack([actor(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
                                    for obs in observations[:-1]])
            np.testing.assert_allclose(actor_actions,actions,atol=2e-6,rtol=1e-5)
            q=critics[0](torch.from_numpy(observations[:-1]),torch.from_numpy(actions)).numpy()/cfg.reward_scale
        np.testing.assert_allclose(q[0],values[step]['q'][-1],atol=1e-4,rtol=1e-6)
        g=discounted_returns(rewards,cfg.gamma)
        np.testing.assert_allclose(g[0],summary[step]['actor']['illustration']['mc'],atol=1e-9,rtol=1e-12)
        bias=q-g
        branches[step]=dict(observations=observations,rewards=rewards,q=q,g=g,bias=bias)
        np.savez_compressed(OUT/f'trajectory-values-{step:07d}.npz',q=q,g=g,bias=bias)
    error_max=max(float(np.abs(b['bias'][:1000-probe['flight_step']]).max()) for b in branches.values())
    limit=args.bias_limit if args.bias_limit is not None else max(50,50*np.ceil(error_max/50))
    assert limit > 0 and limit >= error_max, 'Bias range must contain every displayed segment'
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    cmap=plt.get_cmap('RdBu_r')
    stats['trajectory_coloring']={'quantity':'online Q minus realized discounted reward suffix G',
        'gamma':cfg.gamma,'range':[-float(limit),float(limit)],'shared_prefix':'grey, not assessed',
        'stochastic_repetitions':1,'continuation_values':'natural termination or stored diagnostic cutoff'}
    fig,axes=plt.subplots(1,4,figsize=(6.8,2.0),layout='constrained',sharex=True,sharey=True)
    floor=min(-.8,float(terrain[:,1].min())-.3)
    for col,(ax,step) in enumerate(zip(axes,STEPS)):
        b=branches[step]
        remaining=1000-probe['flight_step']
        trajectory=np.vstack((prefix,b['observations'][:remaining+1]))
        reward=float(probe['prefix_return']+b['rewards'][:remaining].sum())
        np.testing.assert_allclose(reward,summary[step]['actor']['illustration']['performance_return'],atol=1e-10)
        xy=trajectory[:,:2] * [10,20/3] + [0,.6]
        fixed=np.asarray(probe['obs'][:2]) * [10,20/3] + [0,.6]
        ax.fill_between(terrain[:,0],floor,terrain[:,1],color='0.94',zorder=0)
        ax.plot(terrain[:,0],terrain[:,1],color='0.65',lw=.7,zorder=1)
        ax.plot(pad,[0,0],color='0.2',lw=3,solid_capstyle='butt',zorder=2)
        join=probe['flight_step']
        ax.plot(xy[:join+1,0],xy[:join+1,1],color='0.55',lw=1.25,zorder=3)
        continuation=xy[join:]
        ax.plot(continuation[:,0],continuation[:,1],color='0.35',lw=2.0,
                solid_capstyle='round',zorder=3)
        segments=np.stack((continuation[:-1],continuation[1:]),axis=1)
        colored=LineCollection(segments,cmap=cmap,norm=norm,linewidth=1.6,capstyle='round',zorder=4)
        colored.set_array(b['bias'][:len(segments)])
        ax.add_collection(colored)
        ax.scatter(*xy[0],facecolors='white',edgecolors='0.25',s=13,zorder=5)
        ax.scatter(*xy[-1],marker='X',color=cmap(norm(b['bias'][len(segments)-1])),
                   edgecolors='0.2',linewidths=.4,s=23,zorder=5)
        ax.scatter(*fixed,marker='s',color='black',s=18,zorder=6)
        if col==0:ax.annotate(f"Step {probe['flight_step']}",fixed,xytext=(7,0),textcoords='offset points',
                              va='center',fontsize=7,zorder=7)
        for boundary in (-10,10):ax.axvline(boundary,color='0.65',ls='--',lw=.6,zorder=0)
        ax.set(xlim=(-10.7,10.7),ylim=(floor,10.65),xticks=[-10,0,10],yticks=[0,5,10])
        ax.grid(alpha=.15,lw=.5);ax.tick_params(length=2,pad=2)
        checkpoint='1M' if step == 1000000 else f'{step/1000:g}k'
        ending=summary[step]['actor']['illustration']['ending']
        outcome=('timeout' if summary[step]['actor']['illustration']['performance_timeout'] else
                 {'game_over':'collision','out_of_bounds':'boundary exit','lander_asleep':'rest','time_limit':'timeout'}.get(ending,ending))
        ax.set_title(f'({chr(97+col)}) {checkpoint}, {outcome}\nCumulative reward = {reward:.1f}',fontsize=7.5,pad=5)
        stats['checkpoints'].append({'step':step,'displayed_return':reward,'ending':summary[step]['actor']['illustration']['ending'],
            'bias_min':float(b['bias'][:remaining].min()),'bias_max':float(b['bias'][:remaining].max())})
    colorbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,
                         orientation='horizontal',fraction=.10,pad=.03,shrink=.65,aspect=45,
                         ticks=[-limit,0,limit])
    colorbar.ax.tick_params(labelsize=7,length=2,pad=1)
    colorbar.set_label('Bias',fontsize=7,labelpad=2)
    fig.supxlabel('Horizontal position from pad centre (m)',fontsize=8)
    fig.supylabel('Height above pad surface (m)',fontsize=8)
    export(fig,f'{args.name_prefix}-trajectories')

    captions = r'''\subsection{A fixed-state case study}
\qquad Figures~\ref{fig:ddpg-state75-values} and~\ref{fig:ddpg-state75-trajectories} examine one state from a selected failed final-policy flight. They show how value calibration and actor continuations differ across checkpoints; this single-state comparison does not establish a causal training mechanism.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/ddpg-state75-heatmaps.png}
\caption{Action-value maps at the state before action 75 of the final Mac Stage A DDPG flight with training seed 21 and evaluation seed 40009, critic LN on and actor LN off. Columns correspond, from left to right, to 20k, 50k, 300k and 1M training transitions. Panels (a)--(d) show the frozen online critic $Q$; panels (e)--(h) show $Q-\bar G$. Each $G$ rollout forces the plotted action once, then follows that checkpoint's frozen actor; returns include the first reward, use $\gamma=0.99$, and stop at natural termination or 10,000 continuation steps, without critic-filled tails. The grid has nine equally spaced values per action coordinate; the exact actor action is evaluated separately. $\bar G$ averages five paired future-randomness seeds per action. Every panel uses its own linear colour range in original reward units: compare colourbar numbers, rather than colours, across panels. Stars, triangles and circles mark the actor action, grid maximum of $Q$, and grid maximum of $\bar G$, respectively. The critic is optimistic throughout the grid; large absolute error alone does not demonstrate harmful action misranking. This is an exploratory case study with no significance test.}
\label{fig:ddpg-state75-values}
\end{figure}
\FloatBarrier

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/ddpg-state75-trajectories.pdf}
\caption{One actor continuation from the same fixed state for each checkpoint: (a) 20k, (b) 50k, (c) 300k and (d) 1M training transitions. All trajectories share the final reference flight's first 75 transitions; at the square marker, each follows its own frozen checkpoint actor without exploration noise, using the original continuation randomness. Only (d) is the complete original final-policy flight; the earlier panels are continuations from the reference state, rather than full episodes generated by those actors from reset. $R$ is the raw, undiscounted cumulative reward of the specific displayed trajectory, including the shared prefix, through natural termination or the 1,000-transition performance boundary. Axes give physical metres, with horizontal origin at the pad centre and vertical origin at its surface. Grey shading and the dark bar show the seed-specific terrain and pad; dashed verticals mark exit boundaries. Circles mark starts, squares the intervention state, and crosses termination. Panels (a) and (c) end in collision; (b) and (d) exit the boundary. After the square marker, segment colour shows the frozen online critic's $Q(s_t,a_t)-G_t$, where $G_t$ is the discounted remaining reward on that same illustrated trajectory ($\gamma=0.99$), including the current transition. All panels share a symmetric colour scale: red indicates overestimation and blue underestimation. The shared prefix is grey and is not assessed; a thin grey outline keeps near-zero errors visible. Each curve and its colour show one realization, not an average over future-randomness seeds; the final-policy boundary exit occurs at transition 213.}
\label{fig:ddpg-state75-trajectories}
\end{figure}
\FloatBarrier
'''
    condition=provenance.get('condition',{'actor_layer_norm':False,'critic_layer_norm':True})
    if not condition['critic_layer_norm']:
        # This control starts from the original LN-on flight, not its own final flight.
        captions = r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/NAME-heatmaps.png}
\caption{DDPG control with actor LN off and critic LN off, training seed 21. All checkpoints are evaluated at the identical physical state before action 75 of the original critic-LN-on final flight, evaluation seed 40009. Columns show 20k, 50k, 300k and 1M transitions; rows show $Q$ and $Q-\bar G$. The 9 by 9 action grid, five paired future-randomness seeds per action, discount $\gamma=0.99$ and diagnostic cutoff of 10,000 continuation steps match the LN-on analysis. Each rollout forces the selected action once and then follows its own checkpoint actor. Numerical colourbars use independent panel ranges in original reward units. Stars, triangles and circles mark the actor action, grid maximum of $Q$ and grid maximum of $\bar G$. This is one selected state, not a training-seed population comparison.}
\label{fig:NAME-values}
\end{figure}
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/NAME-trajectories.pdf}
\caption{One continuation per LN-off checkpoint from the same reference state. The grey prefix comprises the original critic-LN-on flight's first 75 transitions; the square marks the switch to the corresponding LN-off actor. These are controlled continuations, not full LN-off flights from reset. Subtitles report the undiscounted cumulative reward of each displayed trajectory including the prefix, through termination or the 1,000-transition performance boundary. Axes give metres relative to the pad centre and surface; terrain and the dark pad bar are reconstructed from evaluation seed 40009. Segment colour after the square shows $Q(s_t,a_t)-G_t$, using the realized discounted remaining rewards of that same trajectory. The symmetric bias colour scale matches the LN-on figure; red indicates overestimation, blue underestimation. No exploration noise is added; original continuation randomness is reused. Each trajectory is one realization.}
\label{fig:NAME-trajectories}
\end{figure}
'''.replace('NAME',args.name_prefix)
    captions = captions.replace('ddpg-state75', args.name_prefix)
    captions = captions.replace('action 75', f"action {probe['flight_step']}")
    captions = captions.replace('first 75 transitions', f"first {probe['flight_step']} transitions")
    captions = captions.replace('Panels (a) and (c) end in collision; (b) and (d) exit the boundary.',
                                'Subtitles give the ending of each displayed continuation.')
    if any(float((values[step]['q'][:-1]-values[step]['g'][:-1]).min()) <= 0 for step in STEPS):
        captions = captions.replace('The critic is optimistic throughout the grid;',
                                    'The colourbar numbers describe the sign and magnitude of error;')
    (OUT / 'state-action.tex').write_text(captions)
    stats['condition']=condition
    stats['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stats['checkpoints_provenance']=provenance['checkpoints']
    stats['inputs']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [DATA/'summary.json',DATA/'landing_geometry.npz',*sorted(DATA.glob('values-*.npz')),*sorted(DATA.glob('branch-*-actor.npz'))]}
    (OUT/'figure-statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(OUT)


if __name__=='__main__':main()
