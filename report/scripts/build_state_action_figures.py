"""Build the two report figures for the seed-21 fixed-state case study.

Run from the repository root:
    uv run --locked python report/scripts/build_state_action_figures.py
Inputs: generated/stage-a-seed21-eval40009-state75-20261009/{state.json,
summary.json,values-*.npz,branch-*-actor.npz,landing_geometry.npz} and the
original final on-policy diagnostic file. Outputs: report/generated/state-action/.
No training, checkpoints or rollout measurements are changed.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np

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
                               cmap=cmap,vmin=low,vmax=high,rasterized=False)
            for name, marker, color, size in MARKERS:
                point=ax.scatter(*summary[step][name]['action'],marker=marker,c=color,s=size,
                           edgecolors='black',linewidths=.55,zorder=5,clip_on=False)
                if name == 'mc_best':
                    point.set_path_effects([pe.Stroke(linewidth=2.3,foreground='white'),pe.Normal()])
            ax.axvline(0,color='white',ls=':',lw=.5,alpha=.65)
            for threshold in (-.5,.5):ax.axhline(threshold,color='white',ls=':',lw=.5,alpha=.65)
            ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xticks=[-1,0,1],yticks=[-1,0,1])
            ax.tick_params(length=2,pad=2)
            ax.text(0,1.035,f'({chr(97+4*row+col)})',transform=ax.transAxes,fontsize=8)
            bar=fig.colorbar(image,ax=ax,orientation='horizontal',pad=.045,fraction=.06,aspect=22,
                             ticks=[low,(low+high)/2,high],format='%.1f')
            bar.ax.tick_params(labelsize=6.5,length=2,pad=2)
            label=f'{quantity} · range width {high-low:.1f}'
            if row:
                sd=v['sd'][:-1]
                label+=f'\nMC SD: {sd.min():.1f}–{sd.max():.1f}'
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
    export(fig,'ddpg-state75-heatmaps')

    with np.load(ROOT / 'outputs/mac-stage-a-failure-input/mac-stage-a-20261008/A/ddpg-a0-c1-seed21/diagnostics/on_policy-1000000.npz') as d:
        a,b=d['trajectory_offsets'][9:11]
        prefix=d['trajectory_obs'][int(a):int(b)][:probe['flight_step']].copy()
    with np.load(DATA / 'landing_geometry.npz') as g:
        # Invert the observation scaling, using metres relative to the pad surface.
        terrain=g['terrain'] * [10,20/3] + [0,.6]
        pad=g['pad_x'] * 10
        pad_height=float(g['pad_y'] * (20/3) + .6)
    np.testing.assert_allclose(pad_height,0,atol=1e-12)
    fig,axes=plt.subplots(1,4,figsize=(6.8,2.0),layout='constrained',sharex=True,sharey=True)
    floor=min(-.8,float(terrain[:,1].min())-.3)
    for col,(ax,step) in enumerate(zip(axes,STEPS)):
        with np.load(DATA / f'branch-{step:07d}-actor.npz') as b:
            remaining=1000-probe['flight_step']
            trajectory=np.vstack((prefix,b['observations'][:remaining+1]))
            reward=float(probe['prefix_return']+b['rewards'][:remaining].sum())
        np.testing.assert_allclose(reward,summary[step]['actor']['illustration']['performance_return'],atol=1e-10)
        xy=trajectory[:,:2] * [10,20/3] + [0,.6]
        fixed=np.asarray(probe['obs'][:2]) * [10,20/3] + [0,.6]
        ax.fill_between(terrain[:,0],floor,terrain[:,1],color='0.94',zorder=0)
        ax.plot(terrain[:,0],terrain[:,1],color='0.65',lw=.7,zorder=1)
        ax.plot(pad,[0,0],color='0.2',lw=3,solid_capstyle='butt',zorder=2)
        ax.plot(xy[:,0],xy[:,1],color='#ad3f36',lw=1.25,zorder=3)
        ax.scatter(*xy[0],facecolors='white',edgecolors='0.25',s=13,zorder=5)
        ax.scatter(*xy[-1],marker='X',color='#ad3f36',s=23,zorder=5)
        ax.scatter(*fixed,marker='s',color='black',s=18,zorder=6)
        if col==0:ax.annotate('Step 75',fixed,xytext=(7,0),textcoords='offset points',
                              va='center',fontsize=7,zorder=7)
        for boundary in (-10,10):ax.axvline(boundary,color='0.65',ls='--',lw=.6,zorder=0)
        ax.set(xlim=(-10.7,10.7),ylim=(floor,10.65),xticks=[-10,0,10],yticks=[0,5,10])
        ax.grid(alpha=.15,lw=.5);ax.tick_params(length=2,pad=2)
        ax.text(.03,.98,f'({chr(97+col)})\n$R={reward:.1f}$',transform=ax.transAxes,va='top',fontsize=8,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
        stats['checkpoints'].append({'step':step,'displayed_return':reward,'ending':summary[step]['actor']['illustration']['ending']})
    fig.supxlabel('Horizontal position from pad centre (m)',fontsize=8)
    fig.supylabel('Height above pad surface (m)',fontsize=8)
    export(fig,'ddpg-state75-trajectories')

    captions = r'''\subsection{A fixed-state case study}
\qquad Figures~\ref{fig:ddpg-state75-values} and~\ref{fig:ddpg-state75-trajectories} examine one state from a selected failed final-policy flight. They show how value calibration and actor continuations differ across checkpoints; this single-state comparison does not establish a causal training mechanism.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/ddpg-state75-heatmaps.png}
\caption{Action-value maps at the state before action 75 of the final Mac Stage A DDPG flight with training seed 21 and evaluation seed 40009, critic LN on and actor LN off. Columns correspond, from left to right, to 20k, 50k, 300k and 1M training transitions. Panels (a)--(d) show the frozen online critic $Q$; panels (e)--(h) show $Q-\bar G$. Each $G$ rollout forces the plotted action once, then follows that checkpoint's frozen actor; returns include the first reward, use $\gamma=0.99$, and stop at natural termination or 10,000 continuation steps, without critic-filled tails. The grid has nine equally spaced values per action coordinate; the exact actor action is evaluated separately. $\bar G$ averages five paired future-randomness seeds per action; the reported MC SD range is the minimum--maximum sample SD across grid cells, measuring continuation variability, not training-seed uncertainty. Every panel uses its own linear colour range in original reward units: compare colourbar numbers, rather than colours, across panels. Stars, triangles and circles mark the actor action, grid maximum of $Q$, and grid maximum of $\bar G$, respectively. Dashed guides mark engine activation thresholds. The critic is optimistic throughout the grid; large absolute error alone does not demonstrate harmful action misranking. This is an exploratory case study with no significance test.}
\label{fig:ddpg-state75-values}
\end{figure}
\FloatBarrier

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/ddpg-state75-trajectories.pdf}
\caption{One actor continuation from the same fixed state for each checkpoint: (a) 20k, (b) 50k, (c) 300k and (d) 1M training transitions. All trajectories share the final reference flight's first 75 transitions; at the square marker, each follows its own frozen checkpoint actor without exploration noise, using the original continuation randomness. Only (d) is the complete original final-policy flight; the earlier panels are continuations from the reference state, rather than full episodes generated by those actors from reset. $R$ is the raw, undiscounted cumulative reward of the specific displayed trajectory, including the shared prefix, through natural termination or the 1,000-transition performance boundary. Axes give physical metres, with horizontal origin at the pad centre and vertical origin at its surface. Grey shading and the dark bar show the seed-specific terrain and pad; dashed verticals mark exit boundaries. Circles mark starts, squares the intervention state, and crosses termination. Panels (a) and (c) end in collision; (b) and (d) exit the boundary. Each curve is one illustrative realization, not a stochastic mean; the final-policy boundary exit occurs at transition 213.}
\label{fig:ddpg-state75-trajectories}
\end{figure}
\FloatBarrier
'''
    (OUT / 'state-action.tex').write_text(captions)
    stats['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stats['inputs']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [DATA/'summary.json',DATA/'landing_geometry.npz',*sorted(DATA.glob('values-*.npz')),*sorted(DATA.glob('branch-*-actor.npz'))]}
    (OUT/'figure-statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(OUT)


if __name__=='__main__':main()
