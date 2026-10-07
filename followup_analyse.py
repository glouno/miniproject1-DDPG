"""Seed-level factorial analysis; also works on completed subsets."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from study_diagnostics import atomic_json


def paired(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(419)
    lo,hi=np.quantile(x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1),[.025,.975])
    signs=np.array(list(itertools.product((-1,1),repeat=len(x))))
    p=float((np.abs((signs*x).mean(axis=1))>=abs(x.mean())-1e-12).mean())
    return dict(mean=float(x.mean()),ci95=[float(lo),float(hi)],p_raw=p,n=len(x),differences=x.tolist())


def holm(items):
    previous=0.
    for i,key in enumerate(sorted(items,key=lambda k:items[k]['p_raw'])):
        previous=max(previous,min(1.,items[key]['p_raw']*(len(items)-i)))
        items[key]['p_holm']=previous


def analyse(root,stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((root/'manifest.json').read_text())
    out=root/'analysis'/stage;out.mkdir(parents=True,exist_ok=True)
    groups={};by_seed={}
    for j in manifest['jobs']:
        d=root/j['directory']
        if j['stage']!=stage or not (d/'summary.json').exists():continue
        metrics=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]
        groups.setdefault(j['cell'],[]).append((j,metrics))
        by_seed[j['cell'],j['config']['seed']]=metrics[-1]['on_policy']['return']
    if not groups:return
    conditions=list(groups);algorithms=('ddpg','td3') if stage=='A' else ('ddpg',)
    fig,axes=plt.subplots(len(algorithms),3,figsize=(15,4*len(algorithms)),squeeze=False)
    summaries={}
    for name,runs in groups.items():
        steps=[m['step'] for m in runs[0][1]]
        row=algorithms.index(runs[0][0]['config']['algorithm'])
        finals=[m[-1]['on_policy'] for _,m in runs]
        summaries[name]=dict(n=len(runs),return_mean=float(np.mean([f['return'] for f in finals])),
            return_sd=float(np.std([f['return'] for f in finals],ddof=1)) if len(runs)>1 else None,
            bias_mean=float(np.mean([f['bias_q1'] for f in finals])),rmse_mean=float(np.mean([f['rmse_q1'] for f in finals])))
        summaries[name].update(mean_q1=float(np.mean([f['mean_q1'] for f in finals])),
            mean_mc=float(np.mean([f['mean_mc'] for f in finals])),
            censored_episodes=sum(f['censored'] for f in finals),
            horizon_delta_max=max(f['horizon_delta_max'] for f in finals),
            common_bias_mean=float(np.mean([ms[-1]['common_probe']['bias_q1'] for _,ms in runs])),
            common_rmse_mean=float(np.mean([np.sqrt(np.mean([e['rmse_q1']**2 for e in ms[-1]['common_probe']['episodes']])) for _,ms in runs])))
        for col,key in enumerate(('return','bias_q1','rmse_q1')):
            values=np.array([[m['on_policy'][key] for m in ms] for _,ms in runs])
            ax=axes[row,col];line=ax.plot(steps,values.mean(axis=0),label=f'{name} (n={len(runs)})')[0]
            rng=np.random.default_rng(419);idx=rng.integers(0,len(runs),(10000,len(runs)))
            lo,hi=np.quantile(values[idx].mean(axis=1),[.025,.975],axis=0)
            ax.fill_between(steps,lo,hi,color=line.get_color(),alpha=.15)
            ax.set(xlabel='Collected training transitions',title=f'{algorithms[row].upper()} {key}')
            ax.grid(alpha=.2)
    for ax in axes[:,0]:ax.legend(fontsize=7)
    for ax in axes[:,1]:ax.axhline(0,color='black',lw=.7)
    fig.suptitle('Means and pointwise seed-bootstrap intervals; individual runs in TensorBoard')
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);fig.savefig(out/'learning.pdf');plt.close(fig)
    fig,axes=plt.subplots(len(algorithms),3,figsize=(15,4*len(algorithms)),squeeze=False)
    for name,runs in groups.items():
        row=algorithms.index(runs[0][0]['config']['algorithm']);steps=[m['step'] for m in runs[0][1]]
        for col,key in enumerate(('bias_q1','rmse_q1')):
            values=np.array([[m['common_probe'][key] if key=='bias_q1' else np.sqrt(np.mean([e['rmse_q1']**2 for e in m['common_probe']['episodes']])) for m in ms] for _,ms in runs])
            ax=axes[row,col];line=ax.plot(steps,values.mean(axis=0),label=name)[0]
            rng=np.random.default_rng(419);idx=rng.integers(0,len(runs),(10000,len(runs)))
            lo,hi=np.quantile(values[idx].mean(axis=1),[.025,.975],axis=0)
            ax.fill_between(steps,lo,hi,color=line.get_color(),alpha=.15)
            ax.set(xlabel='Training transitions',title=f'{algorithms[row]} common probes {key}')
        axes[row,2].scatter([ms[-1]['on_policy']['bias_q1'] for _,ms in runs],
            [ms[-1]['on_policy']['return'] for _,ms in runs],label=name)
    for row in range(len(algorithms)):
        axes[row,0].axhline(0,color='black',lw=.7);axes[row,0].legend(fontsize=7)
        axes[row,2].set(xlabel='Final on-policy signed bias',ylabel='Final episode return',title='One point per training seed; association only')
        for ax in axes[row]:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=150);fig.savefig(out/'calibration.pdf');plt.close(fig)
    contrasts={};supporting={}
    if stage=='A':
        for algo in algorithms:
            seeds=sorted({s for (cell,s) in by_seed if cell.startswith(algo)})
            names=[f'{algo}-a{a}-c{c}' for a,c in ((0,0),(0,1),(1,0),(1,1))]
            seeds=[s for s in seeds if all((n,s) in by_seed for n in names)]
            if not seeds:continue
            q=lambda a,c,s:by_seed[f'{algo}-a{a}-c{c}',s]
            contrasts[algo+'_critic_LN_actor_off']=paired([q(0,1,s)-q(0,0,s) for s in seeds])
            contrasts[algo+'_actor_LN_critic_on']=paired([q(1,1,s)-q(0,1,s) for s in seeds])
            supporting[algo+'_interaction']=paired([q(1,1,s)-q(0,1,s)-q(1,0,s)+q(0,0,s) for s in seeds])
    else:
        cells={(j['config']['critic_lr'],j['config']['actor_lr']):j['cell'] for j in manifest['jobs']
               if j['stage']=='B' and j['config']['critic_layer_norm']}
        seeds=sorted({s for (_,s) in by_seed})
        seeds=[s for s in seeds if all((n,s) in by_seed for n in cells.values())]
        if seeds:
            q=lambda c,a,s:by_seed[cells[c,a],s]
            contrasts['critic_lr_lower']=paired([np.mean([q(1e-4,a,s)-q(3e-4,a,s) for a in (1e-4,3e-4)]) for s in seeds])
            contrasts['actor_lr_lower']=paired([np.mean([q(c,1e-4,s)-q(c,3e-4,s) for c in (1e-4,3e-4)]) for s in seeds])
            contrasts['interaction']=paired([q(1e-4,1e-4,s)-q(3e-4,1e-4,s)-q(1e-4,3e-4,s)+q(3e-4,3e-4,s) for s in seeds])
    holm(contrasts)
    expected=sum(j['stage']==stage for j in manifest['jobs'])
    atomic_json(out/'summary.json',dict(stage=stage,completed=sum(len(r) for r in groups.values()),expected=expected,
        partial=sum(len(r) for r in groups.values())!=expected,conditions=summaries,primary_return_contrasts=contrasts,
        supporting_interactions=supporting,note='Seed is unit; sign-flip tests assume symmetry/exchangeability. Partial results are monitoring only.'))
    print(f'Analysed {stage}: {sum(len(r) for r in groups.values())}/{expected} runs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--stage',choices=['A','B'],required=True)
    a=p.parse_args();analyse(a.input,a.stage)
