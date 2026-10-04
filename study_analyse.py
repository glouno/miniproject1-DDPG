"""Aggregate independent training seeds; auto-run after a completed study."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from study_diagnostics import atomic_json


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(419)
    samples=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(samples,[.025,.975]).tolist()


def analyse(root):
    manifest=json.loads((root/'manifest.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    runs=[]
    for job in manifest['jobs']:
        directory=root/job['directory'];summary=json.loads((directory/'summary.json').read_text())
        assert summary['status']=='complete' and summary['transitions']==manifest['config']['max_steps']
        metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        expected=list(range(0,manifest['config']['max_steps']+1,manifest['config']['eval_interval']))
        if expected[-1]!=manifest['config']['max_steps']:
            expected.append(manifest['config']['max_steps'])
        assert [m['step'] for m in metrics]==expected
        runs.append(dict(job=job,metrics=metrics))
    n=len(manifest['seeds']);steps=np.array(expected)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    columns=[('on_policy','return','Episode return (undiscounted)'),('on_policy','bias_q1','Q1 − discounted MC'),('on_policy','rmse_q1','Mean episode Q1 RMSE'),
             ('common_probe','bias_q1','Common-probe Q1 − MC'),('common_probe','rmse_q1','Common-probe Q1 RMSE'),('on_policy','horizon_delta_max','Max observed horizon sensitivity')]
    for ax,(kind,key,title) in zip(axes.flat,columns):
        for algo in ('ddpg','td3'):
            for ln in (False,True):
                selected=[r for r in runs if r['job']['algorithm']==algo and r['job']['layer_norm']==ln]
                values=np.array([[m[kind][key] for m in r['metrics']] for r in selected]).T
                rng=np.random.default_rng(419);indices=rng.integers(0,n,(10000,n))
                draws=values[:,indices].mean(axis=2)
                lo,hi=np.quantile(draws,[.025,.975],axis=1)
                line=ax.plot(steps,values.mean(axis=1),label=f'{algo.upper()} LN {"on" if ln else "off"}')[0]
                ax.fill_between(steps,lo,hi,color=line.get_color(),alpha=.14)
        ax.set_title(title);ax.set_xlabel('Training transitions');ax.grid(alpha=.2)
        if key=='bias_q1':ax.axhline(0,color='black',lw=.6)
    axes[0,0].legend(fontsize=8);fig.suptitle(f'{n} training seeds/condition; shaded: pointwise 95% seed-bootstrap intervals')
    fig.tight_layout();fig.savefig(out/'learning_and_calibration.png',dpi=160);fig.savefig(out/'learning_and_calibration.pdf');plt.close(fig)
    rows=[]
    for algo in ('ddpg','td3'):
        for ln in (False,True):
            selected=[r for r in runs if r['job']['algorithm']==algo and r['job']['layer_norm']==ln]
            row=dict(algorithm=algo,layer_norm=ln,seeds=n)
            for kind,key,_ in columns[:-1]:
                mean,lo,hi=interval([r['metrics'][-1][kind][key] for r in selected])
                row[f'{kind}_{key}']=mean;row[f'{kind}_{key}_low']=lo;row[f'{kind}_{key}_high']=hi
            auc=[np.trapezoid([m['on_policy']['return'] for m in r['metrics']],steps)/steps[-1] for r in selected]
            row['normalized_auc']=float(np.mean(auc));rows.append(row)
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    effects=[]
    for algo in ('ddpg','td3'):
        for kind,key,_ in columns[:-1]:
            differences=[]
            for seed in manifest['seeds']:
                pair={r['job']['layer_norm']:r['metrics'][-1][kind][key] for r in runs if r['job']['algorithm']==algo and r['job']['seed']==seed}
                differences.append(pair[True]-pair[False])
            mean,lo,hi=interval(differences)
            effects.append(dict(algorithm=algo,metric=f'{kind}_{key}',on_minus_off=mean,low=lo,high=hi,seed_differences=differences))
    atomic_json(out/'paired_effects.json',effects)
    lines=['# Long-study working results','',f'{len(runs)} completed runs; {n} independent training seeds per condition; {steps[-1]:,} steps/run.',
           'This is local analysis, not a replacement for the shared Samskrift report.','',
           '| Algorithm | LN | Final return | On-policy signed Q1 error | On-policy RMSE | Common-probe signed Q1 error |',
           '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['algorithm'].upper()} | {row['layer_norm']} | {row['on_policy_return']:.1f} | {row['on_policy_bias_q1']:.1f} | {row['on_policy_rmse_q1']:.1f} | {row['common_probe_bias_q1']:.1f} |")
    lines+=['','## Paired normalization differences','', 'Values are LN on minus LN off; brackets are descriptive 95% percentile-bootstrap intervals over paired training seeds.','']
    for effect in effects:
        lines.append(f"- {effect['algorithm'].upper()} {effect['metric']}: {effect['on_minus_off']:.2f} [{effect['low']:.2f}, {effect['high']:.2f}].")
    lines+=['','Intervals are pointwise/descriptive, do not adjust for multiple comparisons, and are imprecise with small seed counts.',
            'Trajectories and sampled pairs are not independent training replicates. No causal effect is inferred from bias/performance associations.',
            'Common probes fix the physical starting state and first action; subsequent actions use each frozen current policy.',
            'Censored Monte Carlo values are finite observed suffixes; horizon sensitivity and censor counts are in raw metrics.','',
            'Inspect individual seeds, convergence and horizon sensitivity before interpreting endpoint differences or adding an ablation.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(completed_runs=len(runs),seeds_per_condition=n,steps=steps.tolist(),status='passed'))
    print('ANALYSIS COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    analyse(parser.parse_args().input.resolve())
