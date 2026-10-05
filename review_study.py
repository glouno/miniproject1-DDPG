"""Post-run interpretation tables and plots; reads the saved frozen study."""
import argparse
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    root=args.input.resolve();manifest=json.loads((root/'manifest.json').read_text());status=json.loads((root/'status.json').read_text());assert status['status']=='complete'
    jobs=[]
    for job in manifest['jobs']:
        run=root/job['directory'];metrics=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        summary=json.loads((run/'summary.json').read_text());assert summary['transitions']==manifest['config']['max_steps']
        jobs.append(dict(job=job,metrics=metrics,summary=summary))
    z=ZoneInfo('Europe/Paris');start=datetime.fromisoformat(status['started_at']);end=datetime.fromtimestamp((root/'status.json').stat().st_mtime,z)
    attempts=[datetime.fromisoformat(l.split(maxsplit=2)[2]) for p in (root/'logs').glob('*.log') for l in p.read_text().splitlines() if l.startswith('ATTEMPT ')]
    runtime=dict(main_batch_seconds=status['elapsed_seconds'],total_elapsed_seconds=(end-min(attempts)).total_seconds(),
                 main_batch_start=start.astimezone(z).isoformat(),training_end=(start+timedelta(seconds=status['elapsed_seconds'])).astimezone(z).isoformat(),
                 first_attempt=min(attempts).astimezone(z).isoformat(),finished_at=end.isoformat(),
                 mean_run_minutes=float(np.mean([j['summary']['elapsed_seconds'] for j in jobs])/60),
                 cumulative_worker_wall_hours=sum(j['summary']['elapsed_seconds'] for j in jobs)/3600,
                 note='Worker wall-hours are not measured CPU-hours; they include diagnostics and I/O.')
    rows=[]
    for algo in ('ddpg','td3'):
        for ln in (False,True):
            selected=[j for j in jobs if j['job']['algorithm']==algo and j['job']['layer_norm']==ln]
            final=[j['metrics'][-1] for j in selected];v=np.array([m['on_policy']['return'] for m in final]);v300=np.array([next(m for m in j['metrics'] if m['step']==300000)['on_policy']['return'] for j in selected])
            row=dict(algorithm=algo,layer_norm=ln,return_mean=float(v.mean()),return_seed_sd=float(v.std(ddof=1)),individual_returns=v.tolist(),
                     return_300k=float(v300.mean()),change_300k_500k=float(v.mean()-v300.mean()),seeds_above_200=int(sum(v>=200)),
                     final_censored=sum(m['on_policy']['censored'] for m in final),
                     final_probe_censored=sum(m['common_probe']['censored'] for m in final),
                     all_checkpoint_horizon_delta_max=max(m[kind]['horizon_delta_max'] for j in selected for m in j['metrics'] for kind in ('on_policy','common_probe')))
            for kind in ('on_policy','common_probe'):
                row[f'{kind}_bias']=float(np.mean([m[kind]['bias_q1'] for m in final]))
                row[f'{kind}_rmse']=float(np.mean([np.sqrt(np.mean([ep['rmse_q1']**2 for ep in m[kind]['episodes']])) if kind=='common_probe' else m[kind]['rmse_q1'] for m in final]))
            rows.append(row)
    effects=json.loads((root/'analysis/paired_effects.json').read_text());effects_return=[e for e in effects if e['metric']=='on_policy_return']
    td3=next(e for e in effects_return if e['algorithm']=='td3');diff=np.array(td3['seed_differences']);largest=np.argmax(diff)
    extra=dict(td3_median_paired_return_change=float(np.median(diff)),td3_return_improved_seeds=int(sum(diff>0)),
               td3_largest_paired_improvement=float(diff[largest]),td3_largest_improvement_seed=manifest['seeds'][largest],
               td3_mean_change_excluding_largest=float(np.delete(diff,largest).mean()))
    out=root/'analysis';(out/'review.json').write_text(json.dumps(dict(runtime=runtime,conditions=rows,interpretation_checks=extra),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1.45,1]})
    ax=axes[0]
    for index,row in enumerate(rows):
        selected=[j for j in jobs if j['job']['algorithm']==row['algorithm'] and j['job']['layer_norm']==row['layer_norm']]
        steps=np.array([m['step'] for m in selected[0]['metrics']]);v=np.array([[m['on_policy']['return'] for m in j['metrics']] for j in selected]).T
        rng=np.random.default_rng(419);indices=rng.integers(0,len(selected),(10000,len(selected)));samples=v[:,indices].mean(axis=2);lo,hi=np.quantile(samples,[.025,.975],axis=1)
        ax.plot(steps/1000,v.mean(axis=1),color=f'C{index}',label=f"{row['algorithm'].upper()} · LN {'on' if row['layer_norm'] else 'off'}")
        ax.fill_between(steps/1000,lo,hi,color=f'C{index}',alpha=.15)
    ax.set(xlabel='Training steps (thousands)',ylabel='Mean episode return',title='Learning curves · 10 training seeds')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1]
    for x,e in enumerate(effects_return):
        d=np.array(e['seed_differences']);jitter=np.linspace(-.12,.12,len(d))
        ax.scatter(x+jitter,d,color=f'C{1 if x==0 else 3}',label='Individual paired seeds' if x==0 else None)
        ax.errorbar(x,e['on_minus_off'],yerr=[[e['on_minus_off']-e['low']],[e['high']-e['on_minus_off']]],fmt='D',color='black',capsize=5,label='Mean and 95% bootstrap interval' if x==0 else None)
    ax.axhline(0,color='grey',lw=1);ax.set(xticks=[0,1],xticklabels=['DDPG','TD3'],ylabel='Final return change: LN on − off',title='Normalization effect at 500k')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'review_learning.png',dpi=170);fig.savefig(out/'review_learning.pdf');plt.close(fig)
    lines=['# Long-study interpretation','',f"All {len(jobs)} runs completed, {manifest['config']['max_steps']:,} training transitions each. Main batch: {status['elapsed_seconds']/60:.2f} minutes; total elapsed including restart/analysis: {runtime['total_elapsed_seconds']/60:.2f} minutes.",
           'Times are wall-clock elapsed times, not measured CPU time. Runtime and per-seed details are in review.json.','',
           '| Condition | Final return ± training-seed SD | Return at 300k | On-policy Q1−MC | Mean episode RMSE | Common-probe Q1−MC | Common-probe pooled RMSE |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['algorithm'].upper()} LN {'on' if r['layer_norm'] else 'off'} | {r['return_mean']:.1f} ± {r['return_seed_sd']:.1f} | {r['return_300k']:.1f} | {r['on_policy_bias']:.1f} | {r['on_policy_rmse']:.1f} | {r['common_probe_bias']:.1f} | {r['common_probe_rmse']:.1f} |")
    lines+=['','DDPG normalization lowered final performance and raised optimism/calibration error in every paired seed. The same direction persists on common probes, so changed initial state/action visitation alone does not explain it. Continuation policies still differ.',
            'TD3 normalization has the best mean final return and lower between-seed spread, but the paired performance interval includes zero. Seven seeds improve; one large rescued-seed gain drives most of the mean difference. The other nine seeds average only '+f"{extra['td3_mean_change_excluding_largest']:.1f} points improvement.",
            'TD3 critics are slightly pessimistic on policy. Normalization raises their predictions toward the observed returns: this reduces underestimation, rather than reducing positive overestimation.',
            'Near-zero common-probe mean signed error for TD3 LN does not mean all predictions are accurate: positive and negative errors can cancel, while pooled RMSE remains about 52. These probes also include deliberately off-policy actions.',
            'The 500k budget matters: DDPG without LN gains about 105 return points from 300k; both TD3 settings gain about 40–49. Extending this unchanged DDPG+LN configuration is not supported by its flat/poor mean curve.',
            'The study supports an algorithm-dependent effect of critic LN under this fixed configuration. It does not identify a causal effect of bias on performance, nor a universal effect of normalization. A separately frozen target-bias ablation is the next step for that causal question.','',
            'Ten training seeds determine uncertainty. Bootstrap intervals are descriptive, pointwise and not adjusted for multiple comparisons. Evaluation episodes and trajectory pairs are not independent training replicates.',
            'All trajectories were retained. Final censor counts and maximum observed horizon sensitivity are in review.json; small empirical sensitivity does not prove a universal bound on unobserved future rewards. Raw returns use up to 10k diagnostic steps.','',
            'This local working analysis is not a replacement for the live shared Samskrift report.']
    formatted=[]
    for line in lines:
        formatted.append(line)
        if line and not line.startswith('|') and not line.startswith('#'):
            formatted.append('')
    (out/'interpretation.md').write_text('\n'.join(formatted)+'\n')
    print(json.dumps(dict(runtime=runtime,interpretation_checks=extra),indent=2))


if __name__=='__main__':
    main()
