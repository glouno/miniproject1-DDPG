"""Deterministic, read-only network audits and matched continuous action maps."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experiment import load_checkpoint, make_env, discounted_returns
from .study_diagnostics import atomic_json, atomic_npz


def norm(parameters, gradient=False):
    values = [p.grad if gradient else p for p in parameters]
    values = [v.detach() for v in values if v is not None]
    return float(torch.sqrt(sum(v.square().sum() for v in values))) if values else 0.


def distribution(tensor):
    x = tensor.detach().float().flatten().cpu()
    return dict(mean=float(x.mean()), sd=float(x.std(unbiased=False)),
                quantiles=torch.quantile(x, torch.tensor([0., .1, .5, .9, 1.])).tolist())


def network_audit(actor, critics, target_actor, target_critics, bank):
    # Copies ensure gradients, hooks and modes never touch training networks.
    actor, critics, target_actor, target_critics = [copy.deepcopy(x).cpu()
        for x in (actor, critics, target_actor, target_critics)]
    obs = torch.tensor([p['obs'] for p in bank], dtype=torch.float32)
    result = {}
    for name, model in [('actor', actor), *[(f'critic{i+1}', q) for i,q in enumerate(critics)]]:
        layers = {}
        handles = []
        def hook(key):
            def record(module, inputs, outputs):
                entry = dict(output=distribution(outputs), zero_fraction=float((outputs == 0).float().mean()))
                if isinstance(module, nn.LayerNorm):
                    var = inputs[0].detach().var(dim=-1, unbiased=False)
                    entry.update(input_variance=distribution(var), small_variance_fraction=float((var < 1e-5).float().mean()),
                                 gain=distribution(module.weight), offset=distribution(module.bias))
                if isinstance(module, nn.ReLU):
                    entry['always_zero_feature_fraction'] = float((outputs == 0).all(dim=0).float().mean())
                layers[key] = entry
            return record
        for key, layer in model.model.named_children():
            handles.append(layer.register_forward_hook(hook(f'{key}-{type(layer).__name__}')))
        with torch.no_grad():
            if name == 'actor':
                logits = model.model(obs)
                actions = model(obs)
                result['actor_logits'] = distribution(logits)
                result['actor_saturation_fraction'] = float((torch.tanh(logits).abs() > .95).float().mean())
                # LunarLander-v3 engine activation rules, not reverse thrust.
                result['main_engine_on_fraction'] = float((actions[:,0] > 0).float().mean())
                result['side_engine_on_fraction'] = float((actions[:,1].abs() > .5).float().mean())
            else:
                model(obs, actor(obs))
        for handle in handles:
            handle.remove()
        result[name] = dict(layers=layers, parameter_norm=norm(model.parameters()))
        if name != 'actor':
            result[name]['head_weight_norm'] = norm([model.model[-1].weight])
            result[name]['head_bias'] = distribution(model.model[-1].bias)
            with torch.no_grad():
                features=model.model[:-1](torch.cat((obs,actor(obs)),dim=-1))
                singular=torch.linalg.svdvals(features-features.mean(dim=0))
                probabilities=singular/singular.sum().clamp_min(1e-12)
                result[name]['feature_effective_rank']=float(torch.exp(-(probabilities*probabilities.clamp_min(1e-12).log()).sum())) if singular.sum()>0 else 0.
                result[name]['feature_sd']=distribution(features.std(dim=0,unbiased=False))
    with torch.enable_grad():
        a = actor(obs).detach().requires_grad_(True)
        for i,q in enumerate(critics):
            gradient = torch.autograd.grad(q(obs,a).sum(), a)[0]
            result[f'critic{i+1}_action_gradient_norm'] = distribution(gradient.norm(dim=-1))
        jacobian = []
        for s in obs:
            jacobian.append(torch.autograd.functional.jacobian(lambda x: actor(x.unsqueeze(0))[0], s).norm())
        result['actor_observation_jacobian_norm'] = distribution(torch.stack(jacobian))
    with torch.no_grad():
        a = actor(obs)
        result['policy_target_distance'] = float((a-target_actor(obs)).square().mean().sqrt())
        for i,(q,tq) in enumerate(zip(critics,target_critics)):
            result[f'critic{i+1}_target_distance'] = float((q(obs,a)-tq(obs,a)).square().mean().sqrt())
        if len(critics)==2:
            result['critic_disagreement'] = float((critics[0](obs,a)-critics[1](obs,a)).abs().mean())
    return result


def phase_errors(pairs):
    obs, action, t = pairs['obs'], pairs['action'], pairs['t']
    horizon=max(int(k[2:]) for k in pairs.files if k.startswith('mc'))
    e = pairs['q'][:,0] - pairs[f'mc{horizon}']
    masks = dict(first_100=t<100, later=t>=100,
        airborne=(obs[:,6]==0)&(obs[:,7]==0), contact=(obs[:,6]>0)|(obs[:,7]>0),
        tilted=np.abs(obs[:,4])>.2, level=np.abs(obs[:,4])<=.2,
        descending=obs[:,3]<0, ascending=obs[:,3]>=0,
        saturated=np.max(np.abs(action),axis=1)>.95)
    return {key:dict(count=int(m.sum()), fraction=float(m.mean()), bias=float(e[m].mean()),
                     rmse=float(np.sqrt(np.mean(e[m]**2)))) for key,m in masks.items() if m.any()}


def matched_return(cfg, actor, probe, action):
    env = make_env(cfg, len(probe['prefix'])+cfg.diagnostic_horizon)
    obs,_ = env.reset(seed=probe['seed'])
    for a in probe['prefix']:
        obs,_,term,trunc,_=env.step(np.asarray(a,dtype=np.float32))
        assert not (term or trunc)
    np.testing.assert_array_equal(obs, np.asarray(probe['obs'],dtype=np.float32))
    rewards=[]; term=trunc=False
    with torch.inference_mode():
        while not (term or trunc):
            a=np.asarray(action,dtype=np.float32) if not rewards else actor(torch.tensor(obs).unsqueeze(0))[0].numpy()
            obs,r,term,trunc,_=env.step(a); rewards.append(r)
    env.close()
    return dict(mc=float(discounted_returns(rewards,cfg.gamma)[0]), censored=not bool(term),
                length=len(rewards)), np.asarray(rewards)


def action_maps(checkpoint, probes, destination, grid_size=9, count=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg,actor,critics,step=load_checkpoint(checkpoint)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    axis=np.linspace(-1,1,grid_size,dtype=np.float32)
    main,side=np.meshgrid(axis,axis)
    actions=np.column_stack((main.ravel(),side.ravel()))
    records=[]
    for index,probe in enumerate(probes[:count]):
        obs=torch.tensor(probe['obs'],dtype=torch.float32).repeat(len(actions),1)
        with torch.inference_mode():
            qs=np.column_stack([q(obs,torch.from_numpy(actions)).numpy() for q in critics])
            chosen=actor(obs[:1])[0].numpy()
        candidates=[('actor',chosen),('zero',np.zeros(2)),('diagonal+',np.array([.5,.5])),
                    ('diagonal-',np.array([-.5,-.5])),('Q1 grid max',actions[qs[:,0].argmax()])]
        tests=[]
        for name,action in candidates:
            with torch.inference_mode():
                prediction=float(critics[0](obs[:1],torch.tensor(action,dtype=torch.float32).unsqueeze(0)))
            result,rewards=matched_return(cfg,actor,probe,action)
            tests.append(dict(name=name,action=action.tolist(),q=prediction,**result))
            atomic_npz(destination/f'probe{index}-candidate{len(tests)-1}.npz',rewards=rewards)
        atomic_npz(destination/f'probe{index}-grid.npz',actions=actions,q=qs,obs=obs[0].numpy())
        fig,axes=plt.subplots(1,2,figsize=(11.5,5.2))
        im=axes[0].pcolormesh(main,side,qs[:,0].reshape(main.shape),shading='auto',cmap='viridis')
        axes[0].scatter(*chosen,c='white',edgecolors='black',marker='*',s=140,label='Actor')
        best=actions[qs[:,0].argmax()];axes[0].scatter(*best,c='red',marker='x',s=60,label='Grid Q1 max')
        axes[0].set(xlabel='Main-engine command',ylabel='Side-engine command',title=f'Probe {index}: predicted Q1')
        axes[0].legend();fig.colorbar(im,ax=axes[0],label='Discounted return prediction')
        x=np.arange(len(tests));axes[1].plot(x,[r['q'] for r in tests],'o-',label='Predicted Q1')
        axes[1].plot(x,[r['mc'] for r in tests],'s-',label='Matched return estimate')
        censored=[i for i,r in enumerate(tests) if r['censored']]
        if censored:
            axes[1].scatter(censored,[tests[i]['mc'] for i in censored],marker='x',s=100,
                color='red',label=f'Cut off at {cfg.diagnostic_horizon:,} steps')
        axes[1].set_xticks(x,[r['name'] for r in tests],rotation=25)
        axes[1].set(ylabel='Discounted return',title='Forced first action, then frozen actor')
        critic_ln=cfg.layer_norm if cfg.critic_layer_norm is None else cfg.critic_layer_norm
        fig.suptitle(f'{cfg.algorithm.upper()}, actor LN {cfg.actor_layer_norm}, critic LN {critic_ln}; seed {cfg.seed}, step {step:,}')
        axes[1].legend(fontsize=8);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(destination/f'probe{index}.png',dpi=150)
        fig.savefig(destination/f'probe{index}.pdf');plt.close(fig)
        records.append(dict(index=index,probe=probe,candidates=tests,
                            note='Grid-max candidate selected by Q; censored MC is a finite approximation.'))
    atomic_json(destination/'maps.json',records)
    atomic_json(destination/'provenance.json',dict(checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        config=asdict(cfg),grid_size=grid_size,probe_count=len(records)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=6);p.add_argument('--grid-size',type=int,default=9)
    a=p.parse_args();action_maps(a.checkpoint,json.loads(a.probes.read_text()),a.output,a.grid_size,a.count)
