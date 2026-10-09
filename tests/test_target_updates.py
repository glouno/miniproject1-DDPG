"""Target schedule correctness and exact recovery across a hard-copy boundary."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from lunarlander_rl.experiment import Config, initialize, update_targets
from lunarlander_rl.study_diagnostics import build_probes
from lunarlander_rl.study_train import train_study
from lunarlander_rl.target_runner import jobs_for


class TargetChecks(unittest.TestCase):
    def test_plan_changes_only_target_schedule(self):
        jobs = jobs_for()
        self.assertEqual(len(jobs),20)
        self.assertEqual({j['config']['seed'] for j in jobs},set(range(20,30)))
        for j in jobs:
            c = j['config']
            self.assertEqual((c['algorithm'],c['critic_layer_norm'],c['actor_layer_norm']),('ddpg',True,False))
            self.assertEqual((c['actor_lr'],c['critic_lr'],c['max_steps']),(3e-4,3e-4,1_000_000))
        left,right = jobs[0]['config'],jobs[1]['config']
        self.assertEqual({k for k in left if left[k]!=right[k]}, {'tau','hard_target_interval'})

    def test_copy_is_periodic_and_exact_for_parameters_and_buffers(self):
        cfg = Config(critic_layer_norm=True,hard_target_interval=400)
        env,a,q = initialize(cfg); env.close()
        ta,tq = copy.deepcopy(a),copy.deepcopy(q)
        initial = copy.deepcopy(ta.state_dict())
        with torch.no_grad():
            for p in [*a.parameters(),*q.parameters()]: p.add_(1)
        update_targets(a,ta,q,tq,cfg,399)
        for k,v in ta.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
        update_targets(a,ta,q,tq,cfg,400)
        for online,target in ((a,ta),(q,tq)):
            for k,v in online.state_dict().items():torch.testing.assert_close(v,target.state_dict()[k],rtol=0,atol=0)
        copied = copy.deepcopy(ta.state_dict())
        with torch.no_grad():
            for p in a.parameters():p.add_(1)
        update_targets(a,ta,q,tq,cfg,401)
        for k,v in ta.state_dict().items():torch.testing.assert_close(v,copied[k],rtol=0,atol=0)

    def test_both_interventions_exact_resume(self):
        for job in jobs_for('smoke'):
            cfg = replace(Config(**job['config']),max_steps=1100,eval_interval=1100)
            bank = build_probes(cfg,2,41000,5)
            with self.subTest(condition=job['cell']),tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                train_study(cfg,root/'full',bank,horizons=(150,300))
                # Pause at critic round 750; resume crosses the copy at round 800.
                train_study(cfg,root/'pause',bank,horizons=(150,300),stop_after=849)
                train_study(cfg,root/'pause',bank,horizons=(150,300),resume=True)
                x=torch.load(root/'full/checkpoint-0001100.pt',weights_only=True)
                y=torch.load(root/'pause/checkpoint-0001100.pt',weights_only=True)
                for section in ('actor','critics','target_actor','target_critics'):
                    for key in x[section]:torch.testing.assert_close(x[section][key],y[section][key],rtol=0,atol=0)
                self.assertEqual((root/'full/optimization.jsonl').read_text(),(root/'pause/optimization.jsonl').read_text())


if __name__=='__main__':unittest.main()
