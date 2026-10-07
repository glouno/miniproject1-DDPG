"""Scientific interventions, non-invasive diagnostics and interruption recovery."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from experiment import Config, initialize
from study_diagnostics import build_probes
from study_train import train_study
from mechanism_diagnostics import network_audit, matched_return
from followup_runner import jobs_for,prepare,verify


class FollowupChecks(unittest.TestCase):
    def cfg(self,**kwargs):
        return Config(max_steps=1100,learning_starts=100,batch_size=32,replay_capacity=1200,
            actor_hidden=(16,16),critic_hidden=(16,16),eval_interval=1100,
            eval_episodes=1,train_horizon=30,diagnostic_horizon=100,
            seed=77,isolated_initialization=True,resume_interval=250,**kwargs)

    def test_factorial_plan(self):
        jobs=jobs_for('full');self.assertEqual(len(jobs),140)
        self.assertEqual(sum(j['config']['max_steps'] for j in jobs),140_000_000)
        self.assertEqual(len({j['directory'] for j in jobs}),140)
        self.assertEqual(sum(j['stage']=='A' for j in jobs),80)

    def test_frozen_inputs_reject_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'queue'
            prepare(root,'smoke',2);verify(root)
            source=root/'source/experiment.py';original=source.read_bytes()
            source.write_bytes(original+b'\n')
            with self.assertRaisesRegex(RuntimeError,'Frozen file changed'):verify(root)
            source.write_bytes(original)
            manifest=root/'manifest.json';manifest.write_text(manifest.read_text()+'\n')
            with self.assertRaisesRegex(RuntimeError,'Frozen manifest changed'):verify(root)

    def test_independent_initialization_and_placement(self):
        env,a,qs=initialize(self.cfg());env.close()
        env,aln,qs2=initialize(self.cfg(actor_layer_norm=True));env.close()
        for x,y in zip(qs.parameters(),qs2.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        env,a2,qln=initialize(self.cfg(critic_layer_norm=True));env.close()
        for x,y in zip(a.parameters(),a2.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(sum(isinstance(m,nn.LayerNorm) for m in aln.modules()),2)
        self.assertEqual(sum(isinstance(m,nn.LayerNorm) for m in qln[0].modules()),2)
        self.assertIsInstance(qln[0].model[-1],nn.Linear)
        for left,right in ((a,aln),(qs[0],qln[0])):
            linear_left=[m for m in left.modules() if isinstance(m,nn.Linear)]
            linear_right=[m for m in right.modules() if isinstance(m,nn.Linear)]
            for x,y in zip(linear_left,linear_right):
                torch.testing.assert_close(x.weight,y.weight,rtol=0,atol=0)
                torch.testing.assert_close(x.bias,y.bias,rtol=0,atol=0)

    def test_audit_does_not_change_rng_or_weights(self):
        cfg=self.cfg(actor_layer_norm=True,critic_layer_norm=True)
        env,a,q=initialize(cfg);env.close();bank=build_probes(cfg,2,41000,5)
        rng=torch.get_rng_state().clone();weights=[p.detach().clone() for p in [*a.parameters(),*q.parameters()]]
        data=network_audit(a,q,a,q,bank)
        self.assertIn('critic1_action_gradient_norm',data)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for p,w in zip([*a.parameters(),*q.parameters()],weights):torch.testing.assert_close(p,w,rtol=0,atol=0)
        result,rewards=matched_return(cfg,a,bank[0],[.5,-.5])
        self.assertAlmostEqual(result['mc'],float(np.dot(rewards,cfg.gamma**np.arange(len(rewards)))),places=10)

    def test_all_eight_conditions_exact_resume(self):
        for algo in ('ddpg','td3'):
            for a in (False,True):
                for c in (False,True):
                    with self.subTest(algorithm=algo,actor_ln=a,critic_ln=c):
                        cfg=self.cfg(algorithm=algo,actor_layer_norm=a,critic_layer_norm=c,instrumentation=True)
                        bank=build_probes(cfg,2,41000,5)
                        with tempfile.TemporaryDirectory() as tmp:
                            root=Path(tmp)
                            train_study(cfg,root/'full',bank,horizons=(50,100))
                            train_study(cfg,root/'pause',bank,horizons=(50,100),stop_after=760)
                            with (root/'pause/optimization.jsonl').open('a') as f:f.write('{incomplete')
                            train_study(cfg,root/'pause',bank,horizons=(50,100),resume=True)
                            x=torch.load(root/'full/checkpoint-0001100.pt',weights_only=True)
                            y=torch.load(root/'pause/checkpoint-0001100.pt',weights_only=True)
                            for section in ('actor','critics','target_actor','target_critics'):
                                for k in x[section]:torch.testing.assert_close(x[section][k],y[section][k],rtol=0,atol=0)
                            self.assertEqual((root/'full/optimization.jsonl').read_text(),(root/'pause/optimization.jsonl').read_text())
                            self.assertEqual((root/'full/audits/audit-0001100.json').read_text(),(root/'pause/audits/audit-0001100.json').read_text())
                            plain=replace(cfg,instrumentation=False)
                            train_study(plain,root/'plain',bank,horizons=(50,100))
                            z=torch.load(root/'plain/checkpoint-0001100.pt',weights_only=True)
                            for section in ('actor','critics'):
                                for k in x[section]:torch.testing.assert_close(x[section][k],z[section][k],rtol=0,atol=0)

    def test_corrupt_latest_checkpoint_falls_back(self):
        cfg=self.cfg(instrumentation=True)
        bank=build_probes(cfg,2,41000,5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train_study(cfg,root/'full',bank,horizons=(50,100))
            train_study(cfg,root/'pause',bank,horizons=(50,100),stop_after=760)
            (root/'pause/resume.pt').write_bytes(b'broken checkpoint')
            train_study(cfg,root/'pause',bank,horizons=(50,100),resume=True)
            x=torch.load(root/'full/checkpoint-0001100.pt',weights_only=True)
            y=torch.load(root/'pause/checkpoint-0001100.pt',weights_only=True)
            for section in ('actor','critics'):
                for k in x[section]:torch.testing.assert_close(x[section][k],y[section][k],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
