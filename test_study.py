"""Independent return checks and exact interrupted-vs-uninterrupted training."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiment import Config, train, load_checkpoint
from study_diagnostics import build_probes, evaluate_study
from study_train import train_study


class StudyChecks(unittest.TestCase):
    def config(self):
        return Config(max_steps=400,learning_starts=100,batch_size=32,replay_capacity=500,
                      actor_hidden=(16,16),critic_hidden=(16,16),eval_interval=200,
                      eval_episodes=2,train_horizon=100,diagnostic_horizon=300,seed=77)

    def test_exact_resume_and_canonical_training(self):
        for algorithm in ('ddpg','td3'):
            for norm in (False,True):
                with self.subTest(algorithm=algorithm,layer_norm=norm):
                    self.check_resume(replace(self.config(),algorithm=algorithm,layer_norm=norm))

    def check_resume(self,cfg):
        bank=build_probes(cfg,count=2,prefix_steps=5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train_study(cfg,root/'continuous',bank,horizons=(150,300))
            train_study(cfg,root/'interrupted',bank,horizons=(150,300),stop_after=149)
            # Append an incomplete journal to mimic a process dying during a write.
            with (root/'interrupted/training_episodes.jsonl').open('a') as f:
                f.write('{incomplete')
            train_study(cfg,root/'interrupted',bank,horizons=(150,300),resume=True)
            train(cfg,root/'canonical')
            snapshots=[torch.load(root/name/'checkpoint-0000400.pt',weights_only=True) for name in ('continuous','interrupted','canonical')]
            for section in ('actor','critics'):
                for key,value in snapshots[0][section].items():
                    for other in snapshots[1:]:
                        torch.testing.assert_close(value,other[section][key],rtol=0,atol=0)
            self.assertEqual((root/'continuous/training_episodes.jsonl').read_text(),(root/'interrupted/training_episodes.jsonl').read_text())
            a=json.loads((root/'continuous/metrics.jsonl').read_text().splitlines()[-1])
            b=json.loads((root/'interrupted/metrics.jsonl').read_text().splitlines()[-1])
            self.assertEqual(a['on_policy'],b['on_policy']);self.assertEqual(a['common_probe'],b['common_probe'])

    def test_all_trajectories_and_direct_returns(self):
        from experiment import initialize
        cfg=replace(self.config(),train_horizon=10,diagnostic_horizon=50)
        env,actor,critics=initialize(cfg);env.close()
        bank=build_probes(cfg,count=2,prefix_steps=5)
        with tempfile.TemporaryDirectory() as tmp:
            m=evaluate_study(cfg,actor,critics,0,Path(tmp),bank,horizons=(25,50))
            for kind in ('on_policy','common_probe'):
                pairs=np.load(Path(tmp)/f'{kind}-0000000.npz')
                self.assertEqual(len(m[kind]['episodes']),2)
                self.assertEqual(set(pairs['episode']),{0,1})
                for j in range(len(pairs['q'])):
                    ep=int(pairs['episode'][j]);t=int(pairs['t'][j]);start=int(pairs['offsets'][ep]);end=int(pairs['offsets'][ep+1])
                    rewards=pairs['rewards'][start:end]
                    for h in (25,50):
                        suffix=rewards[t:h];direct=np.dot(suffix,cfg.gamma**np.arange(len(suffix)))
                        self.assertAlmostEqual(float(pairs[f'mc{h}'][j]),float(direct),places=10)
                obs=torch.from_numpy(pairs['obs']).float();actions=torch.from_numpy(pairs['action']).float()
                with torch.no_grad():
                    predicted=critics[0](obs,actions).numpy()
                np.testing.assert_allclose(predicted,pairs['q'][:,0],atol=1e-6)
            common=np.load(Path(tmp)/'common_probe-0000000.npz')
            np.testing.assert_array_equal(common['action'],np.asarray([b['action'] for b in bank],dtype=np.float32))


if __name__=='__main__':
    unittest.main()
