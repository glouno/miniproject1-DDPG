"""Physical-state matching, paired randomness and independent MC arithmetic."""
import unittest
import numpy as np

from lunarlander_rl.experiment import Config, initialize
from lunarlander_rl.state_action_evaluation import action_grid, reconstruct, rollout, paired_interval
from lunarlander_rl.study_diagnostics import build_probes


class StateActionChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=Config(algorithm='ddpg',seed=77,actor_hidden=(8,8),critic_hidden=(8,8),
                        train_horizon=20,diagnostic_horizon=50)
        env,self.actor,_=initialize(self.cfg);env.close()
        self.probe=build_probes(self.cfg,count=1,seed_start=41000,prefix_steps=5)[0]
        self.probe['prefix_return']=0.0

    def test_grid_coordinates_and_order(self):
        axis,actions=action_grid(9)
        self.assertEqual(actions.shape,(81,2))
        np.testing.assert_array_equal(axis,np.arange(-1,1.01,.25,dtype=np.float32))
        np.testing.assert_array_equal(actions[40],[0,0])
        np.testing.assert_array_equal(actions[1],[-.75,-1])

    def test_reconstruction_and_equivalent_commands(self):
        env,obs=reconstruct(self.cfg,self.probe,123)
        np.testing.assert_array_equal(obs,np.asarray(self.probe['obs'],dtype=np.float32));env.close()
        first=rollout(self.cfg,self.actor,self.probe,[-1,0],123,capture=True)
        second=rollout(self.cfg,self.actor,self.probe,[0,.5],123,capture=True)
        np.testing.assert_array_equal(first['rewards'],second['rewards'])
        np.testing.assert_array_equal(first['observations'],second['observations'])
        self.assertEqual(first['mc'],second['mc'])

    def test_mc_and_performance_boundary(self):
        result=rollout(self.cfg,self.actor,self.probe,[.5,-.75],456,capture=True)
        expected=sum(float(r)*self.cfg.gamma**i for i,r in enumerate(result['rewards']))
        self.assertAlmostEqual(result['mc'],expected,places=12)
        self.assertEqual(result['performance_timeout'],len(result['rewards'])>15)
        self.assertAlmostEqual(result['performance_return'],sum(result['rewards'][:15]),places=12)

    def test_paired_interval_identical_values(self):
        result=paired_interval([1,10,-3],[1,10,-3])
        self.assertEqual(result,dict(mean=0.0,ci95=[0.0,0.0]))


if __name__=='__main__':unittest.main()
