"""Scientific correctness checks. Run: python -m unittest discover -s RL/miniproject."""
import unittest
import numpy as np
import torch
from experiment import Actor, Critic, Config, bellman_target, discounted_returns, initialize


class ScientificChecks(unittest.TestCase):
    def test_returns_and_terminal_mask(self):
        np.testing.assert_allclose(discounted_returns([1, 2, 3], .5), [2.75, 3.5, 3])
        next_q = torch.tensor([4., 4.], requires_grad=True)
        result = bellman_target(torch.tensor([2., 2.]), torch.tensor([True, False]), next_q, .5)
        torch.testing.assert_close(result, torch.tensor([2., 4.]))
        self.assertFalse(result.requires_grad)

    def test_critic_normalization_and_initialization(self):
        torch.manual_seed(4)
        plain = Critic(8, 2, (16, 16), False)
        torch.manual_seed(4)
        normalized = Critic(8, 2, (16, 16), True)
        self.assertEqual(sum(isinstance(x, torch.nn.LayerNorm) for x in normalized.modules()), 2)
        self.assertFalse(any(isinstance(x, torch.nn.LayerNorm) for x in plain.modules()))
        a = [x for x in plain.modules() if isinstance(x, torch.nn.Linear)]
        b = [x for x in normalized.modules() if isinstance(x, torch.nn.Linear)]
        for x, y in zip(a, b):
            torch.testing.assert_close(x.weight, y.weight)
            torch.testing.assert_close(x.bias, y.bias)
        out = normalized(torch.zeros(1, 8), torch.zeros(1, 2))
        self.assertEqual(tuple(out.shape), (1,))

    def test_actor_gradient_and_physical_bounds(self):
        actor = Actor(8, (16, 16), [-2., -1.], [2., 1.])
        critic = Critic(8, 2, (16, 16), True).requires_grad_(False)
        obs = torch.randn(4, 8)
        action = actor(obs)
        self.assertTrue(bool(((action >= actor.low) & (action <= actor.high)).all()))
        (-critic(obs, action).mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_td3_independent_critics_and_env_flags(self):
        cfg = Config()
        env, actor, critics = initialize(cfg)
        self.assertEqual(len(critics), 2)
        self.assertFalse(torch.equal(critics[0].model[0].weight, critics[1].model[0].weight))
        obs, _ = env.reset(seed=123)
        self.assertEqual(obs.shape, (8,))
        self.assertEqual(env.action_space.shape, (2,))
        env.step(np.zeros(2, dtype=np.float32))
        env.close()


if __name__ == '__main__':
    unittest.main()
