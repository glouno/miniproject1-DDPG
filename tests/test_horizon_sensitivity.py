import unittest
import numpy as np

from lunarlander_rl.horizon_sensitivity import compare, paired_statistics


class HorizonSensitivityTests(unittest.TestCase):
    def test_paired_cutoffs_and_equal_episode_weight(self):
        # Episode 0 extends past cutoff; episode 1 terminates early. Unequal
        # sampled-pair counts must not give episode 0 more aggregation weight.
        data = dict(q=np.array([[10.], [20.], [30.]]),
                    rewards=np.array([1., 2., 4., 8., 3.]), offsets=np.array([0, 4, 5]),
                    episode=np.array([0, 0, 1]), t=np.array([0, 1, 0]))
        episodes, times, deltas = compare(data, .5, short=2, long=4)
        np.testing.assert_allclose(deltas, [-2., -4., 0.])
        self.assertEqual(episodes[0]["delta"], -3.)
        self.assertEqual(episodes[1]["delta"], 0.)
        self.assertEqual(np.mean([e["delta"] for e in episodes]), -1.5)
        # A 2-step window at t=1 contains r_2; cutting episode at t=2 does not.
        self.assertEqual(episodes[0]["per_state_short_delta"], -2.)

    def test_invalid_sample_after_cutoff(self):
        data = dict(q=np.array([[1.]]), rewards=np.ones(4), offsets=np.array([0, 4]),
                    episode=np.array([0]), t=np.array([2]))
        with self.assertRaisesRegex(ValueError, "precede"):
            compare(data, .99, short=2, long=4)

    def test_unchanged_episodes(self):
        result = paired_statistics(np.zeros(20))
        self.assertEqual(result["bootstrap_ci95"], [0., 0.])
        self.assertEqual(result["paired_t_p"], 1.)
        self.assertEqual(result["exact_two_sided_sign_p"], 1.)

    def test_exact_sign_test_excludes_zero_changes(self):
        result = paired_statistics([1.] * 13 + [0.] * 7)
        self.assertEqual(result["exact_two_sided_sign_p"], 2 / 2 ** 13)
        self.assertEqual(result["unchanged_episodes"], 7)
        self.assertGreater(result["bootstrap_ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
