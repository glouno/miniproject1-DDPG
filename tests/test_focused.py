"""Reward-unit consistency, fixed target cadence and exact intervention recovery."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lunarlander_rl.experiment import Config, initialize, bellman_target
from lunarlander_rl.focused_runner import jobs_for
from lunarlander_rl.study_diagnostics import build_probes, evaluate_study
from lunarlander_rl.study_train import train_study
from rl_mind.nn import soft_update
from rl_mind.data import ReplayBuffer


class FocusedChecks(unittest.TestCase):
    def config(self, **kwargs):
        return Config(algorithm="ddpg", actor_layer_norm=False, critic_layer_norm=True,
            isolated_initialization=True, instrumentation=True, seed=77,
            max_steps=1100, learning_starts=100, batch_size=32, replay_capacity=1200,
            actor_hidden=(16, 16), critic_hidden=(16, 16), eval_interval=1100,
            eval_episodes=1, train_horizon=100, diagnostic_horizon=300, resume_interval=250,
            **kwargs)

    def test_plan_reuses_baseline_and_stage_a_seeds(self):
        jobs = jobs_for()
        self.assertEqual(len(jobs), 30)
        self.assertEqual({j["config"]["seed"] for j in jobs}, set(range(20, 30)))
        self.assertEqual({(j["config"]["reward_scale"], j["config"]["ddpg_actor_delay"])
                          for j in jobs}, {(0.1, 1), (1.0, 2), (1.0, 3)})
        self.assertTrue(all(j["config"]["max_steps"] == 1_000_000 for j in jobs))

    def test_training_scales_reward_once_and_preserves_bootstrap(self):
        cfg = replace(self.config(reward_scale=0.1), max_steps=110)
        bank = build_probes(cfg, 2, 41000, 5)
        sampled = []
        sample = ReplayBuffer.sample
        def capture(buffer, size):
            batch = sample(buffer, size)
            sampled.append(batch.reward.clone())
            return batch
        def check(reward, terminated, next_q, gamma):
            torch.testing.assert_close(reward, sampled[-1] * 0.1, rtol=0, atol=0)
            expected = reward + gamma * next_q * (~terminated).float()
            result = bellman_target(reward, terminated, next_q, gamma)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            return result
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(ReplayBuffer, "sample", capture), \
             patch("lunarlander_rl.study_train.bellman_target", side_effect=check):
            train_study(cfg, Path(tmp)/"run", bank, horizons=(150, 300))
        self.assertEqual(len(sampled), 11)

    def test_raw_evaluation_units(self):
        cfg = self.config(reward_scale=0.1)
        env, actor, critics = initialize(cfg); env.close()
        bank = build_probes(cfg, 2, 41000, 5)
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_study(cfg, actor, critics, 0, tmp, bank, horizons=(150, 300))
            with np.load(Path(tmp)/"on_policy-0000000.npz") as pairs:
                with torch.no_grad():
                    native = critics[0](torch.from_numpy(pairs["obs"]).float(),
                                        torch.from_numpy(pairs["action"]).float()).numpy()
                np.testing.assert_allclose(pairs["q"][:, 0], native / 0.1, atol=1e-5)
                for i, t in enumerate(pairs["t"]):
                    rewards = pairs["rewards"][int(t):]
                    mc = np.dot(rewards, cfg.gamma**np.arange(len(rewards)))
                    self.assertAlmostEqual(pairs["mc300"][i], mc, places=10)

    def test_delay_keeps_targets_updating_each_round(self):
        for delay in (2, 3):
            cfg = replace(self.config(ddpg_actor_delay=delay), max_steps=110)
            bank = build_probes(cfg, 2, 41000, 5)
            with tempfile.TemporaryDirectory() as tmp, \
                 patch("lunarlander_rl.study_train.soft_update", wraps=soft_update) as update:
                train_study(cfg, tmp, bank, horizons=(150, 300))
                summary = json.loads((Path(tmp)/"summary.json").read_text())
                self.assertEqual(summary["critic_rounds"], 11)
                self.assertEqual(summary["actor_updates"], 11 // delay)
                self.assertEqual(update.call_count, 22)

    def test_all_interventions_exact_resume(self):
        for scale, delay in ((0.1, 1), (1.0, 2), (1.0, 3)):
            cfg = self.config(reward_scale=scale, ddpg_actor_delay=delay)
            bank = build_probes(cfg, 2, 41000, 5)
            with self.subTest(scale=scale, delay=delay), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                train_study(cfg, root/"full", bank, horizons=(150, 300))
                train_study(cfg, root/"pause", bank, horizons=(150, 300), stop_after=760)
                train_study(cfg, root/"pause", bank, horizons=(150, 300), resume=True)
                x = torch.load(root/"full/checkpoint-0001100.pt", weights_only=True)
                y = torch.load(root/"pause/checkpoint-0001100.pt", weights_only=True)
                for section in ("actor", "critics", "target_actor", "target_critics"):
                    for key in x[section]:
                        torch.testing.assert_close(x[section][key], y[section][key], rtol=0, atol=0)
                self.assertEqual((root/"full/optimization.jsonl").read_text(),
                                 (root/"pause/optimization.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
