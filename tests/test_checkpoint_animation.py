"""Check paired baselines and a real capture/export round trip."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lunarlander_rl.checkpoint_animation import build, compact_rollouts, map_values, render_saved
from lunarlander_rl.experiment import Config, initialize, save_checkpoint
from lunarlander_rl.study_diagnostics import build_probes


class CheckpointAnimationChecks(unittest.TestCase):
    def test_compaction_loads_compressed_arrays_once(self):
        class CountedArrays(dict):
            def __getitem__(self, key):
                counts[key] = counts.get(key, 0) + 1
                return super().__getitem__(key)
        counts = {}
        arrays = CountedArrays(offsets=np.array([0, 2, 4]),
            observations=np.zeros((4, 8)), rollout_q=np.zeros(4), rollout_g=np.zeros(4))
        record = dict(final_observation=[0]*8, future_seed=55000, length=2,
                      ending='time_limit', censored=True, mc=0., performance_return=0.)
        result = compact_rollouts(arrays, dict(rollouts=[record, record]), repeats=2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(counts, dict(offsets=1, observations=1, rollout_q=1, rollout_g=1))

    def test_delta_g_uses_paired_exact_actor_baseline(self):
        q = np.array([7., 3., 5.])
        samples = np.array([[101., 201., 301.], [90., 210., 310.], [100., 200., 300.]])
        values = map_values(q, samples)
        np.testing.assert_array_equal(values['delta_q'], [2, -2, 0])
        np.testing.assert_array_equal(values['delta_g'], [1, 10/3, 0])
        self.assertEqual(values['delta_sd'][0], 0.)
        self.assertEqual(values['delta_sd'][-1], 0.)
        self.assertGreater(values['sd'][0], 0.)

    def test_export_all_checkpoints_and_capture_return_arithmetic(self):
        cfg = Config(algorithm='ddpg', seed=77, actor_hidden=(8, 8), critic_hidden=(8, 8),
                     train_horizon=20, diagnostic_horizon=30)
        env, actor, critics = initialize(cfg); env.close()
        probe = build_probes(cfg, count=1, seed_start=41000, prefix_steps=5)[0]
        probe.update(prefix_return=0., flight_step=5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root/'run'; source = root/'source'; out = root/'output'
            run.mkdir(); source.mkdir()
            # Deliberately different from the original hand-picked four steps.
            for step in (0, 17):
                save_checkpoint(run/f'checkpoint-{step:07d}.pt', cfg, actor, critics, step)
            (source/'state.json').write_text(json.dumps(probe))
            (source/'provenance.json').write_text(json.dumps(dict(
                gamma=cfg.gamma, diagnostic_cutoff=cfg.diagnostic_horizon,
                future_seeds=[55000, 55001], grid_size=2, checkpoints={})))
            build(run, source, out, workers=1, grid_size=2, seeds=(55000, 55001))
            self.assertTrue((out/'index.html').is_file())
            original_manifest = (out/'manifest.json').read_bytes()
            render_saved(out)
            self.assertEqual((out/'manifest.json').read_bytes(), original_manifest)
            self.assertTrue((out/'render_provenance.json').is_file())
            self.assertNotIn('__PAYLOAD__', (out/'index.html').read_text())
            self.assertEqual(len(list(out.glob('capture-*.npz'))), 2)
            with np.load(out/'capture-0000017.npz') as data:
                metadata = json.loads((out/'capture-0000017.json').read_text())
                self.assertEqual(len(data['offsets']), 5*2+1)
                for i, (lo, hi) in enumerate(zip(data['offsets'][:-1], data['offsets'][1:])):
                    expected = sum(float(r)*cfg.gamma**t for t, r in enumerate(data['rewards'][lo:hi]))
                    self.assertAlmostEqual(data['samples'].ravel()[i], expected, places=10)
                    self.assertAlmostEqual(data['rollout_g'][lo], expected, places=10)
                    self.assertEqual(metadata['rollouts'][i]['length'], hi-lo)
                    self.assertTrue(metadata['rollouts'][i]['censored'])
                compact = compact_rollouts(data, metadata, repeats=2, max_points=4)
                self.assertEqual(len(compact), 5)
                self.assertEqual(len(compact[0]), 2)
                trajectory = compact[0][0]
                self.assertEqual(trajectory['t'][-1], trajectory['length'])
                self.assertEqual(len(trajectory['xy']), len(trajectory['q'])+1)
            # The cache may be reopened, but changed evaluation settings must fail.
            build(run, source, out, workers=1, grid_size=2, seeds=(55000, 55001))
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                build(run, source, out, workers=1, grid_size=3, seeds=(55000, 55001))


if __name__ == '__main__':
    unittest.main()
