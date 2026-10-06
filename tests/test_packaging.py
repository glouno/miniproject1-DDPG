"""Exercise editable imports and real subprocess launches after the src migration."""
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lunarlander_rl.experiment import Config
from lunarlander_rl.paths import frozen_command, project_root
from lunarlander_rl.study_runner import prepare


class PackagingChecks(unittest.TestCase):
    def config(self):
        return Config(max_steps=40, learning_starts=20, batch_size=8, replay_capacity=50,
                      actor_hidden=(8, 8), critic_hidden=(8, 8), eval_interval=40,
                      eval_episodes=1, train_horizon=20, diagnostic_horizon=20, seed=77)

    def run_process(self, command, cwd):
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=120)
        logs = ""
        if result.returncode:
            logs = "\n".join(p.read_text() for p in (Path(cwd).parent / 'logs').glob('*.log'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr + logs)
        return result.stdout

    def test_editable_import_from_unrelated_directory(self):
        metadata = importlib.metadata.distribution('miniproject1-ddpg')
        origin = json.loads(metadata.read_text('direct_url.json'))
        self.assertTrue(origin['dir_info']['editable'])
        with tempfile.TemporaryDirectory() as tmp:
            # Isolated mode excludes cwd and PYTHONPATH but retains installed packages.
            output = self.run_process([sys.executable, '-I', '-c',
                'from lunarlander_rl.paths import project_root; print(project_root())'], tmp)
            self.assertEqual(Path(output.strip()), project_root())

    def test_pilot_workers_launch_as_installed_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text(json.dumps(asdict(self.config())))
            self.run_process([sys.executable, '-m', 'lunarlander_rl.run_pilot',
                              '--config', str(config), '--output', str(root / 'pilot'),
                              '--seeds', '77', '--workers', '1'], root)
            manifest = json.loads((root / 'pilot/manifest.json').read_text())
            self.assertEqual(manifest['expected_runs'], 4)
            for job in manifest['jobs']:
                run = root / 'pilot' / job['directory']
                self.assertEqual(json.loads((run / 'summary.json').read_text())['transitions'], 40)
                self.assertTrue((run / 'source/lunarlander_rl/experiment.py').is_file())

    def test_frozen_study_imports_and_runs_its_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            # The research CLI fixes the 2,500/5,000/10,000 diagnostic horizons.
            config.write_text(json.dumps(asdict(replace(self.config(), diagnostic_horizon=10_000))))
            out = root / 'study'
            prepare(out, config, [77], workers=1, max_hours=1, probes_count=2, prefix_steps=3)
            source = out / 'source'
            imported = self.run_process([sys.executable, '-c',
                'import lunarlander_rl.experiment as e; print(e.__file__)'], source)
            self.assertEqual(Path(imported.strip()).resolve(), (source / 'lunarlander_rl/experiment.py').resolve())
            self.run_process(frozen_command(source, 'study_runner') + ['--output', str(out)], source)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
            self.assertEqual(json.loads((out / 'analysis/verification.json').read_text())['completed_runs'], 4)
            manifest = json.loads((out / 'manifest.json').read_text())
            for name, digest in manifest['source_sha256'].items():
                self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)

    def test_legacy_frozen_snapshot_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            self.assertEqual(frozen_command(source, 'study_runner'),
                             [sys.executable, str(source.resolve() / 'study_runner.py')])


if __name__ == '__main__':
    unittest.main()
