"""Saved artifacts must prove completion before reporting or reusing a run."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class StudyCompletionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'source').mkdir()
        (self.root / 'logs').mkdir()

    def manifest(self, names):
        value = dict(jobs=[dict(directory=name, algorithm='ddpg', layer_norm=False, seed=77)
                           for name in names], expected_runs=len(names), config=dict(max_steps=20),
                     source_sha256={}, config_sha256='bound', probes_sha256='bound',
                     lock_sha256='bound', max_hours=1, workers=1)
        (self.root / 'manifest.json').write_text(json.dumps(value))
        return value

    def summary(self, name, value):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        (directory / 'summary.json').write_text(value if isinstance(value, str) else json.dumps(value))

    def test_reporter_counts_only_valid_summaries_without_site_packages(self):
        names = ['missing', 'malformed', 'failed', 'short', 'nonobject', 'boolean', 'valid']
        self.manifest(names)
        self.summary('malformed', '{incomplete')
        self.summary('failed', dict(status='failed', transitions=20))
        self.summary('short', dict(status='complete', transitions=19))
        self.summary('nonobject', [])
        self.summary('boolean', dict(status='complete', transitions=True))
        self.summary('valid', dict(status='complete', transitions=20))
        source = Path(__file__).resolve().parents[1] / 'src'
        result = subprocess.run([sys.executable, '-S', '-m', 'lunarlander_rl.study_status',
                                 '--input', str(self.root)], check=True, capture_output=True,
                                text=True, env=dict(os.environ, PYTHONPATH=str(source)))
        report = json.loads(result.stdout)
        self.assertEqual(report['completed'], 1)
        self.assertEqual(report['expected_runs'], 7)
        self.assertEqual(report['status'], 'starting')
        self.assertEqual({entry['run'] for entry in report['unresolved_summaries']}, set(names[:-1]))
        self.assertTrue(all(entry['reason'] for entry in report['unresolved_summaries']))

    def run_fixture(self, names, process=None, analysis_code=0):
        from lunarlander_rl import study_runner
        self.manifest(names)
        with patch.object(study_runner, 'sha', return_value='bound'), \
             patch.object(study_runner.subprocess, 'Popen', process or Mock()) as launch, \
             patch.object(study_runner.subprocess, 'run', return_value=Mock(returncode=analysis_code)) as analyse, \
             contextlib.redirect_stdout(io.StringIO()):
            result = study_runner.run(self.root)
        status = json.loads((self.root / 'status.json').read_text())
        return result, status, launch, analyse

    def test_invalid_existing_summary_is_preserved_and_other_jobs_are_reused(self):
        self.summary('bad', '{incomplete')
        self.summary('good', dict(status='complete', transitions=20))
        result, status, launch, analyse = self.run_fixture(['bad', 'good'])
        self.assertEqual(result, 1)
        self.assertEqual(status['status'], 'incomplete_requires_attention')
        self.assertEqual(status['jobs']['bad']['status'], 'failed')
        self.assertIn('summary_error', status['jobs']['bad'])
        self.assertTrue(status['jobs']['good']['reused'])
        self.assertEqual((self.root / 'bad/summary.json').read_text(), '{incomplete')
        launch.assert_not_called()
        analyse.assert_not_called()

    def test_zero_exit_with_invalid_fresh_summary_is_not_complete(self):
        def launch(*args, **kwargs):
            self.summary('fresh', dict(status='complete', transitions=19))
            return Mock(pid=123, wait=Mock(return_value=0))
        result, status, _, analyse = self.run_fixture(['fresh'], Mock(side_effect=launch))
        self.assertEqual(result, 1)
        self.assertEqual(status['jobs']['fresh']['exit_code'], 0)
        self.assertIn('summary_error', status['jobs']['fresh'])
        analyse.assert_not_called()

    def test_valid_fresh_and_reused_summaries_preserve_analysis_outcome(self):
        self.summary('reused', dict(status='complete', transitions=20))
        def launch(*args, **kwargs):
            self.summary('fresh', dict(status='complete', transitions=20))
            return Mock(pid=123, wait=Mock(return_value=0))
        result, status, process, analyse = self.run_fixture(['reused', 'fresh'], Mock(side_effect=launch))
        self.assertEqual(result, 0)
        self.assertEqual(status['status'], 'complete')
        self.assertTrue(status['jobs']['reused']['reused'])
        self.assertEqual(process.call_count, 1)
        analyse.assert_called_once()

    def test_valid_training_with_failed_analysis_stays_distinct(self):
        self.summary('good', dict(status='complete', transitions=20))
        result, status, launch, analyse = self.run_fixture(['good'], analysis_code=1)
        self.assertEqual(result, 1)
        self.assertEqual(status['status'], 'training_complete_analysis_failed')
        self.assertEqual(status['jobs']['good']['status'], 'complete')
        launch.assert_not_called()
        analyse.assert_called_once()


if __name__ == '__main__':
    unittest.main()
