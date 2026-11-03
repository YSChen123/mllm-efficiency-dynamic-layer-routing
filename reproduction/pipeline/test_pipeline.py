import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name('pipeline.py')


class PipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('pipeline', MODULE_PATH)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def result(self, *, count=2, limit=None, metric=0.5):
        out = self.root / 'attempt'
        out.mkdir(exist_ok=True)
        obj = {'results': {'scienceqa_img': {'exact_match,none': metric}},
               'config': {'limit': limit},
               'n-samples': {'scienceqa_img': {'original': 2, 'effective': count}}}
        (out / 'run_results.json').write_text(json.dumps(obj))
        (out / 'run_samples_scienceqa_img.jsonl').write_text('\n'.join(json.dumps({'doc_id': x}) for x in range(count))+'\n')
        record = {'target_budget': 1.0, 'scheduler_observed_budget': 1.0,
                  'generation_wall_seconds': 0.1, 'generation_cuda_event_ms': 99.0,
                  'executed_layers': 32, 'total_layers': 32, 'layer_skipping_ratio': 0.0}
        (out / 'measurements.jsonl').write_text((json.dumps(record)+'\n')*count)
        return out

    def test_exit_zero_with_missing_results_is_failure(self):
        with self.assertRaises(ValueError):
            self.p.validate_result(self.root, 'scienceqa_img', expected_count=2)

    def test_limited_run_cannot_become_formal_success(self):
        with self.assertRaises(ValueError):
            self.p.validate_result(self.result(limit=2), 'scienceqa_img', expected_count=2)

    def test_missing_samples_or_nonfinite_metrics_fail(self):
        with self.assertRaises(ValueError):
            self.p.validate_result(self.result(count=1), 'scienceqa_img', expected_count=2)
        with self.assertRaises(ValueError):
            self.p.validate_result(self.result(metric=float('nan')), 'scienceqa_img', expected_count=2)

    def test_mme_requires_finite_cognition_and_perception(self):
        out = self.result()
        path = out / 'run_results.json'
        data = json.loads(path.read_text())
        data['results'] = {'mme': {'mme_perception_score,none': 1000.0, 'mme_cognition_score,none': None}}
        data['n-samples'] = {'mme': {'original': 2, 'effective': 2}}
        path.write_text(json.dumps(data))
        (out / 'run_samples_scienceqa_img.jsonl').rename(out / 'run_samples_mme.jsonl')
        with self.assertRaises(ValueError):
            self.p.validate_result(out, 'mme', expected_count=2)

    def test_resume_rejects_different_dataset_revision(self):
        out = self.result()
        state_path = self.root / 'state.json'
        self.p.atomic_json(state_path, {'status': 'COMPLETED', 'exit_code': 0, 'output': str(out), 'count': 2,
                                      'dataset_revision': 'old-revision'})
        self.assertFalse(self.p.completed_state(state_path, 'scienceqa_img', expected_revision='new-revision'))

    def test_complete_results_validate_and_resume_without_writes(self):
        out = self.result()
        result = self.p.validate_result(out, 'scienceqa_img', expected_count=2)
        state_path = self.root / 'state.json'
        state = {'status': 'COMPLETED', 'exit_code': 0, 'output': str(out), 'count': 2, 'budget': '1.0'}
        self.p.atomic_json(state_path, state)
        before = (out / 'run_results.json').read_bytes()
        self.assertTrue(self.p.completed_state(state_path, 'scienceqa_img'))
        self.assertEqual(before, (out / 'run_results.json').read_bytes())
        self.assertEqual(result['metrics']['exact_match,none'], 0.5)

    def test_truncated_measurements_cannot_resume_or_cleanup(self):
        out = self.result()
        path = out / 'measurements.jsonl'
        path.write_text(path.read_text().splitlines()[0]+'\n')
        state_path = self.root / 'state.json'
        self.p.atomic_json(state_path, {'status': 'COMPLETED', 'exit_code': 0, 'output': str(out), 'count': 2, 'budget': '1.0'})
        with self.assertRaises(ValueError):
            self.p.completed_state(state_path, 'scienceqa_img')

    def test_cleanup_rejects_failed_incomplete_or_foreign_paths(self):
        datasets = self.root / 'datasets'
        data = datasets / 'scienceqa_img'
        data.mkdir(parents=True)
        states = [{'status': 'COMPLETED', 'exit_code': 0}] * 3
        with self.assertRaises(ValueError):
            self.p.assert_cleanup_safe(data, datasets, states)
        with self.assertRaises(ValueError):
            self.p.assert_cleanup_safe(self.root, datasets, states * 2)
        states = [{'status': 'FAILED', 'exit_code': 1}] * 4
        with self.assertRaises(ValueError):
            self.p.assert_cleanup_safe(data, datasets, states)

    def test_validated_four_budget_cleanup_accepts_only_owned_directory(self):
        datasets = self.root / 'datasets'
        data = datasets / 'scienceqa_img'
        data.mkdir(parents=True)
        (data / '.adallava-owned').write_text('scienceqa_img')
        states = [{'status': 'COMPLETED', 'exit_code': 0, 'validated': True, 'budget': b}
                  for b in self.p.BUDGETS]
        self.p.assert_cleanup_safe(data, datasets, states)
        link = datasets / 'pope'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.p.assert_cleanup_safe(link, datasets, states)


if __name__ == '__main__':
    unittest.main()
