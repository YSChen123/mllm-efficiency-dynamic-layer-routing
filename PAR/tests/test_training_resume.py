import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from par_repro.config import load_config
from par_repro.io import atomic_json, file_sha
from par_repro.math_core import sample_layer_actions
from par_repro.training import train


class TrainingRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        config = load_config()
        config['router'] = dict(input_size=16, hidden_size=16, intermediate_size=32, num_heads=4,
                                candidate_count=16, init_std=.02, head_hidden_size=8)
        config['training'].update(epochs=2, reported_epochs=[1, 2])
        folder = root / 'preferences/joint/training'
        (folder / 'items').mkdir(parents=True)
        (folder / 'features').mkdir()
        files = {}
        for i in range(4):
            generator = torch.Generator().manual_seed(i)
            pack = {'visual': torch.randn(576, 16, generator=generator), 'text': torch.randn(i + 1, 16, generator=generator)}
            feature = folder / f'features/{i:05d}.pt'
            torch.save(pack, feature)
            actions = [{'layers': layers, 'tokens': list(range(400))} for layers in sample_layer_actions(i)]
            item = {'identity': 'test-data', 'actions': actions, 'chosen_index': 0, 'rejected_index': 4,
                    'positive_indices': [0, 1], 'negative_indices': [3, 4], 'feature': f'features/{i:05d}.pt',
                    'feature_sha256': file_sha(feature)}
            metadata = folder / f'items/{i:05d}.json'
            atomic_json(metadata, item)
            files[str(metadata.relative_to(folder))] = file_sha(metadata)
            files[item['feature']] = item['feature_sha256']
        manifest = {'count': 4, 'scope': 'SMOKE_ONLY', 'identity': 'test-data', 'files': files}
        atomic_json(folder / 'manifest.json', manifest)
        atomic_json(root / 'prepared.json', {'status': 'PREPARED', 'source_commit': 'test', 'models': {}})
        return config, folder, manifest

    def cpu_patches(self):
        return patch.object(torch.nn.Module, 'cuda', lambda obj, *a, **kw: obj), patch.object(torch.Tensor, 'cuda', lambda obj, *a, **kw: obj)

    def test_failure_before_first_save_has_recoverable_step_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, folder, manifest = self.fixture(root)
            a, b = self.cpu_patches()
            with a, b, patch('par_repro.preferences.checked_preferences', return_value=(folder, manifest)), patch('par_repro.batching.accumulated_update', side_effect=RuntimeError('injected interruption')):
                with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    train(root, cfg, 'early')
                checkpoint = root / 'checkpoints/early/latest.pt'
                self.assertTrue(checkpoint.exists())
                self.assertEqual(torch.load(checkpoint, weights_only=True)['step'], 0)
                with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    train(root, cfg, 'early', resume=True)

    def test_resume_optimizer_next_update_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, folder, manifest = self.fixture(root)
            a, b = self.cpu_patches()
            with a, b, patch('par_repro.preferences.checked_preferences', return_value=(folder, manifest)):
                train(root, cfg, 'continuous')
                train(root, cfg, 'resumed', max_steps=1)
                train(root, cfg, 'resumed', resume=True)
            first = torch.load(root / 'checkpoints/continuous/latest.pt', weights_only=True)
            second = torch.load(root / 'checkpoints/resumed/latest.pt', weights_only=True)
            self.assertEqual(first['step'], second['step'])
            for key in first['router']:
                self.assertTrue(torch.equal(first['router'][key], second['router'][key]), key)


if __name__ == '__main__':
    unittest.main()
