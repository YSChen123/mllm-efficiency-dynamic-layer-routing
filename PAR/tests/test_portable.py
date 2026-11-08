import json
import tempfile
import unittest
from pathlib import Path
from par_repro.io import atomic_json, bind_identity, file_sha, check_prediction_prefix, safe_extract_zip, reconcile_step_log
from par_repro.data import choose_training, choose_validation, seed_prompt, image_id
from par_repro.metrics import pope_scores, seed_scores
from par_repro.checkpoints import validate_legacy_configuration
from par_repro.config import load_config


class PortableTests(unittest.TestCase):
    def test_changed_identity_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'identity.json'
            bind_identity(p, {'beta': 1})
            bind_identity(p, {'beta': 1})
            with self.assertRaises(ValueError):
                bind_identity(p, {'beta': 2})
            self.assertEqual(json.loads(p.read_text()), {'beta': 1})

    def test_training_ids_deterministic_and_ordered(self):
        rows = [{'id': i, 'image': f'{i:012d}.jpg'} for i in range(6000)]
        a = choose_training(rows)
        b = choose_training(rows)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5000)
        self.assertEqual(len({r['source_index'] for r in a}), 5000)
        self.assertEqual(a[0]['record'], rows[a[0]['source_index']])

    def test_validation_excludes_images_and_keeps_unique(self):
        rows = [{'image': f'{i:012d}.jpg', 'conversations': [{'from': 'human', 'value': 'q'}]} for i in range(12)]
        rows.append(rows[-1])
        selected = choose_validation(rows, {0, 1}, {2, 3}, 8, 20261108)
        ids = [image_id(row['record']['image']) for row in selected]
        self.assertEqual(set(ids), set(range(4, 12)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_composite_seed_ids_and_partial_results(self):
        keys = [('3', '49914'), ('9', '49914')]
        rows = [{'category': '3', 'question_id': 49914, 'text': 'A'}]
        self.assertEqual(check_prediction_prefix(rows, keys), 1)
        with self.assertRaises(ValueError):
            check_prediction_prefix(rows, keys, complete=True)
        with self.assertRaises(ValueError):
            check_prediction_prefix(rows + rows, keys)

    def test_zip_path_escape_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            z = Path(folder) / 'bad.zip'
            with zipfile.ZipFile(z, 'w') as archive:
                archive.writestr('../escaped.txt', 'bad')
            with self.assertRaises(ValueError):
                safe_extract_zip(z, Path(folder) / 'images')
            self.assertFalse((Path(folder) / 'escaped.txt').exists())

    def test_pope_original_parser_and_complete_groups(self):
        rows = []
        for group in ['adversarial', 'popular', 'random']:
            rows += [{'category': group, 'text': 'Yes, it is.', 'label': 'yes'},
                     {'category': group, 'text': 'There is no dog.', 'label': 'no'}]
        scores = pope_scores(rows)
        self.assertEqual(scores['macro_f1_percent'], 100.)
        with self.assertRaises(ValueError):
            pope_scores(rows[:2])

    def test_seed_exact_string_match_and_prompt(self):
        rows = [{'text': 'A', 'answer': 'A', 'category': '1'}, {'text': 'A.', 'answer': 'A', 'category': '1'}]
        self.assertEqual(seed_scores(rows)['accuracy_percent'], 50.)
        case = {'question': 'Q', 'choice_a': 'a', 'choice_b': 'b', 'choice_c': 'c', 'choice_d': 'd'}
        self.assertEqual(seed_prompt(case), "Q\nA. a\nB. b\nC. c\nD. d\nAnswer with the option's letter from the given choices directly.")

    def test_resume_archives_unsaved_log_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'steps.jsonl'
            original = '{"step": 1}\n{"step": 2}\n{"step": '
            p.write_text(original)
            reconcile_step_log(p, 1)
            self.assertEqual(p.read_text(), '{"step": 1}\n')
            archives = list(Path(folder).glob('steps.unsaved.*.jsonl'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(), original)
            with self.assertRaises(ValueError):
                reconcile_step_log(p, 2)

    def test_converter_cannot_promote_smoke_to_formal(self):
        cfg = load_config()
        old = {'base': {'router': cfg['router']}, 'trial': {'attention_mode': 'bidirectional', 'global_batch_size': 4}}
        with self.assertRaises(ValueError):
            validate_legacy_configuration(old, 'x:scope=smoke:count=4:trial=x', cfg)
        validate_legacy_configuration(old, 'x:scope=formal:count=5000:trial=x', cfg)
        old['trial']['attention_mode'] = 'causal'
        with self.assertRaises(ValueError):
            validate_legacy_configuration(old, 'x:scope=formal:count=5000:trial=x', cfg)


if __name__ == '__main__':
    unittest.main()
