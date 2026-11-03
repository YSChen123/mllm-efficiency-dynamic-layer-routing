import importlib.util,json,sys,tempfile,unittest
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools'))
from materialize import materialize

class MigrationTests(unittest.TestCase):
    def test_new_server_runtime_has_no_old_paths_or_success_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'new-server'
            materialize(REPO,root)
            for directory in ['scripts','training_repro','evaluation_repro/scripts']:
                for p in (root/directory).rglob('*'):
                    if p.is_file() and p.suffix in {'.py','.sh','.json'}:
                        text=p.read_text(encoding='utf-8')
                        self.assertNotIn('/home/SigConv',text,str(p))
            self.assertTrue((root/'training_repro/source/AdaLLaVA/src/adallava/train/train.py').is_file())
            self.assertTrue((root/'LLaVA/llava/train/llava_trainer.py').is_file())
            self.assertFalse((root/'training_repro/smoke/SMOKE_PASSED.json').exists())
            self.assertFalse(list((root/'configs/states').glob('*.json')))
            self.assertEqual(json.loads((root/'configs/datasets/mme.json').read_text())['revision'],'d6c9023f017b564f7b3ccccf5348166bce8fdbcd')

    def test_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'existing';root.mkdir();(root/'important').write_text('keep')
            with self.assertRaises(FileExistsError):materialize(REPO,root)
            self.assertEqual((root/'important').read_text(),'keep')

    def test_tmux_shell_metacharacters_are_rejected_before_creating_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['ada(test)','ada&test','ada|test','ada>test']:
                root=Path(folder)/name
                with self.assertRaises(ValueError):materialize(REPO,root)
                self.assertFalse(root.exists())

    def test_executed_source_is_unmodified_by_relocation(self):
        source=REPO/'reproduction/training/train_entry.py'
        original=source.read_bytes()
        with tempfile.TemporaryDirectory() as folder:materialize(REPO,Path(folder)/'runtime')
        self.assertEqual(source.read_bytes(),original)

if __name__=='__main__':unittest.main()
