import json,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate import prepare_role
import evaluate

class FreshEvaluationTests(unittest.TestCase):
    def model(self,root):
        model=root/'model';model.mkdir();(model/'config.json').write_text('{}')
        (model/'model-1.safetensors').write_bytes(b'fake-only-for-identity-test')
        (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'model-1.safetensors'}}))
        return model

    def test_changed_model_cannot_reuse_result_directory(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);model=self.model(root);out=root/'output'
            # Linux creates these real links in the deployment rehearsal; Windows
            # requires admin/developer mode for symlinks. This test isolates identity.
            with patch.object(Path,'symlink_to'):
                prepare_role(root,out,'official',model,[])
            (model/'model-1.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity'):prepare_role(root,out,'official',model,[])

    def test_reproduced_requires_actual_training_proof(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);model=self.model(root)
            with self.assertRaises(FileNotFoundError):prepare_role(root,root/'output','reproduced',model,[])
            self.assertFalse((root/'output').exists())

    def test_main_records_status_even_when_optional_collection_fails(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);out=root/'output';(out/'configs/datasets').mkdir(parents=True)
            (out/'configs/datasets/scienceqa_img.json').write_text(json.dumps({'count':3}))
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduction/evaluation'))
            import eval_policy
            calls=[]
            def collect():raise OSError('summary disk failure')
            fake=SimpleNamespace(environment=lambda task:{},validate_measurements=lambda *a:[],
                TASKS={'scienceqa_img':()},run_eval=lambda *a,**k:calls.append((a,k)),collect=collect,
                atomic_json=lambda path,value:Path(path).write_text(json.dumps(value)))
            argv=['evaluate.py','--root',str(root),'--output',str(out),'--model',str(root/'model'),
                '--role','official','--tasks','scienceqa_img','--budgets','1.0','--smoke']
            with patch.object(evaluate,'prepare_role'),patch.object(sys,'argv',argv),patch.dict(sys.modules,{'pipeline':fake,'eval_policy':eval_policy}):
                evaluate.main()
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((out/'EVALUATION_STATUS.json').read_text())['status'],'SMOKE_PASSED')

if __name__=='__main__':unittest.main()
