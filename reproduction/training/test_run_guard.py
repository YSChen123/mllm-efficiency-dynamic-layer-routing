import json,tempfile,unittest
from pathlib import Path
import run_guard

class RunGuardTest(unittest.TestCase):
    def test_partial_newer_checkpoint_never_displaces_complete_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for step in [2,5]:(root/f'checkpoint-{step}').mkdir()
            cp=root/'checkpoint-2'
            (cp/'trainer_state.json').write_text(json.dumps({'global_step':2}))
            (cp/'COMPLETE.json').write_text(json.dumps({'step':2}))
            self.assertEqual(run_guard.latest_complete(root),cp)
    def test_quota_reserve_includes_incoming_checkpoint(self):
        self.assertFalse(run_guard.has_capacity(100,30,65,10))
        self.assertTrue(run_guard.has_capacity(110,30,65,10))
    def test_formal_gate_requires_full_annotation_and_all_images(self):
        ready={'status':'READY','total_samples':665298,'unique_images':349033}
        self.assertFalse(run_guard.data_ready(ready))
        ready['unique_images']=349034
        self.assertTrue(run_guard.data_ready(ready))
    def test_checkpoint_requires_matching_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint-2';cp.mkdir()
            (cp/'trainer_state.json').write_text(json.dumps({'global_step':1}))
            (cp/'COMPLETE.json').write_text(json.dumps({'step':2}))
            self.assertIsNone(run_guard.latest_complete(Path(tmp)))

if __name__=='__main__':unittest.main()
