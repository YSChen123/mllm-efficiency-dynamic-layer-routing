import json,os,subprocess,sys,tempfile,unittest,venv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from link_training_env import link_training_site

class TrainingEnvironmentTests(unittest.TestCase):
    def test_inherits_eval_packages_and_editables_keeps_train_override_first(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);evalenv=root/'evaluation';trainenv=root/'training'
            venv.create(evalenv,with_pip=False);venv.create(trainenv,with_pip=False)
            def python(env):return env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            def site(env):return Path(subprocess.check_output([str(python(env)),'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip())
            evaluation_site=site(evalenv);training_site=site(trainenv)
            (evaluation_site/'shared_marker.py').write_text('value="evaluation"')
            project=root/'source';project.mkdir();(project/'editable_marker.py').write_text('value="editable"')
            (evaluation_site/'project.pth').write_text(str(project)+'\n')
            (evaluation_site/'override_marker.py').write_text('value="old-nccl"')
            (training_site/'override_marker.py').write_text('value="train-local-nccl"')
            command=[str(python(trainenv)),'-c','import shared_marker,editable_marker,override_marker;print(shared_marker.value,editable_marker.value,override_marker.value)']
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            link_training_site(python(evalenv),python(trainenv))
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'evaluation editable train-local-nccl')

if __name__=='__main__':unittest.main()
