import json
from pathlib import Path
import shlex
import tempfile
import unittest

from roe_repro.common import ROOT, load_config, render_training_script, select_training_rows,checkpoint_view


class PortableTests(unittest.TestCase):
    def test_relocated_checkpoint_changes_only_vision_path_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'checkpoint';source.mkdir()
            original={'mm_vision_tower':'/old/server/clip','hidden_size':4096,'torch_dtype':'bfloat16'}
            raw=json.dumps(original).encode();(source/'config.json').write_bytes(raw)
            (source/'pytorch_model.bin').write_bytes(b'trained weights')
            vision=root/'clip';vision.mkdir()
            target=checkpoint_view({'checkpoint':source,'vision_model':vision},root/'run')
            got=json.loads((target/'config.json').read_text())
            self.assertEqual(got,dict(original,mm_vision_tower=str(vision.resolve())))
            self.assertEqual((source/'config.json').read_bytes(),raw)
            self.assertEqual((target/'pytorch_model.bin').read_bytes(),b'trained weights')

    def test_config_paths_are_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'config.json'
            file.write_text(json.dumps({'runtime':'runs','base_model':'assets/base'}))
            c=load_config(file)
            self.assertEqual(c['runtime'],Path(tmp)/'runs')
            self.assertEqual(c['base_model'],Path(tmp)/'assets/base')

    def test_release_sampler_keeps_original_six_rows_per_ten_in_order(self):
        self.assertEqual(select_training_rows(list(range(23))),
                         [0,1,2,3,4,5,10,11,12,13,14,15,20,21,22])

    def test_training_render_changes_paths_only_even_with_spaces(self):
        source=(ROOT/'upstream/RoE/RoE/scripts/v1_5/finetune_RoE.sh').read_text()
        c={k:Path('/tmp/path with spaces')/k for k in
           ['base_model','vision_model','train_subset','images_dir','checkpoint']}
        actual=shlex.split(render_training_script(source,c),comments=True)
        expected=shlex.split(source,comments=True)
        replacements={'liuhaotian/llava-v1.5-7b':str(c['base_model']),
          'openai/clip-vit-large-patch14-336':str(c['vision_model']),
          '/path/to/data/json':str(c['train_subset']),'/data/qiong_code/data/':str(c['images_dir']),
          '/path/to/checkpoint':str(c['checkpoint'])}
        self.assertEqual(actual,[replacements.get(x,x) for x in expected])
        self.assertEqual(actual[actual.index('--skip_ratio')+1],'0.3')


if __name__=='__main__':unittest.main()
