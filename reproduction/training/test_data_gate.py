import io, json, tarfile, tempfile, unittest
from pathlib import Path
from PIL import Image
import data_gate

class DataGateTest(unittest.TestCase):
    def test_original_bytes_are_preserved_and_only_referenced_images_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=io.BytesIO(); Image.new('RGB',(8,8)).save(image,format='JPEG'); raw=image.getvalue()
            archive=root/'source.tar.gz'
            with tarfile.open(archive,'w:gz') as stream:
                for name,content in [('coco/images/train2017/a.jpg',raw),('coco/images/train2017/unused.jpg',raw),('../outside.jpg',raw)]:
                    entry=tarfile.TarInfo(name); entry.size=len(content); stream.addfile(entry,io.BytesIO(content))
            refs=['coco/train2017/a.jpg']
            report=data_gate.extract_referenced(archive,root/'images',refs)
            self.assertEqual(report['found'],1)
            self.assertEqual((root/'images'/refs[0]).read_bytes(),raw)
            self.assertFalse((root/'images/coco/train2017/unused.jpg').exists())
            self.assertFalse((root/'outside.jpg').exists())
    def test_missing_image_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'missing'):
                data_gate.validate_images(Path(tmp),['ocr_vqa/images/notfound.jpg'])
    def test_corrupt_image_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'bad.jpg').write_bytes(b'broken jpeg')
            with self.assertRaisesRegex(ValueError,'unreadable'):
                data_gate.validate_images(root,['bad.jpg'])
    def test_archive_path_normalization(self):
        self.assertEqual(data_gate.canonical_name('ocr-vqa/images/abc.jpg'),'ocr_vqa/images/abc.jpg')
        self.assertEqual(data_gate.canonical_name('coco/images/train2017/a.jpg'),'coco/train2017/a.jpg')
        self.assertIsNone(data_gate.canonical_name('../evil.jpg'))
    def test_existing_original_image_mismatch_blocks_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=io.BytesIO();Image.new('RGB',(8,8)).save(image,format='JPEG');raw=image.getvalue()
            (root/'images/coco/train2017').mkdir(parents=True)
            (root/'images/coco/train2017/a.jpg').write_bytes(raw+b'original source marker')
            archive=root/'source.tar.gz'
            with tarfile.open(archive,'w:gz') as stream:
                entry=tarfile.TarInfo('coco/images/train2017/a.jpg');entry.size=len(raw);stream.addfile(entry,io.BytesIO(raw))
            with self.assertRaisesRegex(ValueError,'differs from existing'):
                data_gate.extract_referenced(archive,root/'images',['coco/train2017/a.jpg'])

if __name__=='__main__': unittest.main()
