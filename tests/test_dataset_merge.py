import csv
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from merge_player_datasets import detection_labels, build
from halisaha.training_dataset import inspect_training_dataset


class DatasetIntegrationTests(unittest.TestCase):
    def test_mixed_polygon_and_box_become_detection_rows(self):
        text, count = detection_labels('0 .5 .5 .2 .2\n0 .1 .2 .3 .2 .3 .6 .1 .6')
        self.assertEqual(count, 1)
        rows = [list(map(float, line.split())) for line in text.splitlines()]
        self.assertEqual(len(rows), 2)
        for actual, expected in zip(rows[1], [0, .2, .4, .2, .4]):
            self.assertAlmostEqual(actual, expected)

    def test_bad_coordinates_and_classes_rejected(self):
        for row in ['1 .5 .5 .2 .2', '0 nan .5 .2 .2', '0 .99 .5 .2 .2', '0 .5 .5 0 .2']:
            with self.subTest(row=row), self.assertRaises(ValueError):
                detection_labels(row)

    def fixture(self, root):
        own, friend, output = root/'own', root/'friend', root/'merged'
        own.mkdir()
        records = []
        for split in ['train', 'val']:
            (own/'images'/split).mkdir(parents=True)
            (own/'labels'/split).mkdir(parents=True)
            (own/'images'/split/'old.jpg').write_bytes(('own'+split).encode())
            (own/'labels'/split/'old.txt').write_text('0 .5 .5 .2 .2\n')
            records.append(dict(split=split, image=f'images/{split}/old.jpg', reviewed='1'))
        with (own/'manifest.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        (own/'data_coco80.yaml').write_text('names: {0: person}\n')
        for split in ['train','valid','test']:
            (friend/split/'images').mkdir(parents=True)
            (friend/split/'labels').mkdir(parents=True)
            (friend/split/'images'/'friend.jpg').write_bytes(('friend'+split).encode())
            (friend/split/'labels'/'friend.txt').write_text('0 .5 .5 .2 .2\n')
        (friend/'train'/'images'/'empty.jpg').write_bytes(b'empty')
        (friend/'train'/'labels'/'empty.txt').write_text('')
        for name in ['README.dataset.txt', 'README.roboflow.txt', 'data.yaml']:
            (friend/name).write_text('source metadata')
        return own, friend, output

    def test_merge_preserves_splits_sources_and_review_flags(self):
        with tempfile.TemporaryDirectory() as d:
            own, friend, out = self.fixture(Path(d))
            before = (friend/'train/labels/friend.txt').read_bytes()
            report = build(own, friend, out)
            self.assertEqual(report['counts'], dict(train=2, val=2, test=1))
            self.assertEqual(len(report['excluded']), 1)
            status = inspect_training_dataset(out, out/'data.yaml')
            self.assertEqual(status, dict(train=(1,2), val=(1,2), test=(0,1)))
            self.assertEqual((friend/'train/labels/friend.txt').read_bytes(), before)
            with self.assertRaises(FileExistsError):
                build(own, friend, out)

    def test_wrong_manifest_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            own, friend, out = self.fixture(Path(d)); build(own, friend, out)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                inspect_training_dataset(own, out/'data.yaml')

    def test_changed_reviewed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            own, friend, out = self.fixture(Path(d)); build(own, friend, out)
            (out/'labels/train/own_old.txt').write_text('0 .4 .4 .1 .1\n')
            with self.assertRaisesRegex(ValueError, 'has changed'):
                inspect_training_dataset(out, out/'data.yaml')

    def test_unmanifested_image_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            own, friend, out = self.fixture(Path(d)); build(own, friend, out)
            (out/'images/train/extra.jpg').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'missing from manifest'):
                inspect_training_dataset(out, out/'data.yaml')

    def test_train_val_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            own, friend, out = self.fixture(Path(d)); build(own, friend, out)
            path = out/'data.yaml'; data = yaml.safe_load(path.read_text())
            data['val'] = data['train']; path.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, 'same images'):
                inspect_training_dataset(out, path)


if __name__ == '__main__':
    unittest.main()
