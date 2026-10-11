"""Bind training review status to the actual images selected by the dataset YAML."""
import csv
import hashlib
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def inspect_training_dataset(dataset, data_yaml):
    dataset, data_yaml = Path(dataset).resolve(), Path(data_yaml).resolve()
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get('path') or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root
    if root.resolve() != dataset:
        raise ValueError('data.yaml root does not match --dataset; review manifest would belong to different data')
    with (dataset / 'manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    for row in rows:
        path = (dataset / row['image']).resolve()
        if path in lookup:
            raise ValueError(f'Duplicate manifest image: {path}')
        lookup[path] = row
    status, selected = {}, {}
    for split in ('train', 'val', 'test'):
        if not data.get(split):
            if split != 'test':
                raise ValueError(f'Missing {split} split')
            status[split] = (0, 0)
            continue
        entries = data[split] if isinstance(data[split], list) else [data[split]]
        images = []
        for entry in entries:
            path = (root / entry).resolve()
            if path.is_dir():
                images.extend(p.resolve() for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
            elif path.is_file() and path.suffix == '.txt':
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line:
                        item = Path(line)
                        images.append((path.parent / item).resolve() if not item.is_absolute() else item.resolve())
            else:
                raise ValueError(f'Unsupported or missing dataset path: {path}')
        if not images or len(images) != len(set(images)):
            raise ValueError(f'Empty or duplicated images in {split}')
        selected[split] = set(images)
        reviewed = 0
        for image in images:
            if image not in lookup or not image.is_file():
                raise ValueError(f'Image missing from manifest or disk: {image}')
            row = lookup[image]
            # Manifest split identifies label location even for legacy text-list subsets.
            label = dataset / 'labels' / row['split'] / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            if row.get('reviewed') == '1':
                expected = row.get('label_sha256')
                if expected and hashlib.sha256(label.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Reviewed label has changed: {label}')
                reviewed += 1
        status[split] = (reviewed, len(images))
    if selected['train'] & selected['val']:
        raise ValueError('Train and validation contain the same images')
    if selected['train'] & selected.get('test', set()):
        raise ValueError('Train and test contain the same images')
    return status
