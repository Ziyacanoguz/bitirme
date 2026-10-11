"""Create a traceable detector dataset without changing either source."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detection_labels(text):
    """Normalize person boxes and convert polygons to their enclosing boxes."""
    boxes, converted = [], 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if not all(math.isfinite(v) for v in values) or values[0] != 0:
            raise ValueError(f'Invalid person annotation at line {number}')
        if len(values) == 5:
            _, x, y, w, h = values
        elif len(values) >= 7 and len(values) % 2 == 1:
            xs, ys = values[1::2], values[2::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            converted += 1
        else:
            raise ValueError(f'Invalid annotation shape at line {number}')
        if (w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-5
                or max(x+w/2, y+h/2) > 1.00001):
            raise ValueError(f'Invalid box at line {number}')
        boxes.append(f'0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}')
    return '\n'.join(boxes) + ('\n' if boxes else ''), converted


def build(own, friend, output):
    own, friend, output = (Path(p).resolve() for p in (own, friend, output))
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    records, exclusions, seen = [], [], {}

    def add(image, label, split, source, original):
        text, converted = detection_labels(label.read_text())
        if source == 'friend' and not text:
            exclusions.append(dict(image=str(image), label=str(label), source_split=split,
                                   reason='empty_label_requires_visual_review'))
            return
        image_hash = digest(image)
        if image_hash in seen:
            raise ValueError(f'Duplicate image content: {image} and {seen[image_hash]}')
        seen[image_hash] = str(image)
        if source == 'own' and original.get('label_sha256') and digest(label) != original['label_sha256']:
            raise ValueError(f'Reviewed source label changed: {label}')
        name = f'{source}_{image.name}'
        row = dict(split=split, match=original.get('match', 'unknown_friend_match'),
                   image=f'images/{split}/{name}', video=original.get('video', ''),
                   camera=original.get('camera', image.name.split('_')[0]),
                   time_seconds=original.get('time_seconds', ''), frame=original.get('frame', ''),
                   prelabel_count=len(text.splitlines()), reviewed=original.get('reviewed', '0'),
                   source=source, source_image=str(image), source_label=str(label),
                   source_split=original.get('split', split), image_sha256=image_hash,
                   source_label_sha256=digest(label), label_sha256='',
                   converted_polygons=converted)
        records.append((row, image, text))

    with (own / 'manifest.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['reviewed'] != '1':
                raise ValueError('Own dataset must be a reviewed snapshot')
            image = own / row['image']
            add(image, own / 'labels' / row['split'] / (image.stem + '.txt'),
                row['split'], 'own', row)
    for source_split, split in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        images = sorted((friend / source_split / 'images').glob('*.jpg'))
        if not images:
            raise ValueError(f'No images in {source_split}')
        for image in images:
            add(image, friend / source_split / 'labels' / (image.stem + '.txt'),
                split, 'friend', {'split': source_split})
    # Validate all inputs before creating any output; copies are independently editable.
    output.mkdir(parents=True)
    for row, image, text in records:
        destination = output / row['image']
        label = output / 'labels' / row['split'] / (destination.stem + '.txt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
        label.write_text(text)
        row['label_sha256'] = digest(label)
    with (output / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0][0]))
        writer.writeheader()
        writer.writerows(row for row, _, _ in records)
    for filename, names in [('data.yaml', {0: 'person'}), ('data_coco80.yaml',
                           yaml.safe_load((own / 'data_coco80.yaml').read_text())['names'])]:
        data = dict(path=str(output), train='images/train', val='images/val', test='images/test', names=names)
        (output / filename).write_text(yaml.safe_dump(data, sort_keys=False))
    provenance = output / 'provenance'
    provenance.mkdir()
    for name in ['README.dataset.txt', 'README.roboflow.txt', 'data.yaml']:
        shutil.copy2(friend / name, provenance / ('friend_' + name))
    summary = dict(schema='halisaha_merged_dataset_v1', own=str(own), friend=str(friend),
                   counts=dict(Counter(row['split'] for row, _, _ in records)),
                   source_counts=dict(Counter(row['source'] for row, _, _ in records)),
                   annotations=sum(int(row['prelabel_count']) for row, _, _ in records),
                   converted_polygons=sum(row['converted_polygons'] for row, _, _ in records),
                   excluded=exclusions,
                   friend_matches_differ_from_own='confirmed_by_user_2026-09-27',
                   friend_internal_match_split_verified=False,
                   split_policy='Preserve existing splits; friend valid renamed val. No test images moved to training.',
                   review_policy='Own review flags preserved; friend annotations require review, not auto-approved.',
                   test_limit='Friend test is export holdout only; internal match/time independence is unknown.',
                   geometry='Original image pixels retained; square stretch not reversed.')
    (output / 'merge.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--own', type=Path, default=ROOT / 'datasets/halisaha_reviewed407_20260904')
    p.add_argument('--friend', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.own, args.friend, args.output), indent=2))
