"""Immutable reviewed-only dataset with a shared temporal validation window."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


def choose_split(row, start, end, guard):
    second = float(row["time_seconds"])
    if start <= second <= end:
        return "val"
    if start - guard <= second <= end + guard:
        return "excluded_guard"
    return "train"


def validate_label(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid person label: {path}:{number}")
        x, y, w, h = map(float, parts[1:])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box: {path}:{number}")
        if min(x - w / 2, y - h / 2) < -1e-5 or max(x + w / 2, y + h / 2) > 1.00001:
            raise ValueError(f"Box outside image: {path}:{number}")
    return len(lines)


def main():
    p = argparse.ArgumentParser(description="Onayli etiketleri zaman ayrimli ve bagimsiz kopyaya al")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val-start", type=float, default=900)
    p.add_argument("--val-end", type=float, default=1040)
    p.add_argument("--guard-seconds", type=float, default=35)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Snapshot already exists: {args.output}")
    if args.val_end <= args.val_start or args.guard_seconds < 0:
        raise ValueError("Invalid validation interval")
    with (args.dataset / "manifest.csv").open(newline="", encoding="utf-8") as f:
        reviewed = [r for r in csv.DictReader(f) if r["split"] == "train" and r.get("reviewed") == "1"]
    counts, groups, records, boxes = Counter(), Counter(), [], Counter()
    # Validate everything before creating the snapshot. Original labels are never changed.
    for row in reviewed:
        split = choose_split(row, args.val_start, args.val_end, args.guard_seconds)
        image = args.dataset / row["image"]
        label = args.dataset / "labels/train" / (image.stem + ".txt")
        if not image.exists():
            raise FileNotFoundError(image)
        count = validate_label(label)
        records.append((dict(row), split, image, label, count))
        counts[split] += 1
        groups[f"{row['match']}/{row['camera']}/{split}"] += 1
    if counts["train"] < 20 or counts["val"] < 8:
        raise ValueError("Not enough reviewed samples in the selected train/val windows")
    output_rows = []
    for row, split, image, label, count in records:
        if split == "excluded_guard":
            continue
        destination_image = args.output / "images" / split / image.name
        destination_label = args.output / "labels" / split / label.name
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination_image)
        shutil.copy2(label, destination_label)
        row.update(split=split, image=str(destination_image.relative_to(args.output)),
                   label_sha256=hashlib.sha256(destination_label.read_bytes()).hexdigest())
        output_rows.append(row)
        boxes[split] += count
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    template = yaml.safe_load((args.dataset / "data_coco80.yaml").read_text())
    template.update(path=str(args.output.resolve()), train="images/train", val="images/val")
    template.pop("test", None)  # Never present validation as an independent test set.
    (args.output / "data_coco80.yaml").write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    summary = dict(source=str(args.dataset.resolve()), reviewed=len(reviewed), counts=dict(counts),
                   boxes=dict(boxes), groups=dict(groups), validation_seconds=[args.val_start, args.val_end],
                   guard_seconds=args.guard_seconds,
                   note="Same-match temporal validation, not a held-out-match test. Test match remains untouched.")
    (args.output / "snapshot.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
