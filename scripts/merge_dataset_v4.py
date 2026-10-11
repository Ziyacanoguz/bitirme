"""Merge halisaha_merged_v3 with verified usable auto-mined frames from dataset5(selim) into halisaha_merged_v4."""
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_labels(text):
    """Ensure person boxes are normalized class 0."""
    boxes = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        values = list(map(float, line.split()))
        if not all(math.isfinite(v) for v in values):
            continue
        if len(values) == 5:
            _, x, y, w, h = values
        elif len(values) >= 7 and len(values) % 2 == 1:
            xs, ys = values[1::2], values[2::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue

        w = min(max(w, 0.001), 1.0)
        h = min(max(h, 0.001), 1.0)
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)

        boxes.append(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
    return "\n".join(boxes) + ("\n" if boxes else "")


def build_merged_v4(output_path: Path):
    if output_path.exists():
        print(f"Cleaning existing {output_path}...")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    for split in ("train", "val"):
        (output_path / "images" / split).mkdir(parents=True)
        (output_path / "labels" / split).mkdir(parents=True)

    seen_hashes = {}
    stats = Counter()

    # 1. First copy entire halisaha_merged_v3
    v3_root = ROOT / "datasets/halisaha_merged_v3"
    print(f"Importing baseline from {v3_root}...")
    for split in ("train", "val"):
        img_dir = v3_root / "images" / split
        lbl_dir = v3_root / "labels" / split
        if not img_dir.exists():
            continue
        for img_p in sorted(img_dir.glob("*.*")):
            if img_p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            h = digest(img_p)
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            
            seen_hashes[h] = img_p.name
            out_img = output_path / "images" / split / img_p.name
            out_lbl = output_path / "labels" / split / f"{img_p.stem}.txt"
            
            shutil.copy2(img_p, out_img)
            shutil.copy2(lbl_p, out_lbl)
            stats[f"v3_{split}"] += 1

    print(f"Baseline imported: {stats['v3_train']} train, {stats['v3_val']} val images.")

    # 2. Import 100% verified auto-mined sets from dataset5(selim)
    mined_sources = [
        ("mined_20260914", ROOT / "dataset5(selim)/mined_20260914"),
        ("mined_20260914b", ROOT / "dataset5(selim)/mined_20260914b"),
        ("mined_20260925", ROOT / "dataset5(selim)/mined_20260925"),
    ]

    new_added = 0
    duplicate_count = 0

    for mname, mroot in mined_sources:
        if not mroot.exists():
            print(f"Warning: {mroot} not found!")
            continue
        img_dir = mroot / "images"
        lbl_dir = mroot / "labels"
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"Processing {mname} ({len(imgs)} images)...")
        
        for idx, img_p in enumerate(imgs):
            h = digest(img_p)
            if h in seen_hashes:
                duplicate_count += 1
                continue
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            
            lbl_content = lbl_p.read_text(encoding="utf-8").strip()
            if not lbl_content:
                continue
            
            clean_lbl = normalize_labels(lbl_content)
            if not clean_lbl.strip():
                continue

            # Deterministic split: 85% train, 15% val based on hash
            # (or sequential split to avoid adjacent frame leakage: every 7th frame to val)
            target_split = "val" if (idx % 7 == 0) else "train"
            
            prefix = f"d5_{mname}_"
            new_img_name = f"{prefix}{img_p.name}"
            new_lbl_name = f"{prefix}{img_p.stem}.txt"

            out_img = output_path / "images" / target_split / new_img_name
            out_lbl = output_path / "labels" / target_split / new_lbl_name

            shutil.copy2(img_p, out_img)
            out_lbl.write_text(clean_lbl, encoding="utf-8")
            
            seen_hashes[h] = new_img_name
            new_added += 1
            stats[f"d5_{target_split}"] += 1

    print(f"Successfully added {new_added} new images from dataset5 ({duplicate_count} duplicates skipped).")

    # Count boxes
    total_boxes = 0
    split_boxes = Counter()
    split_images = Counter()
    for split in ("train", "val"):
        lbl_dir = output_path / "labels" / split
        for lf in lbl_dir.glob("*.txt"):
            count = len([l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()])
            split_boxes[split] += count
            total_boxes += count
            split_images[split] += 1

    print("=== HALISAHA MERGED V4 SUMMARY ===")
    print(f"Train images: {split_images['train']} | boxes: {split_boxes['train']}")
    print(f"Val images:   {split_images['val']} | boxes: {split_boxes['val']}")
    print(f"Total images: {sum(split_images.values())} | boxes: {total_boxes}")

    # Generate dataset.yaml
    data_yaml = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }
    with (output_path / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"Written dataset.yaml to {output_path / 'dataset.yaml'}")


if __name__ == "__main__":
    out_dir = ROOT / "datasets/halisaha_merged_v4"
    build_merged_v4(out_dir)
