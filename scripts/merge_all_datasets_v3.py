"""Merge dataset2, dataset3, dataset4, and halisaha_merged_v1 into halisaha_merged_v3."""
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


def normalize_labels(text):
    """Normalize person boxes and convert polygons to their enclosing boxes."""
    boxes, converted = [], 0
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        values = list(map(float, line.split()))
        if not all(math.isfinite(v) for v in values):
            continue
        
        cls_id = int(values[0])
        if cls_id != 0:
            cls_id = 0
        
        if len(values) == 5:
            _, x, y, w, h = values
        elif len(values) >= 7 and len(values) % 2 == 1:
            xs, ys = values[1::2], values[2::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            converted += 1
        else:
            continue

        w = min(max(w, 0.001), 1.0)
        h = min(max(h, 0.001), 1.0)
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)

        boxes.append(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
    return "\n".join(boxes) + ("\n" if boxes else ""), converted


def build_merged_v3(output_path: Path):
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    records = []
    seen_hashes = {}

    sources = [
        {"name": "merged_v1", "root": ROOT / "datasets/halisaha_merged_v1"},
        {"name": "dataset2", "root": ROOT / "dataset2"},
        {"name": "dataset3", "root": ROOT / "dataset3"},
        {"name": "dataset4", "root": ROOT / "dataset4"},
    ]

    for src in sources:
        sname = src["name"]
        sroot = src["root"]
        if not sroot.exists():
            print(f"Warning: {sroot} does not exist, skipping.")
            continue

        split_map = {
            "train": "train",
            "val": "val",
            "valid": "val",
            "test": "test"
        }

        for folder_split in ["train", "val", "valid", "test"]:
            target_split = split_map[folder_split]

            img_dir = sroot / "images" / folder_split
            if not img_dir.exists():
                img_dir = sroot / folder_split / "images"
            if not img_dir.exists():
                img_dir = sroot / folder_split

            lbl_dir = sroot / "labels" / folder_split
            if not lbl_dir.exists():
                lbl_dir = sroot / folder_split / "labels"
            if not lbl_dir.exists():
                lbl_dir = sroot / folder_split

            if not img_dir.exists():
                continue

            images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
            for img in images:
                lbl = lbl_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    alt_lbl = sroot / "labels" / f"{img.stem}.txt"
                    if alt_lbl.exists():
                        lbl = alt_lbl

                lbl_text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
                norm_text, converted = normalize_labels(lbl_text)

                img_hash = digest(img)
                if img_hash in seen_hashes:
                    print(f"Skipping duplicate image {img} (matches {seen_hashes[img_hash]})")
                    continue
                seen_hashes[img_hash] = str(img)

                new_img_name = f"{sname}_{img.name}"
                dest_img_rel = f"images/{target_split}/{new_img_name}"
                dest_lbl_rel = f"labels/{target_split}/{sname}_{img.stem}.txt"

                dest_img = output_path / dest_img_rel
                dest_lbl = output_path / dest_lbl_rel

                dest_img.parent.mkdir(parents=True, exist_ok=True)
                dest_lbl.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(img, dest_img)
                dest_lbl.write_text(norm_text, encoding="utf-8")

                box_count = len([line for line in norm_text.splitlines() if line.strip()])

                row = {
                    "split": target_split,
                    "match": f"match_{sname}",
                    "image": dest_img_rel,
                    "video": sname,
                    "camera": "cam0",
                    "time_seconds": "0",
                    "frame": "0",
                    "prelabel_count": str(box_count),
                    "reviewed": "1",
                    "source": sname,
                    "source_image": str(img),
                    "source_label": str(lbl) if lbl.exists() else "",
                    "source_split": folder_split,
                    "image_sha256": img_hash,
                    "source_label_sha256": digest(lbl) if lbl.exists() else "",
                    "label_sha256": digest(dest_lbl),
                    "converted_polygons": str(converted),
                }
                records.append(row)

    fieldnames = list(records[0].keys())
    with (output_path / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    data_yaml_content = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person"},
    }
    with (output_path / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml_content, f, sort_keys=False)

    coco_names = {0: "person"}
    for i in range(1, 80):
        coco_names[i] = f"class_{i}"
    data_coco80_content = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": coco_names,
    }
    with (output_path / "data_coco80.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_coco80_content, f, sort_keys=False)

    counts = Counter(r["split"] for r in records)
    sources_cnt = Counter(r["source"] for r in records)
    total_boxes = sum(int(r["prelabel_count"]) for r in records)

    summary = {
        "schema": "halisaha_merged_dataset_v3",
        "output": str(output_path.resolve()),
        "counts": dict(counts),
        "source_counts": dict(sources_cnt),
        "total_images": len(records),
        "total_boxes": total_boxes,
    }
    (output_path / "merge.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Merge v3 completed successfully!")
    print(json.dumps(summary, indent=2))
    return summary


def build_dataset4_standalone(output_path: Path):
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    records = []
    sroot = ROOT / "dataset4"
    split_map = {"train": "train", "val": "val", "valid": "val", "test": "test"}

    for folder_split in ["train", "valid", "test"]:
        target_split = split_map[folder_split]
        img_dir = sroot / folder_split / "images" if (sroot / folder_split / "images").exists() else sroot / folder_split
        lbl_dir = sroot / folder_split / "labels" if (sroot / folder_split / "labels").exists() else sroot / folder_split

        for img in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
            lbl = lbl_dir / f"{img.stem}.txt"
            lbl_text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
            norm_text, converted = normalize_labels(lbl_text)

            dimg = output_path / "images" / target_split / f"dataset4_{img.name}"
            dlbl = output_path / "labels" / target_split / f"dataset4_{img.stem}.txt"

            dimg.parent.mkdir(parents=True, exist_ok=True)
            dlbl.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(img, dimg)
            dlbl.write_text(norm_text, encoding="utf-8")

            box_count = len([line for line in norm_text.splitlines() if line.strip()])
            records.append({
                "split": target_split, "match": "dataset4", "image": f"images/{target_split}/dataset4_{img.name}",
                "video": "dataset4", "camera": "cam0", "time_seconds": "0", "frame": "0",
                "prelabel_count": str(box_count), "reviewed": "1", "source": "dataset4",
                "source_image": str(img), "source_label": str(lbl), "source_split": folder_split,
                "image_sha256": digest(img), "source_label_sha256": digest(lbl) if lbl.exists() else "",
                "label_sha256": digest(dlbl), "converted_polygons": str(converted)
            })

    with (output_path / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    data_yaml_content = {
        "path": str(output_path.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "person"}
    }
    with (output_path / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml_content, f, sort_keys=False)

    print("dataset4_standalone created! Total images:", len(records))


if __name__ == "__main__":
    build_merged_v3(ROOT / "datasets/halisaha_merged_v3")
    build_dataset4_standalone(ROOT / "datasets/dataset4_standalone")
