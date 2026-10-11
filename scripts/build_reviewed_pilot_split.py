import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kontrol edilmis train goruntulerinden gecici pilot split olustur"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/halisaha_player_v2",
    )
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="reviewed_pilot200")
    return parser.parse_args()


def allocate_validation(groups, requested):
    total = sum(len(rows) for rows in groups.values())
    allocation = {
        key: max(1, int(requested * len(rows) / total))
        for key, rows in groups.items()
    }
    while sum(allocation.values()) < requested:
        key = max(groups, key=lambda item: len(groups[item]) - allocation[item])
        allocation[key] += 1
    while sum(allocation.values()) > requested:
        candidates = [key for key, count in allocation.items() if count > 1]
        key = max(candidates, key=lambda item: allocation[item])
        allocation[key] -= 1
    return allocation


def main():
    args = parse_args()
    manifest = args.dataset / "manifest.csv"
    template = args.dataset / "data_coco80.yaml"
    if not manifest.exists() or not template.exists():
        raise FileNotFoundError("Manifest veya data_coco80.yaml bulunamadi")

    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    reviewed = [
        row
        for row in rows
        if row["split"] == "train" and row.get("reviewed") == "1"
    ]
    if len(reviewed) <= args.validation_count:
        raise RuntimeError("Pilot split icin yeterli kontrol edilmis train goruntusu yok")

    groups = defaultdict(list)
    for row in reviewed:
        groups[(row.get("match", "unknown"), row["camera"])].append(row)
    allocation = allocate_validation(groups, args.validation_count)
    rng = random.Random(args.seed)
    train_rows = []
    val_rows = []
    for key in sorted(groups):
        selected = list(groups[key])
        rng.shuffle(selected)
        count = allocation[key]
        val_rows.extend(selected[:count])
        train_rows.extend(selected[count:])

    train_list = args.dataset / f"{args.prefix}_train.txt"
    val_list = args.dataset / f"{args.prefix}_val.txt"
    train_list.write_text(
        "\n".join(str((args.dataset / row["image"]).resolve()) for row in train_rows)
        + "\n",
        encoding="utf-8",
    )
    val_list.write_text(
        "\n".join(str((args.dataset / row["image"]).resolve()) for row in val_rows)
        + "\n",
        encoding="utf-8",
    )

    template_lines = template.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(template_lines):
        if line.startswith("train:"):
            template_lines[index] = f"train: {train_list.name}"
        elif line.startswith("val:"):
            template_lines[index] = f"val: {val_list.name}"
        elif line.startswith("test:"):
            template_lines[index] = f"test: {val_list.name}"
    yaml_path = args.dataset / f"data_{args.prefix}_coco80.yaml"
    yaml_path.write_text("\n".join(template_lines) + "\n", encoding="utf-8")

    print(f"Reviewed toplam: {len(reviewed)}")
    print(f"Pilot train: {len(train_rows)}")
    print(f"Pilot validation: {len(val_rows)}")
    for key in sorted(groups):
        train_count = sum(
            (row.get("match", "unknown"), row["camera"]) == key
            for row in train_rows
        )
        val_count = sum(
            (row.get("match", "unknown"), row["camera"]) == key
            for row in val_rows
        )
        print(f"  {key[0]}/{key[1]}: train={train_count} val={val_count}")
    print("Data YAML:", yaml_path)


if __name__ == "__main__":
    main()
