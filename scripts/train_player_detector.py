import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from halisaha.training_dataset import inspect_training_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kontrol edilmis halisaha veri setiyle YOLO player detector egit"
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "datasets/halisaha_player"
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Varsayilan dataset/data.yaml yerine kullanilacak veri tanimi",
    )
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--name", default="yolo26n_halisaha_pilot")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=("auto", "AdamW", "SGD"), default="auto")
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Dondurulacak ilk model katmani sayisi",
    )
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Veri baglantisini ve onay durumunu denetle; egitim baslatma")
    return parser.parse_args()


def review_status(manifest_path):
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    status = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        reviewed = sum(row.get("reviewed") == "1" for row in selected)
        status[split] = (reviewed, len(selected))
    return status


def main():
    args = parse_args()
    data_yaml = args.data_yaml or (args.dataset / "data.yaml")
    manifest = args.dataset / "manifest.csv"
    for path in (data_yaml, manifest, args.model):
        if not path.exists():
            raise FileNotFoundError(path)

    status = inspect_training_dataset(args.dataset, data_yaml)
    print("Etiket kontrol durumu:")
    for split, (reviewed, total) in status.items():
        print(f"  {split}: {reviewed}/{total}")
    required_incomplete = {
        split: values for split, values in status.items() if split in ("train", "val") and values[0] < values[1]
    }
    if args.dry_run:
        print(json.dumps(dict(data_yaml=str(data_yaml.resolve()), review_status=status,
                              ready_for_reviewed_training=not bool(required_incomplete)), indent=2))
        return
    if required_incomplete and not args.allow_unreviewed:
        raise RuntimeError(
            "Train ve validation etiketleri tamamen kontrol edilmedi. "
            "Once scripts/review_yolo_labels.py ile kutulari duzelt."
        )

    import torch
    from ultralytics import YOLO

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS bu Python ortaminda kullanilamiyor. Komutu normal macOS Terminal'de "
            "calistir veya once MPS kurulumunu duzelt."
        )

    model = YOLO(str(args.model))
    train_args = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        cache="disk",
        patience=15,
        seed=42,
        deterministic=True,
        close_mosaic=min(10, max(args.epochs // 5, 1)),
        plots=True,
        project=str(PROJECT_ROOT / "outputs/training"),
        name=args.name,
        optimizer=args.optimizer,
    )
    if args.learning_rate is not None:
        # Ultralytics auto silently ignores lr0. An explicit rate must take effect.
        if args.optimizer == "auto":
            train_args["optimizer"] = "AdamW"
            print("Ozel ogrenme hizi icin AdamW kullaniliyor (auto lr0'i yok sayar).")
        train_args["lr0"] = args.learning_rate
    if args.freeze is not None:
        train_args["freeze"] = args.freeze
    model.train(**train_args)


if __name__ == "__main__":
    main()
