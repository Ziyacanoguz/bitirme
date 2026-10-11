import argparse
import csv
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.field_projection import FieldProjector  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Track ayak noktalarını ortak saha düzlemine taşı")
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video boyutu okunamadi: {args.video}")
    projector = FieldProjector(args.calibration, width, height)
    if args.output.exists():
        raise FileExistsError(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.tracks.open(newline="", encoding="utf-8") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        fieldnames = list(dict.fromkeys(list(reader.fieldnames or []) +
                                       ["field_x", "field_y", "calibration_supported"]))
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            row = projector.project([row])[0]
            row["field_x"] = f"{row['field_x']:.6f}"
            row["field_y"] = f"{row['field_y']:.6f}"
            writer.writerow(row)
            count += 1
    print(f"Dönüştürülen track satırı: {count}")
    print(f"Çıktı: {args.output}")


if __name__ == "__main__":
    main()
