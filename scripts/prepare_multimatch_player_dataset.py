import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from halisaha.roi import apply_mask, create_mask, load_roi  # noqa: E402
from prepare_player_dataset import (  # noqa: E402
    load_rectangle,
    nms,
    prepare_directories,
    sample_times,
    save_yolo_labels,
    yolo_detections,
)


CAMERA_SETTINGS = {
    "camera1": {
        "roi": PROJECT_ROOT / "configs/roi_camera1.json",
        "far_crop": PROJECT_ROOT / "configs/far_crop_camera1.json",
    },
    "camera2": {
        "roi": PROJECT_ROOT / "configs/roi_camera2.json",
        "far_crop": PROJECT_ROOT / "configs/far_crop_camera2.json",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Farkli maclardan sizintisiz YOLO oyuncu veri seti olustur"
    )
    parser.add_argument(
        "--matches-dir",
        type=Path,
        default=PROJECT_ROOT / "assets/videos/matches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/halisaha_player_v2",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt"
        ),
    )
    parser.add_argument("--train-per-camera-match", type=int, default=120)
    parser.add_argument("--val-per-camera-match", type=int, default=20)
    parser.add_argument("--test-per-camera-match", type=int, default=60)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--crop-image-size", type=int, default=1920)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def video_entries(matches_dir):
    plan = {
        "train": {
            "matches": ("match_01", "match_02"),
            "start": 300.0,
            "end": 2400.0,
        },
        "val": {
            "matches": ("match_01", "match_02"),
            "start": 2600.0,
            "end": 3350.0,
        },
        "test": {
            "matches": ("match_03",),
            "start": 300.0,
            "end": 3350.0,
        },
    }
    entries = []
    for split, split_plan in plan.items():
        for match_name in split_plan["matches"]:
            for camera_name, settings in CAMERA_SETTINGS.items():
                entries.append(
                    {
                        "split": split,
                        "match": match_name,
                        "camera": camera_name,
                        "video": matches_dir / match_name / f"{camera_name}.mp4",
                        "roi": settings["roi"],
                        "far_crop": settings["far_crop"],
                        "start": split_plan["start"],
                        "end": split_plan["end"],
                    }
                )
    return entries


def write_dataset_yaml(output):
    single_class = "\n".join(
        [
            f"path: {output.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: player",
            "",
        ]
    )
    (output / "data.yaml").write_text(single_class, encoding="utf-8")

    template = PROJECT_ROOT / "datasets/halisaha_player/data_coco80.yaml"
    lines = template.read_text(encoding="utf-8").splitlines()
    lines[0] = f"path: {output.resolve()}"
    (output / "data_coco80.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "classes.txt").write_text("player\n", encoding="utf-8")


def main():
    args = parse_args()
    entries = video_entries(args.matches_dir)
    for path in [args.model, *[entry["video"] for entry in entries]]:
        if not path.exists():
            raise FileNotFoundError(path)
    for settings in CAMERA_SETTINGS.values():
        for path in settings.values():
            if not path.exists():
                raise FileNotFoundError(path)

    prepare_directories(args.output, args.overwrite)
    model = YOLO(str(args.model))
    per_split = {
        "train": args.train_per_camera_match,
        "val": args.val_per_camera_match,
        "test": args.test_per_camera_match,
    }
    expected = sum(per_split[entry["split"]] for entry in entries)
    completed = 0
    manifest_rows = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": 0, "val": 0, "test": 0}

    for entry in entries:
        capture = cv2.VideoCapture(str(entry["video"]))
        if not capture.isOpened():
            raise RuntimeError(f"Video acilamadi: {entry['video']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        polygon = load_roi(entry["roi"], width, height)
        mask = create_mask((height, width, 3), polygon)
        crop_x1, crop_y1, crop_x2, crop_y2 = load_rectangle(
            entry["far_crop"], width, height
        )
        count = per_split[entry["split"]]

        for sample_index, time_seconds in enumerate(
            sample_times(entry["start"], entry["end"], count)
        ):
            frame_number = round(time_seconds * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Kare okunamadi: {entry['match']} {entry['camera']} "
                    f"frame={frame_number}"
                )

            masked = apply_mask(frame, mask)
            full_detections = yolo_detections(
                model, masked, args.confidence, args.image_size
            )
            crop = masked[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_detections = yolo_detections(
                model, crop, args.confidence, args.crop_image_size
            )
            if len(crop_detections):
                crop_detections[:, [0, 2]] += crop_x1
                crop_detections[:, [1, 3]] += crop_y1
            detections = nms(
                np.concatenate((full_detections, crop_detections), axis=0),
                args.nms_iou,
            )
            detections = np.asarray(
                [
                    detection
                    for detection in detections
                    if cv2.pointPolygonTest(
                        polygon,
                        (
                            float((detection[0] + detection[2]) / 2),
                            float(detection[3]),
                        ),
                        False,
                    )
                    >= 0
                ],
                dtype=np.float32,
            ).reshape(-1, 6)

            split = entry["split"]
            stem = (
                f"{entry['match']}_{entry['camera']}_{frame_number:06d}"
            )
            relative_image = Path("images") / split / f"{stem}.jpg"
            image_path = args.output / relative_image
            label_path = args.output / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
            label_count = save_yolo_labels(
                label_path, detections, width, height
            )
            manifest_rows.append(
                {
                    "split": split,
                    "match": entry["match"],
                    "image": str(relative_image),
                    "video": str(entry["video"].relative_to(PROJECT_ROOT)),
                    "camera": entry["camera"],
                    "time_seconds": f"{time_seconds:.3f}",
                    "frame": frame_number,
                    "prelabel_count": label_count,
                    "reviewed": 0,
                }
            )
            split_counts[split] += 1
            label_counts[split] += label_count
            completed += 1
            if completed % 10 == 0 or completed == expected:
                print(
                    f"Hazirlanan: {completed}/{expected} | "
                    f"son={entry['match']}/{entry['camera']}/{split} "
                    f"etiket={label_count}"
                )
        capture.release()

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_dataset_yaml(args.output)

    print("Veri seti hazirlandi:", args.output)
    print("Goruntu sayilari:", split_counts)
    print("On etiket sayilari:", label_counts)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
