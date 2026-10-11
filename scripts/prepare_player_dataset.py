import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.roi import apply_mask, create_mask, load_roi  # noqa: E402


CAMERAS = (
    {
        "name": "camera1",
        "video": PROJECT_ROOT / "assets/videos/camera1.mp4",
        "roi": PROJECT_ROOT / "configs/roi_camera1.json",
        "far_crop": PROJECT_ROOT / "configs/far_crop_camera1.json",
    },
    {
        "name": "camera2",
        "video": PROJECT_ROOT / "assets/videos/camera2.mp4",
        "roi": PROJECT_ROOT / "configs/roi_camera2.json",
        "far_crop": PROJECT_ROOT / "configs/far_crop_camera2.json",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iki kamera videosundan YOLO oyuncu veri seti ve on etiket olustur"
    )
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/yolo26n.pt")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "datasets/halisaha_player"
    )
    parser.add_argument("--train-per-camera", type=int, default=40)
    parser.add_argument("--val-per-camera", type=int, default=10)
    parser.add_argument("--test-per-camera", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--crop-image-size", type=int, default=1920)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rectangle(path: Path, width: int, height: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    x1, y1, x2, y2 = map(float, data["rectangle"])
    sx = width / float(data.get("image_width", width))
    sy = height / float(data.get("image_height", height))
    values = [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]
    x1, y1, x2, y2 = values
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def yolo_detections(model, image, confidence, image_size):
    result = model.predict(
        image, classes=[0], conf=confidence, imgsz=image_size, verbose=False
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.cpu().numpy().astype(np.float32)[:, None]
    cls = result.boxes.cls.cpu().numpy().astype(np.float32)[:, None]
    return np.concatenate((xyxy, conf, cls), axis=1)


def box_iou(one, many):
    left = np.maximum(one[:2], many[:, :2])
    right = np.minimum(one[2:4], many[:, 2:4])
    wh = np.maximum(0.0, right - left)
    intersection = wh[:, 0] * wh[:, 1]
    area_one = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    area_many = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(
        0.0, many[:, 3] - many[:, 1]
    )
    return intersection / np.maximum(area_one + area_many - intersection, 1e-6)


def nms(detections, threshold):
    if len(detections) == 0:
        return detections
    order = np.argsort(-detections[:, 4])
    keep = []
    while len(order):
        current = order[0]
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(detections[current], detections[remaining]) < threshold]
    return detections[keep]


def sample_times(start, end, count):
    if count <= 0:
        return []
    step = (end - start) / count
    return [start + (index + 0.5) * step for index in range(count)]


def save_yolo_labels(path, detections, width, height):
    lines = []
    for x1, y1, x2, y2, _confidence, _class_id in detections:
        x1 = float(np.clip(x1, 0, width - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 2 or box_height < 2:
            continue
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        lines.append(
            f"0 {center_x:.6f} {center_y:.6f} {box_width / width:.6f} {box_height / height:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def prepare_directories(root, overwrite):
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite and (any(image_dir.iterdir()) or any(label_dir.iterdir())):
            raise RuntimeError(
                f"Veri seti bos degil: {root}. Yeniden olusturmak icin --overwrite kullan."
            )
        if overwrite:
            for path in image_dir.glob("*.jpg"):
                path.unlink()
            for path in label_dir.glob("*.txt"):
                path.unlink()


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    prepare_directories(args.output, args.overwrite)
    model = YOLO(str(args.model))

    # Zaman araliklari birbirinden ayridir; ardisik karelerin farkli split'lere
    # sizmasi engellenir.
    split_plan = {
        "train": (300.0, 1600.0, args.train_per_camera),
        "val": (1800.0, 2300.0, args.val_per_camera),
        "test": (2700.0, 3350.0, args.test_per_camera),
    }
    manifest_rows = []
    split_counts = defaultdict_int()

    for camera in CAMERAS:
        for path in (camera["video"], camera["roi"], camera["far_crop"]):
            if not path.exists():
                raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(camera["video"]))
        if not capture.isOpened():
            raise RuntimeError(f"Video acilamadi: {camera['video']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        polygon = load_roi(camera["roi"], width, height)
        mask = create_mask((height, width, 3), polygon)
        crop_x1, crop_y1, crop_x2, crop_y2 = load_rectangle(
            camera["far_crop"], width, height
        )

        for split, (start, end, count) in split_plan.items():
            for sample_index, time_seconds in enumerate(sample_times(start, end, count)):
                frame_number = round(time_seconds * fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Kare okunamadi: {camera['name']} frame={frame_number}"
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

                stem = f"{camera['name']}_{frame_number:06d}"
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
                        "image": str(relative_image),
                        "video": str(camera["video"].relative_to(PROJECT_ROOT)),
                        "camera": camera["name"],
                        "time_seconds": f"{time_seconds:.3f}",
                        "frame": frame_number,
                        "prelabel_count": label_count,
                        "reviewed": 0,
                    }
                )
                split_counts[split] += 1
                print(
                    f"[{split}] {camera['name']} {sample_index + 1}/{count} "
                    f"frame={frame_number} etiket={label_count}"
                )
        capture.release()

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: player",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (args.output / "classes.txt").write_text("player\n", encoding="utf-8")
    print("Veri seti hazirlandi:", args.output)
    print("Goruntu sayilari:", dict(split_counts))
    print("Manifest:", manifest_path)
    print("YOLO data.yaml:", data_yaml)


def defaultdict_int():
    return {"train": 0, "val": 0, "test": 0}


if __name__ == "__main__":
    main()
