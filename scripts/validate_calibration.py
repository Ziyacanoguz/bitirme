import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.calibration import load_calibration  # noqa: E402
from halisaha.roi import apply_mask, create_mask, load_roi  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="İki kamera saha kalibrasyonunu görsel olarak doğrula")
    parser.add_argument("--camera1", type=Path, default=PROJECT_ROOT / "assets/videos/camera1.mp4")
    parser.add_argument("--camera2", type=Path, default=PROJECT_ROOT / "assets/videos/camera2.mp4")
    parser.add_argument("--calibration1", type=Path, default=PROJECT_ROOT / "configs/calibration_camera1.json")
    parser.add_argument("--calibration2", type=Path, default=PROJECT_ROOT / "configs/calibration_camera2.json")
    parser.add_argument("--roi1", type=Path, default=PROJECT_ROOT / "configs/roi_camera1.json")
    parser.add_argument("--roi2", type=Path, default=PROJECT_ROOT / "configs/roi_camera2.json")
    parser.add_argument("--time", type=float, default=1800.0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reports/calibration")
    return parser.parse_args()


def read_frame(path: Path, timestamp: float):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Kare okunamadı: {path}")
    return frame


def warp_field(frame, calibration_path, roi_path, output_size):
    height, width = frame.shape[:2]
    homography = load_calibration(calibration_path, width, height)
    polygon = load_roi(roi_path, width, height)
    masked = apply_mask(frame, create_mask(frame.shape, polygon))
    output_width, output_height = output_size
    scale = np.asarray(
        [[output_width - 1, 0, 0], [0, output_height - 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    pixel_homography = scale @ homography
    warped = cv2.warpPerspective(masked, pixel_homography, output_size)
    return warped


def draw_grid(image, label):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for index in range(11):
        x = round(index * (width - 1) / 10)
        color = (0, 255, 255) if index in (0, 5, 10) else (80, 80, 80)
        cv2.line(canvas, (x, 0), (x, height - 1), color, 1)
    for index in range(7):
        y = round(index * (height - 1) / 6)
        color = (0, 255, 255) if index in (0, 3, 6) else (80, 80, 80)
        cv2.line(canvas, (0, y), (width - 1, y), color, 1)
    cv2.rectangle(canvas, (0, 0), (270, 45), (0, 0, 0), -1)
    cv2.putText(canvas, label, (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame1 = read_frame(args.camera1, args.time)
    frame2 = read_frame(args.camera2, args.time)
    size = (args.width, args.height)
    field1 = warp_field(frame1, args.calibration1, args.roi1, size)
    field2 = warp_field(frame2, args.calibration2, args.roi2, size)
    blend = cv2.addWeighted(field1, 0.5, field2, 0.5, 0)
    difference = cv2.absdiff(field1, field2)

    outputs = {
        "camera1_topdown.jpg": draw_grid(field1, "Kamera 1 - ortak saha"),
        "camera2_topdown.jpg": draw_grid(field2, "Kamera 2 - ortak saha"),
        "overlay_topdown.jpg": draw_grid(blend, "Kamera 1 + Kamera 2"),
        "difference_topdown.jpg": draw_grid(difference, "Mutlak fark"),
    }
    for name, image in outputs.items():
        path = args.output_dir / name
        cv2.imwrite(str(path), image)
        print(path)


if __name__ == "__main__":
    main()
