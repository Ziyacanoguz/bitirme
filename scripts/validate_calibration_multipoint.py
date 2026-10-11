import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.calibration import apply_tps_transform, load_multipoint_calibration  # noqa: E402
from halisaha.roi import apply_mask, create_mask, load_roi  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="İki kameranın 7 noktalı TPS kalibrasyonunu doğrula")
    parser.add_argument("--time", type=float, default=1800.0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reports/calibration_7pt")
    return parser.parse_args()


def read_frame(path: Path, timestamp: float):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Kare okunamadı: {path}")
    return frame


def warp_tps(frame, calibration_path: Path, roi_path: Path, output_size):
    height, width = frame.shape[:2]
    polygon = load_roi(roi_path, width, height)
    masked = apply_mask(frame, create_mask(frame.shape, polygon))
    calibration = load_multipoint_calibration(calibration_path)
    output_width, output_height = output_size
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, output_width),
        np.linspace(0.0, 1.0, output_height),
    )
    field_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    image_normalized = apply_tps_transform(
        field_points,
        calibration["field_control_points"],
        calibration["inverse_weights"],
        calibration["inverse_affine"],
    )
    map_x = (image_normalized[:, 0] * width).reshape(output_height, output_width).astype(np.float32)
    map_y = (image_normalized[:, 1] * height).reshape(output_height, output_width).astype(np.float32)
    return cv2.remap(masked, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def draw_grid(image, label):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for index in range(11):
        x = round(index * (width - 1) / 10)
        color = (0, 255, 255) if index in (0, 5, 10) else (70, 70, 70)
        cv2.line(canvas, (x, 0), (x, height - 1), color, 1)
    for index in range(7):
        y = round(index * (height - 1) / 6)
        color = (0, 255, 255) if index in (0, 3, 6) else (70, 70, 70)
        cv2.line(canvas, (0, y), (width - 1, y), color, 1)
    cv2.rectangle(canvas, (0, 0), (320, 45), (0, 0, 0), -1)
    cv2.putText(canvas, label, (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video1 = PROJECT_ROOT / "assets/videos/camera1.mp4"
    video2 = PROJECT_ROOT / "assets/videos/camera2.mp4"
    frame1 = read_frame(video1, args.time)
    frame2 = read_frame(video2, args.time)
    size = (args.width, args.height)
    field1 = warp_tps(
        frame1,
        PROJECT_ROOT / "configs/calibration_camera1_7pt.json",
        PROJECT_ROOT / "configs/roi_camera1.json",
        size,
    )
    field2 = warp_tps(
        frame2,
        PROJECT_ROOT / "configs/calibration_camera2_7pt.json",
        PROJECT_ROOT / "configs/roi_camera2.json",
        size,
    )
    blend = cv2.addWeighted(field1, 0.5, field2, 0.5, 0)
    difference = cv2.absdiff(field1, field2)
    outputs = {
        "camera1_topdown_7pt.jpg": draw_grid(field1, "Kamera 1 - TPS"),
        "camera2_topdown_7pt.jpg": draw_grid(field2, "Kamera 2 - TPS"),
        "overlay_topdown_7pt.jpg": draw_grid(blend, "Kamera 1 + Kamera 2 - TPS"),
        "difference_topdown_7pt.jpg": draw_grid(difference, "Mutlak fark - TPS"),
    }
    for name, image in outputs.items():
        path = args.output_dir / name
        cv2.imwrite(str(path), image)
        print(path)


if __name__ == "__main__":
    main()
