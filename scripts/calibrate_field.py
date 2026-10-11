import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.calibration import compute_homography, save_calibration  # noqa: E402


POINT_LABELS = [
    "SOL KALE - UST KENAR",
    "SAG KALE - UST KENAR",
    "SAG KALE - ALT KENAR",
    "SOL KALE - ALT KENAR",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Kamerayı ortak saha düzlemine kalibre et")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time", type=float, default=1800.0)
    return parser.parse_args()


def main():
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(args.time * fps))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Kalibrasyon karesi okunamadı.")

    height, width = frame.shape[:2]
    points = []
    window = "Saha Kalibrasyonu"

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        for index, point in enumerate(points):
            cv2.circle(canvas, point, 8, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(index + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        if len(points) > 1:
            cv2.polylines(canvas, [np.asarray(points)], False, (0, 255, 255), 2)
        instruction = (
            f"{len(points)+1}/4: {POINT_LABELS[len(points)]}"
            if len(points) < 4
            else "ENTER: kaydet | U: geri al | R: sifirla"
        )
        cv2.rectangle(canvas, (0, 0), (width, 55), (0, 0, 0), -1)
        cv2.putText(canvas, instruction, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("u"), 8) and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, ord("s")) and len(points) == 4:
            break
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    image_points = np.asarray(points, dtype=np.float32)
    homography = compute_homography(image_points)
    save_calibration(args.output, str(args.video), width, height, image_points, homography)
    print(f"Kalibrasyon kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
