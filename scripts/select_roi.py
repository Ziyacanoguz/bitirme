import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.roi import save_roi  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Video üzerinde saha ROI poligonu seç")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time", type=float, default=300.0, help="Referans saniyesi")
    parser.add_argument("--display-width", type=int, default=1280)
    return parser.parse_args()


def main():
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")

    capture.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("ROI için referans kare okunamadı.")

    height, width = frame.shape[:2]
    scale = min(1.0, args.display_width / width)
    display_size = (int(width * scale), int(height * scale))
    base = cv2.resize(frame, display_size)
    points = []
    window = "Saha ROI Secimi"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("Saha sınırlarını sırayla sol tıklayarak seç.")
    print("ENTER/S = Kaydet | U = Son noktayı geri al | R = Sıfırla | Q/ESC = İptal")

    while True:
        preview = base.copy()
        if points:
            for point in points:
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
            for first, second in zip(points, points[1:]):
                cv2.line(preview, first, second, (0, 255, 255), 2)
            if len(points) >= 3:
                cv2.line(preview, points[-1], points[0], (0, 255, 0), 2)

        cv2.putText(
            preview,
            f"Nokta: {len(points)} | ENTER/S kaydet | U geri | R sifirla | Q iptal",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("s")):
            if len(points) < 3:
                print("ROI için en az 3 nokta gerekli.")
                continue
            original_points = [
                (round(x / scale), round(y / scale)) for x, y in points
            ]
            save_roi(args.output, original_points, width, height, str(args.video))
            print(f"ROI kaydedildi: {args.output}")
            break
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (ord("q"), 27):
            print("ROI seçimi iptal edildi.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
