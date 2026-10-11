import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


# =========================
# YOLO MODEL
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Halı saha oyuncu tespiti")
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "assets/videos/camera1.mp4",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/yolo26n.pt",
    )
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=1280)
    return parser.parse_args()


args = parse_args()
model = YOLO(str(args.model))


# =========================
# VIDEO
# =========================

video = cv2.VideoCapture(str(args.video))

if not video.isOpened():
    print("Video acilmadi.")
    exit()


fps = video.get(cv2.CAP_PROP_FPS)

print(f"FPS: {fps}")
print("Player detection başladı.")
print("Q = Çık")


while True:

    ret, frame = video.read()

    if not ret:
        break


    # =========================
    # YOLO
    # =========================

    results = model.predict(
        frame,
        classes=[0],       # sadece insan
        conf=args.confidence,
        imgsz=args.image_size,
        verbose=False
    )


    # =========================
    # SADECE KUTULAR
    # =========================

    annotated_frame = results[0].plot(
        labels=False,
        conf=False,
        line_width=1
    )


    # =========================
    # EKRANA SIĞDIR
    # =========================

    annotated_frame = cv2.resize(
        annotated_frame,
        (1280, 720)
    )


    cv2.imshow(
        "Hali Saha Player Detection",
        annotated_frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


video.release()

cv2.destroyAllWindows()
