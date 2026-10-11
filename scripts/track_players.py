import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.roi import apply_mask, create_mask, load_roi  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO + ByteTrack/BoT-SORT ile oyuncu takibi"
    )
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
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort"),
        default="bytetrack",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=None,
        help="Özel tracker YAML dosyası",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="select_roi.py ile oluşturulan ROI JSON dosyası",
    )
    parser.add_argument("--start", type=float, default=300.0, help="Başlangıç saniyesi")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="İşlenecek süre (saniye)"
    )
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--show", action="store_true", help="İşlerken görüntüyü göster")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/tracks",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.video.exists():
        raise FileNotFoundError(f"Video bulunamadı: {args.video}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model bulunamadı: {args.model}")
    if args.start < 0:
        raise ValueError("Başlangıç zamanı negatif olamaz.")
    if args.duration <= 0:
        raise ValueError("Süre sıfırdan büyük olmalıdır.")
    if args.roi is not None and not args.roi.exists():
        raise FileNotFoundError(f"ROI bulunamadı: {args.roi}")
    if args.tracker_config is not None and not args.tracker_config.exists():
        raise FileNotFoundError(f"Tracker ayarı bulunamadı: {args.tracker_config}")


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = min(int(round(args.start * fps)), max(total_frames - 1, 0))
    requested_frames = int(round(args.duration * fps))
    end_frame = min(start_frame + requested_frames, total_frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    camera_name = args.video.stem
    start_label = f"{args.start:g}s".replace(".", "p")
    duration_label = f"{args.duration:g}s".replace(".", "p")
    stem = f"{camera_name}_{args.tracker}_{start_label}_{duration_label}"
    video_output = args.output_dir / f"{stem}.mp4"
    csv_output = args.output_dir / f"{stem}.csv"

    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Çıktı videosu açılamadı: {video_output}")

    model = YOLO(str(args.model))
    tracker_config = (
        str(args.tracker_config)
        if args.tracker_config is not None
        else f"{args.tracker}.yaml"
    )
    roi_polygon = None
    roi_mask = None
    if args.roi is not None:
        roi_polygon = load_roi(args.roi, width, height)
        roi_mask = create_mask((height, width, 3), roi_polygon)
    processed = 0
    track_ids = set()
    started_at = time.perf_counter()

    with csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "camera_id",
                "frame_number",
                "timestamp_seconds",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "foot_x",
                "foot_y",
            ]
        )

        while start_frame + processed < end_frame:
            ok, frame = capture.read()
            if not ok:
                break

            frame_number = start_frame + processed
            timestamp = frame_number / fps
            inference_frame = apply_mask(frame, roi_mask) if roi_mask is not None else frame
            result = model.track(
                inference_frame,
                persist=True,
                tracker=tracker_config,
                classes=[0],
                conf=args.confidence,
                imgsz=args.image_size,
                verbose=False,
            )[0]

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                identities = boxes.id.int().cpu().tolist()

                for identity, confidence, box in zip(
                    identities, confidences, coordinates
                ):
                    x1, y1, x2, y2 = map(float, box)
                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2
                    track_ids.add(identity)
                    csv_writer.writerow(
                        [
                            camera_name,
                            frame_number,
                            f"{timestamp:.3f}",
                            identity,
                            f"{float(confidence):.5f}",
                            f"{x1:.2f}",
                            f"{y1:.2f}",
                            f"{x2:.2f}",
                            f"{y2:.2f}",
                            f"{foot_x:.2f}",
                            f"{foot_y:.2f}",
                        ]
                    )

            annotated = frame.copy()
            if roi_polygon is not None:
                cv2.polylines(annotated, [roi_polygon], True, (0, 255, 255), 2)
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                identities = boxes.id.int().cpu().tolist()
                for identity, confidence, box in zip(identities, confidences, coordinates):
                    x1, y1, x2, y2 = map(int, box)
                    color = (
                        (37 * identity) % 205 + 50,
                        (17 * identity) % 205 + 50,
                        (29 * identity) % 205 + 50,
                    )
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID {identity} | {float(confidence):.2f}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
            cv2.putText(
                annotated,
                f"{args.tracker.upper()} | frame {frame_number} | {timestamp:.2f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(annotated)
            processed += 1

            if args.show:
                preview = cv2.resize(annotated, (1280, 720))
                cv2.imshow("Hali Saha Object Tracking", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    elapsed = time.perf_counter() - started_at
    processing_fps = processed / elapsed if elapsed else 0.0

    capture.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Tracker: {args.tracker}")
    print(f"İşlenen kare: {processed}")
    print(f"Üretilen farklı track ID: {len(track_ids)}")
    print(f"İşleme hızı: {processing_fps:.2f} FPS")
    print(f"Video çıktısı: {video_output}")
    print(f"Track verisi: {csv_output}")


if __name__ == "__main__":
    main()
