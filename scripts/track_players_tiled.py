import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from boxmot.trackers.bbox.bytetrack import ByteTrack
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.detection_fusion import fuse_full_and_crop_detections  # noqa: E402
from halisaha.roi import apply_mask, create_mask, load_roi  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ana kare + büyütülmüş uzak saha crop'u ile YOLO ve ByteTrack"
    )
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "assets/videos/camera1.mp4")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/yolo26n.pt")
    parser.add_argument("--roi", type=Path, default=PROJECT_ROOT / "configs/roi_camera1.json")
    parser.add_argument(
        "--far-crop",
        type=Path,
        default=PROJECT_ROOT / "configs/far_crop_camera1.json",
    )
    parser.add_argument("--start", type=float, default=1800.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--crop-image-size", type=int, default=1280)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.30,
        help="ByteTrack'in yeni bir izi etkinleştirme güven eşiği",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.80,
        help="ByteTrack association eşleştirme eşiği",
    )
    parser.add_argument("--track-buffer", type=int, default=75)
    parser.add_argument("--device", choices=("cpu", "mps"), default=None)
    parser.add_argument("--save-detections", action="store_true",
                        help="Tracker ayarlarini YOLO'yu tekrar calistirmadan denemek icin NPZ kaydet")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/tracks_tiled"
    )
    return parser.parse_args()


def load_rectangle(path: Path, width: int, height: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    x1, y1, x2, y2 = map(float, data["rectangle"])
    sx = width / float(data.get("image_width", width))
    sy = height / float(data.get("image_height", height))
    values = [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]
    x1, y1, x2, y2 = values
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def yolo_detections(model, image, confidence, image_size, device=None):
    result = model.predict(
        image, classes=[0], conf=confidence, imgsz=image_size, verbose=False, device=device
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.cpu().numpy().astype(np.float32)[:, None]
    cls = result.boxes.cls.cpu().numpy().astype(np.float32)[:, None]
    return np.concatenate((xyxy, conf, cls), axis=1)


def main():
    args = parse_args()
    for path in (args.video, args.model, args.roi, args.far_crop):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0 or args.start < 0 or args.duration <= 0:
        raise ValueError("Gecersiz video zaman bilgisi veya baslangic/sure")
    start_frame = min(round(args.start * fps), total - 1)
    end_frame = min(start_frame + round(args.duration * fps), total)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    polygon = load_roi(args.roi, width, height)
    mask = create_mask((height, width, 3), polygon)
    crop_x1, crop_y1, crop_x2, crop_y2 = load_rectangle(args.far_crop, width, height)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("Uzak alan kirpmasi bos")

    model = YOLO(str(args.model))
    tracker = ByteTrack(
        min_conf=args.confidence,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        frame_rate=round(fps),
        per_class=False,
    )

    stem = (
        f"{args.video.stem}_tiled_{args.image_size}_{args.confidence:g}"
        f"_track{args.track_thresh:g}"
    )
    video_path = args.output_dir / f"{stem}.mp4"
    csv_path = args.output_dir / f"{stem}.csv"
    if any(path.exists() for path in (video_path, csv_path, csv_path.with_suffix(".json"))):
        capture.release()
        raise FileExistsError("Mevcut deneyin uzerine yazilmaz; yeni --output-dir secin")
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Video yazici acilamadi: {video_path}")
    cached_detections, offsets = [], [0]
    unique_ids = set()
    processed = 0
    full_detection_count = 0
    crop_detection_count = 0
    merged_detection_count = 0
    started = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.writer(stream)
        csv_writer.writerow(
            ["frame", "time", "track_id", "confidence", "x1", "y1", "x2", "y2"]
        )
        while start_frame + processed < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            masked = apply_mask(frame, mask)
            full_dets = yolo_detections(model, masked, args.confidence, args.image_size, args.device)
            crop = masked[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_dets = yolo_detections(
                model, crop, args.confidence, args.crop_image_size, args.device
            )
            crop_dets, merged = fuse_full_and_crop_detections(
                full_dets,
                crop_dets,
                crop_x=crop_x1,
                crop_y=crop_y1,
                iou_threshold=args.nms_iou,
            )
            if args.save_detections:
                cached_detections.append(merged.copy())
                offsets.append(offsets[-1] + len(merged))
            tracks = tracker.update(merged, frame)

            full_detection_count += len(full_dets)
            crop_detection_count += len(crop_dets)
            merged_detection_count += len(merged)
            annotated = frame.copy()
            cv2.polylines(annotated, [polygon], True, (0, 255, 255), 2)
            cv2.rectangle(
                annotated, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 255, 0), 2
            )
            for track in tracks:
                x1, y1, x2, y2, identity, confidence = track[:6]
                identity = int(identity)
                unique_ids.add(identity)
                color = (
                    (37 * identity) % 205 + 50,
                    (17 * identity) % 205 + 50,
                    (29 * identity) % 205 + 50,
                )
                cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
                cv2.putText(
                    annotated,
                    f"ID {identity} | {confidence:.2f}",
                    (round(x1), max(20, round(y1) - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                csv_writer.writerow(
                    [
                        start_frame + processed,
                        f"{(start_frame + processed) / fps:.3f}",
                        identity,
                        f"{confidence:.5f}",
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                    ]
                )
            cv2.putText(
                annotated,
                f"TILED ByteTrack | full {len(full_dets)} + far {len(crop_dets)} -> {len(merged)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            writer.write(annotated)
            processed += 1
            if processed % 250 == 0:
                print(f"Kare: {processed}/{end_frame - start_frame} | "
                      f"{processed / (time.perf_counter() - started):.1f} FPS", flush=True)
            if args.show:
                cv2.imshow("Tiled tracking", cv2.resize(annotated, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    elapsed = time.perf_counter() - started
    capture.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    metadata = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    metadata.update(fps=fps, width=width, height=height, start_frame=start_frame,
                    processed_frames=processed, elapsed_seconds=elapsed,
                    output_csv=str(csv_path.resolve()), unique_track_ids=len(unique_ids),
                    detections_after_nms=merged_detection_count)
    csv_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.save_detections:
        flat = np.concatenate(cached_detections) if cached_detections else np.empty((0, 6), np.float32)
        np.savez_compressed(csv_path.with_suffix(".detections.npz"), detections=flat,
                            offsets=np.asarray(offsets, dtype=np.int64))
    print(f"İşlenen kare: {processed}")
    print(f"Ana görüntü tespitleri: {full_detection_count}")
    print(f"Uzak crop tespitleri: {crop_detection_count}")
    print(f"NMS sonrası tespitler: {merged_detection_count}")
    print(f"Benzersiz track ID: {len(unique_ids)}")
    print(f"İşleme hızı: {processed / elapsed:.2f} FPS")
    print(f"Video: {video_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
