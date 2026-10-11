import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from boxmot.reid.core import ReID


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Track kutularından OSNet görünüş vektörü çıkar")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models/osnet_x0_25_msmt17.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--min-height", type=float, default=32)
    parser.add_argument("--max-samples-per-id", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    rows_by_frame = defaultdict(list)
    with args.tracks.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = int(row.get("frame", row.get("frame_number", 0)))
            rows_by_frame[frame].append(row)
    selected_frames = sorted(rows_by_frame)[:: max(args.sample_step, 1)]
    candidates = defaultdict(list)
    for frame_number in selected_frames:
        for row in rows_by_frame[frame_number]:
            width = float(row["x2"]) - float(row["x1"])
            height = float(row["y2"]) - float(row["y1"])
            confidence = float(row["confidence"])
            if confidence < args.min_confidence or height < args.min_height or width < 8:
                continue
            quality = confidence * np.sqrt(width * height)
            candidates[int(row["track_id"])].append((frame_number, row, quality))
    selected_rows = defaultdict(list)
    for items in candidates.values():
        # Sample throughout each track, selecting a clear crop from each time slice.
        if args.max_samples_per_id > 0 and len(items) > args.max_samples_per_id:
            groups = np.array_split(np.arange(len(items)), args.max_samples_per_id)
            items = [max((items[int(i)] for i in group), key=lambda item: item[2]) for group in groups]
        for frame_number, row, _quality in items:
            selected_rows[frame_number].append(row)

    capture = cv2.VideoCapture(str(args.video))
    reid = ReID(weights=args.weights, device=args.device, half=False).model
    features_by_id = defaultdict(list)

    for frame_number in sorted(selected_rows):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        rows = selected_rows[frame_number]
        boxes = np.asarray(
            [[float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])] for row in rows],
            dtype=np.float32,
        )
        embeddings = reid.get_features(boxes, frame)
        for row, embedding in zip(rows, embeddings):
            features_by_id[int(row["track_id"])].append(np.asarray(embedding, dtype=np.float32))
    capture.release()

    identities = sorted(features_by_id)
    means = []
    counts = []
    for identity in identities:
        values = np.asarray(features_by_id[identity], dtype=np.float32)
        mean = values.mean(axis=0)
        norm = np.linalg.norm(mean)
        means.append(mean / norm if norm else mean)
        counts.append(len(values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(identities, dtype=np.int32),
        embeddings=np.asarray(means, dtype=np.float32),
        counts=np.asarray(counts, dtype=np.int32),
    )
    print(f"Track embedding sayısı: {len(identities)}")
    print(f"Çıktı: {args.output}")


if __name__ == "__main__":
    main()
