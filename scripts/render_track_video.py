"""Render a local/stitched track CSV with a visible identity disclaimer."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", default="Camera 1 - Local track IDs")
    p.add_argument("--start", type=float, default=1800)
    p.add_argument("--duration", type=float, default=60)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    by_frame = defaultdict(list)
    with args.tracks.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])].append(row)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Video cannot be opened")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Video writer cannot be opened")
    start = round(args.start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_number in range(start, start + round(args.duration * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        for row in by_frame[frame_number]:
            identity = int(row["track_id"])
            x1, y1, x2, y2 = [round(float(row[k])) for k in ("x1", "y1", "x2", "y2")]
            color = ((37 * identity) % 205 + 50, (17 * identity) % 205 + 50, (29 * identity) % 205 + 50)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {identity}", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        cv2.rectangle(frame, (0, 0), (w, 66), (15, 23, 37), -1)
        cv2.putText(frame, args.label, (24, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2)
        cv2.putText(frame, "IDs are provisional, not verified player identities", (24, 53), cv2.FONT_HERSHEY_SIMPLEX, .52, (120, 200, 255), 1)
        writer.write(frame)
    cap.release()
    writer.release()
    print(args.output)


if __name__ == "__main__":
    main()
