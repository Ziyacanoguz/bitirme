import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Yerel track ID yerine global ID videosu üret")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--camera", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    local_to_global = {}
    id_column = f"camera{args.camera}_id"
    with args.pairs.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            local_to_global[int(row[id_column])] = int(row["global_id"])

    rows_by_frame = defaultdict(list)
    with args.tracks.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = int(row.get("frame", row.get("frame_number", 0)))
            rows_by_frame[frame].append(row)
    if not rows_by_frame:
        raise RuntimeError("Track verisi boş.")

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    first_frame = min(rows_by_frame)
    last_frame = max(rows_by_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    for frame_number in range(first_frame, last_frame + 1):
        ok, frame = capture.read()
        if not ok:
            break
        for row in rows_by_frame.get(frame_number, []):
            local_id = int(row["track_id"])
            global_id = local_to_global.get(local_id)
            x1, y1, x2, y2 = [round(float(row[key])) for key in ("x1", "y1", "x2", "y2")]
            if global_id is None:
                color = (100, 100, 100)
                label = f"L{local_id}"
            else:
                color = ((37 * global_id) % 205 + 50, (17 * global_id) % 205 + 50, (29 * global_id) % 205 + 50)
                label = f"G{global_id} | L{local_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"Kamera {args.camera} | Global ID", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        writer.write(frame)
    capture.release()
    writer.release()
    print(f"Global ID videosu: {args.output}")


if __name__ == "__main__":
    main()
