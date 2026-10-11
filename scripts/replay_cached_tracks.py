"""Run ByteTrack on identical cached detections, without another detector pass."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from boxmot.trackers.bbox.bytetrack import ByteTrack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--match-thresh", type=float, default=.8)
    p.add_argument("--track-thresh", type=float, default=.3)
    p.add_argument("--track-buffer", type=int, default=75)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads(args.metadata.read_text())
    data = np.load(args.metadata.with_suffix(".detections.npz"), allow_pickle=False)
    detections, offsets = data["detections"], data["offsets"]
    if len(offsets) != meta["processed_frames"] + 1 or offsets[-1] != len(detections):
        raise ValueError("Cache length does not match metadata")
    tracker = ByteTrack(min_conf=meta["confidence"], track_thresh=args.track_thresh,
                        match_thresh=args.match_thresh, track_buffer=args.track_buffer,
                        frame_rate=round(meta["fps"]), per_class=False)
    # ByteTrack has no appearance encoder; image pixels do not participate in matching.
    frame = np.zeros((meta["height"], meta["width"], 3), dtype=np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identities, count = set(), 0
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "track_id", "confidence", "x1", "y1", "x2", "y2"])
        for i in range(len(offsets) - 1):
            tracks = tracker.update(detections[offsets[i]:offsets[i + 1]].copy(), frame)
            index = meta["start_frame"] + i
            for track in tracks:
                x1, y1, x2, y2, identity, confidence = track[:6]
                identities.add(int(identity))
                count += 1
                writer.writerow([index, f"{index / meta['fps']:.3f}", int(identity), f"{confidence:.5f}",
                                 *[f"{v:.2f}" for v in (x1, y1, x2, y2)]])
    print(f"Rows={count}, unique IDs={len(identities)}, output={args.output}")


if __name__ == "__main__":
    main()
