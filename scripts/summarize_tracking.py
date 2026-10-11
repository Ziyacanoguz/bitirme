"""Diagnostic tracking counts only; these are not ground-truth ID accuracy metrics."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def summarize(path, fps=25, start=45000, frames=1500):
    by_id, per_frame = defaultdict(list), Counter()
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index, identity = int(row["frame"]), int(row["track_id"])
            by_id[identity].append(index)
            per_frame[index] += 1
    counts = np.array([per_frame[i] for i in range(start, start + frames)])
    return dict(rows=int(counts.sum()), mean_tracks=float(counts.mean()), min_tracks=int(counts.min()),
                max_tracks=int(counts.max()), p95_tracks=float(np.percentile(counts, 95)),
                unique_ids=len(by_id), ids_under_1s=sum(len(v) < fps for v in by_id.values()),
                ids_observed_10s=sum(len(v) >= 10 * fps for v in by_id.values()),
                zero_track_frames=int((counts == 0).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", action="append", required=True, help="name=csv_path")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=45000)
    p.add_argument("--frames", type=int, default=1500)
    p.add_argument("--fps", type=float, default=25)
    args = p.parse_args()
    result = {}
    for spec in args.tracks:
        name, path = spec.split("=", 1)
        result[name] = summarize(path, args.fps, args.start_frame, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
