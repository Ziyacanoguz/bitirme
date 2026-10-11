"""Input validation shared by reproducible tracking and Global-ID pipelines."""
import csv
import math
from pathlib import Path


TRACK_COLUMNS = {"frame", "track_id", "confidence", "x1", "y1", "x2", "y2"}


def inspect_tracking_csv(path):
    """Return auditable structural facts without claiming tracking accuracy."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = TRACK_COLUMNS - fields
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        frames, identities, seen = set(), set(), set()
        duplicates = malformed = invalid_boxes = 0
        rows = 0
        for line, row in enumerate(reader, 2):
            rows += 1
            try:
                frame = int(row["frame"])
                identity = int(row["track_id"])
                confidence = float(row["confidence"])
                x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            except (TypeError, ValueError):
                malformed += 1
                continue
            numeric = (confidence, x1, y1, x2, y2)
            if (frame < 0 or identity < 0 or not all(math.isfinite(value) for value in numeric)
                    or not 0 <= confidence <= 1):
                malformed += 1
                continue
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
            key = (frame, identity)
            duplicates += int(key in seen)
            seen.add(key)
            frames.add(frame)
            identities.add(identity)
    if not rows:
        raise ValueError(f"{path}: tracking CSV is empty")
    if malformed:
        raise ValueError(f"{path}: {malformed} malformed tracking rows")
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "frames": frames,
        "frame_min": min(frames),
        "frame_max": max(frames),
        "frame_count": len(frames),
        "local_id_count": len(identities),
        "duplicate_frame_id_rows": duplicates,
        "invalid_box_rows": invalid_boxes,
    }


def aligned_frame_count(first, second, offset_frames=0):
    """Count frame-number overlap for C1 f and C2 f+offset.

    This is a structural coverage check, not visual/audio synchronization proof.
    The legacy function name is kept for callers that already import it.
    """
    return len(first["frames"] & {frame - int(offset_frames) for frame in second["frames"]})


def assert_tracking_integrity(info):
    """Reject ambiguous rows before segmentation and identity association."""
    problems = []
    if info["duplicate_frame_id_rows"]:
        problems.append(f"{info['duplicate_frame_id_rows']} duplicate (frame, track_id) rows")
    if info["invalid_box_rows"]:
        problems.append(f"{info['invalid_box_rows']} invalid boxes")
    if problems:
        raise ValueError(f"{info['path']}: " + "; ".join(problems))


def public_track_info(info):
    """Remove in-memory frame sets before JSON serialization."""
    return {key: value for key, value in info.items() if key != "frames"}
