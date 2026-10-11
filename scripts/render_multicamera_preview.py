"""Render two synchronized camera panels with conservative provisional ID labels."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from create_global_id_review_pack import (  # noqa: E402
    file_provenance,
    load_mapping,
    load_track_index,
    resolve_fusion_context,
)


GLOBAL_LABEL_STATUSES = {"fused", "single_view"}


def _optional_int(value, *, field):
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error


def load_fused_rows(path):
    required = {
        "status",
        "global_id",
        "source_frame_camera1",
        "source_frame_camera2",
        "camera1_id",
        "camera2_id",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Fused CSV missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    # A completed fusion run may have no calibration-valid observations.
    # In that case the preview still shows the local tracks, without G labels.
    return rows


def build_global_label_lookup(rows, mapping):
    """Map exact source observations to a safe provisional global ID or ``None``.

    A component mapping alone is deliberately insufficient.  Only a fused row
    whose status is ``fused`` or ``single_view`` may expose a G label.  All
    conflicts, invalid positions and unmatched observations remain camera-local.
    """
    result = {}
    for line, row in enumerate(rows, 2):
        status = str(row.get("status", "")).strip().lower()
        global_id = _optional_int(row.get("global_id"), field=f"global_id on line {line}")
        safe_global_id = global_id if status in GLOBAL_LABEL_STATUSES else None
        if safe_global_id is not None and safe_global_id < 1:
            raise ValueError(f"Invalid global_id on fused line {line}")
        for camera in (1, 2):
            frame = _optional_int(
                row.get(f"source_frame_camera{camera}"),
                field=f"source_frame_camera{camera} on line {line}",
            )
            segment = _optional_int(
                row.get(f"camera{camera}_id"),
                field=f"camera{camera}_id on line {line}",
            )
            if frame is None and segment is None:
                continue
            if frame is None or segment is None or frame < 0 or segment < 0:
                raise ValueError(f"Incomplete source observation on fused line {line}")
            if safe_global_id is not None and mapping.get((camera, segment)) != safe_global_id:
                raise ValueError(
                    f"Fused line {line} disagrees with mapping for camera {camera}, segment {segment}"
                )
            key = (camera, frame, segment)
            if key in result:
                raise ValueError(f"Duplicate fused source observation: {key}")
            result[key] = safe_global_id
    return result


def index_tracks_by_frame(track_index):
    frames = defaultdict(list)
    for segment, observations in track_index.items():
        for frame, row in observations.items():
            frames[frame].append((segment, row))
    return {
        frame: sorted(values, key=lambda item: item[0])
        for frame, values in sorted(frames.items())
    }


def aligned_track_frames(first, second, offset_frames):
    """Return C1-frame coordinates observed in both tracking streams."""
    return set(first) & {frame - offset_frames for frame in second}


def observation_label(camera, frame, segment, row, global_lookup):
    global_id = global_lookup.get((camera, frame, segment))
    if global_id is not None:
        return f"G{global_id}"
    local_id = row.get("local_track_id", segment)
    try:
        local_id = int(local_id)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid local track ID for camera {camera}, segment {segment}") from error
    return f"C{camera}:L{local_id}"


def color_for_label(label):
    """Stable BGR color; the same G label has the same color in both panels."""
    digest = sha256(label.encode("utf-8")).digest()
    return tuple(55 + int(value) % 176 for value in digest[:3])


def video_info(capture, path):
    if not capture.isOpened():
        raise RuntimeError(f"Video could not be opened: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or min(width, height, frames) <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return {"fps": fps, "width": width, "height": height, "frames": frames}


def rectangle_overlap_area(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _clamped_label_box(x, y, label_width, label_height, panel_width, panel_height):
    x = min(max(0, int(x)), max(0, panel_width - label_width))
    y = min(max(0, int(y)), max(0, panel_height - label_height))
    return x, y, x + label_width, y + label_height


def choose_label_box(
    detection_box,
    label_size,
    panel_size,
    occupied,
    *,
    gap=4,
):
    """Choose the nearest deterministic non-overlapping label position.

    The first candidate is immediately above the detection.  If it collides
    with the camera header or another label, positions immediately below and
    then progressively farther above/below are tried.  The returned boolean is
    true only when a leader line is useful.
    """
    x1, y1, x2, y2 = detection_box
    label_width, label_height = label_size
    panel_width, panel_height = panel_size
    horizontal = [
        x1,
        round((x1 + x2 - label_width) / 2),
        x2 - label_width,
        x1 - label_width - gap,
        x2 + gap,
    ]
    vertical = [y1 - label_height - gap, y2 + gap]
    for level in range(1, 5):
        vertical.extend(
            [
                y1 - (level + 1) * (label_height + gap),
                y2 + gap + level * (label_height + gap),
            ]
        )
    candidates = []
    for y in vertical:
        for x in horizontal:
            box = _clamped_label_box(
                x, y, label_width, label_height, panel_width, panel_height
            )
            if box not in candidates:
                candidates.append(box)
    for index, box in enumerate(candidates):
        if not any(rectangle_overlap_area(box, other) for other in occupied):
            return box, index > 0
    # Extremely crowded frame: minimize covered label area, then displacement.
    preferred = candidates[0]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            sum(rectangle_overlap_area(item[1], other) for other in occupied),
            abs(item[1][0] - preferred[0]) + abs(item[1][1] - preferred[1]),
            item[0],
        ),
    )
    index, box = ranked[0]
    return box, index > 0


def draw_readable_label(panel, label, color, detection_box, occupied):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    padding = 5
    stripe = 5
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    label_size = (
        text_width + padding * 2 + stripe,
        text_height + baseline + padding * 2,
    )
    label_box, displaced = choose_label_box(
        detection_box,
        label_size,
        (panel.shape[1], panel.shape[0]),
        occupied,
    )
    lx1, ly1, lx2, ly2 = label_box
    if displaced:
        bx1, by1, bx2, by2 = detection_box
        anchor_y = by1 if ly2 <= by1 else by2
        anchor = (round((bx1 + bx2) / 2), anchor_y)
        target = (
            min(max(anchor[0], lx1), lx2 - 1),
            ly2 if ly2 <= anchor_y else ly1,
        )
        cv2.line(panel, anchor, target, color, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (lx1, ly1), (lx2, ly2), (15, 18, 20), -1)
    cv2.rectangle(panel, (lx1, ly1), (lx1 + stripe, ly2), color, -1)
    cv2.rectangle(panel, (lx1, ly1), (lx2, ly2), (225, 225, 225), 1)
    cv2.putText(
        panel,
        label,
        (lx1 + stripe + padding, ly1 + padding + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    occupied.append(label_box)


def render_panel(frame, observations, camera, source_frame, panel_width, global_lookup):
    height, width = frame.shape[:2]
    panel_height = max(1, round(height * panel_width / width))
    panel = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    sx, sy = panel_width / width, panel_height / height
    header_height = 38
    occupied = [(0, 0, panel_width, header_height)]
    for segment, row in observations:
        label = observation_label(camera, source_frame, segment, row, global_lookup)
        color = color_for_label(label)
        x1, y1, x2, y2 = (
            float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        )
        first = (
            min(panel_width - 1, max(0, round(x1 * sx))),
            min(panel_height - 1, max(0, round(y1 * sy))),
        )
        second = (
            min(panel_width - 1, max(0, round(x2 * sx))),
            min(panel_height - 1, max(0, round(y2 * sy))),
        )
        if second[0] <= first[0] or second[1] <= first[1]:
            continue
        cv2.rectangle(panel, first, second, color, 2)
        draw_readable_label(
            panel,
            label,
            color,
            (first[0], first[1], second[0], second[1]),
            occupied,
        )
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, header_height), (10, 13, 16), -1)
    cv2.addWeighted(overlay, 0.86, panel, 0.14, 0, panel)
    cv2.putText(
        panel,
        f"Kamera {camera} | kare {source_frame}",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iki kamerayi gecici Global ID etiketleriyle yan yana goster"
    )
    for name in ("video1", "video2", "tracks1", "tracks2", "fusion-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Kamera 1 kare numarasi; varsayilan iki track akiminin ilk ortak karesi",
    )
    parser.add_argument("--panel-width", type=int, default=960)
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("video1", "video2", "tracks1", "tracks2"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.fusion_dir = args.fusion_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.fusion_dir.is_dir():
        raise NotADirectoryError(args.fusion_dir)
    run_path = args.fusion_dir / "run.json"
    mapping_path = args.fusion_dir / "mapping.csv"
    fused_path = args.fusion_dir / "fused.csv"
    for path in (run_path, mapping_path, fused_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("Output video or metadata already exists")
    if not math.isfinite(args.duration) or args.duration <= 0 or args.panel_width < 320:
        raise ValueError("Duration must be finite/positive and panel-width must be at least 320")

    fusion_run, offset_frames = resolve_fusion_context(
        run_path, (args.tracks1, args.tracks2)
    )
    mapping, mapping_identity_column = load_mapping(mapping_path)
    track_indexes = [load_track_index(path) for path in (args.tracks1, args.tracks2)]
    tracks_by_frame = [index_tracks_by_frame(index) for index in track_indexes]
    global_lookup = build_global_label_lookup(load_fused_rows(fused_path), mapping)
    common = sorted(aligned_track_frames(tracks_by_frame[0], tracks_by_frame[1], offset_frames))
    if not common:
        raise ValueError("Tracking streams have no common frame at the fusion offset")
    start_frame = common[0] if args.start_frame is None else args.start_frame
    if start_frame not in set(common):
        raise ValueError("start-frame is not present in both aligned tracking streams")

    captures = [cv2.VideoCapture(str(path)) for path in (args.video1, args.video2)]
    writer = None
    processed = 0
    try:
        videos = [video_info(capture, path) for capture, path in zip(captures, (args.video1, args.video2))]
        if abs(videos[0]["fps"] - videos[1]["fps"]) > .01:
            raise ValueError("Camera FPS differs; preview requires frame-aligned videos")
        requested = max(1, round(args.duration * videos[0]["fps"]))
        end_frame = min(
            start_frame + requested,
            common[-1] + 1,
            videos[0]["frames"],
            videos[1]["frames"] - offset_frames,
        )
        if end_frame <= start_frame or start_frame + offset_frames < 0:
            raise ValueError("Requested preview range is outside the source videos")
        panel_heights = [
            max(1, round(video["height"] * args.panel_width / video["width"]))
            for video in videos
        ]
        body_height = max(panel_heights)
        header_height = 52
        output_size = (args.panel_width * 2, body_height + header_height)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            videos[0]["fps"],
            output_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Video writer could not be opened: {args.output}")
        captures[0].set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        captures[1].set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset_frames)
        for canonical_frame in range(start_frame, end_frame):
            source_frames = (canonical_frame, canonical_frame + offset_frames)
            decoded = [capture.read() for capture in captures]
            if not all(ok for ok, _frame in decoded):
                raise RuntimeError(f"Could not decode aligned frame {canonical_frame}")
            panels = [
                render_panel(
                    decoded[camera - 1][1],
                    tracks_by_frame[camera - 1].get(source_frames[camera - 1], []),
                    camera,
                    source_frames[camera - 1],
                    args.panel_width,
                    global_lookup,
                )
                for camera in (1, 2)
            ]
            canvas = np.full(
                (body_height + header_height, args.panel_width * 2, 3),
                18,
                dtype=np.uint8,
            )
            for index, panel in enumerate(panels):
                y = header_height + (body_height - panel.shape[0]) // 2
                x = index * args.panel_width
                canvas[y:y + panel.shape[0], x:x + panel.shape[1]] = panel
            cv2.putText(
                canvas,
                "G = gecici Global ID (insan dogrulamasi gerekli) | C#:L# = kamera ici ID",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            processed += 1
    finally:
        for capture in captures:
            capture.release()
        if writer is not None:
            writer.release()

    if processed <= 0:
        raise RuntimeError("No preview frames were written")
    metadata = {
        "schema": "halisaha_multicamera_preview_v1",
        "status": "complete",
        "identity_status": "provisional_not_human_verified",
        "label_policy": (
            "G only for exact fused/single_view source-frame+segment rows; "
            "conflict/invalid/unmatched/missing observations remain camera-local"
        ),
        "offset_frames": offset_frames,
        "offset_source": "fusion_run",
        "start_frame_camera1": start_frame,
        "start_frame_camera2": start_frame + offset_frames,
        "processed_frames": processed,
        "duration_seconds": processed / videos[0]["fps"],
        "fps": videos[0]["fps"],
        "panel_width": args.panel_width,
        "mapping_identity_column": mapping_identity_column,
        "fusion_identity_status": fusion_run.get("identity_status", "unknown"),
        "provenance": {
            "fusion_run": file_provenance(run_path, digest=True),
            "mapping": file_provenance(mapping_path, digest=True),
            "fused": file_provenance(fused_path, digest=True),
            "tracks": [file_provenance(path, digest=True) for path in (args.tracks1, args.tracks2)],
            "videos": [file_provenance(path) for path in (args.video1, args.video2)],
        },
        "output": str(args.output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
