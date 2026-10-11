"""2D Bird's-Eye Tactical Minimap Radar for Halı Saha Video Analysis.

Draws a 2D soccer pitch diagram and plots real-time player positions (field_x, field_y)
with trajectory trail lines and G1..G14 labels.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def draw_pitch_diagram(
    width: int = 800,
    height: int = 500,
    margin: int = 40,
    bg_color: tuple[int, int, int] = (34, 112, 58),
    line_color: tuple[int, int, int] = (240, 240, 240),
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw a clean, standard 2D soccer pitch diagram."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Pitch boundary
    px1, py1 = margin, margin
    px2, py2 = width - margin, height - margin
    pw, ph = px2 - px1, py2 - py1

    # Outer border
    cv2.rectangle(img, (px1, py1), (px2, py2), line_color, line_thickness)

    # Halfway line
    mid_x = px1 + pw // 2
    cv2.line(img, (mid_x, py1), (mid_x, py2), line_color, line_thickness)

    # Center circle
    center_pt = (mid_x, py1 + ph // 2)
    radius = int(ph * 0.18)
    cv2.circle(img, center_pt, radius, line_color, line_thickness)
    cv2.circle(img, center_pt, 4, line_color, -1)

    # Goal boxes (Left and Right)
    box_h = int(ph * 0.5)
    box_w = int(pw * 0.16)
    box_y1 = py1 + (ph - box_h) // 2
    box_y2 = box_y1 + box_h

    # Left Penalty Area
    cv2.rectangle(img, (px1, box_y1), (px1 + box_w, box_y2), line_color, line_thickness)
    # Right Penalty Area
    cv2.rectangle(img, (px2 - box_w, box_y1), (px2, box_y2), line_color, line_thickness)

    # Small Goal Areas
    s_box_h = int(ph * 0.25)
    s_box_w = int(pw * 0.06)
    s_box_y1 = py1 + (ph - s_box_h) // 2
    s_box_y2 = s_box_y1 + s_box_h

    cv2.rectangle(img, (px1, s_box_y1), (px1 + s_box_w, s_box_y2), line_color, line_thickness)
    cv2.rectangle(img, (px2 - s_box_w, s_box_y1), (px2, s_box_y2), line_color, line_thickness)

    # Corner arcs
    arc_r = 12
    cv2.ellipse(img, (px1, py1), (arc_r, arc_r), 0, 0, 90, line_color, line_thickness)
    cv2.ellipse(img, (px1, py2), (arc_r, arc_r), 0, 270, 360, line_color, line_thickness)
    cv2.ellipse(img, (px2, py1), (arc_r, arc_r), 0, 90, 180, line_color, line_thickness)
    cv2.ellipse(img, (px2, py2), (arc_r, arc_r), 0, 180, 270, line_color, line_thickness)

    return img


def get_distinct_colors(count: int = 20) -> list[tuple[int, int, int]]:
    """Generate distinct BGR colors for player markers."""
    colors = []
    for i in range(count):
        hue = int(180 * i / count)
        hsv = np.uint8([[[hue, 220, 240]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


DEFAULT_COLORS = get_distinct_colors(25)


def render_tactical_radar_frame(
    pitch_img: np.ndarray,
    frame_rows: list[dict],
    trajectory_history: dict[int, list[tuple[int, int]]],
    width: int = 800,
    height: int = 500,
    margin: int = 40,
    trail_length: int = 8,
) -> np.ndarray:
    """Render active player positions and motion trails on the pitch diagram."""
    canvas = pitch_img.copy()

    px1, py1 = margin, margin
    pw, ph = width - 2 * margin, height - 2 * margin

    # Draw player positions and trails
    for row in frame_rows:
        gid_val = row.get("global_id")
        if gid_val is None or gid_val == "":
            continue
        try:
            gid = int(float(gid_val))
            fx = float(row["field_x"])
            fy = float(row["field_y"])
        except (ValueError, TypeError, KeyError):
            continue

        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            continue

        cx = int(px1 + fx * pw)
        cy = int(py1 + fy * ph)

        # Update trajectory history
        if gid not in trajectory_history:
            trajectory_history[gid] = []
        trajectory_history[gid].append((cx, cy))
        if len(trajectory_history[gid]) > trail_length:
            trajectory_history[gid].pop(0)

        color = DEFAULT_COLORS[gid % len(DEFAULT_COLORS)]

        # Draw motion trail (fading dots)
        trail = trajectory_history[gid]
        for idx, (tx, ty) in enumerate(trail[:-1]):
            alpha = (idx + 1) / len(trail)
            r = max(2, int(6 * alpha))
            cv2.circle(canvas, (tx, ty), r, color, -1)

        # Draw main player circle
        cv2.circle(canvas, (cx, cy), 13, (20, 20, 20), -1)  # Outer shadow
        cv2.circle(canvas, (cx, cy), 11, color, -1)  # Player color
        cv2.circle(canvas, (cx, cy), 11, (255, 255, 255), 1)  # White border

        # Draw label (G1, G2, etc.)
        label = f"G{gid}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tw, th = text_size

        # Background box for text
        bx1, by1 = cx - tw // 2 - 2, cy - 16 - th
        bx2, by2 = cx + tw // 2 + 2, cy - 14
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (10, 10, 10), -1)
        cv2.putText(
            canvas,
            label,
            (cx - tw // 2, cy - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def render_tactical_radar_video(
    fused_csv_path: Path,
    output_mp4_path: Path,
    duration: float = 20.0,
    fps: float = 25.0,
    width: int = 800,
    height: int = 500,
):
    """Render a standalone 2D Tactical Minimap Radar video from fused track rows."""
    fused_csv_path = Path(fused_csv_path).resolve()
    output_mp4_path = Path(output_mp4_path).resolve()
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    with fused_csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows_by_frame = defaultdict(list)
    for row in rows:
        try:
            rows_by_frame[int(row["frame"])].append(row)
        except (ValueError, KeyError):
            pass

    frames = sorted(rows_by_frame.keys())
    if not frames:
        raise ValueError(f"No valid frame rows in {fused_csv_path}")

    start_frame = frames[0]
    total_frames = min(len(frames), int(duration * fps))

    pitch_bg = draw_pitch_diagram(width=width, height=height)
    trajectory_history: dict[int, list[tuple[int, int]]] = defaultdict(list)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (width, height))

    for idx in range(total_frames):
        frame_idx = start_frame + idx
        frame_rows = rows_by_frame.get(frame_idx, [])
        radar_frame = render_tactical_radar_frame(
            pitch_bg, frame_rows, trajectory_history, width=width, height=height
        )

        # Draw frame header title
        time_sec = idx / fps
        header_text = f"2D Taktik Radar | Kare: {idx+1}/{total_frames} ({time_sec:.1f}s)"
        cv2.putText(
            radar_frame,
            header_text,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(radar_frame)

    writer.release()
    print(f"[Radar] 2D Taktik Radar videosu yazıldı: {output_mp4_path}")
