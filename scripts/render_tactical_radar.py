#!/usr/bin/env python3
"""Render standalone 2D Tactical Minimap Radar video and 3-panel composite video."""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halisaha.tactical_radar import (
    draw_pitch_diagram,
    render_tactical_radar_frame,
    render_tactical_radar_video,
)
from halisaha.multicamera import frame_index


def main():
    parser = argparse.ArgumentParser(description="2D Kuş Bakışı Taktik Radar videosu ve 3 panelli bileşik video oluştur")
    parser.add_argument("--fused-csv", type=Path, required=True, help="Konsolide fused.csv dosyasının yolu")
    parser.add_argument("--video1", type=Path, help="Kamera 1 video yolu")
    parser.add_argument("--video2", type=Path, help="Kamera 2 video yolu")
    parser.add_argument("--tracks1", type=Path, help="Kamera 1 segment/track CSV")
    parser.add_argument("--tracks2", type=Path, help="Kamera 2 segment/track CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Çıktı klasörü")
    parser.add_argument("--duration", type=float, default=20.0, help="Saniye cinsinden video süresi")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_csv = args.fused_csv.resolve()
    radar_mp4 = out_dir / "tactical_radar_20s.mp4"

    print(f"[Radar] Standalone 2D Taktik Radar oluşturuluyor -> {radar_mp4}")
    render_tactical_radar_video(
        fused_csv_path=fused_csv,
        output_mp4_path=radar_mp4,
        duration=args.duration,
        fps=args.fps,
        width=800,
        height=500,
    )

    # Render composite video if videos and tracks are provided
    if args.video1 and args.video2 and args.tracks1 and args.tracks2:
        v1_path = args.video1.resolve()
        v2_path = args.video2.resolve()
        t1_path = args.tracks1.resolve()
        t2_path = args.tracks2.resolve()

        if all(p.is_file() for p in (v1_path, v2_path, t1_path, t2_path)):
            composite_mp4 = out_dir / "global_id_with_radar_20s.mp4"
            print(f"[Radar] 3 Panelli Taktik Radar Bileşik Videosu oluşturuluyor -> {composite_mp4}")

            # Read fused rows
            with fused_csv.open(newline="", encoding="utf-8") as f:
                fused_rows = list(csv.DictReader(f))

            rows_by_frame = defaultdict(list)
            for r in fused_rows:
                try:
                    rows_by_frame[int(r["frame"])].append(r)
                except (ValueError, KeyError):
                    pass

            frames = sorted(rows_by_frame.keys())
            start_frame = frames[0] if frames else 0
            total_frames = min(len(frames), int(args.duration * args.fps))

            cap1 = cv2.VideoCapture(str(v1_path))
            cap2 = cv2.VideoCapture(str(v2_path))

            cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Output canvas size: Top = Camera 1 (640x360) + Camera 2 (640x360) -> 1280x360
            # Bottom = 2D Radar (1280x420) -> Total = 1280x780
            c_width = 1280
            c_height = 780

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(composite_mp4), fourcc, args.fps, (c_width, c_height))

            pitch_bg = draw_pitch_diagram(width=c_width, height=420, margin=35)
            trajectory_history = defaultdict(list)

            for idx in range(total_frames):
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break

                # Resize camera views to 640x360
                f1_resized = cv2.resize(frame1, (640, 360))
                f2_resized = cv2.resize(frame2, (640, 360))

                # Add camera titles
                cv2.putText(f1_resized, "Kamera 1 (Sol Saha)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(f2_resized, "Kamera 2 (Sag Saha)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                top_panel = np.hstack((f1_resized, f2_resized))

                # Render 2D radar for this frame
                frame_idx = start_frame + idx
                frame_rows = rows_by_frame.get(frame_idx, [])
                radar_panel = render_tactical_radar_frame(
                    pitch_bg, frame_rows, trajectory_history, width=c_width, height=420, margin=35
                )

                # Add radar panel header title
                sec = idx / args.fps
                cv2.putText(
                    radar_panel,
                    f"2D KUŞ BAKIŞI TAKTİK RADAR (Ortak Saha Konumları G1..G14) | {sec:.1f}s ({idx+1}/{total_frames})",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                composite_frame = np.vstack((top_panel, radar_panel))
                writer.write(composite_frame)

            cap1.release()
            cap2.release()
            writer.release()
            print(f"[Radar] 3 Panelli Taktik Radar videosu hazır: {composite_mp4}")


if __name__ == "__main__":
    main()
