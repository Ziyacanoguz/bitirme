#!/usr/bin/env python3
"""Run dual-camera tracking and Global ID pipeline using champion YOLO11s model."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/ziyacan/Desktop/Hali_Saha_Project/project_v3")
PY_BOXMOT = ROOT / ".venv-boxmot/bin/python"
BOXMOT_DIR = Path("/Users/ziyacan/Desktop/Hali_Saha_Project/boxmot-master")

MODEL = ROOT / "outputs/training/yolo11s_merged_v3_1280/weights/best.pt"
EXP_DIR = ROOT / "outputs/experiments/yolo11s_global_id_20261008/13m_40s"

V1 = ROOT / "assets/videos/matches/match_03/camera1.mp4"
V2 = ROOT / "assets/videos/matches/match_03/camera2.mp4"
C1 = ROOT / "configs/calibration_camera1_lines_v1.json"
C2 = ROOT / "configs/calibration_camera2_lines_v1.json"
ROI1 = ROOT / "configs/roi_camera1.json"
ROI2 = ROOT / "configs/roi_camera2.json"
CROP1 = ROOT / "configs/far_crop_camera1.json"
CROP2 = ROOT / "configs/far_crop_camera2.json"

START_SEC = 780.0
START_FRAME = 19500
DURATION_SEC = 40.0


def run_cmd(cmd: list):
    env = os.environ.copy()
    if BOXMOT_DIR.exists():
        env["PYTHONPATH"] = f"{BOXMOT_DIR}:{env.get('PYTHONPATH', '')}"
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True, env=env)


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    cam1_dir = EXP_DIR / "camera1"
    cam2_dir = EXP_DIR / "camera2"
    gid_out = EXP_DIR / "global_id"

    # Step 1: Camera 1 Track
    t1_csvs = list(cam1_dir.glob("*.csv"))
    if not t1_csvs:
        print("=== STEP 1: TRACKING CAMERA 1 WITH YOLO11s ===")
        run_cmd([
            PY_BOXMOT, ROOT / "scripts/track_players_tiled.py",
            "--video", V1, "--model", MODEL,
            "--roi", ROI1, "--far-crop", CROP1,
            "--start", START_SEC, "--duration", DURATION_SEC,
            "--confidence", 0.15, "--image-size", 1920, "--crop-image-size", 1280,
            "--track-thresh", 0.30, "--match-thresh", 0.65, "--track-buffer", 75,
            "--device", "mps", "--save-detections", "--output-dir", cam1_dir,
        ])
        t1_csvs = list(cam1_dir.glob("*.csv"))
    t1_csv = t1_csvs[0]
    print("Camera 1 tracking CSV:", t1_csv)

    # Step 2: Camera 2 Track
    t2_csvs = list(cam2_dir.glob("*.csv"))
    if not t2_csvs:
        print("=== STEP 2: TRACKING CAMERA 2 WITH YOLO11s ===")
        run_cmd([
            PY_BOXMOT, ROOT / "scripts/track_players_tiled.py",
            "--video", V2, "--model", MODEL,
            "--roi", ROI2, "--far-crop", CROP2,
            "--start", START_SEC, "--duration", DURATION_SEC,
            "--confidence", 0.15, "--image-size", 1920, "--crop-image-size", 1280,
            "--track-thresh", 0.30, "--match-thresh", 0.65, "--track-buffer", 75,
            "--device", "mps", "--save-detections", "--output-dir", cam2_dir,
        ])
        t2_csvs = list(cam2_dir.glob("*.csv"))
    t2_csv = t2_csvs[0]
    print("Camera 2 tracking CSV:", t2_csv)

    # Step 3: Run Global ID Fusion
    if not (gid_out / "pipeline.json").exists():
        import shutil
        if gid_out.exists():
            shutil.rmtree(gid_out)
        print("=== STEP 3: RUNNING GLOBAL ID PIPELINE ===")
        run_cmd([
            PY_BOXMOT, ROOT / "scripts/run_global_id_pipeline.py",
            "--video1", V1, "--video2", V2,
            "--tracks1", t1_csv, "--tracks2", t2_csv,
            "--calibration1", C1, "--calibration2", C2,
            "--output-dir", gid_out,
            "--offset-frames", 0,
            "--sync-method", "assumed_offset",
            "--sync-note", "Match 03 13m_40s offset 0 with YOLO11s",
            "--block-frames", 50, "--max-gap", 8, "--sample-step", 5,
            "--max-samples-per-id", 4, "--reid-device", "cpu",
            "--min-support", 8, "--min-fraction", 0.45, "--max-distance", 0.10,
        ])

    # Step 4: Consolidate Global IDs
    cons = gid_out / "global_ids_consolidated"
    if not (cons / "consolidation_info.json").exists():
        print("=== STEP 4: CONSOLIDATING GLOBAL IDS ===")
        run_cmd([
            PY_BOXMOT, ROOT / "scripts/consolidate_global_ids.py",
            "--pipeline-dir", gid_out,
            "--target-players", "auto",
            "--max-appearance-distance", 0.45,
            "--max-overlap-tolerance", 35,
        ])

    # Step 5: Render Multicamera Preview Video
    final_mp4 = EXP_DIR / "global_id_11s_40s.mp4"
    raw_mp4 = EXP_DIR / "global_id_11s_40s_raw.mp4"
    if not final_mp4.exists():
        print("=== STEP 5: RENDERING VIDEO ===")
        if not raw_mp4.exists():
            run_cmd([
                PY_BOXMOT, ROOT / "scripts/render_multicamera_preview.py",
                "--video1", V1, "--video2", V2,
                "--tracks1", gid_out / "camera1_segments.csv",
                "--tracks2", gid_out / "camera2_segments.csv",
                "--fusion-dir", cons,
                "--output", raw_mp4,
                "--duration", DURATION_SEC,
                "--start-frame", START_FRAME,
                "--panel-width", 960,
            ])
        run_cmd([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", raw_mp4, "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", 20, "-pix_fmt", "yuv420p", "-movflags", "+faststart", final_mp4
        ])
        raw_mp4.unlink(missing_ok=True)
        raw_mp4.with_suffix(".json").unlink(missing_ok=True)

    print("=== PIPELINE COMPLETE ===")
    print("Output video:", final_mp4)


if __name__ == "__main__":
    main()
