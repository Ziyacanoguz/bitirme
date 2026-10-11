"""Autonomous 2-hour pipeline:
1. Run match_03 dual-camera Global ID pipeline with 1280px High-Res model.
2. Train YOLO11s (9.4M params) at 1280px for 60 epochs on halisaha_merged_v3.
3. Compare metrics and produce comprehensive report.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/Users/ziyacan/Desktop/Hali_Saha_Project/project_v3")
PY_TRAIN = ROOT / ".venv-train/bin/python"
PY_BOXMOT = ROOT / ".venv-boxmot/bin/python"
BOXMOT_DIR = Path("/Users/ziyacan/Desktop/Hali_Saha_Project/boxmot-master")

OUT_GID = ROOT / "outputs/experiments/highres_1280_global_id_20261004"
MODEL_1280 = ROOT / "outputs/training/yolo26n_merged_v3_1280_highres/weights/best.pt"
STATUS_FILE = OUT_GID / "pipeline_status.json"

V1 = ROOT / "assets/videos/matches/match_03/camera1.mp4"
V2 = ROOT / "assets/videos/matches/match_03/camera2.mp4"
C1 = ROOT / "configs/calibration_camera1_lines_v1.json"
C2 = ROOT / "configs/calibration_camera2_lines_v1.json"
ROI1 = ROOT / "configs/roi_camera1.json"
ROI2 = ROOT / "configs/roi_camera2.json"
CROP1 = ROOT / "configs/far_crop_camera1.json"
CROP2 = ROOT / "configs/far_crop_camera2.json"

CLIPS = [
    ("13m_40s", 780.0, 19500, 40.0, "global_id_1280_40s"),
    ("30m_30s", 1800.0, 45000, 30.0, "global_id_1280_30s"),
]


def update_status(stage: str, **kwargs):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **kwargs,
    }
    STATUS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{payload['updated_at']}] STAGE: {stage} {kwargs}")


def run_cmd(cmd: list, env_extra: dict | None = None):
    env = os.environ.copy()
    if BOXMOT_DIR.exists():
        env["PYTHONPATH"] = f"{BOXMOT_DIR}:{env.get('PYTHONPATH', '')}"
    if env_extra:
        env.update(env_extra)
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True, env=env)


def run_phase1_global_id():
    print("=== STARTING PHASE 1: GLOBAL ID DUAL-CAMERA BENCHMARK ===")
    update_status("phase1_start")
    results = {}

    for clip_name, start_sec, start_frame, duration_sec, stem in CLIPS:
        clip_dir = OUT_GID / clip_name
        cam1_dir = clip_dir / "camera1"
        cam2_dir = clip_dir / "camera2"
        gid_out = clip_dir / "global_id"

        cam1_dir.mkdir(parents=True, exist_ok=True)
        cam2_dir.mkdir(parents=True, exist_ok=True)

        # 1. Track Camera 1
        t1_csvs = list(cam1_dir.glob("*.csv"))
        if not t1_csvs:
            update_status("tracking_cam1", clip=clip_name)
            run_cmd([
                PY_BOXMOT, ROOT / "scripts/track_players_tiled.py",
                "--video", V1, "--model", MODEL_1280,
                "--roi", ROI1, "--far-crop", CROP1,
                "--start", start_sec, "--duration", duration_sec,
                "--confidence", 0.15, "--image-size", 1920, "--crop-image-size", 1280,
                "--track-thresh", 0.30, "--match-thresh", 0.80, "--track-buffer", 75,
                "--device", "mps", "--save-detections", "--output-dir", cam1_dir,
            ])
            t1_csvs = list(cam1_dir.glob("*.csv"))
        t1_csv = t1_csvs[0]

        # 2. Track Camera 2
        t2_csvs = list(cam2_dir.glob("*.csv"))
        if not t2_csvs:
            update_status("tracking_cam2", clip=clip_name)
            run_cmd([
                PY_BOXMOT, ROOT / "scripts/track_players_tiled.py",
                "--video", V2, "--model", MODEL_1280,
                "--roi", ROI2, "--far-crop", CROP2,
                "--start", start_sec, "--duration", duration_sec,
                "--confidence", 0.15, "--image-size", 1920, "--crop-image-size", 1280,
                "--track-thresh", 0.30, "--match-thresh", 0.80, "--track-buffer", 75,
                "--device", "mps", "--save-detections", "--output-dir", cam2_dir,
            ])
            t2_csvs = list(cam2_dir.glob("*.csv"))
        t2_csv = t2_csvs[0]

        # 3. Run Global ID Fusion
        if not (gid_out / "pipeline.json").exists():
            import shutil
            if gid_out.exists():
                shutil.rmtree(gid_out)
            update_status("global_id_fusion", clip=clip_name)
            run_cmd([
                PY_BOXMOT, ROOT / "scripts/run_global_id_pipeline.py",
                "--video1", V1, "--video2", V2,
                "--tracks1", t1_csv, "--tracks2", t2_csv,
                "--calibration1", C1, "--calibration2", C2,
                "--output-dir", gid_out,
                "--offset-frames", 0,
                "--sync-method", "assumed_offset",
                "--sync-note", f"Match 03 {clip_name} offset 0 with 1280px model",
                "--block-frames", 50, "--max-gap", 8, "--sample-step", 5,
                "--max-samples-per-id", 4, "--reid-device", "cpu",
                "--min-support", 8, "--min-fraction", 0.45, "--max-distance", 0.10,
            ])

        # 4. Consolidate Global IDs targeting 14 players
        cons = gid_out / "global_ids_consolidated"
        if not (cons / "consolidation_info.json").exists():
            update_status("consolidating_global_ids", clip=clip_name)
            run_cmd([
                PY_BOXMOT, ROOT / "scripts/consolidate_global_ids.py",
                "--pipeline-dir", gid_out,
                "--target-players", 14,
                "--max-appearance-distance", 0.45,
                "--max-overlap-tolerance", 35,
            ])

        # 5. Render Multicamera Preview Video
        final_mp4 = clip_dir / f"{stem}.mp4"
        raw_mp4 = clip_dir / f"{stem}_raw.mp4"
        if not final_mp4.exists():
            update_status("rendering_video", clip=clip_name)
            if not raw_mp4.exists():
                run_cmd([
                    PY_BOXMOT, ROOT / "scripts/render_multicamera_preview.py",
                    "--video1", V1, "--video2", V2,
                    "--tracks1", gid_out / "camera1_segments.csv",
                    "--tracks2", gid_out / "camera2_segments.csv",
                    "--fusion-dir", cons,
                    "--output", raw_mp4,
                    "--duration", duration_sec,
                    "--start-frame", start_frame,
                    "--panel-width", 960,
                ])
            run_cmd([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", raw_mp4, "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", 20, "-pix_fmt", "yuv420p", "-movflags", "+faststart", final_mp4
            ])
            raw_mp4.unlink(missing_ok=True)
            raw_mp4.with_suffix(".json").unlink(missing_ok=True)

        info = json.loads((cons / "consolidation_info.json").read_text(encoding="utf-8"))
        fusion = json.loads((gid_out / "global_ids/run.json").read_text(encoding="utf-8"))
        results[clip_name] = {
            "video": str(final_mp4.resolve()),
            "original_global_ids": info["original_global_id_count"],
            "consolidated_global_ids": info["consolidated_global_id_count"],
            "shared_components": fusion["shared_components"],
            "total_components": fusion["total_components"],
            "fused_rows": fusion["totals"].get("fused_rows", 0),
            "single_view_rows": fusion["totals"].get("single_view_rows", 0),
            "unmatched_rows": fusion["totals"].get("unmatched_rows", 0),
            "conflict_rows": fusion["totals"].get("conflict_rows", 0),
            "disagreement_frames": fusion["totals"].get("disagreement_frames", 0),
        }

    (OUT_GID / "global_id_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate Markdown report
    lines = [
        "# 1280px High-Res Model İle İki Kameralı Global ID Test Raporu",
        "",
        "> Karşılaştırma: Dünkü Model (17 & 16 ID) vs Bugünkü 1280px High-Res Model",
        "",
    ]
    for clip, v in results.items():
        lines += [
            f"## {clip}",
            "",
            f"- **Video Çıktısı**: `{v['video']}`",
            f"- **İlk Ham Bileşen Sayısı**: {v['original_global_ids']}",
            f"- **Konsolide Edilmiş Global ID (Sahadaki Oyuncu Sayısı)**: **`{v['consolidated_global_ids']}`**",
            f"- **İki Kamerada Ortak Bileşen**: {v['shared_components']} / {v['total_components']}",
            f"- **Fused (İki Kamerada Eşleşen) Satır**: {v['fused_rows']}",
            f"- **Tek Kamera Satırı**: {v['single_view_rows']}",
            f"- **Çelişki Satırı**: {v['conflict_rows']}",
            f"- **Görüş Uyuşmazlığı Olan Kare**: {v['disagreement_frames']}",
            "",
        ]
    (OUT_GID / "GLOBAL_ID_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_status("phase1_complete", results=results)
    print("=== PHASE 1 COMPLETED SUCCESSFULLY ===")


def run_phase2_yolo11s():
    print("=== STARTING PHASE 2: YOLO11s (9.4M) ARCHITECTURE TRAINING ===")
    update_status("phase2_start")
    train_dir = ROOT / "outputs/training/yolo11s_merged_v3_1280"
    data_yaml = ROOT / "datasets/halisaha_merged_v3/data.yaml"

    cmd = [
        PY_TRAIN, ROOT / "scripts/train_player_detector.py",
        "--dataset", ROOT / "datasets/halisaha_merged_v3",
        "--data-yaml", data_yaml,
        "--model", "yolo11s.pt",
        "--epochs", 60,
        "--image-size", 1280,
        "--batch", 2,
        "--device", "mps",
        "--name", "yolo11s_merged_v3_1280",
        "--allow-unreviewed",
    ]
    run_cmd(cmd)

    # Evaluate YOLO11s vs YOLO26n 1280px
    eval_out = ROOT / "outputs/evaluation/yolo11s_vs_yolo26n_1280"
    eval_cmd = [
        PY_TRAIN, ROOT / "scripts/evaluate_detector_models.py",
        "--data", data_yaml,
        "--split", "val",
        "--image-size", 1280,
        "--model", f"yolo26n_1280px={MODEL_1280}",
        "--model", f"yolo11s_1280px={train_dir}/weights/best.pt",
        "--output-dir", eval_out,
    ]
    run_cmd(eval_cmd)
    update_status("phase2_complete")
    print("=== PHASE 2 COMPLETED SUCCESSFULLY ===")


def main():
    try:
        run_phase1_global_id()
        run_phase2_yolo11s()
        update_status("all_complete")
    except Exception as e:
        err = traceback.format_exc()
        (OUT_GID / "error.log").write_text(err, encoding="utf-8")
        update_status("failed", error=str(e))
        print("Pipeline failed:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
