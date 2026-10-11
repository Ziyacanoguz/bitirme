"""Single-command player detection, tracking, Re-ID extraction and ID stitching."""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Bir kamera icin tekrar uretilebilir oyuncu takibi ve ID birlestirme")
    for name in ("video", "model", "roi", "far-crop", "calibration", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--camera", default="Kamera 1")
    p.add_argument("--start", type=float, default=1800)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--match-thresh", type=float, default=.8)
    p.add_argument("--confidence", type=float, default=.15)
    p.add_argument("--image-size", type=int, default=1920)
    p.add_argument("--crop-image-size", type=int, default=1280)
    p.add_argument("--track-thresh", type=float, default=.30)
    p.add_argument("--track-buffer", type=int, default=75)
    p.add_argument("--reid-weights", type=Path, default=ROOT / "models/osnet_x0_25_msmt17.pt")
    args = p.parse_args()
    for name in ("video", "model", "roi", "far_crop", "calibration", "reid_weights"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    if args.output_dir.exists():
        raise FileExistsError("Mevcut sonuclari korumak icin yeni output-dir secin")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if (not all(math.isfinite(v) for v in (fps, args.start, args.duration))
            or fps <= 0 or args.duration <= 0 or args.start < 0
            or round(args.start * fps) >= total_frames):
        raise ValueError("Invalid video or time interval")
    if (any(not math.isfinite(v) or not 0 < v <= 1
            for v in (args.confidence, args.track_thresh, args.match_thresh))
            or min(args.image_size, args.crop_image_size, args.track_buffer) < 1):
        raise ValueError("Invalid detector/tracker settings")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    progress = dict(status="running", camera=args.camera, commands=[],
                    settings={key: str(value.resolve()) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                    identity_status="unverified_camera_local_track_ids",
                    note="IDs remain provisional; one person can split or multiple people can mix.")

    def save():
        (out / "run.json").write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")

    def run(script, *options):
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, options)]
        progress["commands"].append(command)
        save()
        subprocess.run(command, cwd=ROOT, check=True)

    try:
        run("track_players_tiled.py", "--video", args.video, "--model", args.model,
            "--roi", args.roi, "--far-crop", args.far_crop, "--start", args.start,
            "--duration", args.duration, "--confidence", args.confidence,
            "--image-size", args.image_size, "--crop-image-size", args.crop_image_size,
            "--track-thresh", args.track_thresh, "--track-buffer", args.track_buffer,
            "--device", args.device, "--save-detections",
            "--match-thresh", args.match_thresh, "--output-dir", out / "tracks")
        track = next((out / "tracks").glob("*.csv"))
        run("project_tracks_to_field.py", "--tracks", track, "--video", args.video,
            "--calibration", args.calibration, "--output", out / "field.csv")
        run("extract_track_embeddings.py", "--tracks", out / "field.csv", "--video", args.video,
            "--weights", args.reid_weights,
            "--sample-step", 10, "--max-samples-per-id", 24, "--device", "cpu",
            "--output", out / "reid.npz")
        run("stitch_tracklets.py", "--tracks", out / "field.csv", "--reid", out / "reid.npz",
            "--output-tracks", out / "stitched.csv", "--output-reid", out / "stitched_reid.npz",
            "--mapping", out / "stitch_mapping.csv", "--max-gap", round(fps),
            "--max-position", .04, "--max-appearance", .22, "--ambiguity-margin", .15,
            "--endpoint-window", 3)
        run("audit_tracking_quality.py", "--tracks", f"raw={track}",
            "--tracks", f"stitched={out / 'stitched.csv'}", "--fps", fps,
            "--start-frame", round(args.start * fps),
            "--end-frame", min(round(args.start * fps) + round(args.duration * fps), total_frames) - 1,
            "--output-dir", out / "quality_audit")
        run("render_track_video.py", "--tracks", out / "stitched.csv", "--video", args.video,
            "--output", out / "tracking_raw.mp4", "--start", args.start,
            "--duration", args.duration, "--label", "Player tracking IDs - local / provisional")
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i",
                            str(out / "tracking_raw.mp4"), "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart",
                            str(out / "tracking_preview.mp4")], check=True)
        progress["status"] = "complete"
    except Exception as error:
        progress.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print(out)


if __name__ == "__main__":
    main()
