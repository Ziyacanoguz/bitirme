"""Reproducible two-camera field projection, segmentation, Re-ID and Global-ID fusion."""
import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from halisaha.pipeline_validation import (  # noqa: E402
    aligned_frame_count,
    assert_tracking_integrity,
    inspect_tracking_csv,
    public_track_info,
)


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Video could not be opened: {path}")
    result = {
        "path": str(path.resolve()),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if (not math.isfinite(result["fps"]) or result["fps"] <= 0
            or result["width"] <= 0 or result["height"] <= 0 or result["frames"] <= 0):
        raise ValueError(f"Invalid video metadata: {path}")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Iki kameranin hazir takip CSV'lerinden tekrar uretilebilir Global ID verisi uret"
    )
    for name in ("video1", "video2", "tracks1", "tracks2", "calibration1", "calibration2"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reid-weights", type=Path, default=ROOT / "models/osnet_x0_25_msmt17.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset-frames", type=int, default=0,
                        help="C1 f karesi C2 f+offset ile eslesir")
    parser.add_argument(
        "--sync-method",
        required=True,
        choices=("hardware", "metadata", "manual_event", "audio", "motion", "assumed_offset"),
        help="Ofsetin yontemi; dogrulanmamis on calismalar icin assumed_offset kullanin",
    )
    parser.add_argument(
        "--sync-note",
        default="",
        help="Senkronizasyon kaniti/notu; pipeline goruntu icerigini kendi basina dogrulamaz",
    )
    parser.add_argument(
        "--min-frame-overlap", type=float, default=.90,
        help="Secilen ofsette gerekli en dusuk takip-kare numarasi ortusme orani",
    )
    parser.add_argument("--block-frames", type=int, default=50)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--max-samples-per-id", type=int, default=4)
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-fraction", type=float, default=.45)
    parser.add_argument("--max-distance", type=float, default=.10)
    parser.add_argument("--dry-run", action="store_true",
                        help="Girdileri ve komutlari dogrula; model calistirma veya dosya yazma")
    args = parser.parse_args(argv)
    args.sync_note = args.sync_note.strip()
    if args.sync_method in {"manual_event", "assumed_offset"} and not args.sync_note:
        parser.error(f"--sync-note is required with --sync-method {args.sync_method}")
    return args


def commands(args, fps):
    out = args.output_dir.resolve()
    py = sys.executable
    script = ROOT / "scripts"
    stages = [
        ("audit_local_tracks", [py, script / "audit_tracking_quality.py",
                               "--tracks", f"camera1={args.tracks1}",
                               "--tracks", f"camera2={args.tracks2}",
                               "--fps", fps, "--output-dir", out / "quality_audit"]),
        ("project_camera1", [py, script / "project_tracks_to_field.py", "--tracks", args.tracks1,
                             "--calibration", args.calibration1, "--video", args.video1,
                             "--output", out / "camera1_field.csv"]),
        ("project_camera2", [py, script / "project_tracks_to_field.py", "--tracks", args.tracks2,
                             "--calibration", args.calibration2, "--video", args.video2,
                             "--output", out / "camera2_field.csv"]),
        ("segment_camera1", [py, script / "segment_camera_tracks.py", "--tracks", out / "camera1_field.csv",
                             "--output", out / "camera1_segments.csv", "--block-frames", args.block_frames,
                             "--max-gap", args.max_gap]),
        ("segment_camera2", [py, script / "segment_camera_tracks.py", "--tracks", out / "camera2_field.csv",
                             "--output", out / "camera2_segments.csv", "--block-frames", args.block_frames,
                             "--max-gap", args.max_gap]),
    ]
    for camera in (1, 2):
        stages.append((f"reid_camera{camera}", [
            py, script / "extract_track_embeddings.py", "--tracks", out / f"camera{camera}_segments.csv",
            "--video", getattr(args, f"video{camera}"), "--weights", args.reid_weights,
            "--sample-step", args.sample_step, "--max-samples-per-id", args.max_samples_per_id,
            "--device", args.reid_device, "--output", out / f"camera{camera}_segment_reid.npz",
        ]))
    stages.append(("fuse_global_ids", [
        py, script / "fuse_camera_tracks.py",
        "--camera1", out / "camera1_segments.csv", "--camera2", out / "camera2_segments.csv",
        "--reid1", out / "camera1_segment_reid.npz", "--reid2", out / "camera2_segment_reid.npz",
        "--output-dir", out / "global_ids", "--offset-frames", args.offset_frames,
        "--fps", fps, "--min-support", args.min_support, "--min-fraction", args.min_fraction,
        "--max-distance", args.max_distance,
    ]))
    return [(name, [str(value) for value in command]) for name, command in stages]


def main():
    args = parse_args()
    path_names = ("video1", "video2", "tracks1", "tracks2", "calibration1",
                  "calibration2", "reid_weights")
    for name in path_names:
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    numeric = [args.block_frames, args.max_gap, args.sample_step, args.max_samples_per_id,
               args.min_support, args.min_fraction, args.max_distance, args.min_frame_overlap]
    if (any(not math.isfinite(float(value)) or value <= 0 for value in numeric)
            or args.min_fraction > 1 or args.min_frame_overlap > 1):
        raise ValueError(
            "Pipeline thresholds must be finite and positive; fractions must be <= 1"
        )
    videos = [video_info(path) for path in (args.video1, args.video2)]
    if abs(videos[0]["fps"] - videos[1]["fps"]) > .01:
        raise ValueError("Camera FPS differs; resample explicitly before Global-ID fusion")
    tracks = [inspect_tracking_csv(path) for path in (args.tracks1, args.tracks2)]
    for track in tracks:
        assert_tracking_integrity(track)
    for camera, (track, video) in enumerate(zip(tracks, videos), 1):
        if track["frame_max"] >= video["frames"]:
            raise ValueError(f"Camera {camera} tracking frame exceeds source video")
    frame_overlap = aligned_frame_count(tracks[0], tracks[1], args.offset_frames)
    frame_overlap_fraction = frame_overlap / min(info["frame_count"] for info in tracks)
    if frame_overlap_fraction < args.min_frame_overlap:
        raise ValueError(
            "Tracking CSV frame-number overlap is below --min-frame-overlap at the selected offset"
        )
    plan = commands(args, videos[0]["fps"])
    manifest = {
        "schema": "halisaha_global_id_pipeline_v1",
        "status": "dry_run" if args.dry_run else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity_status": "provisional_not_human_verified",
        "units": "normalized_not_meters",
        "videos": videos,
        "tracks": [public_track_info(info) for info in tracks],
        "synchronization": {
            "method": args.sync_method,
            "note": args.sync_note,
            "offset_frames": args.offset_frames,
            "content_verified_by_pipeline": False,
            "warning": "Frame-number overlap is not proof of visual/audio synchronization.",
        },
        "frame_number_overlap_count": frame_overlap,
        "frame_number_overlap_fraction": frame_overlap_fraction,
        "settings": {key: value for key, value in vars(args).items()
                     if not isinstance(value, Path) and key != "dry_run"},
        "stages": [{"name": name, "command": command, "status": "pending"}
                   for name, command in plan],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "pipeline.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    save()
    started = time.perf_counter()
    active_stage = None
    try:
        for stage, (_name, command) in zip(manifest["stages"], plan):
            active_stage = stage
            stage["status"] = "running"
            stage_started = time.perf_counter()
            save()
            subprocess.run(command, cwd=ROOT, check=True)
            stage.update(status="complete", elapsed_seconds=time.perf_counter() - stage_started)
            active_stage = None
            save()
        manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                        output_global_ids=str((args.output_dir / "global_ids/fused.csv").resolve()))
    except Exception as error:
        if active_stage is not None:
            active_stage.update(
                status="failed",
                elapsed_seconds=time.perf_counter() - stage_started,
                error=f"{type(error).__name__}: {error}",
            )
        manifest.update(status="failed", elapsed_seconds=time.perf_counter() - started,
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(manifest["output_global_ids"])


if __name__ == "__main__":
    main()
