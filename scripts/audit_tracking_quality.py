#!/usr/bin/env python3
"""Write a ground-truth-free diagnostic bundle for one or more tracking CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from halisaha.tracking_audit import AuditConfig, audit_tracking_csv, public_summary


DETAIL_TABLES = {
    "per_frame": [
        "frame", "active_track_count", "count_change", "mean_confidence",
        "low_confidence_count", "high_iou_pair_count",
    ],
    "per_track": [
        "track_id", "start_frame", "end_frame", "observed_frames", "observed_seconds",
        "lifespan_frames", "presence_coverage_rate", "is_short_track", "starts_inside_window",
        "ends_inside_window", "gap_episode_count", "missing_frames_inside_span", "max_gap_frames",
        "jump_event_count", "max_step_box_heights_per_frame", "mean_confidence", "min_confidence",
        "low_confidence_observations", "median_box_height_px",
    ],
    "potential_fragment_links": [
        "old_track_id", "new_track_id", "old_end_frame", "new_start_frame", "elapsed_frames",
        "distance_box_heights_per_frame", "box_height_ratio", "endpoint_iou",
    ],
    "high_iou_pair_events": ["frame", "track_id_a", "track_id_b", "iou"],
}


def _parse_track_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, raw_path = value.split("=", 1)
    else:
        raw_path = value
        name = Path(raw_path).stem
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    if not name:
        raise argparse.ArgumentTypeError("track name cannot be empty")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"tracking CSV not found: {path}")
    return name, path


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentage(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _markdown_report(audits: dict[str, dict], created_at: str) -> str:
    lines = [
        "# Takip Kalitesi Tanısal Denetimi",
        "",
        f"Oluşturulma zamanı (UTC): `{created_at}`",
        "",
        "> Bu rapor gerçek kimlik etiketi kullanmaz. Aşağıdaki değerler doğruluk, IDF1, HOTA, "
        "MOTA, precision veya recall değildir; yalnızca inceleme önceliği oluşturan risk göstergeleridir.",
        "",
        "| Kayıt | Kare kapsamı | Ort. aktif iz | Benzersiz ID | Kısa ID | Boşluklu ID | Sıçramalı ID | Olası parçalanma bağı | Yüksek IoU olayı |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, audit in audits.items():
        coverage = audit["frame_range"]["csv_frame_coverage_rate"]
        counts = audit["counts"]
        risks = audit["fragmentation_risk_indicators"]
        overlap = audit["overlap_and_duplicate_risk"]
        lines.append(
            f"| {name} | {_percentage(coverage)} | {counts['active_tracks_per_frame']['mean']:.2f} | "
            f"{counts['unique_track_ids']} | {risks['short_track_count']} | "
            f"{risks['tracks_with_internal_gaps']} | {risks['tracks_with_jump_events']} | "
            f"{risks['potential_fragment_link_count']} | {overlap['different_id_high_iou_pair_events']} |"
        )
    lines.extend(
        [
            "",
            "## Nasıl yorumlanmalı?",
            "",
            "- **Kısa ID:** Belirlenen süreden az gözlenen bir izdir. Kısa süreli gerçek giriş/çıkış da olabilir.",
            "- **Boşluk:** Aynı ID'nin CSV'de bir veya daha fazla kare kaybolup tekrar görünmesidir.",
            "- **Sıçrama:** Ayak noktası ardışık karede, ortalama kutu yüksekliğine göre olağandışı uzaklığa gider.",
            "- **Olası parçalanma bağı:** Bir ID bittikten kısa süre sonra yakın konumda başka ID başlar. Otomatik birleştirme yapılmaz.",
            "- **Yüksek IoU:** Farklı ID kutuları büyük ölçüde üst üste gelir. Bu çift kutu olabileceği gibi gerçek oyuncu örtüşmesi de olabilir.",
            "- **Kare kapsamı:** CSV'de en az bir iz olan kare oranıdır; sahadaki bütün oyuncuların bulunduğunu göstermez.",
            "",
            "Her kayıt için ayrıntılı `per_frame`, `per_track`, `potential_fragment_links` ve "
            "`high_iou_pair_events` CSV dosyaları aynı klasörde bulunur.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks", action="append", required=True, type=_parse_track_spec,
        metavar="NAME=CSV", help="Tekrarlanabilir; NAME verilmezse dosya adı kullanılır.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--short-track-seconds", type=float, default=1.0)
    parser.add_argument("--low-confidence", type=float, default=0.25)
    parser.add_argument("--jump-threshold-box-heights", type=float, default=0.75)
    parser.add_argument("--duplicate-iou", type=float, default=0.70)
    parser.add_argument("--fragment-max-gap", type=int, default=5)
    parser.add_argument("--fragment-max-distance-per-frame", type=float, default=0.75)
    parser.add_argument("--fragment-max-height-ratio", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    names = [name for name, _ in args.tracks]
    if len(set(names)) != len(names):
        raise ValueError("tracking input names must be unique")

    config = AuditConfig(
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        short_track_seconds=args.short_track_seconds,
        low_confidence_threshold=args.low_confidence,
        jump_threshold_box_heights=args.jump_threshold_box_heights,
        duplicate_iou_threshold=args.duplicate_iou,
        fragment_max_frame_gap=args.fragment_max_gap,
        fragment_max_distance_per_frame=args.fragment_max_distance_per_frame,
        fragment_max_height_ratio=args.fragment_max_height_ratio,
    )
    config.validate()
    audits = {name: audit_tracking_csv(path, config) for name, path in args.tracks}

    args.output_dir.mkdir(parents=True)
    for name, audit in audits.items():
        for table, fields in DETAIL_TABLES.items():
            _write_csv(args.output_dir / f"{name}_{table}.csv", audit[table], fields)

    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "report_kind": "tracking_quality_diagnostics_without_ground_truth",
        "created_at_utc": created_at,
        "ground_truth_used": False,
        "accuracy_claim": False,
        "datasets": audits,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"created_at_utc": created_at, "datasets": {name: public_summary(value) for name, value in audits.items()}},
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(_markdown_report(audits, created_at), encoding="utf-8")
    print(json.dumps({name: public_summary(value) for name, value in audits.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
