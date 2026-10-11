#!/usr/bin/env python3
"""Consolidate global IDs in a pipeline run folder down to target player count."""

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halisaha.identity_consolidation import consolidate_global_identities


def load_reid_npz(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        return {}
    data = np.load(path)
    result = {}
    id_key = "ids" if "ids" in data else "track_ids" if "track_ids" in data else "identities" if "identities" in data else None
    emb_key = "embeddings" if "embeddings" in data else "features" if "features" in data else None
    if id_key and emb_key:
        for tid, emb in zip(data[id_key], data[emb_key]):
            result[int(tid)] = emb
    else:
        for key in data.files:
            if key.startswith("track_") or key.isdigit():
                tid = int(key.replace("track_", ""))
                result[tid] = data[key]
    return result


def estimate_match_player_count(fused_rows: list[dict], min_players: int = 10, max_players: int = 18) -> int:
    """Estimate ground-truth player count (e.g. 12 for 6v6, 14 for 7v7) using spatial NMS on fused frames."""
    from collections import defaultdict
    by_frame = defaultdict(list)
    for r in fused_rows:
        if r.get("field_x") and r.get("field_y"):
            by_frame[int(r["frame"])].append((float(r["field_x"]), float(r["field_y"])))
    if not by_frame:
        return 14
    counts = []
    for frame, pts in by_frame.items():
        clusters = []
        for x, y in pts:
            if not any(np.hypot(c[0] - x, c[1] - y) < 0.06 for c in clusters):
                clusters.append((x, y))
        counts.append(len(clusters))
    p95 = np.percentile(counts, 95)
    # Round to nearest even number: 10, 12, 14, 16
    even_estimate = int(round(p95 / 2.0) * 2)
    return max(min_players, min(max_players, even_estimate))


def main():
    parser = argparse.ArgumentParser(description="Parçalı Global ID'leri hedef oyuncu sayısına konsolide et")
    parser.add_argument("--pipeline-dir", type=Path, required=True, help="Global ID pipeline çalışma klasörü (ör. outputs/runs/global_id_demo_20s)")
    parser.add_argument("--target-players", type=str, default="auto", help="Hedef oyuncu sayısı: 'auto' (otomatik tespit), 12, 14, 16 vb.")
    parser.add_argument("--max-appearance-distance", type=float, default=0.45, help="Maksimum Re-ID görünüş mesafesi eşiği")
    parser.add_argument("--max-overlap-tolerance", type=int, default=35, help="Takip geçişlerinde (handover) tolere edilecek azami örtüşme karesi")
    args = parser.parse_args()

    pipe_dir = args.pipeline_dir.resolve()
    global_dir = pipe_dir / "global_ids"
    if not global_dir.is_dir():
        raise NotADirectoryError(global_dir)

    fused_path = global_dir / "fused.csv"
    mapping_path = global_dir / "mapping.csv"

    # Read fused rows
    with fused_path.open(newline="", encoding="utf-8") as f:
        fused_rows = list(csv.DictReader(f))

    # Read mapping rows
    with mapping_path.open(newline="", encoding="utf-8") as f:
        mapping_rows = list(csv.DictReader(f))

    # Read Re-ID embeddings
    reid1_path = pipe_dir / "camera1_segment_reid.npz"
    reid2_path = pipe_dir / "camera2_segment_reid.npz"
    if not reid1_path.is_file():
        reid1_path = pipe_dir / "camera1_reid.npz"
    if not reid2_path.is_file():
        reid2_path = pipe_dir / "camera2_reid.npz"

    feat1 = load_reid_npz(reid1_path)
    feat2 = load_reid_npz(reid2_path)

    print(f"[Consolidate] Girdi: {len(fused_rows)} fused satırı, {len(mapping_rows)} mapping satırı")
    print(f"[Consolidate] Yüklenen Re-ID vektörleri: Kamera 1 = {len(feat1)}, Kamera 2 = {len(feat2)}")

    if str(args.target_players).strip().lower() == "auto":
        target_players = estimate_match_player_count(fused_rows)
        print(f"[Consolidate] Otomatik algılanan maç kadrosu: {target_players} oyuncu ({target_players//2}v{target_players//2})")
    else:
        target_players = int(args.target_players)
        print(f"[Consolidate] Belirtilen hedef oyuncu sayısı: {target_players}")

    c_fused, c_mapping, cmap = consolidate_global_identities(
        fused_rows,
        mapping_rows,
        feat1,
        feat2,
        target_players=target_players,
        max_appearance_distance=args.max_appearance_distance,
        max_overlap_tolerance=args.max_overlap_tolerance,
    )

    orig_gids = len(set(int(r["global_id"]) for r in fused_rows if r.get("global_id") != "" and r.get("global_id") is not None))
    distinct_fused_gids = len(set(int(r["global_id"]) for r in c_fused if r.get("global_id") != "" and r.get("global_id") is not None))
    print(f"[Consolidate] Başarılı! Orijinal Global ID sayısı: {orig_gids} -> Konsolide Edilmiş Oyuncu ID Sayısı: {distinct_fused_gids}")

    # Create consolidated directory for preview rendering
    cons_dir = pipe_dir / "global_ids_consolidated"
    cons_dir.mkdir(parents=True, exist_ok=True)

    out_fused_path = cons_dir / "fused.csv"
    out_mapping_path = cons_dir / "mapping.csv"

    if c_fused:
        with out_fused_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=c_fused[0].keys())
            writer.writeheader()
            writer.writerows(c_fused)

    if c_mapping:
        with out_mapping_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=c_mapping[0].keys())
            writer.writeheader()
            writer.writerows(c_mapping)

    # Copy run.json to consolidated directory
    orig_run_json = global_dir / "run.json"
    if orig_run_json.is_file():
        (cons_dir / "run.json").write_text(orig_run_json.read_text(encoding="utf-8"), encoding="utf-8")

    # Save consolidation metadata
    info = {
        "status": "complete",
        "original_global_id_count": orig_gids,
        "consolidated_global_id_count": distinct_fused_gids,
        "target_players": args.target_players,
        "max_appearance_distance": args.max_appearance_distance,
        "cluster_map": {str(k): v for k, v in cmap.items()},
    }
    (cons_dir / "consolidation_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[Consolidate] Kaydedildi:\n  - {out_fused_path}\n  - {out_mapping_path}")


if __name__ == "__main__":
    main()
