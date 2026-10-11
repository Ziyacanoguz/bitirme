import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="İki kamera track'lerini saha konumuyla eşleştir")
    parser.add_argument("--camera1", type=Path, required=True)
    parser.add_argument("--camera2", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.14)
    parser.add_argument("--min-pair-frames", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/global_id_test/association")
    return parser.parse_args()


def load_rows(path: Path):
    frames = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = int(row.get("frame", row.get("frame_number", 0)))
            point = np.asarray([float(row["field_x"]), float(row["field_y"])], dtype=np.float64)
            if np.isfinite(point).all() and (-0.1 <= point[0] <= 1.1) and (-0.1 <= point[1] <= 1.1):
                frames[frame].append(
                    {
                        "id": int(row["track_id"]),
                        "point": point,
                        "confidence": float(row["confidence"]),
                    }
                )
    return frames


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera1 = load_rows(args.camera1)
    camera2 = load_rows(args.camera2)
    common_frames = sorted(set(camera1) & set(camera2))
    pair_observations = defaultdict(list)
    accepted_by_frame = {}

    for frame in common_frames:
        first = camera1[frame]
        second = camera2[frame]
        if not first or not second:
            continue
        first_points = np.asarray([item["point"] for item in first])
        second_points = np.asarray([item["point"] for item in second])
        costs = np.linalg.norm(first_points[:, None, :] - second_points[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(costs)
        accepted = []
        for row_index, column_index in zip(rows, columns):
            distance = float(costs[row_index, column_index])
            if distance <= args.max_distance:
                id1 = first[row_index]["id"]
                id2 = second[column_index]["id"]
                pair_observations[(id1, id2)].append(distance)
                accepted.append((id1, id2, distance, first[row_index]["point"], second[column_index]["point"]))
        accepted_by_frame[frame] = accepted

    best_for_camera1 = {}
    best_for_camera2 = {}
    for pair, distances in pair_observations.items():
        score = (len(distances), -float(np.median(distances)))
        id1, id2 = pair
        if score > best_for_camera1.get(id1, ((-1, 0), None))[0]:
            best_for_camera1[id1] = (score, id2)
        if score > best_for_camera2.get(id2, ((-1, 0), None))[0]:
            best_for_camera2[id2] = (score, id1)

    stable_pairs = []
    for id1, (score, id2) in best_for_camera1.items():
        if best_for_camera2.get(id2, (None, None))[1] != id1:
            continue
        distances = pair_observations[(id1, id2)]
        if len(distances) >= args.min_pair_frames:
            stable_pairs.append((id1, id2, len(distances), float(np.median(distances)), float(np.mean(distances))))
    stable_pairs.sort(key=lambda item: (-item[2], item[3]))

    summary_path = args.output_dir / "stable_pairs.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["global_id", "camera1_id", "camera2_id", "matched_frames", "median_distance", "mean_distance"])
        for global_id, (id1, id2, count, median, mean) in enumerate(stable_pairs):
            writer.writerow([global_id, id1, id2, count, f"{median:.6f}", f"{mean:.6f}"])

    best_frame = max(accepted_by_frame, key=lambda frame: len(accepted_by_frame[frame]), default=None)
    visual_path = args.output_dir / "best_frame_matches.jpg"
    if best_frame is not None:
        canvas = np.full((700, 1200, 3), (35, 90, 35), dtype=np.uint8)
        for index in range(11):
            x = round(index * 1199 / 10)
            cv2.line(canvas, (x, 0), (x, 699), (70, 120, 70), 1)
        for index in range(7):
            y = round(index * 699 / 6)
            cv2.line(canvas, (0, y), (1199, y), (70, 120, 70), 1)
        for id1, id2, distance, point1, point2 in accepted_by_frame[best_frame]:
            p1 = (round(point1[0] * 1199), round(point1[1] * 699))
            p2 = (round(point2[0] * 1199), round(point2[1] * 699))
            cv2.line(canvas, p1, p2, (255, 255, 255), 2)
            cv2.circle(canvas, p1, 7, (0, 255, 0), -1)
            cv2.circle(canvas, p2, 7, (255, 255, 0), -1)
            cv2.putText(canvas, f"{id1}/{id2}", (p1[0] + 8, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, f"frame {best_frame} | matches {len(accepted_by_frame[best_frame])}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(visual_path), canvas)

    all_distances = [distance for values in pair_observations.values() for distance in values]
    print(f"Ortak kare: {len(common_frames)}")
    print(f"Kabul edilen karelik eşleşme: {len(all_distances)}")
    print(f"Kararlı kamera-ID çifti: {len(stable_pairs)}")
    if all_distances:
        print(f"Medyan saha mesafesi: {np.median(all_distances):.4f}")
    print(f"Özet: {summary_path}")
    print(f"Görsel: {visual_path}")


if __name__ == "__main__":
    main()
