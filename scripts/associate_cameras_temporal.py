import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Konum + hareket + hafıza ile iki kamera global ID eşleştirme")
    parser.add_argument("--camera1", type=Path, required=True)
    parser.add_argument("--camera2", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.18)
    parser.add_argument("--velocity-scale", type=float, default=0.025)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--history-bonus", type=float, default=0.45)
    parser.add_argument("--history-ttl", type=int, default=20)
    parser.add_argument("--history-min-hits", type=int, default=5)
    parser.add_argument("--min-pair-frames", type=int, default=5)
    parser.add_argument("--reid1", type=Path, default=None)
    parser.add_argument("--reid2", type=Path, default=None)
    parser.add_argument("--reid-weight", type=float, default=0.30)
    parser.add_argument("--appearance-gate", type=float, default=0.45)
    parser.add_argument("--edge-y-threshold", type=float, default=0.80)
    parser.add_argument("--edge-max-distance", type=float, default=0.30)
    parser.add_argument("--edge-reid-weight", type=float, default=2.0)
    parser.add_argument("--edge-rescue-appearance", type=float, default=0.18)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/global_id_test/association_temporal")
    return parser.parse_args()


def load_rows(path: Path):
    frames = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = int(row.get("frame", row.get("frame_number", 0)))
            point = np.asarray([float(row["field_x"]), float(row["field_y"])], dtype=np.float64)
            if np.isfinite(point).all() and (-0.12 <= point[0] <= 1.12) and (-0.12 <= point[1] <= 1.12):
                frames[frame].append(
                    {"id": int(row["track_id"]), "point": point, "confidence": float(row["confidence"])}
                )
    return frames


def load_embeddings(path):
    if path is None:
        return {}
    data = np.load(path)
    return {int(identity): embedding for identity, embedding in zip(data["ids"], data["embeddings"])}


def add_velocities(frames):
    previous = {}
    smoothed = {}
    for frame in sorted(frames):
        for item in frames[frame]:
            identity = item["id"]
            velocity = np.zeros(2, dtype=np.float64)
            if identity in previous:
                old_frame, old_point = previous[identity]
                gap = frame - old_frame
                if 0 < gap <= 8:
                    raw = (item["point"] - old_point) / gap
                    velocity = 0.65 * smoothed.get(identity, raw) + 0.35 * raw
            smoothed[identity] = velocity
            previous[identity] = (frame, item["point"])
            item["velocity"] = velocity


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera1 = load_rows(args.camera1)
    camera2 = load_rows(args.camera2)
    embeddings1 = load_embeddings(args.reid1)
    embeddings2 = load_embeddings(args.reid2)
    add_velocities(camera1)
    add_velocities(camera2)
    common_frames = sorted(set(camera1) & set(camera2))

    pair_observations = defaultdict(list)
    matches_by_frame = {}
    active_for_1 = {}
    active_for_2 = {}
    pair_streak = defaultdict(int)
    pair_last_frame = {}

    for frame in common_frames:
        first = camera1[frame]
        second = camera2[frame]
        if not first or not second:
            continue
        p1 = np.asarray([item["point"] for item in first])
        p2 = np.asarray([item["point"] for item in second])
        v1 = np.asarray([item["velocity"] for item in first])
        v2 = np.asarray([item["velocity"] for item in second])
        position = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
        velocity = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=2)
        distance_gate = np.full_like(position, args.max_distance)
        appearance = np.full_like(position, 0.5)
        appearance_available = np.zeros_like(position, dtype=bool)
        if embeddings1 and embeddings2:
            for row_index, item1 in enumerate(first):
                feature1 = embeddings1.get(item1["id"])
                if feature1 is None:
                    continue
                for column_index, item2 in enumerate(second):
                    feature2 = embeddings2.get(item2["id"])
                    if feature2 is not None:
                        appearance[row_index, column_index] = 1.0 - float(np.clip(np.dot(feature1, feature2), -1.0, 1.0))
                        appearance_available[row_index, column_index] = True

        edge_region = (
            (p1[:, None, 1] >= args.edge_y_threshold)
            & (p2[None, :, 1] >= args.edge_y_threshold)
        )
        appearance_rescue = (
            edge_region
            & appearance_available
            & (appearance <= args.edge_rescue_appearance)
        )
        distance_gate[appearance_rescue] = args.edge_max_distance
        appearance_weight = np.full_like(position, args.reid_weight)
        appearance_weight[edge_region & appearance_available] = args.edge_reid_weight
        cost = position / distance_gate + args.velocity_weight * np.minimum(
            velocity / args.velocity_scale, 2.0
        )
        cost += appearance_weight * appearance * appearance_available

        for row_index, item1 in enumerate(first):
            prior = active_for_1.get(item1["id"])
            if prior is None or frame - prior[1] > args.history_ttl:
                continue
            for column_index, item2 in enumerate(second):
                pair = (item1["id"], item2["id"])
                if item2["id"] == prior[0] and pair_streak[pair] >= args.history_min_hits:
                    cost[row_index, column_index] -= args.history_bonus

        invalid = position > distance_gate
        invalid |= appearance_available & (appearance > args.appearance_gate)
        cost[invalid] = 1e6
        rows, columns = linear_sum_assignment(cost)
        accepted = []
        for row_index, column_index in zip(rows, columns):
            if invalid[row_index, column_index]:
                continue
            item1 = first[row_index]
            item2 = second[column_index]
            id1, id2 = item1["id"], item2["id"]
            pos_distance = float(position[row_index, column_index])
            vel_distance = float(velocity[row_index, column_index])
            appearance_distance = float(appearance[row_index, column_index])
            pair_observations[(id1, id2)].append((frame, pos_distance, vel_distance, appearance_distance))
            pair = (id1, id2)
            if pair_last_frame.get(pair) == frame - 1:
                pair_streak[pair] += 1
            else:
                pair_streak[pair] = 1
            pair_last_frame[pair] = frame
            active_for_1[id1] = (id2, frame)
            active_for_2[id2] = (id1, frame)
            accepted.append((id1, id2, pos_distance, vel_distance, appearance_distance, item1["point"], item2["point"]))
        matches_by_frame[frame] = accepted

    def pair_rank(observations):
        positions = [item[1] for item in observations]
        velocities = [item[2] for item in observations]
        appearances = [item[3] for item in observations]
        return (len(observations), -float(np.median(positions)), -float(np.median(velocities)), -float(np.median(appearances)))

    best1 = {}
    best2 = {}
    for pair, observations in pair_observations.items():
        id1, id2 = pair
        rank = pair_rank(observations)
        if rank > best1.get(id1, ((-1, 0, 0, 0), None))[0]:
            best1[id1] = (rank, id2)
        if rank > best2.get(id2, ((-1, 0, 0, 0), None))[0]:
            best2[id2] = (rank, id1)

    stable = []
    for id1, (_rank, id2) in best1.items():
        if best2.get(id2, (None, None))[1] != id1:
            continue
        observations = pair_observations[(id1, id2)]
        if len(observations) < args.min_pair_frames:
            continue
        position_values = [item[1] for item in observations]
        velocity_values = [item[2] for item in observations]
        appearance_values = [item[3] for item in observations]
        stable.append(
            (
                id1,
                id2,
                len(observations),
                float(np.median(position_values)),
                float(np.mean(position_values)),
                float(np.median(velocity_values)),
                float(np.median(appearance_values)),
            )
        )
    stable.sort(key=lambda item: (-item[2], item[3]))
    global_for_pair = {(id1, id2): index for index, (id1, id2, *_rest) in enumerate(stable)}

    summary_path = args.output_dir / "stable_pairs_temporal.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["global_id", "camera1_id", "camera2_id", "matched_frames", "median_position", "mean_position", "median_velocity", "appearance_distance"])
        for global_id, values in enumerate(stable):
            writer.writerow([global_id, values[0], values[1], values[2], f"{values[3]:.6f}", f"{values[4]:.6f}", f"{values[5]:.6f}", f"{values[6]:.6f}"])

    frame_path = args.output_dir / "frame_matches_temporal.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "global_id", "camera1_id", "camera2_id", "position_distance", "velocity_distance", "appearance_distance"])
        for frame, matches in sorted(matches_by_frame.items()):
            for id1, id2, position_distance, velocity_distance, appearance_distance, *_points in matches:
                writer.writerow([frame, global_for_pair.get((id1, id2), ""), id1, id2, f"{position_distance:.6f}", f"{velocity_distance:.6f}", f"{appearance_distance:.6f}"])

    best_frame = max(matches_by_frame, key=lambda key: len(matches_by_frame[key]), default=None)
    visual_path = args.output_dir / "best_frame_temporal.jpg"
    if best_frame is not None:
        canvas = np.full((700, 1200, 3), (35, 90, 35), dtype=np.uint8)
        for index in range(11):
            cv2.line(canvas, (round(index * 1199 / 10), 0), (round(index * 1199 / 10), 699), (70, 120, 70), 1)
        for index in range(7):
            cv2.line(canvas, (0, round(index * 699 / 6)), (1199, round(index * 699 / 6)), (70, 120, 70), 1)
        for id1, id2, pos_distance, _vel_distance, _appearance_distance, point1, point2 in matches_by_frame[best_frame]:
            one = (round(point1[0] * 1199), round(point1[1] * 699))
            two = (round(point2[0] * 1199), round(point2[1] * 699))
            cv2.line(canvas, one, two, (255, 255, 255), 2)
            cv2.circle(canvas, one, 7, (0, 255, 0), -1)
            cv2.circle(canvas, two, 7, (255, 255, 0), -1)
            global_id = global_for_pair.get((id1, id2))
            label = f"G{global_id} {id1}/{id2}" if global_id is not None else f"{id1}/{id2}"
            cv2.putText(canvas, label, (one[0] + 8, one[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, f"frame {best_frame} | temporal matches {len(matches_by_frame[best_frame])}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(visual_path), canvas)

    accepted_count = sum(len(values) for values in matches_by_frame.values())
    print(f"Ortak kare: {len(common_frames)}")
    print(f"Kabul edilen karelik eşleşme: {accepted_count}")
    print(f"Kararlı global çift: {len(stable)}")
    print(f"Özet: {summary_path}")
    print(f"Kare eşleşmeleri: {frame_path}")
    print(f"Görsel: {visual_path}")


if __name__ == "__main__":
    main()
