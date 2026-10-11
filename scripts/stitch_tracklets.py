import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ayni kamerada kisa sure kaybolup yeni ID alan tracklet'leri "
            "saha konumu, hareket ve Re-ID gorunusune gore birlestir"
        )
    )
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--reid", type=Path, required=True)
    parser.add_argument("--output-tracks", type=Path, required=True)
    parser.add_argument("--output-reid", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=75)
    parser.add_argument("--max-position", type=float, default=0.11)
    parser.add_argument("--max-appearance", type=float, default=0.38)
    parser.add_argument("--appearance-weight", type=float, default=0.65)
    parser.add_argument("--gap-weight", type=float, default=0.15)
    parser.add_argument("--endpoint-window", type=int, default=7)
    parser.add_argument("--allow-missing-appearance", action="store_true",
                        help="Re-ID kaniti olmadan birlestirmeye izin ver (varsayilan: kapali)")
    parser.add_argument("--ambiguity-margin", type=float, default=0.15,
                        help="Yakin maliyetli alternatif varsa birlestirme yapma")
    return parser.parse_args()


def load_tracks(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Track verisi bos: {path}")
    required = {"frame", "track_id", "field_x", "field_y"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Eksik sutunlar: {', '.join(sorted(missing))}")
    return fieldnames, rows


def load_embeddings(path: Path):
    data = np.load(path)
    counts = data["counts"] if "counts" in data.files else np.ones(len(data["ids"]), dtype=np.int32)
    return {
        int(identity): (np.asarray(embedding, dtype=np.float64), int(count))
        for identity, embedding, count in zip(data["ids"], data["embeddings"], counts)
    }


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def endpoint(points, at_start, window):
    subset = points[:window] if at_start else points[-window:]
    return np.median(subset, axis=0)


def end_velocity(frames, points, window):
    frames = np.asarray(frames[-window:], dtype=np.float64)
    points = np.asarray(points[-window:], dtype=np.float64)
    if len(frames) < 3 or frames[-1] == frames[0]:
        return np.zeros(2, dtype=np.float64)
    centered = frames - frames.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-9:
        return np.zeros(2, dtype=np.float64)
    return (centered[:, None] * (points - points.mean(axis=0))).sum(axis=0) / denominator


def build_tracklets(rows, endpoint_window):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["track_id"])].append(row)

    tracklets = {}
    for identity, values in grouped.items():
        values.sort(key=lambda row: int(row["frame"]))
        frames = np.asarray([int(row["frame"]) for row in values], dtype=np.int32)
        points = np.asarray(
            [[float(row["field_x"]), float(row["field_y"])] for row in values],
            dtype=np.float64,
        )
        finite = np.isfinite(points).all(axis=1)
        frames = frames[finite]
        points = points[finite]
        if not len(frames):
            continue
        tracklets[identity] = {
            "start": int(frames[0]),
            "end": int(frames[-1]),
            "start_point": endpoint(points, True, endpoint_window),
            "end_point": endpoint(points, False, endpoint_window),
            "end_velocity": end_velocity(frames, points, endpoint_window),
            "observations": len(frames),
        }
    return tracklets


def appearance_distance(first, second, embeddings):
    one = embeddings.get(first)
    two = embeddings.get(second)
    if one is None or two is None:
        return None
    return 1.0 - float(np.clip(np.dot(normalized(one[0]), normalized(two[0])), -1.0, 1.0))


def candidate_links(tracklets, embeddings, args):
    candidates = []
    identities = sorted(tracklets, key=lambda identity: tracklets[identity]["start"])
    for first in identities:
        one = tracklets[first]
        for second in identities:
            if first == second:
                continue
            two = tracklets[second]
            gap = two["start"] - one["end"]
            if gap <= 0 or gap > args.max_gap:
                continue
            prediction = one["end_point"] + one["end_velocity"] * gap
            position = float(np.linalg.norm(prediction - two["start_point"]))
            if position > args.max_position:
                continue
            appearance = appearance_distance(first, second, embeddings)
            if appearance is None and not args.allow_missing_appearance:
                continue
            if appearance is not None and appearance > args.max_appearance:
                continue
            appearance_cost = 0.50 if appearance is None else appearance
            cost = (
                position / args.max_position
                + args.appearance_weight * appearance_cost
                + args.gap_weight * gap / args.max_gap
            )
            candidates.append((cost, first, second, gap, position, appearance))
    return sorted(candidates)


def select_links(candidates, ambiguity_margin=0.0):
    predecessor = {}
    successor = {}
    selected = []
    candidates = sorted(candidates)
    # Mutual best match with a margin: fewer merges are preferable to mixing two players.
    by_first, by_second = defaultdict(list), defaultdict(list)
    for candidate in candidates:
        by_first[candidate[1]].append(candidate)
        by_second[candidate[2]].append(candidate)
    for candidate in candidates:
        _cost, first, second, *_rest = candidate
        if ambiguity_margin > 0:
            outgoing, incoming = by_first[first], by_second[second]
            if outgoing[0] != candidate or incoming[0] != candidate:
                continue
            if any(len(options) > 1 and options[1][0] - candidate[0] < ambiguity_margin
                   for options in (outgoing, incoming)):
                continue
        if first in successor or second in predecessor:
            continue
        successor[first] = second
        predecessor[second] = first
        selected.append(candidate)
    return predecessor, successor, selected


def make_mapping(tracklets, predecessor, successor):
    mapping = {}
    for identity in tracklets:
        if identity in predecessor:
            continue
        current = identity
        while True:
            mapping[current] = identity
            if current not in successor:
                break
            current = successor[current]
    for identity in tracklets:
        mapping.setdefault(identity, identity)
    return mapping


def save_tracks(path, fieldnames, rows, mapping):
    output_fields = list(fieldnames)
    insert_at = output_fields.index("track_id") + 1
    output_fields.insert(insert_at, "source_track_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for original in rows:
            row = dict(original)
            source_id = int(row["track_id"])
            row["source_track_id"] = source_id
            row["track_id"] = mapping.get(source_id, source_id)
            writer.writerow(row)


def save_embeddings(path, embeddings, mapping):
    grouped = defaultdict(list)
    for identity, (embedding, count) in embeddings.items():
        grouped[mapping.get(identity, identity)].append((embedding, count))
    identities = sorted(grouped)
    means = []
    counts = []
    for identity in identities:
        values = grouped[identity]
        total = sum(count for _embedding, count in values)
        mean = sum(embedding * count for embedding, count in values) / max(total, 1)
        means.append(normalized(mean).astype(np.float32))
        counts.append(total)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(identities, dtype=np.int32),
        embeddings=np.asarray(means, dtype=np.float32),
        counts=np.asarray(counts, dtype=np.int32),
    )


def save_mapping(path, tracklets, mapping, predecessor, selected):
    link_for_second = {candidate[2]: candidate for candidate in selected}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "stitched_id",
                "source_track_id",
                "start_frame",
                "end_frame",
                "observations",
                "predecessor_id",
                "gap_frames",
                "position_distance",
                "appearance_distance",
                "link_cost",
            ]
        )
        for source_id in sorted(tracklets, key=lambda identity: (mapping[identity], tracklets[identity]["start"])):
            link = link_for_second.get(source_id)
            writer.writerow(
                [
                    mapping[source_id],
                    source_id,
                    tracklets[source_id]["start"],
                    tracklets[source_id]["end"],
                    tracklets[source_id]["observations"],
                    predecessor.get(source_id, ""),
                    "" if link is None else link[3],
                    "" if link is None else f"{link[4]:.6f}",
                    "" if link is None or link[5] is None else f"{link[5]:.6f}",
                    "" if link is None else f"{link[0]:.6f}",
                ]
            )


def main():
    args = parse_args()
    fieldnames, rows = load_tracks(args.tracks)
    embeddings = load_embeddings(args.reid)
    tracklets = build_tracklets(rows, args.endpoint_window)
    candidates = candidate_links(tracklets, embeddings, args)
    predecessor, successor, selected = select_links(candidates, args.ambiguity_margin)
    mapping = make_mapping(tracklets, predecessor, successor)

    save_tracks(args.output_tracks, fieldnames, rows, mapping)
    save_embeddings(args.output_reid, embeddings, mapping)
    save_mapping(args.mapping, tracklets, mapping, predecessor, selected)

    stitched_count = len(set(mapping.values()))
    print(f"Girdi tracklet sayisi: {len(tracklets)}")
    print(f"Secilen birlestirme: {len(selected)}")
    print(f"Birlestirme sonrasi ID: {stitched_count}")
    print(f"Track ciktisi: {args.output_tracks}")
    print(f"Re-ID ciktisi: {args.output_reid}")
    print(f"Esleme tablosu: {args.mapping}")


if __name__ == "__main__":
    main()
