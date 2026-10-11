"""Temporal-constrained Re-ID appearance clustering for Global ID consolidation.

Reduces fragmented global component IDs down to the physical player count (~12-14)
by clustering non-overlapping tracklets using long-term OSNet Re-ID embeddings.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float | None:
    if u is None or v is None:
        return None
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u <= 1e-9 or norm_v <= 1e-9:
        return None
    similarity = np.dot(u, v) / (norm_u * norm_v)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def consolidate_global_identities(
    fused_rows: list[dict],
    mapping_rows: list[dict],
    features1: dict[int, np.ndarray] | None = None,
    features2: dict[int, np.ndarray] | None = None,
    *,
    target_players: int = 14,
    max_appearance_distance: float = 0.45,
    max_overlap_tolerance: int = 0,
    max_overlap_ratio: float = 0.25,
    strict_overlap_appearance_gate: float = 0.25,
) -> tuple[list[dict], list[dict], dict[int, int]]:
    """Cluster non-overlapping global component IDs using Re-ID embeddings.

    Returns: (consolidated_fused_rows, consolidated_mapping_rows, global_id_map)
    """
    features1 = features1 or {}
    features2 = features2 or {}

    # 1. Group observations and frame intervals by global_id
    frames_by_gid: dict[int, set[int]] = defaultdict(set)
    rows_by_gid: dict[int, list[dict]] = defaultdict(list)
    for row in fused_rows:
        gid = row.get("global_id")
        if gid != "" and gid is not None:
            gid_int = int(gid)
            frame = int(row["frame"])
            frames_by_gid[gid_int].add(frame)
            rows_by_gid[gid_int].append(row)

    # Also check mapping_rows for complete frame coverage
    track_node_features: dict[tuple[int, int], np.ndarray] = {}
    for identity, feat in features1.items():
        track_node_features[(1, int(identity))] = feat
    for identity, feat in features2.items():
        track_node_features[(2, int(identity))] = feat

    nodes_by_gid: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in mapping_rows:
        gid = int(row["global_id"])
        camera = int(row["camera"])
        track_id = int(row["track_id"])
        nodes_by_gid[gid].add((camera, track_id))

    unique_gids = sorted(nodes_by_gid.keys())
    if not unique_gids:
        return fused_rows, mapping_rows, {}

    # 2. Compute aggregated Re-ID feature vector for each global_id component
    gid_features: dict[int, np.ndarray | None] = {}
    for gid in unique_gids:
        feats = []
        for node in nodes_by_gid[gid]:
            feat = track_node_features.get(node)
            if feat is not None:
                norm = np.linalg.norm(feat)
                if norm > 1e-9:
                    feats.append(feat / norm)
        if feats:
            avg_feat = np.mean(feats, axis=0)
            norm = np.linalg.norm(avg_feat)
            gid_features[gid] = avg_feat / norm if norm > 1e-9 else None
        else:
            gid_features[gid] = None

    # 3. Disjoint set (Union-Find) with temporal overlap conflict checks
    parent = {gid: gid for gid in unique_gids}
    member_gids = {gid: {gid} for gid in unique_gids}
    member_frames = {gid: set(frames_by_gid[gid]) for gid in unique_gids}

    def root(n: int) -> int:
        if parent[n] != n:
            parent[n] = root(parent[n])
        return parent[n]

    # Extract position endpoints and average position for each component
    gid_endpoints: dict[int, tuple[float, float, int, int]] = {}
    for gid, rows in rows_by_gid.items():
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda r: int(r["frame"]))
        first_r, last_r = sorted_rows[0], sorted_rows[-1]
        try:
            gid_endpoints[gid] = (
                float(last_r["field_x"]),
                float(last_r["field_y"]),
                int(last_r["frame"]),
                int(first_r["frame"]),
            )
        except (KeyError, ValueError):
            pass

    def can_merge(g1: int, g2: int) -> tuple[bool, float | None]:
        r1, r2 = root(g1), root(g2)
        if r1 == r2:
            return False, None
        # Temporal overlap check: two tracks active at the same frame cannot be the same player
        # unless it is a minor handover/split tracklet with small overlap and confident appearance.
        overlap = member_frames[r1] & member_frames[r2]
        if overlap:
            if max_overlap_tolerance <= 0:
                return False, None
            ov_len = len(overlap)
            min_len = min(len(member_frames[r1]), len(member_frames[r2]))
            if ov_len > max_overlap_tolerance or (min_len > 0 and (ov_len / min_len) > max_overlap_ratio):
                return False, None
            f1, f2 = gid_features[r1], gid_features[r2]
            dist = _cosine_distance(f1, f2)
            if dist is not None and dist <= strict_overlap_appearance_gate:
                return True, dist
            return False, None

        # Re-ID feature distance
        f1, f2 = gid_features[r1], gid_features[r2]
        dist = _cosine_distance(f1, f2)
        if dist is not None and dist <= max_appearance_distance:
            return True, dist

        # Spatial + temporal endpoint fallback if Re-ID feature is missing
        if (f1 is None or f2 is None) and r1 in gid_endpoints and r2 in gid_endpoints:
            x1, y1, end1, start1 = gid_endpoints[r1]
            x2, y2, end2, start2 = gid_endpoints[r2]
            # Gap in frames
            if end1 < start2:
                gap = start2 - end1
                p_dist = float(np.hypot(x1 - x2, y1 - y2))
            elif end2 < start1:
                gap = start1 - end2
                p_dist = float(np.hypot(x1 - x2, y1 - y2))
            else:
                gap = 0
                p_dist = float(np.hypot(x1 - x2, y1 - y2))

            if gap <= 75 and p_dist <= 0.08:
                spatial_cost = 0.25 + p_dist
                return True, spatial_cost

        return False, None

    # 4. Agglomerative merging loop
    while True:
        # Stop condition: check how many distinct active player roots exist in fused_rows
        fused_active_roots = set(
            root(int(r["global_id"]))
            for r in fused_rows
            if r.get("global_id") != "" and r.get("global_id") is not None
        )
        if len(fused_active_roots) <= target_players:
            break

        current_roots = sorted(set(root(g) for g in unique_gids))
        if len(current_roots) <= target_players:
            break

        best_pair = None
        best_dist = float("inf")

        for i, r1 in enumerate(current_roots):
            for r2 in current_roots[i + 1 :]:
                eligible, dist = can_merge(r1, r2)
                if eligible and dist is not None and dist < best_dist:
                    best_dist = dist
                    best_pair = (r1, r2)

        if best_pair is None or best_dist > max_appearance_distance:
            break

        r1, r2 = best_pair
        if r2 < r1:
            r1, r2 = r2, r1
        parent[r2] = r1
        member_gids[r1] |= member_gids.pop(r2)
        member_frames[r1] |= member_frames.pop(r2)
        # Update feature average
        feats = [f for f in (gid_features[r1], gid_features[r2]) if f is not None]
        if feats:
            avg_f = np.mean(feats, axis=0)
            norm = np.linalg.norm(avg_f)
            gid_features[r1] = avg_f / norm if norm > 1e-9 else None

    # 5. Canonical re-numbering: sort clusters by earliest frame -> G1, G2, ..., G_N
    final_clusters: list[tuple[int, int, set[int]]] = []
    for r in sorted(set(root(g) for g in unique_gids)):
        all_frames = member_frames[r]
        min_frame = min(all_frames) if all_frames else 999999999
        final_clusters.append((min_frame, r, member_gids[r]))

    final_clusters.sort(key=lambda item: (item[0], item[1]))

    raw_to_canonical: dict[int, int] = {}
    for canonical_id, (_, _, original_gids) in enumerate(final_clusters, 1):
        for original_gid in original_gids:
            raw_to_canonical[original_gid] = canonical_id

    # 6. Update mapping rows
    consolidated_mapping = []
    for row in mapping_rows:
        new_row = dict(row)
        orig_gid = int(row["global_id"])
        if orig_gid in raw_to_canonical:
            new_row["global_id"] = raw_to_canonical[orig_gid]
        consolidated_mapping.append(new_row)

    # 7. Update fused rows
    consolidated_fused = []
    for row in fused_rows:
        new_row = dict(row)
        gid = row.get("global_id")
        if gid != "" and gid is not None:
            orig_gid = int(gid)
            if orig_gid in raw_to_canonical:
                c_id = raw_to_canonical[orig_gid]
                new_row["global_id"] = c_id
                if row.get("display_label", "").startswith("G"):
                    new_row["display_label"] = f"G{c_id}"
        consolidated_fused.append(new_row)

    return consolidated_fused, consolidated_mapping, raw_to_canonical
