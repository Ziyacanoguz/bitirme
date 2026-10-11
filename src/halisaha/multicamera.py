"""Conservative, deterministic and auditable offline multi-camera identity fusion.

Scores emitted by this module are heuristic evidence scores in ``[0, 1]``. They
are useful for ranking and review, but are deliberately not called probabilities:
there is no human-labelled cross-camera validation set yet.
"""

from collections import Counter, defaultdict
from hashlib import sha256
import json

import numpy as np


ASSOCIATION_SCHEMA_VERSION = "global-id-evidence-v2"


def _finite_scalar(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def valid_position(row):
    try:
        point = np.asarray([float(row["field_x"]), float(row["field_y"])])
        confidence = float(row["confidence"])
        return bool(
            np.isfinite(point).all()
            and (point >= 0).all()
            and (point <= 1).all()
            and int(row.get("calibration_supported", 0)) == 1
            and np.isfinite(confidence)
            and 0.15 <= confidence <= 1.0
        )
    except (KeyError, ValueError, TypeError):
        return False


def xy(row):
    return np.asarray([float(row["field_x"]), float(row["field_y"])])


def appearance_distance(a, b, features1, features2):
    first, second = features1.get(a), features2.get(b)
    if first is None or second is None:
        return None
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    return None if norm <= 1e-9 else 1 - float(np.clip(np.dot(first, second) / norm, -1, 1))


def frame_index(rows):
    """Return one deterministic, highest-confidence observation per frame/ID."""
    frames = defaultdict(dict)
    for row in rows:
        frame, identity = int(row["frame"]), int(row["track_id"])
        old = frames[frame].get(identity)
        if old is None or float(row["confidence"]) > float(old["confidence"]):
            frames[frame][identity] = row
    return {
        frame: [values[identity] for identity in sorted(values)]
        for frame, values in sorted(frames.items())
    }


def _longest_consecutive_run(frames):
    ordered = sorted(set(frames))
    longest = current = 0
    previous = None
    for frame in ordered:
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    return longest


def _cross_evidence_score(support, fraction, median_distance, appearance, *, min_support,
                          max_distance, appearance_gate):
    """Ranking score, not a calibrated probability."""
    support_quality = min(1.0, support / max(1.0, 2.0 * min_support))
    fraction_quality = float(np.clip(fraction, 0, 1))
    position_quality = 1.0 - float(np.clip(median_distance / max_distance, 0, 1))
    # Missing appearance must not receive the neutral score of a measured match.
    appearance_quality = (
        0.25 if appearance is None
        else 1.0 - float(np.clip(appearance / appearance_gate, 0, 1))
    )
    return float(np.clip(
        0.35 * support_quality
        + 0.30 * fraction_quality
        + 0.20 * position_quality
        + 0.15 * appearance_quality,
        0,
        1,
    ))


def _continuity_evidence_score(gap, position_distance, appearance):
    appearance_quality = 1.0 - float(np.clip(appearance / 0.22, 0, 1))
    position_quality = 1.0 - float(np.clip(position_distance / 0.045, 0, 1))
    gap_quality = 1.0 - float(np.clip((gap - 1) / 2, 0, 1))
    return float(np.clip(0.55 * appearance_quality + 0.35 * position_quality + 0.10 * gap_quality, 0, 1))


class ConflictSafeComponents:
    """Never unite distinct simultaneous tracks from the same camera."""

    def __init__(self, observations):
        nodes = sorted(observations)
        self.parent = {node: node for node in nodes}
        self.members = {node: {node} for node in nodes}
        self.observations = observations

    def root(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.root(self.parent[node])
        return self.parent[node]

    def overlap_conflict(self, first, second):
        a, b = self.root(first), self.root(second)
        if a == b:
            return False
        return any(
            one[0] == two[0] and bool(self.observations[one] & self.observations[two])
            for one in self.members[a]
            for two in self.members[b]
        )

    def merge(self, first, second):
        a, b = self.root(first), self.root(second)
        if a == b:
            return True
        if self.overlap_conflict(first, second):
            return False
        # A stable root makes the output independent of CSV row order.
        if b < a:
            a, b = b, a
        self.parent[b] = a
        self.members[a] |= self.members.pop(b)
        return True


def _component_appearance_conflict(components, first, second, node_features, gate):
    if gate is None:
        return None
    a, b = components.root(first), components.root(second)
    if a == b:
        return None
    distances = []
    for one in components.members[a]:
        feature1 = node_features.get(one)
        if feature1 is None:
            continue
        for two in components.members[b]:
            feature2 = node_features.get(two)
            if feature2 is None:
                continue
            norm = np.linalg.norm(feature1) * np.linalg.norm(feature2)
            if norm > 1e-9:
                distances.append(1 - float(np.clip(np.dot(feature1, feature2) / norm, -1, 1)))
    if distances and max(distances) > gate:
        return float(max(distances))
    return None


def _continuity_candidates(rows1, rows2, features1, features2):
    output = []
    for camera, rows, features in ((1, rows1, features1), (2, rows2, features2)):
        segments = defaultdict(list)
        for row in rows:
            segments[int(row["track_id"])].append(row)
        by_local = defaultdict(list)
        for segment, values in segments.items():
            values.sort(key=lambda row: int(row["frame"]))
            if "local_track_id" in values[0] and values[0]["local_track_id"] != "":
                by_local[int(values[0]["local_track_id"])].append(
                    (int(values[0]["frame"]), segment, values)
                )
        for local_id, entries in sorted(by_local.items()):
            entries.sort()
            for (_, first_id, first), (_, second_id, second) in zip(entries, entries[1:]):
                gap = int(second[0]["frame"]) - int(first[-1]["frame"])
                position_distance = (
                    float(np.linalg.norm(xy(first[-1]) - xy(second[0])))
                    if valid_position(first[-1]) and valid_position(second[0]) else None
                )
                appearance = appearance_distance(first_id, second_id, features, features)
                reason = "eligible"
                if gap < 1 or gap > 3:
                    reason = "continuity_gap_out_of_range"
                elif position_distance is None:
                    reason = "continuity_invalid_endpoint"
                elif appearance is None:
                    reason = "continuity_appearance_missing"
                elif appearance > 0.22:
                    reason = "continuity_appearance_mismatch"
                elif position_distance > 0.045:
                    reason = "continuity_position_jump"
                output.append(dict(
                    camera=camera,
                    source_local_id=local_id,
                    first_track_id=first_id,
                    second_track_id=second_id,
                    first_end_frame=int(first[-1]["frame"]),
                    second_start_frame=int(second[0]["frame"]),
                    gap_frames=gap,
                    position_distance=position_distance if position_distance is not None else "",
                    appearance=appearance if appearance is not None else "",
                    evidence_score=(
                        _continuity_evidence_score(gap, position_distance, appearance)
                        if reason == "eligible" else ""
                    ),
                    accepted=False,
                    component_link=False,
                    reason=reason,
                ))
    return output


def associate(rows1, rows2, features1, features2, *, offset_frames=0, min_support=15,
              max_distance=0.10, appearance_gate=0.45, margin=0.12, min_fraction=0.35,
              min_consecutive_support=1, max_p95_distance=None,
              missing_appearance_min_support=None, missing_appearance_min_fraction=None,
              missing_appearance_max_p95=None, component_appearance_gate=0.50,
              return_diagnostics=False):
    """Associate short tracks while retaining every decision needed for an audit."""
    required_thresholds = (
        min_support,
        max_distance,
        appearance_gate,
        margin,
        min_fraction,
        min_consecutive_support,
    )
    optional_thresholds = (
        max_p95_distance,
        missing_appearance_min_support,
        missing_appearance_min_fraction,
        missing_appearance_max_p95,
        component_appearance_gate,
    )
    if (
        not all(_finite_scalar(value) for value in required_thresholds)
        or not all(value is None or _finite_scalar(value) for value in optional_thresholds)
        or min_support < 1 or max_distance <= 0 or appearance_gate <= 0 or margin < 0
        or not 0 <= min_fraction <= 1 or min_consecutive_support < 1
        or max_p95_distance is not None and max_p95_distance <= 0
        or missing_appearance_min_support is not None and missing_appearance_min_support < 1
        or missing_appearance_min_fraction is not None and not 0 <= missing_appearance_min_fraction <= 1
        or missing_appearance_max_p95 is not None and missing_appearance_max_p95 <= 0
        or component_appearance_gate is not None and component_appearance_gate <= 0
    ):
        raise ValueError("Invalid association thresholds")

    frames1, frames2 = frame_index(rows1), frame_index(rows2)
    evidence = defaultdict(list)
    observations, valid_frames = defaultdict(set), defaultdict(set)
    for camera, rows in ((1, rows1), (2, rows2)):
        for row in rows:
            node = (camera, int(row["track_id"]))
            frame = int(row["frame"]) - (offset_frames if camera == 2 else 0)
            observations[node].add(frame)
            if valid_position(row):
                valid_frames[node].add(frame)

    for frame in sorted(frames1):
        first = [row for row in frames1[frame] if valid_position(row)]
        second = [row for row in frames2.get(frame + offset_frames, []) if valid_position(row)]
        if not first or not second:
            continue
        distance = np.linalg.norm(
            np.asarray([xy(row) for row in first])[:, None, :]
            - np.asarray([xy(row) for row in second])[None, :, :],
            axis=2,
        )
        cost = np.full(distance.shape, np.inf)
        appearances = {}
        for i, one in enumerate(first):
            for j, two in enumerate(second):
                a, b = int(one["track_id"]), int(two["track_id"])
                appearance = appearance_distance(a, b, features1, features2)
                appearances[(i, j)] = appearance
                if distance[i, j] > max_distance:
                    continue
                if appearance is not None and appearance > appearance_gate:
                    continue
                if appearance is None and distance[i, j] > max_distance * 0.65:
                    continue
                cost[i, j] = (
                    distance[i, j] / max_distance
                    + (0.35 * appearance / appearance_gate if appearance is not None else 0.2)
                )
        # Reject ties/near-ties; no forced Hungarian assignment of uncertain observations.
        for i in range(len(first)):
            j = int(np.argmin(cost[i]))
            best = cost[i, j]
            if not np.isfinite(best) or int(np.argmin(cost[:, j])) != i:
                continue
            alternatives_row = np.delete(cost[i], j)
            alternatives_col = np.delete(cost[:, j], i)
            if (
                alternatives_row.size and np.min(alternatives_row) - best < margin
                or alternatives_col.size and np.min(alternatives_col) - best < margin
            ):
                continue
            evidence[(int(first[i]["track_id"]), int(second[j]["track_id"]))].append(
                (frame, float(distance[i, j]), float(best), appearances[(i, j)])
            )

    candidates = []
    for (camera1_id, camera2_id), support_rows in sorted(evidence.items()):
        frames = [item[0] for item in support_rows]
        support = len(support_rows)
        overlap = len(valid_frames[(1, camera1_id)] & valid_frames[(2, camera2_id)])
        fraction = support / max(1, overlap)
        distances = [item[1] for item in support_rows]
        costs = [item[2] for item in support_rows]
        appearance = appearance_distance(camera1_id, camera2_id, features1, features2)
        median_distance = float(np.median(distances))
        p95_distance = float(np.quantile(distances, 0.95))
        candidate = dict(
            camera1_id=camera1_id,
            camera2_id=camera2_id,
            first_frame=min(frames),
            last_frame=max(frames),
            support_frames=json.dumps(sorted(set(frames)), separators=(",", ":")),
            support_span_frames=max(frames) - min(frames) + 1,
            support=support,
            longest_consecutive_support=_longest_consecutive_run(frames),
            overlap=overlap,
            support_fraction=fraction,
            median_distance=median_distance,
            p95_distance=p95_distance,
            median_match_cost=float(np.median(costs)),
            appearance=appearance if appearance is not None else "",
            appearance_available=int(appearance is not None),
            evidence_score=_cross_evidence_score(
                support,
                fraction,
                median_distance,
                appearance,
                min_support=min_support,
                max_distance=max_distance,
                appearance_gate=appearance_gate,
            ),
            accepted=False,
            component_link=False,
            reason="pending",
        )
        if support < min_support:
            candidate["reason"] = "insufficient_support"
        elif candidate["longest_consecutive_support"] < min_consecutive_support:
            candidate["reason"] = "insufficient_consecutive_support"
        elif fraction < min_fraction:
            candidate["reason"] = "insufficient_support_fraction"
        elif max_p95_distance is not None and p95_distance > max_p95_distance:
            candidate["reason"] = "excessive_p95_distance"
        elif appearance is None and (
            support < (missing_appearance_min_support or min_support)
            or fraction < (
                min_fraction if missing_appearance_min_fraction is None
                else missing_appearance_min_fraction
            )
            or missing_appearance_max_p95 is not None and p95_distance > missing_appearance_max_p95
        ):
            candidate["reason"] = "appearance_missing_weak_geometry"
        else:
            candidate["reason"] = "eligible"
        candidates.append(candidate)

    node_features = {
        **{(1, int(identity)): feature for identity, feature in features1.items()},
        **{(2, int(identity)): feature for identity, feature in features2.items()},
    }
    components = ConflictSafeComponents(observations)

    # Artificial segments of one uninterrupted local tracker ID are considered
    # before cross-camera edges. A weaker cross-camera edge must not block this
    # stronger local continuity evidence.
    continuity = _continuity_candidates(rows1, rows2, features1, features2)
    for row in sorted(
        continuity,
        key=lambda item: (
            -(float(item["evidence_score"]) if item["evidence_score"] != "" else -1),
            item["camera"],
            item["first_track_id"],
            item["second_track_id"],
        ),
    ):
        if row["reason"] != "eligible":
            continue
        first = (row["camera"], row["first_track_id"])
        second = (row["camera"], row["second_track_id"])
        if components.overlap_conflict(first, second):
            row["reason"] = "continuity_same_camera_overlap_conflict"
            continue
        conflict = _component_appearance_conflict(
            components, first, second, node_features, component_appearance_gate
        )
        if conflict is not None:
            row["reason"] = "continuity_component_appearance_conflict"
            row["component_max_appearance"] = conflict
            continue
        was_separate = components.root(first) != components.root(second)
        components.merge(first, second)
        row.update(
            accepted=True,
            component_link=was_separate,
            reason="continuity_component_link" if was_separate else "continuity_corroborating_edge",
        )

    # Highest-quality evidence gets first claim on a component. Stable tie-breaks
    # make shuffled input rows produce identical decisions.
    candidates.sort(key=lambda row: (
        -row["evidence_score"],
        -row["support"],
        row["median_distance"],
        row["camera1_id"],
        row["camera2_id"],
    ))
    for row in candidates:
        if row["reason"] != "eligible":
            continue
        first, second = (1, row["camera1_id"]), (2, row["camera2_id"])
        if components.overlap_conflict(first, second):
            row["reason"] = "same_camera_overlap_conflict"
            continue
        conflict = _component_appearance_conflict(
            components, first, second, node_features, component_appearance_gate
        )
        if conflict is not None:
            row["reason"] = "component_appearance_conflict"
            row["component_max_appearance"] = conflict
            continue
        was_separate = components.root(first) != components.root(second)
        components.merge(first, second)
        row.update(
            accepted=True,
            component_link=was_separate,
            reason="cross_camera_component_link" if was_separate else "cross_camera_corroborating_edge",
        )

    groups = [set(members) for members in components.members.values()]
    groups.sort(key=lambda members: (
        min(min(observations[node]) for node in members),
        min(members),
    ))
    mapping, shared = {}, set()
    for global_id, members in enumerate(groups, 1):
        if len({node[0] for node in members}) == 2:
            shared.add(global_id)
        for node in sorted(members):
            mapping[node] = global_id

    diagnostics = dict(
        schema_version=ASSOCIATION_SCHEMA_VERSION,
        score_semantics="heuristic_evidence_not_probability",
        thresholds=dict(
            min_support=min_support,
            min_consecutive_support=min_consecutive_support,
            max_distance=max_distance,
            max_p95_distance=max_p95_distance,
            appearance_gate=appearance_gate,
            ambiguity_margin=margin,
            min_fraction=min_fraction,
            missing_appearance_min_support=missing_appearance_min_support or min_support,
            missing_appearance_min_fraction=(
                min_fraction if missing_appearance_min_fraction is None
                else missing_appearance_min_fraction
            ),
            missing_appearance_max_p95=missing_appearance_max_p95,
            component_appearance_gate=component_appearance_gate,
        ),
        candidate_decisions=dict(Counter(row["reason"] for row in candidates)),
        continuity_decisions=dict(Counter(row["reason"] for row in continuity)),
        continuity=sorted(
            continuity,
            key=lambda row: (row["camera"], row["first_track_id"], row["second_track_id"]),
        ),
    )
    result = (mapping, shared, candidates)
    return (*result, diagnostics) if return_diagnostics else result


def _component_uid(nodes):
    canonical = "|".join(f"{camera}:{identity}" for camera, identity in sorted(nodes))
    return sha256(canonical.encode("ascii")).hexdigest()[:16]


def build_component_audit(mapping, shared, candidates, diagnostics, rows1, rows2, *, offset_frames=0):
    """Summarize graph evidence and return lookup tables used by frame fusion."""
    nodes_by_component = defaultdict(list)
    for node, global_id in mapping.items():
        nodes_by_component[global_id].append(node)

    rows_by_node = defaultdict(list)
    source_local = {}
    for camera, rows in ((1, rows1), (2, rows2)):
        for row in rows:
            node = (camera, int(row["track_id"]))
            rows_by_node[node].append(row)
            value = row.get("local_track_id", row["track_id"])
            source_local[node] = int(value)

    links = []
    pair_evidence = {}
    for row in candidates:
        if not row["accepted"]:
            continue
        first, second = (1, int(row["camera1_id"])), (2, int(row["camera2_id"]))
        score = float(row["evidence_score"])
        pair_evidence[(first, second)] = score
        if row["component_link"]:
            links.append((first, second, score, "cross_camera"))
    for row in diagnostics["continuity"]:
        if row["accepted"] and row["component_link"]:
            links.append((
                (int(row["camera"]), int(row["first_track_id"])),
                (int(row["camera"]), int(row["second_track_id"])),
                float(row["evidence_score"]),
                "local_continuity",
            ))

    component_rows, component_evidence, component_uids = [], {}, {}
    for global_id, nodes in sorted(nodes_by_component.items()):
        nodes = sorted(nodes)
        uid = _component_uid(nodes)
        component_uids[global_id] = uid
        member_links = [
            link for link in links
            if mapping[link[0]] == global_id and mapping[link[1]] == global_id
        ]
        scores = [link[2] for link in member_links]
        evidence_score = min(scores) if scores else None
        component_evidence[global_id] = evidence_score
        canonical_frames = {
            int(row["frame"]) - (offset_frames if node[0] == 2 else 0)
            for node in nodes for row in rows_by_node[node]
        }
        component_rows.append(dict(
            global_id=global_id,
            component_uid=uid,
            shared=int(global_id in shared),
            identity_evidence_score=evidence_score if evidence_score is not None else "",
            evidence_semantics="weakest_graph_link_heuristic_not_probability",
            first_frame=min(canonical_frames),
            last_frame=max(canonical_frames),
            observed_canonical_frames=len(canonical_frames),
            camera1_segment_count=sum(node[0] == 1 for node in nodes),
            camera2_segment_count=sum(node[0] == 2 for node in nodes),
            camera1_track_ids="|".join(str(node[1]) for node in nodes if node[0] == 1),
            camera2_track_ids="|".join(str(node[1]) for node in nodes if node[0] == 2),
            camera1_source_local_ids="|".join(
                map(str, sorted({source_local[node] for node in nodes if node[0] == 1}))
            ),
            camera2_source_local_ids="|".join(
                map(str, sorted({source_local[node] for node in nodes if node[0] == 2}))
            ),
            cross_camera_component_links=sum(link[3] == "cross_camera" for link in member_links),
            continuity_component_links=sum(link[3] == "local_continuity" for link in member_links),
        ))
    return component_rows, component_evidence, component_uids, pair_evidence


def observation_weight(row):
    height = max(1, float(row["y2"]) - float(row["y1"]))
    width = max(1, float(row["x2"]) - float(row["x1"]))
    # Higher-resolution full-height boxes are usually a better foot observation.
    body_quality = min(1.0, height / max(1.0, width * 2))
    return max(0.01, float(row["confidence"]) ** 2 * min(height, 180) ** 2 * body_quality)


def fuse_frames(rows1, rows2, mapping, shared, *, offset_frames=0, fps=25,
                max_disagreement=0.10, pair_evidence=None, component_evidence=None,
                component_uids=None):
    if (
        not _finite_scalar(fps)
        or not _finite_scalar(max_disagreement)
        or fps <= 0
        or max_disagreement <= 0
    ):
        raise ValueError("Invalid fusion settings")
    pair_evidence = pair_evidence or {}
    component_evidence = component_evidence or {}
    component_uids = component_uids or {}
    first, second = frame_index(rows1), frame_index(rows2)
    frames = sorted(set(first) | {frame - offset_frames for frame in second})
    output, totals = [], Counter()
    # Conflict observations have camera-specific IDs, never masquerade as a fused point.
    fallback_ids = {
        node: max(mapping.values(), default=0) + index + 1
        for index, node in enumerate(sorted(mapping))
    }
    for frame in frames:
        groups = defaultdict(list)
        for camera, rows in (
            (1, first.get(frame, [])),
            (2, second.get(frame + offset_frames, [])),
        ):
            for row in rows:
                if valid_position(row):
                    node = (camera, int(row["track_id"]))
                    groups[mapping[node]].append((camera, row))
                else:
                    totals["invalid_position_rows"] += 1
        for global_id in sorted(groups):
            observations = sorted(groups[global_id], key=lambda item: item[0])
            points = np.asarray([xy(row) for _, row in observations])
            disagreement = (
                float(np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2)))
                if len(points) > 1 else 0.0
            )
            conflict = len(points) > 1 and disagreement > max_disagreement
            bundles = [[item] for item in observations] if conflict else [observations]
            if conflict:
                totals["disagreement_events"] += 1
            for bundle in bundles:
                weights = np.asarray([observation_weight(row) for _, row in bundle])
                point = np.average([xy(row) for _, row in bundle], axis=0, weights=weights)
                cameras = sorted(camera for camera, _ in bundle)
                node = (bundle[0][0], int(bundle[0][1]["track_id"]))
                shared_identity = global_id in shared and not conflict
                identity = fallback_ids[node] if conflict else global_id
                local_label = bundle[0][1].get("local_track_id", node[1])
                label = f"G{global_id}" if shared_identity else f"C{node[0]}L{local_label}"
                status = (
                    "conflict" if conflict else "fused" if len(bundle) > 1
                    else "single_view" if shared_identity else "unmatched"
                )
                totals[status + "_rows"] += 1
                direct_score = None
                if len(bundle) == 2:
                    direct_score = pair_evidence.get((
                        (1, int(bundle[0][1]["track_id"])),
                        (2, int(bundle[1][1]["track_id"])),
                    ))
                identity_score = (
                    None if conflict or not shared_identity else
                    direct_score if direct_score is not None else component_evidence.get(global_id)
                )
                identity_kind = (
                    "none_conflict" if conflict else "none_unmatched" if not shared_identity else
                    "direct_pair" if direct_score is not None else "component_graph"
                )
                row_by_camera = {camera: row for camera, row in bundle}
                output.append(dict(
                    frame=frame,
                    time=f"{frame / fps:.3f}",
                    source_frame_camera1=frame if 1 in row_by_camera else "",
                    source_frame_camera2=frame + offset_frames if 2 in row_by_camera else "",
                    offset_frames_applied=offset_frames,
                    track_id=identity,
                    global_id=global_id if shared_identity else "",
                    component_uid=component_uids.get(global_id, "") if shared_identity else "",
                    display_label=label,
                    identity_evidence_score=identity_score if identity_score is not None else "",
                    identity_evidence_kind=identity_kind,
                    identity_evidence_semantics="heuristic_not_probability",
                    field_x=float(point[0]),
                    field_y=float(point[1]),
                    position_disagreement=disagreement,
                    confidence=max(float(row["confidence"]) for _, row in bundle),
                    detection_confidence_min=min(float(row["confidence"]) for _, row in bundle),
                    calibration_supported=1,
                    status=status,
                    cameras="+".join(map(str, cameras)),
                    camera1_id=(int(row_by_camera[1]["track_id"]) if 1 in row_by_camera else ""),
                    camera2_id=(int(row_by_camera[2]["track_id"]) if 2 in row_by_camera else ""),
                    camera1_local_track_id=(
                        int(row_by_camera[1].get("local_track_id", row_by_camera[1]["track_id"]))
                        if 1 in row_by_camera else ""
                    ),
                    camera2_local_track_id=(
                        int(row_by_camera[2].get("local_track_id", row_by_camera[2]["track_id"]))
                        if 2 in row_by_camera else ""
                    ),
                    observation_count=len(bundle),
                ))
    # Backward-compatible alias with clarified semantics in new reports.
    totals["disagreement_frames"] = totals["disagreement_events"]
    return output, dict(totals)


def audit_time_offset(rows1, rows2, candidates, radius=8):
    """Motion-consistency sweep only; do not silently apply an uncertain offset."""
    tracks = []
    for rows in (rows1, rows2):
        by_id = defaultdict(dict)
        for row in rows:
            if valid_position(row):
                by_id[int(row["track_id"])][int(row["frame"])] = xy(row)
        tracks.append(by_id)
    pairs = [row for row in candidates if row["accepted"] and row["support"] >= 100]
    scores = {offset: [] for offset in range(-radius, radius + 1)}
    per_pair = []
    for pair in pairs:
        first = tracks[0][pair["camera1_id"]]
        second = tracks[1][pair["camera2_id"]]
        pair_scores = {}
        for offset in scores:
            errors = []
            for frame in first:
                if (
                    frame - 5 not in first or frame + offset not in second
                    or frame + offset - 5 not in second
                ):
                    continue
                delta1 = first[frame] - first[frame - 5]
                delta2 = second[frame + offset] - second[frame + offset - 5]
                if max(np.linalg.norm(delta1), np.linalg.norm(delta2)) < 0.006:
                    continue
                errors.append(np.linalg.norm(delta1 - delta2))
            if len(errors) >= 50:
                pair_scores[offset] = float(np.median(errors))
                scores[offset].append(pair_scores[offset])
        if len(pair_scores) == len(scores):
            per_pair.append(dict(
                camera1_id=pair["camera1_id"],
                camera2_id=pair["camera2_id"],
                best_offset=min(pair_scores, key=pair_scores.get),
            ))
    aggregate = {offset: float(np.median(values)) for offset, values in scores.items() if values}
    best = min(aggregate, key=aggregate.get) if aggregate else None
    return dict(
        score_by_offset_frames=aggregate,
        best_offset_candidate=best,
        per_pair=per_pair,
        applied_automatically=False,
        note=(
            "Motion-only diagnostic; fixed geometry and detector foot noise remain. "
            "Audio is not evaluated by this diagnostic."
        ),
    )
