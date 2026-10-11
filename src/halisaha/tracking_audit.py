"""Ground-truth-free diagnostics for player tracking CSV files.

The values produced here are risk indicators, not tracking accuracy metrics.  In
particular, a short track or a high-IoU pair can be legitimate.  IDF1, HOTA and
ID-switch counts require manually verified identities and are intentionally not
estimated by this module.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_COLUMNS = {"frame", "track_id", "confidence", "x1", "y1", "x2", "y2"}


@dataclass(frozen=True)
class AuditConfig:
    fps: float = 25.0
    start_frame: int | None = None
    end_frame: int | None = None
    short_track_seconds: float = 1.0
    low_confidence_threshold: float = 0.25
    jump_threshold_box_heights: float = 0.75
    duplicate_iou_threshold: float = 0.70
    fragment_max_frame_gap: int = 5
    fragment_max_distance_per_frame: float = 0.75
    fragment_max_height_ratio: float = 2.0

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.start_frame is not None and self.start_frame < 0:
            raise ValueError("start_frame cannot be negative")
        if self.end_frame is not None and self.end_frame < 0:
            raise ValueError("end_frame cannot be negative")
        if self.start_frame is not None and self.end_frame is not None:
            if self.end_frame < self.start_frame:
                raise ValueError("end_frame must be >= start_frame")
        if self.short_track_seconds <= 0:
            raise ValueError("short_track_seconds must be positive")
        for name in ("low_confidence_threshold", "duplicate_iou_threshold"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.jump_threshold_box_heights <= 0:
            raise ValueError("jump_threshold_box_heights must be positive")
        if self.fragment_max_frame_gap < 1:
            raise ValueError("fragment_max_frame_gap must be >= 1")
        if self.fragment_max_distance_per_frame <= 0:
            raise ValueError("fragment_max_distance_per_frame must be positive")
        if self.fragment_max_height_ratio < 1:
            raise ValueError("fragment_max_height_ratio must be >= 1")


@dataclass(frozen=True)
class Observation:
    frame: int
    track_id: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    line_number: int = 0

    @property
    def valid_box(self) -> bool:
        values = (self.x1, self.y1, self.x2, self.y2)
        return all(math.isfinite(value) for value in values) and self.x2 > self.x1 and self.y2 > self.y1

    @property
    def valid_confidence(self) -> bool:
        return math.isfinite(self.confidence) and 0 <= self.confidence <= 1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


def _id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _descriptive(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values]
    if not items:
        return {key: None for key in ("min", "p05", "median", "mean", "p95", "max")}
    return {
        "min": min(items),
        "p05": _quantile(items, 0.05),
        "median": statistics.median(items),
        "mean": statistics.fmean(items),
        "p95": _quantile(items, 0.95),
        "max": max(items),
    }


def _iou(first: Observation, second: Observation) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = first.width * first.height
    second_area = second.width * second.height
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def load_tracking_csv(path: str | Path) -> list[Observation]:
    """Load a detector/tracker CSV and reject values that cannot be audited."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if "frame_number" in fields and "frame" not in fields:
            fields.add("frame")
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(sorted(missing))}")
        observations = []
        for line_number, row in enumerate(reader, 2):
            try:
                frame_val = row.get("frame") if "frame" in row else row.get("frame_number")
                frame = int(frame_val)
                track_id = str(row["track_id"]).strip()
                confidence = float(row["confidence"])
                x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: malformed tracking row") from error
            if frame < 0 or not track_id:
                raise ValueError(f"{source}:{line_number}: invalid frame or empty track_id")
            observations.append(Observation(frame, track_id, confidence, x1, y1, x2, y2, line_number))
    if not observations:
        raise ValueError(f"{source}: tracking CSV is empty")
    return observations


def _canonicalize(observations: Iterable[Observation]) -> tuple[list[Observation], int]:
    """Keep one observation per frame/ID while retaining the duplicate-row count."""
    chosen: dict[tuple[int, str], Observation] = {}
    duplicate_rows = 0
    for observation in observations:
        key = (observation.frame, observation.track_id)
        previous = chosen.get(key)
        if previous is not None:
            duplicate_rows += 1
        rank = (observation.valid_box, observation.valid_confidence, observation.confidence)
        previous_rank = (
            (previous.valid_box, previous.valid_confidence, previous.confidence)
            if previous is not None
            else (False, False, float("-inf"))
        )
        if previous is None or rank > previous_rank:
            chosen[key] = observation
    return sorted(chosen.values(), key=lambda item: (item.frame, _id_sort_key(item.track_id))), duplicate_rows


def _normalized_step(first: Observation, second: Observation) -> float | None:
    if not first.valid_box or not second.valid_box:
        return None
    elapsed = second.frame - first.frame
    if elapsed <= 0:
        return None
    distance = math.dist(first.foot, second.foot)
    reference_height = (first.height + second.height) / 2.0
    if reference_height <= 0:
        return None
    return distance / reference_height / elapsed


def audit_tracking(
    observations: Iterable[Observation],
    config: AuditConfig | None = None,
    *,
    source: str | Path | None = None,
) -> dict:
    """Calculate interpretable tracking-risk diagnostics without ground truth."""
    config = config or AuditConfig()
    config.validate()
    raw = list(observations)
    if not raw:
        raise ValueError("tracking observations are empty")
    canonical, duplicate_rows = _canonicalize(raw)

    inferred_start = min(item.frame for item in canonical)
    inferred_end = max(item.frame for item in canonical)
    start_frame = inferred_start if config.start_frame is None else config.start_frame
    end_frame = inferred_end if config.end_frame is None else config.end_frame
    if any(item.frame < start_frame or item.frame > end_frame for item in canonical):
        raise ValueError("configured frame range does not contain every CSV observation")

    by_frame: dict[int, list[Observation]] = defaultdict(list)
    by_id: dict[str, list[Observation]] = defaultdict(list)
    for observation in canonical:
        by_frame[observation.frame].append(observation)
        by_id[observation.track_id].append(observation)

    invalid_box_rows = sum(not item.valid_box for item in canonical)
    invalid_confidence_rows = sum(not item.valid_confidence for item in canonical)
    valid_confidences = [item.confidence for item in canonical if item.valid_confidence]
    low_confidence_rows = sum(
        item.valid_confidence and item.confidence < config.low_confidence_threshold for item in canonical
    )

    overlap_events = []
    overlap_count_by_frame: Counter[int] = Counter()
    per_frame = []
    previous_count: int | None = None
    for frame in range(start_frame, end_frame + 1):
        frame_items = by_frame.get(frame, [])
        valid_items = [item for item in frame_items if item.valid_box]
        for index, first in enumerate(valid_items):
            for second in valid_items[index + 1 :]:
                value = _iou(first, second)
                if value >= config.duplicate_iou_threshold:
                    overlap_count_by_frame[frame] += 1
                    overlap_events.append(
                        {
                            "frame": frame,
                            "track_id_a": first.track_id,
                            "track_id_b": second.track_id,
                            "iou": value,
                        }
                    )
        confidences = [item.confidence for item in frame_items if item.valid_confidence]
        active_count = len(frame_items)
        per_frame.append(
            {
                "frame": frame,
                "active_track_count": active_count,
                "count_change": None if previous_count is None else active_count - previous_count,
                "mean_confidence": statistics.fmean(confidences) if confidences else None,
                "low_confidence_count": sum(
                    value < config.low_confidence_threshold for value in confidences
                ),
                "high_iou_pair_count": overlap_count_by_frame[frame],
            }
        )
        previous_count = active_count

    short_frame_limit = config.short_track_seconds * config.fps
    per_track = []
    endpoint_data: dict[str, tuple[Observation, Observation]] = {}
    for track_id in sorted(by_id, key=_id_sort_key):
        items = sorted(by_id[track_id], key=lambda item: item.frame)
        endpoint_data[track_id] = (items[0], items[-1])
        deltas = [second.frame - first.frame for first, second in zip(items, items[1:])]
        gaps = [delta - 1 for delta in deltas if delta > 1]
        steps = [
            value
            for first, second in zip(items, items[1:])
            if second.frame - first.frame == 1
            for value in [_normalized_step(first, second)]
            if value is not None
        ]
        jump_count = sum(value > config.jump_threshold_box_heights for value in steps)
        confidence_values = [item.confidence for item in items if item.valid_confidence]
        heights = [item.height for item in items if item.valid_box]
        lifespan_frames = items[-1].frame - items[0].frame + 1
        per_track.append(
            {
                "track_id": track_id,
                "start_frame": items[0].frame,
                "end_frame": items[-1].frame,
                "observed_frames": len(items),
                "observed_seconds": len(items) / config.fps,
                "lifespan_frames": lifespan_frames,
                "presence_coverage_rate": len(items) / lifespan_frames,
                "is_short_track": len(items) < short_frame_limit,
                "starts_inside_window": items[0].frame > start_frame,
                "ends_inside_window": items[-1].frame < end_frame,
                "gap_episode_count": len(gaps),
                "missing_frames_inside_span": sum(gaps),
                "max_gap_frames": max(gaps, default=0),
                "jump_event_count": jump_count,
                "max_step_box_heights_per_frame": max(steps, default=None),
                "mean_confidence": statistics.fmean(confidence_values) if confidence_values else None,
                "min_confidence": min(confidence_values, default=None),
                "low_confidence_observations": sum(
                    value < config.low_confidence_threshold for value in confidence_values
                ),
                "median_box_height_px": statistics.median(heights) if heights else None,
            }
        )

    fragment_candidates = []
    track_ids = sorted(endpoint_data, key=_id_sort_key)
    for old_id in track_ids:
        _, old_end = endpoint_data[old_id]
        for new_id in track_ids:
            if new_id == old_id:
                continue
            new_start, _ = endpoint_data[new_id]
            elapsed = new_start.frame - old_end.frame
            if not 1 <= elapsed <= config.fragment_max_frame_gap:
                continue
            normalized_step = _normalized_step(old_end, new_start)
            if normalized_step is None:
                continue
            height_ratio = max(old_end.height, new_start.height) / min(old_end.height, new_start.height)
            if (
                normalized_step <= config.fragment_max_distance_per_frame
                and height_ratio <= config.fragment_max_height_ratio
            ):
                fragment_candidates.append(
                    {
                        "old_track_id": old_id,
                        "new_track_id": new_id,
                        "old_end_frame": old_end.frame,
                        "new_start_frame": new_start.frame,
                        "elapsed_frames": elapsed,
                        "distance_box_heights_per_frame": normalized_step,
                        "box_height_ratio": height_ratio,
                        "endpoint_iou": _iou(old_end, new_start),
                    }
                )
    fragment_candidates.sort(
        key=lambda row: (row["distance_box_heights_per_frame"], row["elapsed_frames"])
    )

    expected_frames = end_frame - start_frame + 1
    frame_counts = [row["active_track_count"] for row in per_frame]
    nonempty_frames = sum(count > 0 for count in frame_counts)
    changed_by_three = sum(
        abs(row["count_change"]) >= 3 for row in per_frame if row["count_change"] is not None
    )
    short_tracks = sum(row["is_short_track"] for row in per_track)
    gap_tracks = sum(row["gap_episode_count"] > 0 for row in per_track)
    jump_tracks = sum(row["jump_event_count"] > 0 for row in per_track)
    internal_starts = sum(row["starts_inside_window"] for row in per_track)
    internal_ends = sum(row["ends_inside_window"] for row in per_track)
    median_active = statistics.median(frame_counts)

    return {
        "report_kind": "tracking_quality_diagnostics_without_ground_truth",
        "ground_truth_used": False,
        "accuracy_claim": False,
        "source": str(Path(source).resolve()) if source is not None else None,
        "configuration": asdict(config),
        "limitations": [
            "These values are diagnostic risk indicators, not accuracy, IDF1, HOTA, MOTA, precision or recall.",
            "A high-IoU pair may be two genuinely occluding players; it is only a duplicate-box risk.",
            "A nearby ending/starting ID pair is only a possible fragmentation link and is not auto-merged.",
            "Frame coverage measures CSV output presence, not whether every real player was detected.",
        ],
        "input_integrity": {
            "raw_rows": len(raw),
            "canonical_observations": len(canonical),
            "duplicate_frame_id_rows": duplicate_rows,
            "invalid_box_rows": invalid_box_rows,
            "invalid_confidence_rows": invalid_confidence_rows,
        },
        "frame_range": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "expected_frames": expected_frames,
            "frames_with_any_track": nonempty_frames,
            "frames_without_tracks": expected_frames - nonempty_frames,
            "csv_frame_coverage_rate": nonempty_frames / expected_frames,
        },
        "counts": {
            "unique_track_ids": len(by_id),
            "active_tracks_per_frame": _descriptive(frame_counts),
            "large_count_change_frames_abs_ge_3": changed_by_three,
        },
        "confidence": {
            **_descriptive(valid_confidences),
            "threshold": config.low_confidence_threshold,
            "low_confidence_observations": low_confidence_rows,
            "low_confidence_rate": low_confidence_rows / len(valid_confidences) if valid_confidences else None,
        },
        "fragmentation_risk_indicators": {
            "short_track_limit_seconds": config.short_track_seconds,
            "short_track_count": short_tracks,
            "short_track_rate": short_tracks / len(per_track),
            "tracks_starting_inside_window": internal_starts,
            "tracks_ending_inside_window": internal_ends,
            "tracks_with_internal_gaps": gap_tracks,
            "tracks_with_jump_events": jump_tracks,
            "potential_fragment_link_count": len(fragment_candidates),
            "identity_turnover_ratio_vs_median_active": (
                len(by_id) / median_active if median_active > 0 else None
            ),
        },
        "overlap_and_duplicate_risk": {
            "same_frame_same_id_duplicate_rows": duplicate_rows,
            "different_id_high_iou_pair_events": len(overlap_events),
            "frames_with_high_iou_pair": len(overlap_count_by_frame),
            "iou_threshold": config.duplicate_iou_threshold,
        },
        "per_frame": per_frame,
        "per_track": per_track,
        "potential_fragment_links": fragment_candidates,
        "high_iou_pair_events": overlap_events,
    }


def audit_tracking_csv(path: str | Path, config: AuditConfig | None = None) -> dict:
    return audit_tracking(load_tracking_csv(path), config, source=path)


def public_summary(audit: Mapping) -> dict:
    """Drop potentially large detail tables for concise machine-readable summaries."""
    detail_keys = {"per_frame", "per_track", "potential_fragment_links", "high_iou_pair_events"}
    return {key: value for key, value in audit.items() if key not in detail_keys}
