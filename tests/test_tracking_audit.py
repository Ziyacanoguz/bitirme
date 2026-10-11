import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halisaha.tracking_audit import AuditConfig, Observation, audit_tracking, audit_tracking_csv


FIELDS = ["frame", "track_id", "confidence", "x1", "y1", "x2", "y2"]


def observation(frame, track_id="1", x=0, confidence=.8, width=10, height=20):
    return Observation(frame, str(track_id), confidence, x, 0, x + width, height)


class TrackingAuditTests(unittest.TestCase):
    def test_frame_coverage_short_tracks_gaps_and_confidence(self):
        rows = [
            observation(10, 1, confidence=.9),
            observation(11, 1, confidence=.2),
            observation(13, 1, confidence=.8),
            observation(13, 2, x=100, confidence=.7),
        ]
        config = AuditConfig(fps=2, start_frame=10, end_frame=14, short_track_seconds=1)
        result = audit_tracking(rows, config)
        self.assertEqual(result["frame_range"]["frames_without_tracks"], 2)
        self.assertAlmostEqual(result["frame_range"]["csv_frame_coverage_rate"], 3 / 5)
        self.assertEqual(result["confidence"]["low_confidence_observations"], 1)
        tracks = {row["track_id"]: row for row in result["per_track"]}
        self.assertEqual(tracks["1"]["gap_episode_count"], 1)
        self.assertEqual(tracks["1"]["missing_frames_inside_span"], 1)
        self.assertTrue(tracks["2"]["is_short_track"])

    def test_jump_only_uses_consecutive_frames(self):
        rows = [
            observation(1, 1, x=0),
            observation(2, 1, x=100),
            observation(4, 1, x=1000),
        ]
        result = audit_tracking(rows, AuditConfig(jump_threshold_box_heights=.75))
        track = result["per_track"][0]
        self.assertEqual(track["jump_event_count"], 1)
        self.assertEqual(track["gap_episode_count"], 1)

    def test_duplicate_row_and_high_iou_pair_are_separate_risks(self):
        rows = [
            observation(1, 1, x=0, confidence=.7),
            observation(1, 1, x=0, confidence=.9),
            observation(1, 2, x=1, confidence=.8),
        ]
        result = audit_tracking(rows, AuditConfig(duplicate_iou_threshold=.7))
        self.assertEqual(result["input_integrity"]["duplicate_frame_id_rows"], 1)
        self.assertEqual(result["input_integrity"]["canonical_observations"], 2)
        self.assertEqual(result["overlap_and_duplicate_risk"]["different_id_high_iou_pair_events"], 1)
        self.assertEqual(result["per_frame"][0]["active_track_count"], 2)

    def test_nearby_end_and_start_become_candidate_not_automatic_merge(self):
        rows = [
            observation(1, "old", x=0),
            observation(2, "old", x=1),
            observation(3, "new", x=2),
            observation(4, "new", x=3),
            observation(3, "far", x=500),
        ]
        result = audit_tracking(rows, AuditConfig(fragment_max_frame_gap=2))
        pairs = {(row["old_track_id"], row["new_track_id"]) for row in result["potential_fragment_links"]}
        self.assertIn(("old", "new"), pairs)
        self.assertNotIn(("old", "far"), pairs)
        self.assertEqual(result["counts"]["unique_track_ids"], 3)

    def test_csv_validation_and_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(dict(frame=1, track_id=4, confidence=.8, x1=0, y1=0, x2=10, y2=20))
            result = audit_tracking_csv(path)
            self.assertEqual(result["counts"]["unique_track_ids"], 1)
            self.assertEqual(result["source"], str(path.resolve()))

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            path.write_text("frame,track_id\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                audit_tracking_csv(path)


if __name__ == "__main__":
    unittest.main()
