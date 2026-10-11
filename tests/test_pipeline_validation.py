import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halisaha.pipeline_validation import (
    aligned_frame_count,
    assert_tracking_integrity,
    inspect_tracking_csv,
    public_track_info,
)


FIELDS = ["frame", "track_id", "confidence", "x1", "y1", "x2", "y2"]


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as stream:
        # Some validation tests intentionally omit a column from the header.
        # Ignore the corresponding extra value in the fixture row so the
        # production validator, rather than DictWriter, observes the bad file.
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row(frame, identity=1):
    return dict(frame=frame, track_id=identity, confidence=.8, x1=1, y1=2, x2=11, y2=22)


class PipelineValidationTests(unittest.TestCase):
    def test_inspection_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            write_csv(path, [row(10), row(10), row(11, 2)])
            info = inspect_tracking_csv(path)
            self.assertEqual(info["rows"], 3)
            self.assertEqual(info["frame_count"], 2)
            self.assertEqual(info["local_id_count"], 2)
            self.assertEqual(info["duplicate_frame_id_rows"], 1)
            self.assertNotIn("frames", public_track_info(info))

    def test_aligned_frames_respect_offset(self):
        first = {"frames": {10, 11, 12}}
        second = {"frames": {12, 13, 20}}
        self.assertEqual(aligned_frame_count(first, second, 2), 2)

    def test_missing_column_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            write_csv(path, [row(1)], fields=FIELDS[:-1])
            with self.assertRaisesRegex(ValueError, "missing columns"):
                inspect_tracking_csv(path)

    def test_malformed_row_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            bad = row("not-a-frame")
            write_csv(path, [bad])
            with self.assertRaisesRegex(ValueError, "malformed"):
                inspect_tracking_csv(path)

    def test_non_finite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            bad = row(1)
            bad["x1"] = "nan"
            write_csv(path, [bad])
            with self.assertRaisesRegex(ValueError, "malformed"):
                inspect_tracking_csv(path)

    def test_pipeline_integrity_rejects_duplicates_and_bad_boxes(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assert_tracking_integrity({
                "path": "tracks.csv", "duplicate_frame_id_rows": 1, "invalid_box_rows": 0,
            })
        with self.assertRaisesRegex(ValueError, "invalid boxes"):
            assert_tracking_integrity({
                "path": "tracks.csv", "duplicate_frame_id_rows": 0, "invalid_box_rows": 2,
            })


if __name__ == "__main__":
    unittest.main()
