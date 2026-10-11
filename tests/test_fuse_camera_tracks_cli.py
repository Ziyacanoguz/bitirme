import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fuse_camera_tracks.py"
SPEC = importlib.util.spec_from_file_location("fuse_camera_tracks_cli", SCRIPT)
fusion_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fusion_cli)


class FusionCliEmptyEvidenceTests(unittest.TestCase):
    def run_empty_evidence_case(self, directory, *, calibration_supported):
        root = Path(directory)
        args = [str(SCRIPT), "--output-dir", str(root / "result")]
        for camera, x in ((1, 0.1), (2, 0.9)):
            tracks = root / f"camera{camera}.csv"
            rows = [dict(
                frame=frame, track_id=camera, confidence=0.9,
                x1=10, y1=10, x2=20, y2=30, field_x=x, field_y=0.5,
                calibration_supported=calibration_supported,
            ) for frame in range(4)]
            with tracks.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            features = root / f"camera{camera}.npz"
            np.savez(features, ids=np.array([camera]), embeddings=np.array([[1.0, 0.0]]))
            args.extend((f"--camera{camera}", str(tracks), f"--reid{camera}", str(features)))
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            fusion_cli.main()
        return root / "result"

    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def test_no_cross_camera_candidates_is_a_complete_unmatched_run(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_empty_evidence_case(directory, calibration_supported=1)
            fields, candidates = self.read_csv(result / "candidates.csv")
            self.assertIn("support_frames", fields)
            self.assertIn("accepted", fields)
            self.assertEqual(candidates, [])
            _, fused = self.read_csv(result / "fused.csv")
            self.assertEqual(len(fused), 8)
            self.assertEqual({row["status"] for row in fused}, {"unmatched"})
            report = json.loads((result / "run.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["candidate_pairs"], 0)
            self.assertEqual(report["shared_components"], 0)
            self.assertTrue((result / "time_offset_audit.json").is_file())

    def test_no_valid_positions_produces_empty_fused_csv_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_empty_evidence_case(directory, calibration_supported=0)
            fields, fused = self.read_csv(result / "fused.csv")
            self.assertIn("source_frame_camera1", fields)
            self.assertIn("status", fields)
            self.assertEqual(fused, [])
            report = json.loads((result / "run.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["totals"]["invalid_position_rows"], 8)
            self.assertEqual(report["total_components"], 2)


if __name__ == "__main__":
    unittest.main()
